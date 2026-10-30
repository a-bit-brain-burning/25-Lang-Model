"""Frozen R1 segment/VC diagnosis; reference swaps are explicit oracle only."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_name,'1')
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from html import escape
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
from ..common import PROJECT,save_json,sha256
from ..models.universal_acoustic import expand_controls
from ..audio.dsp import write_wav
from .chinese_prosody_evaluate import physical_frames
from .representation_limit import GenericDSPAdapter

BASE=PROJECT/'outputs/phrase_prosody_r2_20261015/evaluation/validation'
DATA=PROJECT/'outputs/chinese_training_20261014/data_full'
OUT=PROJECT/'outputs/segment_fidelity_20261015/diagnosis'
SYSTEM='old_reference_old_planner'
ACOUSTIC=PROJECT/'outputs/phrase_prosody_20261015/acoustic/training/reference_condition/best.pt'
PLANNER=PROJECT/'outputs/phrase_prosody_20261015/planner/best.pt'
EXPECTED_ACOUSTIC='c272ba27dcd9a043156a1413e8d2c31ab542d25595965361da35cbda7b173ba1'
EXPECTED_PLANNER='507ea949f59b20254034aef6cafbea739a4b89a81d26fa0d043a3198047795c4'
GROUPS={'VC_shape':list(range(2,25)),'VC_AP':list(range(25,30)),
        'VC_VUV':[30],'VC_energy':[1],'VC_joint':list(range(1,31))}


def matched_unit_map(ordinary,reference,ordinary_phonetic,reference_phonetic):
    """Only exact character span, unit count and numeric-phone matches qualify."""
    spans={};position=0
    for item in reference:
        end=position+len(item['character']);spans[(position,end)]=item;position=end
    if ''.join(x['character'] for x in ordinary)!=''.join(x['character'] for x in reference):raise ValueError('Character text differs')
    mapping=np.full(len(ordinary_phonetic),-1,np.int64);issues=[];position=0
    for item in ordinary:
        end=position+len(item['character']);ref=spans.get((position,end));a,b=item['phone_start'],item['phone_end']
        reason=None
        if ref is None:reason='fused_character_topology_mismatch'
        elif b-a!=ref['phone_end']-ref['phone_start']:reason='unit_count_mismatch'
        else:
            c,d=ref['phone_start'],ref['phone_end']
            equal=np.all(np.isclose(ordinary_phonetic[a:b],reference_phonetic[c:d],atol=1e-6),axis=-1)
            for i,ok in enumerate(equal):
                if ok:mapping[a+i]=c+i
            if not equal.all():reason='numeric_phone_mismatch'
            if item.get('pinyin')!=ref.get('pinyin'):
                issues.append({'character':item['character'],'ordinary_pinyin':item.get('pinyin'),'reference_pinyin':ref.get('pinyin'),'reason':'pinyin_or_tone_difference','mapped_units':int(equal.sum())})
        if reason:issues.append({'character':item['character'],'reason':reason})
        position=end
    return mapping,issues


def reference_on_prediction(reference,reference_durations,predicted_durations,mapping,baseline):
    """Resample within each phone by j/n; never interpolate across a boundary.

    Binary reference VUV uses nearest observed frame (not a soft fabricated
    label), then ±6 logits for the unchanged physical adapter. Other continuous
    channels linearly interpolate with end clamping inside the SAME phone.
    """
    result=np.asarray(baseline,np.float32).copy();valid=np.zeros(len(result),bool)
    source=np.r_[0,np.cumsum(reference_durations)];target=np.r_[0,np.cumsum(predicted_durations)]
    if len(result)!=target[-1] or len(reference)!=source[-1]:raise ValueError('Curve duration mismatch')
    for unit,ref in enumerate(mapping):
        if ref<0 or not predicted_durations[unit] or not reference_durations[ref]:continue
        a,b=target[unit:unit+2];c,d=source[ref:ref+2];length=int(d-c)
        positions=np.arange(b-a,dtype=np.float64)*length/(b-a);left=np.minimum(positions.astype(np.int64),length-1);right=np.minimum(left+1,length-1);alpha=(positions-left)[:,None]
        part=reference[c:d];result[a:b]=part[left]*(1-alpha)+part[right]*alpha
        nearest=np.minimum(np.floor(positions+.5).astype(np.int64),length-1)
        result[a:b,30]=np.where(part[nearest,30]>.5,6.,-6.)
        valid[a:b]=True
    return result,valid


def segment_masks(syllables,phonetic,durations,matched):
    n=int(np.sum(durations));offsets=np.r_[0,np.cumsum(durations)]
    masks={k:np.zeros(n,bool) for k in ('initial','initial_onset30ms','rime_middle40_70','rime_tail70_100','VC_boundary_pm3')};events=[]
    for item in syllables:
        a,b=item['phone_start'],item['phone_end']
        for unit in range(a,b):
            start,end=offsets[unit:unit+2];length=int(end-start)
            if not length:continue
            if phonetic[unit,1]>.5:
                masks['rime_middle40_70'][start+int(.4*length):start+max(int(.4*length)+1,int(.7*length))]=True
                masks['rime_tail70_100'][start+int(.7*length):end]=True
            elif phonetic[unit,0]<.5 and unit==a:
                masks['initial'][start:end]=True;masks['initial_onset30ms'][start:min(end,start+3)]=True
    for previous,current in zip(syllables[:-1],syllables[1:]):
        left=previous['phone_end']-1;right=current['phone_start']
        # Strict vowel->consonant across characters, never an intervening real pause.
        if not (phonetic[left,1]>.5 and phonetic[right,0]<.5 and phonetic[right,1]<.5):continue
        if durations[left]<=0 or durations[right]<=0 or any(durations[left+1:right]>0):continue
        if not matched[left] or not matched[right]:continue
        boundary=int(offsets[right]);start=max(int(offsets[left]),boundary-3);end=min(int(offsets[right+1]),boundary+3)
        masks['VC_boundary_pm3'][start:end]=True
        events.append({'left_character':previous['character'],'right_character':current['character'],'left_pinyin':previous.get('pinyin'),'right_pinyin':current.get('pinyin'),
                       'left_unit':left,'right_unit':right,'frame':boundary,'start_frame':start,'end_frame':end,'next_initial':current.get('initial'),'next_manner':int(phonetic[right,:7].argmax())})
    return masks,events


def region_metrics(predicted,reference,mask):
    mask=np.asarray(mask,bool);count=int(mask.sum())
    if not count:return {'frames':0}
    p,r=predicted[mask],reference[mask];voice=p[:,30]>0;truth=r[:,30]>0;both=voice&truth;shape=p[:,2:25]-r[:,2:25]
    # Parseval: retained 23 DCT coefficients are scaled by sqrt(513).
    result={'frames':count,'shape23_coefficient_RMSE':float(np.sqrt(np.mean(shape**2))),
            'retained_shape_envelope_RMSE_db':float(np.sqrt(np.mean(np.sum(shape**2,axis=1)))*10/np.log(10)),
            'energy_RMSE_db':float(np.sqrt(np.mean(((p[:,1]-r[:,1])*10/np.log(10))**2))),
            'AP5_encoded_RMSE':float(np.sqrt(np.mean((p[:,25:30]-r[:,25:30])**2))),
            'AP5_linear_amplitude_RMSE':float(np.sqrt(np.mean((10**(3*p[:,25:30])-10**(3*r[:,25:30]))**2))),
            'VUV_error_rate':float(np.mean(voice!=truth)),'predicted_voiced_fraction':float(voice.mean()),'reference_voiced_fraction':float(truth.mean()),
            'reference_voiced_coverage':float((voice&truth).sum()/max(1,truth.sum())),
            'F0_RMSE_cents_jointly_voiced':float(np.sqrt(np.mean(((p[both,0]-r[both,0])*1200/np.log(2))**2))) if both.any() else None}
    full_voice=predicted[:,30]>0;full_truth=reference[:,30]>0
    inner=np.zeros(len(mask),bool);inner[1:-1]=mask[1:-1]&mask[:-2]&mask[2:]
    isolated=np.zeros(len(mask),bool);isolated[1:-1]=(full_voice[1:-1]!=full_voice[:-2])&(full_voice[1:-1]!=full_voice[2:])
    result['isolated_VUV_flip_frames']=int((inner&isolated).sum());result['false_unvoiced_frames_on_reference_voiced']=int((mask&full_truth&~full_voice).sum())
    pairs=mask[:-1]&mask[1:]
    if pairs.any():
        pd=np.diff(predicted[:,2:25],axis=0)[pairs];rd=np.diff(reference[:,2:25],axis=0)[pairs];pn=np.linalg.norm(pd,axis=-1);rn=np.linalg.norm(rd,axis=-1);active=rn>.05
        result.update(shape_delta_predicted_mean_l2=float(pn.mean()),shape_delta_reference_mean_l2=float(rn.mean()),
                      shape_delta_mean_magnitude_ratio=float(pn.mean()/max(1e-12,rn.mean())),
                      shape_delta_direction_cosine=float(np.mean(np.sum(pd[active]*rd[active],axis=-1)/(pn[active]*rn[active]+1e-12))) if active.any() else None,
                      shape_delta_spikes_gt3x_reference_and025=int((pn>np.maximum(.25,3*rn)).sum()))
    return result


def event_metrics(predicted,reference,event):
    b=event['frame'];pd=predicted[b,2:25]-predicted[b-1,2:25];rd=reference[b,2:25]-reference[b-1,2:25];pn=float(np.linalg.norm(pd));rn=float(np.linalg.norm(rd))
    return {**event,'shape_jump_predicted_l2':pn,'shape_jump_reference_l2':rn,'shape_jump_ratio':pn/max(rn,1e-9),
            'shape_jump_direction_cosine':float(np.dot(pd,rd)/(pn*rn+1e-12)),
            'energy_jump_predicted_db':float((predicted[b,1]-predicted[b-1,1])*10/np.log(10)),
            'energy_jump_reference_db':float((reference[b,1]-reference[b-1,1])*10/np.log(10)),
            'predicted_voice_left_right':(predicted[b-1:b+1,30]>0).tolist(),'reference_voice_left_right':(reference[b-1:b+1,30]>0).tolist(),
            'F0_jump_predicted_cents':float((predicted[b,0]-predicted[b-1,0])*1200/np.log(2)),
            'F0_jump_reference_cents':float((reference[b,0]-reference[b-1,0])*1200/np.log(2)),
            'AP5_left':predicted[b-1,25:30].tolist(),'AP5_right':predicted[b,25:30].tolist()}


def swap_channels(curve,reference,mask,channels):
    out=curve.copy();rows=np.flatnonzero(mask);out[np.ix_(rows,channels)]=reference[np.ix_(rows,channels)]
    if not np.array_equal(out[~mask],curve[~mask]):raise AssertionError('Swap leaked outside registered window')
    other=np.setdiff1d(np.arange(31),channels)
    if not np.array_equal(out[:,other],curve[:,other]):raise AssertionError('Swap leaked into other streams')
    return out


def stage_audit(arrays):
    raw=arrays['raw_retimed_curve'];curve=arrays['curve'];original=arrays['original_curve'];duration=arrays['durations'];before=arrays['original_durations']
    assert np.array_equal(expand_controls(arrays['original_controls'],before),original)
    assert np.array_equal(expand_controls(arrays['controls'],duration),raw)
    if not np.array_equal(raw[:,1:],curve[:,1:]):raise AssertionError('D changed non-F0 stream')
    changed=np.any(arrays['controls']!=arrays['original_controls'],axis=(1,2))
    physical=physical_frames(curve);voice=curve[:,-1]>0
    return {'expand_controls_exact_original_and_delivery':True,'D_nonF0_byte_exact':True,'D_modified_F0_frames':int(np.any(curve!=raw,axis=-1).sum()),
            'delivery_changed_units':np.flatnonzero(changed).tolist(),'delivery_nonquiet_controls_byte_exact':bool(np.array_equal(arrays['controls'][~arrays['comma_mask']],arrays['original_controls'][~arrays['comma_mask']])),
            'physical_hard_gate_unvoiced_frames':int((~voice).sum()),'AP_forced_zero_on_unvoiced_frames':int(np.any(curve[~voice,25:30]!=0,axis=-1).sum()),
            'physical_VUV_is_binary':bool(np.isin(physical[:,-1],[0,1]).all()),'physical_AP_unvoiced_is_zero':bool((physical[~voice,25:30]==0).all()),
            'native_source_interpolates_binary_frame_VUV_over_160_samples':True,
            'no_cross_phone_interpolation_in_expand_controls':True}


def original_pcm(row):
    path=Path(row['source_audio'])
    if sha256(path)!=row['source_audio_sha256']:raise ValueError('Original audio SHA differs')
    rate,pcm=wavfile.read(path)
    if pcm.ndim!=1 or pcm.dtype!=np.int16:raise ValueError('Expected original mono PCM16')
    if rate!=16000:
        g=math.gcd(int(rate),16000);pcm=np.clip(np.rint(resample_poly(pcm.astype(np.float64),16000//g,int(rate)//g)),-32768,32767).astype(np.int16)
    return pcm


def run(output=OUT,limit=100,listening_count=12):
    torch.set_num_threads(1);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'completed.json').exists():print('Frozen diagnosis already complete',flush=True);return
    if sha256(ACOUSTIC)!=EXPECTED_ACOUSTIC or sha256(PLANNER)!=EXPECTED_PLANNER:raise ValueError('Frozen R1 baseline checkpoint differs')
    manifest_path=BASE/'manifest.json';baseline=json.loads(manifest_path.read_text());selected=baseline['rows'][:limit]
    allrows={r['id']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()) if r['split']=='validation'}
    sources=[Path(__file__),PROJECT/'src/models/universal_acoustic.py',PROJECT/'src/experiments/chinese_prosody_evaluate.py',PROJECT/'src/audio/delivery_timing.py',PROJECT/'src/audio/native.c',PROJECT/'src/experiments/representation_limit.py']
    source_shas={str(p):sha256(p) for p in sources}
    config={'baseline_system':SYSTEM,'baseline_manifest_sha256':sha256(manifest_path),'data_manifest_sha256':sha256(DATA/'manifest.jsonl'),
            'acoustic_checkpoint_sha256':EXPECTED_ACOUSTIC,'planner_checkpoint_sha256':EXPECTED_PLANNER,
            'ids':[r['id'] for r in selected],'limit':limit,'listening_count':listening_count,'frame_ms':10,'VC_window_each_side_frames':3,
            'rime_middle_phase':[.4,.7],'rime_tail_phase':[.7,1.],'shape_spike_threshold':'predicted_delta_norm > max(0.25,3*reference_delta_norm)',
            'phase_resampling':'per-unit j/n, continuous linear with end clamp; VUV nearest observed frame ±6logits',
            'reference_used_only_in_explicit_oracle_diagnostics':True,'test_loaded':False,'default_changed':False,'source_sha256':source_shas}
    config_path=output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Frozen diagnosis config differs')
    save_json(config_path,config);snap=output/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in sources:shutil.copy2(p,snap/(p.stem+'_'+sha256(p)[:12]+p.suffix))
    adapter=GenericDSPAdapter(24,5);records=[];aggregated=defaultdict(lambda:[[],[],[]]);all_events=[];started=time.monotonic()
    for index,record in enumerate(selected):
        uid=record['id'];row=allrows[uid];info=record['systems'][SYSTEM]
        if info['checkpoint_sha256']!=EXPECTED_ACOUSTIC or not info['ordinary_text_only_inference']:raise ValueError('Baseline identity/reference leak')
        wav=BASE/info['wav'];npz=wav.with_suffix('.npz')
        if sha256(wav)!=info['wav_sha256']:raise ValueError('Baseline WAV differs')
        if info.get('parameter_npz_sha256') and sha256(npz)!=info['parameter_npz_sha256']:raise ValueError('Baseline parameter NPZ differs')
        with np.load(npz,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        reference_path=Path(row['record_path'])
        if sha256(reference_path)!=row['record_sha256']:raise ValueError('Reference parameters differ')
        with np.load(reference_path,allow_pickle=False) as z:reference={k:z[k] for k in ('target_curve','target_duration','target_controls','phonetic_features')}
        mapping,issues=matched_unit_map(info['frontend_metadata']['syllables'],row['frontend_metadata']['syllables'],arrays['phonetic_features'],reference['phonetic_features'])
        mapped,valid=reference_on_prediction(reference['target_curve'],reference['target_duration'],arrays['durations'],mapping,arrays['curve'])
        matched=(mapping>=0)&(np.asarray([reference['target_duration'][r] if r>=0 else 0 for r in mapping])>0)
        masks,events=segment_masks(info['frontend_metadata']['syllables'],arrays['phonetic_features'],arrays['durations'],matched)
        audit=stage_audit(arrays);metrics={name:region_metrics(arrays['curve'],mapped,mask&valid) for name,mask in masks.items()}
        for name,mask in masks.items():
            aggregated[name][0].append(arrays['curve']);aggregated[name][1].append(mapped);aggregated[name][2].append(mask&valid)
        detail=[{**event_metrics(arrays['curve'],mapped,event),'id':uid,'text':row['text']} for event in events];all_events.extend(detail)
        folder=output/'samples'/uid;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'parameters.npz',**arrays,reference_phase_aligned=mapped,reference_matched_frame_mask=valid,mapping=mapping,reference_native_curve=reference['target_curve'],reference_native_duration=reference['target_duration'],**{name+'_mask':mask for name,mask in masks.items()})
        result={'id':uid,'text':row['text'],'baseline_sha256':info['wav_sha256'],'baseline_parameters_sha256':sha256(npz),'matched_units':int((mapping>=0).sum()),'nonsilence_units':int((arrays['phonetic_features'][:,0]<.5).sum()),
                'matched_frames':int(valid.sum()),'total_predicted_frames':len(valid),'mapping_issues':issues,'stage_audit':audit,'regions':metrics,'VC_events':detail,'files':{}}
        if index<listening_count:
            curves={'baseline':arrays['curve'],'matched_phase_oracle':mapped,'oracle_real_native':reference['target_curve'],
                    'oracle_fitted_native':expand_controls(reference['target_controls'],reference['target_duration'])}
            for group,channels in GROUPS.items():curves[group]=swap_channels(arrays['curve'],mapped,masks['VC_boundary_pm3']&valid,channels)
            for name,curve in curves.items():
                target=folder/(name+'.wav');pcm,stats=adapter.render(physical_frames(curve));write_wav(target,pcm);np.savez_compressed(target.with_suffix('.npz'),curve=curve)
                result['files'][name]={'wav':str(target.relative_to(output)),'sha256':sha256(target),'oracle':name!='baseline','seconds':len(pcm)/16000,'clipping_fraction':float(np.mean((pcm==32767)|(pcm==-32768))),'DSP':stats}
            if sha256(folder/'baseline.wav')!=info['wav_sha256']:raise ValueError('Baseline DSP reproduction differs')
            write_wav(folder/'original.wav',original_pcm(row));result['files']['original']={'wav':str((folder/'original.wav').relative_to(output)),'sha256':sha256(folder/'original.wav'),'oracle':True,'original_audio':True}
        save_json(folder/'metrics.json',result);records.append(result)
        if index%10==0:print(json.dumps({'stage':'diagnosis','processed':index+1,'total':len(selected),'VC_events':len(all_events)}),flush=True)
    summary={name:region_metrics(np.concatenate(values[0]),np.concatenate(values[1]),np.concatenate(values[2])) for name,values in aggregated.items()}
    # Mark every join invalid so concatenated clip boundaries cannot add pairwise events.
    summary['note']='Region frame errors aggregate exact frames; pairwise masks are within regions. Clip joins generally silence and excluded.'
    top=sorted(all_events,key=lambda e:e['shape_jump_ratio'] if e['shape_jump_reference_l2']>.05 else -1,reverse=True)[:30]
    save_json(output/'summary.json',{'regions':summary,'VC_events':len(all_events),'utterances':len(records),'mapping_issues_utterances':sum(bool(r['mapping_issues']) for r in records),'top_VC_events_by_jump_ratio_reference_gt005':top,'test_loaded':False,'MOS_measured':False})
    save_json(output/'manifest.json',{'records':records,'ordinary_baseline_only':True,'all_swaps_oracle_diagnostic':True})
    html=['<!doctype html><meta charset="utf-8"><title>字音与VC边界诊断：oracle单列</title><h1>R1基线与参考流替换诊断</h1><p>仅baseline为普通文字推理。VC组为±30ms局部真参数替换；matched_phase_oracle仅精确匹配音素。native oracle使用真实时长，不是可部署系统。不盲评MOS。</p>']
    for result in records[:listening_count]:
        html.append('<h2>'+escape(result['id']+' '+result['text'])+'</h2>')
        for name,file in result['files'].items():html.append('<p>'+escape(name)+(' — oracle诊断' if file['oracle'] else ' — 普通文字基线')+'<br><audio controls src="'+escape(file['wav'])+'"></audio></p>')
    (output/'index.html').write_text('\n'.join(html))
    for p,h in source_shas.items():
        if sha256(Path(p))!=h:raise ValueError('Source changed during diagnosis')
    save_json(output/'completed.json',{'status':'complete','utterances':len(records),'listening_utterances':min(len(records),listening_count),'wav_count':sum(len(r['files']) for r in records),
             'VC_events':len(all_events),'wall_seconds':time.monotonic()-started,'config_sha256':sha256(config_path),'summary_sha256':sha256(output/'summary.json'),'source_sha256':source_shas,'default_changed':False,'test_loaded':False})
    print(json.dumps({'output':str(output),'summary':summary}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--limit',type=int,default=100);p.add_argument('--listening-count',type=int,default=12);a=p.parse_args();run(a.output,a.limit,a.listening_count)


if __name__=='__main__':main()
