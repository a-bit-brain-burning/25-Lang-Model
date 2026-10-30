"""Fixed-representation and parameter substitutions on reference time axes.

REFERENCE_CONDITION_D1 is a training-matched diagnostic, not ordinary text
inference. All true-axis streams share quiet-span policy, D/F0 guard and DSP.
Natural text predictions are rendered and reported separately.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch

from ..common import PROJECT,save_json,sha256
from ..audio.dsp import write_wav
from ..audio.delivery_timing import apply_numeric_delivery
from ..models.boundary_acoustic import load_boundary_acoustic
from ..models.universal_acoustic import load_acoustic,expand_controls
from ..models.phrase_prosody import load_planner
from ..frontend.chinese_delivery_policy import numeric_duration_policy
from .phrase_acoustic_inference import append_plan
from .chinese_prosody_evaluate import apply_profile,physical_frames
from .representation_limit import GenericDSPAdapter
from .boundary_trajectory_diagnostics import trajectory_metrics,write_csv,audio_metrics
from .segment_boundary_diagnose import segment_masks,matched_unit_map

GROUPS={'F0':(0,),'energy':(1,),'AP_VUV':(25,26,27,28,29,30),
        'shape23':tuple(range(2,25)),'F0_AP_VUV':(0,25,26,27,28,29,30)}


def reference_logits(curve):
    result=np.asarray(curve,np.float32).copy()
    if result.ndim!=2 or result.shape[1]!=31:raise ValueError('Expected frame31 reference')
    result[:,30]=np.where(result[:,30]>.5,3.,-3.)
    return result


def substitute(prediction,oracle,group):
    """F0 is substituted only where the donor says it is voiced/defined."""
    if group not in GROUPS:raise ValueError('Unknown group')
    predicted=np.asarray(prediction,np.float32);reference=np.asarray(oracle,np.float32)
    if predicted.shape!=reference.shape or predicted.ndim!=2 or predicted.shape[1]!=31:
        raise ValueError('Substitution requires identical 31D time axes')
    result=predicted.copy()
    for column in GROUPS[group]:
        valid=reference[:,30]>0 if column==0 else np.ones(len(result),bool)
        result[valid,column]=reference[valid,column]
    return result


def reference_features(reference,row):
    return {'phonetic_features':reference['phonetic_features'],
        'context_features':reference['context_features'],
        'syllable_phone_ranges':np.asarray([(s['phone_start'],s['phone_end'])for s in row['frontend_metadata']['syllables']],np.int64)}


def comma_units(row,units):
    after=set();position=0
    for char in row['text']:
        if '\u3400'<=char<='\u9fff':position+=1
        elif char in ',，':after.add(position)
    mask=np.zeros(units,bool);position=0;syllables=row['frontend_metadata']['syllables']
    for i,s in enumerate(syllables[:-1]):
        position+=len(s['character']);index=s['phone_end']
        if position in after and index<syllables[i+1]['phone_start']:mask[index]=True
    return mask


def uniform_postprocess(curve,durations,features,quiet_units,quiet_log_energy=-30.):
    """Same existing policy on frame/oracle/mixed/predicted true-axis streams.

    Durations stay fixed. Quiet span manipulation is the frame equivalent of
    apply_numeric_delivery, followed by the unchanged guarded D/F0 profile.
    """
    source=np.asarray(curve,np.float32);durations=np.asarray(durations,np.int64)
    if len(source)!=int(durations.sum()) or len(quiet_units)!=len(durations):raise ValueError('Time axis mismatch')
    quiet=np.repeat(np.asarray(quiet_units,bool),durations)
    prepared=source.copy();prepared[quiet,1]=np.minimum(prepared[quiet,1],quiet_log_energy)
    prepared[quiet,25:30]=0.;prepared[quiet,30]=-12.
    result,audit,modified=apply_profile(features,{'curve':prepared,'durations':durations},True)
    changed=np.any(result!=source,axis=0)
    return result,{'quiet_frames':int(quiet.sum()),'D_modified_F0_frames':int(modified.sum()),
        'changed_channels':np.flatnonzero(changed).tolist(),'duration_changed':False,'D_audit':audit}


def masks_and_events(row,reference):
    dur=reference['target_duration'];n=int(dur.sum());offsets=np.r_[0,dur.cumsum()]
    masks,events=segment_masks(row['frontend_metadata']['syllables'],reference['phonetic_features'],dur,np.ones(len(dur),bool))
    masks['whole']=np.ones(n,bool);masks['boundary_pm40ms']=np.zeros(n,bool)
    for b in offsets[1:-1]:masks['boundary_pm40ms'][max(0,int(b)-4):min(n,int(b)+4)]=True
    masks['nonboundary']=~masks['boundary_pm40ms']
    masks['sentence_tail_300ms']=np.zeros(n,bool)
    syl=row['frontend_metadata']['syllables']
    if syl:
        b=int(offsets[syl[-1]['phone_end']]);masks['sentence_tail_300ms'][max(0,b-30):b]=True
    # Add last pronounced syllable endings before actual positive pauses.
    pauses=[]
    for s in syl:
        end=s['phone_end'];stop=int(offsets[end])
        if end<len(dur) and dur[end]>0 and reference['phonetic_features'][end,0]>.5:
            pauses.append({'kind':'pre_pause_tail','frame':stop,'left_unit':end-1,'right_unit':end,
                'start_frame':max(0,stop-20),'end_frame':min(n,stop+20),'left_character':s['character']})
    return masks,events+pauses


def extra_metrics(predicted,reference,mask):
    result=trajectory_metrics(predicted,reference,mask)
    selected=np.flatnonzero(mask);voice=predicted[:,30]>0
    pairs=mask[:-1]&mask[1:];joint=pairs&voice[:-1]&voice[1:]
    if joint.any():
        delta=np.diff(predicted[:,0])*1200/np.log(2)
        result['absolute_F0_delta_cents_mean']=float(abs(delta[joint]).mean())
        result['F0_delta_gt200cents_fraction']=float((abs(delta[joint])>200).mean())
    result['VUV_transitions']=int(np.sum(pairs&(voice[:-1]!=voice[1:])))
    if len(selected):result['mean_energy_log']=float(predicted[selected,1].mean())
    return result


def render_natural(model,plan,cfg,adapter):
    features=append_plan(plan['frontend'],plan['unit_plan']);prediction=model.predict_features(features)
    policy=numeric_duration_policy(features,prediction['durations'],**{k:cfg[k]for k in
        ('comma_min_frames','kinship_max_ratio','minimum_rime_frames','minimum_neutral_scale','include_final_kinship')})
    controls,durations,_=apply_numeric_delivery(prediction['controls'],prediction['durations'],policy['target_durations'],policy['comma_mask'],quiet_log_energy=cfg['quiet_log_energy'])
    raw=expand_controls(controls,durations);curve,audit=uniform_postprocess(raw,durations,features,policy['comma_mask'],cfg['quiet_log_energy'])
    pcm,dsp=adapter.render(physical_frames(curve))
    return features,prediction,durations,raw,curve,pcm,{'postprocess':audit,'DSP':dsp,'duration_policy':policy['metadata']}


def listening_page(output,records,seed=20261016):
    output.mkdir(parents=True,exist_ok=True);rng=random.Random(seed);key=[]
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>连续性参数替换诊断</title>',
        '<style>body{font:17px sans-serif;max-width:1050px;margin:30px auto;padding:0 16px}audio{width:280px}label{display:inline-block;margin:8px}</style>',
        '<h1>同轴连续性诊断</h1><p>真实时长、参考训练native条件，仅用于参数组诊断。整句保持同一DSP状态后截取；不是自然文本推理。每组只比较一个问题：尾部/边界是否完整、过渡是否连贯、舒适度是否退化。没有人工评分结论。</p>']
    systems=[['M3_D1','O_repr','O_frame'],['M3_D1','swap_F0','swap_AP_VUV','swap_F0_AP_VUV'],['M3_D1','swap_shape23','swap_energy']]
    for i,record in enumerate(records[:9]):
        selected=list(systems[i%3]);rng.shuffle(selected);pcm=record['pcm'];frames=record['frames']
        event=record['events'][-1]if record['events']else {'frame':max(1,frames//2),'kind':'middle_context'}
        center=event['frame'];start=max(0,center-75);stop=min(frames,center+75)
        arrays={name:pcm[name][start*160:stop*160].astype(np.float64)/32768 for name in selected}
        gains={name:10**(-25/20)/max(1e-12,float(np.sqrt(np.mean(x*x))))for name,x in arrays.items()}
        maximum=max(float(abs(arrays[name]).max(initial=0))*gains[name]for name in selected)
        shared=min(1.,.95/max(maximum,1e-12));html.append(f'<section><h2>BDG{i+1:02}</h2><p>{escape(record["text"])}</p>')
        for j,name in enumerate(selected):
            label=f'BDG{i+1:02}_{chr(65+j)}';source=pcm[name][start*160:stop*160]
            write_wav(output/(label+'_raw.wav'),source);matched=np.clip(np.rint(arrays[name]*gains[name]*shared*32768),-32768,32767).astype(np.int16)
            write_wav(output/(label+'.wav'),matched);html.append(f'<label>{label}<br><audio controls preload="none" src="{label}.wav"></audio></label>')
            key.append({'label':label,'id':record['id'],'system':name,'source':str(record['paths'][name]),
                'start_frame':start,'stop_frame':stop,'event':event,'gain_dB':float(20*np.log10(gains[name]*shared)),
                'full_sentence_DSP_before_excerpt':True,'reference_condition':True,'raw_sha256':sha256(output/(label+'_raw.wav')),'matched_sha256':sha256(output/(label+'.wav'))})
        html.append('</section>')
    (output/'index.html').write_text('\n'.join(html));save_json(output.parent/'B_diagnosis_answers.json',{'seed':seed,'mapping':key,'gain_algorithm':'whole excerpt RMS to -25dBFS, group-shared attenuation if peak>.95; no frame normalization; raw retained'})


def run(root,limit=12):
    root=Path(root);output=root/'transition_diagnosis/B';output.mkdir(parents=True,exist_ok=True)
    representation=root/'representation_audit';representation.mkdir(parents=True,exist_ok=True)
    if (output/'COMPLETED.json').exists():raise ValueError('Frozen output already exists')
    torch.set_num_threads(1);selection=root/'jobs/diagnostic_set.json';rows=json.loads(selection.read_text())['rows'][:limit]
    checkpoints={'M3':PROJECT/'outputs/boundary_trajectory_20261015_1610/training/seed20261028/M3/latest.pt',
        'B_accept':PROJECT/'outputs/phrase_prosody_20261015/acoustic/training/reference_condition/best.pt'}
    models={'M3':load_boundary_acoustic(checkpoints['M3']),'B_accept':load_acoustic(checkpoints['B_accept'])}
    planner_path=PROJECT/'outputs/phrase_prosody_20261015/planner/best.pt';planner=load_planner(planner_path)
    cfg_path=PROJECT/'configs/chinese_articulation_v1.json';cfg=json.loads(cfg_path.read_text());adapter=GenericDSPAdapter(24,5)
    frozenpaths=[cfg_path,PROJECT/'configs/chinese_linked_f0_guarded_v1.json',PROJECT/'src/audio/native.c',PROJECT/'src/audio/native_stream.c',PROJECT/'src/experiments/chinese_prosody_evaluate.py',PROJECT/'src/audio/delivery_timing.py',PROJECT/'src/models/universal_acoustic.py']
    config={'status':'RUNNING','seed':20261016,'selected_ids':[r['id']for r in rows],'selection_sha256':sha256(selection),
        'split_counts':{s:sum(r['split']==s for r in rows)for s in ('train','validation')},
        'postprocess_hashes':{str(p):sha256(p)for p in frozenpaths},'checkpoints':{n:{'path':str(p),'sha256':sha256(p)}for n,p in checkpoints.items()},
        'planner_sha256':sha256(planner_path),'DSP':{'dim':24,'AP_bands':5,'taps':128,'gain':1.,'seed':1729},
        'axis':'REFERENCE_CONDITION_D1; frozen reference-native phone/context22+prosody12+predicted_text_plan8, true durations',
        'natural_axis':'current frozen frontend + fixed text planner + predicted controls/durations; separate table, no direct frame errors against reference',
        'common_true_axis_policy':'same actual-comma quiet-span frame adapter, unchanged D/F0 guard, physical conversion and full-sentence C DSP; durations fixed',
        'F0_substitution':'O_repr logF0 copied only on donor-voiced frames; no unvoiced frequency fabricated',
        'human_listening':'NOT_RUN','test_used':False,'source_sha256':sha256(Path(__file__))}
    save_json(output/'config.json',config);summary_rows=[];natural_rows=[];metric_rows=[];per_frame_rows=[];listening=[];representation_rows=[]
    from .chinese_data import fit31
    for index,row in enumerate(rows):
        uid=row['id'];folder=output/'cases'/uid;folder.mkdir(parents=True,exist_ok=True)
        if sha256(Path(row['record_path']))!=row['record_sha256']:raise ValueError('Reference changed '+uid)
        with np.load(row['record_path'],allow_pickle=False)as z:reference={k:z[k]for k in z.files}
        durations=reference['target_duration'];features=reference_features(reference,row);quiet=comma_units(row,len(durations))
        aux=Path(row['features_path']);plan_cache=PROJECT/'outputs/phrase_prosody_20261015/labels/predicted'/f'{uid}.npy'
        conditioned={**features,'context_features':np.concatenate([reference['context_features'],np.load(aux),np.load(plan_cache)],axis=-1),'edge_sil_mask':reference['edge_sil_mask']}
        frame=reference_logits(reference['target_curve']);o8=expand_controls(reference['target_controls'],durations)
        if index<2:
            rebuilt,mask=fit31(reference['target_curve'],durations)
            error=float(abs(rebuilt-reference['target_controls']).max(initial=0))
            representation_rows.append({'id':uid,'target_generation_recheck_max_abs':error,'target_controls_exact':bool(np.array_equal(rebuilt,reference['target_controls']))})
            if error>1e-6:raise ValueError('Target regeneration mismatch '+uid)
        predictions={name:model.predict_features(conditioned)for name,model in models.items()}
        curves={'O_frame':frame,'O_repr':o8}
        for name,value in predictions.items():curves[('M3'if name=='M3'else'B_accept')+'_D1']=expand_controls(value['controls'],durations)
        for group in GROUPS:curves['swap_'+group]=substitute(curves['M3_D1'],o8,group)
        masks,events=masks_and_events(row,reference);pcm_by={};paths={};true_post,_=uniform_postprocess(frame,durations,features,quiet,cfg['quiet_log_energy'])
        np.savez_compressed(folder/'frozen_inputs.npz',**reference,prosody12=np.load(aux),predicted_plan8=np.load(plan_cache),quiet_units=quiet,
            M3_controls=predictions['M3']['controls'],M3_predicted_durations=predictions['M3']['durations'],B_accept_controls=predictions['B_accept']['controls'])
        for system,pre in curves.items():
            post,audit=uniform_postprocess(pre,durations,features,quiet,cfg['quiet_log_energy']);pcm,dsp=adapter.render(physical_frames(post))
            path=folder/(system+'.wav');write_wav(path,pcm);pcm_by[system]=pcm;paths[system]=path
            np.savez_compressed(folder/(system+'.npz'),pre=pre,post=post,durations=durations,quiet_units=quiet)
            save_json(folder/(system+'.json'),{'axis':config['axis'],'postprocess':audit,'DSP':dsp,'audio':audio_metrics(pcm),'wav_sha256':sha256(path)})
            for region,mask in masks.items():
                for phase,pred,truth in [('pre',pre,frame),('post',post,true_post)]:
                    metric=extra_metrics(pred,truth,mask);flat={k:v for k,v in metric.items()if not isinstance(v,dict)}
                    metric_rows.append({'id':uid,'split':row['split'],'system':system,'phase':phase,'region':region,**flat})
            for event in events:
                a,b=event['start_frame'],event['end_frame'];mask=np.zeros(len(frame),bool);mask[a:b]=True
                metric=extra_metrics(post,true_post,mask)
                summary_rows.append({'id':uid,'split':row['split'],'system':system,'event_kind':event.get('kind','cross_character_VC'),
                    'boundary_frame':event['frame'],'left_unit':event['left_unit'],'right_unit':event['right_unit'],
                    **{k:v for k,v in metric.items()if not isinstance(v,dict)}})
                for t in range(a,b):
                    per_frame_rows.append({'id':uid,'system':system,'event_kind':event.get('kind','cross_character_VC'),'boundary_frame':event['frame'],'frame':t,
                        'relative_ms':(t-event['frame'])*10,'logF0':float(post[t,0]),'voiced':int(post[t,30]>0),'energy_log':float(post[t,1]),
                        **{f'AP{k}_encoded':float(post[t,25+k])for k in range(5)}})
        plan=planner.predict_text(row['text'])
        for name,model in models.items():
            tick=time.perf_counter();front,pred,nd,npre,npost,pcm,audit=render_natural(model,plan,cfg,adapter)
            path=folder/(name+'_NATURAL.wav');write_wav(path,pcm)
            np.savez_compressed(folder/(name+'_NATURAL.npz'),pre=npre,post=npost,durations=nd,controls=pred['controls'],context_features=front['context_features'],phonetic_features=front['phonetic_features'])
            mapping,issues=matched_unit_map(front['metadata']['syllables'],row['frontend_metadata']['syllables'],front['phonetic_features'],reference['phonetic_features'])
            natural_rows.append({'id':uid,'split':row['split'],'system':name,'axis':'PURE_TEXT_NATURAL_PREDICTED_DURATION','seconds':len(pcm)/16000,
                'wall_seconds':time.perf_counter()-tick,'ordinary_units':len(nd),'reference_units':len(durations),'matched_speech_units':int((mapping>=0).sum()),
                'wave':str(path),'clipping_fraction':audio_metrics(pcm)['clipping_fraction'],'no_direct_unaligned_frame_error':True})
            save_json(folder/(name+'_NATURAL.json'),{**audit,'frontend_metadata':front['metadata'],'mapping_diagnostic_issues':issues,'reference_condition_used':False})
        listening.append({'id':uid,'text':row['text'],'pcm':pcm_by,'paths':paths,'frames':len(frame),'events':events})
        save_json(output/'RUN_STATE.json',{'status':'RUNNING','completed_utterances':index+1,'total':len(rows)})
    write_csv(output/'per_region.csv',metric_rows);write_csv(output/'per_transition.csv',summary_rows);write_csv(output/'per_frame_transition.csv',per_frame_rows);write_csv(output/'natural_separate.csv',natural_rows)
    grouped={}
    for phase in ('pre','post'):
        grouped[phase]={}
        for system in curves:
            grouped[phase][system]={}
            for region in ('whole','boundary_pm40ms','rime_tail70_100','sentence_tail_300ms'):
                parts=[r for r in metric_rows if r['system']==system and r['phase']==phase and r['region']==region]
                fields=('retained_shape_envelope_RMSE_db','F0_RMSE_cents_jointly_voiced','energy_RMSE_db','AP5_encoded_RMSE','VUV_error_rate','shape_delta_direction_cosine')
                grouped[phase][system][region]={k:float(np.mean([r[k]for r in parts if isinstance(r.get(k),(int,float))]))if any(isinstance(r.get(k),(int,float))for r in parts)else None for k in fields}
    save_json(output/'summary.json',{'status':'COMPLETED','utterances':len(rows),'aggregate_unit':'utterance mean','metrics':grouped,'natural_rows':len(natural_rows),
        'reference_condition_D1_not_natural':True,'MOS_or_human_preference':'NOT_RUN','F0_AP_VUV_reference':'automatic extractor/publisher alignment; not independently gold',
        'frame_transition_CSV':'per_frame_transition.csv','paired_metric_caution':'parameter replacement intentionally improves replaced-channel errors; cannot establish perceived mechanism without listening'})
    prior=PROJECT/'outputs/boundary_trajectory_20261015_1610/verification'
    save_json(representation/'B_reuse_and_spotcheck.json',{'status':'COMPLETED','prior_verification_directory':str(prior),
        'scope':'reuse existing C/NumPy interpolation validation; only first2 selected target fits regenerated, all selected O8 expanded by unchanged public function',
        'target_spotchecks':representation_rows,'channels':31,'controls':8,'shape_definition':'DCT-II(log WORLD power envelope)[1:24]/sqrt(513), 23 retained coefficients',
        'F0':'log Hz in neural; exp to Hz only when binary VUV>0','energy':'ln waveform mean-square energy',
        'AP':'band amplitude a=10^(3*encoded), encoded in[-1,0], equivalent 20log10 amplitude/60',
        'voicing':'target frames probability0/1 converted to +/-3; model controls logit, physical threshold>0',
        'timing':'j/n position*7; no boundary endpoint duplication; fixed 10ms frame','O8_not_mathematical_lower_bound':True})
    listening_page(root/'listen/diagnosis/B',listening)
    if config['postprocess_hashes']!={str(p):sha256(p)for p in frozenpaths}:raise ValueError('Fixed processing source changed during diagnosis')
    shutil.copyfile(__file__,output/'source_snapshot.py');save_json(output/'COMPLETED.json',{'status':'COMPLETED','utterances':len(rows),'true_axis_systems':len(curves),
        'true_axis_waves':len(rows)*len(curves),'natural_waves':len(natural_rows),'parameter_groups':list(GROUPS),'postprocessor_SHA_unchanged':True,'source_sha256':sha256(Path(__file__))})
    return grouped


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--limit',type=int,default=12)
    args=parser.parse_args();run(args.root,args.limit)


if __name__=='__main__':main()
