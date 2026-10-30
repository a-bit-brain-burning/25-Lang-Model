"""Frozen historical reproduction, true-duration D1 and oracle stream swaps.

No test data, planner updates, DSP changes, or subjective ratings are used.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
import random
import re
import shutil
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
import torch
from ..common import PROJECT, save_json, sha256
from ..models.universal_acoustic import expand_controls, load_acoustic
from .chinese_prosody_evaluate import physical_frames, apply_profile
from .representation_limit import GenericDSPAdapter
from .segment_boundary_diagnose import (BASE, DATA, SYSTEM, ACOUSTIC, PLANNER,
    EXPECTED_ACOUSTIC, EXPECTED_PLANNER, matched_unit_map,
    reference_on_prediction, segment_masks, region_metrics)

DEFAULT = PROJECT/'outputs/boundary_trajectory_20261015_1610/diagnostics'
HISTORY = PROJECT/'outputs/segment_fidelity_20261015/diagnosis'


def frozen_rows(limit=100):
    baseline = json.loads((BASE/'manifest.json').read_text())['rows'][:limit]
    sources = {r['id']:r for r in map(json.loads, (DATA/'manifest.jsonl').read_text().splitlines())
               if r['split'] == 'validation'}
    for item in baseline:
        row = sources[item['id']]
        info = item['systems'][SYSTEM]
        path = (BASE/info['wav']).with_suffix('.npz')
        if sha256(path) != info['parameter_npz_sha256']:
            raise ValueError('Frozen ordinary baseline NPZ changed')
        if sha256(Path(row['record_path'])) != row['record_sha256']:
            raise ValueError('Reference record changed')
        with np.load(path, allow_pickle=False) as z:
            prediction = {k:z[k] for k in z.files}
        with np.load(row['record_path'], allow_pickle=False) as z:
            reference = {k:z[k] for k in z.files}
        yield row, info, prediction, reference


def true_axis_controls(row, info, arrays, reference, controls):
    """Map only exact character/phone matches. Unmatched frames are excluded.

    O8 filling outside valid mask only makes array shapes finite; those frames
    are excluded from all candidate comparisons and never used in swap audio.
    """
    mapping, issues = matched_unit_map(info['frontend_metadata']['syllables'],
        row['frontend_metadata']['syllables'], arrays['phonetic_features'], reference['phonetic_features'])
    out = reference['target_controls'].copy()
    matched = np.zeros(len(out), bool)
    for i, j in enumerate(mapping):
        if j >= 0:
            out[j] = controls[i]
            matched[j] = True
    # Exact whole topology permits aligned edge/punctuation silence too.
    exact = np.array_equal(arrays['phonetic_features'], reference['phonetic_features'])
    if exact:
        out = np.asarray(controls, np.float32).copy()
        matched[:] = True
    valid = np.repeat(matched, reference['target_duration'])
    return out, valid, matched, exact, issues


def trajectory_metrics(predicted, reference, mask):
    result = region_metrics(predicted, reference, mask)
    for span in (1, 2, 4):
        pairs = mask[:-span] & mask[span:]
        pd = (predicted[span:,2:25]-predicted[:-span,2:25])[pairs]
        rd = (reference[span:,2:25]-reference[:-span,2:25])[pairs]
        pn, rn = np.linalg.norm(pd, axis=1), np.linalg.norm(rd, axis=1)
        active = rn > .05
        cos = np.sum(pd[active]*rd[active], axis=1)/(pn[active]*rn[active]+1e-12)
        result[f'delta_{span*10}ms'] = {
            'eligible_pairs':int(len(pd)), 'active_pairs':int(active.sum()),
            'coverage':float(active.mean()) if len(active) else None,
            'direction_cosine':float(cos.mean()) if len(cos) else None,
            'negative_cosine_fraction':float(np.mean(cos < 0)) if len(cos) else None,
            'magnitude_MAE':float(np.mean(abs(pn-rn))) if len(pn) else None,
            'predicted_mean_l2':float(pn.mean()) if len(pn) else None,
            'reference_mean_l2':float(rn.mean()) if len(rn) else None}
    return result


def collect_regions(row, reference, valid, matched):
    dur, pho = reference['target_duration'], reference['phonetic_features']
    masks, events = segment_masks(row['frontend_metadata']['syllables'], pho, dur, matched)
    masks['whole_matched'] = valid.copy()
    boundary = np.zeros(len(valid), bool)
    for b in np.cumsum(dur)[:-1]:
        boundary[max(0,int(b)-4):min(len(valid),int(b)+4)] = True
    masks['all_phone_boundary_pm40ms'] = boundary & valid
    masks['nonboundary_matched'] = ~boundary & valid
    return {k:v & valid for k,v in masks.items()}, events


def metric_summary(streams):
    # False separator frames prevent cross-utterance derivative pairs, including k=4.
    output = {}
    for region, parts in streams.items():
        p, r, m = [], [], []
        for pp, rr, mm in parts:
            p.extend([pp, np.zeros((4,31),np.float32)])
            r.extend([rr, np.zeros((4,31),np.float32)])
            m.extend([mm, np.zeros(4,bool)])
        output[region] = trajectory_metrics(np.concatenate(p), np.concatenate(r), np.concatenate(m))
    return output


def write_csv(path, records):
    if not records:
        return
    keys = sorted(set().union(*(r.keys() for r in records)))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(records)


def evaluate_checkpoint(checkpoint, output, limit=100, predictor=None):
    """D1 on frozen ordinary inputs. Custom predictor returns controls/durations.

    A predictor receives the baseline arrays dict containing the original
    phonetic32/context42 (already frozen text-planner output). It must return
    a dict with N x8 x31 controls and N durations, without reference arguments.
    """
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'completed.json').exists():
        raise ValueError('Candidate evaluation already frozen')
    if predictor is None:
        model = load_acoustic(Path(checkpoint))
        predictor = model.predict_features
    streams = defaultdict(list)
    rows, details, duration_errors = [], [], []
    for row, info, arrays, reference in frozen_rows(limit):
        # The saved NPZ lacks edge flags. Recover them from ordinary text only;
        # otherwise edge-silence duration scoring would use the wrong gate.
        from ..frontend.phrase_plan import make_frontend
        frontend = make_frontend().encode(row['text'])
        if not np.array_equal(frontend['phonetic_features'],arrays['phonetic_features']):
            raise ValueError('Current ordinary frontend differs from frozen baseline')
        ordinary = dict(arrays)
        ordinary['edge_sil_mask'] = np.asarray(frontend['edge_sil_mask'],bool)
        result = predictor(ordinary)
        controls, valid, matched, exact, issues = true_axis_controls(
            row, info, arrays, reference, result['controls'])
        prediction = expand_controls(controls, reference['target_duration'])
        truth = reference['target_curve'].copy()
        truth[:,30] = np.where(truth[:,30] > .5, 6., -6.)
        masks, events = collect_regions(row, reference, valid, matched)
        metrics = {k:trajectory_metrics(prediction, truth, mask) for k, mask in masks.items()}
        for key, mask in masks.items():
            streams[key].append((prediction, truth, mask))
            simple = {k:v for k,v in metrics[key].items() if not isinstance(v,dict)}
            rows.append({'id':row['id'], 'region':key, **simple,
                **{'delta10_'+k:v for k,v in metrics[key]['delta_10ms'].items()}})
        if exact:
            duration_errors.extend(abs(np.asarray(result['durations'])-reference['target_duration']).tolist())
        details.append({'id':row['id'],'exact_numeric_topology':exact,'mapping_issues':issues,
                        'valid_frames':int(valid.sum()),'total_frames':len(valid),'regions':metrics})
    summary = {'status':'COMPLETED','utterances':len(details),'regions':metric_summary(streams),
        'duration_MAE_ms_exact_topology':float(np.mean(duration_errors)*10) if duration_errors else None,
        'duration_units':len(duration_errors),'exact_topology_utterances':sum(x['exact_numeric_topology'] for x in details),
        'axis':'D1 true durations, ordinary text frontend and frozen text planner inputs',
        'D2':'NOT_APPLICABLE: duration prediction is parallel output, not control-point input',
        'unmatched_target_filled_frames_excluded':True,'test_loaded':False,
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(Path(checkpoint))}
    save_json(output/'summary.json',summary)
    save_json(output/'per_utterance.json', details)
    write_csv(output/'per_utterance.csv',rows)
    save_json(output/'completed.json',{'status':'COMPLETED','summary_sha256':sha256(output/'summary.json')})
    return summary


def run_supplement(output=DEFAULT/'supplement', limit=100):
    """Same-mask O8 and per-boundary strata without changing frozen main run."""
    from collections import Counter
    from ..frontend.phrase_plan import make_frontend
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'completed.json').exists():raise ValueError('Frozen supplement exists')
    sources=[r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()) if r['split']=='train']
    freq=Counter()
    for row in sources:
        syllables=row['frontend_metadata']['syllables']
        for s in syllables:
            freq.update(zip(s['ipa'][:-1],s['ipa'][1:]))
        for l,r in zip(syllables[:-1],syllables[1:]):
            if l['boundary_after_level']==0:
                freq[(l['ipa'][-1],r['ipa'][0])]+=1
    adapter=GenericDSPAdapter(24,5)
    oracle_streams=defaultdict(list)
    predicted_streams=defaultdict(list)
    rows=[]
    aud=[]
    for row,info,arrays,reference in frozen_rows(limit):
        controls,valid,matched,exact,issues=true_axis_controls(row,info,arrays,reference,arrays['original_controls'])
        dur,phonetic=reference['target_duration'],reference['phonetic_features']
        predicted=expand_controls(controls,dur)
        oracle=expand_controls(reference['target_controls'],dur)
        truth=reference['target_curve'].copy()
        truth[:,30]=np.where(truth[:,30]>.5,6.,-6.)
        masks,_=collect_regions(row,reference,valid,matched)
        for name,mask in masks.items():
            oracle_streams[name].append((oracle,truth,mask))
            predicted_streams[name].append((predicted,truth,mask))
        offsets=np.r_[0,np.cumsum(dur)]
        unit_chars={}
        labels={}
        for char_index,s in enumerate(row['frontend_metadata']['syllables']):
            for i,label in zip(range(s['phone_start'],s['phone_end']),s['ipa']):
                unit_chars[i]=char_index
                labels[i]=label
        active=np.flatnonzero(dur>0)
        for left,right in zip(active[:-1],active[1:]):
            if not matched[left] or not matched[right]:continue
            b=int(offsets[right])
            if not 0<b<len(truth):continue
            start=max(int(offsets[left]),b-4)
            end=min(int(offsets[right+1]),b+4)
            mask=np.zeros(len(truth),bool)
            mask[start:end]=True
            mask &= valid
            silence=bool(phonetic[left,0]>.5 or phonetic[right,0]>.5)
            cross=unit_chars.get(left,-1)!=unit_chars.get(right,-1)
            vc=bool(cross and phonetic[left,1]>.5 and phonetic[right,0]<.5 and phonetic[right,1]<.5)
            count=freq[(labels.get(left,'SIL'),labels.get(right,'SIL'))]
            base={'id':row['id'],'left_unit':int(left),'right_unit':int(right),'frame':b,
                'left_phone':labels.get(left,'SIL'),'right_phone':labels.get(right,'SIL'),
                'boundary_kind':'cross_character_VC' if vc else ('same_character' if not cross else 'other_cross_phone'),
                'right_manner_index':int(phonetic[right,:7].argmax()),
                'phone_pair_train_text_count':int(count),'frequency_group':'rare_lt20' if count<20 else 'common_ge20',
                'shorter_side_frames':int(min(dur[left],dur[right])),
                'duration_group':'short_le4frames' if min(dur[left],dur[right])<=4 else 'long_gt4frames',
                'pause_adjacent':silence,'window_frames':int(mask.sum())}
            for name,curve in [('P8_D1',predicted),('O8',oracle)]:
                metric=trajectory_metrics(curve,truth,mask)
                delta=metric['delta_10ms']
                rows.append({**base,'system':name,'shape_RMSE_db':metric['retained_shape_envelope_RMSE_db'],
                    'direction_cosine':delta['direction_cosine'],'direction_coverage':delta['coverage'],
                    'negative_cosine_fraction':delta['negative_cosine_fraction'],'magnitude_MAE':delta['magnitude_MAE']})
        if exact and len(aud)<12:
            pcm,_=adapter.render(physical_frames(arrays['curve']))
            _,original=wavfile.read(BASE/info['wav'])
            aud.append({'id':row['id'],'PCM_byte_exact':bool(np.array_equal(pcm,original)),
                'samples':len(pcm),'baseline_wav_sha256':info['wav_sha256']})
    strata={}
    for key in ('boundary_kind','right_manner_index','frequency_group','duration_group','pause_adjacent'):
        strata[key]={}
        for value in sorted({str(r[key]) for r in rows}):
            strata[key][value]={}
            for system in ('P8_D1','O8'):
                parts=[r for r in rows if str(r[key])==value and r['system']==system]
                metrics={metric:float(np.mean([r[metric] for r in parts if r[metric] is not None]))
                    for metric in ('shape_RMSE_db','direction_cosine','direction_coverage','negative_cosine_fraction','magnitude_MAE')}
                strata[key][value][system]={'boundaries':len(parts),'utterances':len({r['id'] for r in parts}),**metrics}
    summary={'status':'COMPLETED','same_mask_O8':metric_summary(oracle_streams),
        'same_mask_P8_D1':metric_summary(predicted_streams),'strata_event_weighted':strata,
        'frequency_definition':'training-only frontend IPA adjacency text count; rare<20 fixed before validation aggregation',
        'strata_caveat':'event-weighted overlapping windows; descriptive only, not independent observations or significance',
        'natural_baseline_fixed_DSP_rebuild':aud,'test_loaded':False,'source_sha256':sha256(Path(__file__))}
    if not all(x['PCM_byte_exact'] for x in aud):raise AssertionError('Frozen baseline DSP differs')
    write_csv(output/'per_boundary.csv',rows)
    save_json(output/'summary.json',summary)
    shutil.copy2(__file__,output/'source_snapshot.py')
    save_json(output/'completed.json',{'status':'COMPLETED','boundary_rows':len(rows),'baseline_DSP_exact':len(aud),
        'summary_sha256':sha256(output/'summary.json')})
    print(json.dumps({'status':'COMPLETED','output':str(output),'boundaries':len(rows)//2,'baseline_DSP_exact':len(aud)}),flush=True)
    return summary


def audio_metrics(pcm):
    x = pcm.astype(np.float64)/32768
    hz, power = welch(x, fs=16000, nperseg=min(1024,len(x)), noverlap=512)
    bands = {}
    for name,lo,hi in [('low',80,1000),('mid',1000,4000),('high',4000,8001)]:
        bands[name] = float(power[(hz>=lo)&(hz<hi)].sum())
    energy = np.array([np.mean(part**2) for part in np.array_split(x[:len(x)//320*320],len(x)//320)])
    active = energy > max(float(energy.max())*.001,1e-12)
    jumps = 10*np.log10(np.maximum(energy[1:],1e-12)/np.maximum(energy[:-1],1e-12))
    both = active[1:] & active[:-1]
    return {'peak':float(abs(x).max()),'RMS_dbFS':float(10*np.log10(max(np.mean(x*x),1e-12))),
        'clipping_fraction':float(np.mean((pcm==32767)|(pcm==-32768))),
        'band_energy':bands,'high_to_low_db':float(10*np.log10(max(bands['high'],1e-20)/max(bands['low'],1e-20))),
        'active_20ms_energy_up_jumps_gt6dB':int(np.sum(jumps[both]>6)),
        'max_active_20ms_energy_up_jump_db':float(np.max(jumps[both])) if both.any() else None}


def run(output=DEFAULT, limit=100, listening_count=12):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'completed.json').exists():
        raise ValueError('Frozen diagnostics already exist')
    save_json(output/'status.json',{'status':'RUNNING'})
    if sha256(ACOUSTIC)!=EXPECTED_ACOUSTIC or sha256(PLANNER)!=EXPECTED_PLANNER:
        raise ValueError('Frozen checkpoint identity differs')
    config = {'status':'RUNNING','seed':20261015,'ids':[r[0]['id'] for r in frozen_rows(limit)],
        'historical_axis':'P8 predicted durations; per-phone j/n reference resampling; O8 true durations',
        'new_axis':'true reference durations; unchanged ordinary predicted controls; no time-aligned natural score',
        'shape_domain':'DCT-II(log WORLD spectral envelope)[1:24]/sqrt(513); 10/ln(10) Parseval dB',
        'direction':'23 raw physical shape coefficients; reference L2>.05; denominator pn*rn+1e-12',
        'direction_spans_frames':[1,2,4],'historical_VC_window_frames_each_side':3,
        'new_all_boundary_window_frames_each_side':4,'frame_ms':10,'sample_rate':16000,
        'nearzero_predicted_direction':'retained with epsilon denominator, not dropped',
        'normalization_changed':False,'test_loaded':False,'MOS_measured':False,
        'DSP':{'taps':128,'seed':1729,'gain':1.,'dim':24,'AP_bands':5},
        'loudness':'whole-utterance unweighted RMS target -24dBFS, shared per-sentence peak attenuation to .98; no per-frame gains',
        'source_sha256':sha256(Path(__file__)),'acoustic_sha256':sha256(ACOUSTIC),
        'planner_sha256':sha256(PLANNER),'data_manifest_sha256':sha256(DATA/'manifest.jsonl'),
        'baseline_manifest_sha256':sha256(BASE/'manifest.json'),
        'DSP_source_sha256':sha256(PROJECT/'src/audio/native.c')}
    save_json(output/'config.json', config)
    shutil.copy2(__file__, output/'source_snapshot.py')
    adapter = GenericDSPAdapter(24,5)
    old_streams = defaultdict(list)
    o8_streams = defaultdict(list)
    d1_streams = defaultdict(list)
    records, csvrows, key, blind_rows = [], [], [], []
    rng = random.Random(config['seed'])
    html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>参数组盲听</title>',
        '<h1>同句参数组盲听</h1><p>所有匿名样本使用真实音素时长和相同DSP，仅供诊断。请分别评价清晰度、过渡自然度、刺耳程度／舒适度（1–5），并注明设备。逐句响度匹配；没有自动试听结论。</p>']
    for index,(row,info,arrays,reference) in enumerate(frozen_rows(limit)):
        uid = row['id']
        mapping, issues = matched_unit_map(info['frontend_metadata']['syllables'],row['frontend_metadata']['syllables'],
            arrays['phonetic_features'],reference['phonetic_features'])
        mapped, oldvalid = reference_on_prediction(reference['target_curve'],reference['target_duration'],
            arrays['durations'],mapping,arrays['curve'])
        oldmatched=(mapping>=0)&np.asarray([reference['target_duration'][j]>0 if j>=0 else False for j in mapping])
        oldmasks,_ = segment_masks(info['frontend_metadata']['syllables'],arrays['phonetic_features'],arrays['durations'],oldmatched)
        for name,mask in oldmasks.items():
            old_streams[name].append((arrays['curve'],mapped,mask & oldvalid))
        truth = reference['target_curve'].copy()
        truth[:,30] = np.where(truth[:,30]>.5,6.,-6.)
        oracle = expand_controls(reference['target_controls'],reference['target_duration'])
        omasks,_ = collect_regions(row,reference,np.ones(len(truth),bool),np.ones(len(reference['target_duration']),bool))
        for name,mask in omasks.items():
            o8_streams[name].append((oracle,truth,mask))
        controls, valid, matched, exact, _ = true_axis_controls(row,info,arrays,reference,arrays['original_controls'])
        predicted = expand_controls(controls,reference['target_duration'])
        masks,events = collect_regions(row,reference,valid,matched)
        result={'id':uid,'text':row['text'],'exact_numeric_topology':exact,'mapping_issues':issues,
            'source_record_sha256':row['record_sha256'],'matched_frames':int(valid.sum()),'total_frames':len(valid),
            'VC_events':len(events),'regions':{},'audio':{}}
        for name,mask in masks.items():
            d1_streams[name].append((predicted,truth,mask))
            result['regions'][name]=trajectory_metrics(predicted,truth,mask)
            csvrows.append({'id':uid,'region':name,**{k:v for k,v in result['regions'][name].items() if not isinstance(v,dict)}})
        if exact and len(blind_rows)<listening_count:
            folder=output/'samples'/uid
            folder.mkdir(parents=True,exist_ok=True)
            # Fixed deployment F0 guard, same true durations. It never changes spectra.
            features={'phonetic_features':arrays['phonetic_features'],'context_features':arrays['context_features'],
                'syllable_phone_ranges':[(s['phone_start'],s['phone_end']) for s in info['frontend_metadata']['syllables']]}
            p,_,_=apply_profile(features,{'curve':predicted,'durations':reference['target_duration']},True)
            curves={'H0':p,'H1':oracle,'R':truth}
            for name,channels in {'H2':list(range(2,25)),'H3':[0,*range(25,31)],'H4':[1],
                                  'H5':[0,*range(2,31)]}.items():
                curve=p.copy()
                curve[:,channels]=oracle[:,channels]
                curves[name]=curve
            pcms={}
            for name,curve in curves.items():
                pcm,stats=adapter.render(physical_frames(curve))
                path=folder/(name+'_raw.wav')
                wavfile.write(path,16000,pcm)
                result['audio'][name]={'raw':str(path.relative_to(output)),'raw_sha256':sha256(path),
                    'raw_metrics':audio_metrics(pcm),'DSP':stats,'oracle':name!='H0'}
                pcms[name]=pcm
            gains={n:10**(-24/20)/max(np.sqrt(np.mean((pcm.astype(np.float64)/32768)**2)),1e-12) for n,pcm in pcms.items()}
            largest=max(float(abs(pcm.astype(np.float64)/32768).max())*gains[n] for n,pcm in pcms.items())
            shared=min(1.,.98/max(largest,1e-12))
            order=list(pcms)
            rng.shuffle(order)
            html.append('<h2>'+escape(uid+' '+row['text'])+'</h2>')
            for column,name in enumerate(order):
                label=f'S{len(blind_rows)+1:02}_{chr(65+column)}'
                pcm=pcms[name]
                gain=gains[name]*shared
                matched_pcm=np.clip(np.rint(pcm.astype(np.float64)*gain),-32768,32767).astype(np.int16)
                path=folder/(label+'.wav')
                wavfile.write(path,16000,matched_pcm)
                result['audio'][name].update(matched=str(path.relative_to(output)),matched_sha256=sha256(path),
                    gain_db=float(20*np.log10(gain)),matched_metrics=audio_metrics(matched_pcm))
                html.append('<p>'+label+'<br><audio controls preload="none" src="'+escape(str(path.relative_to(output)))+'"></audio></p>')
                key.append({'label':label,'id':uid,'system':name})
            blind_rows.append(uid)
            np.savez_compressed(folder/'same_axis_curves.npz',durations=reference['target_duration'],**curves)
            save_json(folder/'metrics.json',result)
        records.append(result)
        if index%10==0:
            print(json.dumps({'stage':'diagnostics','utterances':index+1,'listening':len(blind_rows)}),flush=True)
    (output/'index.html').write_text('\n'.join(html+['</html>']),encoding='utf-8')
    save_json(output/'answer_key_private.json',key)
    write_csv(output/'ratings.csv',[{'label':x['label'],'clarity_1_5':'','transition_1_5':'','comfort_1_5':'','device':'','comment':''} for x in key])
    save_json(output/'per_utterance.json',records)
    write_csv(output/'per_utterance.csv',csvrows)
    summaries={'historical_P8_predicted_axis':metric_summary(old_streams),
        'O8_true_axis':metric_summary(o8_streams),'P8_D1_true_axis_matched':metric_summary(d1_streams)}
    old=json.loads((HISTORY/'summary.json').read_text())['regions']
    old_o8=json.loads((HISTORY/'native_floor_summary.json').read_text())['same100_true_duration_fitted8knot_oracle']
    checks=[]
    for region in old:
        if not isinstance(old[region],dict):continue
        for metric in ('retained_shape_envelope_RMSE_db','shape_delta_direction_cosine'):
            for name,target in [('historical_P8_predicted_axis',old),('O8_true_axis',old_o8)]:
                actual=summaries[name][region][metric]
                previous=target[region][metric]
                checks.append({'system':name,'region':region,'metric':metric,'previous':previous,
                               'actual':actual,'absolute_difference':abs(actual-previous)})
    summary={'status':'COMPLETED','utterances':len(records),'listening_ids':blind_rows,
        'systems':summaries,'historical_reproduction':checks,
        'historical_numeric_match_max_error':max(x['absolute_difference'] for x in checks),
        'historical_comparison_warning':'Old P8 and O8 use different timelines and masks; not a strict same-axis headroom comparison',
        'audio_count':len(key)*2,'listening_ratings':'NOT_RUN: no human ratings','test_loaded':False,
        'natural_inference_gain':'NOT_RUN: only unchanged old natural baseline and explicit oracle diagnostics'}
    save_json(output/'summary.json',summary)
    links=re.findall('src="([^"]+)"',(output/'index.html').read_text())
    if not all((output/p).is_file() for p in links):raise ValueError('Missing listening links')
    all_audio=[(name,rec['audio'][name]) for rec in records for name in rec['audio']]
    audio_summary={name:{'utterances':len([x for n,x in all_audio if n==name]),
        'mean_raw_high_to_low_db':float(np.mean([x['raw_metrics']['high_to_low_db'] for n,x in all_audio if n==name])),
        'raw_clipping_frames':sum(x['raw_metrics']['clipping_fraction']>0 for n,x in all_audio if n==name),
        'matched_mean_RMS_dbFS':float(np.mean([x['matched_metrics']['RMS_dbFS'] for n,x in all_audio if n==name]))}
        for name in ('H0','H1','H2','H3','H4','H5','R')}
    save_json(output/'audio_summary.json',audio_summary)
    a=summaries['historical_P8_predicted_axis']['VC_boundary_pm3']
    b=summaries['O8_true_axis']['VC_boundary_pm3']
    c=summaries['P8_D1_true_axis_matched']['VC_boundary_pm3']
    report=f'''# 实际完成的接缝与参数替换诊断

状态 COMPLETED；100句为先前已使用validation，不是独立test。单线程CPU；未训练模型、未更新planner、未修改DSP、未人工评分。

历史口径重算：P8 VC谱RMSE {a['retained_shape_envelope_RMSE_db']:.6f} dB，方向 {a['shape_delta_direction_cosine']:.6f}；O8 {b['retained_shape_envelope_RMSE_db']:.6f} dB，方向 {b['shape_delta_direction_cosine']:.6f}。对历史保存值最大绝对差 {summary['historical_numeric_match_max_error']:.3g}。

历史P8使用预测时长、逐音素j/n重采样真值；O8使用真实时长。二者窗口物理位置与覆盖不同，不能将数字差全归预测器。新D1沿真实时长、保留原预测控制点与普通文字planner输入：匹配VC谱RMSE {c['retained_shape_envelope_RMSE_db']:.6f} dB，方向 {c['shape_delta_direction_cosine']:.6f}，10ms方向覆盖 {c['delta_10ms']['coverage']:.3%}，负方向比例 {c['delta_10ms']['negative_cosine_fraction']:.3%}。未匹配帧全部排除。模型时长是并行输出，不是控制点输入，D2不适用。

谱域为WORLD包络自然对数DCT-II的1–23阶除sqrt(513)，保留谱域Parseval RMSE乘10/ln(10)，不是完整STFT误差。方向真值范数阈值固定0.05；预测零向量不剔除。增补10/20/40ms指标与历史指标并列。

盲听选冻结100句中最先{len(blind_rows)}个全音素数值拓扑一致句；完整100句指标未据此筛选。H0真实时长预测；H1全O8；H2仅谱O8；H3 F0/AP5/VUV一起O8；H4仅能量O8；H5谱加激励O8；R真实逐帧。所有使用相同128tap、seed1729、gain1 C DSP和物理VUV/AP适配。原始与整句RMS匹配各{len(key)}份，共{len(key)*2}份WAV；匹配算法和每句增益见config/per_utterance。匿名顺序和问题分别为清晰度、过渡、舒适度；答案单独answer_key_private.json。未加入低通、降音量方案或POSTFILTER。

audio_summary.json只提供频带能量线索；频带变化、零削波不能证明柔和或更清晰。尚需人工试听，未编造偏好或MOS。没有新增自然推理收益结论。

入口：`python -m project.src.experiments.boundary_trajectory_diagnostics --output ...`。evaluate_checkpoint可复用冻结普通输入评估候选D1；自定义predictor只接收普通预测数组，不接收参考。
'''
    (output/'REPORT.md').write_text(report,encoding='utf-8')
    if not (output/'FAILED_STEP.md').exists():
        (output/'FAILED_STEP.md').write_text('No failed execution steps in completed diagnostic run. Human listening NOT_RUN.\n')
    completed={'status':'COMPLETED','wall_seconds':time.monotonic()-started,
        'utterances':len(records),'WAVs':len(key)*2,'HTML_links_verified':len(links),
        'summary_sha256':sha256(output/'summary.json'),'source_sha256':sha256(Path(__file__))}
    save_json(output/'completed.json',completed)
    save_json(output/'status.json',completed)
    print(json.dumps(completed),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--listening-count',type=int,default=12)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--supplement',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.supplement:
        run_supplement(args.output/'supplement',args.limit)
    elif args.checkpoint:
        evaluate_checkpoint(args.checkpoint,args.output,args.limit)
    else:
        run(args.output,args.limit,args.listening_count)


if __name__=='__main__':
    main()
