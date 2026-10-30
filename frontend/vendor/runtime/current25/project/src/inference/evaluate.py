"""Frozen, fixed-split evaluation on the original absolute timeline (no DTW)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.fft import idct
from scipy.io import wavfile
import torch

from ..common import PROJECT, WORKSPACE, save_json, sha256
from ..audio.dsp import DSPAdapter, write_wav
from ..audio.extractor import read_audio
from ..models.acoustic import load_acoustic, expand_controls, resource_estimate
from ..models.residual import load_residual
from ..frontend import from_text


def timeline_metrics(predicted, target, durations=None, reference_durations=None):
    """Reference-length scoring: missing prediction = silence; overrun separately counted."""
    n = len(target)
    pred = np.zeros((n, 29), np.float32)
    pred[:, 1] = np.log(1e-12)
    pred[:, 28] = -12
    pred[:min(n, len(predicted))] = predicted[:n]
    target_voice, pred_voice = target[:, 28] > .5, pred[:, 28] > 0
    active = target[:, 1] > max(float(target[:, 1].max()) - np.log(1000), np.log(1e-10))
    joint = target_voice & pred_voice
    # DC-free reconstructed 513-bin log power shape plus measured log energy.
    # This is the compact representation's spectral envelope, not raw FFT LSD.
    modes = np.zeros((n, 513), np.float32)
    modes[:, 1:24] = (pred[:, 2:25] - target[:, 2:25]) * np.sqrt(513)
    db = (idct(modes, type=2, norm='ortho', axis=1) +
          (pred[:, 1] - target[:, 1])[:, None]) * (10 / np.log(10))
    cents = (pred[joint, 0] - target[joint, 0]) * (1200 / np.log(2))
    result = {'reference_frames': n, 'predicted_frames': len(predicted),
        'missing_frames_scored_as_silence': max(0, n-len(predicted)),
        'overrun_frames_not_in_reference_metric': max(0, len(predicted)-n),
        'active_frames': int(active.sum()), 'spectral_active_RMSE_db': float(np.sqrt(np.mean(db[active]**2))) if active.any() else None,
        'spectral_all_RMSE_db': float(np.sqrt(np.mean(db**2))),
        'energy_active_RMSE_db': float(np.sqrt(np.mean(((pred[active, 1]-target[active, 1])*10/np.log(10))**2))) if active.any() else None,
        'AP3_active_RMSE': float(np.sqrt(np.mean((pred[active, 25:28]-target[active, 25:28])**2))) if active.any() else None,
        'VUV_error_rate': float(np.mean(target_voice != pred_voice)),
        'reference_voiced_frames': int(target_voice.sum()), 'joint_voiced_frames': int(joint.sum()),
        'F0_joint_voiced_RMSE_cents': float(np.sqrt(np.mean(cents**2))) if len(cents) else None,
        'F0_joint_voiced_MAE_cents': float(np.mean(np.abs(cents))) if len(cents) else None,
        'joint_voiced_reference_coverage': float(joint.sum()/target_voice.sum()) if target_voice.any() else None,
        'predicted_voiced_fraction': float(pred_voice.mean()),
        'sentence_duration_error_ms': 10 * (len(predicted)-n),
        'sentence_duration_absolute_error_ms': 10 * abs(len(predicted)-n),
        'sentence_duration_absolute_relative_error': abs(len(predicted)-n)/n,
        'alignment': 'absolute reference timeline, zero time warp; pad short predictions as silence and clip long predictions; overrun separately reported'}
    if durations is not None and reference_durations is not None:
        if len(durations) != len(reference_durations):
            raise ValueError('Phone sequence mismatch: boundary comparison unavailable')
        boundary = np.cumsum(durations)[:-1] - np.cumsum(reference_durations)[:-1]
        result['internal_phone_boundary_MAE_ms'] = float(np.abs(boundary).mean()*10) if len(boundary) else 0.
        result['phone_duration_MAE_ms'] = float(np.abs(durations-reference_durations).mean()*10)
    return result


def waveform_metrics(pred_pcm, original):
    predicted = pred_pcm.astype(np.float64)/32768
    aligned = np.zeros(len(original), np.float64)
    aligned[:min(len(original), len(predicted))] = predicted[:len(original)]
    metrics = {'RMS': float(np.sqrt(np.mean(predicted**2))),
        'clipping_fraction': float(np.mean((pred_pcm == 32767)|(pred_pcm == -32768))),
        'waveform_alignment': 'pad/clip at original sample count; no shift/time warp'}
    try:
        sys.path.append(str(WORKSPACE/'unit_experiment_deps'))
        from pystoi import stoi
        metrics['STOI_original_timeline'] = float(stoi(original, aligned, 16000, extended=False))
    except ImportError:
        metrics['STOI_original_timeline'] = None
        metrics['STOI_status'] = 'pystoi not installed'
    return metrics


def aggregate(rows):
    result = {}
    for name in sorted({r['system'] for r in rows}):
        part = [r for r in rows if r['system'] == name]
        keys = sorted({k for r in part for k,v in r.items() if isinstance(v, (int,float)) and not isinstance(v,bool)})
        result[name] = {'count': len(part), **{k:float(np.mean([r[k] for r in part if r.get(k) is not None])) for k in keys}}
        result[name]['aggregation'] = 'unweighted utterance means; ASR uses pooled corpus edit counts in its separate report'
    return result


def run(args):
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(x) for x in (args.data/'manifest.jsonl').read_text().splitlines()]
    by_id = {r['id']:r for r in all_rows}
    ids = json.loads((args.data/'split.json').read_text())[args.split]
    if args.limit:
        ids = ids[:args.limit]
    snapshots, models = {}, {}
    for cp in args.checkpoint:
        name = cp.parent.name
        snapshot = output/'snapshots'/name/cp.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copyfile(cp, snapshot)
        digest = sha256(snapshot)
        models[name] = load_acoustic(snapshot)
        state = torch.load(snapshot, map_location='cpu', weights_only=True)
        snapshots[name] = {'path':str(snapshot), 'sha256':digest, 'source':str(cp),
            'training_step':state.get('step'), 'checkpoint_saved_utc':state.get('saved_utc'),
            'training_validation':state.get('validation'), 'training_lineage':state.get('lineage'),
            'resources':resource_estimate(models[name])}
    residual, alpha = load_residual(args.residual) if args.residual else (None, 0.)
    residual_model = None
    if residual is not None:
        matching = [name for name, snapshot in snapshots.items() if snapshot['sha256'] == residual.baseline_sha256]
        if len(matching) != 1:
            raise ValueError('Residual must match exactly one frozen acoustic model SHA256')
        residual_model = matching[0]
    config = {'created_utc':datetime.now(timezone.utc).isoformat(), 'data':str(args.data.resolve()),
        'manifest_sha256':sha256(args.data/'manifest.jsonl'), 'split_sha256':sha256(args.data/'split.json'),
        'split':args.split, 'ids':ids, 'checkpoint_snapshots':snapshots,
        'residual':str(args.residual) if args.residual else None, 'residual_sha256':sha256(args.residual) if args.residual else None,
        'residual_alpha':alpha, 'residual_model':residual_model, 'threads':args.threads, 'DSP':{'gain':1.,'seed':1729,'taps':args.taps},
        'diagnostic_true_duration':args.true_duration, 'ordinary_inference_uses_only_text':True,
        'spectral_metric':'24 channel energy + DC-free DCT23 compact envelope reconstructed to513bins, dB RMSE on reference active(-30dB relative peak) frames',
        'MOS':'not measured; objective metrics and ASR do not estimate subjective MOS',
        'test_is_for_final_locked_model_only':args.split=='test'}
    save_json(output/'config.json', config)
    dsp = DSPAdapter(taps=args.taps)
    results = []
    start = time.monotonic()
    for number, identity in enumerate(ids, 1):
        row = by_id[identity]
        directory = output/'samples'/identity
        directory.mkdir(parents=True, exist_ok=True)
        with np.load(args.data/row['record_path'], allow_pickle=False) as z:
            target = z['target_curve'].copy()
            true_duration = z['target_duration'].copy()
            phone_ids = z['phone_ids'].copy()
        original = read_audio(row['source_audio'])
        original_pcm = np.rint(original*32768).clip(-32768,32767).astype(np.int16)
        write_wav(directory/'original.wav', original_pcm)
        save_json(directory/'reference.json', {'id':identity,'text':row['text'], 'split':args.split,
            'source_audio':row['source_audio'],'source_audio_sha256':sha256(row['source_audio']),
            'target_record_sha256':sha256(args.data/row['record_path'])})
        oracle, oracle_dsp = dsp.render(target, vuv='probability', sample_count=len(original))
        write_wav(directory/'real_parameter_oracle.wav', oracle)
        oracle_metric = {'id':identity,'system':'real_parameter_oracle','uses_reference_acoustics':True,
            'uses_reference_duration':True, **waveform_metrics(oracle, original), 'DSP_RTF':oracle_dsp['RTF']}
        save_json(directory/'real_parameter_oracle.metrics.json', oracle_metric)
        results.append(oracle_metric)
        features = from_text(row['text'])
        if not np.array_equal(features['phone_ids'], phone_ids):
            raise ValueError(f'{identity}: text-only frontend phone IDs differ from audited record')
        for name,model in models.items():
            tick = time.perf_counter()
            predicted = model.predict_features(features)
            acoustic_wall = time.perf_counter()-tick
            versions = [(name, predicted['controls'], predicted['durations'], False)]
            if args.true_duration:
                versions.append((name+'__true_duration_diagnostic', predicted['controls'], true_duration, True))
            if residual is not None and name == residual_model:
                versions.append((name+'__residual', residual.correct(predicted['controls'], alpha), predicted['durations'], False))
                if args.true_duration:
                    versions.append((name+'__residual_true_duration_diagnostic', residual.correct(predicted['controls'], alpha), true_duration, True))
            for system,controls,durations,diagnostic in versions:
                curve = expand_controls(controls,durations)
                pcm, dsp_metric = dsp.render(curve)
                write_wav(directory/(system+'.wav'), pcm)
                np.savez_compressed(directory/(system+'.npz'), controls=controls,durations=durations,curve=curve)
                metrics = {'id':identity,'system':system,'uses_reference_acoustics':False,
                    'uses_reference_duration':diagnostic,'ordinary_text_only_inference':not diagnostic,
                    'checkpoint_sha256':snapshots[name]['sha256'],
                    **timeline_metrics(curve,target,durations,true_duration), **waveform_metrics(pcm,original),
                    'acoustic_seconds':acoustic_wall,'DSP_RTF':dsp_metric['RTF'],
                    'wav_sha256':sha256(directory/(system+'.wav'))}
                save_json(directory/(system+'.metrics.json'),metrics)
                results.append(metrics)
        save_json(output/'per_utterance_metrics.json', results)
        save_json(output/'summary.json',{'config':config,'systems':aggregate(results),'completed_utterances':number,
            'total_utterances':len(ids),'wall_seconds':time.monotonic()-start})
        print(json.dumps({'completed':number,'total':len(ids),'id':identity,'wall_seconds':time.monotonic()-start}),flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=PROJECT/'outputs/data_large')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--residual',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--true-duration',action='store_true')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--taps',type=int,default=128)
    run(p.parse_args())


if __name__ == '__main__':
    main()
