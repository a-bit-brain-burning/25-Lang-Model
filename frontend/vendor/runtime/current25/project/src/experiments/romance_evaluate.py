"""Validation-only Romance39 diagnostics on explicitly separated time axes.

No training loop is imported. Reference-duration curves use unchanged predicted
control points, not re-prediction with true duration. Natural predicted-duration
curves receive unit-duration statistics only: no unaligned frame spectral loss.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
import csv
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from ..common import save_json, sha256
from ..models.universal_acoustic import UniversalAcousticModel, expand_controls
from ..models import romance_word_planner as wp
from .eu_quality_spectrum_basis import PhysicalSpectrum
from .romance_listen import ACOUSTIC_FORMAT, load_system

DB = 10/math.log(10)
CENTS = 1200/math.log(2)
EPS = 1e-6
PROTOCOL = {
    'schema': 'romance_validation_physical513_events_v1',
    'split': 'validation only; no test or train recordings decoded',
    'aligned_axis': 'D1_REFERENCE_DURATION_UNCHANGED_PREDICTED_CONTROLS_NOT_NATURAL',
    'natural_axis': 'ORDINARY_TEXT_PREDICTED_DURATION_NO_UNALIGNED_FRAME_ERROR',
    'representation_floor': 'raw O8 controls expanded on reference durations -> original O_frame',
    'spectrum': 'PhysicalSpectrum(32,700) evaluated in FP64,513 bins,10/ln10 logpower dB; waveform energy normalization',
    'shape_direction': 'normalized physical513 logpower minus log waveform energy, converted to dB',
    'boundaries': 'adjacent positive-duration units; union [b-4,b+4) frames; overlaps counted once per region',
    'delta': 'k=1,2,4 frames; all k+1 intervening frames must remain inside region',
    'cosine': 'reference L2 change norm>1e-6 dB; predicted norm<=1e-6 retained with cosine0; coverage/negative fraction reported',
    'magnitude': 'absolute difference of change L2 norms divided by sqrt513, dB RMS units',
    'F0': 'trusted reference voiced/F0-confidence mask; values and differences retain prediction errors even when predicted VUV is wrong; joint-voiced value result additionally reported',
    'plan': 'only text-predicted plan enters inference; reference plan constructed afterward for diagnostics using frozen train baselines',
    'history': 'new physical513 protocol; do not mix with historical coefficient-domain seam results',
    'human_listening': 'NOT_RUN', 'MOS': 'NOT_RUN', 'test': 'NOT_RUN',
}


def scalar(value):
    return float(value) if np.isfinite(value) else None


def pair_mask(mask, stride):
    mask = np.asarray(mask, bool)
    if len(mask) <= stride:
        return np.zeros(0, bool)
    result = np.ones(len(mask)-stride, bool)
    for shift in range(stride+1):
        result &= mask[shift:shift+len(result)]
    return result


def boundary_regions(duration, front):
    duration = np.asarray(duration, np.int64)
    p = np.asarray(front['phonetic_features'])
    word = np.asarray(front['lexical_word_indices'])
    if duration.shape != (len(p),) or word.shape != duration.shape or np.any(duration < 0):
        raise ValueError('Invalid reference boundary topology')
    frames = int(duration.sum())
    boundaries = np.r_[0, duration.cumsum()]
    active = np.flatnonzero(duration > 0)
    masks = {'all': np.zeros(frames, bool)}
    counts = {'all': 0}
    items = []
    for left, right in zip(active[:-1], active[1:]):
        b = int(boundaries[right])
        if p[left, 0] > .5 or p[right, 0] > .5:
            kind = 'pause_adjacent'
        elif p[left, 1] > .5 and p[right, 2:5].sum() > .5:
            kind = 'vowel_to_voiced_obstruent' if p[right, 17] > .5 else 'vowel_to_unvoiced_obstruent'
        elif p[left, 1] > .5 and p[right, 5] > .5:
            kind = 'vowel_to_nasal'
        elif p[left, 1] > .5 and p[right, 1] <= .5:
            kind = 'vowel_to_other_consonant'
        elif p[left, 1] <= .5 and p[right, 1] > .5:
            kind = 'consonant_to_vowel'
        elif p[left, 1] > .5 and p[right, 1] > .5:
            kind = 'vowel_to_vowel'
        else:
            kind = 'other'
        regions = ['all', kind]
        if word[left] >= 0 and word[right] >= 0:
            regions.append('cross_word' if word[left] != word[right] else 'within_word')
        for name in regions:
            masks.setdefault(name, np.zeros(frames, bool))[max(0, b-4):min(frames, b+4)] = True
            counts[name] = counts.get(name, 0)+1
        items.append({'frame': b, 'left_unit': int(left), 'right_unit': int(right), 'kind': kind,
                      'cross_word': bool(word[left] >= 0 and word[right] >= 0 and word[left] != word[right])})
    return masks, counts, items


@torch.inference_mode()
def physical_logpower(curve):
    curve = np.asarray(curve, np.float32)
    if curve.ndim != 2 or curve.shape[1] != 39 or not len(curve) or not np.isfinite(curve).all():
        raise ValueError('Expected finite nonempty neural39 curve')
    # FP64 metric reduction avoids FP32 DC-cancellation roundoff creating a
    # spurious >1e-6 change norm in an otherwise constant spectral trajectory.
    return PhysicalSpectrum(32, 700.).double()(torch.from_numpy(curve[:, 1:33].astype(np.float64))).numpy()


def trajectory_metrics(pred_power, ref_power, pred_energy, ref_energy, region):
    p, r = np.asarray(pred_power, np.float64), np.asarray(ref_power, np.float64)
    region = np.asarray(region, bool)
    if p.shape != r.shape or p.ndim != 2 or p.shape[1] != 513 or region.shape != (len(p),):
        raise ValueError('Trajectory errors require the same 513-bin reference frame axis')
    shape_p = (p-np.asarray(pred_energy)[:, None])*DB
    shape_r = (r-np.asarray(ref_energy)[:, None])*DB
    result = {'frames': int(region.sum()),
        'physical_RMSE_dB': scalar(np.sqrt(np.mean(((p-r)[region]*DB)**2))) if region.any() else None,
        'shape_RMSE_dB': scalar(np.sqrt(np.mean((shape_p-shape_r)[region]**2))) if region.any() else None,
        'shape_3k8k_RMSE_dB': scalar(np.sqrt(np.mean((shape_p-shape_r)[region, 193:]**2))) if region.any() else None}
    for k in (1, 2, 4):
        valid = pair_mask(region, k)
        dp, dr = shape_p[k:]-shape_p[:-k], shape_r[k:]-shape_r[:-k]
        pn, rn = np.linalg.norm(dp, axis=-1), np.linalg.norm(dr, axis=-1)
        eligible = valid & (rn > EPS)
        cosine = np.zeros(len(dp), np.float64)
        nonzero = eligible & (pn > EPS)
        cosine[nonzero] = np.clip((dp[nonzero]*dr[nonzero]).sum(1)/(pn[nonzero]*rn[nonzero]), -1., 1.)
        tag = f'delta{k}_'
        result.update({tag+'eligible_pairs': int(valid.sum()), tag+'direction_pairs': int(eligible.sum()),
            tag+'direction_coverage': float(eligible.sum()/valid.sum()) if valid.any() else None,
            tag+'cosine': scalar(cosine[eligible].mean()) if eligible.any() else None,
            tag+'negative_fraction': float((cosine[eligible] < 0).mean()) if eligible.any() else None,
            tag+'predicted_nearzero_fraction': float((pn[eligible] <= EPS).mean()) if eligible.any() else None,
            tag+'change_RMSE_dB': scalar(np.sqrt(np.mean((dp-dr)[valid]**2))) if valid.any() else None,
            tag+'magnitude_MAE_dB_RMS': scalar(np.mean(abs(pn-rn)[valid])/math.sqrt(513)) if valid.any() else None,
            tag+'predicted_magnitude_dB_RMS': scalar(pn[valid].mean()/math.sqrt(513)) if valid.any() else None,
            tag+'reference_magnitude_dB_RMS': scalar(rn[valid].mean()/math.sqrt(513)) if valid.any() else None})
    return result


def duration_metrics(pred, ref, front):
    p, r = np.asarray(pred), np.asarray(ref)
    if p.shape != r.shape or p.shape != np.asarray(front['plan_owner']).shape or np.any(p < 0) or np.any(r < 0) or not r.sum():
        raise ValueError('Duration metrics need matching unit topology and positive reference time')
    lexical, pause = np.asarray(front['lexical_word_indices']) >= 0, np.asarray(front['internal_pause_mask'])
    return {'duration_units': len(p), 'duration_MAE_ms': float(abs(p-r).mean()*10),
        'lexical_duration_MAE_ms': float(abs(p-r)[lexical].mean()*10) if lexical.any() else None,
        'pause_duration_MAE_ms': float(abs(p-r)[pause].mean()*10) if pause.any() else None,
        'pause_presence_error': float(((p > 0) != (r > 0))[pause].mean()) if pause.any() else None,
        'sentence_duration_ratio': float(p.sum()/r.sum()), 'predicted_frames': int(p.sum()), 'reference_frames': int(r.sum())}


def parameter_metrics(pred, reference, duration, front, strict):
    p, r = np.asarray(pred), np.asarray(reference)
    if p.shape != r.shape or p.ndim != 2 or p.shape[1] != 39:
        raise ValueError('Parameter errors require the same reference time axis')
    raw_voice, voiced = r[:, 38] > .5, p[:, 38] > 0
    trusted = raw_voice & (r[:, 0] >= math.log(50)) & (r[:, 0] <= math.log(500)) & (r[:, 1]*DB >= -45)
    if strict is not None:
        trusted &= np.asarray(strict['frame_f0_valid'], bool)
    energy_difference = (p[:, 1]-r[:, 1])*DB
    pitch_difference = (p[:, 0]-r[:, 0])*CENTS
    joint = trusted & voiced
    lexical_frames = np.repeat(np.asarray(front['lexical_word_indices']) >= 0, duration)
    offsets = np.r_[0, np.asarray(duration).cumsum()]
    tail = np.zeros(len(p), bool)
    for first, stop in np.asarray(front['word_phone_ranges']):
        end = int(offsets[stop]); begin = max(int(offsets[first]), end-4)
        tail[begin:end] = True
    result = {'F0_trusted_frames': int(trusted.sum()), 'F0_trusted_coverage': float(trusted.mean()),
        'F0_RMSE_cents': scalar(np.sqrt(np.mean(pitch_difference[trusted]**2))) if trusted.any() else None,
        'F0_MAE_cents': scalar(abs(pitch_difference[trusted]).mean()) if trusted.any() else None,
        'F0_joint_voiced_frames': int(joint.sum()),
        'F0_joint_voiced_RMSE_cents': scalar(np.sqrt(np.mean(pitch_difference[joint]**2))) if joint.any() else None,
        'trusted_reference_missed_voicing_fraction': float((~voiced[trusted]).mean()) if trusted.any() else None,
        'energy_RMSE_dB': scalar(np.sqrt(np.mean(energy_difference**2))),
        'lexical_energy_RMSE_dB': scalar(np.sqrt(np.mean(energy_difference[lexical_frames]**2))) if lexical_frames.any() else None,
        'word_tail_energy_MAE_dB': scalar(abs(energy_difference[tail]).mean()) if tail.any() else None,
        'word_tail_frames': int(tail.sum()),
        'AP_MAE_normalized_log': scalar(abs(p[:, 33:38]-r[:, 33:38])[raw_voice].mean()) if raw_voice.any() else None,
        'AP_reference_voiced_frames': int(raw_voice.sum()), 'VUV_error': float((voiced != raw_voice).mean()),
        'VUV_false_positive': float(voiced[~raw_voice].mean()) if (~raw_voice).any() else None,
        'VUV_false_negative': float((~voiced[raw_voice]).mean()) if raw_voice.any() else None}
    if strict is not None:
        weight = np.asarray(strict['frame_voice_weight'])
        labels = np.asarray(strict['frame_voice']) > .5
        result['VUV_strict_weighted_error'] = float(np.sum(weight*(voiced != labels))/max(weight.sum(), 1e-12))
    for k in (1, 2, 4):
        valid = pair_mask(trusted, k)
        error = pitch_difference[k:]-pitch_difference[:-k]
        result[f'F0_delta{k}_trusted_pairs'] = int(valid.sum())
        result[f'F0_delta{k}_RMSE_cents'] = scalar(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None
    return result


def natural_statistics(curve):
    curve = np.asarray(curve)
    voiced = curve[:, 38] > 0
    return {'natural_frame_count': len(curve), 'natural_audio_seconds': len(curve)/100,
        'natural_voiced_fraction': float(voiced.mean()),
        'natural_voiced_mean_F0_Hz': scalar(np.exp(curve[voiced, 0]).mean()) if voiced.any() else None,
        'natural_mean_energy_dB': scalar(curve[:, 1].mean()*DB),
        'natural_AP_mean': scalar(curve[:, 33:38][voiced].mean()) if voiced.any() else None}


def load_comparison(path, language, front, planner_sha):
    state = torch.load(path, map_location='cpu', weights_only=True)
    lineage = state.get('lineage', state)
    if state.get('format') != ACOUSTIC_FORMAT or lineage.get('language', language) != language:
        raise ValueError('Comparison format/language mismatch')
    for key, value in (('lexicon_sha256', front.lexicon_sha256), ('dictionary_sha256', front.lexicon_sha256),
                       ('frontend_signature', front.signature), ('planner_sha256', planner_sha)):
        if key in lineage and lineage[key] != value:
            raise ValueError('Frozen comparison identity differs: '+key)
    config = state['config']
    if tuple(config.get(k) for k in ('phonetic_dim', 'context_dim', 'spectral_dim', 'ap_bands')) != (36, 27, 32, 5):
        raise ValueError('Comparison checkpoint must retain the same36/context27/warped39 schema')
    model = UniversalAcousticModel(config, state['model']['means'], state['model']['scales'])
    model.load_state_dict(state['model'], strict=True)
    return model.eval(), state


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)


def summarize(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for group, chosen in grouped.items():
        numeric = {k for row in chosen for k, v in row.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        means, counts = {}, {}
        for key in sorted(numeric):
            values = [row[key] for row in chosen if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])]
            if values:
                means[key] = float(np.mean(values))
                if key.endswith(('_frames', '_pairs', '_units')) or key in ('frames', 'boundary_count'):
                    counts[key] = sum(values)
        output.append({**dict(zip(group_keys, group)), 'recordings': len(chosen),
                       'mean_per_recording': means, 'support_totals': counts})
    return output


def evaluate(data, lexicon, training, language, output, *, initial=None, zero_control=False):
    torch.set_num_threads(1)
    data, training, output = map(Path, (data, training, output))
    if output.exists():
        raise FileExistsError('Use a new evaluation output directory')
    output.mkdir(parents=True)
    save_json(output/'status.json', {'status': 'RUNNING'})
    save_json(output/'protocol.json', PROTOCOL)
    started = time.perf_counter()
    try:
        manifest = data/'manifest.jsonl'; receipt = json.loads((data/'completed.json').read_text())
        if receipt.get('status') != 'COMPLETED' or receipt.get('manifest_sha256') != sha256(manifest):
            raise ValueError('Data manifest is not atomically completed/frozen')
        if receipt.get('dictionary_sha256', sha256(lexicon)) != sha256(lexicon):
            raise ValueError('Data/frozen dictionary SHA mismatch')
        rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        validation = [r for r in rows if r.get('split') == 'validation']
        if not validation or len({r['id'] for r in validation}) != len(validation):
            raise ValueError('Missing or duplicate validation recording')
        front, planner, _, identity = load_system(language, lexicon, training)
        best_path = training/'acoustic/best.pt'
        best, best_state = load_comparison(best_path, language, front, identity['planner_sha256'])
        if best_state.get('lineage', {}).get('manifest_sha256', sha256(manifest)) != sha256(manifest):
            raise ValueError('Checkpoint/data manifest mismatch')
        systems = [('BEST_PREDICTED_PLAN', best, False)]
        if zero_control:
            systems.append(('BEST_ZERO_PLAN_CONTROL', best, True))
        if initial is not None:
            initial_model, _ = load_comparison(initial, language, front, identity['planner_sha256'])
            systems.append(('INITIAL_TRANSFER_PREDICTED_PLAN', initial_model, False))
        checkpoints = {str(best_path): sha256(best_path), str(training/'planner/best.pt'): identity['planner_sha256']}
        if initial is not None:
            checkpoints[str(initial)] = sha256(initial)
        strict_rows = None
        strict_receipt = data/'strict/completed.json'
        if strict_receipt.exists():
            marker = json.loads(strict_receipt.read_text()); statistics = data/'strict/statistics.json'
            if (marker.get('status') != 'COMPLETED' or marker.get('source_manifest_sha256') != sha256(manifest)
                    or marker.get('statistics_sha256') != sha256(statistics)):
                raise ValueError('Strict automatic-confidence publication differs')
            strict_rows = {r['id']: r for r in json.loads(statistics.read_text())}
        baseline_path = training/'planner/baselines.json'
        phone_baselines = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        save_json(output/'identity.json', {'checkpoints': checkpoints, 'identity': identity,
            'manifest_sha256': sha256(manifest), 'strict_used': strict_rows is not None,
            'planner_baselines_sha256': sha256(baseline_path) if phone_baselines is not None else None,
            'script_sha256': sha256(__file__), 'reference_plan_used_for_inference': False})
        sentence_rows, boundary_rows, plan_rows = [], [], []
        (output/'predictions').mkdir()
        for index, row in enumerate(validation):
            path = Path(row['record_path']); path = path if path.is_absolute() else data/path
            if sha256(path) != row['sha256']:
                raise ValueError('Reference NPZ changed: '+row['id'])
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            f = front.encode(row['text'])
            for key in ('phonetic_features', 'context_features', 'plan_owner', 'lexical_word_indices', 'internal_pause_mask'):
                if key not in a or not np.array_equal(a[key], f[key]):
                    raise ValueError('Ordinary-text/reference topology differs: '+key)
            raw_duration = np.asarray(a.get('durations', a.get('target_duration')))
            if (raw_duration.shape != (len(f['ipa']),) or not np.isfinite(raw_duration).all()
                    or np.any(raw_duration < 0) or np.any(raw_duration != np.rint(raw_duration))):
                raise ValueError('Invalid reference unit durations')
            duration = raw_duration.astype(np.int64)
            reference = np.asarray(a.get('curve', a.get('target_curve')), np.float32)
            if reference.shape != (int(duration.sum()), 39):
                raise ValueError('Reference frame axis mismatch')
            strict = None
            if strict_rows is not None:
                overlay = data/'strict'/(row['id']+'.npz')
                if row['id'] not in strict_rows or sha256(overlay) != strict_rows[row['id']]['sha256']:
                    raise ValueError('Strict overlay changed')
                with np.load(overlay, allow_pickle=False) as z:
                    strict = {k: z[k] for k in z.files}
            o8 = expand_controls(a['target_controls'], duration)
            power, o8power = physical_logpower(reference), physical_logpower(o8)
            masks, boundary_counts, _ = boundary_regions(duration, f)
            plan = planner.predict_frontend(f)
            reference_plan, reference_mask = None, None
            reference_plan_computed = False
            floor = {'id': row['id'], 'system': 'O8_RAW_REFERENCE_FLOOR', 'axis': 'REFERENCE_DURATION_REPRESENTATION_DIAGNOSTIC', 'target': 'O_FRAME'}
            floor.update(parameter_metrics(o8, reference, duration, f, strict))
            for prefix, region in [('whole_', np.ones(len(reference), bool)), ('boundary_', masks['all']), ('nonboundary_', ~masks['all'])]:
                floor.update({prefix+k: v for k, v in trajectory_metrics(o8power, power, o8[:, 1], reference[:, 1], region).items()})
            floor['boundary_count'] = boundary_counts['all']
            sentence_rows.append(floor)
            for name, model, zero in systems:
                word_plan = np.zeros_like(plan['word_plan']) if zero else plan['word_plan']
                unit_plan = np.zeros_like(plan['unit_plan']) if zero else plan['unit_plan']
                features = {**f, 'context_features': np.column_stack((f['context_features'], unit_plan)).astype(np.float32)}
                prediction = model.predict_features(features)
                if not reference_plan_computed:
                    # Reference plans are assessed only after text-only forward
                    # prediction, and never replace any input or output field.
                    if phone_baselines is not None and planner.normalization:
                        measurement = wp.reference_measurements(f, {**a, **(strict or {}), 'split': 'validation'})
                        raw, mask = wp.raw_plan(measurement, phone_baselines)
                        reference_plan, reference_mask = wp.normalized_plan(raw, mask, planner.normalization)
                    reference_plan_computed = True
                aligned = expand_controls(prediction['controls'], duration)
                predicted_power = physical_logpower(aligned)
                durations = duration_metrics(prediction['durations'], duration, f)
                input_difference = {'input_plan_mean_absolute': float(abs(word_plan).mean()),
                                    'input_plan_vs_predicted_mean_absolute': float(abs(word_plan-plan['word_plan']).mean())}
                for j, field in enumerate(wp.PLAN_NAMES):
                    input_difference['input_'+field+'_mean'] = float(word_plan[:, j].mean())
                    input_difference['input_'+field+'_vs_predicted_MAE'] = float(abs(word_plan[:, j]-plan['word_plan'][:, j]).mean())
                    if reference_mask is not None:
                        valid = reference_mask[:, j]
                        input_difference['plan_'+field+'_valid_words'] = int(valid.sum())
                        input_difference['plan_'+field+'_reference_MAE'] = float(abs(word_plan[valid, j]-reference_plan[valid, j]).mean()) if valid.any() else None
                plan_rows.append({'id': row['id'], 'system': name, **input_difference})
                natural = {'id': row['id'], 'system': name, 'axis': PROTOCOL['natural_axis'], 'target': 'UNIT_DURATION_ONLY',
                           **durations, **natural_statistics(prediction['curve']), **input_difference}
                sentence_rows.append(natural)
                same_axis = {'id': row['id'], 'system': name, 'axis': PROTOCOL['aligned_axis'], 'target': 'O_FRAME',
                    **durations, **input_difference, **parameter_metrics(aligned, reference, duration, f, strict),
                    'boundary_count': boundary_counts['all']}
                for prefix, region in [('whole_', np.ones(len(reference), bool)), ('boundary_', masks['all']), ('nonboundary_', ~masks['all'])]:
                    same_axis.update({prefix+k: v for k, v in trajectory_metrics(predicted_power, power, aligned[:, 1], reference[:, 1], region).items()})
                same_axis.update({'vs_O8_boundary_'+k: v for k, v in trajectory_metrics(
                    predicted_power, o8power, aligned[:, 1], o8[:, 1], masks['all']).items()})
                sentence_rows.append(same_axis)
                for kind, region in masks.items():
                    boundary_rows.append({'id': row['id'], 'system': name, 'axis': PROTOCOL['aligned_axis'], 'target': 'O_FRAME',
                        'kind': kind, 'boundary_count': boundary_counts[kind],
                        **trajectory_metrics(predicted_power, power, aligned[:, 1], reference[:, 1], region)})
                np.savez_compressed(output/'predictions'/(f'{index:04}_{name}.npz'),
                    controls=prediction['controls'], natural_durations=prediction['durations'], natural_curve=prediction['curve'],
                    reference_duration_diagnostic_curve=aligned, reference_durations=duration,
                    word_plan=word_plan, unit_plan=unit_plan, reference_inputs_used_for_natural_inference=np.asarray(False))
            for kind, region in masks.items():
                boundary_rows.append({'id': row['id'], 'system': 'O8_RAW_REFERENCE_FLOOR', 'axis': floor['axis'], 'target': 'O_FRAME',
                    'kind': kind, 'boundary_count': boundary_counts[kind],
                    **trajectory_metrics(o8power, power, o8[:, 1], reference[:, 1], region)})
            save_json(output/'status.json', {'status': 'RUNNING', 'recordings': index+1, 'total': len(validation)})
        if any(sha256(p) != checksum for p, checksum in checkpoints.items()) or sha256(manifest) != receipt['manifest_sha256']:
            raise ValueError('Frozen weights/data changed during evaluation')
        write_csv(output/'per_sentence.csv', sentence_rows); write_csv(output/'boundary_by_type.csv', boundary_rows)
        write_csv(output/'plan_by_sentence.csv', plan_rows)
        summary = {'status': 'COMPLETED', 'language': language, 'validation_recordings': len(validation),
            'systems': [x[0] for x in systems], 'protocol': PROTOCOL,
            'sentence_aggregates': summarize(sentence_rows, ['system', 'axis', 'target']),
            'boundary_aggregates': summarize(boundary_rows, ['system', 'axis', 'target', 'kind']),
            'aggregation': 'equal recording means, counts summed separately; no framewise significance or MOS inference',
            'reference_plan_diagnostic': 'COMPLETED' if phone_baselines is not None and planner.normalization else 'NOT_RUN: frozen planner phone baselines/normalization unavailable',
            'test': 'NOT_RUN', 'human_listening': 'NOT_RUN', 'MOS': None, 'wall_seconds': time.perf_counter()-started}
        save_json(output/'results.json', {'records': sentence_rows, 'boundary_records': boundary_rows, 'plan_records': plan_rows})
        save_json(output/'summary.json', summary); save_json(output/'completed.json', summary); save_json(output/'status.json', summary)
        return summary
    except Exception:
        save_json(output/'status.json', {'status': 'FAILED', 'traceback': traceback.format_exc()})
        (output/'FAILED_STEP.md').write_text('# FAILED_STEP\n\n'+traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--lexicon', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--language', choices=('es', 'fr', 'it', 'pt'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--zero-control', action='store_true')
    args = parser.parse_args()
    summary = evaluate(args.data, args.lexicon, args.training, args.language, args.output,
                       initial=args.initial, zero_control=args.zero_control)
    print(json.dumps({'status': summary['status'], 'validation_recordings': summary['validation_recordings'],
                      'systems': summary['systems'], 'wall_seconds': summary['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
