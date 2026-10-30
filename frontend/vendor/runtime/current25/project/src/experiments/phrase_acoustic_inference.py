"""Text-plan-conditioned acoustic inference; source-derived plans are explicit diagnostics."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..common import PROJECT, save_json, sha256
from ..frontend.phrase_plan import PLAN_DIM, PLAN_NAMES, make_frontend, plan_for_units, project_by_characters
from ..models.universal_acoustic import load_acoustic, expand_controls
from ..frontend.chinese_delivery_policy import numeric_duration_policy
from ..audio.delivery_timing import apply_numeric_delivery
from ..audio.dsp import write_wav
from .chinese_prosody_evaluate import apply_profile, physical_frames
from .representation_limit import GenericDSPAdapter


def base34_features(frontend_result):
    features = dict(frontend_result)
    context = np.asarray(features['context_features'], np.float32)
    if context.shape[1] == 22:
        context = np.concatenate((context, np.asarray(features['prosody_features'], np.float32)), -1)
    if context.shape != (len(features['phonetic_features']), 34):
        raise ValueError('Expected frozen phonetic32/context34 frontend')
    features['context_features'] = context
    return features


def append_plan(features, unit_plan):
    result = base34_features(features)
    plan = np.asarray(unit_plan, np.float32)
    if plan.shape != (len(result['phonetic_features']), PLAN_DIM) or not np.isfinite(plan).all():
        raise ValueError('Expected finite native-unit N x8 plan')
    result['context_features'] = np.concatenate((result['context_features'], plan), -1)
    return result


def render_features(text, checkpoint, output, features, provenance):
    tick = time.perf_counter()
    checkpoint, output = Path(checkpoint), Path(output)
    digest = sha256(checkpoint)
    model = load_acoustic(checkpoint)
    if model.config['context_dim'] != 42 or model.channels != 31 or model.config['continuous_voicing']:
        raise ValueError('Expected context42 binary31 conditioned model')
    if sha256(checkpoint) != digest:
        raise ValueError('Checkpoint changed during load')
    parameters = sum(p.numel() for p in model.parameters())
    planner_parameters = int(provenance.get('planner_parameters', 0))
    if parameters + planner_parameters > 1000000:
        raise ValueError('Acoustic plus planner exceeds1M')
    if len(features['phonetic_features']) > 256:
        raise ValueError('Split sentence into at most256 native units')
    cfg = json.loads((PROJECT/'configs/chinese_articulation_v1.json').read_text())
    prediction = model.predict_features(features)
    duration = numeric_duration_policy(features, prediction['durations'], comma_min_frames=cfg['comma_min_frames'],
        kinship_max_ratio=cfg['kinship_max_ratio'], minimum_neutral_scale=cfg['minimum_neutral_scale'],
        minimum_rime_frames=cfg['minimum_rime_frames'], include_final_kinship=cfg['include_final_kinship'])
    controls, durations, adapter = apply_numeric_delivery(prediction['controls'], prediction['durations'],
        duration['target_durations'], duration['comma_mask'], quiet_log_energy=cfg['quiet_log_energy'])
    raw = expand_controls(controls, durations)
    if not 0 < len(raw) <= 3000:
        raise ValueError('Prototype duration out of range')
    curve, transition, modified = apply_profile(features, {'curve': raw, 'durations': durations}, True)
    pcm, dsp = GenericDSPAdapter(24, 5).render(physical_frames(curve))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output, pcm)
    np.savez_compressed(output.with_suffix('.npz'), curve=curve, raw_retimed_curve=raw,
        original_curve=prediction['curve'], controls=controls, original_controls=prediction['controls'],
        durations=durations, original_durations=prediction['durations'],
        comma_mask=duration['comma_mask'], kinship_mask=duration['kinship_neutral_mask'],
        phonetic_features=features['phonetic_features'], context_features=features['context_features'],
        phrase_plan=features['context_features'][:, 34:42], modified_F0_frames=modified)
    pauses = []
    offsets = np.r_[0, durations.cumsum()]
    for i in np.flatnonzero(duration['comma_mask']):
        a, b = int(offsets[i]), int(offsets[i+1])
        region = pcm[(a+2)*160:max(a+2, b-2)*160].astype(np.float64)/32768.
        rms = float(np.sqrt(np.mean(region**2))) if len(region) else None
        pauses.append({'unit': int(i), 'start_frame': a, 'end_frame': b,
            'original_ms': int(prediction['durations'][i])*10, 'repaired_ms': int(durations[i])*10,
            'middle_RMS': rms, 'middle_RMS_dbFS': float(20*np.log10(max(rms, 1e-12))) if rms is not None else None})
    reference = bool(provenance.get('reference_derived', True))
    result = {'text': text, 'profile': cfg['version'], 'checkpoint_sha256': digest,
        'parameters': parameters, 'planner_parameters': planner_parameters, 'total_parameters': parameters+planner_parameters,
        'context_dim': 42, 'wav_sha256': sha256(output), 'ordinary_text_only_inference': not reference,
        'reference_audio_pinyin_timing_used': reference, 'reference_plan_used': reference,
        'phrase_plan_provenance': provenance,
        'frontend_metadata': features['metadata'], 'duration_policy': duration['metadata'],
        'adapter': adapter, 'transition': transition, 'comma_pauses': pauses,
        'seconds': len(pcm)/16000, 'wall_seconds': time.perf_counter()-tick,
        'clipping_fraction': float(np.mean((pcm==32767)|(pcm==-32768))), 'DSP': dsp, 'MOS_measured': False}
    save_json(output.with_suffix('.json'), result)
    return result


def synthesize_plan(text, checkpoint, output, syllable_plan, *, provenance):
    """Explicit Kx8 plan intervention. Caller must declare reference_derived."""
    if 'reference_derived' not in provenance and 'reference_plan_used' not in provenance:
        raise ValueError('Plan provenance must declare reference_derived or reference_plan_used explicitly')
    provenance = dict(provenance)
    if 'reference_derived' not in provenance:
        provenance['reference_derived'] = bool(provenance['reference_plan_used'])
    if 'reference_plan_used' in provenance and bool(provenance['reference_plan_used']) != bool(provenance['reference_derived']):
        raise ValueError('Conflicting reference-plan provenance flags')
    features = make_frontend().encode(text)
    return render_features(text, checkpoint, output,
        append_plan(features, plan_for_units(features, syllable_plan)), dict(provenance))


def synthesize_reference(text, checkpoint, output, reference_syllable_plan,
                         reference_frontend, *, provenance):
    """Reference-plan diagnostic; ordinary frontend itself receives text only."""
    features = make_frontend().encode(text)
    plan = project_by_characters(reference_frontend, reference_syllable_plan, features)
    info = {**provenance, 'mode': 'reference_plan_diagnostic', 'reference_derived': True,
            'ordinary_frontend_received_reference_pinyin': False,
            'routing': 'character spans only; reference character mismatch raises'}
    return render_features(text, checkpoint, output, append_plan(features, plan_for_units(features, plan)), info)


def synthesize(text, checkpoint, output, *, mode='zero', planner_checkpoint=None):
    if mode == 'zero':
        features = make_frontend().encode(text)
        plan = np.zeros((len(features['phonetic_features']), PLAN_DIM), np.float32)
        provenance = {'mode': 'zero', 'reference_derived': False, 'planner_parameters': 0}
    elif mode == 'predicted':
        if planner_checkpoint is None:
            raise ValueError('Pure-text predicted mode requires frozen planner checkpoint')
        from ..models.phrase_prosody import load_planner
        planner = load_planner(planner_checkpoint)
        predicted = planner.predict_text(text)
        features, plan = predicted['frontend'], predicted['unit_plan']
        state = torch.load(planner_checkpoint, map_location='cpu', weights_only=True)
        normalization_sha = state.get('normalization_sha256') or state.get('lineage', {}).get('normalization_sha256')
        if not normalization_sha:
            raise ValueError('Planner checkpoint must record normalization SHA')
        planner_parameters = sum(p.numel() for p in planner.parameters())
        provenance = {'mode': 'predicted', 'reference_derived': False,
            'planner_checkpoint': str(planner_checkpoint), 'planner_checkpoint_sha256': sha256(Path(planner_checkpoint)),
            'normalization_sha256': normalization_sha, 'planner_parameters': planner_parameters,
            'plan_names': list(PLAN_NAMES), 'reference_pinyin_audio_or_timing_used': False}
    else:
        raise ValueError('Ordinary inference supports zero/predicted only; use explicit reference API')
    return render_features(text, checkpoint, output, append_plan(features, plan), provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', required=True); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--mode', choices=('zero','predicted'), default='zero')
    p.add_argument('--planner-checkpoint', type=Path); a=p.parse_args(); torch.set_num_threads(1)
    synthesize(a.text, a.checkpoint, a.output, mode=a.mode, planner_checkpoint=a.planner_checkpoint)


if __name__ == '__main__': main()
