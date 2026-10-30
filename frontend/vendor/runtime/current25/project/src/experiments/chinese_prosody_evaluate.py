"""Frozen Mandarin prosody comparison: ordinary text synthesis and offline CER.

Reference pinyin, timing and audio never enter the predicted synthesis arms.
The original and acoustic-parameter oracle are separate validation diagnostics.
"""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from math import gcd
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from ..common import PROJECT, WORKSPACE, save_json, sha256
from ..audio.dsp import write_wav
from ..audio.trajectory_continuity import ContinuityConfig
from ..audio.trajectory_continuity_guarded import guarded_bridge
from ..frontend.chinese_linking_policy import numeric_policy
from ..frontend.chinese_training import ChineseTrainingFrontend
from ..models.universal_acoustic import load_acoustic, expand_controls
from .representation_limit import GenericDSPAdapter

ROOT = PROJECT / 'outputs/chinese_prosody_20261015'
CORPUS = PROJECT / 'outputs/chinese_training_20261014/data_full'
OLD = PROJECT / 'outputs/chinese_training_20261014/selected/acoustic.pt'
PROFILE = PROJECT / 'configs/chinese_linked_f0_guarded_v1.json'
CONFIG = PROJECT / 'configs/chinese_prosody_listening_v1.json'
SYSTEMS = ('old_D', 'continued_D', 'features_D', 'prosody_D')


def physical_frames(curve):
    physical = curve.copy()
    voice = curve[:, -1] > 0
    physical[:, 0] = np.where(voice, np.exp(curve[:, 0]), 0.)
    physical[:, -1] = voice
    physical[~voice, 25:30] = 0.
    return physical


def apply_profile(features, prediction, enabled):
    cfg = json.loads(PROFILE.read_text())
    # D policy remains on its frozen 32+22 contract. Extra model inputs are not
    # interpreted by the policy or generic trajectory processing.
    policy_features = dict(features)
    policy_features['context_features'] = features['context_features'][:, :22]
    base, durations = prediction['curve'], prediction['durations']
    allowed, protected, spans = numeric_policy(policy_features, durations, cfg['protected_syllable_fraction'])
    curve, audit = guarded_bridge(base, durations, allowed, base[:, -1] > 0, protected, spans,
        f0=enabled, spectral_energy=False,
        config=ContinuityConfig(half_window_frames=cfg['half_window_frames'],
            minimum_segment_frames=cfg['minimum_segment_frames'],
            max_f0_change_cents=cfg['max_f0_change_cents']),
        min_direction_cents=cfg['minimum_direction_excursion_cents'])
    changed = audit.pop('modified_mask')
    if not np.array_equal(curve[:, 1:], base[:, 1:]):
        raise AssertionError('D may change F0 only')
    if changed[protected].any():
        raise AssertionError('Protected syllable centers changed')
    return curve, audit, changed


def synthesize(model, frontend, text, output, use_D=True):
    from .chinese_prosody_train import encode_for_model
    tick = time.perf_counter()
    # This call has text only; never a reference pinyin argument.
    features = encode_for_model(frontend, text, model)
    if len(features['phonetic_features']) > 256:
        raise ValueError('Prototype sentence exceeds 256 units')
    prediction = model.predict_features(features)
    curve, audit, changed = apply_profile(features, prediction, use_D)
    if not 0 < len(curve) <= 3000:
        raise ValueError('Prototype duration out of range')
    physical = physical_frames(curve)
    pcm, dsp = GenericDSPAdapter(24, 5).render(physical)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output, pcm)
    np.savez_compressed(output.with_suffix('.npz'), curve=curve,
        baseline_curve=prediction['curve'], controls=prediction['controls'],
        durations=prediction['durations'], modified_frame_mask=changed,
        phonetic_features=features['phonetic_features'], context_features=features['context_features'])
    result = {'text': text, 'wav_sha256': sha256(output),
        'parameters': sum(p.numel() for p in model.parameters()),
        'context_dim': model.config['context_dim'], 'profile': 'D' if use_D else 'raw',
        'seconds': len(pcm) / 16000, 'wall_seconds': time.perf_counter() - tick,
        'ordinary_text_only_inference': True, 'reference_audio_pinyin_timing_used': False,
        'clipping_fraction': float(np.mean((pcm == 32767) | (pcm == -32768))),
        'frontend_metadata': features['metadata'], 'transition': audit, 'DSP': dsp}
    save_json(output.with_suffix('.json'), result)
    return result


def original_and_oracle(row, folder):
    if sha256(Path(row['source_audio'])) != row['source_audio_sha256'] or sha256(Path(row['record_path'])) != row['record_sha256']:
        raise ValueError('Reference WAV or record identity differs from frozen manifest')
    rate, pcm = wavfile.read(row['source_audio'])
    if pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError('Expected original mono PCM16')
    if rate != 16000:
        divisor = gcd(int(rate), 16000)
        audio = resample_poly(pcm.astype(np.float32), 16000 // divisor, int(rate) // divisor)
        pcm = np.clip(np.round(audio), -32768, 32767).astype(np.int16)
    write_wav(folder / 'original.wav', pcm)
    with np.load(row['record_path'], allow_pickle=False) as z:
        curve = z['target_curve'].copy()
    oracle, _ = GenericDSPAdapter(24, 5).render(physical_frames(curve))
    write_wav(folder / 'oracle.wav', oracle)


def freeze_selection(output, config):
    path = output / 'selection.json'
    if path.exists():
        selected = json.loads(path.read_text())
        if selected['config_sha256'] != sha256(CONFIG) or selected['data_manifest_sha256'] != sha256(CORPUS / 'manifest.jsonl'):
            raise ValueError('Frozen selection differs from config or data manifest')
        return selected
    validation = [json.loads(line) for line in (CORPUS / 'manifest.jsonl').read_text().splitlines()
                  if line.strip() and json.loads(line)['split'] == 'validation']
    validation.sort(key=lambda row: row['id'])
    generator = np.random.default_rng(config['seed'])
    indexes = generator.permutation(len(validation))[:config['validation_utterances']]
    rows = [{'id': 'V_' + validation[i]['id'], 'source_id': validation[i]['id'],
             'text': validation[i]['text'], 'cohort': 'validation_reference',
             'reference': validation[i]} for i in indexes]
    rows += [{**item, 'cohort': 'text_challenge'} for item in config['texts']]
    selected = {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': sha256(CONFIG), 'data_manifest_sha256': sha256(CORPUS / 'manifest.jsonl'),
        'no_test_access_for_selection': True, 'rows': rows}
    save_json(path, selected)
    return selected


def listening_html(output, manifest):
    lines = ['<!doctype html><meta charset="utf-8"><title>中文韵律训练对照</title>',
        '<style>body{font:16px sans-serif;max-width:1400px;margin:24px auto;padding:0 16px}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}'
        'audio{width:230px}summary{cursor:pointer}small{color:#555}</style>',
        '<h1>中文轻声与声调：同句盲听</h1>',
        '<p>四列顺序逐句随机，播放文件匿名并匹配同句整体RMS（非感知响度校准）。先听字间连接、轻声、词汇声调和句末。'
        '旧 D、同预算继续训练、仅新增韵律输入、韵律输入加轮廓监督各一列，四组均使用原 D 衔接。'
        '新模型尚未通过人工质量验收；此包不证明MOS提升。</p>',
        '<p>前12句为validation诊断（附原音、真实参数oracle），后24句为预设新文本。'
        '预测组只使用文字，不读取参考拼音、时长或音频；这里不是独立未见test成绩。</p>',
        '<table><tr><th>文本</th><th>A</th><th>B</th><th>C</th><th>D</th></tr>']
    for row in manifest['rows']:
        lines.append('<tr><td>' + escape(row['id'] + ' ' + row['text']) + '</td>')
        for label in 'ABCD':
            relative = row['blind'][label]['wav']
            lines.append(f'<td><audio controls preload="none" src="{escape(relative)}"></audio></td>')
        lines.append('</tr>')
        if row['cohort'] == 'validation_reference':
            lines.append('<tr><td colspan="5"><details><summary>原音与真实参数重建（诊断）</summary>'
                f'原音 <audio controls preload="none" src="{row["original"]}"></audio>'
                f'Oracle <audio controls preload="none" src="{row["oracle"]}"></audio></details></td></tr>')
    lines += ['</table><details><summary>揭晓版本（听后再打开）</summary><ul>']
    for row in manifest['rows']:
        key = '；'.join(label + '=' + row['blind'][label]['system'] for label in 'ABCD')
        lines.append('<li>' + escape(row['id'] + ': ' + key) + '</li>')
    lines += ['</ul></details><p>所有配置、模型SHA、逐句记录见 manifest.json。此处字母逐句随机，'
              '与上一试听包用户选定的 D 无对应关系。</p>']
    (output / 'index.html').write_text('\n'.join(lines))


def blind_audio(output, folder, order, systems):
    """Match utterance RMS using gains only; use one common peak cap per row.

    Raw synthesis files remain unchanged for reproducibility and ASR. These
    simple RMS gains do not constitute a calibrated perceptual loudness test.
    """
    signals = {}
    for name in order:
        rate, pcm = wavfile.read(output / systems[name]['wav'])
        if rate != 16000 or pcm.dtype != np.int16 or pcm.ndim != 1:
            raise ValueError('Blind playback requires mono 16k PCM16')
        audio = pcm.astype(np.float64) / 32768.
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms <= 1e-8:
            raise ValueError('Silent model output cannot be loudness matched')
        signals[name] = (audio, rms, .08 / rms)
    peak = max(float(np.max(np.abs(audio))) * gain for audio, rms, gain in signals.values())
    common = min(1., .92 / peak)
    result = {}
    blind_folder = folder / 'blind'
    blind_folder.mkdir(exist_ok=True)
    for label, name in zip('ABCD', order):
        audio, rms, gain = signals[name]
        pcm = np.round(audio * gain * common * 32768.).astype(np.int16)
        wav = blind_folder / (label + '.wav')
        write_wav(wav, pcm)
        result[label] = {'system': name, 'wav': str(wav.relative_to(output)),
            'wav_sha256': sha256(wav), 'gain': gain * common,
            'source_RMS': rms, 'target_RMS_before_joint_peak_cap': .08,
            'joint_peak_cap_gain': common, 'playback_RMS': float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768.) ** 2))),
            'ASR_uses_this_normalized_audio': False}
    if np.ptp([r['playback_RMS'] for r in result.values()]) > 1e-5:
        raise AssertionError('Blind playback RMS mismatch')
    return result


def generate(args):
    from ..frontend.chinese_prosody import ChineseProsodyFrontend
    config = json.loads(CONFIG.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    selected = freeze_selection(args.output, config)
    paths = {'old_D': OLD, 'continued_D': args.baseline,
             'features_D': args.features, 'prosody_D': args.prosody}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(args.selection.read_text())
    digests = {key: sha256(path) for key, path in paths.items()}
    for arm, key in (('baseline', 'continued_D'), ('prosody', 'prosody_D')):
        if selection['checkpoints'][arm]['sha256'] != digests[key]:
            raise ValueError('Checkpoint differs from completed paired-training selection: ' + arm)
    features_selection_path = args.features.parent / 'completed.json'
    features_selection = json.loads(features_selection_path.read_text())
    if features_selection['checkpoint_sha256'] != digests['features_D']:
        raise ValueError('Features-only checkpoint differs from completed selection')
    loaded = {}
    models = {}
    for key, path in paths.items():
        identity = str(path.resolve())
        if identity not in loaded:
            loaded[identity] = load_acoustic(path)
            if sha256(path) != digests[key]:
                raise ValueError('Checkpoint changed while being loaded')
        models[key] = loaded[identity]
    for name, model in models.items():
        expected_context = 34 if name in ('features_D', 'prosody_D') else 22
        if model.config['context_dim'] != expected_context or model.channels != 31:
            raise ValueError('Checkpoint contract mismatch: ' + name)
    fronts = {'old_D': ChineseTrainingFrontend(), 'continued_D': ChineseTrainingFrontend(),
              'features_D': ChineseProsodyFrontend(), 'prosody_D': ChineseProsodyFrontend()}
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
        'config': config, 'selection_sha256': sha256(args.output / 'selection.json'),
        'training_selection_sha256': sha256(args.selection),
        'features_only_selection_sha256': sha256(features_selection_path),
        'profile_sha256': sha256(PROFILE), 'checkpoints': {k: {'path': str(v), 'sha256': sha256(v)}
                                                       for k, v in paths.items()},
        'ordinary_text_only_prediction': True, 'MOS_measured': False, 'rows': []}
    generator = np.random.default_rng(config['seed'] + 1)
    for row in selected['rows']:
        folder = args.output / 'samples' / row['id']
        folder.mkdir(parents=True, exist_ok=True)
        result = {k: v for k, v in row.items() if k != 'reference'}
        result['systems'] = {}
        for name in SYSTEMS:
            wav = folder / (name + '.wav')
            metrics = synthesize(models[name], fronts[name], row['text'], wav, True)
            result['systems'][name] = {'wav': str(wav.relative_to(args.output)), **metrics}
        if row['cohort'] == 'validation_reference':
            original_and_oracle(row['reference'], folder)
            result['original'] = str((folder / 'original.wav').relative_to(args.output))
            result['oracle'] = str((folder / 'oracle.wav').relative_to(args.output))
            result['original_wav_sha256'] = sha256(folder / 'original.wav')
            result['oracle_wav_sha256'] = sha256(folder / 'oracle.wav')
            result['reference'] = {k: row['reference'][k] for k in ('id', 'split', 'source_audio_sha256', 'record_sha256')}
        order = generator.permutation(SYSTEMS).tolist()
        result['blind'] = blind_audio(args.output, folder, order, result['systems'])
        manifest['rows'].append(result)
        save_json(args.output / 'manifest.json', manifest)
        print(json.dumps({'generated': len(manifest['rows']), 'id': row['id']}, ensure_ascii=False), flush=True)
    manifest['complete'] = True
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(args.output / 'manifest.json', manifest)
    listening_html(args.output, manifest)
    summary = {'utterances': len(manifest['rows']), 'predicted_WAVs': 4 * len(manifest['rows']),
        'reference_WAVs': 2 * config['validation_utterances'],
        'RMS_matched_anonymous_playback_WAVs': 4 * len(manifest['rows']), 'systems': {}}
    for system in SYSTEMS:
        group = [r['systems'][system] for r in manifest['rows']]
        summary['systems'][system] = {'parameters': group[0]['parameters'],
            'maximum_clipping_fraction': max(r['clipping_fraction'] for r in group),
            'total_audio_seconds': sum(r['seconds'] for r in group),
            'ordinary_text_only': all(r['reference_audio_pinyin_timing_used'] is False for r in group)}
    save_json(args.output / 'audio_summary.json', summary)


def asr(args):
    from .chinese_asr_evaluate import ASR_SHA, ROOT as CHINESE, aggregate, normalize
    from ..inference.asr_evaluate import edits
    os.environ['NUMBA_CACHE_DIR'] = str(PROJECT / 'outputs/cache/numba')
    sys.path.extend([str(CHINESE / 'deps'), str(WORKSPACE / 'oracle_experiment_deps')])
    import whisper
    from opencc import OpenCC
    manifest = json.loads((args.output / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Generate all comparisons before ASR')
    checkpoint = CHINESE / 'asr/downloads/tiny.pt'
    if sha256(checkpoint) != ASR_SHA:
        raise ValueError('ASR checkpoint SHA mismatch')
    decoder = {'language': 'zh', 'task': 'transcribe', 'fp16': False, 'temperature': 0.,
               'beam_size': 1, 'condition_on_previous_text': False}
    model = whisper.load_model(str(checkpoint), device='cpu')
    converter = OpenCC('t2s')
    output = args.output / 'asr'
    output.mkdir(exist_ok=True)
    config = {'manifest_sha256': sha256(args.output / 'manifest.json'), 'decoder': decoder,
        'ASR_checkpoint_sha256': ASR_SHA, 'cohort': 'validation_reference_only',
        'reference_prompt_used': False, 'MOS_measured': False, 'ASR_to_MOS_conversion': False,
        'score_normalization': 'NFKC/punctuation-space-symbol removal; literal and OpenCC t2s; no homophone substitutions'}
    save_json(output / 'config.json', config)
    rows = []
    total_jobs = 6 * sum(r['cohort'] == 'validation_reference' for r in manifest['rows'])
    for row in manifest['rows']:
        if row['cohort'] != 'validation_reference':
            continue
        for system in (*SYSTEMS, 'original', 'oracle'):
            relative = row[system] if system in ('original', 'oracle') else row['systems'][system]['wav']
            path = args.output / relative
            digest = sha256(path)
            expected = row[system + '_wav_sha256'] if system in ('original', 'oracle') else row['systems'][system]['wav_sha256']
            if digest != expected:
                raise ValueError('ASR audio differs from frozen synthesis manifest')
            cache = output / (row['id'] + '__' + system + '.json')
            result = json.loads(cache.read_text()) if cache.exists() else None
            if not result or result.get('wav_sha256') != digest or result.get('decoder') != decoder or result.get('ASR_checkpoint_sha256') != ASR_SHA or result.get('reference') != row['text']:
                rate, pcm = wavfile.read(path)
                if rate != 16000 or pcm.ndim != 1 or pcm.dtype != np.int16:
                    raise ValueError('Expected 16k mono PCM16 ASR input')
                started = time.perf_counter()
                decoded = model.transcribe(pcm.astype(np.float32) / 32768., verbose=None, **decoder)
                reference = normalize(row['text'])
                hypothesis = normalize(decoded['text'])
                simple_reference = normalize(row['text'], converter)
                simple_hypothesis = normalize(decoded['text'], converter)
                result = {'id': row['id'], 'system': system, 'wav_sha256': digest,
                    'decoder': decoder, 'ASR_checkpoint_sha256': ASR_SHA,
                    'reference': row['text'], 'hypothesis': decoded['text'],
                    'literal_reference': reference, 'literal_hypothesis': hypothesis,
                    'simplified_reference': simple_reference, 'simplified_hypothesis': simple_hypothesis,
                    'reference_characters': len(simple_reference),
                    'literal_character_errors': edits(reference, hypothesis),
                    'simplified_character_errors': edits(simple_reference, simple_hypothesis),
                    'ASR_wall_seconds': time.perf_counter() - started}
                save_json(cache, result)
            rows.append(result)
            save_json(output / 'per_utterance.json', rows)
            save_json(output / 'summary.json', {'config': config, 'systems': aggregate(rows),
                'completed_jobs': len(rows), 'total_jobs': total_jobs, 'complete': len(rows) == total_jobs})
            print(json.dumps({'ASR_done': len(rows), 'system': system, 'id': row['id']}, ensure_ascii=False), flush=True)


def diagnose(args):
    """Separate reference-phonetic/reference-timing diagnostic, never TTS MOS.

    Duration estimates are predicted; frame-F0 comparisons expand predicted
    controls on publisher timings solely to isolate contour errors. No such
    reference conditioning is used by generate() or ordinary synthesis.
    """
    from ..frontend.chinese_prosody import ChineseProsodyFrontend
    from .chinese_train import NumericRecords
    from .chinese_prosody_train import encode_for_model
    selection = json.loads(args.selection.read_text())
    paths = {'old': OLD, 'continued': args.baseline, 'features': args.features, 'prosody': args.prosody}
    for arm, key in (('baseline', 'continued'), ('prosody', 'prosody')):
        if selection['checkpoints'][arm]['sha256'] != sha256(paths[key]):
            raise ValueError('Diagnostic checkpoint identity mismatch')
    if json.loads((args.features.parent / 'completed.json').read_text())['checkpoint_sha256'] != sha256(args.features):
        raise ValueError('Features-only diagnostic checkpoint identity mismatch')
    models = {name: load_acoustic(path) for name, path in paths.items()}
    front = ChineseProsodyFrontend()
    records = NumericRecords(CORPUS, 'validation')
    accumulated = {system: {} for system in models}
    per_syllable = []
    cents = 1200 / np.log(2)
    for row, arrays in records:
        features = front.encode(row['text'], pinyin=row['publisher_pinyin'])
        for name in ('phonetic_features', 'context_features', 'edge_sil_mask'):
            if not np.array_equal(features[name], arrays[name]):
                raise ValueError('Reference diagnostic must preserve frozen native input')
        reference_durations = arrays['target_duration'].astype(np.int64)
        offsets = np.r_[0, reference_durations.cumsum()]
        target = arrays['target_curve']
        for system, model in models.items():
            numeric = dict(features)
            if model.config['context_dim'] == 34:
                numeric['context_features'] = np.concatenate((features['context_features'], features['prosody_features']), -1)
            prediction = model.predict_features(numeric)
            aligned = expand_controls(prediction['controls'], reference_durations)
            for item in features['metadata']['syllables']:
                start, end = item['phone_start'], item['phone_end']
                low, high = offsets[start], offsets[end]
                voice = target[low:high, -1] > .5
                residual = (aligned[low:high, 0] - target[low:high, 0])[voice] * cents
                centered = residual - residual.mean() if len(residual) else residual
                predicted_duration = int(prediction['durations'][start:end].sum()) * 10
                real_duration = int(reference_durations[start:end].sum()) * 10
                categories = ['all_syllables', 'surface_tone_' + str(item['surface_tone_category'])]
                if item['surface_tone_category'] == 5:
                    categories += ['neutral', 'neutral_after_' + str(item['previous_surface_tone'])]
                if item['citation_known'] and item['surface_tone_category'] == 2 and item['citation_tone'] in (2, 3):
                    categories += ['citation_' + str(item['citation_tone']) + '_surface_2']
                for category in categories:
                    state = accumulated[system].setdefault(category, {'syllables': 0, 'voiced_frames': 0,
                        'pitch_squared_cents': 0., 'centered_pitch_squared_cents': 0.,
                        'pitch_signed_cents': 0., 'duration_abs_ms': 0., 'duration_signed_ms': 0.,
                        'reference_durations_ms': [], 'predicted_durations_ms': []})
                    state['syllables'] += 1
                    state['voiced_frames'] += len(residual)
                    state['pitch_squared_cents'] += float(np.sum(residual ** 2))
                    state['centered_pitch_squared_cents'] += float(np.sum(centered ** 2))
                    state['pitch_signed_cents'] += float(residual.sum())
                    state['duration_abs_ms'] += abs(predicted_duration - real_duration)
                    state['duration_signed_ms'] += predicted_duration - real_duration
                    state['reference_durations_ms'].append(real_duration)
                    state['predicted_durations_ms'].append(predicted_duration)
                per_syllable.append({'id': row['id'], 'system': system, 'character': item['character'],
                    'surface_tone': item['surface_tone_category'], 'citation_tone': item['citation_tone'],
                    'citation_known': item['citation_known'], 'previous_tone': item['previous_surface_tone'],
                    'reference_duration_ms': real_duration, 'predicted_duration_ms': predicted_duration,
                    'voiced_frames': len(residual),
                    'F0_RMSE_cents': float(np.sqrt(np.mean(residual ** 2))) if len(residual) else None,
                    'centered_F0_RMSE_cents': float(np.sqrt(np.mean(centered ** 2))) if len(residual) else None})
    summary = {}
    for system, categories in accumulated.items():
        summary[system] = {}
        for category, state in categories.items():
            n, count = state['syllables'], state['voiced_frames']
            summary[system][category] = {'syllables': n, 'voiced_frames': count,
                'F0_RMSE_cents_on_gold_voicing': float(np.sqrt(state['pitch_squared_cents'] / count)) if count else None,
                'centered_F0_RMSE_cents_on_gold_voicing': float(np.sqrt(state['centered_pitch_squared_cents'] / count)) if count else None,
                'F0_mean_bias_cents': state['pitch_signed_cents'] / count if count else None,
                'duration_MAE_ms': state['duration_abs_ms'] / n,
                'duration_mean_bias_ms': state['duration_signed_ms'] / n,
                'reference_duration_std_ms': float(np.std(state['reference_durations_ms'])),
                'predicted_duration_std_ms': float(np.std(state['predicted_durations_ms']))}
    output = args.output / 'reference_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'config.json', {'cohort': 'validation_520', 'records': len(records),
        'checkpoints_sha256': {key: sha256(path) for key, path in paths.items()},
        'reference_phonetics_used': True, 'reference_durations_used_for_frame_F0_only': True,
        'ordinary_text_TTS_evaluation': False, 'reference_duration_used_for_duration_prediction': False,
        'postprocessing': 'none; raw learned controls isolate acoustic-model change',
        'test_used': False, 'MOS_measured': False, 'tone_group_labels': 'frontend-derived, not independently verified tone labels',
        'warning': 'Gold-voicing/reference-timing F0 errors do not measure predicted voicing or end-to-end tone correctness.'})
    save_json(output / 'per_syllable.json', per_syllable)
    save_json(output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('freeze', 'generate', 'asr', 'diagnose'), default='generate')
    parser.add_argument('--output', type=Path, default=ROOT / 'listening_test')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'training/baseline/best.pt')
    parser.add_argument('--prosody', type=Path, default=ROOT / 'training/prosody/best.pt')
    parser.add_argument('--features', type=Path, default=ROOT / 'features_only_experiment/training/prosody/best.pt')
    parser.add_argument('--selection', type=Path, default=ROOT / 'selection_ready.json')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == 'freeze':
        args.output.mkdir(parents=True, exist_ok=True)
        freeze_selection(args.output, json.loads(CONFIG.read_text()))
    elif args.mode == 'generate':
        generate(args)
    elif args.mode == 'diagnose':
        diagnose(args)
    else:
        asr(args)


if __name__ == '__main__':
    main()
