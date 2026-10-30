"""Pure-text Romance39 listening pack, with frozen predicted timing and DSP.

The optional zero-plan arm is a control intervention in the same acoustic model,
not an equally trained baseline or evidence of improved quality. No reference
recordings, reference phonetics or reference durations enter synthesis.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
import csv
import html
import json
from pathlib import Path
import platform
import random
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from ..audio.dsp import write_wav
from ..common import save_json, sha256
from ..frontend.romance import RomanceFrontend
from ..models.english39_acoustic import contract as acoustic_contract, render_curve
from ..models.romance_word_planner import load_word_planner
from ..models.universal_acoustic import UniversalAcousticModel

ACOUSTIC_FORMAT = 'tiny_romance_acoustic39_v1'
CHALLENGES = {
    'es': ['La casa está cerca de la plaza.', 'Esta mañana vamos a escuchar una historia.'],
    'fr': ['La maison est près de la place.', 'Ce matin nous allons écouter une histoire.'],
    'it': ['La casa è vicino alla piazza.', 'Questa mattina ascoltiamo una storia.'],
    'pt': ['A casa fica perto da praça.', 'Esta manhã vamos ouvir uma história.'],
}


def rss_bytes():
    # Linux ru_maxrss is KiB. This includes Torch/runtime, not a deployment claim.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def load_system(language, lexicon, training):
    training = Path(training)
    acoustic_path, planner_path = training/'acoustic/best.pt', training/'planner/best.pt'
    hashes = {str(p): sha256(p) for p in (acoustic_path, planner_path, Path(lexicon))}
    front = RomanceFrontend(language, lexicon)
    planner = load_word_planner(planner_path)
    state = torch.load(acoustic_path, map_location='cpu', weights_only=True)
    if state.get('format') != ACOUSTIC_FORMAT or state.get('language', language) != language:
        raise ValueError('Acoustic checkpoint format/language mismatch')
    config = state['config']
    if any(config.get(k) != v for k, v in (
        ('phonetic_dim', 36), ('context_dim', 27), ('spectral_dim', 32), ('ap_bands', 5),
    )):
        raise ValueError('Expected explicit Romance36 + context27 -> warped70039 checkpoint')
    for key in ('dictionary_sha256', 'lexicon_sha256'):
        if key in state and state[key] != front.lexicon_sha256:
            raise ValueError('Acoustic checkpoint/frozen dictionary SHA mismatch')
    if 'frontend_signature' in state and state['frontend_signature'] != front.signature:
        raise ValueError('Acoustic checkpoint/frontend signature mismatch')
    lineage=state.get('lineage',{})
    if lineage.get('lexicon_sha256',front.lexicon_sha256)!=front.lexicon_sha256 or lineage.get('frontend_signature',front.signature)!=front.signature:
        raise ValueError('Acoustic lineage/frontend identity mismatch')
    if lineage.get('planner_sha256',hashes[str(planner_path)])!=hashes[str(planner_path)]:
        raise ValueError('Acoustic lineage/planner identity mismatch')
    if lineage.get('language',language)!=language or config.get('language',language)!=language:
        raise ValueError('Acoustic lineage language mismatch')
    model = UniversalAcousticModel(config, state['model']['means'], state['model']['scales'])
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    if hashes != {str(p): sha256(p) for p in (acoustic_path, planner_path, Path(lexicon))}:
        raise ValueError('Checkpoint or lexicon changed during load')
    return front, planner, model, {
        'language': language, 'acoustic_checkpoint': str(acoustic_path.resolve()),
        'planner_checkpoint': str(planner_path.resolve()),
        'acoustic_sha256': hashes[str(acoustic_path)], 'planner_sha256': hashes[str(planner_path)],
        'lexicon_sha256': front.lexicon_sha256, 'frontend_signature': front.signature,
        'acoustic_step': state.get('step'), 'planner_step': getattr(planner, 'checkpoint_step', None),
        'acoustic_parameters': sum(p.numel() for p in model.parameters()),
        'planner_parameters': sum(p.numel() for p in planner.parameters()),
        'frontend_contract': front.contract,
        'acoustic_output_contract': {key: value for key, value in acoustic_contract('warped39').items()
            if key in ('mode', 'channels', 'spectral', 'channel_names', 'control_points',
                       'frame_period_ms', 'sample_rate', 'AP_edges_Hz', 'AP_encoding', 'DSP', 'postfilter')},
        'device': 'cpu',
        'postfilter': None, 'reference_inputs': False,
    }


def select_cases(data, language):
    data = Path(data)
    manifest = data/'manifest.jsonl'
    receipt = json.loads((data/'completed.json').read_text())
    if receipt.get('status') != 'COMPLETED' or receipt.get('manifest_sha256') != sha256(manifest):
        raise ValueError('Wait for the frozen completed data manifest')
    if receipt.get('language', language) != language:
        raise ValueError('Data language differs from listening language')
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    validation = [r for r in rows if r.get('split') == 'validation'][:6]
    if not validation:
        raise ValueError('No validation text in frozen manifest')
    # Read only IDs and ordinary text. No record NPZ, reference WAV or phones.
    cases = [{'id': f'VAL{i+1:02}', 'source_id': row['id'], 'text': row['text'],
              'split': 'validation', 'reference_available_but_not_loaded': True}
             for i, row in enumerate(validation)]
    cases.extend({'id': f'NEW{i+1:02}', 'source_id': None, 'text': text,
                  'split': 'constructed_unseen_text', 'reference_available_but_not_loaded': False}
                 for i, text in enumerate(CHALLENGES[language]))
    ids = [r['source_id'] for r in cases if r['source_id'] is not None]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate validation ID')
    train_text = {r['text'].strip() for r in rows if r.get('split') == 'train'}
    for case in cases:
        if case['split'] == 'constructed_unseen_text' and case['text'].strip() in train_text:
            case['split'] = 'constructed_text_matches_train_not_unseen'
    return cases, {'manifest_sha256': sha256(manifest), 'selection': 'first6 validation manifest order, no quality selection',
                   'validation_requested': 6, 'validation_available_selected': len(validation),
                   'test_targets_or_audio_opened': False, 'reference_inputs_used': False}


@torch.inference_mode()
def synthesize_case(text, system, front, planner, model, output, identity):
    output = Path(output)
    if output.exists() or output.with_suffix('.npz').exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Refuse to overwrite listening artifacts: '+str(output))
    started = time.perf_counter()
    raw = front.encode(text)
    plan = planner.predict_frontend(raw)
    word_plan, unit_plan = plan['word_plan'], plan['unit_plan']
    if system == 'ZERO_PLAN_CONTROL':
        word_plan, unit_plan = np.zeros_like(word_plan), np.zeros_like(unit_plan)
    elif system != 'PREDICTED_PLAN':
        raise ValueError('Unknown ordinary-text system')
    features = {**raw, 'context_features': np.column_stack((raw['context_features'], unit_plan)).astype(np.float32)}
    prediction_started = time.perf_counter()
    predicted = model.predict_features(features)
    inference_seconds = time.perf_counter()-prediction_started
    if predicted['controls'].shape != (len(raw['ipa']), 8, 39):
        raise ValueError('Predicted controls differ from fixed39 contract')
    curve = np.asarray(predicted['curve'], np.float32)
    if (curve.ndim != 2 or curve.shape[1] != 39 or not len(curve)
            or not np.isfinite(curve).all() or len(curve) != np.asarray(predicted['durations']).sum()):
        raise ValueError('Invalid predicted native time axis')
    dsp_started = time.perf_counter()
    pcm, dsp = render_curve(curve, 'warped39')
    dsp_seconds = time.perf_counter()-dsp_started
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output, pcm)
    np.savez_compressed(output.with_suffix('.npz'), controls=predicted['controls'],
                        durations=predicted['durations'], curve=curve, word_plan=word_plan, unit_plan=unit_plan,
                        phonetic_features=raw['phonetic_features'], context_features=features['context_features'],
                        lexical_word_indices=raw['lexical_word_indices'], plan_owner=raw['plan_owner'],
                        internal_pause_mask=raw['internal_pause_mask'], ipa=np.asarray(raw['ipa']))
    pcmf = pcm.astype(np.float64)/32768
    wall = time.perf_counter()-started
    metadata = {'status': 'COMPLETED', 'system': system, 'text': text,
        'axis': 'ORDINARY_TEXT_PREDICTED_DURATION', 'plan_source': 'PREDICTED_TEXT_ONLY' if system == 'PREDICTED_PLAN' else 'ZERO_PLAN_CONTROL_INTERVENTION',
        'reference_phonetics_or_audio_or_duration_used': False, 'identity': identity,
        'units': len(raw['ipa']), 'frames': len(curve), 'audio_seconds': len(pcm)/16000,
        'wall_seconds': wall, 'acoustic_inference_seconds': inference_seconds, 'DSP_seconds': dsp_seconds,
        'RTF': wall/(len(pcm)/16000), 'host_process_peak_RSS_bytes': rss_bytes(),
        'PCM_peak': float(np.max(abs(pcmf))), 'PCM_RMS': float(np.sqrt(np.mean(pcmf**2))),
        'clipping_fraction': float(np.mean(abs(pcm.astype(np.int32)) >= 32767)),
        'wave_sha256': sha256(output), 'npz_sha256': sha256(output.with_suffix('.npz')),
        'postfilter': None, 'loudness_matching': None, 'full_sentence_continuous_DSP': True,
        'DSP': dsp, 'human_listening': 'NOT_RUN', 'MOS': None}
    save_json(output.with_suffix('.json'), metadata)
    return metadata


def _html_document(title, sections, explanation):
    return ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>body{max-width:950px;margin:32px auto;padding:0 18px;font:17px/1.7 sans-serif;'
            'color:#17252f;background:#f6f7f8}section{background:white;padding:18px;margin:18px 0;'
            'border-radius:12px}audio{width:100%;max-width:650px}small{color:#566}h2{margin:0}</style>'
            '<title>'+html.escape(title)+'</title><h1>'+html.escape(title)+'</h1><p>'+html.escape(explanation)+'</p>'
            '<p>分别关注：发音清晰度；词间停顿/轻读；连续自然度；重音与节奏；舒适度。'
            '未进行真人盲听，无MOS结论。原始音频未响度匹配，能量控制差异保留。</p>'+''.join(sections)+'</html>')


def build_pages(output, cases, results, failures, seed):
    output = Path(output)
    (output/'blind/audio').mkdir(parents=True)
    labelled, blind, mapping = [], [], []
    rng = random.Random(seed)
    for case in cases:
        cid = case['id']
        head = '<section><h2>'+html.escape(cid)+'</h2><p>'+html.escape(case['text'])+'</p>'
        label = head
        concealed = head
        versions = [r for r in results if r['case_id'] == cid]
        for row in versions:
            relative = row['wave_relative']
            label += '<p>'+html.escape(row['system'])+'</p><audio controls preload="none" src="'+html.escape(relative, quote=True)+'"></audio>'
        shuffled = list(versions); rng.shuffle(shuffled)
        for i, row in enumerate(shuffled):
            letter = chr(65+i)
            anonymous = cid+'_'+letter+'.wav'
            shutil.copyfile(output/row['wave_relative'], output/'blind/audio'/anonymous)
            concealed += '<p>版本 '+letter+'</p><audio controls preload="none" src="audio/'+anonymous+'"></audio>'
            mapping.append({'case_id': cid, 'letter': letter, 'system': row['system'],
                            'anonymous_wave': 'blind/audio/'+anonymous, 'wave_sha256': row['wave_sha256']})
        failed = [r for r in failures if r['case_id'] == cid]
        for row in failed:
            message = '<p><strong>FAILED：</strong>'+html.escape(row['reason'])+'</p>'
            label += message; concealed += message
        labelled.append(label+'</section>'); blind.append(concealed+'</section>')
    note = '纯文本预测时长与8×39控制点，固定warped700 C DSP。ZERO_PLAN_CONTROL仅是同模型计划置零干预，不是匹配训练基线。'
    (output/'index.html').write_text(_html_document('新语种模型：有标签试听', labelled, note), encoding='utf8')
    (output/'blind/index.html').write_text(_html_document('新语种模型：匿名同句试听', blind,
        '版本顺序按固定seed随机；答案映射位于试听目录外侧的blind_map.json。'), encoding='utf8')
    save_json(output/'blind_map.json', {'seed': seed, 'mapping': mapping, 'human_listening': 'NOT_RUN'})


def create_pack(root, language, data, lexicon, training, *, output=None, seed=20261017, zero_control=True):
    torch.set_num_threads(1)
    output = Path(output or Path(root)/language/'listen')
    if output.exists():
        raise FileExistsError('Use a new listening output directory: '+str(output))
    output.mkdir(parents=True)
    started = time.perf_counter()
    save_json(output/'status.json', {'status': 'RUNNING', 'language': language})
    save_json(output/'config.json', {'root': str(Path(root).resolve()), 'language': language,
        'data': str(Path(data).resolve()), 'lexicon': str(Path(lexicon).resolve()),
        'training': str(Path(training).resolve()), 'output': str(output.resolve()),
        'seed': seed, 'zero_control': zero_control, 'script_sha256': sha256(__file__),
        'command_argv': sys.argv, 'python_executable': sys.executable})
    try:
        cases, selection = select_cases(data, language)
        front, planner, model, identity = load_system(language, lexicon, training)
        save_json(output/'manifest.json', {'cases': cases, 'selection': selection, 'identity': identity,
                  'systems': ['PREDICTED_PLAN']+(['ZERO_PLAN_CONTROL'] if zero_control else [])})
        results, failures = [], []
        for case in cases:
            # Preflight ordinary text once. OOV is recorded once per case; never fixed.
            try:
                front.encode(case['text'])
            except ValueError as error:
                failures.append({'case_id': case['id'], 'status': 'FAILED', 'reason': str(error), 'text': case['text']})
                continue
            for system in ['PREDICTED_PLAN']+(['ZERO_PLAN_CONTROL'] if zero_control else []):
                relative = 'wav/raw/'+case['id']+'_'+system+'.wav'
                try:
                    result = synthesize_case(case['text'], system, front, planner, model, output/relative, identity)
                    results.append({'case_id': case['id'], 'split': case['split'], 'source_id': case['source_id'],
                                    'wave_relative': relative, **result})
                except Exception as error:
                    failures.append({'case_id': case['id'], 'system': system, 'status': 'FAILED', 'reason': str(error),
                                     'text': case['text'], 'traceback': traceback.format_exc()})
        build_pages(output, cases, results, failures, seed)
        save_json(output/'results.json', {'status': 'COMPLETED', 'records': results, 'failed': failures})
        fields = ['case_id', 'source_id', 'split', 'system', 'status', 'audio_seconds', 'frames', 'units',
                  'PCM_peak', 'PCM_RMS', 'clipping_fraction', 'wall_seconds', 'RTF', 'host_process_peak_RSS_bytes',
                  'wave_relative', 'wave_sha256']
        with (output/'per_sentence.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(results)
        summary = {'status': 'COMPLETED' if results else 'FAILED', 'language': language, 'selection': selection,
            'cases': len(cases), 'completed_audio_count': len(results), 'failed_cases_or_versions': failures,
            'identity': identity, 'parameters_total': identity['acoustic_parameters']+identity['planner_parameters'],
            'wall_seconds': time.perf_counter()-started, 'host_process_peak_RSS_bytes': rss_bytes(),
            'output': str(output.resolve()), 'human_listening': 'NOT_RUN', 'MOS': None,
            'ordinary_inference': True, 'reference_inputs_used': False, 'ARM_or_target_hardware_test': False,
            'deployment_memory_or_realtime_claim': False}
        save_json(output/'resource_report.json', {
            'status': summary['status'], 'device': 'cpu', 'threads': torch.get_num_threads(),
            'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
            'acoustic_parameters': identity['acoustic_parameters'], 'planner_parameters': identity['planner_parameters'],
            'parameters_total': summary['parameters_total'],
            'acoustic_file_bytes': Path(identity['acoustic_checkpoint']).stat().st_size,
            'planner_file_bytes': Path(identity['planner_checkpoint']).stat().st_size,
            'checkpoint_files_may_include_optimizer': True,
            'host_process_peak_RSS_bytes': summary['host_process_peak_RSS_bytes'],
            'RSS_scope': 'Linux process high-water mark including Python/Torch/frontend; not model-only or deployment memory',
            'wall_seconds': summary['wall_seconds'], 'audio_seconds': sum(r['audio_seconds'] for r in results),
            'maximum_PCM_peak': max((r['PCM_peak'] for r in results), default=None),
            'maximum_clipping_fraction': max((r['clipping_fraction'] for r in results), default=None),
            'per_sentence_metrics': 'per_sentence.csv', 'target_hardware_test': False})
        if failures:
            (output/'FAILED_STEP.md').write_text('# Per-case failures\n\n'+
                '\n'.join('- '+r['case_id']+': '+r['reason'] for r in failures)+'\n\nNo pronunciation was patched. Other cases continued.\n')
        save_json(output/'completed.json', summary); save_json(output/'status.json', summary)
        return summary
    except Exception as error:
        save_json(output/'status.json', {'status': 'FAILED', 'language': language, 'reason': str(error), 'traceback': traceback.format_exc()})
        (output/'FAILED_STEP.md').write_text('# FAILED_STEP\n\n'+traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--language', choices=tuple(CHALLENGES), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--lexicon', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--no-zero-control', action='store_true')
    args = parser.parse_args()
    print(json.dumps(create_pack(args.root, args.language, args.data, args.lexicon, args.training,
                                output=args.output, seed=args.seed, zero_control=not args.no_zero_control), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
