"""Publish independent, audited real LJSpeech labels for a text-only tiny model.

Reuses existing real WORLD and automatic alignment caches, but recomputes
waveform-energy labels and fits all controls without any old neural prediction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from ..audio.extractor import read_audio, short_time_energy
from ..audio.schema import CHANNEL_NAMES, ENERGY_DEFINITION, clip_controls
from ..frontend import from_text

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
DEFAULT_DATA = PROJECT/'outputs/data'
RECORD_KEYS = ('phone_ids', 'phonemes', 'text_features', 'language_id', 'target_controls', 'target_duration', 'control_mask', 'duration_mask', 'edge_mask', 'target_curve', 'frame_mask', 'stress', 'word_boundary')

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def interpolation_basis(n):
    p = np.arange(n, dtype=np.float64) / n * 7
    left = np.minimum(p.astype(np.int64), 6)
    alpha = p-left
    matrix = np.zeros((n, 8))
    matrix[np.arange(n), left] = 1-alpha
    matrix[np.arange(n), left+1] = alpha
    return matrix

def fit_controls(target, durations, ridge=.05):
    """Fit real observations in the same eight-knot basis used in inference.

UV pitch/AP and zero-length edge phones are missing observations. Their safe
neutral values carry zero loss weight; no legacy model is used for any label.
"""
    durations = np.asarray(durations, dtype=np.int64)
    if int(durations.sum()) != len(target) or np.any(durations < 0):
        raise ValueError('Actual phone boundaries must exactly cover real frames')
    result = np.zeros((len(durations), 8, 29), dtype=np.float32)
    result[..., 0] = np.log(150)
    result[..., 1] = -12
    result[..., 28] = -3
    mask = np.zeros_like(result)
    offset = 0
    for phone, n in enumerate(durations):
        if not n:
            continue
        t = target[offset:offset+n].astype(np.float64)
        offset += n
        x = interpolation_basis(n)
        voice = t[:, 28] > .5
        for columns, valid in [(list(range(1, 25))+[28], np.ones(n, dtype=bool)), ([0, 25, 26, 27], voice)]:
            if not np.any(valid):
                continue
            y = t[:, columns].copy()
            if 28 in columns:
                y[:, -1] = np.where(voice, 3., -3.)
            xx = x[valid]
            yy = y[valid]
            anchor = np.column_stack([np.interp(np.arange(8)/7, np.arange(n)[valid]/n, yy[:, c]) for c in range(len(columns))])
            fit = np.linalg.solve(xx.T@xx + ridge*np.eye(8), xx.T@yy + ridge*anchor)
            result[phone][:, columns] = fit
            mask[phone][:, columns] = 1
    return clip_controls(result), mask

def load_records(data_dir=DEFAULT_DATA, split='train'):
    data_dir = Path(data_dir)
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown dataset split')
    rows = [json.loads(line) for line in (data_dir/'manifest.jsonl').read_text().splitlines() if line.strip()]
    records = []
    for row in rows:
        if row['split'] != split:
            continue
        path = data_dir/row.get('record_path', row.get('path'))
        with np.load(path, allow_pickle=False) as asset:
            records.append((row, {key: asset[key] for key in asset.files}))
    return records

def flat_records(records):
    keys = ('phone_ids', 'text_features', 'language_id', 'target_controls', 'target_duration', 'control_mask', 'duration_mask', 'edge_mask')
    out = {key: np.concatenate([arrays[key] for _, arrays in records]) for key in keys}
    out['context_ids'] = np.concatenate([context_ids(arrays['phone_ids']) for _, arrays in records])
    out['record_index'] = np.concatenate([np.full(len(arrays['phone_ids']), i) for i, (_, arrays) in enumerate(records)])
    return out

def context_ids(ids):
    padded = np.pad(np.asarray(ids), (2, 2))
    return np.stack([padded[i:i+len(ids)] for i in range(5)], axis=-1)

def unpack_source_record(record, source, published):
    """Adapt published real WORLD/alignment schema without importing parent code."""
    identifier = record['id']['value']
    metadata = record['metadata']['value']
    plain = from_text(record['text']['value'], oov_policy='error')
    candidate = metadata.get('publisher_normalized_text') or published.get(identifier)
    front = plain
    if candidate:
        try:
            restored = from_text(candidate, oov_policy='error')
            if restored['normalized_text'] == plain['normalized_text'] and restored['phones'] == plain['phones']:
                front = restored
        except ValueError:
            pass
    if front['phones'] != record['phonemes']['value']:
        raise ValueError('Actual alignment phones differ from text for '+identifier)
    asset_info = record['spectral24']['value']
    asset = source/asset_info['path']
    with np.load(asset, allow_pickle=False) as z:
        target = np.column_stack([np.log(np.maximum(z['f0'], 1)), z['spectral24'], z['ap3'], z['vuv']]).astype(np.float32)
    bounds = [(round(p['start_s']*100), round(p['end_s']*100)) for p in record['phoneme_boundaries']['value']]
    counts = np.asarray([b-a for a, b in bounds], dtype=np.int64)
    if not bounds or bounds[0][0] != 0 or bounds[-1][1] != len(target) or any(b!=c for (_, b), (c, _) in zip(bounds, bounds[1:])):
        raise ValueError('Actual phoneme boundaries do not partition real audio frames')
    n = len(counts)
    arrays = {'phone_ids': front['phone_ids'].numpy(), 'phonemes': np.asarray(front['phones']), 'text_features': front['text_features'].numpy(), 'language_id': np.ones(n, dtype=np.int64), 'target_duration': counts, 'duration_mask': np.ones(n, dtype=np.float32), 'edge_mask': front['edge_sil_mask'].numpy(), 'target_curve': target}
    row = {'id': identifier, 'split': metadata['split'], 'source_group': metadata['source_group'], 'text': front['punctuated_text'], 'language': 'en', 'language_id': 1, 'phone_count': n, 'target_frames': len(target), 'source_audio': record['audio_path']['value'], 'source_audio_sha256': metadata['audio_sha256'], 'raw_audio_sha256': metadata['raw_audio_sha256'], 'source_acoustics': str(asset), 'source_acoustics_sha256': asset_info['sha256'], 'boundary_quality': 'automatic_unreviewed'}
    return row, arrays

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=WORKSPACE/'dataset')
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--max-recordings', type=int, default=0, help='0 uses every cached real recording')
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        raise ValueError('Published artifacts must stay inside project/')
    begin = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    (output/'records').mkdir(exist_ok=True)
    source_manifest = source/'manifest.jsonl'
    rows = [json.loads(x) for x in source_manifest.read_text().splitlines() if x.strip()]
    if args.max_recordings:
        rows = rows[:args.max_recordings]
    protected = {str(source_manifest): sha256(source_manifest)}
    manifest, summaries = [], {}
    groups = {s: set() for s in ('train', 'validation', 'test')}
    ids, texts, audio_hashes = set(), set(), set()
    transcript_path = WORKSPACE.parent/'tts/data/external/ljspeech-1.1/extracted/LJSpeech-1.1/metadata.csv'
    published = {r[0]: r[2] for line in transcript_path.read_text().splitlines() if len(r:=line.split('|', 2)) == 3}
    if transcript_path.exists():
        protected[str(transcript_path)] = sha256(transcript_path)
    with (output/'prepare_progress.jsonl').open('w') as log:
        for index, row in enumerate(rows):
            schema_record = row if isinstance(row.get('id'), dict) else None
            if schema_record is not None:
                row, schema_arrays = unpack_source_record(schema_record, source, published)
            identifier, split = row['id'], row['split']
            group = row['source_group']
            if split not in groups or group != identifier.split('-')[0]:
                raise ValueError('Invalid authoritative source-series assignment')
            if identifier in ids or row['text'] in texts or row['source_audio_sha256'] in audio_hashes:
                raise ValueError('Duplicate id, canonical transcript, or audio hash')
            ids.add(identifier); texts.add(row['text']); audio_hashes.add(row['source_audio_sha256'])
            groups[split].add(group)
            oldpath = None
            if schema_record is None:
                oldpath = source.parent/row['path']
                if not oldpath.exists():
                    oldpath = source/row['path']
                if sha256(oldpath) != row['sha256']:
                    raise ValueError('Real cached record hash mismatch: '+identifier)
            wav_path = Path(row['source_audio'])
            if sha256(wav_path) != row['source_audio_sha256']:
                raise ValueError('Actual source WAV hash mismatch: '+identifier)
            source_acoustics = Path(row['source_acoustics'])
            if sha256(source_acoustics) != row['source_acoustics_sha256']:
                raise ValueError('Actual WORLD source asset hash mismatch: '+identifier)
            if oldpath is not None:
                protected[str(oldpath)] = row['sha256']
            protected[str(wav_path)] = row['source_audio_sha256']
            protected[str(source_acoustics)] = row['source_acoustics_sha256']
            if schema_record is None:
                with np.load(oldpath, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in ('phone_ids', 'phonemes', 'text_features', 'language_id', 'target_duration', 'duration_mask', 'edge_mask', 'target_curve')}
            else:
                arrays = schema_arrays
            n = len(arrays['phone_ids'])
            if arrays['text_features'].shape != (n, 12) or arrays['target_curve'].shape[1] != 29:
                raise ValueError('Frontend feature or acoustic schema mismatch')
            audio = read_audio(wav_path)
            target = arrays['target_curve'].astype(np.float32)
            energy = short_time_energy(audio, len(target))
            old_energy = target[:, 1].copy()
            target[:, 1] = np.log(np.maximum(energy, 1e-12))
            controls, control_mask = fit_controls(target, arrays['target_duration'])
            arrays['target_curve'] = target
            arrays['target_controls'] = controls
            arrays['control_mask'] = control_mask
            frame_mask = np.ones_like(target)
            frame_mask[:, [0, 25, 26, 27]] = target[:, 28:29]
            arrays['frame_mask'] = frame_mask
            # The frozen pronunciation and 12 text-only features must reproduce
            # the real aligned phone sequence exactly. CMU stress is additional.
            front = from_text(row['text'], oov_policy='error')
            if front['phones'] != arrays['phonemes'].tolist() or not np.array_equal(front['phone_ids'].numpy(), arrays['phone_ids']):
                raise ValueError('Text-derived phone contract mismatch: '+identifier)
            if not np.allclose(front['text_features'].numpy(), arrays['text_features'], atol=1e-7, rtol=0):
                raise ValueError('Text-derived linguistic feature mismatch: '+identifier)
            arrays['stress'] = np.asarray(front['stress'], dtype=np.int64)
            arrays['word_boundary'] = arrays['text_features'][:, :2].astype(bool)
            path = output/'records'/f'{identifier}.npz'
            np.savez_compressed(path, **arrays)
            result = {**row, 'record_path': str(path.relative_to(output)), 'path': str(path.relative_to(output)), 'sha256': sha256(path), 'publisher_normalized_text': published.get(identifier), 'source_cached_record': str(oldpath) if oldpath else str(source_manifest), 'source_cached_record_sha256': row.get('sha256', protected[str(source_manifest)]), 'source_wav_samples': len(audio), 'source_audio_seconds': len(audio)/16000, 'energy_definition': ENERGY_DEFINITION, 'historical_baseline_used': False}
            manifest.append(result)
            summary = summaries.setdefault(split, {'recordings': 0, 'phones': 0, 'frames': 0, 'wav_seconds': 0., 'duration_zero_edge_phones': 0})
            summary['recordings'] += 1
            summary['phones'] += n
            summary['frames'] += len(target)
            summary['wav_seconds'] += len(audio)/16000
            summary['duration_zero_edge_phones'] += int(np.count_nonzero(arrays['target_duration']==0))
            event = {'id': identifier, 'split': split, 'recordings': index+1, 'wall_seconds': time.monotonic()-begin, 'log_energy_correction_mean': float(np.mean(target[:, 1]-old_energy))}
            log.write(json.dumps(event)+'\n'); log.flush()
            if (index+1) % 100 == 0:
                print(json.dumps(event), flush=True)
    disjoint = not any(groups[a]&groups[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    if not disjoint:
        raise ValueError('Source-series leakage detected')
    (output/'manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in manifest))
    save_json(output/'split.json', {split: [row['id'] for row in manifest if row['split']==split] for split in groups})
    save_json(output/'source_hashes.json', protected)
    save_json(output/'prepare_config.json', {'source': str(source), 'output': str(output), 'max_recordings': args.max_recordings, 'ridge': .05, 'frame_period_ms': 10, 'energy_definition': ENERGY_DEFINITION, 'control_points': 8, 'target_controls_vuv': 'ridge-fit +/-3 surrogate logits; target_curve vuv is binary', 'seed': None, 'synthetic_acoustic_labels': False})
    report = {'status': 'complete', 'dataset': 'LJSpeech 1.1', 'language': 'English', 'speaker_count': 1, 'official_url': 'https://keithito.com/LJ-Speech-Dataset/', 'download_action': 'reused already-downloaded official corpus and real WORLD/alignment caches; no duplicate download', 'source_manifest': str(source_manifest), 'source_manifest_sha256': protected[str(source_manifest)], 'counts': summaries, 'source_groups': {s: sorted(gs) for s, gs in groups.items()}, 'source_groups_disjoint': disjoint, 'duplicate_ids': 0, 'duplicate_canonical_texts': 0, 'duplicate_source_audio_hashes': 0, 'all_source_assets_sha256_verified': True, 'channel_names': CHANNEL_NAMES, 'acoustic_dimensions': '24 spectral + 3 AP + separate logF0/VUV = 29 model columns', 'energy_definition': ENERGY_DEFINITION, 'WORLD_shape': 'orthonormal DCT-II(log power) / sqrt(513), retained modes1..23', 'AP_encoding': 'band power-average amplitude ratio converted dB/60; bands0/1000/4000/8000Hz', 'label_method': 'real frame labels -> regularized eight-point perphone fits; no historical neural prediction input or label', 'unobserved_labels': 'unvoiced F0/AP and zero-length edge phones have loss mask0; neutral placeholders excluded from loss', 'alignment_quality': 'automatic, unreviewed; no human phone-boundary accuracy claim', 'stress': 'real CMU stress annotation when phone order matches; -1 is consonant or unavailable stress; baseline12D features do not use stress', 'frontend_phone_and_feature_exact_checks': len(manifest), 'test_independence': 'source-series separated, but historical evaluation reused; not a new blind test', 'wall_seconds': time.monotonic()-begin, 'record_bytes': sum(p.stat().st_size for p in (output/'records').glob('*.npz')), 'parent_directory_written': False}
    save_json(output/'provenance.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
