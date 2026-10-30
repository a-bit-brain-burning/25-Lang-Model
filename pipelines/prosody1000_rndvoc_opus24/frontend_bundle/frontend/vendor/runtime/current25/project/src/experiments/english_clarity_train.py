"""Paired English fine-tuning with identical initialization and sampled data.

Run from repository root. Existing completed runs are immutable. --resume uses
the optimizer, RNG and sampler history from each arm's latest checkpoint.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..common import PROJECT, save_json, sha256
from ..frontend.english_clarity import supervision_groups
from ..frontend.universal_english import from_legacy_arrays
from ..models.universal_acoustic import load_acoustic, load_acoustic_state, resource_estimate
from ..training.clarity_objective import clarity_terms, frame_logits, weighted_clarity
from . import universal_train as base
from .upgrade_common import LazyRecords


def configure():
    # Reuse the frozen binary AP5 objective, with an explicit 31D contract.
    base.SPEC, base.BANDS, base.D, base.CONTINUOUS = 24, 5, 31, False


class Records(LazyRecords):
    def __init__(self, root, split):
        if split not in ('train', 'validation'):
            raise ValueError('This experiment does not load test arrays')
        super().__init__(root, split, 24, 5, cache_size=128)
        for row in self.rows:
            path = self.root / row['record_path']
            if not path.is_file():
                path = self.root / 'records' / Path(row['record_path']).name
            row['record_path'] = str(path.resolve())

    def __getitem__(self, index):
        row, arrays = super().__getitem__(index)
        if 'phonetic_features' not in arrays:
            arrays.update(from_legacy_arrays(arrays['phone_ids'], arrays['text_features'], arrays['stress']))
            arrays['_groups'] = supervision_groups(arrays['phone_ids'], arrays['text_features'], arrays['stress'])
        return row, arrays


def batch(records, indices):
    indices = list(indices)
    f, t, m, d = base.batch_records(records, indices)
    shape = f['valid_mask'].shape
    groups = {key: torch.zeros(shape, dtype=torch.float32 if key == 'duration_weight' else torch.bool)
              for key in ('vowel', 'obstruent', 'lexical', 'stressed_vowel', 'final_obstruent', 'duration_weight')}
    left, right = [], []
    for i, index in enumerate(indices):
        a = records[index][1]
        n = len(a['phone_ids'])
        for key in groups:
            groups[key][i, :n] = torch.as_tensor(a['_groups'][key])
        pairs = a['_groups']['vowel_pairs'] + i * shape[1]
        left.extend(pairs[:, 0].tolist())
        right.extend(pairs[:, 1].tolist())
    groups['pair_left'] = torch.tensor(left, dtype=torch.long)
    groups['pair_right'] = torch.tensor(right, dtype=torch.long)
    return f, t, m, d, groups


@torch.inference_mode()
def evaluate(model, records, decoder, weights, selection_supervision='control'):
    model.eval()
    totals, sums, counts = {}, {}, {}

    def accumulate(key, value, mask):
        mask = torch.broadcast_to(mask, value.shape)
        sums[key] = sums.get(key, 0.) + float((value * mask).sum())
        counts[key] = counts.get(key, 0.) + float(mask.sum())

    for start in range(0, len(records), 8):
        indices = range(start, min(start + 8, len(records)))
        f, t, m, d, g = batch(records, indices)
        out = model({k: f[k] for k in base.TEXT_INPUT_KEYS})
        loss, terms = base.objective(out, t, m, d, f, model.scales, decoder,
                                     10**8, model.voiced_log_prior)
        extra = clarity_terms(out, t, m, d, f, model.scales, g)
        frame_base = loss - .7 * terms['vuv_balanced_bce'] + .7 * extra['frame_vuv']
        selection_base = frame_base if selection_supervision == 'frame' else loss
        terms = {'base_total': loss, 'frame_base_total': frame_base,
                 'selection_score': selection_base + weighted_clarity(extra, weights),
                 **{'clarity_' + k: v for k, v in extra.items()},
                 **{k: v for k, v in terms.items() if k != 'total'}}
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.) + float(value) * len(indices)
        error = (out['duration_frames'] - d).abs() * 10
        for group in ('lexical', 'vowel', 'obstruent', 'stressed_vowel', 'final_obstruent'):
            accumulate(group + '_duration_mae_ms', error, g[group] & (d > 0))
        # The actual frame targets are used only for this reference-time diagnostic.
        logits = frame_logits(out['controls'], f)
        true_voice = f['_frame_voice'] > .5
        lexical = g['lexical'].flatten()[f['_frame_indices']]
        accumulate('false_voiced_rate', (logits > 0).float(), ~true_voice & lexical)
        accumulate('false_unvoiced_rate', (logits <= 0).float(), true_voice & lexical)
        pp, tt = out['controls'][..., 2:6, 1:25], t[..., 2:6, 1:25]
        db_error = (decoder(pp) - decoder(tt)) * (10 / math.log(10))
        active = g['vowel'][..., None, None] & (d > 0)[..., None, None]
        accumulate('vowel_mid_spectral_mse_db', db_error.square(), active)
        pitch = (out['controls'][..., 0] - t[..., 0]) * (1200 / math.log(2))
        accumulate('f0_mse_cents', pitch.square(), m[..., 0] > 0)
        ap = (out['controls'][..., 25:30] - t[..., 25:30]).square()
        accumulate('ap5_mse', ap, m[..., 25:30] > 0)
    result = {key: value / len(records) for key, value in totals.items()}
    result.update({key: sums[key] / max(counts[key], 1) for key in sums})
    result['vowel_mid_spectral_rmse_db'] = math.sqrt(result.pop('vowel_mid_spectral_mse_db'))
    result['f0_rmse_cents'] = math.sqrt(result.pop('f0_mse_cents'))
    result['diagnostic_counts'] = counts
    result['recordings'] = len(records)
    return result


def audit(config, output, train, validation):
    ids = [set(r['id'] for r in records.rows) for records in (train, validation)]
    if not all(ids) or ids[0] & ids[1]:
        raise ValueError('Invalid train/validation split')
    digests = []
    for records in (train, validation):
        for row in records.rows:
            digest = sha256(row['record_path'])
            if digest != row['sha256']:
                raise ValueError('Frozen record SHA mismatch: ' + row['id'])
            digests.append({'id': row['id'], 'split': row['split'], 'sha256': digest})
    sources = {}
    snapshot = output / 'source_snapshot'
    paths = sorted((PROJECT / 'src').rglob('*.py')) + sorted((PROJECT / 'src/frontend/assets').glob('*'))
    for path in paths:
        if path.is_file():
            relative = path.relative_to(PROJECT)
            dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            sources[str(relative)] = sha256(path)
    lineage = {'manifest_sha256': sha256(Path(config['data_dir']) / 'manifest.jsonl'),
               'parent_sha256': sha256(config['parent']), 'train_recordings': len(train),
               'train_hours': sum(r['target_frames'] for r in train.rows) / 360000,
               'validation_recordings': len(validation), 'test_arrays_loaded': False,
               'normalization': 'parent train-only buffers retained', 'record_hashes_verified': len(digests),
               'source_sha256': sources, 'command': sys.argv,
               'network_input_keys': list(base.TEXT_INPUT_KEYS),
               'alignment': 'historical automatic unreviewed; not human gold',
               'historical_test': 'previously inspected; not an independent confirmation set'}
    save_json(output / 'record_hashes.json', digests)
    save_json(output / 'split.json', {'train': sorted(ids[0]), 'validation': sorted(ids[1])})
    save_json(output / 'lineage.json', lineage)
    return lineage


def train_arm(arm, config, output, train, validation, lineage, resume=False):
    folder = output / arm
    folder.mkdir(exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True) if resume and (folder / 'latest.pt').exists() else None
    model = load_acoustic_state(state) if state else load_acoustic(config['parent'])
    if model.channels != 31 or model.config['continuous_voicing'] or model.config['context_dim'] != 22:
        raise ValueError('Requires the numeric binary AP5 parent')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    decoder = base.PhysicalSpectrum()
    started = time.monotonic()
    step, history, best_step = 0, [], 0
    if state:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng']['torch'])
        random.setstate(state['rng']['python'])
        rng.bit_generator.state = state['rng']['sampler']
        step, history = state['step'], state['sample_history']
        best_step = state['best_step']
        best_score = state['best_score']
    else:
        initial = evaluate(model, validation, decoder, config['weights'], config.get('selection_supervision', 'control'))
        best_score = initial['selection_score']
        save_json(folder / 'validation_000000.json', initial)
        base.save_checkpoint(folder / 'initial.pt', model, optimizer, 0, config, initial, lineage, rng)
        shutil.copyfile(folder / 'initial.pt', folder / 'best.pt')
        print(json.dumps({'arm': arm, 'step': 0, 'validation': initial}), flush=True)
    save_json(folder / 'resources.json', resource_estimate(model))
    with (folder / 'training.jsonl').open('a') as log:
        for step in range(step + 1, config['steps'] + 1):
            model.train()
            indices = rng.choice(len(train), min(config['batch_records'], len(train)), replace=False)
            history.append(hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
            f, t, m, d, groups = batch(train, indices)
            pred = model({k: f[k] for k in base.TEXT_INPUT_KEYS})
            loss, base_terms = base.objective(pred, t, m, d, f, model.scales, decoder,
                                     config['physical_samples'], model.voiced_log_prior)
            extra = clarity_terms(pred, t, m, d, f, model.scales, groups)
            training_loss = loss + (weighted_clarity(extra, config['weights']) if arm == 'clarity' else 0.)
            if arm == 'clarity' and config.get('replace_control_vuv', False):
                training_loss = training_loss - .7 * base_terms['vuv_balanced_bce'] + .7 * extra['frame_vuv']
            if not torch.isfinite(training_loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            training_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            lr = config['learning_rate'] * (.15 + .85 * .5 * (1 + math.cos(math.pi * step / config['steps'])))
            for group in optimizer.param_groups:
                group['lr'] = lr
            if step == 1 or step % 100 == 0:
                row = {'arm': arm, 'step': step, 'loss': float(training_loss.detach()),
                       'base': float(loss.detach()), 'auxiliary': {k: float(v.detach()) for k, v in extra.items()},
                       'gradient_norm': float(gradient), 'learning_rate': lr, 'wall_seconds': time.monotonic() - started}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
            if step % config['eval_every'] == 0 or step == config['steps']:
                metrics = evaluate(model, validation, decoder, config['weights'], config.get('selection_supervision', 'control'))
                if metrics['selection_score'] < best_score:
                    best_score, best_step = metrics['selection_score'], step
                    base.save_checkpoint(folder / 'best.pt', model, optimizer, step, config, metrics, lineage, rng)
                base.save_checkpoint(folder / 'latest.pt', model, optimizer, step, config, metrics, lineage, rng)
                latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
                latest.update(sample_history=history, best_score=best_score, best_step=best_step)
                torch.save(latest, folder / 'latest.tmp')
                (folder / 'latest.tmp').replace(folder / 'latest.pt')
                save_json(folder / f'validation_{step:06d}.json', metrics)
                save_json(folder / 'progress.json', {'status': 'running', 'step': step, 'best_step': best_step})
                print(json.dumps({'arm': arm, 'step': step, 'score': metrics['selection_score'], 'best_step': best_step}), flush=True)
    selected = load_acoustic(folder / 'best.pt')
    selected_state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(v, selected_state['model'][k]) for k, v in selected.state_dict().items())
    save_json(folder / 'completed.json', {'status': 'complete', 'steps': step, 'best_step': best_step,
              'best_score': best_score, 'checkpoint_sha256': sha256(folder / 'best.pt'),
              'sample_history_sha256': hashlib.sha256(''.join(history).encode()).hexdigest(),
              'checkpoint_reload_exact': True, 'resources': resource_estimate(model),
              'wall_seconds_this_invocation': time.monotonic() - started})
    save_json(folder / 'progress.json', {'status': 'complete', 'step': step, 'best_step': best_step})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('selection_supervision', 'control') not in ('control', 'frame'):
        raise ValueError('Unknown selection supervision')
    if args.output:
        config['output'] = str(args.output)
    output = Path(config['output'])
    if output.exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory, or --resume for an interrupted run')
    if args.resume:
        if config != json.loads((output / 'config.json').read_text()):
            raise ValueError('Resume config differs')
        lineage = json.loads((output / 'lineage.json').read_text())
        if sha256(config['parent']) != lineage['parent_sha256'] or sha256(Path(config['data_dir']) / 'manifest.jsonl') != lineage['manifest_sha256']:
            raise ValueError('Resume data/parent differs')
        if any(sha256(PROJECT / k) != v for k, v in lineage['source_sha256'].items()):
            raise ValueError('Resume source differs; start a new run')
        for row in json.loads((output / 'record_hashes.json').read_text()):
            record = Path(config['data_dir']) / 'records' / (row['id'] + '.npz')
            if sha256(record) != row['sha256']:
                raise ValueError('Resume record differs: ' + row['id'])
    output.mkdir(parents=True, exist_ok=True)
    configure()
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    train, validation = Records(config['data_dir'], 'train'), Records(config['data_dir'], 'validation')
    if not args.resume:
        save_json(output / 'config.json', config)
        lineage = audit(config, output, train, validation)
    save_json(output / 'status.json', {'status': 'running', 'command': sys.argv})
    try:
        for arm in ('control', 'clarity'):
            train_arm(arm, config, output, train, validation, lineage, args.resume)
        summaries = {arm: json.loads((output / arm / 'completed.json').read_text()) for arm in ('control', 'clarity')}
        assert summaries['control']['sample_history_sha256'] == summaries['clarity']['sample_history_sha256']
        save_json(output / 'training_completed.json', {'arms': summaries, 'identical_sample_order': True,
                  'test_evaluated': False, 'quality_improved': 'not established; requires evaluation and listening'})
        save_json(output / 'status.json', {'status': 'training_complete', 'next': 'paired text-only synthesis and listening'})
    except BaseException as exc:
        save_json(output / 'status.json', {'status': 'interrupted', 'error': repr(exc), 'resume': '--resume'})
        raise


if __name__ == '__main__':
    main()
