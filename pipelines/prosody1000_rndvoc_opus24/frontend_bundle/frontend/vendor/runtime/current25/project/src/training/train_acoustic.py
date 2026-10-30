"""Train the small joint acoustic/duration baseline on real paired labels.

Only phone IDs and twelve text-derived features enter the network. Statistics,
optimization and model selection use train/validation; test is never loaded.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..models.acoustic import FORMAT, TinyAcousticModel, load_acoustic, load_acoustic_state, resource_estimate
from .prepare_data import load_records

TEXT_INPUT_KEYS = ('phone_ids', 'text_features', 'valid_mask', 'edge_sil_mask', 'stress')
TRAIN_FIELDS = ('phone_ids', 'text_features', 'target_controls', 'control_mask', 'target_duration',
                'target_curve', 'stress', 'edge_sil_mask', 'edge_mask')


def load_training_records(data_dir, split):
    """Load only used labels/features, leaving redundant frame masks on disk."""
    if split not in ('train', 'validation'):
        raise ValueError('Training loader refuses the test split')
    data_dir = Path(data_dir)
    rows = [json.loads(line) for line in (data_dir / 'manifest.jsonl').read_text().splitlines() if line.strip()]
    result = []
    for row in rows:
        if row['split'] != split:
            continue
        with np.load(data_dir / row.get('record_path', row.get('path')), allow_pickle=False) as asset:
            result.append((row, {key: asset[key] for key in TRAIN_FIELDS if key in asset.files}))
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def normalization(records, vuv_supervision='control'):
    total = np.zeros(29, np.float64)
    square = total.copy()
    count = total.copy()
    voiced = units = 0.
    for _, a in records:
        t, mask = a['target_controls'].reshape(-1, 29), a['control_mask'].reshape(-1, 29)
        total += (t * mask).sum(0)
        square += (t * t * mask).sum(0)
        count += mask.sum(0)
        voice = 1 / (1 + np.exp(-np.clip(t[:, 28], -30, 30)))
        if vuv_supervision == 'frame':
            voiced += a['target_curve'][:, 28].sum()
            units += len(a['target_curve'])
        else:
            voiced += (voice * mask[:, 28]).sum()
            units += mask[:, 28].sum()
    mean = total / np.maximum(count, 1)
    scales = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean ** 2, 0))
    scales = np.maximum(scales, np.array([.1, .5] + [.05] * 26 + [1.]))
    mean[28], scales[28] = 0., 1.
    return mean.astype(np.float32), scales.astype(np.float32), float(voiced / units)


def batch_records(records, indexes):
    chosen = [records[int(i)][1] for i in indexes]
    width = max(len(a['phone_ids']) for a in chosen)
    b = len(chosen)
    ids = torch.zeros(b, width, dtype=torch.long)
    text = torch.zeros(b, width, 12)
    valid = torch.zeros(b, width, dtype=torch.bool)
    edge = torch.zeros_like(valid)
    target = torch.zeros(b, width, 8, 29)
    mask = torch.zeros_like(target)
    duration = torch.zeros(b, width)
    stress = torch.full((b, width), -1., dtype=torch.float32)
    frame_indices, frame_fractions, frame_voice = [], [], []
    for i, a in enumerate(chosen):
        n = len(a['phone_ids'])
        ids[i, :n] = torch.from_numpy(a['phone_ids'].astype(np.int64))
        text[i, :n] = torch.from_numpy(a['text_features'].astype(np.float32))
        valid[i, :n] = True
        if 'edge_sil_mask' in a:
            edge[i, :n] = torch.from_numpy(a['edge_sil_mask'].astype(bool))
        else:
            edge[i, 0] = edge[i, n - 1] = True
        target[i, :n] = torch.from_numpy(a['target_controls'].astype(np.float32))
        mask[i, :n] = torch.from_numpy(a['control_mask'].astype(np.float32))
        duration[i, :n] = torch.from_numpy(a['target_duration'].astype(np.float32))
        if 'stress' in a:
            stress[i, :n] = torch.from_numpy(a['stress'].astype(np.float32))
        counts = a['target_duration'].astype(np.int64)
        frame_indices.append(np.repeat(np.arange(n, dtype=np.int64) + i * width, counts))
        frame_fractions.extend([np.arange(count, dtype=np.float32) / count for count in counts if count > 0])
        frame_voice.append(a['target_curve'][:, 28].astype(np.float32))
    return {'phone_ids': ids, 'text_features': text, 'valid_mask': valid, 'edge_sil_mask': edge,
            'stress': stress, '_frame_indices': torch.from_numpy(np.concatenate(frame_indices)),
            '_frame_fractions': torch.from_numpy(np.concatenate(frame_fractions)),
            '_frame_voice': torch.from_numpy(np.concatenate(frame_voice))}, target, mask, duration


class PhysicalSpectrum(nn.Module):
    def __init__(self, bins=513):
        super().__init__()
        x = torch.arange(bins, dtype=torch.float32) + .5
        modes = torch.arange(1, 24, dtype=torch.float32)
        self.register_buffer('basis', math.sqrt(2) * torch.cos(math.pi / bins * modes[:, None] * x))
        w = torch.ones(bins)
        w[0] = w[-1] = .5
        self.register_buffer('log_weights', w.log())
        self.logsum = math.log(bins - 1)

    def forward(self, spectral):
        shape = (spectral[..., 1:] @ self.basis).clamp(-80, 80)
        norm = torch.logsumexp(shape + self.log_weights, -1, keepdim=True) - self.logsum
        return spectral[..., :1] + shape - norm


def objective(out, target, mask, duration, features, scales, decoder, physical_samples=768, voiced_log_prior=0.,
              vuv_supervision='control'):
    p = out['controls']
    norm_error = (p - target) / scales
    avg = lambda x, m: (x * m).sum() / m.sum().clamp_min(1)
    spectral = avg(norm_error[..., 2:25].square(), mask[..., 2:25])
    energy = avg(norm_error[..., 1].square(), mask[..., 1])
    pitch = avg(F.smooth_l1_loss(norm_error[..., 0], torch.zeros_like(norm_error[..., 0]), reduction='none'), mask[..., 0])
    ap = avg(norm_error[..., 25:28].square(), mask[..., 25:28])
    if vuv_supervision == 'frame':
        positions = features['_frame_fractions'] * 7
        left = positions.long().clamp(0, 6)
        alpha = positions - left
        c = p.reshape(-1, 8, 29)[features['_frame_indices'], :, 28]
        row = torch.arange(len(left))
        logits = c[row, left] * (1 - alpha) + c[row, left + 1] * alpha - voiced_log_prior
        voice = features['_frame_voice']
        bce = F.binary_cross_entropy_with_logits(logits, voice, reduction='none')
        vuv = .5 * avg(bce, voice) + .5 * avg(bce, 1 - voice)
    else:
        voice = torch.sigmoid(target[..., 28])
        # Control surrogate labels retain the fitted transitions. The explicit
        # frame variant instead trains against the original binary WORLD VUV.
        bce = F.binary_cross_entropy_with_logits(p[..., 28] - voiced_log_prior, voice, reduction='none')
        vuv = .5 * avg(bce, mask[..., 28] * voice) + .5 * avg(bce, mask[..., 28] * (1 - voice))
    active = (mask[..., 1] > 0).flatten().nonzero().flatten()
    if len(active) > physical_samples:
        # Random control subsampling is training-only; evaluation computes all controls.
        active = active[torch.randperm(len(active))[:physical_samples]]
    pp, tt = p.reshape(-1, 29)[active], target.reshape(-1, 29)[active]
    db_error = (decoder(pp[:, 1:25]) - decoder(tt[:, 1:25])) / math.log(10)
    physical = db_error.square().mean()
    valid, edge = features['valid_mask'], features['edge_sil_mask']
    logduration = F.smooth_l1_loss(torch.log1p(out['duration_frames']), torch.log1p(duration), reduction='none', beta=.5)
    local = avg(logduration, valid)
    delta = out['duration_frames'] - duration
    count = valid.sum(-1)
    position = torch.arange(1, duration.shape[1] + 1)[None]
    inner = valid & (position < count[:, None])
    boundary = avg(F.smooth_l1_loss(delta.cumsum(-1) / (5 * position.sqrt()), torch.zeros_like(delta), reduction='none'), inner)
    sentence = F.smooth_l1_loss(delta.sum(-1) / (5 * count.sqrt()), torch.zeros_like(count, dtype=torch.float32))
    edge_bce = avg(F.binary_cross_entropy_with_logits(out['edge_nonzero_logits'], (duration > 0).float(), reduction='none'), edge & valid)
    timing = local + .3 * boundary + .3 * sentence + .1 * edge_bce
    total = physical + .15 * spectral + .1 * energy + .3 * pitch + .2 * ap + .7 * vuv + .3 * timing
    return total, {'total': total, 'physical_logpower_mse_db_div10': physical, 'shape_normalized_mse': spectral,
                  'energy_normalized_mse': energy, 'f0_normalized_huber': pitch, 'ap_normalized_mse': ap,
                  'vuv_balanced_bce': vuv, 'duration': timing, 'duration_log_huber': local,
                  'duration_boundary': boundary, 'duration_sentence': sentence, 'edge_nonzero_bce': edge_bce}


@torch.inference_mode()
def evaluate(model, records, decoder):
    model.eval()
    totals = {}
    frames_error = voice_correct = voice_count = duration_error = phone_count = 0.
    for start in range(0, len(records), 8):
        f, t, m, d = batch_records(records, range(start, min(start + 8, len(records))))
        out = model({k: f[k] for k in TEXT_INPUT_KEYS})
        _, terms = objective(out, t, m, d, f, model.scales, decoder, physical_samples=10000000,
                             voiced_log_prior=model.voiced_log_prior,
                             vuv_supervision=model.config.get('vuv_supervision', 'control'))
        weight = min(8, len(records) - start)
        for k, v in terms.items():
            totals[k] = totals.get(k, 0.) + float(v) * weight
        vm = m[..., 28] > 0
        voice_correct += (((out['controls'][..., 28] > 0) == (t[..., 28] > 0)) & vm).sum().item()
        voice_count += vm.sum().item()
        valid = f['valid_mask']
        duration_error += ((out['duration_frames'] - d).abs() * valid).sum().item()
        phone_count += valid.sum().item()
        frames_error += (out['duration_frames'].sum(-1) - d.sum(-1)).abs().sum().item()
    result = {k: v / len(records) for k, v in totals.items()}
    result.update(duration_mae_frames=duration_error / phone_count,
                  sentence_length_mae_seconds=.01 * frames_error / len(records),
                  control_voicing_accuracy=voice_correct / voice_count, recordings=len(records))
    return result


def save_checkpoint(path, model, optimizer, step, config, validation, lineage, rng=None):
    state = {'format': FORMAT, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'step': step, 'config': model.config, 'training_config': config,
             'validation': validation, 'lineage': lineage, 'saved_utc': datetime.now(timezone.utc).isoformat()}
    if rng is not None:
        state['rng'] = {'torch': torch.get_rng_state(), 'sampler': rng.bit_generator.state, 'python': random.getstate()}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('data_dir', 'output', 'steps', 'seconds'):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'running.json', {'pid_in_tool_namespace': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
               'command': 'python -B -m project.src.training.train_acoustic --config ' + str(args.config),
               'status': 'running; completed.json marks normal completion'})
    torch.set_num_threads(min(4, config.get('threads', 1)))
    torch.set_num_interop_threads(1)
    seed = config.get('seed', 20261014)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    train = load_training_records(Path(config['data_dir']), 'train')
    validation = load_training_records(Path(config['data_dir']), 'validation')
    train_ids = {str(r.get('id')) for r, _ in train}
    val_ids = {str(r.get('id')) for r, _ in validation}
    if train_ids & val_ids or not train or not validation:
        raise ValueError('Invalid train/validation split')
    means, scales, prior = normalization(train, config.get('vuv_supervision', 'control'))
    model_config = {**config['model'], 'voiced_prior': prior, 'vuv_supervision': config.get('vuv_supervision', 'control')}
    model = TinyAcousticModel(model_config, means, scales)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', .001), weight_decay=1e-4)
    start_step = 0
    resume_sha256 = None
    if args.resume:
        resume_bytes = args.resume.read_bytes()
        resume_sha256 = hashlib.sha256(resume_bytes).hexdigest()
        state = torch.load(io.BytesIO(resume_bytes), map_location='cpu', weights_only=True)
        model = load_acoustic_state(state)
        for name in ('width', 'embedding_dim', 'dilations', 'vocabulary_size', 'feature_dim', 'control_points'):
            if name in config['model'] and config['model'][name] != model.config[name]:
                raise ValueError('Resume architecture mismatch for ' + name)
        if 'dropout' in config['model']:
            model.config['dropout'] = config['model']['dropout']
            for block in model.blocks:
                block.dropout.p = config['model']['dropout']
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', .001), weight_decay=1e-4)
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = config.get('learning_rate', .001)
        start_step = state['step']
        if 'rng' in state:
            torch.set_rng_state(state['rng']['torch'])
            random.setstate(state['rng']['python'])
            rng.bit_generator.state = state['rng']['sampler']
        model.config['voiced_prior'] = prior
        model.config['vuv_supervision'] = config.get('vuv_supervision', 'control')
        model.voiced_log_prior.fill_(math.log(prior / (1 - prior)))
    decoder = PhysicalSpectrum()
    manifest = Path(config['data_dir']) / 'manifest.jsonl'
    source_paths = [Path(__file__), Path(__file__).resolve().parents[1] / 'models/acoustic.py',
                    Path(__file__).with_name('prepare_data.py')]
    source_hashes = {}
    (out / 'source_snapshot').mkdir(exist_ok=True)
    for source_path in source_paths:
        content = source_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        source_hashes[str(source_path.resolve())] = digest
        (out / 'source_snapshot' / (source_path.stem + '_' + digest[:12] + '.py')).write_bytes(content)
    lineage = {'data_dir': str(Path(config['data_dir']).resolve()),
               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
               'train_recordings': len(train), 'validation_recordings': len(validation),
               'test_loaded': False, 'statistics_source': 'train only',
               'model_input_keys': ['phone_ids', 'text_features'] + (['stress'] if model.config['feature_dim'] == 13 else []),
               'stress_encoding': 'dictionary stress/2; consonant or unknown -0.5' if model.config['feature_dim'] == 13 else None,
               'resume': str(args.resume) if args.resume else None, 'resume_checkpoint_sha256': resume_sha256,
               'source_sha256': source_hashes, 'loaded_record_fields': list(TRAIN_FIELDS)}
    write_json(out / 'config.json', config)
    write_json(out / 'normalization.json', {'means': model.means.tolist(), 'scales': model.scales.tolist(),
               'voiced_prior': float(torch.sigmoid(model.voiced_log_prior)),
               'normalization_source': 'parent checkpoint buffers retained' if args.resume else 'current train only'})
    write_json(out / 'resources.json', resource_estimate(model))
    write_json(out / 'lineage.json', lineage)
    best_score = float('inf')
    best_step = 0
    if args.resume:
        # Validate the resumed model on this run's validation set before updates.
        # Existing best in the same output folder must never be silently lost.
        with torch.random.fork_rng(devices=[]):
            candidate = load_acoustic(out / 'best.pt') if (out / 'best.pt').exists() else model
        parent_validation = evaluate(candidate, validation, decoder)
        best_score = parent_validation['total']
        best_step = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)['step'] if (out / 'best.pt').exists() else start_step
        if not (out / 'best.pt').exists():
            save_checkpoint(out / 'best.pt', model, optimizer, start_step, config, parent_validation, lineage, rng)
        write_json(out / 'resume_validation.json', parent_validation)
    learning_rate = config.get('learning_rate', .001)
    steps = config.get('steps', 20000)
    wall = config.get('seconds', 1800.)
    eval_every = config.get('eval_every', 500)
    record_count = config.get('batch_records', 12)
    final_validation = None
    step = start_step
    print(json.dumps({'start': lineage, 'parameters': resource_estimate(model)['parameters']}), flush=True)
    with (out / 'training.jsonl').open('a') as log:
        for update in range(1, steps + 1):
            if time.monotonic() - started > wall - 15:
                break
            model.train()
            indices = rng.choice(len(train), min(record_count, len(train)), replace=False)
            features, target, mask, duration = batch_records(train, indices)
            prediction = model({k: features[k] for k in TEXT_INPUT_KEYS})
            loss, terms = objective(prediction, target, mask, duration, features, model.scales, decoder,
                                    config.get('physical_samples', 768), model.voiced_log_prior,
                                    model.config.get('vuv_supervision', 'control'))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite acoustic loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            step = start_step + update
            progress = min(1., max(update / steps, (time.monotonic() - started) / wall))
            lr = learning_rate * (.15 + .85 * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            if update == 1 or update % 100 == 0:
                row = {'step': step, 'wall_seconds': time.monotonic() - started,
                       'loss': {k: float(v.detach()) for k, v in terms.items()}, 'learning_rate': lr,
                       'gradient_norm': float(gradient_norm), 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
            if update == 100 or update % eval_every == 0:
                final_validation = evaluate(model, validation, decoder)
                write_json(out / f'validation_step{step:06d}.json', final_validation)
                save_checkpoint(out / 'latest.pt', model, optimizer, step, config, final_validation, lineage, rng)
                score = final_validation['total']
                if score < best_score:
                    best_score, best_step = score, step
                    save_checkpoint(out / 'best.pt', model, optimizer, step, config, final_validation, lineage, rng)
                print(json.dumps({'step': step, 'validation': final_validation, 'best_step': best_step}), flush=True)
        final_validation = evaluate(model, validation, decoder)
        save_checkpoint(out / 'latest.pt', model, optimizer, step, config, final_validation, lineage, rng)
        if final_validation['total'] < best_score:
            best_score, best_step = final_validation['total'], step
            save_checkpoint(out / 'best.pt', model, optimizer, step, config, final_validation, lineage, rng)
    # Checkpoint reload is an actual runtime interface check.
    loaded = load_acoustic(out / 'best.pt')
    feature = batch_records(validation, [0])[0]
    original = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(v, original['model'][k]) for k, v in loaded.state_dict().items())
    write_json(out / 'completed.json', {'status': 'complete', 'final_step': step, 'best_step': best_step,
               'best_validation_total': best_score, 'latest_validation': final_validation,
               'wall_seconds': time.monotonic() - started, 'test_evaluated': False,
               'checkpoint_reload_exact': True, 'resources': resource_estimate(model)})
    write_json(out / 'running.json', {'pid_in_tool_namespace': os.getpid(), 'status': 'complete',
                                    'finished_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'complete': str(out / 'completed.json'), 'best_step': best_step}), flush=True)


if __name__ == '__main__':
    main()
