#!/usr/bin/env python3
"""Bounded standalone reproduction of the original acoustic training objective."""
import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from new25_acoustic.batching import acoustic_forward, collate
from new25_acoustic.contracts import LANGUAGES, json_write, safe_path, sha256
from new25_acoustic.data import BalancedSchedule, Corpus
from new25_acoustic.losses import acoustic_loss
from new25_acoustic.model import Acoustic

CHECKPOINT_FORMAT = 'new25_acoustic_training_v1'


def source_identity():
    paths = sorted((ROOT / 'new25_acoustic').glob('*.py')) + [Path(__file__).resolve()]
    files = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': files}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class BudgetStop(Exception):
    pass


class Run:
    def __init__(self, path, config, identity, resume):
        self.path = Path(path).resolve()
        if resume:
            if not self.path.is_dir():
                raise ValueError('--resume requires an existing training run')
        else:
            self.path.mkdir(parents=True, exist_ok=False)
        self.lock = (self.path / 'owner.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.config = config
        self.stop = False
        self.started = time.monotonic()
        self.initial_used = 0
        self.budget_path = self.path / 'budget.json'
        if resume:
            self.budget = json.loads(self.budget_path.read_text())
            if self.budget['identity'] != identity:
                raise ValueError('Source/config/data/mode changed; start a new run')
            self.initial_used = self.budget['used_seconds']
        else:
            now = time.time()
            self.budget = {'identity': identity, 'started_unix': now,
                           'deadline_unix': now + config['total_wall_seconds'],
                           'used_seconds': 0, 'launches': 0}
        self.budget['launches'] += 1
        self.previous_signals = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous_signals[sig] = signal.signal(sig, self.signal_stop)
        self.save_budget()

    def signal_stop(self, *_):
        self.stop = True

    def save_budget(self):
        self.budget['used_seconds'] = self.initial_used + time.monotonic() - self.started
        json_write(self.budget_path, self.budget)

    def check(self, stage_started=None, remaining_seconds=None):
        self.save_budget()
        if self.budget['launches'] > self.config['max_launches']:
            raise BudgetStop('Launch budget exhausted')
        roots = (self.path, self.path.parent, ROOT)
        if self.stop or any((root / marker).exists() for root in roots
                            for marker in ('STOP', 'PAUSE', 'SAFETY_STOP')):
            raise BudgetStop('Stop requested; markers preserved')
        if time.time() >= self.budget['deadline_unix']:
            raise BudgetStop('Wall-time budget exhausted')
        if self.budget['used_seconds'] >= self.config['total_active_seconds']:
            raise BudgetStop('Active-time budget exhausted')
        if stage_started is not None and time.monotonic() - stage_started >= remaining_seconds:
            raise BudgetStop('Acoustic active-time budget exhausted')
        if shutil.disk_usage(self.path).free < 4 * 1024**3:
            raise BudgetStop('4 GiB free-space floor')

    def close(self):
        self.save_budget()
        for sig, handler in self.previous_signals.items():
            signal.signal(sig, handler)
        self.lock.close()


def checkpoint(directory, payload):
    path = directory / f'step_{payload["step"]:08d}.pt'
    base = path
    number = 0
    while path.exists():
        number += 1
        path = base.with_name(base.stem + f'_save_{number:02d}' + base.suffix)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    temp.replace(path)
    json_write(directory / 'latest.json', {'path': path.name, 'sha256': sha256(path),
                                          'step': payload['step'], 'time': time.time()})


def warm_start(model, path, corpus):
    path = Path(path).resolve()
    for name, expected in (('vocabulary.json', corpus.vocab), ('normalization.json', corpus.norm)):
        if json.loads((path.parent / name).read_text()) != expected:
            raise ValueError('Warm-start metadata differs from prepared corpus: ' + name)
    weights = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError('--warm-start requires a pure Acoustic state_dict')
    model.load_state_dict(weights, strict=True)
    return sha256(path)


def train(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    cfg = config['acoustic']
    for key in ('max_steps', 'max_seconds', 'lr', 'batch_size'):
        if cfg[key] <= 0:
            raise ValueError('Training bound must be positive: ' + key)
    for key in ('total_wall_seconds', 'total_active_seconds', 'max_launches', 'grad_clip',
                'max_consecutive_overflows', 'status_every', 'save_every'):
        if config[key] <= 0:
            raise ValueError('Configuration value must be positive: ' + key)
    device = torch.device(args.device)
    if device.type != 'cuda' and not (args.smoke_test and device.type == 'cpu'):
        raise ValueError('Formal training requires CUDA; CPU requires explicit --smoke-test')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.smoke_test:
        torch.set_num_threads(args.cpu_threads)
    corpus = Corpus(args.prepared)
    if not args.smoke_test and not corpus.receipt.get('heldout_for_this_run_available'):
        raise ValueError('Formal run requires a sealed group-separated test selected before training')
    run_path = Path(args.run).resolve()
    if run_path.is_relative_to(corpus.prepared):
        raise ValueError('Run output must be outside the frozen prepared corpus')
    source = source_identity()
    identity = {'source': source['sha256'], 'config': sha256(config_path),
                'data': sha256(corpus.prepared / 'COMPLETE.json'),
                'smoke_test': bool(args.smoke_test), 'device_type': device.type}
    run = Run(run_path, config, identity, args.resume)
    try:
        run.check()
    except BudgetStop as error:
        json_write(run.path / 'STOPPED.json', {'reason': str(error), 'quality_pass': False})
        run.close()
        return 3
    json_write(run.path / 'SOURCE_IDENTITY.json', source)
    json_write(run.path / 'PREPARED_RECEIPT.json', corpus.receipt)
    json_write(run.path / 'CONFIG.json', config)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    model = Acoustic(len(corpus.vocab)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(.8, .99), weight_decay=1e-5)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    sampler = BalancedSchedule(corpus, config['seed'])
    step = attempts = overflow = 0
    used_stage = 0.0
    exposure = {language: 0.0 for language in LANGUAGES}
    sample_counts = {language: 0 for language in LANGUAGES}
    origin = {'kind': 'from_scratch', 'weights_sha256': None, 'original_optimizer_restored': False}
    if args.resume:
        if (run.path / 'completed.json').exists():
            raise ValueError('Completed run cannot be resumed; use a new output directory')
        pointer = json.loads((run.path / 'latest.json').read_text())
        cp_path = safe_path(run.path, pointer['path'])
        if sha256(cp_path) != pointer['sha256']:
            raise ValueError('Checkpoint hash differs from latest.json')
        # Own local training checkpoints contain Python/NumPy RNG state. Never
        # use this unrestricted loader for an untrusted downloaded checkpoint.
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        if cp.get('format') != CHECKPOINT_FORMAT or cp['identity'] != identity:
            raise ValueError('Only an unchanged run created by this script can resume')
        if cp['vocab'] != corpus.vocab or cp['normalization'] != corpus.norm:
            raise ValueError('Checkpoint vocabulary/normalization changed')
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        scaler.load_state_dict(cp['scaler'])
        sampler.load_state_dict(cp['sampler'])
        restore_rng(cp['rng'])
        step, attempts, overflow = cp['step'], cp['attempts'], cp['consecutive_overflows']
        used_stage, exposure, sample_counts = cp['stage_seconds'], cp['exposure'], cp['sample_counts']
        origin = cp['origin']
    elif args.warm_start:
        origin = {'kind': 'model_only_warm_start_new_run',
                  'weights_sha256': warm_start(model, args.warm_start, corpus),
                  'original_optimizer_restored': False}
    json_write(run.path / 'ORIGIN.json', origin)
    started = time.monotonic()
    max_steps = min(cfg['max_steps'], 2) if args.smoke_test else cfg['max_steps']

    def save(reason):
        run.save_budget()
        checkpoint(run.path, {
            'format': CHECKPOINT_FORMAT, 'phase': 'acoustic', 'model': model.state_dict(),
            'vocab': corpus.vocab, 'normalization': corpus.norm, 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'rng': rng_state(), 'sampler': sampler.state_dict(),
            'step': step, 'attempts': attempts, 'consecutive_overflows': overflow,
            'stage_seconds': used_stage + time.monotonic() - started,
            'exposure': exposure, 'sample_counts': sample_counts, 'identity': identity,
            'source': source, 'data_receipt': corpus.receipt, 'config': config,
            'budget': run.budget, 'origin': origin, 'reason': reason,
            'smoke_test': bool(args.smoke_test), 'quality_pass': False,
        })

    try:
        while step < max_steps:
            run.check(started, cfg['max_seconds'] - used_stage)
            indices = sampler.next_batch(1 if args.smoke_test else cfg['batch_size'])
            batch = collate(corpus, indices, device)
            language = batch['rows'][0]['language']
            optimizer.zero_grad(set_to_none=True)
            attempts += 1
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred, phone_pred = acoustic_forward(model, batch)
            loss, terms = acoustic_loss(pred, batch, phone_pred)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            old_scale = scaler.get_scale()
            if not amp and not torch.isfinite(grad_norm):
                raise FloatingPointError('Nonfinite gradient')
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                overflow += 1
                if overflow >= config['max_consecutive_overflows']:
                    raise FloatingPointError('Repeated FP16 overflow; checkpoint preserved')
                continue
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('Nonfinite gradient')
            overflow = 0
            step += 1
            exposure[language] += sum(batch['lengths']) / 100
            sample_counts[language] += len(indices)
            record = {'phase': 'acoustic', 'step': step, 'attempts': attempts,
                      'actual_cuda_updates': step if amp else 0, 'language': language,
                      'loss': float(loss.detach()),
                      'loss_terms': {key: float(value.detach()) for key, value in terms.items()},
                      'grad_norm': float(grad_norm), 'exposure_seconds_by_language': exposure,
                      'sample_visits_by_language': sample_counts,
                      'stage_seconds': used_stage + time.monotonic() - started,
                      'smoke_test': bool(args.smoke_test)}
            with (run.path / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if step % config['status_every'] == 0 or args.smoke_test:
                json_write(run.path / 'status.json', record)
            if step % config['save_every'] == 0:
                save('periodic')
            if args.smoke_test and args.smoke_stop_after and step >= args.smoke_stop_after:
                raise BudgetStop('Explicit smoke stop for resume verification')
        save('step_bound')
        pointer = json.loads((run.path / 'latest.json').read_text())
        json_write(run.path / 'completed.json', {'status': 'MAX_STEPS_REACHED', 'step': step,
                   'training_bound_only': True, 'quality_pass': False,
                   'smoke_test': bool(args.smoke_test), 'checkpoint_sha256': pointer['sha256']})
        return 0
    except BudgetStop as error:
        save(str(error))
        json_write(run.path / 'STOPPED.json', {'reason': str(error), 'step': step, 'quality_pass': False})
        return 3
    except BaseException as error:
        save('failure')
        json_write(run.path / 'FAILED.json', {'reason': repr(error), 'step': step, 'attempts': attempts})
        raise
    finally:
        run.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, help='Frozen prepared cache with COMPLETE.json')
    parser.add_argument('--run', required=True, help='Explicit new output directory; existing only with --resume')
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'acoustic_train.json'))
    parser.add_argument('--device', default='cuda')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true', help='Resume this script\'s own run and optimizer')
    mode.add_argument('--warm-start', help='Pure model-only Acoustic state_dict; creates a new run')
    parser.add_argument('--smoke-test', action='store_true', help='At most two batch-one interface updates')
    parser.add_argument('--smoke-stop-after', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=1, help='CPU smoke-test threads')
    args = parser.parse_args()
    if args.smoke_stop_after and not args.smoke_test:
        parser.error('--smoke-stop-after requires --smoke-test')
    if args.cpu_threads <= 0 or args.smoke_stop_after < 0:
        parser.error('Thread count must be positive and smoke stop nonnegative')
    raise SystemExit(train(args))
