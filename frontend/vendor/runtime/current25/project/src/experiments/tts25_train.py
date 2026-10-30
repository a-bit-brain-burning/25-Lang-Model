"""Scratch-only local training dispatcher preserving the 25 registered architectures.

The data publisher supplies numeric inputs and a hash-bound text frontend adapter.
No pre-existing checkpoint is opened by model construction or training. Checkpoint
loading is restricted to this run's recovery, validation-best and exported copies.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
import collections
import contextlib
import fcntl
import faulthandler
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback
import sys
import numpy as np
import torch
from torch.nn import functional as F

from ..common import sha256
from ..models.universal_acoustic import UniversalAcousticModel
from ..models.boundary_acoustic import BoundaryAcousticModel
from ..models.romance_routed_acoustic import RoutedRomanceAcousticModel
from ..models.phrase_prosody import PhraseProsodyPlanner
from ..models.english_word_planner import EnglishWordPlanner
from ..models.romance_word_planner import RomanceWordPlanner
from ..training.recording_epochs import RecordingEpochScheduler
from .eu_quality_spectrum_basis import PhysicalSpectrum
from .english_strict_train import strict_common
from .romance_train import atomic_checkpoint

SEED = 20261023
faulthandler.enable(all_threads=True)


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def construct(spec, role, means=None, scales=None, vocabulary=None):
    component = spec['components'][role]
    config = dict(component['config'])
    # Keep layer shapes, discard old training-data priors and identity bindings.
    if role == 'acoustic':
        config['voiced_prior'] = .5
        for stale in ('lexicon_sha256', 'frontend_semantic_sha256'):
            config.pop(stale, None)
    if role == 'acoustic':
        cls = {'routed39': RoutedRomanceAcousticModel, 'boundary31': BoundaryAcousticModel}.get(spec['family'], UniversalAcousticModel)
        model = cls(config, means, scales)
    elif spec['language'] == 'zh':
        model = PhraseProsodyPlanner(config, vocabulary)
    elif spec['language'] == 'en':
        model = EnglishWordPlanner(config)
    else:
        model = RomanceWordPlanner(config)
    actual = sum(p.numel() for p in model.parameters())
    if actual != component['parameters']:
        raise ValueError(f"Architecture parameter mismatch {spec['language']}/{role}: {actual}")
    return model


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def audit_data(root, spec, *, pilot=False):
    """Gate formal training on actual unique whole-recording WAV duration and SHAs."""
    import wave
    root = Path(root); manifest = root / 'manifest.jsonl'
    receipt = json.loads((root / 'completed.json').read_text())
    if receipt.get('status') != 'COMPLETED' or receipt.get('manifest_sha256') != sha256(manifest):
        raise ValueError('Data must be frozen and hash bound')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = set(); texts = {}; wavs = {}; groups = {}; counts = collections.Counter(); seconds = collections.Counter()
    for row in rows:
        split = row['split']; uid = row['id']
        if split not in ('train', 'validation') or uid in ids:
            raise ValueError('Only distinct train/validation records are permitted')
        ids.add(uid)
        normalized = ' '.join(row['text'].casefold().split())
        if normalized in texts:
            raise ValueError('Repeated normalized text cannot count towards the duration gate')
        texts[normalized] = split
        group = row.get('source_group')
        if group is None: raise ValueError('Every formal sample needs its source_group')
        if group in groups and groups[group] != split: raise ValueError('Source-group split leakage')
        groups[group] = split
        record = resolve(root, row['record_path'])
        if sha256(record) != row.get('sha256', row.get('record_sha256')):
            raise ValueError('NPZ identity changed: ' + uid)
        wav = resolve(root, row['wav'])
        digest = sha256(wav)
        if digest != row['wav_sha256']: raise ValueError('WAV identity changed: ' + uid)
        if digest in wavs: raise ValueError('Repeated recordings cannot count toward the duration gate')
        wavs[digest] = split
        with wave.open(str(wav), 'rb') as w:
            actual = w.getnframes() / w.getframerate()
            if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError('Student input must be mono16k PCM16')
        if abs(actual - float(row['seconds'])) > 1 / 16000 + 1e-8:
            raise ValueError('Claimed duration differs from whole recording')
        if actual <= 0: raise ValueError('Empty recording')
        for key, hash_key in (('textgrid', 'textgrid_sha256'),):
            if key in row and sha256(resolve(root, row[key])) != row[hash_key]:
                raise ValueError('Alignment identity changed')
        counts[split] += 1; seconds[split] += actual
    minimum = {'train': 0 if pilot else 18000, 'validation': 0 if pilot else 360}
    for split in minimum:
        if not counts[split] or seconds[split] + 1e-8 < minimum[split]:
            raise ValueError(f"{split} has {seconds[split]:.3f}s, requires {minimum[split]}s")
    strict_path = root / 'strict/completed.json'
    if not strict_path.exists(): raise ValueError('Train-fitted strict F0 overlay is required')
    strict = json.loads(strict_path.read_text())
    if strict.get('status') != 'COMPLETED' or strict.get('source_manifest_sha256') != sha256(manifest):
        raise ValueError('Strict F0 lineage differs')
    if strict.get('fit_split', strict.get('normalization_fit_split', 'train')) != 'train':
        raise ValueError('Strict F0 fitting must use train only')
    return rows, {'status': 'PASSED', 'pilot': pilot, 'counts': dict(counts), 'seconds': dict(seconds),
                  'manifest_sha256': sha256(manifest), 'strict_receipt_sha256': sha256(strict_path),
                  'durations_from': 'unique complete mono16k PCM16 recordings; validation excluded from train',
                  'source_group_disjoint': True, 'test_opened': False}


class NumericBank:
    """Bounded-memory numeric NPZ view; no frontend/language logic in the model."""
    def __init__(self, root, rows, spec, capacity=32):
        self.root = Path(root); self.rows = rows; self.spec = spec; self.cache = collections.OrderedDict(); self.capacity = capacity
        self.train = [r for r in rows if r['split'] == 'train']; self.validation = [r for r in rows if r['split'] == 'validation']
        self.strict = {}
        stats = self.root / 'strict/statistics.json'
        if stats.exists():
            self.strict = {r['id']: r for r in json.loads(stats.read_text())}
        self.plans = {}
        self.prepared_planner = {}

    def get(self, row):
        uid = row['id']
        if uid in self.cache: self.cache.move_to_end(uid); return self.cache[uid]
        with np.load(resolve(self.root, row['record_path']), allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        strict_path = self.root / 'strict' / (uid + '.npz')
        if strict_path.exists():
            digest = self.strict.get(uid, {}).get('sha256')
            if digest and sha256(strict_path) != digest: raise ValueError('Strict overlay identity changed')
            with np.load(strict_path, allow_pickle=False) as z: a.update({k: z[k] for k in z.files})
        a.update(self.prepared_planner.get(uid, {}))
        a['durations'] = np.asarray(a.get('durations', a.get('target_duration')), np.int64)
        n = len(a['durations']); c = self.spec['components']['acoustic']['config']; channels = c['channels']
        for key in ('phonetic_features', 'context_features', 'target_controls', 'control_mask'):
            if key not in a or not np.isfinite(a[key]).all(): raise ValueError('Missing/nonfinite numeric input ' + key)
        if a['phonetic_features'].shape != (n, c['phonetic_dim']) or a['target_controls'].shape != (n, 8, channels) or a['control_mask'].shape != a['target_controls'].shape:
            raise ValueError('Numeric architecture/target shape differs')
        extra = 8 if self.spec['language'] == 'zh' else 5 if 'planner' in self.spec['components'] else 0
        if a['context_features'].shape != (n, c['context_dim'] - extra): raise ValueError('Text context dimension differs')
        if not np.isfinite(a['durations']).all() or (a['durations'] < 0).any() or not a['durations'].sum(): raise ValueError('Invalid duration labels')
        frames = int(a['durations'].sum())
        if 'frame_voice' not in a:
            curve = a.get('curve', a.get('target_curve'))
            if curve is None: raise ValueError('Missing WORLD frame voicing labels')
            a['frame_voice'] = np.asarray(curve[:, -1], np.float32)
        a.setdefault('frame_voice_weight', np.ones(frames, np.float32))
        if a['frame_voice'].shape != (frames,) or a['frame_voice_weight'].shape != (frames,): raise ValueError('Frame topology mismatch')
        if not np.isfinite(a['frame_voice']).all() or not np.isfinite(a['frame_voice_weight']).all() or ((a['frame_voice'] < 0) | (a['frame_voice'] > 1)).any() or ((a['frame_voice_weight'] < 0) | (a['frame_voice_weight'] > 1)).any(): raise ValueError('Invalid frame VUV/confidence')
        if ((a['control_mask'] < 0) | (a['control_mask'] > 1)).any(): raise ValueError('Invalid confidence masks')
        self.cache[uid] = a
        while len(self.cache) > self.capacity: self.cache.popitem(last=False)
        return a

    def normalization(self, progress=None):
        channels = self.spec['output']['neural_channels']; total = np.zeros(channels); square = total.copy(); count = total.copy()
        for index,row in enumerate(self.train):
            a = self.get(row); t = a['target_controls'].astype(np.float64); m = a['control_mask'].astype(np.float64)
            total += (t*m).sum((0, 1)); square += (t*t*m).sum((0, 1)); count += m.sum((0, 1))
            if progress and (index%128==0 or index+1==len(self.train)):progress(index+1,len(self.train))
        mean = total / np.maximum(count, 1); scale = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean**2, 0))
        scale = np.maximum(scale, np.r_[.1, .5, np.full(channels-3, .05), 1.]); mean[-1] = 0; scale[-1] = 1
        return mean.astype(np.float32), scale.astype(np.float32)

    def acoustic_batch(self, rows, device):
        arrays = [self.get(row) for row in rows]; width = max(len(a['durations']) for a in arrays)
        b = len(rows); c = self.spec['components']['acoustic']['config']; d = c['channels']
        f = {'phonetic_features': torch.zeros(b, width, c['phonetic_dim']), 'context_features': torch.zeros(b, width, c['context_dim']),
             'valid_mask': torch.zeros(b, width, dtype=torch.bool), 'edge_sil_mask': torch.zeros(b, width, dtype=torch.bool)}
        target = torch.zeros(b, width, 8, d); mask = torch.zeros_like(target); duration = torch.zeros(b, width)
        ix = []; fractions = []; voice = []; confidence = []
        for j, (row, a) in enumerate(zip(rows, arrays)):
            n = len(a['durations']); f['phonetic_features'][j, :n] = torch.from_numpy(a['phonetic_features'])
            ctx = a['context_features']
            if 'planner' in self.spec['components']:
                plan = self.plans[row['id']]; owner = np.asarray(a['planner_owner'], np.int64)
                unit_plan = np.zeros((n, plan.shape[-1]), np.float32); good = owner >= 0
                if np.any(owner[good] >= len(plan)): raise ValueError('Planner ownership is outside words/syllables')
                unit_plan[good] = plan[owner[good]]; ctx = np.column_stack((ctx, unit_plan))
            f['context_features'][j, :n] = torch.from_numpy(ctx)
            f['valid_mask'][j, :n] = True
            f['edge_sil_mask'][j, :n] = torch.from_numpy(a.get('edge_sil_mask', np.r_[True, np.zeros(max(0, n-2), bool), True]))
            target[j, :n] = torch.from_numpy(a['target_controls']); mask[j, :n] = torch.from_numpy(a['control_mask']); duration[j, :n] = torch.from_numpy(a['durations'].astype(np.float32))
            ix.append(np.repeat(np.arange(n) + j*width, a['durations']))
            fractions.extend([np.arange(k, dtype=np.float32)/k for k in a['durations'] if k])
            voice.append(a['frame_voice']); confidence.append(a['frame_voice_weight'])
        f['_frame_indices'] = torch.from_numpy(np.concatenate(ix)).long(); f['_frame_fractions'] = torch.from_numpy(np.concatenate(fractions))
        f['_frame_voice'] = torch.from_numpy(np.concatenate(voice).astype(np.float32)); f['_frame_voice_weight'] = torch.from_numpy(np.concatenate(confidence).astype(np.float32))
        return {k: v.to(device) for k, v in f.items()}, target.to(device), mask.to(device), duration.to(device)

    def planner_batch(self, rows, device):
        arrays = [self.get(r) for r in rows]; width = max(len(a['planner_numeric']) for a in arrays)
        dim = self.spec['components']['planner']['config'].get('feature_dim', 66); plan_dim = 8 if self.spec['language'] == 'zh' else 5
        f = {'numeric': torch.zeros(len(rows), width, dim), 'boundary': torch.zeros(len(rows), width), 'valid_mask': torch.zeros(len(rows), width, dtype=torch.bool)}
        if plan_dim == 8: f['char_ids'] = torch.zeros(len(rows), width, dtype=torch.long)
        target = torch.zeros(len(rows), width, plan_dim); mask = torch.zeros_like(target)
        for j, a in enumerate(arrays):
            n = len(a['planner_numeric']); f['numeric'][j, :n] = torch.from_numpy(a['planner_numeric']); f['valid_mask'][j, :n] = True
            f['boundary'][j, :n] = torch.from_numpy(a.get('planner_boundary', np.zeros(n, np.float32)))
            if plan_dim == 8: f['char_ids'][j, :n] = torch.from_numpy(a['planner_char_ids']).long()
            target[j, :n] = torch.from_numpy(a['planner_target']); mask[j, :n] = torch.from_numpy(a['planner_mask'])
        return {k: v.to(device) for k, v in f.items()}, target.to(device), mask.to(device)


def acoustic_objective(model, packed, decoder, spec, all_physical=False):
    f, target, mask, duration = packed
    if spec['family'] in ('routed39','english39'):
        # Preserve the validated39D physical/clarity/pause objective using numeric
        # articulatory groups only; no word or language policy enters the model.
        from ..training.english39_objective import objective as routed_objective
        from .romance_train import WEIGHTS
        valid=f['valid_mask'];phone=f['phonetic_features'];lexical=valid & (phone[...,0]<.5)
        vowel=lexical & (phone[...,1]>.5);obstruent=lexical & (phone[...,2:5].sum(-1)>.5)
        pause=valid & ~lexical
        if pause.shape[1]:pause[:,0]=False;pause[torch.arange(len(pause),device=pause.device),valid.sum(-1)-1]=False
        f['_internal_pause_mask']=pause
        left=[];right=[];width=valid.shape[1]
        for i in range(len(valid)):
            positions=vowel[i].nonzero().flatten()+i*width
            left.append(positions[:-1]);right.append(positions[1:])
        context=f['context_features']
        groups={'vowel':vowel,'obstruent':obstruent,'lexical':lexical,
                'stressed_vowel':vowel & (context[...,12]>=.75) & (context[...,13]>.5),
                'final_obstruent':obstruent & (context[...,1]>.5),
                'duration_weight':torch.where(obstruent,1.5,1.),
                'pair_left':torch.cat(left),'pair_right':torch.cat(right)}
        loss,metrics=routed_objective(model,(f,target,mask,duration,groups),decoder,{'physical_samples':768,'weights':WEIGHTS},all_physical)
        return loss,{**metrics,'total':loss}
    output = model({k: f[k] for k in ('phonetic_features', 'context_features', 'valid_mask', 'edge_sil_mask')})
    common, metrics = strict_common(output, target, mask, duration, f, model.scales, decoder, 10**8 if all_physical else 768, model.voiced_log_prior)
    from ..training.clarity_objective import frame_logits
    logits = frame_logits(output['controls'], f)
    weight = f['_frame_voice_weight']
    bce = F.binary_cross_entropy_with_logits(logits, f['_frame_voice'], reduction='none')
    frame_vuv = (bce*weight).sum()/weight.sum().clamp_min(1)
    # Continuous-voicing families retain the original full-frame BCE policy.
    if spec['components']['acoustic']['config'].get('continuous_voicing'):
        common = common - .7*metrics['vuv_balanced_bce'] + .7*frame_vuv
    else:
        common = common + .15*frame_vuv
    return common, {**metrics, 'frame_vuv_bce': frame_vuv, 'total': common}


def planner_objective(output, target, mask):
    if target.shape[-1] == 5:
        from ..models.romance_word_planner import objective
        return objective(output, target, mask.bool())
    boundary_mask = mask[..., :4].amin(-1)
    ce = -(target[..., :4]*output['boundary_logits'].log_softmax(-1)).sum(-1)
    boundary = (ce*boundary_mask).sum()/boundary_mask.sum().clamp_min(1)
    cont_mask = mask[..., 4:]
    cont = F.smooth_l1_loss(output['plan'][..., 4:], target[..., 4:], reduction='none')
    return boundary + (cont*cont_mask).sum()/cont_mask.sum().clamp_min(1)


@contextlib.contextmanager
def gpu_lock(run):
    path = Path(run)/'operations/local_gpu.lock'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def frontend_session(adapter, language, metadata, progress=None):
    """Reuse one unchanged frozen dictionary frontend, with explicit lineage.

    The frozen adapter constructs/loads a 35MB dictionary on every sentence.
    This execution session calls that adapter's same frontend factory and
    numeric encoder once per text, while retaining the dictionary instance.
    No frozen adapter/source/metadata is changed or monkeypatched.
    """
    if not (hasattr(adapter,'RomanceFrontend') and hasattr(adapter,'encode_frontend')):
        return adapter
    frontend=adapter.RomanceFrontend(language,Path(metadata['lexicon_path']))
    if metadata.get('lexicon_sha256') and frontend.lexicon_sha256!=metadata['lexicon_sha256']:raise ValueError('Cached frontend lexicon identity changed')
    if metadata.get('frontend_signature') and frontend.signature!=metadata['frontend_signature']:raise ValueError('Cached frontend identity changed')
    class Session:
        calls=0
        def encode(self, selected_language, text, selected_metadata):
            if selected_language!=language or selected_metadata.get('frontend_signature')!=metadata.get('frontend_signature') or selected_metadata.get('lexicon_path')!=metadata.get('lexicon_path'):raise ValueError('Frontend session metadata changed')
            f=frontend.encode(text);p=adapter.encode_frontend(f)
            result={**f,'planner_numeric':p['numeric'],'planner_boundary':p['boundary'],'planner_owner':f['plan_owner']}
            self.calls+=1
            if progress and (self.calls==1 or self.calls%128==0):progress(self.calls)
            return result
    return Session()


def prepare_romance_planner(bank, run, language, manifest_sha256, progress=None):
    """Use identical train-only numeric wp functions without eager full-corpus Bank.

    NumericBank caches at most32 decompressed records. The preparation stage
    retains only small per-word measurements and numeric frontend metadata;
    full curves/controls are released through the bounded source view.
    """
    metadata=json.loads((bank.root/'frontend_metadata.json').read_text())
    adapter=frontend_session(load_adapter(resolve(bank.root,metadata['adapter_path']),metadata['adapter_sha256']),language,metadata,progress)
    preparation_path=Path(run)/'training_tools/planner_prepare.py'
    preparation=load_adapter(preparation_path,sha256(preparation_path))
    result=preparation.prepare(bank,adapter,metadata,language,manifest_sha256)
    result['preparation_source_sha256']=sha256(preparation_path)
    result['host_record_cache_capacity']=bank.capacity
    result['full_corpus_WAVE_or_curve_cache']=False
    return result


def train_ready_signature(data):
    """A raw completed marker is insufficient; the final barrier binds strict labels."""
    data=Path(data);path=data/'TRAIN_READY.json'
    if not path.exists():return None
    ready=json.loads(path.read_text())
    if ready.get('status') not in ('READY','COMPLETED','TRAIN_READY'):raise ValueError('Formal TRAIN_READY barrier is not complete')
    if ready.get('manifest_sha256')!=sha256(data/'manifest.jsonl'):raise ValueError('TRAIN_READY manifest binding differs')
    strict_digest=ready.get('strict_receipt_sha256',ready.get('strict_completed_sha256'))
    if strict_digest!=sha256(data/'strict/completed.json'):raise ValueError('TRAIN_READY strict binding differs')
    return hashlib.sha256(path.read_bytes()+(data/'completed.json').read_bytes()).hexdigest()


def train_numeric(run, language, *, pilot=False, batch=24, planner_steps=3000, acoustic_steps=8000, resume=False):
    run = Path(run).resolve(); registry_path = run/'architecture/registry.json'
    registry = json.loads(registry_path.read_text()); spec = registry['models'][language]
    out = run/'training'/language; out.mkdir(parents=True, exist_ok=True)
    def preparation_status(stage,**details):
        status={'status':'PREPARING','stage':stage,'language':language,'pid':os.getpid(),'time':time.time(),'python':sys.executable,'python_version':sys.version,'trainer_sha256':sha256(__file__),**details}
        save_json(out/'preparation_status.json',status)
        with (out/'preparation.jsonl').open('a') as log:log.write(json.dumps(status)+'\n')
        print(json.dumps(status),flush=True)
    preparation_status('AUDITING_FROZEN_DATA')
    for source,digest in registry.get('model_sources',{}).items():
        if sha256(source)!=digest:raise ValueError('Registered model source changed: '+source)
    if spec.get('parent') is not None or not spec.get('scratch') or spec['seed'] != SEED: raise ValueError('Scratch-only architecture policy violated')
    if not pilot and (planner_steps < 3000 or acoustic_steps < 8000): raise ValueError('Formal update floor violated')
    if not pilot and train_ready_signature(run/'data'/language) is None:
        raise ValueError('Wait for the final TRAIN_READY publication barrier')
    rows, audit = audit_data(run/'data'/language, spec, pilot=pilot)
    if not pilot:
        reference = json.loads((run/'references'/language/'metadata.json').read_text())
        for row in rows:
            if row.get('teacher_revision') != 'c5fdb5ccb189668d56333f77ba2629f4cd7535f4' or row.get('num_step') != 32:
                raise ValueError('Formal recordings must bind the authorized official teacher revision and32 steps')
            if row.get('voice_sha256') != reference['reference_sha256']:
                raise ValueError('Formal recordings must use this language user reference')
            if not row.get('ASR_checked') or row.get('assessment', {}).get('status') != 'ACCEPTED':
                raise ValueError('Formal recordings must pass per-record ASR screening')
    bank = NumericBank(run/'data'/language, rows, spec)
    save_json(out/'data_audit.json', audit)
    preparation_status('TRAIN_ONLY_NORMALIZATION',records=len(bank.train))
    means, scales = bank.normalization(lambda current,total:preparation_status('TRAIN_ONLY_NORMALIZATION',records=current,total_records=total))
    preparation_status('NORMALIZATION_FINISHED')
    metadata_path = bank.root/'planner_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if language in ('es', 'fr', 'it', 'pt', 'ro') and not metadata:
        preparation_status('FRESH_PLANNER_TARGETS',records=len(rows),dictionary_instance_reused=True)
        metadata = prepare_romance_planner(bank, run, language, audit['manifest_sha256'],lambda n:preparation_status('FRESH_PLANNER_TARGETS',records=n,total_records=len(rows),dictionary_instance_reused=True))
        save_json(out/'planner_metadata.json', metadata)
    if 'planner' in spec['components'] and not metadata:
        frontend_metadata = json.loads((bank.root/'frontend_metadata.json').read_text())
        adapter = load_adapter(resolve(bank.root,frontend_metadata['adapter_path']),frontend_metadata['adapter_sha256'])
        preparation_path=run/'training_tools/planner_prepare.py'
        preparation=load_adapter(preparation_path,sha256(preparation_path))
        metadata=preparation.prepare(bank,adapter,frontend_metadata,language,audit['manifest_sha256'])
        metadata['preparation_source_sha256']=sha256(preparation_path)
        save_json(out/'planner_metadata.json',metadata)
    if 'planner' in spec['components']:
        preparation_status('VERIFYING_PLANNER_NUMERIC_TARGETS')
        if metadata.get('fit_split') != 'train' or metadata.get('source_manifest_sha256') != audit['manifest_sha256']:
            raise ValueError('Planner labels must have this manifest and train-only statistics')
        for row in rows:
            a = bank.get(row)
            for key in ('planner_numeric', 'planner_target', 'planner_mask', 'planner_owner'):
                if key not in a: raise ValueError('Missing prepared numeric planner supervision: '+key)
    frontend_meta_path = bank.root/'frontend_metadata.json'
    if not pilot:
        if not frontend_meta_path.exists(): raise ValueError('A hash-bound ordinary-text frontend is required before formal training')
        fm = {**json.loads(frontend_meta_path.read_text()), **{k:metadata[k] for k in ('vocabulary',) if k in metadata}}
        adapter = frontend_session(load_adapter(resolve(bank.root, fm['adapter_path']), fm['adapter_sha256']),language,fm)
        preparation_status('VERIFYING_PLAIN_TEXT_FRONTEND',records=len(rows),dictionary_instance_reused=hasattr(adapter,'calls'))
        for index,row in enumerate(rows):
            encoded = adapter.encode(language, row['text'], fm)
            recorded = bank.get(row)
            for key in ('phonetic_features', 'context_features'):
                if not np.array_equal(encoded[key], recorded[key]): raise ValueError('Formal extraction and ordinary-text frontend differ: '+row['id']+'/'+key)
            if index%128==0 or index+1==len(rows):preparation_status('VERIFYING_PLAIN_TEXT_FRONTEND',records=index+1,total_records=len(rows))
    preparation_status('CPU_PREPARATION_FINISHED_WAITING_FOR_GPU_LOCK')
    lineage = {'initialization': 'random_scratch', 'parent': None, 'parent_checkpoint_read': False, 'seed': SEED,
               'registry_sha256': sha256(registry_path), 'manifest_sha256': audit['manifest_sha256'],
               'strict_receipt_sha256': audit['strict_receipt_sha256'], 'trainer_sha256': sha256(__file__),
               'language': language, 'test_opened': False, 'normalization_fit_split': 'train',
               'frontend_metadata_sha256':sha256(frontend_meta_path) if frontend_meta_path.exists() else None,
               'planner_preparation_sha256':metadata.get('preparation_source_sha256'),
               'fresh_normalization_sha256':hashlib.sha256(means.tobytes()+scales.tobytes()).hexdigest()}
    lineage['frontend_execution']='same frozen frontend factory and numeric encoder; one explicitly cached dictionary instance per preparation pass'
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    if not (snapshot/'tts25_train.py').exists():shutil.copy2(__file__,snapshot/'tts25_train.py')
    for source in registry.get('model_sources',{}):
        destination=snapshot/'models'/Path(source).name;destination.parent.mkdir(exist_ok=True)
        if not destination.exists():shutil.copy2(source,destination)
    torch.set_num_threads(1); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    with gpu_lock(run):
        preparation_status('GPU_LOCK_ACQUIRED')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; no silent CPU fallback')
        device = 'cuda'; torch.cuda.manual_seed_all(SEED)
        for phase in (['planner', 'acoustic'] if 'planner' in spec['components'] else ['acoustic']):
            folder = out/phase; folder.mkdir(exist_ok=True)
            if (folder/'completed.json').exists():
                if phase == 'planner': _publish_plans(bank, _load_own(folder/'best.pt', spec, phase, metadata).to(device), device)
                continue
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            model = construct(spec, phase, means, scales, metadata.get('vocabulary')).to(device)
            minimum = planner_steps if phase == 'planner' else acoustic_steps
            steps_per_epoch = math.ceil(len(bank.train)/batch); epochs = math.ceil(minimum/steps_per_epoch)
            schedule = RecordingEpochScheduler(len(bank.train), batch, SEED, epochs)
            config = {'seed': SEED, 'parent': None, 'initialization': 'random_scratch', 'phase': phase, 'language': language,
                      'batch': batch, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch, 'minimum_updates': minimum,
                      'steps': schedule.total_steps, 'sample_exposures': schedule.total_exposures,
                      'complete_recording_epochs': True, 'sampling_order_sha256': schedule.order_sha256,
                      'learning_rate': .001 if phase == 'planner' else .0006, 'weight_decay': .0001,
                      'device': device, 'precision': 'FP32', 'pilot': pilot, 'validation_interval_steps': max(steps_per_epoch, math.ceil(500/steps_per_epoch)*steps_per_epoch),
                      'selection': 'minimum validation total loss', 'normalization_fit_split': 'train', 'test': 'NOT_RUN',
                      'objective': 'strict physical spectrum + duration + F0/AP + frame/control VUV; no architecture change'}
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.0001)
            decoder = PhysicalSpectrum(spec['components']['acoustic']['config']['spectral_dim'], None if spec['family']=='boundary31' else 700).to(device)
            start = 0; last = 0; best = float('inf'); best_step = 0
            if (folder/'initial.pt').exists() and not resume: raise FileExistsError('Existing training requires --resume')
            if resume:
                states = [torch.load(folder/n, map_location=device, weights_only=True) for n in ('latest.pt', 'interrupted.pt') if (folder/n).exists()]
                if states:
                    state = max(states, key=lambda x:x['step'])
                    if state['training_config'] != config or state['lineage'] != lineage: raise ValueError('Recovery identity changed')
                    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
                    start = last = state['step']; best = state['best_score']; best_step = state['best_step']
                    torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']]); random.setstate(state['python_rng'])
            save_json(folder/'config.json', config); save_json(folder/'lineage.json', lineage)
            def checkpoint(name, step, metrics):
                payload = {'format': spec['components'][phase]['format'], 'config': model.config, 'model': model.state_dict(),
                           'optimizer': optimizer.state_dict(), 'step': step, 'training_config': config, 'lineage': lineage,
                           'best_score': best, 'best_step': best_step, 'metrics': metrics, 'normalization': metadata.get('normalization', {}),
                           'vocabulary': metadata.get('vocabulary', {}), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate()}
                if phase=='planner' and language=='en':
                    from ..frontend.english_word_prosody import PLAN_VERSION
                    payload['plan_version']=PLAN_VERSION
                atomic_checkpoint(folder/name, payload)
            def loss_for(items, all_physical=False):
                if phase == 'planner':
                    f, target, mask = bank.planner_batch(items, device)
                    return planner_objective(model(f), target, mask), {}
                return acoustic_objective(model, bank.acoustic_batch(items, device), decoder, spec, all_physical)
            @torch.inference_mode()
            def validate():
                model.eval(); total=0.
                for first in range(0,len(bank.validation),batch):
                    items=bank.validation[first:first+batch];loss,_=loss_for(items,True);total+=float(loss)*len(items)
                return {'loss':total/len(bank.validation),'recordings':len(bank.validation),'split':'validation','reference_duration_is_training_supervision_only':True}
            if start == 0:
                metrics=validate();best=metrics['loss'];checkpoint('initial.pt',0,metrics);checkpoint('best.pt',0,metrics)
                save_json(folder/'initialization.json', {**lineage, 'step':0,'tensor_sha256':fingerprint(model), 'checkpoint_sha256':sha256(folder/'initial.pt'), 'parameters':sum(p.numel() for p in model.parameters())})
            begun=time.perf_counter(); updates=0; torch.cuda.reset_peak_memory_stats()
            try:
                with (folder/'training.jsonl').open('a') as log:
                    for step,indices in enumerate(schedule.batches(start),start+1):
                        model.train(); optimizer.zero_grad(set_to_none=True)
                        loss,_=loss_for([bank.train[i] for i in indices])
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite training gradient')
                        if any(p.device.type!='cuda' for p in model.parameters()) or any(p.grad.device.type!='cuda' for p in model.parameters() if p.grad is not None):raise AssertionError('Actual parameters/gradients must be CUDA')
                        optimizer.step();last=step;updates+=1
                        for group in optimizer.param_groups:group['lr']=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*step/schedule.total_steps)))
                        if step==1 or step%100==0:
                            status={'status':'RUNNING','language':language,'phase':phase,'step':step,'steps':schedule.total_steps,'loss':float(loss.detach()),'seconds':time.perf_counter()-begun,'device':'cuda','actual_optimizer_updates_this_process':updates}
                            save_json(folder/'status.json',status);log.write(json.dumps(status)+'\n');log.flush();print(json.dumps(status),flush=True)
                        if step%config['validation_interval_steps']==0 or step==schedule.total_steps:
                            metrics=validate()
                            improved=metrics['loss']<best
                            if improved:best=metrics['loss'];best_step=step
                            checkpoint('latest.pt',step,metrics)
                            if improved:checkpoint('best.pt',step,metrics)
                            save_json(folder/f'validation_{step:06d}.json',metrics)
                result={'status':'COMPLETED','phase':phase,'steps':last,'best_step':best_step,'best_score':best,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-begun,'checkpoint_sha256':sha256(folder/'best.pt'),'CUDA':{'GPU':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'runtime':torch.version.cuda,'model_parameter_devices':sorted({str(p.device) for p in model.parameters()}),'actual_parameter_gradient_cuda_assertion':'all optimizer updates','optimizer_updates_this_process':updates,'peak_allocated_bytes':torch.cuda.max_memory_allocated()},'lineage':lineage}
                save_json(folder/'completed.json',result);save_json(folder/'status.json',result)
            except BaseException:
                checkpoint('interrupted.pt',last,{})
                save_json(folder/'status.json',{'status':'FAILED','step':last,'traceback':traceback.format_exc(),'recoverable':True});raise
            if phase=='planner': _publish_plans(bank,_load_own(folder/'best.pt',spec,phase,metadata).to(device),device)
            del model,optimizer,decoder;torch.cuda.empty_cache()
    save_json(out/'completed.json',{'status':'COMPLETED','language':language,'parameters':spec['parameters'],'lineage':lineage,'pilot':pilot,'export':'PENDING'})
    return out


def _load_own(path,spec,role,metadata=None):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['lineage']['initialization']!='random_scratch' or state['lineage']['parent'] is not None:raise ValueError('Not a scratch checkpoint from this run')
    if state['lineage'].get('language')!=spec['language']:raise ValueError('Checkpoint language differs')
    model=construct(spec,role,state['model'].get('means'),state['model'].get('scales'),(metadata or {}).get('vocabulary',state.get('vocabulary')))
    model.load_state_dict(state['model']);model.eval();return model


@torch.inference_mode()
def _publish_plans(bank,model,device):
    model.eval()
    for row in bank.rows:
        features,_,_=bank.planner_batch([row],device)
        bank.plans[row['id']]=model(features)['plan'][0].cpu().numpy().astype(np.float32)


def load_adapter(path,expected_sha256):
    path=Path(path)
    if sha256(path)!=expected_sha256:raise ValueError('Frontend adapter source changed')
    spec=importlib.util.spec_from_file_location('tts25_numeric_frontend_'+expected_sha256[:12],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _text_predict(spec,acoustic,planner,adapter,text,metadata):
    # This call accepts text plus language; no sample identity/audio/reference can enter.
    encoded=adapter.encode(spec['language'],text,metadata)
    features={k:encoded[k] for k in ('phonetic_features','context_features','edge_sil_mask') if k in encoded}
    if planner is not None:
        if 'planner_numeric' not in encoded and spec['family']=='routed39':
            from ..models.romance_word_planner import encode_frontend
            numeric=encode_frontend(encoded)
            encoded.update(planner_numeric=numeric['numeric'],planner_boundary=numeric['boundary'],planner_owner=numeric['plan_owner'])
        f={'numeric':torch.from_numpy(encoded['planner_numeric']).float()}
        for name in ('boundary','char_ids'):
            key='planner_'+name
            if key in encoded:f[name]=torch.from_numpy(encoded[key]).long() if name=='char_ids' else torch.from_numpy(encoded[key]).float()
        if 'boundary' not in f:f['boundary']=torch.zeros(len(f['numeric']))
        plan=planner(f)['plan'].detach().cpu().numpy()
        owner=np.asarray(encoded['planner_owner'],np.int64);unit=np.zeros((len(owner),plan.shape[-1]),np.float32);good=owner>=0;unit[good]=plan[owner[good]]
        features['context_features']=np.column_stack((features['context_features'],unit)).astype(np.float32)
    prediction=acoustic.predict_features(features)
    if spec['family']=='routed39':
        from .romance_routed_delivery import quiet_declared_pauses
        prediction=quiet_declared_pauses(encoded,prediction)
    elif spec['language']=='zh':
        from ..frontend.chinese_delivery_policy import numeric_duration_policy
        from .continuity_representation import uniform_postprocess
        from ..models.universal_acoustic import expand_controls
        policy=numeric_duration_policy(encoded,prediction['durations'],kinship_max_ratio=1.)
        prediction['durations']=policy['target_durations']
        raw=expand_controls(prediction['controls'],prediction['durations'])
        prediction['curve'],prediction['delivery_audit']=uniform_postprocess(raw,prediction['durations'],encoded,policy['comma_mask'],-30.)
    return prediction


def export_and_verify(run,language):
    """Freeze inference-only weights; independently load and compare ordinary-text PCM."""
    from ..models.english39_acoustic import render_curve
    from ..audio.dsp import write_wav
    run=Path(run).resolve();spec=json.loads((run/'architecture/registry.json').read_text())['models'][language]
    training=run/'training'/language;data=run/'data'/language
    metadata=json.loads((data/'frontend_metadata.json').read_text())
    planner_metadata_path=training/'planner_metadata.json'
    if planner_metadata_path.exists():
        prepared=json.loads(planner_metadata_path.read_text())
        metadata.update({k:prepared[k] for k in ('vocabulary','normalization','baselines') if k in prepared})
    adapter_path=resolve(data,metadata['adapter_path']);adapter=load_adapter(adapter_path,metadata['adapter_sha256'])
    out=run/'exports'/language
    if (out/'completed.json').exists():return out
    out.mkdir(parents=True,exist_ok=True)
    acoustic=_load_own(training/'acoustic/best.pt',spec,'acoustic',metadata)
    planner=_load_own(training/'planner/best.pt',spec,'planner',metadata) if 'planner' in spec['components'] else None
    for role,model in [('acoustic',acoustic),('planner',planner)]:
        if model is None:continue
        original=torch.load(training/role/'best.pt',map_location='cpu',weights_only=True)
        atomic_checkpoint(out/(role+'.pt'),{k:original[k] for k in ('format','config','model','step','lineage','normalization','vocabulary') if k in original})
    # Preserve the adapter's relative frontend/runtime/training_tools layout.
    adapter_export=out/'frontend/adapter.py';adapter_export.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(adapter_path,adapter_export)
    for sibling in adapter_path.parent.glob('adapter*.py'):
        if sibling.name!='adapter.py':shutil.copy2(sibling,adapter_export.parent/sibling.name)
    if (run/'frontend'/language).exists():shutil.copytree(run/'frontend'/language,out/'frontend'/language,dirs_exist_ok=True)
    if language in ('zh','en'):
        (out/'training_tools').mkdir(exist_ok=True)
        shutil.copy2(run/'training_tools/zh_en_frontend_adapter.py',out/'training_tools/zh_en_frontend_adapter.py')
    exported_metadata=dict(metadata)
    if metadata.get('lexicon_path'):
        lexicon_copy=out/'frontend'/language/Path(metadata['lexicon_path']).name;lexicon_copy.parent.mkdir(parents=True,exist_ok=True)
        if not lexicon_copy.exists():shutil.copy2(metadata['lexicon_path'],lexicon_copy)
        elif sha256(lexicon_copy)!=sha256(metadata['lexicon_path']):raise ValueError('Copied export lexicon identity differs')
        exported_metadata['lexicon_path']=str(lexicon_copy)
    exported_metadata.update(adapter_path=str(adapter_export),adapter_sha256=sha256(adapter_export))
    save_json(out/'frontend_metadata.json',exported_metadata);save_json(out/'architecture.json',spec)
    exported_acoustic=_load_own(out/'acoustic.pt',spec,'acoustic',metadata)
    exported_planner=_load_own(out/'planner.pt',spec,'planner',metadata) if planner is not None else None
    exported_adapter=load_adapter(adapter_export,metadata['adapter_sha256'])
    rows=[json.loads(line) for line in (data/'manifest.jsonl').read_text().splitlines() if line.strip()]
    candidates=sorted((r for r in rows if r['split']=='validation'),key=lambda r:float(r['seconds']))
    selected=[candidates[int(round(q*(len(candidates)-1)))] for q in (0,.2,.4,.6,.8,1)]
    new_texts=metadata.get('novel_texts',[])
    if len(new_texts)<2:raise ValueError('Need two novel ordinary-text examples per language')
    cases=[{'id':'validation_'+str(i),'text':r['text'],'source_seconds':r['seconds']} for i,r in enumerate(selected)]+[{'id':'novel_'+str(i),'text':t} for i,t in enumerate(new_texts[:2])]
    evidence=[]
    for case in cases:
        with torch.inference_mode():
            p=_text_predict(spec,acoustic,planner,adapter,case['text'],metadata)
            q=_text_predict(spec,exported_acoustic,exported_planner,exported_adapter,case['text'],exported_metadata)
        mode='linear31' if spec['family']=='boundary31' else 'warped39'
        # render_curve supports explicit modes; never infer basis from channel count.
        pcm,stats=render_curve(p['curve'],mode);export_pcm,_=render_curve(q['curve'],mode)
        if not np.array_equal(pcm,export_pcm):raise AssertionError('Training/export PCM mismatch')
        path=out/'listening'/(case['id']+'.wav');path.parent.mkdir(exist_ok=True);write_wav(path,pcm)
        evidence.append({**case,'wav':str(path),'sha256':sha256(path),'pcm_exact':True,'pcm_samples':len(pcm),'seconds':len(pcm)/16000,'reference_inputs_used':False,'clipped_fraction':float(np.mean(np.abs(pcm.astype(np.int32))>=32767)),'DSP':stats})
    save_json(out/'acceptance.json',{'status':'PASSED','samples':evidence,'PCM_comparison':'all signed PCM16 samples exact','ordinary_text_reference_reads':False,'MOS':'NOT_MEASURED'})
    save_json(out/'completed.json',{'status':'COMPLETED','language':language,'parameters':spec['parameters'],'samples':len(evidence),'acoustic_sha256':sha256(out/'acoustic.pt'),'planner_sha256':sha256(out/'planner.pt') if planner is not None else None,'automatic_release':False})
    return out


def infer_bundle(bundle,text,output):
    """Ordinary text API: export bundle + text only; write all inference locally."""
    from ..models.english39_acoustic import render_curve
    from ..audio.dsp import write_wav
    bundle=Path(bundle).resolve();output=Path(output)
    spec=json.loads((bundle/'architecture.json').read_text())
    metadata=json.loads((bundle/'frontend_metadata.json').read_text())
    adapter=load_adapter(bundle/'frontend/adapter.py',metadata['adapter_sha256'])
    acoustic=_load_own(bundle/'acoustic.pt',spec,'acoustic',metadata)
    planner=_load_own(bundle/'planner.pt',spec,'planner',metadata) if 'planner' in spec['components'] else None
    with torch.inference_mode():prediction=_text_predict(spec,acoustic,planner,adapter,text,metadata)
    pcm,stats=render_curve(prediction['curve'],'linear31' if spec['family']=='boundary31' else 'warped39')
    output.parent.mkdir(parents=True,exist_ok=True);write_wav(output,pcm)
    save_json(output.with_suffix('.json'),{'text':text,'language':spec['language'],'reference_audio_used':False,'reference_durations_used':False,'reference_pronunciation_used':False,'seconds':len(pcm)/16000,'wav_sha256':sha256(output),'parameters':spec['parameters'],'DSP':stats,'duration_frames':prediction['durations'].tolist()})
    return pcm


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('audit','train','export','infer'))
    p.add_argument('--run',type=Path,required=True);p.add_argument('--language',required=True);p.add_argument('--pilot',action='store_true')
    p.add_argument('--batch',type=int,default=24);p.add_argument('--planner-steps',type=int,default=3000);p.add_argument('--acoustic-steps',type=int,default=8000);p.add_argument('--resume',action='store_true');p.add_argument('--text');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.command=='audit':
        spec=json.loads((a.run/'architecture/registry.json').read_text())['models'][a.language]
        _,result=audit_data(a.run/'data'/a.language,spec,pilot=a.pilot);print(json.dumps(result))
    elif a.command=='train':train_numeric(a.run,a.language,pilot=a.pilot,batch=a.batch,planner_steps=a.planner_steps,acoustic_steps=a.acoustic_steps,resume=a.resume)
    elif a.command=='export':export_and_verify(a.run,a.language)
    else:
        if a.text is None or a.output is None:p.error('infer requires --text and --output')
        infer_bundle(a.run/'exports'/a.language,a.text,a.output)


if __name__=='__main__':main()
