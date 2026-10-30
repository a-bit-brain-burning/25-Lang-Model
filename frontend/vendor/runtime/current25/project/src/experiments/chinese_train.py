"""Chinese paired training with the existing language-neutral numeric model.

No phone IDs or language symbols enter the model. Normalization is computed
from this corpus train partition only; validation selects checkpoints.
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

from ..models.universal_acoustic import FORMAT, UniversalAcousticModel as TinyAcousticModel, load_acoustic, load_acoustic_state, resource_estimate

TEXT_INPUT_KEYS = ('phonetic_features', 'context_features', 'valid_mask', 'edge_sil_mask')
TRAIN_FIELDS = ('phonetic_features', 'context_features', 'target_controls', 'control_mask',
                'target_duration', 'target_curve', 'edge_sil_mask', 'phone_ids', 'stress')
D=31
SPEC=24
BANDS=5
CONTINUOUS=False
from collections import OrderedDict

class NumericRecords:
    def __init__(self, root, split, cache_size=24):
        self.root=Path(root);self.cache=OrderedDict();self.limit=cache_size
        self.rows=[json.loads(s) for s in (self.root/'manifest.jsonl').read_text().splitlines() if s.strip()]
        self.rows=[r for r in self.rows if r['split']==split]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        i=int(i)
        if i<0 or i>=len(self):raise IndexError(i)
        if i not in self.cache:
            row=self.rows[i]
            with np.load(self.root/row['record_path'],allow_pickle=False) as z:
                a={k:z[k] for k in TRAIN_FIELDS if k in z.files}
            n=len(a['phonetic_features'])
            a.setdefault('phone_ids',np.arange(n,dtype=np.int64))
            a.setdefault('text_features',np.zeros((n,12),np.float32))
            assert a['phonetic_features'].shape==(n,32) and a['context_features'].shape==(n,22)
            assert a['target_controls'].shape==(n,8,31) and a['control_mask'].shape==(n,8,31)
            assert a['target_curve'].shape==(int(a['target_duration'].sum()),31)
            for key in TRAIN_FIELDS:
                if key in a and not np.isfinite(a[key]).all():raise ValueError('Nonfinite '+key)
            if not np.isin(a['target_curve'][:,-1],[0.,1.]).all():raise ValueError('Binary frame VUV required')
            self.cache[i]=(row,a)
            if len(self.cache)>self.limit:self.cache.popitem(last=False)
        self.cache.move_to_end(i);return self.cache[i]

def load_training_records(data_dir, split):
    if split not in ('train','validation'):raise ValueError('No test for training')
    return NumericRecords(data_dir,split)

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def normalization(records,vuv_supervision='frame'):
    total=np.zeros(31);square=total.copy();count=total.copy();voiced=frames=0
    for _,a in records:
        t=a['target_controls'].reshape(-1,31).astype(np.float64)
        m=a['control_mask'].reshape(-1,31).astype(np.float64)
        total+=(t*m).sum(0);square+=(t*t*m).sum(0);count+=m.sum(0)
        voiced+=int(a['target_curve'][:,-1].sum());frames+=len(a['target_curve'])
    means=total/np.maximum(count,1)
    scales=np.sqrt(np.maximum(square/np.maximum(count,1)-means**2,0))
    scales=np.maximum(scales,np.array([.1,.5]+[.05]*28+[1.]))
    means[-1]=0;scales[-1]=1
    return means.astype(np.float32),scales.astype(np.float32),float(voiced/max(frames,1))

def warmstart_physical(model,checkpoint):
    """Copy shared English weights; retarget affine output normalization exactly."""
    old=load_acoustic(checkpoint)
    if old.channels!=model.channels:raise ValueError('Warmstart channels mismatch')
    oldstate=old.state_dict();newstate=model.state_dict()
    for name in newstate:
        if name not in ('means','scales','voiced_log_prior'):newstate[name]=oldstate[name].clone()
    oldscale=old.scales.repeat(8);newscale=model.scales.repeat(8)
    oldmean=old.means.repeat(8);newmean=model.means.repeat(8)
    ratio=oldscale/newscale
    newstate['acoustic.2.weight']*=ratio[:,None]
    newstate['acoustic.2.bias']=(oldmean-newmean+oldscale*newstate['acoustic.2.bias'])/newscale
    # The VUV row is raw logit, rather than normalized physical output.
    for row in range(30,8*31,31):
        newstate['acoustic.2.weight'][row]=oldstate['acoustic.2.weight'][row]
        newstate['acoustic.2.bias'][row]=oldstate['acoustic.2.bias'][row]+old.voiced_log_prior-model.voiced_log_prior
    model.load_state_dict(newstate)
    return old

def batch_records(records, indexes):
    chosen = [records[int(i)][1] for i in indexes]
    width = max(len(a['phone_ids']) for a in chosen)
    b = len(chosen)
    ids = torch.zeros(b, width, dtype=torch.long)
    text = torch.zeros(b, width, 12)
    phonetic = torch.zeros(b,width,32)
    context = torch.zeros(b,width,22)
    valid = torch.zeros(b, width, dtype=torch.bool)
    edge = torch.zeros_like(valid)
    target = torch.zeros(b, width, 8, D)
    mask = torch.zeros_like(target)
    duration = torch.zeros(b, width)
    stress = torch.full((b, width), -1., dtype=torch.float32)
    frame_indices, frame_fractions, frame_voice = [], [], []
    for i, a in enumerate(chosen):
        n = len(a['phone_ids'])
        ids[i, :n] = torch.from_numpy(a['phone_ids'].astype(np.int64))
        text[i, :n] = torch.from_numpy(a['text_features'].astype(np.float32))
        phonetic[i,:n]=torch.from_numpy(a['phonetic_features'].astype(np.float32))
        context[i,:n]=torch.from_numpy(a['context_features'].astype(np.float32))
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
        frame_voice.append(a['target_curve'][:, -1].astype(np.float32))
    return {'phonetic_features':phonetic,'context_features':context,'phone_ids': ids, 'text_features': text, 'valid_mask': valid, 'edge_sil_mask': edge,
            'stress': stress, '_frame_indices': torch.from_numpy(np.concatenate(frame_indices)),
            '_frame_fractions': torch.from_numpy(np.concatenate(frame_fractions)),
            '_frame_voice': torch.from_numpy(np.concatenate(frame_voice))}, target, mask, duration


class PhysicalSpectrum(nn.Module):
    def __init__(self, bins=513):
        super().__init__()
        x = torch.arange(bins, dtype=torch.float32) + .5
        modes = torch.arange(1, SPEC, dtype=torch.float32)
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
    spectral = avg(norm_error[..., 2:1+SPEC].square(), mask[..., 2:1+SPEC])
    energy = avg(norm_error[..., 1].square(), mask[..., 1])
    pitch = avg(F.smooth_l1_loss(norm_error[..., 0], torch.zeros_like(norm_error[..., 0]), reduction='none'), mask[..., 0])
    ap = avg(norm_error[..., 1+SPEC:-1].square(), mask[..., 1+SPEC:-1])
    if vuv_supervision == 'frame':
        positions = features['_frame_fractions'] * 7
        left = positions.long().clamp(0, 6)
        alpha = positions - left
        c = p.reshape(-1, 8, D)[features['_frame_indices'], :, -1]
        row = torch.arange(len(left))
        logits = c[row, left] * (1 - alpha) + c[row, left + 1] * alpha - voiced_log_prior
        voice = features['_frame_voice']
        bce = F.binary_cross_entropy_with_logits(logits, voice, reduction='none')
        vuv = bce.mean()  # Proper unbalanced soft-target BCE; no prior subtraction.
    else:
        voice = torch.sigmoid(target[..., -1])
        # Control surrogate labels retain the fitted transitions. The explicit
        # frame variant instead trains against the original binary WORLD VUV.
        bce = F.binary_cross_entropy_with_logits(p[..., -1] - voiced_log_prior, voice, reduction='none')
        vuv = .5 * avg(bce, mask[..., -1] * voice) + .5 * avg(bce, mask[..., -1] * (1 - voice))
    active = (mask[..., 1] > 0).flatten().nonzero().flatten()
    if len(active) > physical_samples:
        # Random control subsampling is training-only; evaluation computes all controls.
        active = active[torch.randperm(len(active))[:physical_samples]]
    pp, tt = p.reshape(-1, D)[active], target.reshape(-1, D)[active]
    db_error = (decoder(pp[:, 1:1+SPEC]) - decoder(tt[:, 1:1+SPEC])) / math.log(10)
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
        vm = m[..., -1] > 0
        voice_correct += (((out['controls'][..., -1] > 0) == (t[..., -1] > 0)) & vm).sum().item()
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


def canonical_init(model, seed):
    # Shared 31-channel numeric model; AP3 baseline has independent band meaning,
    # but copied per-band initial rows avoid output-size RNG confounding.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        canonical=TinyAcousticModel({**model.config,'ap_bands':5}, np.zeros(31), np.ones(31))
    source=canonical.state_dict();dest=model.state_dict()
    indices=list(range(25))+list(range(25,25+BANDS))+[30]
    rows=torch.tensor([k*31+j for k in range(8) for j in indices])
    for name in dest:
        if name in ('means','scales','voiced_log_prior'):continue
        dest[name]=source[name][rows] if name in ('acoustic.2.weight','acoustic.2.bias') else source[name]
    model.load_state_dict(dest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--warmstart',type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.warmstart is None and config.get('warmstart'):args.warmstart=Path(config['warmstart'])
    if config['model'].get('continuous_voicing') or config['model'].get('spectral_dim')!=24 or config['model'].get('ap_bands')!=5:raise ValueError('Chinese baseline fixed31-channel AP5binary contract')
    global D,SPEC,BANDS,CONTINUOUS
    SPEC=config['model']['spectral_dim'];BANDS=config['model']['ap_bands'];D=SPEC+BANDS+2;CONTINUOUS=config['model']['continuous_voicing']
    for key in ('data_dir', 'output', 'steps', 'seconds'):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completed.json').exists() and not args.resume:
        print(json.dumps({'reused_completed_training':str(out),'new_run_requires_new_output':True}),flush=True)
        return
    if args.resume and (out/'completed.json').exists():
        (out/'completed.json').unlink()
    write_json(out / 'running.json', {'pid_in_tool_namespace': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
               'command': 'python -B -m project.src.experiments.chinese_train --config ' + str(args.config),
               'status': 'running; completed.json marks normal completion'})
    publication=Path(config['data_dir'])/'publication_ready.json'
    while not publication.exists():time.sleep(1)
    if config.get('frozen_publication_sha256'):
        published=json.loads(publication.read_text())
        actual=hashlib.sha256((Path(config['data_dir'])/'manifest.jsonl').read_bytes()).hexdigest()
        if actual!=config['frozen_publication_sha256'] or actual!=published['manifest_sha256'] or not published.get('edge_masks_preserved'):
            raise ValueError('Final publication barrier mismatch')
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
    model_config = {**config['model'], 'voiced_prior': .5, 'vuv_supervision': config.get('vuv_supervision', 'control')}
    model = TinyAcousticModel(model_config, means, scales)
    model.eval()
    canonical_init(model, seed)
    warm_sha=None
    if args.warmstart:
        warm_sha=hashlib.sha256(args.warmstart.read_bytes()).hexdigest()
        old=warmstart_physical(model,args.warmstart)
        probe=batch_records(validation,[0])[0]
        with torch.inference_mode():
            oldout=old({k:probe[k] for k in TEXT_INPUT_KEYS});newout=model({k:probe[k] for k in TEXT_INPUT_KEYS})
        assert float((oldout['controls']-newout['controls']).abs().max())<1e-4 and torch.equal(oldout['duration_frames'],newout['duration_frames']), 'Warmstart physical parity failed'
        write_json(out/'warmstart_parity.json',{'physical_control_max_abs':float((oldout['controls']-newout['controls']).abs().max()),'duration_max_abs':float((oldout['duration_frames']-newout['duration_frames']).abs().max()),'source_checkpoint_sha256':warm_sha,'normalization_affine_retargeted':True})
    torch.manual_seed(seed+1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', .001), weight_decay=1e-4)
    if resource_estimate(model)['parameters'] > 200000:raise ValueError('Model exceeds 200k parameter budget')
    torch.save({'format':FORMAT,'config':model.config,'model':model.state_dict(),'step':0},out/'initial.pt')
    start_step = 0
    resume_sha256 = None
    if args.resume:
        resume_bytes = args.resume.read_bytes()
        resume_sha256 = hashlib.sha256(resume_bytes).hexdigest()
        state = torch.load(io.BytesIO(resume_bytes), map_location='cpu', weights_only=True)
        if state.get('lineage',{}).get('manifest_sha256')!=hashlib.sha256((Path(config['data_dir'])/'manifest.jsonl').read_bytes()).hexdigest():raise ValueError('Resume manifest mismatch; use warmstart for a new dataset')
        model = load_acoustic_state(state)
        for name in ('width', 'phonetic_dim', 'dilations', 'context_dim', 'control_points', 'spectral_dim', 'ap_bands', 'acoustic_hidden', 'continuous_voicing'):
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
        model.config['voiced_prior'] = .5
        model.config['vuv_supervision'] = config.get('vuv_supervision', 'control')
        model.voiced_log_prior.zero_()
    decoder = PhysicalSpectrum()
    manifest = Path(config['data_dir']) / 'manifest.jsonl'
    source_paths = [Path(__file__), Path(__file__).resolve().parents[1] / 'models/universal_acoustic.py',
                    Path(__file__).resolve().parents[1] / 'frontend/chinese_training.py', Path(__file__).resolve().parents[1] / 'frontend/universal_chinese.py', Path(__file__).resolve().parents[1] / 'frontend/universal.py', Path(__file__).resolve().parents[1] / 'frontend/phonetic_features.py']
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
               'model_input_keys': ['phonetic_features','context_features','valid_mask','edge_sil_mask'],
               'stress_encoding': 'numeric frontend tone/context; no language symbols in shared model', 'warmstart':str(args.warmstart) if args.warmstart else None,'warmstart_sha256':warm_sha,
               'resume': str(args.resume) if args.resume else None, 'resume_checkpoint_sha256': resume_sha256,
               'source_sha256': source_hashes, 'loaded_record_fields': list(TRAIN_FIELDS)}
    write_json(out / 'config.json', config)
    write_json(out / 'normalization.json', {'means': model.means.tolist(), 'scales': model.scales.tolist(),
               'voiced_prior': float(torch.sigmoid(model.voiced_log_prior)),'observed_train_frame_voiced_fraction':prior, 'voice_supervision':'binary frame WORLD VUV; proper BCE; zero log prior',
               'normalization_source':'Chinese train-only weighted control statistics; warmstart affine retarget if provided'})
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
    sampler_digest=hashlib.sha256()
    print(json.dumps({'start': lineage, 'parameters': resource_estimate(model)['parameters']}), flush=True)
    with (out / 'training.jsonl').open('a') as log:
        for update in range(1, steps + 1):
            if time.monotonic() - started > wall - 15:
                break
            model.train()
            indices = rng.choice(len(train), min(record_count, len(train)), replace=False)
            sampler_digest.update(np.asarray(indices,dtype='<i8').tobytes())
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
            progress = min(1., update / steps)
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
               'checkpoint_reload_exact': True, 'sampler_indices_sha256':sampler_digest.hexdigest(), 'resources': resource_estimate(model)})
    write_json(out / 'running.json', {'pid_in_tool_namespace': os.getpid(), 'status': 'complete',
                                    'finished_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'complete': str(out / 'completed.json'), 'best_step': best_step}), flush=True)


if __name__ == '__main__':
    main()
