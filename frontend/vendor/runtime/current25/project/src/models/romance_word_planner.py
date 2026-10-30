"""Numeric word guidance for independent Romance-language experiments.

Same small BiGRU/plan5 recipe as the English planner, with an explicitly new
162D numeric input. No English phone inventory, function-word list, spelling
rule, reference recording or duration is used for ordinary prediction.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

FORMAT = 'tiny_romance_numeric_word_planner_v1'
PLAN_VERSION = 'romance_word_plan5_pause_v1'
PLAN_NAMES = ('log_duration_residual_z', 'log_energy_residual_z',
              'pitch_level_residual_z', 'pause_probability', 'log1p_pause_frames_z')
PLAN_DIM = 5
PHONETIC_DIM = 36
SCALARS = ('character_length_div20', 'phone_length_div12', 'vowels_div6',
           'primary_stresses_div3', 'secondary_stresses_div3', 'first_primary_position',
           'stress_known_fraction', 'pronunciation_uncertainty', 'word_position',
           'words_remaining_fraction', 'sentence_words_div50', 'sentence_phones_div200',
           'punctuation_before', 'punctuation_after', 'first_word', 'last_word',
           'apostrophe', 'vowel_phone_fraction')
BASE_FEATURE_NAMES = SCALARS + tuple(f'mean_phonetic_{i}' for i in range(PHONETIC_DIM))
FEATURE_NAMES = tuple(prefix + '/' + name for prefix in ('current', 'previous', 'next')
                      for name in BASE_FEATURE_NAMES)
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_SCHEMA_SHA256 = hashlib.sha256('\n'.join(FEATURE_NAMES).encode()).hexdigest()


def _arrays(front):
    phonetic = np.asarray(front.get('phonetic_features', front.get('phonetic')), np.float32)
    context = np.asarray(front.get('context_features', front.get('context')), np.float32)
    lexical = np.asarray(front['lexical_word_indices'], np.int64)
    owner = np.asarray(front['plan_owner'], np.int64)
    words = list(front['words'])
    n, w = len(phonetic), len(words)
    if (phonetic.shape != (n, PHONETIC_DIM) or context.shape != (n, 22)
            or lexical.shape != (n,) or owner.shape != (n,) or not w or not n
            or not np.isfinite(phonetic).all() or not np.isfinite(context).all()):
        raise ValueError('Expected finite phonetic[N,36], context[N,22], words and unit mappings')
    if ((lexical < -1) | (lexical >= w) | (owner < -1) | (owner >= w)).any():
        raise ValueError('Word mapping outside utterance')
    if np.any(owner[lexical >= 0] != lexical[lexical >= 0]):
        raise ValueError('Lexical phone must receive its own word plan')
    # Slot0/1 are the shared manner_silence/manner_vowel bits in phonetic36.
    pause = (lexical < 0) & (owner >= 0)
    if np.any(pause & (phonetic[:, 0] < .5)) or np.any(owner[pause] >= w - 1):
        raise ValueError('Internal pause needs a silence unit and a nonfinal left-word owner')
    if 'internal_pause_mask' in front and not np.array_equal(np.asarray(front['internal_pause_mask'], bool), pause):
        raise ValueError('Internal-pause topology and plan ownership differ')
    vowel = np.asarray(front.get('vowel_mask', phonetic[:, 1] > .5), bool)
    if vowel.shape != (n,) or np.any(vowel & (lexical < 0)):
        raise ValueError('Vowel mask differs from lexical units')
    return phonetic, context, lexical, owner, words, pause, vowel


def encode_frontend(front):
    """Pure text/frontend numeric features; silence never enters phone means."""
    phonetic, context, lexical, owner, words, pause, vowel = _arrays(front)
    n, w = len(phonetic), len(words)
    base = np.zeros((w, len(BASE_FEATURE_NAMES)), np.float32)
    boundary = np.zeros(w, np.float32)
    ranges = []
    confidence = np.asarray(front.get('pronunciation_confidence', np.zeros(w)), np.float32)
    if confidence.shape != (w,) or not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError('Pronunciation confidence must be finite [words] within [0,1]')
    for wi, word in enumerate(words):
        indices = np.flatnonzero(lexical == wi)
        if not len(indices) or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError('Each word needs contiguous lexical phones, excluding silence')
        a, b = int(indices[0]), int(indices[-1] + 1)
        ranges.append([a, b])
        known = context[a:b, 13] > .5
        strength = context[a:b, 12]
        primary = np.flatnonzero(known & (strength >= .75))
        secondary = known & (strength > 0) & (strength < .75)
        before, after = float(context[a, 9]), float(context[b - 1, 8])
        boundary[wi] = after
        scalars = [min(len(word) / 20, 2), min((b - a) / 12, 2), min(vowel[a:b].sum() / 6, 2),
                   min(len(primary) / 3, 2), min(secondary.sum() / 3, 2),
                   float(primary[0] / max(1, b - a - 1)) if len(primary) else 0.,
                   float(known.mean()), 1. - float(confidence[wi]), wi / max(1, w - 1),
                   (w - 1 - wi) / max(1, w - 1), min(w / 50, 2),
                   min((lexical >= 0).sum() / 200, 2), before, after, float(wi == 0),
                   float(wi == w - 1), float("'" in word or '’' in word), float(vowel[a:b].mean())]
        base[wi] = np.r_[scalars, phonetic[a:b].mean(0)]
    previous, following = np.zeros_like(base), np.zeros_like(base)
    previous[1:], following[:-1] = base[:-1], base[1:]
    return {'numeric': np.concatenate((base, previous, following), -1), 'boundary': boundary,
            'word_indices': owner.copy(), 'plan_owner': owner.copy(),
            'lexical_word_indices': lexical.copy(), 'internal_pause_mask': pause.copy(),
            'word_phone_ranges': np.asarray(ranges, np.int64), 'words': words,
            'text_only': True, 'plan_version': PLAN_VERSION,
            'feature_schema_sha256': FEATURE_SCHEMA_SHA256}


def plan_for_units(word_plan, plan_owner):
    plan, owner = np.asarray(word_plan, np.float32), np.asarray(plan_owner, np.int64)
    if (plan.ndim != 2 or plan.shape[1] != PLAN_DIM or owner.ndim != 1
            or not np.isfinite(plan).all() or np.any((owner < -1) | (owner >= len(plan)))):
        raise ValueError('Expected word_plan[words,5] and valid plan_owner[units]')
    result = np.zeros((len(owner), PLAN_DIM), np.float32)
    valid = owner >= 0
    result[valid] = plan[owner[valid]]
    return result


class RomanceWordPlanner(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {'feature_dim': FEATURE_DIM, 'feature_schema_sha256': FEATURE_SCHEMA_SHA256,
                       'width': 48, 'hidden': 32, 'layers': 2, 'dropout': .1, **(config or {})}
        c = self.config
        if c['feature_dim'] != FEATURE_DIM or c['feature_schema_sha256'] != FEATURE_SCHEMA_SHA256:
            raise ValueError('Numeric word feature schema differs')
        self.input, self.norm = nn.Linear(FEATURE_DIM, c['width']), nn.LayerNorm(c['width'])
        self.gru = nn.GRU(c['width'], c['hidden'], num_layers=c['layers'], batch_first=True,
                          bidirectional=True, dropout=c['dropout'] if c['layers'] > 1 else 0.)
        self.head = nn.Sequential(nn.Linear(2 * c['hidden'], 32), nn.GELU(), nn.Linear(32, PLAN_DIM))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[3] = -3.
        self.normalization = {}

    def forward(self, features):
        numeric, boundary = features['numeric'], features['boundary']
        single = numeric.ndim == 2
        if single:
            numeric, boundary = numeric[None], boundary[None]
        if numeric.ndim != 3 or numeric.shape[-1] != FEATURE_DIM or boundary.shape != numeric.shape[:2]:
            raise ValueError('Word input shape mismatch')
        valid = features.get('valid_mask', torch.ones(numeric.shape[:2], dtype=torch.bool, device=numeric.device))
        if valid.ndim == 1:
            valid = valid[None]
        if valid.dtype != torch.bool or valid.shape != numeric.shape[:2]:
            raise ValueError('Word mask must be boolean and match word axis')
        lengths = valid.sum(1)
        if torch.any(lengths < 1) or not torch.equal(valid, torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]):
            raise ValueError('Planner needs a nonempty contiguous word prefix')
        hidden = F.gelu(self.norm(self.input(numeric)))
        packed = pack_padded_sequence(hidden, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=numeric.shape[1])
        raw = self.head(hidden)
        continuous = 3 * torch.tanh(raw / 3)
        next_word = torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None] - 1
        plan = torch.cat((continuous[..., :3], raw[..., 3:4].sigmoid() * next_word[..., None],
                          continuous[..., 4:5] * next_word[..., None]), -1) * valid[..., None]
        result = {'plan': plan, 'pause_logits': raw[..., 3], 'valid_mask': valid}
        return {k: v[0] for k, v in result.items()} if single else result

    @torch.inference_mode()
    def predict_frontend(self, front):
        encoded = encode_frontend(front)
        device = next(self.parameters()).device
        self.eval()
        features = {k: torch.from_numpy(encoded[k]).to(device) for k in ('numeric', 'boundary')}
        plan = self(features)['plan'].cpu().numpy().astype(np.float32)
        return {'frontend': front, 'word_plan': plan,
                'unit_plan': plan_for_units(plan, encoded['plan_owner']),
                'word_indices': encoded['plan_owner'], 'plan_owner': encoded['plan_owner'],
                'word_phone_ranges': encoded['word_phone_ranges'], 'words': encoded['words'],
                'plan_version': PLAN_VERSION, 'plan_source': 'PREDICTED_TEXT_ONLY',
                'pause_source': 'PREDICTED_FROM_TEXT_TRAINED_ON_EXPLICIT_ALIGNMENT_GAPS'}


def _train_only(measurements, split):
    if split != 'train' or any(m.get('split', 'train') != 'train' for m in measurements):
        raise ValueError('Baseline/normalization fitting requires train records only')


def _signature(vector):
    return hashlib.sha256(np.asarray(vector, dtype='<f4').tobytes()).hexdigest()


def reference_measurements(front, record):
    """Reference-only neural39 statistics; use VUV38, never spectral channel30."""
    phonetic, context, lexical, owner, words, pause, vowel = _arrays(front)
    encoded = encode_frontend(front)
    duration_raw = np.asarray(record.get('durations', record.get('target_duration')))
    if (duration_raw.shape != (len(phonetic),) or not np.isfinite(duration_raw).all()
            or np.any(duration_raw < 0) or np.any(duration_raw != np.rint(duration_raw))):
        raise ValueError('Reference unit durations must be nonnegative integers')
    duration = duration_raw.astype(np.int64)
    curve = np.asarray(record.get('curve', record.get('target_curve')), np.float32)
    if curve.shape != (int(duration.sum()), 39) or not np.isfinite(curve).all():
        raise ValueError('Reference must have common-axis neural39 curve')
    for key, expected in (('internal_pause_mask', pause), ('plan_owner', owner), ('lexical_word_indices', lexical)):
        if key in record and not np.array_equal(np.asarray(record[key]), expected):
            raise ValueError('Reference/frontend topology differs: ' + key)
    offsets = np.r_[0, duration.cumsum()]
    energy_valid = np.repeat(vowel, duration) & (curve[:, 38] > 0) & (curve[:, 1] * 10 / math.log(10) >= -45)
    f0_valid = energy_valid & (curve[:, 0] >= math.log(50)) & (curve[:, 0] <= math.log(500))
    if 'frame_f0_valid' in record:
        trusted = np.asarray(record['frame_f0_valid'], bool)
        if trusted.shape != (len(curve),):
            raise ValueError('F0 confidence and reference axis differ')
        f0_valid &= trusted
    sentence_pitch = float(np.median(curve[f0_valid, 0])) if f0_valid.sum() >= 10 else np.nan
    energy_sum, energy_count = np.zeros(len(duration)), np.zeros(len(duration), np.int64)
    for i in np.flatnonzero(vowel):
        a, b = offsets[i:i + 2]
        valid = energy_valid[a:b]
        energy_sum[i], energy_count[i] = curve[a:b, 1][valid].sum(dtype=np.float64), valid.sum()
    w = len(words)
    word_duration, word_energy, word_pitch = np.zeros(w), np.full(w, np.nan), np.full(w, np.nan)
    pause_frames, pause_available = np.zeros(w, np.int64), np.zeros(w, bool)
    for i in np.flatnonzero(pause):
        wi = owner[i]
        pause_available[wi] = True
        pause_frames[wi] += duration[i]
    for wi, (a, b) in enumerate(encoded['word_phone_ranges']):
        word_duration[wi] = duration[a:b].sum()
        count = energy_count[a:b].sum()
        if count >= 3:
            word_energy[wi] = energy_sum[a:b].sum() / count
        lo, hi = offsets[a], offsets[b]
        valid = f0_valid[lo:hi]
        if valid.sum() >= 3 and np.isfinite(sentence_pitch):
            word_pitch[wi] = float(np.median(curve[lo:hi, 0][valid])) - sentence_pitch
    return {'encoded': encoded,
            'phone_keys': [_signature(np.r_[x, context[i, 12:14]]) for i, x in enumerate(phonetic)],
            'lexical_word_indices': lexical, 'duration': duration,
            'word_duration': word_duration, 'word_energy': word_energy,
            'word_pitch_relative': word_pitch, 'energy_sum': energy_sum, 'energy_count': energy_count,
            'pause_frames': pause_frames, 'pause_available': pause_available,
            'split': record.get('split', 'train'),
            'confidence_source': 'automatic alignment/WORLD; voiced vowel>=-45dB; F0 50..500Hz; optional trusted F0 mask'}


def fit_phone_baselines(measurements, split='train'):
    measurements = list(measurements)
    _train_only(measurements, split)
    stats = defaultdict(lambda: np.zeros(4, np.float64))
    for m in measurements:
        for key, lexical, duration, energy, count in zip(m['phone_keys'], m['lexical_word_indices'], m['duration'], m['energy_sum'], m['energy_count']):
            if lexical < 0:
                continue
            for k in ('phone:' + key, 'global'):
                stats[k] += [duration if duration > 0 else 0, int(duration > 0), energy, count]
    if 'global' not in stats or stats['global'][1] < 1:
        raise ValueError('No positive lexical train duration for baseline')
    return {k: {'duration_mean_frames': float(v[0] / max(1, v[1])), 'duration_support': int(v[1]),
                'energy_mean_log_power': float(v[2] / max(1, v[3])), 'energy_frame_support': int(v[3])}
            for k, v in stats.items()}


def _baseline(baselines, phone, field):
    support = 'duration_support' if field == 'duration_mean_frames' else 'energy_frame_support'
    for key in ('phone:' + phone, 'global'):
        item = baselines.get(key)
        if item and item[support] >= (20 if key != 'global' else 1):
            return item[field]
    raise ValueError('Missing train-only phonetic baseline for ' + field)


def raw_plan(measurement, baselines):
    m = measurement
    raw = np.zeros((len(m['word_duration']), PLAN_DIM), np.float32)
    mask = np.zeros_like(raw, bool)
    for wi, (a, b) in enumerate(m['encoded']['word_phone_ranges']):
        expected = sum(_baseline(baselines, p, 'duration_mean_frames') for p in m['phone_keys'][a:b])
        if m['word_duration'][wi] > 0 and expected > 0:
            raw[wi, 0], mask[wi, 0] = math.log(m['word_duration'][wi] / expected), True
        if np.isfinite(m['word_energy'][wi]):
            counts = m['energy_count'][a:b]
            # Nonvowel phones have zero energy count and need no energy baseline.
            expected_energy = sum(c * _baseline(baselines, p, 'energy_mean_log_power')
                                  for p, c in zip(m['phone_keys'][a:b], counts) if c)
            raw[wi, 1] = m['word_energy'][wi] - expected_energy / max(1, counts.sum())
            mask[wi, 1] = True
        if np.isfinite(m['word_pitch_relative'][wi]):
            raw[wi, 2], mask[wi, 2] = m['word_pitch_relative'][wi], True
        if m['pause_available'][wi]:
            raw[wi, 3], mask[wi, 3] = float(m['pause_frames'][wi] > 0), True
        if m['pause_frames'][wi] > 0:
            raw[wi, 4], mask[wi, 4] = np.log1p(m['pause_frames'][wi]), True
    return raw, mask


def fit_plan_normalization(raw_pairs, split='train'):
    _train_only([], split)
    pairs = list(raw_pairs)
    if not pairs:
        raise ValueError('No train plan records')
    values, masks = np.concatenate([x for x, _ in pairs]), np.concatenate([x for _, x in pairs])
    means, scales, counts = [], [], []
    for j, floor in zip((0, 1, 2, 4), (.1, .25, .1, .25)):
        valid = values[masks[:, j], j]
        if not np.isfinite(valid).all():
            raise ValueError('Nonfinite valid plan label')
        means.append(float(valid.mean()) if len(valid) else 0.)
        scales.append(max(floor, float(valid.std())) if len(valid) else floor)
        counts.append(len(valid))
    return {'continuous_indices': [0, 1, 2, 4], 'mean': means, 'scale': scales,
            'valid_word_count': counts, 'fit_split': 'train', 'clip_z': 3.,
            'plan_version': PLAN_VERSION, 'missing_fields_are_masked': True,
            'raw_units': ['log(speech_word_frames/train_expected_phone_frames)',
                          'log_wave_power_residual', 'logHz_relative_to_recording_voiced_vowel_median',
                          'log1p_positive_explicit_pause_frames']}


def normalized_plan(raw, mask, normalization, boundary=None):
    raw, mask = np.asarray(raw, np.float32), np.asarray(mask, bool)
    if raw.ndim != 2 or raw.shape[1] != PLAN_DIM or mask.shape != raw.shape or normalization.get('fit_split') != 'train':
        raise ValueError('Plan/normalization schema mismatch')
    if not np.isfinite(raw[mask]).all():
        raise ValueError('Nonfinite valid plan value')
    result = np.zeros_like(raw)
    columns = np.asarray(normalization['continuous_indices'])
    if not np.array_equal(columns, [0, 1, 2, 4]):
        raise ValueError('Continuous plan field order differs')
    z = (raw[:, columns] - normalization['mean']) / normalization['scale']
    result[:, columns] = np.where(mask[:, columns], np.clip(z, -3, 3), 0)
    result[:, 3] = np.where(mask[:, 3], raw[:, 3], 0)
    return result, mask.copy()


def objective(output, target, mask):
    if target.shape != output['plan'].shape or mask.shape != target.shape or target.shape[-1] != PLAN_DIM:
        raise ValueError('Plan loss shape mismatch')
    if not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite valid plan target')
    target = torch.where(mask, target.float(), torch.zeros_like(target, dtype=torch.float32))
    error = F.smooth_l1_loss(output['plan'].float(), target, reduction='none')
    error = torch.cat((error[..., :3], F.binary_cross_entropy_with_logits(
        output['pause_logits'].float(), target[..., 3], reduction='none')[..., None], error[..., 4:5]), -1)
    dims = tuple(range(error.ndim - 1))
    fields = (error * mask).sum(dims) / mask.sum(dims).clamp_min(1)
    return fields.mean()


def checkpoint_payload(model, normalization=None, step=0):
    return {'format': FORMAT, 'plan_version': PLAN_VERSION, 'feature_schema_sha256': FEATURE_SCHEMA_SHA256,
            'config': dict(model.config), 'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'normalization': model.normalization if normalization is None else normalization, 'step': int(step)}


def load_word_planner(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('format') != FORMAT or state.get('plan_version') != PLAN_VERSION
            or state.get('feature_schema_sha256') != FEATURE_SCHEMA_SHA256):
        raise ValueError('Unknown Romance numeric planner contract')
    model = RomanceWordPlanner(state['config'])
    model.load_state_dict(state['model'], strict=True)
    model.normalization, model.checkpoint_step = state.get('normalization', {}), state.get('step', 0)
    return model.eval()
