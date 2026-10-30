"""Numeric, training-only fidelity losses motivated by clear-speech research.

All targets are measured speech. These terms preserve contrasts; they do not
invent a universal clear-speaking rate, formant displacement or voicing rule.
"""
import torch
from torch.nn import functional as F


def average(value, weight):
    weight = torch.broadcast_to(weight.to(value.dtype), value.shape)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def huber(error):
    return F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')


def frame_logits(controls, features):
    position = features['_frame_fractions'] * 7
    left = position.long().clamp(0, 6)
    alpha = position - left
    values = controls.reshape(-1, 8, controls.shape[-1])[features['_frame_indices'], :, -1]
    rows = torch.arange(len(left), device=controls.device)
    return values[rows, left] * (1 - alpha) + values[rows, left + 1] * alpha


def clarity_terms(output, target, mask, duration, features, scales, groups):
    p = output['controls']
    observed = (duration > 0) & features['valid_mask'] & groups['lexical']
    weights = groups['duration_weight'] * observed
    lp = output['duration_frames'].clamp_min(1).log()
    lt = duration.clamp_min(1).log()
    # Center by lexical geometric mean to learn relative timing without
    # rewarding a global slowdown. Absolute timing stays in the base loss.
    count = observed.sum(-1, keepdim=True).clamp_min(1)
    relative = (lp - (lp * observed).sum(-1, keepdim=True) / count
                - lt + (lt * observed).sum(-1, keepdim=True) / count)
    timing = average(huber(relative), weights)
    vp = p[..., 2:6, 2:25].mean(-2) / scales[2:25]
    vt = target[..., 2:6, 2:25].mean(-2) / scales[2:25]
    vm = mask[..., 2:6, 2:25].amin(-2) * groups['vowel'][..., None] * observed[..., None]
    vowel = average(huber(vp - vt), vm)
    a, b = groups['pair_left'], groups['pair_right']
    flatp, flatt, flatm = (x.flatten(0, 1) for x in (vp, vt, vm))
    contrast = average(huber((flatp[a] - flatp[b]) - (flatt[a] - flatt[b])), flatm[a] * flatm[b])
    # Supervise signed changes inside consonants, including energy and AP5.
    # Mask endpoints independently: missing AP labels never become targets.
    err = (p[..., 1:-1].diff(dim=-2) - target[..., 1:-1].diff(dim=-2)) / scales[1:-1]
    cm = (mask[..., 1:, 1:-1] * mask[..., :-1, 1:-1]
          * groups['obstruent'][..., None, None] * observed[..., None, None])
    transition = average(huber(err), cm)
    logits = frame_logits(p, features)
    # Original WORLD frame VUV, with the same inference threshold as the model.
    # This is unbalanced proper BCE; do not subtract the model's prior again.
    voice = features['_frame_voice']
    lexical_frames = groups['lexical'].flatten()[features['_frame_indices']]
    vuv = average(F.binary_cross_entropy_with_logits(logits, voice, reduction='none'), lexical_frames)
    return {'relative_duration': timing, 'vowel_midpoint': vowel,
            'vowel_contrast': contrast, 'consonant_transition': transition, 'frame_vuv': vuv}


def weighted_clarity(terms, weights):
    return sum(terms[key] * float(weights[key]) for key in terms)
