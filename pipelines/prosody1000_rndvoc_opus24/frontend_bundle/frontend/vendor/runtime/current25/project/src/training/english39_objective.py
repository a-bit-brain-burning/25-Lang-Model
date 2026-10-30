"""Dimension-aware confidence supervision; no linguistic rules or smoothing."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .clarity_objective import average, frame_logits, huber, weighted_clarity
from ..experiments.english_strict_train import strict_common


def clarity(output, target, mask, duration, features, scales, groups, spectral_dim):
    p = output['controls']
    observed = (duration > 0) & features['valid_mask'] & groups['lexical']
    count = observed.sum(-1, keepdim=True).clamp_min(1)
    lp, lt = output['duration_frames'].clamp_min(1).log(), duration.clamp_min(1).log()
    relative = lp-(lp*observed).sum(-1, keepdim=True)/count-lt+(lt*observed).sum(-1, keepdim=True)/count
    timing = average(huber(relative), groups['duration_weight']*observed)
    stop = 1+spectral_dim
    vp = p[..., 2:6, 2:stop].mean(-2)/scales[2:stop]
    vt = target[..., 2:6, 2:stop].mean(-2)/scales[2:stop]
    vm = mask[..., 2:6, 2:stop].amin(-2)*groups['vowel'][..., None]*observed[..., None]
    vowel = average(huber(vp-vt), vm)
    a, b = groups['pair_left'], groups['pair_right']
    fp, ft, fm = (x.flatten(0, 1) for x in (vp, vt, vm))
    contrast = average(huber((fp[a]-fp[b])-(ft[a]-ft[b])), fm[a]*fm[b])
    delta = (p[..., 1:-1].diff(dim=-2)-target[..., 1:-1].diff(dim=-2))/scales[1:-1]
    cm = mask[..., 1:, 1:-1]*mask[..., :-1, 1:-1]*groups['obstruent'][..., None, None]*observed[..., None, None]
    transition = average(huber(delta), cm)
    weight = features.get('_frame_voice_weight', torch.ones_like(features['_frame_voice']))
    lexical = groups['lexical'].flatten()[features['_frame_indices']]
    voice = average(F.binary_cross_entropy_with_logits(frame_logits(p, features),
                    features['_frame_voice'], reduction='none'), lexical*weight)
    return {'relative_duration': timing, 'vowel_midpoint': vowel,
            'vowel_contrast': contrast, 'consonant_transition': transition, 'frame_vuv': voice}


def objective(model, packed, decoder, config, all_physical=False):
    features, target, mask, duration, groups = packed
    out = model({k: features[k] for k in ('phonetic_features', 'context_features', 'valid_mask', 'edge_sil_mask')})
    common, terms = strict_common(out, target, mask, duration, features, model.scales,
                                 decoder, 10**8 if all_physical else config['physical_samples'],
                                 model.voiced_log_prior)
    extras = clarity(out, target, mask, duration, features, model.scales, groups,
                     model.config['spectral_dim'])
    loss = common+weighted_clarity(extras, config['weights'])
    pause = features['_internal_pause_mask'] & features['valid_mask']
    positive, negative = pause & (duration > 0), pause & (duration == 0)
    avg = lambda value, weight: (value*weight).sum()/weight.sum().clamp_min(1)
    gate = F.binary_cross_entropy_with_logits(out['edge_nonzero_logits'], (duration > 0).float(), reduction='none')
    gate_loss = .5*avg(gate, positive)+.5*avg(gate, negative)
    error = F.smooth_l1_loss(torch.log1p(out['positive_frames']), torch.log1p(duration), reduction='none', beta=.5)
    pause_duration = avg(error, positive)
    loss = loss+.05*gate_loss+.2*pause_duration
    return loss, {**terms, **{'clarity_'+k: v for k,v in extras.items()},
                  'pause_gate': gate_loss, 'positive_pause_duration': pause_duration,
                  'total_with_aux': loss}
