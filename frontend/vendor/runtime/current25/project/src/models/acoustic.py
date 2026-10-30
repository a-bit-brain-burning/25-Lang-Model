"""Small text-only, bounded-context acoustic and duration predictor.

Contract: channels = logF0, spectral24(log energy + DCT23), AP3,
voicing logit. Eight control points per phone; 10ms frame durations.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'tiny_hybrid_acoustic_v1'


class ContextBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, mask):
        x = self.norm(h) * mask[..., None]
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return (h + .5 * self.dropout(self.mix(F.gelu(x)))) * mask[..., None]


class TinyAcousticModel(nn.Module):
    def __init__(self, config, means=None, scales=None):
        super().__init__()
        self.config = {'width': 80, 'embedding_dim': 32, 'dilations': [1, 2, 4],
                       'dropout': .05, 'vocabulary_size': 43, 'feature_dim': 12,
                       'control_points': 8, 'frame_period_ms': 10,
                       'max_phone_frames': 150, **config}
        c = self.config
        if c['feature_dim'] not in (12, 13) or c['control_points'] != 8:
            raise ValueError('Only twelve text features (plus optional stress) and eight controls supported')
        w = c['width']
        self.embedding = nn.Embedding(c['vocabulary_size'], c['embedding_dim'])
        self.input = nn.Linear(c['embedding_dim'] + c['feature_dim'], w)
        self.blocks = nn.ModuleList([ContextBlock(w, d, c['dropout']) for d in c['dilations']])
        self.norm = nn.LayerNorm(w)
        self.acoustic = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, 8 * 29))
        self.duration = nn.Sequential(nn.Linear(w, w // 2), nn.GELU(), nn.Linear(w // 2, 2))
        self.register_buffer('means', torch.as_tensor(np.zeros(29) if means is None else means, dtype=torch.float32))
        self.register_buffer('scales', torch.as_tensor(np.ones(29) if scales is None else scales, dtype=torch.float32))
        prior = min(.99, max(.01, float(c.get('voiced_prior', .5))))
        self.register_buffer('voiced_log_prior', torch.tensor(math.log(prior / (1 - prior))))
        nn.init.normal_(self.acoustic[-1].weight, std=.003)
        nn.init.zeros_(self.acoustic[-1].bias)
        nn.init.zeros_(self.duration[-1].weight)
        with torch.no_grad():
            self.duration[-1].bias.copy_(torch.tensor([math.log(math.expm1(.8)), 0.]))

    def forward(self, features):
        ids = features['phone_ids']
        text = features['text_features']
        single = ids.ndim == 1
        if single:
            ids, text = ids[None], text[None]
        if self.config['feature_dim'] == 13 and text.shape[-1] == 12:
            if 'stress' not in features:
                raise ValueError('Stress model requires true dictionary stress metadata')
            stress = torch.as_tensor(features['stress'], dtype=text.dtype, device=text.device)
            if stress.ndim == 1:
                stress = stress[None]
            if stress.shape != ids.shape:
                raise ValueError('Stress sequence must match phones')
            text = torch.cat((text, stress[..., None] / 2.), -1)
        if ids.dtype != torch.long or text.shape != (*ids.shape, self.config['feature_dim']):
            raise ValueError('Phone IDs or text feature dimensions differ from model contract')
        valid = features.get('valid_mask', ids != 0)
        if valid.ndim == 1:
            valid = valid[None]
        edge = features.get('edge_sil_mask', text[..., 10:12].sum(-1) > .5)
        if edge.ndim == 1:
            edge = edge[None]
        h = F.gelu(self.input(torch.cat((self.embedding(ids), text), -1))) * valid[..., None]
        for block in self.blocks:
            h = block(h, valid)
        h = self.norm(h)
        raw = self.acoustic(h).reshape(*ids.shape, 8, 29)
        normalized = self.means + self.scales * raw
        f0 = normalized[..., :1].clamp(math.log(50), math.log(500))
        spectral = torch.cat((normalized[..., 1:2].clamp(-40, 20), normalized[..., 2:25].clamp(-20, 20)), -1)
        ap = normalized[..., 25:28].clamp(-1., 0.)
        vuv = raw[..., 28:29] + self.voiced_log_prior
        controls = torch.cat((f0, spectral, ap, vuv), -1)
        d = self.duration(h)
        positive = F.softplus(d[..., 0]) * 10.
        frames = torch.where(edge, positive * torch.sigmoid(d[..., 1]), positive.clamp_min(1.)) * valid
        output = {'controls': controls, 'duration_frames': frames,
                  'positive_frames': positive, 'edge_nonzero_logits': d[..., 1]}
        return {k: v[0] for k, v in output.items()} if single else output

    @torch.inference_mode()
    def predict_features(self, features):
        self.eval()
        device = self.means.device
        f = {k: torch.as_tensor(features[k], device=device, dtype=torch.long if k == 'phone_ids' else
                               torch.bool if k.endswith('mask') else torch.float32)
             for k in ('phone_ids', 'text_features', 'edge_sil_mask', 'stress') if k in features}
        output = self(f)
        controls = output['controls'].cpu().numpy().astype(np.float32)
        edge = f.get('edge_sil_mask', f['text_features'][:, 10:12].sum(-1) > .5)
        frames = output['positive_frames'].round().clamp(1, self.config['max_phone_frames']).long()
        durations = torch.where(edge & (output['edge_nonzero_logits'] < 0), 0, frames).cpu().numpy()
        return {'controls': controls, 'durations': durations,
                'curve': expand_controls(controls, durations), 'frame_period_ms': self.config['frame_period_ms']}

    predict_record = predict_features


def expand_controls(controls, durations):
    controls = np.asarray(controls, dtype=np.float32)
    durations = np.asarray(durations, dtype=np.int64)
    if controls.shape != (len(durations), 8, 29) or (durations < 0).any():
        raise ValueError('Expected N x 8 x 29 controls and nonnegative durations')
    result = []
    for control, frames in zip(controls, durations):
        if frames:
            position = np.arange(frames, dtype=np.float32) * 7 / frames
            left = np.minimum(position.astype(np.int64), 6)
            alpha = (position - left)[:, None]
            result.append(control[left] * (1 - alpha) + control[left + 1] * alpha)
    return np.concatenate(result).astype(np.float32) if result else np.zeros((0, 29), np.float32)


def load_acoustic(checkpoint):
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    return load_acoustic_state(state)


def load_acoustic_state(state):
    if state.get('format') != FORMAT:
        raise ValueError('Unrecognized acoustic checkpoint format')
    model = TinyAcousticModel(state['config'], state['model']['means'], state['model']['scales'])
    model.load_state_dict(state['model'])
    model.eval()
    return model


def resource_estimate(model, max_phones=256):
    c = model.config
    w, e, depth = c['width'], c['embedding_dim'], len(c['dilations'])
    p = sum(v.numel() for v in model.parameters())
    macs = (e + c['feature_dim']) * w + depth * 4 * w * w + w * w + w * 232 + w * (w // 2) + w
    return {'parameters': p, 'float32_weight_bytes': 4 * p, 'int8_weight_bytes_estimate': p,
            'MAC_per_phone_estimate': macs, 'MAC_for_256_phones_estimate': macs * max_phones,
            'receptive_field_phones': 1 + 2 * sum(c['dilations']),
            'float32_activation_workspace_estimate_bytes': 4 * max_phones * (3 * w + 232 + e + c['feature_dim']),
            'scope': 'acoustic/duration only; excludes frontend, DSP, Python/runtime overhead',
            'MCU_200MHz_verified': False, 'Python_RSS_is_not_embedded_RAM': True}


AcousticModel = TinyAcousticModel
