"""Language independent numeric articulatory/prosodic acoustic model; no language symbols or rules.

Contract: log F0, spectral energy plus bounded DCT shape, AP3 or AP5,
voicing logit (legacy binary controls or continuous frame supervision).
Eight control points per segment; 10 ms durations. Model input is numeric
articulatory/context features, without a phoneme ID table or language rules.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'tiny_universal_acoustic_v2'


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


class UniversalAcousticModel(nn.Module):
    def __init__(self, config, means=None, scales=None):
        super().__init__()
        self.config = {'width': 80, 'dilations': [1, 2, 4],
                       'spectral_dim': 24, 'ap_bands': 3, 'acoustic_hidden': 63, 'dropout': .05, 'phonetic_dim': 32, 'context_dim': 22, 'continuous_voicing': True,
                       'control_points': 8, 'frame_period_ms': 10,
                       'max_phone_frames': 150, **config}
        self.config.pop('embedding_dim', None)  # Historical unused metadata only.
        c = self.config
        if c['control_points'] != 8:
            raise ValueError('Eight controls per segment required')
        w = c['width']
        self.channels = c['spectral_dim'] + c['ap_bands'] + 2
        c['channels'] = self.channels
        self.input = nn.Linear(c['phonetic_dim'] + c['context_dim'], w)
        self.blocks = nn.ModuleList([ContextBlock(w, d, c['dropout']) for d in c['dilations']])
        self.norm = nn.LayerNorm(w)
        self.acoustic = nn.Sequential(nn.Linear(w, c['acoustic_hidden']), nn.GELU(), nn.Linear(c['acoustic_hidden'], 8 * self.channels))
        # Optional direct articulatory path. Numeric features only; existing
        # checkpoints retain their exact network and state-dict contract.
        if int(c.get('segment_skip_hidden', 0)) > 0:
            self.segment_skip = nn.Sequential(
                nn.Linear(c['phonetic_dim'], int(c['segment_skip_hidden'])),
                nn.GELU(), nn.Linear(int(c['segment_skip_hidden']), 8 * self.channels))
            nn.init.zeros_(self.segment_skip[-1].weight)
            nn.init.zeros_(self.segment_skip[-1].bias)
        self.duration = nn.Sequential(nn.Linear(w, w // 2), nn.GELU(), nn.Linear(w // 2, 2))
        self.register_buffer('means', torch.as_tensor(np.zeros(self.channels) if means is None else means, dtype=torch.float32))
        self.register_buffer('scales', torch.as_tensor(np.ones(self.channels) if scales is None else scales, dtype=torch.float32))
        prior = min(.99, max(.01, float(c.get('voiced_prior', .5)) if not c['continuous_voicing'] else .5))
        self.register_buffer('voiced_log_prior', torch.tensor(math.log(prior / (1 - prior))))
        nn.init.normal_(self.acoustic[-1].weight, std=.003)
        nn.init.zeros_(self.acoustic[-1].bias)
        nn.init.zeros_(self.duration[-1].weight)
        with torch.no_grad():
            self.duration[-1].bias.copy_(torch.tensor([math.log(math.expm1(.8)), 0.]))

    def forward(self, features):
        phonetic = features['phonetic_features']
        context = features['context_features']
        single = phonetic.ndim == 2
        if single:
            phonetic, context = phonetic[None], context[None]
        shape = phonetic.shape[:-1]
        if phonetic.shape[-1] != self.config['phonetic_dim'] or context.shape != (*shape, self.config['context_dim']):
            raise ValueError('Numeric universal input dimensions differ from checkpoint contract')
        valid = features.get('valid_mask', torch.ones(shape, dtype=torch.bool, device=phonetic.device))
        edge = features.get('edge_sil_mask', torch.zeros(shape, dtype=torch.bool, device=phonetic.device))
        if valid.ndim == 1: valid = valid[None]
        if edge.ndim == 1: edge = edge[None]
        h = F.gelu(self.input(torch.cat((phonetic, context), -1))) * valid[..., None]
        for block in self.blocks:
            h = block(h, valid)
        h = self.norm(h)
        acoustic_raw = self.acoustic(h)
        if hasattr(self, 'segment_skip'):
            acoustic_raw = acoustic_raw + self.segment_skip(phonetic)
        raw = acoustic_raw.reshape(*shape, 8, self.channels)
        normalized = self.means + self.scales * raw
        f0 = normalized[..., :1].clamp(math.log(50), math.log(500))
        spectral = torch.cat((normalized[..., 1:2].clamp(-40, 20), normalized[..., 2:1+self.config['spectral_dim']].clamp(-20, 20)), -1)
        ap = normalized[..., 1+self.config['spectral_dim']:-1].clamp(-1., 0.)
        vuv = raw[..., -1:] + self.voiced_log_prior
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
        f = {k: torch.as_tensor(features[k], device=device, dtype=torch.bool if k.endswith('mask') else torch.float32)
             for k in ('phonetic_features', 'context_features', 'valid_mask', 'edge_sil_mask') if k in features}
        output = self(f)
        controls = output['controls'].cpu().numpy().astype(np.float32)
        edge = f.get('edge_sil_mask', torch.zeros(f['phonetic_features'].shape[0], dtype=torch.bool, device=device))
        frames = output['positive_frames'].round().clamp(1, self.config['max_phone_frames']).long()
        durations = torch.where(edge & (output['edge_nonzero_logits'] < 0), 0, frames).cpu().numpy()
        return {'controls': controls, 'durations': durations,
                'curve': expand_controls(controls, durations), 'frame_period_ms': self.config['frame_period_ms']}

    predict_record = predict_features


def expand_controls(controls, durations):
    controls = np.asarray(controls, dtype=np.float32)
    durations = np.asarray(durations, dtype=np.int64)
    if controls.shape != (len(durations), 8, controls.shape[-1]) or (durations < 0).any():
        raise ValueError('Expected N x 8 x D controls and nonnegative durations')
    result = []
    for control, frames in zip(controls, durations):
        if frames:
            position = np.arange(frames, dtype=np.float32) * 7 / frames
            left = np.minimum(position.astype(np.int64), 6)
            alpha = (position - left)[:, None]
            result.append(control[left] * (1 - alpha) + control[left + 1] * alpha)
    return np.concatenate(result).astype(np.float32) if result else np.zeros((0, controls.shape[-1]), np.float32)


def load_acoustic(checkpoint):
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    return load_acoustic_state(state)


def load_acoustic_state(state):
    if state.get('format') != FORMAT:
        raise ValueError('Unrecognized acoustic checkpoint format')
    model = UniversalAcousticModel(state['config'], state['model']['means'], state['model']['scales'])
    model.load_state_dict(state['model'])
    model.eval()
    return model


def resource_estimate(model, max_phones=256):
    c = model.config
    w, e, depth = c['width'], c['phonetic_dim'], len(c['dilations'])
    p = sum(v.numel() for v in model.parameters())
    macs = (e + c['context_dim']) * w + depth * 4 * w * w + w * c['acoustic_hidden'] + c['acoustic_hidden'] * 8 * model.channels + w * (w // 2) + w
    skip = int(c.get('segment_skip_hidden', 0))
    macs += (e * skip + skip * 8 * model.channels) if skip else 0
    return {'parameters': p, 'float32_weight_bytes': 4 * p, 'int8_weight_bytes_estimate': p,
            'MAC_per_phone_estimate': macs, 'MAC_for_256_phones_estimate': macs * max_phones,
            'receptive_field_phones': 1 + 2 * sum(c['dilations']),
            'float32_activation_workspace_estimate_bytes': 4 * max_phones * (3 * w + 8 * model.channels + e + c['context_dim']),
            'scope': 'acoustic/duration only; excludes frontend, DSP, Python/runtime overhead',
            'MCU_200MHz_verified': False, 'Python_RSS_is_not_embedded_RAM': True}


AcousticModel = UniversalAcousticModel
