"""Language independent, adjacent-segment spectral residual acoustic model.

The base network and 8x31 representation are unchanged. Each active adjacency
owns the left segment's final two controls and the right segment's first two.
No control belongs to two adjacencies. Residuals use the frozen base training
scales, then the original physical spectral bounds. Duration conditioning is
the model's continuous predicted duration, never a reference duration.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .universal_acoustic import UniversalAcousticModel

FORMAT = 'tiny_boundary_acoustic_v1'


class BoundaryAcousticModel(UniversalAcousticModel):
    """Existing forward/predict API, plus a zero-initialized shared head.

    Hidden states already have the base model's bounded convolutional context.
    The new head reads one adjacent segment, adding at most one segment of
    lookahead beyond the existing base receptive field. It is not recurrent.
    """

    def __init__(self, config, means=None, scales=None):
        super().__init__(config, means, scales)
        if self.channels != 31 or self.config['spectral_dim'] != 24:
            raise ValueError('Boundary head requires the fixed spectral24/AP5 31D contract')
        hidden = int(self.config.get('boundary_hidden', 32))
        if not 1 <= hidden <= 128:
            raise ValueError('boundary_hidden must be in 1..128')
        self.config['boundary_hidden'] = hidden
        self.config['boundary_duration_condition'] = 'log1p_predicted_continuous_frames'
        self.config['boundary_residual_domain'] = 'base_train_scale_times_raw'
        self.config['boundary_adjacency'] = 'deployment_nonzero_units_v2'
        inputs = 2 * self.config['width'] + 2 * self.config['context_dim'] + 2
        self.boundary_head = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, 92))
        nn.init.zeros_(self.boundary_head[-1].weight)
        nn.init.zeros_(self.boundary_head[-1].bias)

    @classmethod
    def from_base(cls, base, *, boundary_hidden=32):
        """Copy all existing weights exactly; only new head initialization is random."""
        if isinstance(base, cls):
            raise ValueError('from_base expects an unmodified base acoustic model')
        model = cls({**base.config, 'boundary_hidden': boundary_hidden}, base.means, base.scales)
        model.to(device=base.means.device, dtype=base.means.dtype)
        result = model.load_state_dict(base.state_dict(), strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {
                'boundary_head.0.weight', 'boundary_head.0.bias',
                'boundary_head.2.weight', 'boundary_head.2.bias'}:
            raise ValueError('Base state contract changed: ' + str(result))
        model.train(base.training)
        return model

    def _apply_boundary(self, controls, hidden, context, frames, active):
        """Apply disjoint spectral residual ownership; accepts batched tensors."""
        n = controls.shape[1]
        if n < 2:
            return controls
        # The same deployment gate used by predict_features determines whether
        # an optional segment exists. Skip zero-duration optional segments,
        # never a reference-duration decision. Reverse cummin avoids per-unit
        # Python loops and GPU synchronization. A positive pause stays active.
        indices = torch.arange(n, device=active.device).expand_as(active)
        candidates = torch.where(active, indices, n)
        nearest = torch.cummin(candidates.flip(1), dim=1).values.flip(1)
        right = torch.cat((nearest[:, 1:], torch.full_like(nearest[:, :1], n)), 1)
        pair = active & (right < n)
        right = right.clamp_max(n-1)
        gather = lambda value: value.gather(1, right[..., None].expand(-1, -1, value.shape[-1]))
        log_frames = torch.log1p(frames.clamp_min(0))[..., None]
        conditioning = torch.cat((hidden, gather(hidden), log_frames, gather(log_frames),
                                  context, gather(context)), -1)
        residual = self.boundary_head(conditioning).reshape(*pair.shape, 2, 2, 23)
        residual = residual * pair[..., None, None, None] * self.scales[2:25]
        # Each segment's first two belong to its left adjacency, final two to
        # its right adjacency. Padding and sentence edges receive no residual.
        correction = torch.zeros_like(controls[..., 2:25])
        correction[:, :, 6:8] = residual[:, :, 0]
        correction[:, :, 0:2] = torch.zeros_like(correction[:, :, 0:2]).scatter_add(
            1, right[..., None, None].expand(-1, -1, 2, 23), residual[:, :, 1])
        spectral = (controls[..., 2:25] + correction).clamp(-20., 20.)
        return torch.cat((controls[..., :2], spectral, controls[..., 25:]), -1)

    def forward(self, features):
        # The following base forward is intentionally explicit: no hooks, no
        # duplicate trunk evaluation, and no change to the shared base class.
        phonetic, context = features['phonetic_features'], features['context_features']
        single = phonetic.ndim == 2
        if single:
            phonetic, context = phonetic[None], context[None]
        shape = phonetic.shape[:-1]
        if phonetic.shape[-1] != self.config['phonetic_dim'] or context.shape != (*shape, self.config['context_dim']):
            raise ValueError('Numeric universal input dimensions differ from checkpoint contract')
        valid = features.get('valid_mask', torch.ones(shape, dtype=torch.bool, device=phonetic.device))
        edge = features.get('edge_sil_mask', torch.zeros(shape, dtype=torch.bool, device=phonetic.device))
        if valid.ndim == 1:
            valid = valid[None]
        if edge.ndim == 1:
            edge = edge[None]
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
        spectral = torch.cat((normalized[..., 1:2].clamp(-40, 20),
                              normalized[..., 2:1+self.config['spectral_dim']].clamp(-20, 20)), -1)
        ap = normalized[..., 1+self.config['spectral_dim']:-1].clamp(-1., 0.)
        vuv = raw[..., -1:] + self.voiced_log_prior
        controls = torch.cat((f0, spectral, ap, vuv), -1)
        d = self.duration(h)
        positive = F.softplus(d[..., 0]) * 10.
        frames = torch.where(edge, positive * torch.sigmoid(d[..., 1]), positive.clamp_min(1.)) * valid
        active = valid & ~(edge & (d[..., 1] < 0))
        controls = self._apply_boundary(controls, h, context, frames, active)
        output = {'controls': controls, 'duration_frames': frames,
                  'positive_frames': positive, 'edge_nonzero_logits': d[..., 1]}
        return {k: v[0] for k, v in output.items()} if single else output


def load_boundary_acoustic_state(state):
    if state.get('format') != FORMAT:
        raise ValueError('Unrecognized boundary acoustic checkpoint format')
    model = BoundaryAcousticModel(state['config'], state['model']['means'], state['model']['scales'])
    model.load_state_dict(state['model'])
    return model.eval()


def load_boundary_acoustic(checkpoint):
    return load_boundary_acoustic_state(torch.load(Path(checkpoint), map_location='cpu', weights_only=True))


def boundary_resource_estimate(model, max_phones=256):
    c = model.config
    inputs = 2*c['width'] + 2*c['context_dim'] + 2
    hidden = c['boundary_hidden']
    parameters = sum(p.numel() for p in model.boundary_head.parameters())
    return {'additional_parameters': parameters, 'float32_weight_bytes': 4*parameters,
            'additional_MAC_per_boundary': inputs*hidden + hidden*92,
            'scratch_float32_bytes_estimate': 4*(inputs+hidden+92),
            'extra_right_lookahead_phones': 1,
            'right_lookahead_scope': 'one active segment; deployment-gated zero segments are skipped',
            'extra_hidden_storage_float32_bytes_estimate': 4*c['width'],
            'batched_training_correction_bytes': 4*max_phones*8*23,
            'target_200MHz_verified': False}
