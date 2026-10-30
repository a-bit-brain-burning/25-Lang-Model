"""Tiny optional parameter-domain correction: final=baseline+alpha*delta.

The network observes only the predicted29 channels, never text/audio labels
at inference. Mean/scale are fitted on training targets by the baseline.
"""
import numpy as np
import torch
from torch import nn


class ParameterResidual(nn.Module):
    def __init__(self, means, scales, width=64):
        super().__init__()
        self.width = int(width)
        self.register_buffer('means', torch.as_tensor(means, dtype=torch.float32).clone())
        self.register_buffer('scales', torch.as_tensor(scales, dtype=torch.float32).clone().clamp_min(.01))
        self.network = nn.Sequential(nn.Linear(29, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 29))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, baseline):
        return self.network((baseline - self.means) / self.scales) * self.scales

    @torch.inference_mode()
    def correct(self, baseline, alpha=1.):
        tensor = torch.as_tensor(baseline, dtype=torch.float32)
        return (tensor + float(alpha) * self(tensor)).cpu().numpy().astype(np.float32)


def load_residual(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'parameter_residual_v1':
        raise ValueError('Unknown residual checkpoint format')
    state = checkpoint['model']
    model = ParameterResidual(state['means'], state['scales'], checkpoint['width'])
    model.load_state_dict(state)
    model.eval()
    model.baseline_sha256 = checkpoint.get('lineage', {}).get('baseline_sha256')
    return model, float(checkpoint.get('selected_alpha', 1.))
