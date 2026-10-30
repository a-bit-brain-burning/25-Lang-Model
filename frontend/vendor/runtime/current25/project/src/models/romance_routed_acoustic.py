"""Numeric36/context22/plan5 acoustic model with explicit D1 timing routing.

The shared weights are unchanged in size. Spectrum/F0/energy/AP/VUV see the
whole text plan. Timing sees rate/pause and excludes the two energy/pitch plan
coordinates in both training and inference. No language or lexical rules.
"""
import torch
from .universal_acoustic import UniversalAcousticModel

FORMAT = 'tiny_romance_routed39_v1'


class RoutedRomanceAcousticModel(UniversalAcousticModel):
    def __init__(self, config, means=None, scales=None):
        if config.get('phonetic_dim') != 36 or config.get('context_dim') != 27:
            raise ValueError('Expected explicit numeric36 + context22 + plan5')
        if config.get('duration_condition_route', 'rate_pause_only_v1') != 'rate_pause_only_v1':
            raise ValueError('Unknown timing plan route')
        super().__init__({**config, 'duration_condition_route':'rate_pause_only_v1'}, means, scales)

    def forward(self, features):
        result = super().forward(features)
        context = features['context_features'].clone()
        context[...,23:25] = 0.
        timing = super().forward({**features, 'context_features':context})
        for name in ('duration_frames','positive_frames','edge_nonzero_logits'):
            result[name] = timing[name]
        return result


def load_routed(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Expected explicitly routed Romance39 checkpoint')
    model=RoutedRomanceAcousticModel(state['config'],state['model']['means'],state['model']['scales'])
    model.load_state_dict(state['model']);model.eval()
    return model,state
