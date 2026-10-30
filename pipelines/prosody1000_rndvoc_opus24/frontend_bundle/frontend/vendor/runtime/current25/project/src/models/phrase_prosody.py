"""Small global phrase planner: numeric text features + characters -> plan8."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

FORMAT = 'tiny_phrase_prosody_v1'


class PhraseProsodyPlanner(nn.Module):
    def __init__(self, config, vocabulary=None):
        super().__init__()
        self.config = {'numeric_dim': 66, 'embedding_dim': 16, 'input_hidden': 64,
                       'gru_hidden': 96, 'head_hidden': 64, 'dropout': .05,
                       **config}
        self.vocabulary = vocabulary or {'<pad>': 0, '<unk>': 1}
        c = self.config
        self.embedding = nn.Embedding(c['vocab_size'], c['embedding_dim'], padding_idx=0)
        self.input = nn.Sequential(nn.Linear(c['numeric_dim'] + c['embedding_dim'], c['input_hidden']), nn.GELU())
        self.gru = nn.GRU(c['input_hidden'], c['gru_hidden'], batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2*c['gru_hidden'], c['head_hidden']), nn.GELU(),
                                  nn.Dropout(c['dropout']), nn.Linear(c['head_hidden'], 8))

    def forward(self, features):
        numeric, characters = features['numeric'], features['char_ids']
        single = numeric.ndim == 2
        if single:
            numeric, characters = numeric[None], characters[None]
        valid = features.get('valid_mask', characters != 0)
        if valid.ndim == 1:
            valid = valid[None]
        if numeric.shape[:2] != characters.shape or numeric.shape[-1] != self.config['numeric_dim']:
            raise ValueError('Invalid text-only planner input dimensions')
        lengths = valid.long().sum(1)
        if torch.any(lengths == 0):
            raise ValueError('Empty planner sentence')
        hidden = self.input(torch.cat((numeric, self.embedding(characters)), -1))
        packed = pack_padded_sequence(hidden, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=numeric.shape[1])
        raw = self.head(encoded)
        boundary = raw[..., :4].softmax(-1)
        continuous = torch.stack((2*raw[..., 4].tanh(), .5*raw[..., 5].tanh(),
                                  2*raw[..., 6].tanh(), 2*raw[..., 7].tanh()), -1)
        result = {'plan': torch.cat((boundary, continuous), -1) * valid[..., None],
                  'boundary_logits': raw[..., :4], 'valid_mask': valid}
        return {k: v[0] for k, v in result.items()} if single else result

    @torch.inference_mode()
    def predict_text(self, text):
        from ..frontend.phrase_plan import encode_text, plan_for_units
        front, arrays = encode_text(text, self.vocabulary)
        features = {'numeric': torch.from_numpy(arrays['numeric']),
                    'char_ids': torch.from_numpy(arrays['char_ids'])}
        self.eval()
        values = self(features)['plan'].cpu().numpy().astype(np.float32)
        return {'frontend': front, 'syllable_plan': values,
                'unit_plan': plan_for_units(front, values)}


def load_planner(checkpoint):
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Unknown phrase planner checkpoint')
    model = PhraseProsodyPlanner(state['config'], state['vocabulary'])
    model.load_state_dict(state['model'])
    model.eval()
    model.normalization = state.get('normalization', {})
    model.checkpoint_step = state['step']
    return model
