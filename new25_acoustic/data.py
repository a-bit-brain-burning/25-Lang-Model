"""Frozen prepared acoustic arrays, normalization, and balanced sampling.

The PCM reader is used only by corpus preparation to verify the source timeline.
Acoustic training never opens source audio.
"""
import copy
import json
import random
import wave
from pathlib import Path

import numpy as np
import torch

from .contracts import LANGUAGES, HOP, SCHEMA, safe_path, sha256


def read_pcm(path, frames):
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError('Expected native 16 kHz mono PCM16')
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype('float32') / 32768
    if len(x) != frames * HOP:
        raise ValueError('PCM/frame timeline mismatch; do not silently shift')
    return x


def frame_layout(durations):
    ids = np.repeat(np.arange(len(durations)), durations)
    pos = []
    for n in durations:
        if n:
            j = np.arange(n, dtype=np.float32)
            pos.append(np.stack([j / max(n - 1, 1), np.full(n, 1 / n), j == 0, j == n - 1], -1))
    if not pos:
        raise ValueError('Empty timeline')
    return ids.astype('int64'), np.concatenate(pos).astype('float32')


class Corpus:
    def __init__(self, prepared):
        self.prepared = Path(prepared).resolve()
        self.receipt = json.loads((self.prepared / 'COMPLETE.json').read_text())
        if self.receipt.get('status') != 'PREPARED' or self.receipt.get('schema') != SCHEMA:
            raise ValueError('Expected a completed compatible prepared corpus')
        metadata = self.receipt['metadata_sha256']
        if not {'manifest.jsonl', 'normalization.json', 'vocabulary.json'} <= metadata.keys():
            raise ValueError('Prepared receipt does not bind required metadata')
        for name, digest in metadata.items():
            if sha256(safe_path(self.prepared, name)) != digest:
                raise ValueError('Prepared metadata changed ' + name)
        self.manifest = self.prepared / 'manifest.jsonl'
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.norm = json.loads((self.prepared / 'normalization.json').read_text())
        self.vocab = json.loads((self.prepared / 'vocabulary.json').read_text())
        if sorted(self.vocab.values()) != list(range(len(self.vocab))):
            raise ValueError('Vocabulary IDs must be contiguous')
        self.verified = {}
        self.by_split = {s: {language: [] for language in LANGUAGES} for s in ('train', 'validation')}
        for i, row in enumerate(self.rows):
            self.by_split[row['split']][row['language']].append(i)
        if any(not ids for languages in self.by_split.values() for ids in languages.values()):
            raise ValueError('Every language needs both train and validation')

    def item(self, i):
        row = self.rows[i]
        path = safe_path(self.prepared, row['cache'])
        self.verify(path, row['cache_sha256'])
        with np.load(path, allow_pickle=False) as arrays:
            item = {key: arrays[key] for key in arrays.files}
        item['row'] = row
        return item

    def verify(self, path, digest):
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if self.verified.get(str(path)) != identity:
            if sha256(path) != digest:
                raise ValueError('Frozen payload changed: ' + str(path))
            self.verified[str(path)] = identity

    def normalized(self, item):
        norm = self.norm[item['row']['language']]
        curve = item['curve'].copy()
        curve[~item['f0_valid'], 0] = norm['mean'][0]
        return (curve - np.asarray(norm['mean'], np.float32)) / np.asarray(norm['std'], np.float32)

    def tensor_item(self, i, device):
        item = self.item(i)
        row = item['row']
        norm = self.norm[row['language']]
        frame_phone, position = frame_layout(item['durations'])
        item['frame_phone'] = frame_phone
        item['position'] = position
        item['target'] = self.normalized(item)
        item['phonetic'] = (item['phonetic'] - np.array(norm['phonetic_mean'], np.float32)) / np.array(norm['phonetic_std'], np.float32)
        item['context'] = (item['context'] - np.array(norm['context_mean'], np.float32)) / np.array(norm['context_std'], np.float32)
        result = {key: torch.from_numpy(value.copy()).to(device)[None]
                  for key, value in item.items() if isinstance(value, np.ndarray)}
        result['lang'] = torch.tensor([LANGUAGES.index(row['language'])], device=device)
        result['row'] = row
        result['norm'] = norm
        return result


def denormalize(pred, norm):
    mean = pred.new_tensor(norm['mean'])
    std = pred.new_tensor(norm['std'])
    raw = pred.float() * std + mean
    # The last acoustic head predicts a voice logit; physical output uses p.
    return torch.cat([raw[:, :, :-1], torch.sigmoid(pred[:, :, -1:])], -1)


class BalancedSchedule:
    """One shuffled epoch per language, round-robin; checkpoint the exact state."""
    def __init__(self, corpus, seed):
        self.rng = random.Random(seed)
        self.cursor = 0
        self.ids = {language: list(ids) for language, ids in corpus.by_split['train'].items()}
        self.offset = {language: 0 for language in LANGUAGES}
        for ids in self.ids.values():
            self.rng.shuffle(ids)

    def next(self):
        language = LANGUAGES[self.cursor % len(LANGUAGES)]
        self.cursor += 1
        if self.offset[language] == len(self.ids[language]):
            self.offset[language] = 0
            self.rng.shuffle(self.ids[language])
        index = self.ids[language][self.offset[language]]
        self.offset[language] += 1
        return index

    def next_batch(self, size):
        language = LANGUAGES[self.cursor % len(LANGUAGES)]
        self.cursor += 1
        result = []
        for _ in range(size):
            if self.offset[language] == len(self.ids[language]):
                self.offset[language] = 0
                self.rng.shuffle(self.ids[language])
            result.append(self.ids[language][self.offset[language]])
            self.offset[language] += 1
        return result

    def state_dict(self):
        return copy.deepcopy({'rng': self.rng.getstate(), 'cursor': self.cursor,
                              'ids': self.ids, 'offset': self.offset})

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])
        self.cursor = state['cursor']
        self.ids = copy.deepcopy(state['ids'])
        self.offset = copy.deepcopy(state['offset'])
