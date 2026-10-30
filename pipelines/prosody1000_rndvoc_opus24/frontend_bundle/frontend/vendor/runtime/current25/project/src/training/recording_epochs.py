"""Deterministic complete-recording epochs without padding or dropped tails.

Indices refer to a caller-frozen, ordered training manifest. Checkpoint lineage
must bind that manifest (or its ordered IDs) as well as order_sha256. The module
does not import Torch, inspect recordings, or touch global NumPy RNG state.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import numbers

import numpy as np

VERSION = 'complete_recording_epochs_pcg64_v1'


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class RecordingEpochScheduler:
    """One permutation per epoch; start_step is completed optimizer updates.

    For N=5,B=2, one epoch yields lengths2,2,1. The next epoch starts a new
    minibatch, so no repeated samples are used to fill a tail. Changing N, B,
    seed or epochs changes the order identity and must reject incompatible
    checkpoint resume rather than silently continue with a new schedule.
    """

    n_records: int
    batch_size: int
    seed: int
    epochs: int

    def __post_init__(self):
        for name, minimum in (('n_records', 1), ('batch_size', 1), ('seed', 0), ('epochs', 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))

    @property
    def steps_per_epoch(self):
        return (self.n_records + self.batch_size - 1) // self.batch_size

    @property
    def total_steps(self):
        return self.epochs * self.steps_per_epoch

    @property
    def total_exposures(self):
        return self.epochs * self.n_records

    def epoch_indices(self, epoch):
        """Fresh independent zero-based epoch order; no prefix replay needed."""
        epoch = _integer(epoch, 'epoch')
        if epoch >= self.epochs:
            raise ValueError('Epoch outside frozen schedule')
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, epoch])))
        return generator.permutation(self.n_records).astype(np.int64, copy=False)

    def batches(self, start_step=0):
        """Yield index arrays after exactly start_step completed batches.

        Resuming at total_steps yields no batches. Mid-epoch resume generates
        only that epoch's order and starts at its exact batch cursor. Arrays
        are independent copies, so caller mutation cannot corrupt later steps.
        """
        start_step = _integer(start_step, 'start_step')
        if start_step > self.total_steps:
            raise ValueError('Resume step outside frozen schedule')
        first_epoch, first_batch = divmod(start_step, self.steps_per_epoch)
        for epoch in range(first_epoch, self.epochs):
            order = self.epoch_indices(epoch)
            begin = first_batch if epoch == first_epoch else 0
            for batch in range(begin, self.steps_per_epoch):
                yield order[batch * self.batch_size:(batch + 1) * self.batch_size].copy()

    @cached_property
    def order_sha256(self):
        """Versioned schedule identity; hashes do not depend on native endian."""
        metadata = {'version': VERSION, 'n_records': self.n_records,
                    'batch_size': self.batch_size, 'seed': self.seed, 'epochs': self.epochs,
                    'tail_policy': 'short_batch_no_drop_no_repetition',
                    'permutation': 'PCG64(SeedSequence([seed,zero_based_epoch]))'}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode() + b'\n')
        for epoch in range(self.epochs):
            digest.update(self.epoch_indices(epoch).astype('<i8', copy=False).tobytes(order='C'))
        return digest.hexdigest()


def scheduler(n_records, batch_size, seed, epochs):
    """Constructor spelling used by the full-data training orchestrator."""
    return RecordingEpochScheduler(n_records, batch_size, seed, epochs)
