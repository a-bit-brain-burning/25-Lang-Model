"""Explicit full-utterance16k PCM stages between a vocoder and BWE24.

This module only defines the insertion contract. No denoiser, trained acoustic
processor or identity audio is fabricated; callers supply their actual module.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Protocol

import numpy as np


class PCM16Stage(Protocol):
    """Each call consumes and returns mono np.int16[N] at16000Hz.

    A stage owns its internal state and compensates its own padding/delay.
    ``reset()`` is called once before processing every independent utterance.
    Its report must be a finite JSON-serializable dict; errors stop the chain.
    """
    name: str
    input_sample_rate: int
    output_sample_rate: int
    learned_parameters: int | None

    def reset(self) -> None: ...

    def process_pcm16(self, pcm: np.ndarray, *, sample_rate: int = 16000) -> tuple[np.ndarray, dict]: ...


class CallablePCM16Stage:
    """Adapt a caller-supplied real processor without guessing its API.

    ``process`` accepts (pcm, sample_rate=16000) and returns (pcm, report).
    Supply the actual reset callback for a stateful processor. ``reset=None``
    explicitly declares the callback stateless or internally self-resetting.
    No inference occurs in this adapter's constructor.
    """
    input_sample_rate = 16000
    output_sample_rate = 16000

    def __init__(self, name: str, process: Callable, *, reset: Callable | None = None,
                 learned_parameters: int | None = None):
        if not isinstance(name, str) or not name.strip() or not callable(process):
            raise ValueError("Stage requires a nonempty name and an actual process callable")
        if reset is not None and not callable(reset):
            raise ValueError("reset must be a callable or explicit None")
        if learned_parameters is not None and (type(learned_parameters) is not int or learned_parameters < 0):
            raise ValueError("learned_parameters must be a nonnegative integer or unknown None")
        self.name, self._process, self._reset = name, process, reset
        self.learned_parameters = learned_parameters

    def reset(self):
        if self._reset is not None:
            self._reset()

    def process_pcm16(self, pcm, *, sample_rate=16000):
        return self._process(pcm, sample_rate=sample_rate)


def _pcm(value, expected_samples=None):
    if (not isinstance(value, np.ndarray) or value.dtype != np.int16
            or value.ndim != 1 or not len(value)):
        raise ValueError("Insertion stages must return a nonempty mono np.int16[N] vector")
    if expected_samples is not None and len(value) != expected_samples:
        raise ValueError("Insertion stage changed length; compensate its delay/padding inside its adapter")
    return np.ascontiguousarray(value)


def _fingerprint(pcm):
    return hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest()


def run_pcm16_stages(pcm, stages):
    """Run supplied modules serially, in list order; preserve the16k timeline.

    Empty stages preserve the input values. No resampling, gain correction,
    automatic delay trimming or silent fallback is applied here. Use separate
    stage/backend instances for concurrently processed utterances.
    """
    current = _pcm(pcm).copy()
    original_samples = len(current)
    reports, names = [], set()
    for index, stage in enumerate(tuple(stages)):
        name = getattr(stage, "name", None)
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Insertion stages need distinct nonempty names")
        names.add(name)
        if (getattr(stage, "input_sample_rate", None) != 16000
                or getattr(stage, "output_sample_rate", None) != 16000):
            raise ValueError("Each insertion stage must explicitly declare16k input and output")
        if not callable(getattr(stage, "reset", None)) or not callable(getattr(stage, "process_pcm16", None)):
            raise ValueError("Insertion stage must implement reset and process_pcm16")
        source_hash = _fingerprint(current)
        stage.reset()
        output = stage.process_pcm16(current.copy(), sample_rate=16000)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("Insertion stage must return (PCM16, report)")
        processed, report = output
        if not isinstance(report, dict):
            raise ValueError("Insertion stage report must be a dict")
        json.dumps(report, allow_nan=False)
        current = _pcm(processed, original_samples).copy()
        count = getattr(stage, "learned_parameters", None)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Invalid inserted module parameter count")
        reports.append({"index": index, "name": name, "input_sample_rate": 16000,
                        "output_sample_rate": 16000, "input_samples": original_samples,
                        "output_samples": len(current), "input_pcm16_sha256": source_hash,
                        "output_pcm16_sha256": _fingerprint(current),
                        "reset_for_this_utterance": True, "learned_parameters": count,
                        "module_report": report})
    return current, reports
