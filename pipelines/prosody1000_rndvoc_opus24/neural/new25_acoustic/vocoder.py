"""The reserved waveform-synthesis interface. This module contains no vocoder."""
from typing import Any, Protocol, Tuple

import torch


SAMPLE_RATE = 16000
HOP_SAMPLES = 160


class Vocoder(Protocol):
    def __call__(self, normalized: torch.Tensor, physical: torch.Tensor,
                 lang: torch.Tensor, state: Any = None) -> Tuple[torch.Tensor, Any]:
        """Return mono float PCM [B,T*160] and the next opaque stream state.

        Both conditions have shape [B,T,31|39]. `physical` uses the physical
        contract in docs/VOCODER_INTERFACE.md; `normalized` is computed from
        those physical values using this release's per-language statistics.
        The caller resets state between utterances and whenever language or
        acoustic mode changes. A streaming implementation carries state across
        chunks of the same utterance and emits exactly 160 samples per frame.
        """
        ...


def validate_condition(normalized, physical, lang):
    if normalized.ndim != 3 or physical.shape != normalized.shape:
        raise ValueError('Conditions must both have shape [B,T,D]')
    if normalized.shape[0] < 1 or normalized.shape[1] < 1 or normalized.shape[2] not in (31, 39):
        raise ValueError('Nonempty 31D or 39D conditions required')
    if not normalized.is_floating_point() or not physical.is_floating_point():
        raise ValueError('Conditions must have floating point dtype')
    if lang.shape != (normalized.shape[0],) or lang.dtype != torch.long:
        raise ValueError('Language must be int64 [B]')
    if not (normalized.device == physical.device == lang.device):
        raise ValueError('All interface tensors must use the same device')
    if not bool(torch.isfinite(normalized).all() and torch.isfinite(physical).all()):
        raise ValueError('Nonfinite acoustic condition')
    if bool(((lang < 0) | (lang >= 25)).any()):
        raise ValueError('Language id outside this release vocabulary')
    expected = torch.where(lang == 24, 31, 39)
    if bool((expected != normalized.shape[-1]).any()):
        raise ValueError('Language and acoustic mode disagree')
    if bool(((physical[..., -1] < 0) | (physical[..., -1] > 1)).any()):
        raise ValueError('Physical voicing must be a probability, not a logit')


def validate_output(pcm, normalized):
    expected = (normalized.shape[0], normalized.shape[1] * HOP_SAMPLES)
    if pcm.shape != expected or not pcm.is_floating_point():
        raise ValueError('Vocoder output must be floating mono PCM [B,T*160]')
    if not bool(torch.isfinite(pcm).all()):
        raise ValueError('Nonfinite PCM output')


class VocoderStub:
    """A deliberate integration point; it never fabricates silent audio."""
    def __call__(self, normalized, physical, lang, state=None):
        validate_condition(normalized, physical, lang)
        raise NotImplementedError('Attach a trained vocoder satisfying docs/VOCODER_INTERFACE.md')
