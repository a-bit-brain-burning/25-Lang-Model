"""Optional original Opus-BWE24 and explicit16k waveform insertion slots.

Importing this package does not compile C code or construct a model.
"""
__all__ = ["OpusBWE24", "RNDVoCWithBWE24", "PCM16Stage", "CallablePCM16Stage", "run_pcm16_stages"]


def __getattr__(name):
    if name in ("OpusBWE24", "RNDVoCWithBWE24"):
        from . import bwe24
        return getattr(bwe24, name)
    if name in ("PCM16Stage", "CallablePCM16Stage", "run_pcm16_stages"):
        from . import chain
        return getattr(chain, name)
    raise AttributeError(name)
