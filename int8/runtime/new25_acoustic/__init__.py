"""New25 acoustic feature prediction; a waveform synthesizer is not bundled."""

from .model import Acoustic, FP32GRUCell, LocalBlock

__all__ = ["Acoustic", "FP32GRUCell", "LocalBlock", "AcousticPredictor"]


def __getattr__(name):
    # Keep `python -m new25_acoustic.inference` free of double-import warnings.
    if name == "AcousticPredictor":
        from .inference import AcousticPredictor

        return AcousticPredictor
    raise AttributeError(name)
