"""Selected FARGAN 4750 with persistent physical-feature and waveform streams."""

__all__ = ["FARGANVocoder"]


def __getattr__(name):
    if name == "FARGANVocoder":
        from .inference import FARGANVocoder
        return FARGANVocoder
    raise AttributeError(name)
