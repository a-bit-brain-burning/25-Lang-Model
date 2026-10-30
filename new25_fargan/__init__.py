"""Selected FARGAN 4750 and physical-feature adapter, offline CPU inference."""

__all__ = ["FARGANVocoder"]


def __getattr__(name):
    if name == "FARGANVocoder":
        from .inference import FARGANVocoder
        return FARGANVocoder
    raise AttributeError(name)
