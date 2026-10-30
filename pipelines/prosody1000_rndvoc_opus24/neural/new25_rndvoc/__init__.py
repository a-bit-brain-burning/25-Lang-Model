"""Selected 25k RNDVoC-Lite offline reference inference.

Lazy imports keep package metadata inspection free of model construction.
"""

__all__ = ["RNDVoCVocoder"]


def __getattr__(name):
    if name == "RNDVoCVocoder":
        from .inference import RNDVoCVocoder
        return RNDVoCVocoder
    raise AttributeError(name)
