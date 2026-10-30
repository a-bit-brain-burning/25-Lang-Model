"""Offline HiFi-GAN V2 waveform synthesis for normalized NEW25 features.

The generator is derived from the official HiFi-GAN implementation. See
third_party/hifi_gan/ for the upstream license and attribution.
"""

__all__ = ["HiFiGANVocoder"]


def __getattr__(name):
    # Keep ``python -m new25_hifigan.inference`` free of eager module imports.
    if name == "HiFiGANVocoder":
        from .inference import HiFiGANVocoder

        return HiFiGANVocoder
    raise AttributeError(name)
