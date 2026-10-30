"""Pinned HiFi-GAN mel and SciPy polyphase resampling reference math.

Synthesis consumes predicted mel from the physical adapter and does not require
reference audio. ``Mel`` and ``to_native`` support optional true-mel comparison.
Resampling uses scipy.signal.resample_poly's default Kaiser window and constant
padding exactly as in the fine-tuning source; it is noncausal.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F

SR = 22050
HOP = 256
FFT = 1024


def mel_matrix():
    """Slaney mel scale and area normalization, 80 bins over 0..8000 Hz."""
    def hz_mel(f):
        f = np.asarray(f, dtype=np.float64)
        m = f / (200 / 3)
        a = f >= 1000
        m[a] = 15 + np.log(f[a] / 1000) / (np.log(6.4) / 27)
        return m

    def mel_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = (200 / 3) * m
        a = m >= 15
        f[a] = 1000 * np.exp((np.log(6.4) / 27) * (m[a] - 15))
        return f

    freq = np.fft.rfftfreq(FFT, 1 / SR)
    points = mel_hz(np.linspace(hz_mel(np.array([0.]))[0], hz_mel(np.array([8000.]))[0], 82))
    diff = np.diff(points)
    ramps = points[:, None] - freq[None, :]
    weights = np.maximum(0, np.minimum(-ramps[:-2] / diff[:-1, None], ramps[2:] / diff[1:, None]))
    weights *= 2 / (points[2:] - points[:-2])[:, None]
    return weights.astype(np.float32)


class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("basis", torch.from_numpy(mel_matrix()), persistent=False)
        self.register_buffer("window", torch.hann_window(FFT), persistent=False)

    def forward(self, y):
        if y.ndim != 2 or y.shape[-1] <= 384 or not torch.isfinite(y).all():
            raise ValueError("Mel requires finite [B,N] native audio with N > 384")
        with torch.autocast(device_type=y.device.type, enabled=False):
            y = F.pad(y.float().unsqueeze(1), (384, 384), mode="reflect").squeeze(1)
            z = torch.stft(y, FFT, HOP, FFT, self.window.float(), center=False,
                           normalized=False, onesided=True, return_complex=True)
            magnitude = (z.real.square() + z.imag.square() + 1e-9).sqrt()
            return (self.basis.float() @ magnitude).clamp_min(1e-5).log()


def _pcm(value):
    array = np.asarray(value)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("PCM must be a nonempty finite numeric vector")
    with np.errstate(over="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("PCM values must fit in FP32")
    return array


def sample_layout(physical_frames):
    """Return exact (16-kHz samples, unpadded native samples, native mel frames)."""
    if type(physical_frames) is not int or physical_frames < 1:
        raise ValueError("physical_frames must be a positive integer")
    length = physical_frames * 160
    native_length = (length * 441 + 319) // 320
    mel_frames = (native_length + HOP - 1) // HOP
    return length, native_length, mel_frames


def to_native(pcm16):
    y = resample_poly(_pcm(pcm16), 441, 320).astype(np.float32)
    length = len(y)
    y = np.pad(y, (0, (-len(y)) % HOP))
    return y, length


def to_16k(pcm, length):
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    y = resample_poly(_pcm(pcm), 320, 441).astype(np.float32)
    if len(y) < length:
        raise ValueError("Unexpected resampling length loss")
    return y[:length]
