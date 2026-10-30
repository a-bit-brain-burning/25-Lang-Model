"""Explicit 24 spectral + 3 aperiodicity contract, with separate F0/VUV controls.

Neural trajectories contain 29 columns, not 27. Spectrum slot zero is the
logarithm of measured waveform mean-square; shape is 23 log-power DCT modes.
"""
from dataclasses import dataclass
import numpy as np

SAMPLE_RATE = 16000
HOP_SAMPLES = 160
CONTROL_POINTS = 8
CHANNEL_NAMES = ['log_f0', 'log_wave_energy'] + [f'shape_{i:02}' for i in range(1, 24)] + ['ap_low', 'ap_mid', 'ap_high', 'vuv_logit']
PARAMETER_DIM = 29
AP_EDGES_HZ = (0, 1000, 4000, 8000)
ENERGY_DEFINITION = '640-sample Hann-window mean-square, centered at frame_index*160; zero padding 320 samples on both sides'

def clip_controls(controls):
    c = np.asarray(controls, dtype=np.float32).copy()
    if c.shape[-1] != 29 or not np.isfinite(c).all():
        raise ValueError('Expected finite (...,29) acoustic controls')
    c[..., 0] = np.clip(c[..., 0], np.log(50), np.log(500))
    c[..., 1] = np.clip(c[..., 1], -40, 20)
    c[..., 2:25] = np.clip(c[..., 2:25], -20, 20)
    c[..., 25:28] = np.clip(c[..., 25:28], -1, 0)
    c[..., 28] = np.clip(c[..., 28], -12, 12)
    return c

def expand_controls(controls, durations):
    """Linear eight-knot interpolation with the historical j/n frame basis."""
    c = np.asarray(controls, dtype=np.float32)
    counts = np.asarray(durations)
    if c.shape != (len(counts), 8, 29) or np.any(counts < 0) or np.any(counts != counts.astype(np.int64)):
        raise ValueError('Invalid controls or noninteger duration frame counts')
    chunks = []
    for knots, n in zip(c, counts.astype(np.int64)):
        if not n:
            continue
        p = np.arange(n, dtype=np.float64) / n * 7
        lo = np.minimum(p.astype(np.int64), 6)
        alpha = p - lo
        chunks.append(knots[lo] * (1-alpha[:, None]) + knots[lo+1] * alpha[:, None])
    if not chunks:
        raise ValueError('No observed acoustic frames')
    return np.concatenate(chunks).astype(np.float32)

def physical_curve(curve):
    """Keep log F0, convert the last neural logit to a voiced probability."""
    c = np.asarray(curve, dtype=np.float32).copy()
    c[..., 28] = 1 / (1 + np.exp(-np.clip(c[..., 28], -60, 60)))
    return c

def dsp_packed(curve):
    """DSP layout is [F0 Hz, spectral24, AP3, binary VUV]."""
    c = clip_controls(curve)
    voiced = c[..., 28] > 0
    c[..., 0] = np.where(voiced, np.exp(c[..., 0]), 0)
    c[..., 28] = voiced
    c[~voiced, 25:28] = 0
    return c

@dataclass
class AcousticParams:
    f0: np.ndarray
    spectral24: np.ndarray
    ap3: np.ndarray
    vuv: np.ndarray
    sample_rate: int = SAMPLE_RATE
    hop_samples: int = HOP_SAMPLES

    def validate(self):
        n = len(self.f0)
        if self.spectral24.shape != (n, 24) or self.ap3.shape != (n, 3) or self.vuv.shape != (n,):
            raise ValueError('Invalid 24+3 plus F0/VUV array shapes')
        if not all(np.isfinite(x).all() for x in [self.f0, self.spectral24, self.ap3, self.vuv]):
            raise ValueError('Nonfinite acoustic parameter')
        if np.any((self.f0 < 0) | (self.f0 > 500)) or np.any((self.ap3 < -1) | (self.ap3 > 0)):
            raise ValueError('Acoustic parameters outside DSP domain')
        if not np.array_equal(self.vuv.astype(bool), self.f0 > 0):
            raise ValueError('F0 and voiced/unvoiced state disagree')
        if np.any((self.spectral24[:, 0] < -40) | (self.spectral24[:, 0] > 20)) or np.any(np.abs(self.spectral24[:, 1:]) > 20):
            raise ValueError('Energy or spectral shape outside DSP domain')
        if self.sample_rate != SAMPLE_RATE or self.hop_samples != HOP_SAMPLES:
            raise ValueError('Prototype supports 16kHz, 10ms frames')
        return self

    def packed(self):
        self.validate()
        return np.column_stack([self.f0, self.spectral24, self.ap3, self.vuv]).astype(np.float32)

    @classmethod
    def from_neural_curve(cls, curve):
        p = dsp_packed(curve)
        return cls(p[:, 0], p[:, 1:25], p[:, 25:28], p[:, 28].astype(bool)).validate()
