"""Real WAV -> compact WORLD shape/F0/AP plus waveform energy, never fake labels.

WORLD is an offline training dependency. It is not required on the target CPU.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import math
import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly
from .schema import AcousticParams, SAMPLE_RATE, HOP_SAMPLES, AP_EDGES_HZ, ENERGY_DEFINITION

def read_audio(path):
    rate, pcm = wavfile.read(Path(path))
    if pcm.ndim != 1:
        raise ValueError('Single-speaker corpus must contain mono WAV')
    if np.issubdtype(pcm.dtype, np.integer):
        audio = pcm.astype(np.float64) / max(abs(np.iinfo(pcm.dtype).min), np.iinfo(pcm.dtype).max)
    else:
        audio = pcm.astype(np.float64)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE//divisor, rate//divisor)
    if not np.isfinite(audio).all() or not len(audio):
        raise ValueError('Empty/nonfinite WAV')
    return np.ascontiguousarray(audio, dtype=np.float64)

def short_time_energy(audio, frame_count=None):
    audio = np.asarray(audio, dtype=np.float64)
    n = math.ceil(len(audio)/HOP_SAMPLES) if frame_count is None else int(frame_count)
    if n < 1 or (n-1)*HOP_SAMPLES >= len(audio):
        raise ValueError('Frame count exceeds actual WAV timeline')
    window = np.hanning(640)
    padded = np.pad(audio, (320, 320), mode='constant')
    return np.asarray([np.sum(padded[i*160:i*160+640]**2 * window)/window.sum() for i in range(n)], dtype=np.float32)

class AcousticExtractor(ABC):
    @abstractmethod
    def extract(self, wav_path):
        """Return AcousticParams in the published physical contract."""

class WorldAcousticExtractor(AcousticExtractor):
    def extract(self, wav_path):
        try:
            import pyworld as pw
        except ImportError as error:
            raise RuntimeError('Install pyworld for offline real-WAV extraction; no synthetic fallback') from error
        audio = read_audio(wav_path)
        f0, times = pw.harvest(audio, 16000, f0_floor=50, f0_ceil=400, frame_period=10)
        f0 = pw.stonemask(audio, f0, times, 16000)
        spectrum = pw.cheaptrick(audio, f0, times, 16000, f0_floor=50, fft_size=1024)
        ap = pw.d4c(audio, f0, times, 16000, fft_size=1024)
        n = math.ceil(len(audio)/160)
        shape = dct(np.log(np.maximum(spectrum[:n], 1e-12)), type=2, norm='ortho', axis=1, workers=1)[:, 1:24] / math.sqrt(spectrum.shape[1])
        energy = short_time_energy(audio, n)
        spectral24 = np.column_stack([np.log(np.maximum(energy, 1e-12)), shape]).astype(np.float32)
        frequencies = np.linspace(0, 8000, ap.shape[1])
        safe = np.clip(ap[:n], .001, 1-1e-12)
        bands = []
        for i, (low, high) in enumerate(zip(AP_EDGES_HZ[:-1], AP_EDGES_HZ[1:])):
            mask = (frequencies >= low) & ((frequencies <= high) if i == 2 else (frequencies < high))
            bands.append(10*np.log10(np.mean(safe[:, mask]**2, axis=1))/60)
        return AcousticParams(f0[:n].astype(np.float32), spectral24, np.column_stack(bands).astype(np.float32), f0[:n]>0).validate()

def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    params = WorldAcousticExtractor().extract(args.wav)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = params.packed()
    target[:, 0] = np.log(np.maximum(target[:, 0], 1))
    np.savez_compressed(args.output, f0=params.f0, spectral24=params.spectral24, ap3=params.ap3, vuv=params.vuv, target_curve=target)
    print(json.dumps({'wav': str(args.wav), 'output': str(args.output), 'frames': len(params.f0), 'energy_definition': ENERGY_DEFINITION}))

if __name__ == '__main__':
    main()
