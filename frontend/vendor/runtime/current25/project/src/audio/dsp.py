"""AcousticParams -> waveform using the existing C source/filter DSP.

Offline training uses log-F0 and VUV logits; the C ABI uses F0 Hz and
binary voicing. Spectrum slot zero is log short-time waveform mean-square
energy. No reference recording is accessed during synthesis.
"""
import ctypes
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.fft import idct
from scipy.io import wavfile
from scipy.signal import butter

from ..common import PROJECT, save_json, sha256

RATE = 16000
HOP = 160
CHANNELS = 29


class Stats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in
                ['excitation_cpu_s', 'filter_build_cpu_s', 'filtering_cpu_s', 'pcm_cpu_s']] + [
        (name, ctypes.c_uint64) for name in ['bank_mac', 'fir_mac', 'basis_mac',
        'recursion_mac', 'generated_samples', 'clipped_pcm', 'exp_calls', 'sqrt_calls', 'context_bytes']]


def clip_curve(curve):
    c = np.asarray(curve, dtype=np.float32).copy()
    if c.ndim != 2 or c.shape[1] != CHANNELS or not len(c) or not np.isfinite(c).all():
        raise ValueError('Expected nonempty finite [frames,29] acoustic trajectory')
    c[:, 0] = np.clip(c[:, 0], np.log(50), np.log(500))
    c[:, 1] = np.clip(c[:, 1], -40, 20)
    c[:, 2:25] = np.clip(c[:, 2:25], -20, 20)
    c[:, 25:28] = np.clip(c[:, 25:28], -1, 0)
    c[:, 28] = np.clip(c[:, 28], -12, 12)
    return c


def to_physical(curve, vuv='logit'):
    c = clip_curve(curve)
    if vuv not in ('logit', 'probability'):
        raise ValueError('VUV representation must be explicit')
    voiced = c[:, 28] > (0 if vuv == 'logit' else .5)
    c[:, 0] = np.where(voiced, np.exp(c[:, 0]), 0)
    c[:, 28] = voiced
    c[~voiced, 25:28] = 0
    return np.ascontiguousarray(c, dtype=np.float32)


def fixed_tables(taps=128):
    modes = np.zeros((23, 513))
    modes[np.arange(23), np.arange(1, 24)] = np.sqrt(513)
    log_shape = idct(modes, type=2, norm='ortho', axis=1)
    cepstrum = np.fft.irfft(.5 * log_shape, n=1024, axis=1)
    cepstrum[:, 1:512] *= 2
    basis = np.ascontiguousarray(cepstrum[:, :taps].T, dtype=np.float32)
    bank = np.asarray([butter(2, x, fs=RATE, output='sos')[0][[0, 1, 2, 4, 5]]
                       for x in (1000, 4000)], dtype=np.float32)
    return basis, bank


def build_native(output=None):
    output = Path(output or PROJECT / 'outputs/native/libcompact_dsp.so').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('native_stream.c')
    core = source.with_name('native.c')
    if output.exists() and output.stat().st_mtime >= max(source.stat().st_mtime, core.stat().st_mtime):
        return output
    command = ['gcc', '-O3', '-std=c99', '-shared', '-fPIC', '-fno-fast-math',
               '-ffp-contract=off', str(source), '-lm', '-o', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    save_json(output.parent / 'build.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'source_sha256': sha256(source),
        'core_sha256': sha256(core), 'algorithm_origin': 'existing tetts C DSP, copied without algorithm changes'})
    if result.returncode:
        raise RuntimeError('DSP build failed: ' + result.stderr)
    return output


class DSPAdapter:
    def __init__(self, taps=128, gain=1., seed=1729, library=None):
        if taps not in (64, 128, 256):
            raise ValueError('Supported FIR lengths:64,128,256')
        self.taps, self.gain, self.seed = taps, float(gain), int(seed)
        self.basis, self.bank = fixed_tables(taps)
        self.library = ctypes.CDLL(str(library or build_native()))
        pf = ctypes.POINTER(ctypes.c_float)
        self.library.tetts_render.argtypes = [pf, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            pf, pf, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_float, pf,
            ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(Stats)]
        self.library.tetts_render.restype = ctypes.c_int
        self.library.tetts_stream_bytes.restype = ctypes.c_size_t
        self.library.tetts_stream_initialize.argtypes = [ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, pf, pf, ctypes.c_uint32, ctypes.c_float]
        self.library.tetts_stream_initialize.restype = ctypes.c_int
        self.library.tetts_stream_frame.argtypes = [ctypes.c_void_p, pf, pf, ctypes.c_int, pf,
            ctypes.POINTER(ctypes.c_int16)]
        self.library.tetts_stream_frame.restype = ctypes.c_int

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def render(self, curve, vuv='logit', sample_count=None):
        packed = to_physical(curve, vuv)
        count = int(sample_count if sample_count is not None else len(packed) * HOP)
        if count < 1 or count > 480000:
            raise ValueError('Waveform duration must be between one sample and30seconds')
        output, pcm, stats = np.empty(count, np.float32), np.empty(count, np.int16), Stats()
        start = time.perf_counter()
        result = self.library.tetts_render(self.pointer(packed), len(packed), 24, 3,
            self.pointer(self.basis), self.pointer(self.bank), self.taps, count, self.seed,
            self.gain, self.pointer(output), pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.byref(stats))
        if result:
            raise RuntimeError(f'C DSP render failed:{result}')
        metrics = {name: getattr(stats, name) for name, _ in stats._fields_}
        metrics.update({'wall_seconds': time.perf_counter() - start, 'seconds': count / RATE,
                        'sample_rate': RATE, 'taps': self.taps, 'gain': self.gain,
                        'seed': self.seed})
        metrics['RTF'] = metrics['wall_seconds'] / metrics['seconds']
        return pcm, metrics

    def blocks(self, curve, vuv='logit', sample_count=None):
        """Yield one PCM hop; native state stays bounded independently of sentence length."""
        packed = to_physical(curve, vuv)
        count = int(sample_count if sample_count is not None else len(packed) * HOP)
        if count < 1 or count > 480000:
            raise ValueError('Invalid sample count')
        memory = ctypes.create_string_buffer(self.library.tetts_stream_bytes())
        result = self.library.tetts_stream_initialize(memory, 24, 3, self.taps,
            self.pointer(self.basis), self.pointer(self.bank), self.seed, self.gain)
        if result:
            raise RuntimeError(f'C DSP initialization failed:{result}')
        for offset in range(0, count, HOP):
            first = min(offset // HOP, len(packed) - 1)
            second = min(first + 1, len(packed) - 1)
            n = min(HOP, count - offset)
            output, pcm = np.empty(n, np.float32), np.empty(n, np.int16)
            result = self.library.tetts_stream_frame(memory, self.pointer(packed[first]),
                self.pointer(packed[second]), n, self.pointer(output),
                pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))
            if result:
                raise RuntimeError(f'C DSP frame failed:{result}')
            yield pcm

    def memory(self):
        return {'native_stream_state_bytes': int(self.library.tetts_stream_bytes()),
                'two_parameter_frames_bytes': 2 * CHANNELS * 4,
                'one_hop_float_and_pcm_bytes': HOP * 6,
                'basis_bytes': self.basis.nbytes, 'bank_bytes': self.bank.nbytes,
                'basis_and_bank_can_be_ROM': True, 'DSP_lookahead_ms': 10,
                'host_Python_runtime_and_whole_trajectory_excluded': True}


def write_wav(path, pcm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, RATE, np.asarray(pcm, dtype=np.int16))
