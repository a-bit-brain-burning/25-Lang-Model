"""The existing New25 raw-log-mel contract, including its exact frame centres."""
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

SR, FFT, HOP, PAD = 22050, 1024, 256, 384


class Frontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(FFT))
        self.register_buffer('mel_basis', torch.from_numpy(librosa_mel(sr=SR, n_fft=FFT, n_mels=80, fmin=0, fmax=8000)))
        self.register_buffer('loss_basis', torch.from_numpy(librosa_mel(sr=SR, n_fft=FFT, n_mels=80, fmin=0, fmax=SR/2)))

    def spectrum(self, wav):
        assert wav.ndim == 2 and wav.shape[-1] % HOP == 0 and wav.shape[-1] > PAD
        x = F.pad(wav[:, None], (PAD, PAD), mode='reflect')[:, 0]
        return torch.stft(x, FFT, HOP, FFT, self.window, center=False, return_complex=True)

    def mel(self, wav, for_loss=False):
        spec = self.spectrum(wav)
        mag = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-9)
        return ((self.loss_basis if for_loss else self.mel_basis) @ mag).clamp_min(1e-5).log()


def aligned_istft(spec, window):
    """Exact OLA for reflect384/center=False analysis; output length = T*256.

    Native torch.istft(center=True) would move this grid by half a hop and
    return one hop fewer. This deterministic boundary operation adds no weights.
    """
    b, f, t = spec.shape
    assert f == FFT//2+1
    frames = torch.fft.irfft(spec, n=FFT, dim=1) * window[None, :, None]
    full_length = FFT + (t - 1) * HOP
    wav = F.fold(frames, (1, full_length), (1, FFT), stride=(1, HOP))[:, 0, 0]
    weights = window.square()[None, :, None].expand(1, -1, t)
    envelope = F.fold(weights, (1, full_length), (1, FFT), stride=(1, HOP))[0, 0, 0]
    wav = wav[:, PAD:PAD+t*HOP]
    envelope = envelope[PAD:PAD+t*HOP]
    assert bool((envelope > 1e-5).all())
    return wav / envelope[None]
