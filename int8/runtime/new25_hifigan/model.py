"""Inference-only official HiFi-GAN generator and trained physical adapter.

Generator/ResBlock math and state names follow jik876/hifi-gan, with upstream
weight normalization already folded into the published convolution weights.
No discriminators, training losses, plotting helpers, or pretrained weights are
needed. Upstream license and attribution live in third_party/hifi_gan/.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1
GENERATOR_CONFIG = {
    "resblock": "1",
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 128,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "num_mels": 80,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "sampling_rate": 22050,
    "fmin": 0,
    "fmax": 8000,
}


def validate_generator_config(config):
    """Require the exact published VCTK V2 architecture/frontend contract."""
    if not isinstance(config, Mapping):
        raise ValueError("Generator config must be a JSON object")
    for name, expected in GENERATOR_CONFIG.items():
        if config.get(name) != expected:
            raise ValueError(f"Unsupported generator config {name}: expected {expected!r}")
    return dict(config)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    """Official residual block with folded weight normalization."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                      padding=get_padding(kernel_size, d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=get_padding(kernel_size, 1))
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """Published 925,985-parameter generator; input [B,80,M], output [B,1,256M]."""

    def __init__(self, config=None):
        super().__init__()
        self.config = validate_generator_config(GENERATOR_CONFIG if config is None else config)
        h = self.config
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.conv_pre = nn.Conv1d(80, h["upsample_initial_channel"], 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            self.ups.append(nn.ConvTranspose1d(
                h["upsample_initial_channel"] // (2 ** i),
                h["upsample_initial_channel"] // (2 ** (i + 1)), k, u,
                padding=(k - u) // 2,
            ))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"]):
                self.resblocks.append(ResBlock1(ch, k, d))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        # Upstream deliberately uses the functional default slope (0.01) here.
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)


class Block(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.dw = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.up = nn.Linear(width, intermediate)
        self.down = nn.Linear(intermediate, width)
        self.gamma = nn.Parameter(torch.full((width,), 1 / 6))

    def forward(self, x, mask):
        return (x + self.gamma * self.down(F.gelu(self.up(self.norm(
            self.dw(x.transpose(1, 2)).transpose(1, 2)
        ))))) * mask


class PhysicalMelAdapter(nn.Module):
    """The 1,902,992-parameter noncausal 31/39-to-80-mel training adapter."""

    def __init__(self):
        super().__init__()
        w = 192
        self.inputs = nn.ModuleDict({str(d): nn.Conv1d(d, w, 7, padding=3) for d in (31, 39)})
        self.language = nn.Embedding(25, w)
        self.norm = nn.LayerNorm(w, eps=1e-6)
        self.blocks = nn.ModuleList([Block(w, 768) for _ in range(6)])
        self.head = nn.Linear(w, 80)

    def forward(self, x, language, frames, mask=None, context_frames=0):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
        mask = mask[:, :, None]
        z = self.norm(self.inputs[str(x.shape[-1])](x.transpose(1, 2)).transpose(1, 2)
                      + self.language(language)[:, None]) * mask
        for layer in self.blocks:
            z = layer(z, mask)
        z = self.head(z).float().transpose(1, 2)
        # Preserve original interpolation, including its left-edge clamping.
        positions = ((torch.arange(frames, device=x.device, dtype=torch.float32) * 256 + 128)
                     * 100 / 22050 - .5 + context_frames)
        lo = positions.floor().long().clamp(0, x.shape[1] - 1)
        hi = (lo + 1).clamp(max=x.shape[1] - 1)
        a = (positions - positions.floor()).clamp(0, 1)
        return z[:, :, lo] * (1 - a) + z[:, :, hi] * a
