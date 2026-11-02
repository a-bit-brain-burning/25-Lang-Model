"""Persistent original centered-k7 physical-to-FARGAN adapter.

Each layer waits for its three real future frames. At EOS only, its original
zero padding becomes available. Pointwise blocks run once per valid frame;
Conv1d evaluates a bounded halo around each incoming batch using the original
PyTorch operator. This keeps the original weights, statistics and receptive
field; it does not make the trained adapter causal.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


class _LayerCursor:
    def __init__(self, conv, finish):
        if (conv.kernel_size, conv.padding, conv.stride, conv.dilation) != (
                (7,), (3,), (1,), (1,)):
            raise ValueError("Expected the released centered k7 adapter convolution")
        self.conv, self.finish = conv, finish
        self.buffer = None
        self.start = self.received = self.emitted = self.max_buffer = 0
        self.finished = False

    def push(self, x, final=False):
        if self.finished:
            raise ValueError("Adapter layer is already finished")
        if self.buffer is None:
            self.buffer = x.clone()
        elif x.shape[1]:
            self.buffer = torch.cat((self.buffer, x), dim=1)
        self.received += int(x.shape[1])
        self.max_buffer = max(self.max_buffer, int(self.buffer.shape[1]))
        end = self.received if final else max(self.emitted, self.received - 3)
        self.finished = bool(final)
        if end == self.emitted:
            return x.new_empty((1, 0, self.conv.out_channels))
        left = max(0, self.emitted - 3)
        if left < self.start:
            raise RuntimeError("Adapter convolution history was released early")
        right = min(self.received, end + 3)
        window = self.buffer[:, left-self.start:right-self.start]
        convolved = self.conv(window.transpose(1, 2)).transpose(1, 2)
        lo, hi = self.emitted-left, end-left
        out = self.finish(convolved[:, lo:hi], window[:, lo:hi])
        self.emitted = end
        keep = max(0, end - 3)
        self.buffer = self.buffer[:, keep-self.start:].clone()
        self.start = keep
        return out


class PhysicalFarganAdapterStream:
    """One independent request, B=1 CPU FP32, physical frames -> 20D features."""

    def __init__(self, adapter, language_id, dimension):
        if dimension not in (31, 39) or len(adapter.blocks) != 6:
            raise ValueError("Expected the original 31/39D six-block adapter")
        self.adapter = adapter
        self.dimension = dimension
        self.language = torch.tensor([language_id], dtype=torch.long)
        self.received = self.emitted = 0
        self.finished = False

        def initial(value, residual):
            return adapter.norm(value + adapter.language(self.language)[:, None])

        self.layers = [_LayerCursor(adapter.inputs[str(dimension)], initial)]
        for block in adapter.blocks:
            def finish(value, residual, block=block):
                return residual + block.gamma * block.down(F.gelu(block.up(block.norm(value))))
            self.layers.append(_LayerCursor(block.dw, finish))

    @torch.inference_mode()
    def push(self, normalized, final=False):
        if self.finished:
            raise ValueError("Adapter stream is already finished")
        if (normalized.ndim != 3 or normalized.shape[0] != 1
                or normalized.shape[2] != self.dimension
                or normalized.device.type != "cpu" or normalized.dtype != torch.float32):
            raise ValueError("Adapter stream requires CPU FP32 [1,T,D]")
        self.received += int(normalized.shape[1])
        x = normalized
        for layer in self.layers:
            x = layer.push(x, final=final)
        features = self.adapter.head(x).float() if x.shape[1] else x.new_empty((1, 0, 20))
        if not torch.isfinite(features).all():
            raise ValueError("Adapter returned nonfinite FARGAN features")
        self.emitted += int(features.shape[1])
        self.finished = bool(final)
        if final and self.emitted != self.received:
            raise RuntimeError("Adapter EOS did not release the complete physical timeline")
        return features

    def report(self):
        return {
            "received_physical_frames": self.received,
            "emitted_feature_frames": self.emitted,
            "future_context_frames": 21,
            "layers": [{"received": x.received, "emitted": x.emitted,
                        "cached_frames": 0 if x.buffer is None else int(x.buffer.shape[1]),
                        "peak_cached_frames": x.max_buffer} for x in self.layers],
            "finished": self.finished,
        }
