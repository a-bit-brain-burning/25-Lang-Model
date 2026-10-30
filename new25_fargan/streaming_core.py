"""Stateful original FARGAN valid-center synthesis, without padding or prefill.

Input is already the adapter's CPU FP32 [1,N,20] feature sequence. A frame n
uses conditioning features[n+2:n+5] and period[n+3]. Thus T features produce
(T-4)*160 samples at original coordinates [400,160*T-240). Chunk boundaries
do not reset excitation, recurrent state, random noise or de-emphasis.
"""
from __future__ import annotations

from collections import deque
import threading

import numpy as np
from scipy.signal import lfilter
import torch


_RNG_LOCK = threading.RLock()


class FARGANGeneratorStream:
    """One independent request using the existing 28-tensor generator.

    The three GRUs, FWConv state, excitation history and previous-prediction
    history persist. Original sig_net, including its evaluation-time n() noise,
    is called unchanged. CPU RNG is restored around every push; different
    streams can be interleaved without changing their noise sequences.
    """

    streaming = True
    sample_rate = 16000
    hop_size = 160

    def __init__(self, generator, seed=0):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in 0..2**63-1")
        if (generator.subframe_size, generator.nb_subframes,
                generator.frame_size, generator.feature_dim) != (40, 4, 160, 20):
            raise ValueError("Expected the original 40x4, 20-feature FARGAN")
        if any(p.device.type != "cpu" or p.dtype != torch.float32
               for p in generator.parameters()):
            raise ValueError("FARGAN streaming requires CPU FP32 generator parameters")
        self.generator = generator
        self.generator.eval()
        self.seed = seed
        self.received = 0
        self.emitted_frames = 0
        self.sig_net_calls = 0
        self.condition_input_rows = 0
        self.condition_valid_rows = 0
        self.finished = False
        self.failed = False
        self._conditioning = deque()
        self._max_condition_rows = 0
        self.prev = torch.zeros((1, 256), dtype=torch.float32)
        self.exc_mem = torch.zeros((1, 256), dtype=torch.float32)
        self.states = (
            torch.zeros((1, 160), dtype=torch.float32),
            torch.zeros((1, 128), dtype=torch.float32),
            torch.zeros((1, 128), dtype=torch.float32),
            torch.zeros((1, 164), dtype=torch.float32),
        )
        # scipy uses the original float64 IIR arithmetic before FP32 output.
        self._deemphasis_zi = np.zeros(1, dtype=np.float64)
        with _RNG_LOCK, torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self._rng_state = torch.random.get_rng_state().clone()

    @torch.inference_mode()
    def push(self, features, final=False):
        if self.finished or self.failed:
            raise RuntimeError("FARGAN stream is already finished or failed")
        if type(final) is not bool:
            raise TypeError("final must be bool")
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[0] != 1 or features.shape[2] != 20
                or features.dtype != torch.float32 or features.device.type != "cpu"):
            raise ValueError("Expected finite CPU float32 features of shape [1,N,20]")
        if not torch.isfinite(features).all():
            raise ValueError("FARGAN features contain nonfinite values")
        count = int(features.shape[1])
        if final and self.received + count < 5:
            raise ValueError("FARGAN valid-center synthesis requires at least five features")
        features = features.detach()
        # This is the released period conversion, including NumPy round/clip.
        with np.errstate(over="ignore"):
            periods = np.round(np.clip(
                256.0 / 2**(features.numpy()[0, :, 18] + 1.5), 32, 255,
            )).astype(np.int64)
        raw_parts = []
        try:
            with _RNG_LOCK, torch.random.fork_rng(devices=[]), torch.autocast("cpu", enabled=False):
                torch.random.set_rng_state(self._rng_state)
                cond_net = self.generator.cond_net
                for local in range(count):
                    absolute = self.received
                    self.received += 1
                    if absolute < 2:
                        # Original cond_net begins with features[:,2:].
                        continue
                    frame = features[:, local, :]
                    period_value = int(periods[local])
                    period = torch.tensor([period_value], dtype=torch.long)
                    embedded = cond_net.pembed(period - 32)
                    encoded = torch.tanh(cond_net.fdense1(torch.cat((frame, embedded), -1)))
                    # Clone only the one gain coordinate, never retain input chunks.
                    self._conditioning.append((encoded, period_value, frame[:, 0:1].clone()))
                    self.condition_input_rows += 1
                    self._max_condition_rows = max(self._max_condition_rows, len(self._conditioning))
                    if len(self._conditioning) < 3:
                        continue
                    if len(self._conditioning) != 3:
                        raise RuntimeError("FARGAN conditioning queue exceeded the valid3 window")
                    window = torch.stack([item[0] for item in self._conditioning], dim=1)
                    conditioned = torch.tanh(cond_net.fconv1(window.permute(0, 2, 1)))
                    conditioned = torch.tanh(cond_net.fdense2(conditioned.permute(0, 2, 1)))
                    cond = conditioned[:, 0, :]
                    center = self._conditioning[1]
                    pitch = torch.tensor([center[1]], dtype=torch.long)
                    self.condition_valid_rows += 1
                    for k in range(self.generator.nb_subframes):
                        # Keep upstream's gain argument even though sig_net itself
                        # recomputes gain from its noisy conditioned input.
                        gain = .03 * 10**(.5 * center[2] / np.sqrt(18.0))
                        out, self.exc_mem, self.prev, self.states = self.generator.sig_net(
                            cond[:, k*80:(k+1)*80], self.prev, self.exc_mem,
                            pitch, self.states, gain=gain,
                        )
                        if out.shape != (1, 40) or not torch.isfinite(out).all():
                            raise ValueError("FARGAN returned invalid subframe PCM")
                        raw_parts.append(out)
                        self.sig_net_calls += 1
                    self.emitted_frames += 1
                    self._conditioning.popleft()
                self._rng_state = torch.random.get_rng_state().clone()
            if any(s.dtype != torch.float32 or not torch.isfinite(s).all()
                   for s in (*self.states, self.exc_mem, self.prev)):
                raise ValueError("FARGAN stream state must remain finite FP32")
            if raw_parts:
                raw = torch.cat(raw_parts, dim=1).numpy()[0]
                pcm, self._deemphasis_zi = lfilter(
                    [1.], [1., -.85], raw, zi=self._deemphasis_zi,
                )
                pcm = pcm.astype(np.float32)
                if not np.isfinite(pcm).all():
                    raise ValueError("FARGAN de-emphasis returned nonfinite PCM")
                output = torch.from_numpy(pcm.copy())[None]
            else:
                output = torch.empty((1, 0), dtype=torch.float32)
            if self.emitted_frames != max(0, self.received - 4):
                raise RuntimeError("FARGAN valid-center frame count mismatch")
            if final:
                self.finished = True
            return output
        except Exception:
            self.failed = True
            raise

    def finish(self):
        """Flush no extra samples: original valid convolution has no right pad."""
        return self.push(torch.empty((1, 0, 20), dtype=torch.float32), final=True)

    def report(self):
        return {
            "streaming": True,
            "seed": self.seed,
            "received_features": self.received,
            "emitted_frames": self.emitted_frames,
            "emitted_samples": self.emitted_frames * self.hop_size,
            "condition_input_rows": self.condition_input_rows,
            "condition_valid_rows": self.condition_valid_rows,
            "sig_net_calls": self.sig_net_calls,
            "state_shapes": [list(s.shape) for s in self.states],
            "state_dtype": "float32",
            "excitation_samples": self.exc_mem.shape[1],
            "previous_prediction_samples": self.prev.shape[1],
            "condition_rows_retained": len(self._conditioning),
            "condition_rows_peak": self._max_condition_rows,
            "deemphasis_state_dtype": str(self._deemphasis_zi.dtype),
            "source_sample_start": 400,
            "source_sample_end": self.received * self.hop_size - 240 if self.received >= 5 else None,
            "teacher_prefill_frames": 0,
            "padding_frames": 0,
            "finished": self.finished,
            "failed": self.failed,
        }
