"""Persistent FARGAN synthesis of the selected model's original valid center.

Consume the existing NEW25 normalized waveform condition. T input frames give
(T-4)*160 samples aligned to [400,160*T-240) on the original utterance timeline.
No edge extension, reference waveform or teacher prefill is applied. The original
offline reference remains available as ``synthesize_offline``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from new25_acoustic.inference import LANGUAGES, MAX_FRAMES, _read_json, load_safetensors
# Share the released AcousticResult validation and PCM16 writer. This does not
# construct a HiFi-GAN model, read its weights, or require its optional frontend.
from new25_hifigan.inference import (
    _language_id, _normalized_tensor, load_acoustic_npz, write_pcm16,
)

from .adapter import PhysicalFarganAdapter
from .adapter_stream import PhysicalFarganAdapterStream
from ._vendor.fargan import FARGAN

SAMPLE_RATE = 16000
HOP_SIZE = 160
CHECKPOINT_SHA256 = "e90567f4f4ac8811d99b6cba910d117207a95a21be05cb5dbb0ebc1cfbc32e71"
METADATA = {
    "format": "new25_fargan_v1",
    "languages": list(LANGUAGES),
    "physical_dimensions": {lang: 31 if lang == "zh" else 39 for lang in LANGUAGES},
    "model_args": [],
    "model_kwargs": {"cond_size": 256, "gamma": 0.9, "softquant": True},
    "sample_rate": SAMPLE_RATE,
    "hop_size": HOP_SIZE,
    "generator_parameters": 817525,
    "adapter_parameters": 1891412,
    "tensor_count": 91,
    "selected_step": 4750,
    "streaming": False,
    "output_alignment": "[400,160*T-240)",
    "teacher_prefill_frames": 0,
    "deemphasis": 0.85,
    "math": "FP32 including recurrent state",
    "parameter_storage": "FP16",
}
# METADATA describes immutable released files, including their historical API.
# Runtime capability is separate: loading never rewrites model_config.json.
RUNTIME_CAPABILITIES = {"streaming": True, "artifact_streaming": False,
                        "adapter_future_context_frames": 21,
                        "output_alignment": "[400,160*T-240)"}


@dataclass
class FARGANStreamState:
    owner: object
    language_id: int
    seed: int
    adapter: PhysicalFarganAdapterStream
    generator: object
    received_frames: int = 0
    output_samples: int = 0
    finished: bool = False

    def report(self):
        return {"received_frames": self.received_frames,
                "output_samples": self.output_samples, "finished": self.finished,
                "adapter": self.adapter.report(), "generator": self.generator.report()}


def _chunk_tensor(normalized, language_id):
    """Same numeric contract as the released reader, also accepting empty blocks."""
    if isinstance(normalized, torch.Tensor):
        if normalized.dtype == torch.bool or normalized.is_complex() or normalized.is_quantized:
            raise ValueError("normalized must contain finite real numeric features")
        x = normalized.detach()
    else:
        value = np.asarray(normalized)
        if value.dtype.kind not in "iuf":
            raise ValueError("normalized must contain finite real numeric features")
        with np.errstate(over="ignore"):
            x = torch.from_numpy(value.astype(np.float32, copy=True))
    dimension = 31 if LANGUAGES[language_id] == "zh" else 39
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != dimension:
        raise ValueError(f"normalized must have shape [1,T,{dimension}]")
    if not 0 <= x.shape[1] <= MAX_FRAMES:
        raise ValueError(f"A chunk must contain 0..{MAX_FRAMES} physical frames")
    x = x.to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("normalized must contain finite values representable in FP32")
    return x


def sample_layout(frames):
    """Coordinates refer to the untrimmed acoustic utterance at 16 kHz."""
    if type(frames) is not int or not 5 <= frames <= MAX_FRAMES:
        raise ValueError(f"FARGAN requires a complete timeline of 5..{MAX_FRAMES} frames")
    return {
        "input_frames": frames,
        "output_frames": frames - 4,
        "output_samples": (frames - 4) * HOP_SIZE,
        "source_sample_start": 400,
        "source_sample_end": frames * HOP_SIZE - 240,
        "sample_rate": SAMPLE_RATE,
        "hop_size": HOP_SIZE,
    }


class FARGANVocoder(nn.Module):
    """B=1 CPU FP32 reference returning float32 [1,(T-4)*160] PCM.

    ``normalized`` is AcousticResult.normalized with an explicit batch axis,
    not raw head logits or physical values. A stream preserves adapter context,
    excitation, recurrent, noise and de-emphasis state across calls. Output
    remains the original valid center, not the reserved T*160 padded protocol.
    """

    streaming = True

    def __init__(self):
        super().__init__()
        # Upstream constructors set seeds. Isolate them from caller RNG state.
        with torch.random.fork_rng(devices=[]):
            self.generator = FARGAN(**METADATA["model_kwargs"])
            self.adapter = PhysicalFarganAdapter()
        self.float().eval().requires_grad_(False)

    @classmethod
    def from_pretrained(cls, model_dir="models/fargan"):
        """Strictly load the released 91-tensor model-only safetensors bundle."""
        root = Path(model_dir)
        metadata = _read_json(root / "model_config.json")
        if metadata != METADATA:
            raise ValueError("Unsupported FARGAN model metadata")
        export = _read_json(root / "EXPORT.json")
        if (export.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
                or export.get("source_checkpoint_step") != 4750
                or export.get("contains_acoustic_weights") is not False):
            raise ValueError("Unexpected selected FARGAN identity")
        for name in ("model_config.json", "vocoder_adapter_fp16.safetensors"):
            path = root / name
            expected = export["files"][name]
            if (path.stat().st_size != expected["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]):
                raise ValueError("FARGAN export identity mismatch: " + name)
        state = load_safetensors(root / "vocoder_adapter_fp16.safetensors")
        model = cls()
        reference = model.state_dict()
        if len(state) != 91 or set(state) != set(reference):
            raise ValueError("Expected exactly 91 generator.* and adapter.* tensors")
        for name, value in state.items():
            if value.shape != reference[name].shape:
                raise ValueError("FARGAN weight shape mismatch: " + name)
        model.load_state_dict(state, strict=True)
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in (("generator", model.generator), ("adapter", model.adapter))}
        if counts != {"generator": 817525, "adapter": 1891412}:
            raise ValueError("Unexpected FARGAN component parameter count")
        return model

    @torch.inference_mode()
    def synthesize_offline(self, normalized, language, *, seed=0):
        """Render the original valid center with zero teacher-prefill frames.

        The official Python FARGAN adds noise even in eval. An explicit seed
        makes repeated calls reproducible without changing the caller's RNG.
        The input durations may be supplied or predicted upstream; only the
        supplied-true-duration path has the recorded 75-case development score.
        """
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in 0..2**63-1")
        language_id = _language_id(language)
        x = _normalized_tensor(normalized, language_id, "cpu")
        layout = sample_layout(int(x.shape[1]))
        if any(p.device.type != "cpu" for p in self.parameters()):
            raise ValueError("Selected FARGAN reference supports CPU inference")
        self.float().eval()
        from scipy.signal import lfilter
        with torch.random.fork_rng(devices=[]), torch.autocast("cpu", enabled=False):
            torch.random.default_generator.manual_seed(seed)
            features = self.adapter(x, torch.tensor([language_id], dtype=torch.long))
            if not torch.isfinite(features).all():
                raise ValueError("Adapter returned nonfinite FARGAN features")
            with np.errstate(over="ignore"):
                periods = np.round(np.clip(256.0 / 2**(features.numpy()[0, :, 18] + 1.5),
                                           32, 255)).astype(np.int64)
            raw, states = self.generator(features, torch.from_numpy(periods)[None], layout["output_frames"])
            if raw.shape != (1, layout["output_samples"]) or not torch.isfinite(raw).all():
                raise ValueError("FARGAN returned invalid PCM")
            if any(item.dtype != torch.float32 for item in states):
                raise ValueError("FARGAN recurrent state must remain FP32")
        # One continuous de-emphasis pass, with the original zero initial state.
        pcm = lfilter([1.], [1., -.85], raw.numpy()[0]).astype(np.float32)
        if not np.isfinite(pcm).all():
            raise ValueError("De-emphasis returned nonfinite PCM")
        return torch.from_numpy(pcm.copy())[None]

    def begin_stream(self, language, seed=0):
        """Create an independent request without mutating any caller RNG."""
        from .streaming_core import FARGANGeneratorStream
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in 0..2**63-1")
        language_id = _language_id(language)
        if any(p.device.type != "cpu" for p in self.parameters()):
            raise ValueError("Selected FARGAN reference supports CPU inference")
        self.float().eval()
        dimension = 31 if LANGUAGES[language_id] == "zh" else 39
        return FARGANStreamState(self, language_id, seed,
                                PhysicalFarganAdapterStream(self.adapter, language_id, dimension),
                                FARGANGeneratorStream(self.generator, seed))

    @torch.inference_mode()
    def synthesize_chunk(self, normalized, language, state=None, seed=0, final=False):
        """Consume new physical frames and return only newly finalized PCM."""
        language_id = _language_id(language)
        x = _chunk_tensor(normalized, language_id)
        if state is None:
            state = self.begin_stream(language, seed=seed)
        if (not isinstance(state, FARGANStreamState) or state.owner is not self
                or state.language_id != language_id):
            raise ValueError("Stream state belongs to another model or language")
        if state.finished:
            raise ValueError("FARGAN stream is already finished; begin a new stream")
        if type(final) is not bool:
            raise ValueError("final must be a bool")
        total = state.received_frames + int(x.shape[1])
        if total > MAX_FRAMES or (final and total < 5):
            raise ValueError(f"A complete FARGAN stream requires 5..{MAX_FRAMES} physical frames")
        with torch.autocast("cpu", enabled=False):
            features = state.adapter.push(x, final=final)
            pcm = state.generator.push(features, final=final)
        if pcm.ndim != 2 or pcm.shape[0] != 1 or pcm.dtype != torch.float32 or not torch.isfinite(pcm).all():
            raise ValueError("FARGAN stream returned invalid PCM")
        state.received_frames = total
        state.output_samples += int(pcm.shape[1])
        state.finished = final
        if final and state.output_samples != sample_layout(total)["output_samples"]:
            raise RuntimeError("FARGAN stream did not preserve the original valid center")
        return pcm, state

    @torch.inference_mode()
    def synthesize(self, normalized, language, *, seed=0, state=None, chunk_frames=None):
        """Collect real streaming output; use synthesize_chunk for immediate delivery."""
        if state is not None:
            raise ValueError("Use synthesize_chunk to continue an existing stream")
        language_id = _language_id(language)
        x = _normalized_tensor(normalized, language_id, "cpu")
        sample_layout(int(x.shape[1]))
        chunk_frames = 16 if chunk_frames is None else chunk_frames
        if type(chunk_frames) is not int or chunk_frames < 1:
            raise ValueError("chunk_frames must be a positive integer")
        state = self.begin_stream(language, seed=seed)
        pieces = []
        for begin in range(0, int(x.shape[1]), chunk_frames):
            end = min(int(x.shape[1]), begin + chunk_frames)
            pcm, state = self.synthesize_chunk(x[:, begin:end], language, state=state,
                                               final=end == x.shape[1])
            pieces.append(pcm)
        return torch.cat(pieces, dim=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, required=True, help="Existing AcousticResult NPZ")
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/fargan"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Explicit PCM16 mono WAV destination")
    parser.add_argument("--metadata", type=Path, help="Defaults to OUTPUT.wav.json")
    args = parser.parse_args(argv)
    normalized = load_acoustic_npz(args.acoustic, args.language)
    model = FARGANVocoder.from_pretrained(args.model_dir)
    waveform = model.synthesize(normalized, args.language, seed=args.seed)
    write_pcm16(args.output, waveform)
    receipt = dict(sample_layout(int(normalized.shape[1])), language=args.language,
                   selected_checkpoint_step=4750, source_checkpoint_sha256=CHECKPOINT_SHA256,
                   seed=args.seed, streaming=True, artifact_streaming=False, teacher_prefill_frames=0,
                   alignment="valid_center", duration_source="supplied_by_acoustic_npz",
                   pcm16_clipped_samples=int((waveform.abs() > 1).sum()))
    metadata = args.metadata if args.metadata is not None else Path(str(args.output) + ".json")
    metadata.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(receipt, output=str(args.output), metadata=str(metadata))))


if __name__ == "__main__":
    main()
