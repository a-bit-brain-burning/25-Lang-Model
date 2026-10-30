"""Offline FARGAN synthesis of the selected model's original valid center.

Consume the existing NEW25 normalized waveform condition. T input frames give
(T-4)*160 samples aligned to [400,160*T-240) on the original utterance timeline.
No edge padding, reference waveform, teacher prefill or streaming is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    not raw head logits or physical values. The trained adapter consumes the
    entire supplied timeline. Each call starts fresh recurrent/excitation and
    de-emphasis states; it is not the reserved T*160 streaming protocol.
    """

    streaming = False

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
    def synthesize(self, normalized, language, *, seed=0, state=None, chunk_frames=None):
        """Render the original valid center with zero teacher-prefill frames.

        The official Python FARGAN adds noise even in eval. An explicit seed
        makes repeated calls reproducible without changing the caller's RNG.
        The input durations may be supplied or predicted upstream; only the
        supplied-true-duration path has the recorded 75-case development score.
        """
        if state is not None or chunk_frames is not None:
            raise ValueError("FARGAN requires a complete utterance; state/chunking are unsupported")
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

    def synthesize_chunk(self, *args, **kwargs):
        raise ValueError("FARGAN has no streaming/chunk API; supply a complete utterance")


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
                   seed=args.seed, streaming=False, teacher_prefill_frames=0,
                   alignment="valid_center", duration_source="supplied_by_acoustic_npz",
                   pcm16_clipped_samples=int((waveform.abs() > 1).sum()))
    metadata = args.metadata if args.metadata is not None else Path(str(args.output) + ".json")
    metadata.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(receipt, output=str(args.output), metadata=str(metadata))))


if __name__ == "__main__":
    main()
