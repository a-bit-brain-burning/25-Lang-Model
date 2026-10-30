"""Offline selected RNDVoC25k: NEW25 condition -> E2 adapter -> raw80 mel.

The CPU reference expands FP16 learned weights to FP32. All three nonlearned
projection buffers, FFT, overlap-add and intermediate computations stay FP32.
Complete utterances are required; no streaming state or chunk API is claimed.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from new25_acoustic.inference import LANGUAGES, MAX_FRAMES, _read_json, load_safetensors
from new25_hifigan.frontend import sample_layout, to_16k
from new25_hifigan.inference import (
    _language_id, _normalized_tensor, load_acoustic_npz, write_pcm16,
)
from new25_hifigan.model import PhysicalMelAdapter

from .model import RNDVoCLite

CHECKPOINT_SHA256 = "45b811b8f2a3ae7eba3742fc2b0115d6898145f42fa5b8c662bb6bf8032e1dd1"
MODEL_CONFIG = {
    "sampling_rate": 22050, "num_mels": 80, "n_fft": 1024,
    "win_size": 1024, "hop_size": 256, "fmin": 0, "fmax": 8000,
    "nrep": 2, "kernel_size": 7, "causal": False,
    "time_type": "convnext_v2", "freq_type": "shuffler",
    "use_shared_encoder": True, "use_shared_decoder": True,
    "use_rnd": True, "phiphiT_learnable": False,
    "null_nstage": 4, "input_channel": 128,
    "hidden_channel": 72, "squeeze_size": 32,
}
BUFFER_NAMES = {"Phi", "PhiT", "PhiTPhi"}


def _verify_file(root, export, name):
    expected = export["files"][name]
    path = root / name
    if (path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]):
        raise ValueError("RNDVoC export identity mismatch: " + name)


def _load_state(module, state, *, learned_dtype, buffers=()):
    reference = module.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(reference):
        raise ValueError("Model tensor names do not match the selected architecture")
    for name, value in state.items():
        expected_dtype = torch.float32 if name in buffers else learned_dtype
        if value.shape != reference[name].shape or value.dtype != expected_dtype:
            raise ValueError("Model tensor shape/dtype mismatch: " + name)
    module.load_state_dict(state, strict=True)


class RNDVoCVocoder(nn.Module):
    """B=1 CPU reference with raw-mel and physical-condition entry points.

    ``synthesize`` consumes AcousticResult.normalized [1,T,31/39], returning
    exactly [1,T*160] samples at 16 kHz. ``synthesize_mel`` consumes raw natural
    log-mel [1,80,M] on the pinned HiFiGAN frame grid, returning [1,M*256] at
    22.05 kHz. The latter uses no physical adapter or reference waveform.
    """

    streaming = False
    output_sample_rate = 16000
    native_sample_rate = 22050

    def __init__(self, config=None):
        super().__init__()
        selected = MODEL_CONFIG if config is None else config
        if not isinstance(selected, Mapping) or dict(selected) != MODEL_CONFIG:
            raise ValueError("Only the selected B4/C128/hidden72/squeeze32 model is supported")
        with torch.random.fork_rng(devices=[]):
            self.generator = RNDVoCLite(SimpleNamespace(**selected))
            self.adapter = PhysicalMelAdapter()
        self.float().eval().requires_grad_(False)

    @classmethod
    def from_pretrained(cls, model_dir="models/rndvoc", *, weights="vocoder_fp16.safetensors"):
        """Load the fixed25k generator and separate E2 adapter without pickle.

        The default preserves the exact FP16 learned-weight / FP32 buffer
        storage contract used by the delivered25k CPU listening panel.
        ``vocoder_fp32.safetensors`` is the unrounded generator alternative;
        the adapter remains its already published FP16 version in either mode.
        """
        if weights not in ("vocoder_fp16.safetensors", "vocoder_fp32.safetensors"):
            raise ValueError("Select one of the two published generator weight files")
        root = Path(model_dir)
        export = _read_json(root / "EXPORT.json")
        if (export.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
                or export.get("source_checkpoint_step") != 25000):
            raise ValueError("Unexpected selected RNDVoC checkpoint")
        for name in (weights, "adapter_fp16.safetensors", "config.json", "model_config.json"):
            _verify_file(root, export, name)
        config = _read_json(root / "config.json")
        metadata = _read_json(root / "model_config.json")
        if (metadata.get("format") != "new25_rndvoc_25000_v1"
                or metadata.get("languages") != list(LANGUAGES)
                or metadata.get("generator_parameters") != 713929
                or metadata.get("adapter_parameters") != 1902992
                or metadata.get("streaming") is not False
                or metadata.get("nonlearned_buffer_dtype") != "F32"):
            raise ValueError("Unsupported RNDVoC interface metadata")
        model = cls(config)
        generator = load_safetensors(root / weights)
        adapter = load_safetensors(root / "adapter_fp16.safetensors")
        if len(generator) != 196 or len(adapter) != 63:
            raise ValueError("Expected 193 learned generator tensors, three buffers and 63 adapter tensors")
        _load_state(model.generator, generator,
                    learned_dtype=torch.float16 if "fp16" in weights else torch.float32,
                    buffers=BUFFER_NAMES)
        _load_state(model.adapter, adapter, learned_dtype=torch.float16)
        counts = (sum(p.numel() for p in model.generator.parameters()),
                  sum(p.numel() for p in model.adapter.parameters()),
                  sum(p.numel() for p in model.generator.buffers()))
        if counts != (713929, 1902992, 345249):
            raise ValueError("Unexpected generator/adapter/buffer element counts")
        model.metadata = metadata
        return model

    def _cpu_reference(self):
        if any(p.device.type != "cpu" for p in self.parameters()):
            raise ValueError("This published reference supports CPU; CUDA packaging has not been revalidated")
        self.float().eval()

    @torch.inference_mode()
    def synthesize_mel(self, raw_logmel):
        """Raw80 log-mel -> 22.05-kHz mono; output length exactly M*256.

        Supply denormalized natural-log mel, not z-scored training targets,
        physical31/39D data, dB, linear magnitudes or a different frame grid.
        """
        self._cpu_reference()
        if isinstance(raw_logmel, torch.Tensor):
            if raw_logmel.dtype == torch.bool or raw_logmel.is_complex() or raw_logmel.is_quantized:
                raise ValueError("mel must contain finite real values")
            mel = raw_logmel.detach().to(device="cpu", dtype=torch.float32)
        else:
            raw = np.asarray(raw_logmel)
            if raw.dtype.kind not in "iuf":
                raise ValueError("mel must contain numeric values")
            mel = torch.from_numpy(raw.astype(np.float32, copy=True))
        max_mel_frames = sample_layout(MAX_FRAMES)[2]
        if (mel.ndim != 3 or mel.shape[:2] != (1, 80)
                or not 1 <= mel.shape[-1] <= max_mel_frames or not torch.isfinite(mel).all()):
            raise ValueError("mel must have finite shape [1,80,M] within the maximum utterance length")
        with torch.autocast("cpu", enabled=False):
            waveform = self.generator(mel)[-1]
        if waveform.shape != (1, mel.shape[-1] * 256) or not torch.isfinite(waveform).all():
            raise ValueError("RNDVoC returned invalid native PCM")
        return waveform

    @torch.inference_mode()
    def synthesize(self, normalized, language, *, state=None, chunk_frames=None):
        """Complete normalized physical timeline -> 16-kHz [1,T*160] PCM."""
        if state is not None or chunk_frames is not None:
            raise ValueError("RNDVoC requires a complete utterance; state/chunking are unsupported")
        self._cpu_reference()
        language_id = _language_id(language)
        x = _normalized_tensor(normalized, language_id, "cpu")
        length, _, mel_frames = sample_layout(int(x.shape[1]))
        lang = torch.tensor([language_id], dtype=torch.long)
        with torch.autocast("cpu", enabled=False):
            mel = self.adapter(x, lang, mel_frames, context_frames=0)
            native = self.synthesize_mel(mel)
        audio = to_16k(native[0].numpy(), length)
        if audio.shape != (length,) or not np.isfinite(audio).all():
            raise ValueError("Invalid resampled waveform")
        return torch.from_numpy(audio.copy())[None]

    def synthesize_chunk(self, *args, **kwargs):
        raise ValueError("RNDVoC has no streaming/chunk API; provide a complete utterance")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, required=True, help="Existing AcousticResult NPZ")
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/rndvoc"))
    parser.add_argument("--weights", choices=("vocoder_fp16.safetensors", "vocoder_fp32.safetensors"),
                        default="vocoder_fp16.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Explicit 16-kHz PCM16 WAV destination")
    args = parser.parse_args(argv)
    normalized = load_acoustic_npz(args.acoustic, args.language)
    model = RNDVoCVocoder.from_pretrained(args.model_dir, weights=args.weights)
    waveform = model.synthesize(normalized, args.language)
    write_pcm16(args.output, waveform)
    print(f"Wrote {waveform.shape[-1]} samples at16000Hz using selected generator step25000")


if __name__ == "__main__":
    main()
