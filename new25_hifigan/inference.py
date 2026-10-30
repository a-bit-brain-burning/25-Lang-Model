"""Single-utterance offline HiFi-GAN synthesis from NEW25 normalized features.

FP16 deployment weights are expanded to FP32 for reference computation. The
physical adapter, generator and polyphase resampler have future context. This
API therefore requires a complete utterance and has no reusable stream state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import wave

import numpy as np
import torch
from torch import nn

from new25_acoustic.inference import LANGUAGES, MAX_FRAMES, _read_json, load_safetensors

from .frontend import sample_layout, to_16k
from .model import Generator, PhysicalMelAdapter, validate_generator_config

SAMPLE_RATE = 16000
HOP_SIZE = 160
METADATA = {
    "format": "new25_hifigan_v1",
    "languages": list(LANGUAGES),
    "physical_dimensions": {language: 31 if language == "zh" else 39 for language in LANGUAGES},
    "adapter_context_frames": 0,
    "generator_native_sample_rate": 22050,
    "output_sample_rate": SAMPLE_RATE,
    "acoustic_hop_size": HOP_SIZE,
    "generator_hop_size": 256,
    "generator_parameters": 925985,
    "adapter_parameters": 1902992,
    "tensor_count": 219,
    "streaming": False,
}


def validate_metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise ValueError("Model metadata must be a JSON object")
    for name, expected in METADATA.items():
        if type(metadata.get(name)) is not type(expected) or metadata.get(name) != expected:
            raise ValueError(f"Unsupported model metadata {name}: expected {expected!r}")
    return dict(metadata)


def _language_id(language):
    if isinstance(language, str):
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        return LANGUAGES.index(language)
    if isinstance(language, torch.Tensor):
        if language.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("language must contain one integer language ID")
        array = language.detach().cpu().numpy()
    else:
        array = np.asarray(language)
    if array.shape != (1,) or array.dtype.kind not in "iu":
        raise ValueError("language must have shape [1] with an integer language ID")
    result = int(array[0])
    if result not in range(len(LANGUAGES)):
        raise ValueError("language ID must be in 0..24")
    return result


def _normalized_tensor(normalized, language_id, device):
    if isinstance(normalized, torch.Tensor):
        if normalized.dtype == torch.bool or normalized.is_complex() or normalized.is_quantized:
            raise ValueError("normalized must contain finite real numeric features")
        x = normalized.detach()
    else:
        value = np.asarray(normalized)
        if value.dtype.kind not in "iuf":
            raise ValueError("normalized must contain finite real numeric features")
        # Copy accepts read-only arrays and arrays with negative strides safely.
        with np.errstate(over="ignore"):
            x = torch.from_numpy(value.astype(np.float32, copy=True))
    dimension = 31 if LANGUAGES[language_id] == "zh" else 39
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != dimension:
        raise ValueError(f"normalized must have shape [1,T,{dimension}] for {LANGUAGES[language_id]}")
    if x.shape[1] not in range(1, MAX_FRAMES + 1):
        raise ValueError(f"A complete utterance must contain 1..{MAX_FRAMES} physical frames")
    x = x.to(device=device, dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("normalized must contain finite values representable in FP32")
    return x


class HiFiGANVocoder(nn.Module):
    """Offline B=1 reference synthesizer returning CPU float32 [1,T*160] audio.

    ``normalized`` is the waveform condition produced by NEW25 acoustic
    inference, including continuous normalized log-F0 and normalized sigmoid
    VUV. It is not the raw physical vector or the acoustic head's logits.
    ``language`` is an integer [1] tensor/array in published language order,
    or its language code. No reference waveform is used.
    """

    streaming = False

    def __init__(self, generator=None, physical_to_mel=None, *, device="cpu"):
        super().__init__()
        selected = torch.device(device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("Reference inference supports cpu or cuda")
        self.generator = Generator() if generator is None else generator
        self.physical_to_mel = PhysicalMelAdapter() if physical_to_mel is None else physical_to_mel
        self.to(device=selected, dtype=torch.float32).eval()
        if sum(p.numel() for p in self.generator.parameters()) != METADATA["generator_parameters"]:
            raise ValueError("Unexpected generator parameter count")
        if sum(p.numel() for p in self.physical_to_mel.parameters()) != METADATA["adapter_parameters"]:
            raise ValueError("Unexpected adapter parameter count")
        if len(self.state_dict()) != METADATA["tensor_count"]:
            raise ValueError("Unexpected generator/adapter tensor count")
        if any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("Model contains nonfinite parameters")

    @classmethod
    def from_pretrained(cls, model_dir="models/hifigan", *, weights="vocoder_adapter_fp16.safetensors", device="cpu"):
        """Strictly load the folded generator/adapter bundle and its metadata.

        Supported files are standard F16/F32 safetensors and a pure tensor
        state dictionary loaded with ``torch.load(weights_only=True)``.
        """
        selected = torch.device(device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("Reference inference supports cpu or cuda")
        root = Path(model_dir)
        config = validate_generator_config(_read_json(root / "config.json"))
        metadata = validate_metadata(_read_json(root / "model_config.json"))
        path = root / weights
        if path.suffix == ".safetensors":
            state = load_safetensors(path)
        elif path.suffix == ".pt":
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Weights must use .safetensors or a weights-only .pt state dictionary")
        model = cls(Generator(config), PhysicalMelAdapter())
        reference = model.state_dict()
        if not isinstance(state, Mapping) or set(state) != set(reference):
            raise ValueError("Expected exactly the 219 generator.* and physical_to_mel.* state keys")
        for name, expected in reference.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape:
                raise ValueError(f"Weight shape mismatch: {name}")
            if tensor.dtype not in (torch.float16, torch.float32) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid weight dtype/value: {name}")
        model.load_state_dict(state, strict=True)
        model.to(device=selected, dtype=torch.float32).eval()
        model.metadata = metadata
        return model

    @torch.inference_mode()
    def synthesize(self, normalized, language, *, state=None, chunk_frames=None):
        """Synthesize one complete utterance, then resample/trim to exactly T*160.

        Any non-None ``state`` or ``chunk_frames`` is rejected. Splitting an
        utterance into separate calls changes context and waveform boundaries.
        """
        if state is not None or chunk_frames is not None:
            raise ValueError("HiFi-GAN requires a complete utterance; stream state and chunking are unsupported")
        language_id = _language_id(language)
        device = next(self.parameters()).device
        x = _normalized_tensor(normalized, language_id, device)
        length, _, mel_frames = sample_layout(int(x.shape[1]))
        lang = torch.tensor([language_id], device=device, dtype=torch.long)
        self.float().eval()
        with torch.autocast(device_type=device.type, enabled=False):
            mel = self.physical_to_mel(x, lang, mel_frames, context_frames=0)
            native = self.generator(mel)
        if native.shape != (1, 1, mel_frames * 256) or not torch.isfinite(native).all():
            raise ValueError("Generator returned an invalid native waveform")
        audio = to_16k(native[0, 0].float().cpu().numpy(), length)
        if not np.isfinite(audio).all():
            raise ValueError("Resampler returned nonfinite audio")
        return torch.from_numpy(audio.copy())[None]

    def synthesize_chunk(self, *args, **kwargs):
        raise ValueError("HiFi-GAN has no streaming/chunk API; call synthesize with a complete utterance")


def load_acoustic_npz(path, language):
    """Read a pickle-free AcousticResult NPZ and check its semantic metadata.

    The acoustic package saves [T,D]; [1,T,D] is also accepted. ``physical``
    never substitutes for the required ``normalized`` waveform condition.
    """
    language_id = _language_id(language)
    with np.load(path, allow_pickle=False) as data:
        if "normalized" not in data:
            raise ValueError("Acoustic NPZ must include normalized waveform-condition features")
        normalized = np.array(data["normalized"], copy=True)
        if normalized.ndim == 2:
            normalized = normalized[None]
        normalized = _normalized_tensor(normalized, language_id, "cpu").numpy()
        for name, expected in (("sample_rate", SAMPLE_RATE), ("hop_size", HOP_SIZE)):
            if name in data and (data[name].shape != () or data[name].dtype.kind not in "iu" or int(data[name]) != expected):
                raise ValueError(f"Acoustic NPZ {name} must be {expected}")
        if "language" in data and (data["language"].shape != () or str(data["language"].item()) != LANGUAGES[language_id]):
            raise ValueError("Acoustic NPZ language does not match --language")
        for name in ("language_id", "lang"):
            if name in data:
                value = data[name]
                if name == "language_id" and value.shape == ():
                    value = value.reshape(1)
                if _language_id(value) != language_id:
                    raise ValueError("Acoustic NPZ language ID does not match --language")
        if "physical" in data:
            physical = np.asarray(data["physical"])
            expected = normalized.shape[1:] if physical.ndim == 2 else normalized.shape
            if physical.shape != expected or physical.dtype.kind not in "iuf" or not np.isfinite(physical).all():
                raise ValueError("Acoustic NPZ physical features must match the finite normalized timeline")
        if "durations" in data:
            durations = np.asarray(data["durations"])
            if (durations.ndim != 1 or durations.dtype.kind not in "iu" or np.any(durations < 0)
                    or sum(int(n) for n in durations) != normalized.shape[1]):
                raise ValueError("Acoustic NPZ durations must sum to the normalized frame count")
    return normalized


def write_pcm16(path, waveform):
    """Explicitly quantize finite float audio to clipped signed PCM16 mono WAV."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    audio = np.asarray(waveform)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 1 or not len(audio) or audio.dtype.kind not in "iuf" or not np.isfinite(audio).all():
        raise ValueError("WAV audio must be a nonempty finite mono waveform")
    # Full-scale -1 maps to -32768; the positive full-scale endpoint saturates.
    pcm = np.rint(np.clip(audio.astype(np.float64), -1, 1) * 32768).clip(-32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", required=True, type=Path, help="NEW25 AcousticResult NPZ with normalized features")
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--model-dir", type=Path, default=Path("models/hifigan"))
    parser.add_argument("--weights", default="vocoder_adapter_fp16.safetensors")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", required=True, type=Path, help="Explicit PCM16 mono WAV destination")
    args = parser.parse_args(argv)
    normalized = load_acoustic_npz(args.acoustic, args.language)
    model = HiFiGANVocoder.from_pretrained(args.model_dir, weights=args.weights, device=args.device)
    waveform = model.synthesize(normalized, args.language)
    write_pcm16(args.output, waveform)
    print(f"Wrote {waveform.shape[-1]} samples at {SAMPLE_RATE} Hz to {args.output} (clipped/rounded PCM16 quantization)")


if __name__ == "__main__":
    main()
