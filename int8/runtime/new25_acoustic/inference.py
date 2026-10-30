"""Single-utterance acoustic inference from existing original frontend features.

The input is already tokenized phone IDs plus raw phonetic/context arrays. This
package does not turn arbitrary text into those features and does not emit audio.
FP16 exports are expanded to FP32 for reference arithmetic, not native FP16 RAM
or performance measurements. Decoder state is explicit; the encoder consumes
the entire phone sequence and has noncausal context.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .model import Acoustic

LANGUAGES = (
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
    "lt", "lv", "nl", "no", "pl", "pt", "ro", "sk", "sl", "sv", "tr", "zh",
)
SAMPLE_RATE = 16000
HOP_SIZE = 160
MAX_FRAMES = 20000
FRONTEND_MODE = "original_text_derived_phone_features_NOT_new_text_frontend"


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_json_object)


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Read the standard dense F16/F32 safetensors subset without extra packages.

    Reject malformed headers, duplicate names, holes, overlapping buffers,
    unsupported dtypes and nonfinite weights. No pickle or executable code is
    accepted. Returned tensors own their storage and are FP32 on the CPU.
    """
    path = Path(path)
    size = path.stat().st_size
    if size < 10 or size > 100 * 1024 * 1024:
        raise ValueError("Unexpected acoustic safetensors file size")
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size < 2 or header_size > min(size - 8, 1024 * 1024):
            raise ValueError("Invalid safetensors header length")
        try:
            header = json.loads(handle.read(header_size), object_pairs_hook=_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid safetensors JSON header") from error
        data = handle.read()
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    result, ranges = {}, []
    for name, spec in header.items():
        if name == "__metadata__":
            if not isinstance(spec, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in spec.items()
            ):
                raise ValueError("Invalid safetensors metadata")
            continue
        if not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor specification: {name}")
        tag = spec["dtype"]
        dtype = {"F16": np.dtype("<f2"), "F32": np.dtype("<f4")}.get(tag) if isinstance(tag, str) else None
        shape, offsets = spec["shape"], spec["data_offsets"]
        if dtype is None:
            raise ValueError(f"Unsupported weight dtype: {spec['dtype']}")
        if not isinstance(shape, list) or not all(type(n) is int and n >= 0 for n in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(n) is int for n in offsets):
            raise ValueError(f"Invalid tensor offsets: {name}")
        begin, end = offsets
        if begin < 0 or end < begin or end > len(data) or end - begin != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"Tensor size/offset mismatch: {name}")
        array = np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=begin).reshape(shape)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite weight: {name}")
        result[name] = torch.from_numpy(array.astype(np.float32, copy=True))
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            raise ValueError("Overlapping buffers or holes in safetensors data")
        cursor = end
    if not result or cursor != len(data):
        raise ValueError("Empty or trailing safetensors data")
    return result


def _finite_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _integer_vector(value, name, length=None):
    array = _finite_array(value, name)
    if array.ndim != 1 or (length is not None and array.shape != (length,)):
        raise ValueError(f"{name} must have shape [{length if length is not None else 'N'}]")
    if array.dtype.kind == "b" or np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain integers")
    if np.any(array < 0) or np.any(array >= np.iinfo(np.int64).max):
        raise ValueError(f"{name} values are outside the nonnegative int64 range")
    return array.astype(np.int64)


def frame_layout(durations):
    """Original frame ownership and [relative_position, 1/duration, first, last]."""
    durations = _integer_vector(durations, "durations")
    if not len(durations) or sum(int(n) for n in durations) not in range(1, MAX_FRAMES + 1):
        raise ValueError(f"Timeline must have 1..{MAX_FRAMES} frames")
    ids = np.repeat(np.arange(len(durations)), durations)
    positions = []
    for n in durations:
        if n:
            j = np.arange(n, dtype=np.float32)
            positions.append(np.stack([j / max(n - 1, 1), np.full(n, 1 / n), j == 0, j == n - 1], -1))
    return ids.astype(np.int64), np.concatenate(positions).astype(np.float32)


def predict_durations(prosody: torch.Tensor, pause: torch.Tensor) -> torch.Tensor:
    """Match tools/evaluate.py, including suppression of a predicted absent pause."""
    if prosody.ndim != 3 or prosody.shape[0] != 1 or prosody.shape[-1] != 5:
        raise ValueError("prosody must have shape [1,N,5]")
    if pause.shape != prosody.shape[:2] or not torch.isfinite(prosody).all():
        raise ValueError("Invalid prosody/pause timeline")
    if not torch.isfinite(pause).all() or not torch.all((pause == 0) | (pause == 1)):
        raise ValueError("pause must contain only 0 or 1")
    duration = F.softplus(prosody[:, :, 0]) * (1 - pause) + F.softplus(prosody[:, :, 1]) * pause
    duration = (duration.exp() - 1).round().clamp(1, 400).long()
    return torch.where((pause > 0) & (prosody[:, :, 2] < 0), 0, duration)


def denormalize(prediction, normalization):
    """Physical VUV uses sigmoid of the head logit, never a hard VUV threshold."""
    mean = prediction.new_tensor(normalization["mean"])
    std = prediction.new_tensor(normalization["std"])
    raw = prediction.float() * std + mean
    return torch.cat([raw[:, :, :-1], torch.sigmoid(prediction[:, :, -1:])], -1)


def waveform_condition(physical, normalization):
    """Continuous predicted log-F0 is normalized without unvoiced imputation."""
    return (physical - physical.new_tensor(normalization["mean"])) / physical.new_tensor(normalization["std"])


@dataclass
class AcousticResult:
    """All arrays exclude the single batch dimension; decoder state retains it."""

    language: str
    physical: np.ndarray
    normalized: np.ndarray
    prediction: np.ndarray
    durations: np.ndarray
    frame_phone: np.ndarray
    position: np.ndarray
    acoustic_state: dict

    def save(self, path):
        """Save a pickle-free NPZ, including explicit final decoder state."""
        arrays = {
            "physical": self.physical, "normalized": self.normalized,
            "prediction": self.prediction, "durations": self.durations,
            "frame_phone": self.frame_phone, "position": self.position,
            "language": np.asarray(self.language), "language_id": np.asarray(LANGUAGES.index(self.language)),
            "sample_rate": np.asarray(SAMPLE_RATE), "hop_size": np.asarray(HOP_SIZE),
            "frontend_mode": np.asarray(FRONTEND_MODE),
            "acoustic_state_gru": self.acoustic_state["gru"],
        }
        arrays.update({f"acoustic_state_conv_{i}": x for i, x in enumerate(self.acoustic_state["conv"])})
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **arrays)


class AcousticPredictor:
    """Strict single-sentence FP32 reference adapter for the published weights."""

    def __init__(self, model: Acoustic, normalization: Mapping, vocabulary: Mapping, device="cpu"):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Reference inference supports cpu or cuda")
        if not isinstance(vocabulary, Mapping) or not vocabulary or any(
            type(i) is not int for i in vocabulary.values()
        ) or sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("Vocabulary IDs must be unique consecutive integers from zero")
        if len(vocabulary) != model.phone.num_embeddings:
            raise ValueError("Vocabulary/model size mismatch")
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.normalization = dict(normalization)
        self.vocabulary = dict(vocabulary)
        if set(self.normalization) != set(LANGUAGES):
            raise ValueError("Normalization must cover exactly the 25 supported languages")
        for language, norm in self.normalization.items():
            dimension = 31 if language == "zh" else 39
            for field, allowed in (("", (dimension,)), ("phonetic_", (32, 36)), ("context_", (22, 34))):
                mean = _finite_array(norm[field + "mean"], field + "mean").astype(np.float32)
                std = _finite_array(norm[field + "std"], field + "std").astype(np.float32)
                if mean.ndim != 1 or len(mean) not in allowed or std.shape != mean.shape or np.any(std <= 0):
                    raise ValueError(f"Invalid {language} {field}normalization dimensions/std")
                if not np.isfinite(mean).all() or not np.isfinite(std).all():
                    raise ValueError("Normalization is outside the FP32 range")
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Model contains nonfinite parameters")

    @classmethod
    def from_pretrained(cls, model_dir="models", *, weights="acoustic_fp16.safetensors", device="cpu"):
        root = Path(model_dir)
        vocabulary = _read_json(root / "vocabulary.json")
        normalization = _read_json(root / "normalization.json")
        path = root / weights
        if path.suffix == ".safetensors":
            state = load_safetensors(path)
        elif path.suffix == ".pt":
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("Weights must use .safetensors or a weights-only .pt state dictionary")
        model = Acoustic(len(vocabulary))
        if not isinstance(state, Mapping) or set(state) != set(model.state_dict()):
            raise ValueError("Expected exactly the 97 acoustic state keys, without acoustic. prefix")
        for name, reference in model.state_dict().items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape:
                raise ValueError(f"Weight shape mismatch: {name}")
            if tensor.dtype not in (torch.float16, torch.float32) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid weight dtype/value: {name}")
        model.load_state_dict(state, strict=True)
        return cls(model, normalization, vocabulary, device)

    @torch.inference_mode()
    def predict(self, phones, phonetic, context, language, *, pause=None, durations=None, chunk_frames=None):
        """Predict from raw original-frontend arrays, optionally using true durations.

        `phones` is integer [N], `phonetic` is [N,32|36], and `context` is
        [N,22|34]. Each feature width must match that language's normalization.
        Predicted durations require an explicit binary pause[N] array from the
        original frontend. Explicit durations[N] may include zeros. The returned
        state is acoustic decoder state, not a waveform synthesizer's state.
        """
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}; expected one of {LANGUAGES}")
        phone_ids = _integer_vector(phones, "phones")
        n = len(phone_ids)
        if not 1 <= n <= MAX_FRAMES or np.any(phone_ids >= len(self.vocabulary)):
            raise ValueError("Phone count or vocabulary ID is outside the supported range")
        # 0 is padding in the trained network; inference contains no padded phones.
        if np.any(phone_ids == 0):
            raise ValueError("Single-utterance input must not contain padding phone ID 0")
        norm = self.normalization[language]
        inputs = {}
        for name, value in (("phonetic", phonetic), ("context", context)):
            value = _finite_array(value, name).astype(np.float32)
            mean, std = np.asarray(norm[name + "_mean"], np.float32), np.asarray(norm[name + "_std"], np.float32)
            if value.shape != (n, len(mean)):
                raise ValueError(f"{name} must have shape [{n},{len(mean)}] for {language}")
            value = (value - mean) / std
            if not np.isfinite(value).all():
                raise ValueError(f"Normalized {name} is nonfinite")
            inputs[name] = torch.from_numpy(value.copy())[None].to(self.device)
        phone_tensor = torch.from_numpy(phone_ids)[None].to(self.device)
        language_tensor = torch.tensor([LANGUAGES.index(language)], device=self.device)
        if chunk_frames is not None and (type(chunk_frames) is not int or chunk_frames < 1):
            raise ValueError("chunk_frames must be a positive integer")
        if pause is not None:
            pause = _finite_array(pause, "pause")
            if pause.shape != (n,) or np.any((pause != 0) & (pause != 1)):
                raise ValueError(f"pause must have shape [{n}] and contain only 0 or 1")
        with torch.autocast(device_type=self.device.type, enabled=False):
            encoded, prosody = self.model.encode(
                phone_tensor, inputs["phonetic"], inputs["context"], language_tensor,
                torch.ones((1, n), device=self.device),
            )
            if not torch.isfinite(encoded).all() or not torch.isfinite(prosody).all():
                raise ValueError("Nonfinite acoustic encoder output")
            if durations is None:
                if pause is None:
                    raise ValueError("Predicted durations require an original-frontend pause array")
                pause_tensor = torch.as_tensor(pause.copy(), dtype=torch.float32, device=self.device)[None]
                duration_array = predict_durations(prosody, pause_tensor)[0].cpu().numpy()
            else:
                duration_array = _integer_vector(durations, "durations", n)
            frame_phone, position = frame_layout(duration_array)
            fp = torch.from_numpy(frame_phone)[None].to(self.device)
            pos = torch.from_numpy(position)[None].to(self.device)
            dimension, frames = len(norm["mean"]), len(frame_phone)
            step = frames if chunk_frames is None else chunk_frames
            outputs, state = [], None
            for offset in range(0, frames, step):
                output, state = self.model.decode(encoded, prosody, fp[:, offset:offset + step],
                    pos[:, offset:offset + step], language_tensor, dimension, state)
                outputs.append(output)
            prediction = torch.cat(outputs, dim=1)
            physical = denormalize(prediction, norm)
            normalized = waveform_condition(physical, norm)
        if any(not torch.isfinite(x).all() for x in (prediction, physical, normalized, state["gru"], *state["conv"])):
            raise ValueError("Nonfinite acoustic decoder output/state")
        array = lambda tensor: tensor.detach().cpu().numpy().copy()
        return AcousticResult(language, array(physical[0]), array(normalized[0]), array(prediction[0]),
            duration_array.copy(), frame_phone, position,
            {"gru": array(state["gru"]), "conv": [array(x) for x in state["conv"]]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="NPZ: phones, raw phonetic/context, and durations or pause")
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--output", required=True, help="Destination acoustic NPZ (no audio)")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--weights", default="acoustic_fp16.safetensors")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-frames", type=int)
    parser.add_argument("--predict-durations", action="store_true", help="Ignore optional durations in the input NPZ; requires pause")
    args = parser.parse_args(argv)
    try:
        with np.load(args.features, allow_pickle=False) as data:
            missing = {"phones", "phonetic", "context"} - set(data.files)
            if missing:
                raise ValueError(f"Missing frontend arrays: {sorted(missing)}")
            if "language" in data and str(data["language"].item()) != args.language:
                raise ValueError("NPZ language disagrees with --language")
            features = {k: data[k].copy() for k in ("phones", "phonetic", "context", "pause", "durations") if k in data}
        if args.predict_durations:
            features.pop("durations", None)
        predictor = AcousticPredictor.from_pretrained(args.model_dir, weights=args.weights, device=args.device)
        result = predictor.predict(language=args.language, chunk_frames=args.chunk_frames, **features)
        result.save(args.output)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "language": result.language,
        "frames": len(result.physical), "dimension": result.physical.shape[-1],
        "seconds": len(result.physical) * HOP_SIZE / SAMPLE_RATE,
        "frontend_mode": FRONTEND_MODE, "audio_generated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
