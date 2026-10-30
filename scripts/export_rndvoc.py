"""Serialize the selected, SHA-bound generator and existing E2 adapter.

Standard-library-only tensor archive conversion. No torch import, module
construction, forward pass, training, quality evaluation or benchmark occurs.
The restricted pickle reader accepts only the three globals in this checkpoint.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import pickle
import struct
import zipfile

CHECKPOINT_SHA256 = "45b811b8f2a3ae7eba3742fc2b0115d6898145f42fa5b8c662bb6bf8032e1dd1"
ADAPTER_SOURCE_SHA256 = "1493330513821fdf877d08bbc62963017ebb3c633ceb5767e3d5e242f042d802"
BUFFERS = {"Phi", "PhiT", "PhiTPhi"}


def sha(value):
    return hashlib.sha256(value).hexdigest()


class TensorDescription:
    def __init__(self, storage, offset, shape, stride, *unused):
        self.storage, self.offset = storage, offset
        self.shape, self.stride = tuple(shape), tuple(stride)


class TensorArchiveReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): TensorDescription,
            ("torch", "FloatStorage"): "F32",
        }
        try:
            return allowed[module, name]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"Unsupported archive global: {module}.{name}") from exc

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5
                or value[0] != "storage" or value[1] != "F32"):
            raise pickle.UnpicklingError("Only the selected archive's F32 storages are supported")
        return {"key": value[2], "elements": value[4]}


def dense_bytes(archive, prefix, tensor):
    if not isinstance(tensor, TensorDescription):
        raise ValueError("Expected a tensor description")
    raw = archive.read(prefix + "/data/" + tensor.storage["key"])
    if len(raw) != tensor.storage["elements"] * 4:
        raise ValueError("Storage length mismatch")
    contiguous = []
    step = 1
    for dimension in reversed(tensor.shape):
        contiguous.insert(0, step)
        step *= dimension
    if tuple(contiguous) == tensor.stride:
        begin = tensor.offset * 4
        end = begin + math.prod(tensor.shape) * 4
        if not 0 <= begin <= end <= len(raw):
            raise ValueError("Tensor storage range mismatch")
        result = raw[begin:end]
    else:
        # PhiT can be stored transposed. Emit logical C-order without changing
        # any FP32 value; do not treat storage order as tensor index order.
        parts = []
        for coordinates in itertools.product(*(range(n) for n in tensor.shape)):
            index = tensor.offset + sum(a * b for a, b in zip(coordinates, tensor.stride))
            if not 0 <= index < tensor.storage["elements"]:
                raise ValueError("Strided tensor range mismatch")
            parts.append(raw[index * 4:index * 4 + 4])
        result = b"".join(parts)
    if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", result)):
        raise ValueError("Nonfinite tensor")
    return result


def safetensors(path, tensors, metadata):
    header = {"__metadata__": metadata}
    pieces, offset = [], 0
    for name, (shape, dtype, raw) in tensors.items():
        header[name] = {"shape": list(shape), "dtype": dtype,
                        "data_offsets": [offset, offset + len(raw)]}
        pieces.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = struct.pack("<Q", len(encoded)) + encoded + b"".join(pieces)
    with path.open("xb") as handle:
        handle.write(payload)
    return {"bytes": len(payload), "sha256": sha(payload), "tensor_bytes": offset}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hifigan-weights", type=Path,
                        default=Path("models/hifigan/vocoder_adapter_fp16.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("models/rndvoc"))
    args = parser.parse_args(argv)
    blob = args.checkpoint.read_bytes()
    if sha(blob) != CHECKPOINT_SHA256:
        raise ValueError("Expected exactly the selected step25000 checkpoint")
    output = args.output
    output.mkdir(exist_ok=True, parents=True)
    proof, fp16, fp32 = {}, OrderedDict(), OrderedDict()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        entry = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = entry.rsplit("/", 1)[0]
        if archive.read(prefix + "/byteorder") != b"little":
            raise ValueError("Only little-endian source archives are supported")
        checkpoint = TensorArchiveReader(io.BytesIO(archive.read(entry))).load()
        if checkpoint["counts"]["generator"] != 25000:
            raise ValueError("Checkpoint step mismatch")
        state = checkpoint["generator"]
        if len(state) != 196 or not BUFFERS <= set(state):
            raise ValueError("Expected 193 learned tensors and three projection buffers")
        for name, tensor in state.items():
            raw = dense_bytes(archive, prefix, tensor)
            stored = raw if name in BUFFERS else b"".join(
                struct.pack("<e", value[0]) for value in struct.iter_unpack("<f", raw))
            tag = "F32" if name in BUFFERS else "F16"
            fp16[name] = tensor.shape, tag, stored
            fp32[name] = tensor.shape, "F32", raw
            proof[name] = {"shape": list(tensor.shape), "learned": name not in BUFFERS,
                           "source_fp32_sha256": sha(raw), "stored_dtype": tag,
                           "stored_sha256": sha(stored), "elements": math.prod(tensor.shape)}
    learned = sum(x["elements"] for x in proof.values() if x["learned"])
    buffers = sum(x["elements"] for x in proof.values() if not x["learned"])
    if (learned, buffers) != (713929, 345249):
        raise ValueError("Unexpected generator/buffer element count")
    files = {}
    metadata = {"format": "pt", "component": "rndvoc_lite", "training_step": "25000",
                "source_checkpoint_sha256": CHECKPOINT_SHA256}
    files["vocoder_fp16.safetensors"] = safetensors(output / "vocoder_fp16.safetensors", fp16, metadata)
    files["vocoder_fp32.safetensors"] = safetensors(output / "vocoder_fp32.safetensors", fp32, metadata)
    adapter_blob = args.hifigan_weights.read_bytes()
    if sha(adapter_blob) != ADAPTER_SOURCE_SHA256:
        raise ValueError("Expected the existing SHA-bound E2 adapter bundle")
    size = struct.unpack("<Q", adapter_blob[:8])[0]
    header = json.loads(adapter_blob[8:8 + size])
    adapter = OrderedDict()
    for name, spec in header.items():
        if not name.startswith("physical_to_mel."):
            continue
        begin, end = spec["data_offsets"]
        if spec["dtype"] != "F16":
            raise ValueError("Expected the already published FP16 adapter")
        adapter[name.removeprefix("physical_to_mel.")] = (
            spec["shape"], "F16", adapter_blob[8 + size + begin:8 + size + end])
    if len(adapter) != 63 or sum(math.prod(x[0]) for x in adapter.values()) != 1902992:
        raise ValueError("Adapter count mismatch")
    files["adapter_fp16.safetensors"] = safetensors(output / "adapter_fp16.safetensors", adapter,
        {"format": "pt", "component": "shared_e2_physical_to_mel_adapter", "training_step": "31403"})
    receipt = {"status": "MODEL_ONLY_TENSOR_SERIALIZATION_COMPLETE", "source_checkpoint_sha256": CHECKPOINT_SHA256,
               "source_checkpoint_step": 25000, "source_identity": checkpoint["source_identity"],
               "files": files, "generator_parameters": learned, "nonlearned_buffer_elements": buffers,
               "parameter_tensors": 193, "buffers_fp32": sorted(BUFFERS), "adapter_parameters": 1902992,
               "adapter_source_file_sha256": ADAPTER_SOURCE_SHA256, "tensors": proof,
               "forward_or_model_instantiation_executed": False, "experiments_or_benchmarks_run": False}
    with (output / "TENSOR_PROVENANCE.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "tensors"}))


if __name__ == "__main__":
    main()
