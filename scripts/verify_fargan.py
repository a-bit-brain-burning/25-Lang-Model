#!/usr/bin/env python3
"""Verify the FARGAN-only release and its existing acoustic dependency, no torch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    model_dir = root / "models/fargan"
    export = json.loads((model_dir / "EXPORT.json").read_text())
    for name, record in export["files"].items():
        path = model_dir / name
        if not path.resolve().is_relative_to(model_dir) or path.is_symlink():
            raise ValueError("Invalid FARGAN export path")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("FARGAN export mismatch: " + name)
    blob = (model_dir / "vocoder_adapter_fp16.safetensors").read_bytes()
    size = struct.unpack("<Q", blob[:8])[0]
    if not 2 <= size <= min(len(blob) - 8, 1024 * 1024):
        raise ValueError("Invalid FARGAN safetensors header")
    tensors = json.loads(blob[8:8 + size])
    counts = {"generator": {"tensors": 0, "parameters": 0},
              "adapter": {"tensors": 0, "parameters": 0}}
    ranges = []
    for name, item in tensors.items():
        component = name.split(".", 1)[0]
        if component not in counts or item["dtype"] != "F16":
            raise ValueError("Unexpected FARGAN component/dtype: " + name)
        count = math.prod(item["shape"])
        begin, end = item["data_offsets"]
        if begin < 0 or end - begin != count * 2:
            raise ValueError("FARGAN tensor byte count mismatch: " + name)
        counts[component]["tensors"] += 1
        counts[component]["parameters"] += count
        ranges.append((begin, end))
    if counts != {"generator": {"tensors": 28, "parameters": 817525},
                  "adapter": {"tensors": 63, "parameters": 1891412}}:
        raise ValueError("FARGAN tensor topology differs")
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            raise ValueError("FARGAN overlapping/gapped tensor ranges")
        cursor = end
    if cursor != 5417874 or cursor != len(blob) - 8 - size:
        raise ValueError("Unexpected FARGAN payload/trailing data")
    if hashlib.sha256(blob[8 + size:]).hexdigest() != export["source_raw_fp16_sha256"]:
        raise ValueError("FARGAN FP16 bytes differ from selected export")
    dependency = export["acoustic_dependency"]
    acoustic_export = json.loads((root / "models/EXPORT.json").read_text())
    if dependency["source_checkpoint_sha256"] != acoustic_export["source_acoustic_checkpoint_sha256"]:
        raise ValueError("Acoustic source checkpoint differs")
    for name, key in (("acoustic_fp16.safetensors", "fp16_weights_sha256"),
                      ("vocabulary.json", "vocabulary_sha256"),
                      ("normalization.json", "normalization_sha256")):
        if sha256(root / "models" / name) != dependency[key]:
            raise ValueError("Existing acoustic dependency mismatch: " + name)
    upstream = json.loads((root / "third_party/opus/PROVENANCE.json").read_text())
    for name, key in (("COPYING", "copying_sha256"),
                      ("LICENSE_PLEASE_READ.txt", "patent_notice_sha256")):
        if sha256(root / "third_party/opus" / name) != upstream[key]:
            raise ValueError("Upstream license/notice changed")
    for name, record in upstream["sources"].items():
        if sha256(root / name) != record["repository_sha256"]:
            raise ValueError("Pinned FARGAN source changed: " + name)
    return {"status": "FARGAN_LAYOUT_SOURCE_SHA_AND_ACOUSTIC_DEPENDENCY_VERIFIED",
            "tensors": len(tensors), "parameters": 2708937, "fp16_tensor_bytes": cursor,
            "new_acoustic_weights": 0, "complete_fargan_chain_parameters": 3895008,
            "complete_fargan_chain_fp16_tensor_bytes": 7790016,
            "quality_or_ram_acceptance": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
