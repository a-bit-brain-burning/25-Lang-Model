"""Static RNDVoC release checks; standard library only, no model imports.

Checks source parsing, file paths, export SHA, tensor headers and value bytes.
Does not execute model constructors, inference, training, tests or benchmarks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CP = "45b811b8f2a3ae7eba3742fc2b0115d6898145f42fa5b8c662bb6bf8032e1dd1"
BUFFERS = {"Phi", "PhiT", "PhiTPhi"}


def read(path):
    return json.loads(path.read_text())


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def tensor_file(path):
    blob = path.read_bytes()
    size = struct.unpack("<Q", blob[:8])[0]
    assert 0 < size < min(1024 * 1024, len(blob) - 8)
    header = json.loads(blob[8:8 + size])
    result, spans = {}, []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        assert item["dtype"] in ("F16", "F32")
        assert all(type(n) is int and n >= 0 for n in item["shape"])
        start, end = item["data_offsets"]
        unit = 2 if item["dtype"] == "F16" else 4
        assert end - start == math.prod(item["shape"]) * unit
        assert 0 <= start <= end <= len(blob) - 8 - size
        raw = blob[8 + size + start:8 + size + end]
        assert all(math.isfinite(x[0]) for x in struct.iter_unpack("<e" if unit == 2 else "<f", raw))
        result[name] = dict(item, raw=raw)
        spans.append((start, end))
    end = 0
    for start, stop in sorted(spans):
        assert start == end
        end = stop
    assert end == len(blob) - 8 - size
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    model_dir = ROOT / "models/rndvoc"
    export = read(model_dir / "EXPORT.json")
    assert export["source_checkpoint_sha256"] == EXPECTED_CP
    assert export["source_checkpoint_step"] == 25000
    checked = []
    for name, identity in export["files"].items():
        p = model_dir / name
        assert p.stat().st_size == identity["bytes"] and digest(p.read_bytes()) == identity["sha256"], name
        checked.append(str(p.relative_to(ROOT)))
    proof = read(model_dir / "TENSOR_PROVENANCE.json")
    fp16 = tensor_file(model_dir / "vocoder_fp16.safetensors")
    fp32 = tensor_file(model_dir / "vocoder_fp32.safetensors")
    assert set(fp16) == set(fp32) == set(proof["tensors"])
    assert len(fp16) == 196
    elements = {"learned": 0, "nonlearned": 0}
    for name, item in fp16.items():
        evidence = proof["tensors"][name]
        assert item["shape"] == fp32[name]["shape"] == evidence["shape"]
        assert digest(fp32[name]["raw"]) == evidence["source_fp32_sha256"]
        assert digest(item["raw"]) == evidence["stored_sha256"]
        assert item["dtype"] == ("F32" if name in BUFFERS else "F16")
        assert fp32[name]["dtype"] == "F32"
        elements["nonlearned" if name in BUFFERS else "learned"] += math.prod(item["shape"])
    assert elements == {"learned": 713929, "nonlearned": 345249}
    adapter = tensor_file(model_dir / "adapter_fp16.safetensors")
    published = tensor_file(ROOT / "models/hifigan/vocoder_adapter_fp16.safetensors")
    assert len(adapter) == 63 and sum(math.prod(x["shape"]) for x in adapter.values()) == 1902992
    for name, value in adapter.items():
        old = published["physical_to_mel." + name]
        assert value["shape"] == old["shape"] and value["dtype"] == old["dtype"] == "F16"
        assert value["raw"] == old["raw"]
    sources = sorted((ROOT / "new25_rndvoc").rglob("*.py"))
    for path in sources + [ROOT / "scripts/export_rndvoc.py", ROOT / "scripts/verify_rndvoc.py"]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    inference = ast.parse((ROOT / "new25_rndvoc/inference.py").read_text())
    declaration = next(n for n in inference.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "MODEL_CONFIG" for t in n.targets))
    assert ast.literal_eval(declaration.value) == read(model_dir / "config.json")
    cls = next(n for n in inference.body if isinstance(n, ast.ClassDef) and n.name == "RNDVoCVocoder")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert set(["from_pretrained", "synthesize", "synthesize_mel", "synthesize_chunk"]) <= set(methods)
    assert [n.arg for n in methods["synthesize"].args.args] == ["self", "normalized", "language"]
    assert [n.arg for n in methods["synthesize"].args.kwonlyargs] == ["state", "chunk_frames"]
    assert [n.arg for n in methods["synthesize_mel"].args.args] == ["self", "raw_logmel"]
    upstream = read(ROOT / "third_party/rndvoc/PROVENANCE.json")
    for name, expected in upstream["released_source_files"].items():
        path = ROOT / "new25_rndvoc/_vendor" / name
        assert digest(path.read_bytes()) == expected
        module = ast.dump(ast.parse(path.read_text()), include_attributes=False)
        assert digest(module.encode()) == upstream["module_ast_sha256"][name]
    for path in [ROOT / "README.md", ROOT / "docs/RNDVOC_INFERENCE.md", ROOT / "docs/RNDVOC_MODEL_CARD.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#")[0]).resolve()
            expected_receipt = args.receipt.resolve() if args.receipt else None
            assert destination.exists() or destination == expected_receipt, (path, target)
    for name in ["new25_acoustic/inference.py", "new25_hifigan/inference.py", "new25_hifigan/frontend.py",
                 "new25_hifigan/model.py", "third_party/rndvoc/LICENSE", "requirements-rndvoc.txt"]:
        assert (ROOT / name).is_file()
    metadata = read(model_dir / "model_config.json")
    assert metadata["streaming"] is False and metadata["nonlearned_buffer_dtype"] == "F32"
    assert metadata["native_10MB_ram_pass"] is False and metadata["human_quality_pass"] is False
    result = {"status": "RNDVOC25000_STATIC_RELEASE_CHECKS_PASSED", "checkpoint_step": 25000,
              "checkpoint_sha256": EXPECTED_CP, "model_files_sha_verified": checked,
              "generator_parameter_tensors": 193, "generator_parameters": elements["learned"],
              "fp32_buffer_elements": elements["nonlearned"], "adapter_tensor_byte_equality": 63,
              "source_ast_parsed_files": len(sources) + 2,
              "entry_signatures_paths_dependencies_and_doc_links_checked": True,
              "tensor_values_and_headers_checked_without_torch": True,
              "model_imported_or_instantiated": False, "inference_or_training_run": False,
              "quality_evaluation_or_performance_benchmark_run": False,
              "existing_development_evidence_reused": True}
    if args.receipt:
        with args.receipt.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
