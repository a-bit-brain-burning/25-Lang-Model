"""Read-only static release checks for original Opus BWE and waveform slots.

Standard library only. No package imports, C build, model construction, audio
processing, inference, training, evaluation or benchmark is executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_functions(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("quantize", "save_wav"):
            result[node.name] = node
        if isinstance(node, ast.ClassDef) and node.name == "OpusBWE24":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                    result["OpusBWE24." + item.name] = item
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    package = ROOT / "new25_postprocess"
    assets = package / "assets/opus_bwe"
    manifest = read(package / "assets/BWE_ASSET_MANIFEST.json")
    proof = read(ROOT / "receipts/opus_bwe24_provenance.json")
    assert manifest["model_commit"] == "ad4799a75530903cd128e23e38c250602f7cea61"
    assert len(manifest["files"]) == 72
    checked = []
    for name, expected in manifest["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        path = assets / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], name
        checked.append(str(path.relative_to(ROOT)))
    assert sha(package / "assets/BWE_ASSET_MANIFEST.json") == proof["asset_manifest_sha256"]
    assert sha(package / "bwe24.py") == proof["published_bwe_module_sha256"]
    source = (package / "bwe24.py").read_text()
    tree = ast.parse(source)
    nodes = original_functions(tree)
    expected = proof["unchanged_original_numeric_and_build_functions_ast_sha256"]
    assert set(nodes) == set(expected)
    for name, node in nodes.items():
        value = ast.dump(node, include_attributes=False).encode()
        assert hashlib.sha256(value).hexdigest() == expected[name], name
    parsed = []
    for path in list(package.rglob("*.py")) + [ROOT / "new25_rndvoc/inference.py", Path(__file__)]:
        ast.parse(path.read_text(), filename=str(path))
        parsed.append(str(path.resolve().relative_to(ROOT)))
    wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RNDVoCWithBWE24")
    ctor = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert [x.arg for x in ctor.args.args] == ["self", "vocoder", "postprocessor"]
    assert [x.arg for x in ctor.args.kwonlyargs] == ["stages"]
    assert ast.literal_eval(ctor.args.kw_defaults[0]) == ()
    chain = ast.parse((package / "chain.py").read_text())
    classes = {n.name: n for n in chain.body if isinstance(n, ast.ClassDef)}
    assert {"PCM16Stage", "CallablePCM16Stage"} <= set(classes)
    for cls in classes.values():
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        assert {"reset", "process_pcm16"} <= set(methods)
        proc = methods["process_pcm16"]
        assert [x.arg for x in proc.args.args] == ["self", "pcm"]
        assert [x.arg for x in proc.args.kwonlyargs] == ["sample_rate"]
    assert any(isinstance(n, ast.FunctionDef) and n.name == "run_pcm16_stages" for n in chain.body)
    assert "from .chain import run_pcm16_stages" in source
    assert "root = Path(__file__).resolve().parent" in source
    assert "root/'assets/BWE_ASSET_MANIFEST.json'" in source
    project = (ROOT / "pyproject.toml").read_text()
    assert '"new25_postprocess*"' in project
    assert 'postprocess = ["scipy>=1.11,<2"]' in project
    block = re.search(r"^new25_postprocess = (\[.*?^\])", project, re.M | re.S)
    assert block
    package_data = set(ast.literal_eval(block.group(1)))
    actual_data = {str(p.relative_to(package)) for p in (package / "assets").rglob("*") if p.is_file()}
    assert actual_data == package_data and len(package_data) == 73
    interface = (ROOT / "new25_rndvoc/inference.py").read_text()
    assert 'parser.add_argument("--bwe24", action="store_true"' in interface
    assert 'parser.add_argument("--postprocess-cache", type=Path' in interface
    assert "from new25_postprocess import OpusBWE24, RNDVoCWithBWE24" in interface
    for path in [ROOT / "README.md", ROOT / "docs/POSTPROCESS_MODULES.md", ROOT / "docs/RNDVOC_INFERENCE.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#")[0]).resolve()
            receipt = args.receipt.resolve() if args.receipt else None
            assert destination.exists() or destination == receipt, (path, target)
    assert proof["prior_runtime_validation"]["status"] == "ORIGINAL_BWE_WRAPPER_EQUAL_LENGTH_AND_RESET_PASSED"
    assert proof["runtime_inference_training_evaluation_or_build_rerun_for_push"] is False
    assert proof["new_intermediate_stage_runtime_validated"] is False
    result = {"status": "ORIGINAL_BWE24_AND_MODULE_SLOTS_STATIC_CHECKS_PASSED",
              "original_model_assets_sha_verified": len(checked), "package_data_paths_verified": len(package_data),
              "preserved_original_function_AST": len(expected), "python_sources_ast_parsed": parsed,
              "cli_and_sdk_signatures_and_callers_checked": True, "documentation_links_checked": True,
              "third_party_license_bytes_preserved": True, "default_middle_stages": [],
              "new_stage_protocol_runtime_validated": False, "package_or_model_imported": False,
              "C_build_or_audio_processing_run": False, "training_inference_evaluation_benchmark_run": False,
              "prior_35k_runtime_evidence_reused": True, "repository_rndvoc_checkpoint_still": 25000}
    if args.receipt:
        with args.receipt.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
