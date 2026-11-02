"""Inference-only removal of repeated original FARGAN weight normalization.

Load the released checkpoint first, then call freeze_weight_norm(vocoder)
before opening streams. This changes the in-memory parameter representation,
not the effective weights, sig_net arithmetic, quantization noise or RNG calls.
It does not write model files, and its state_dict is not the original archive
format. Reload the original archive when an unfrozen model is required.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.weight_norm import WeightNorm


def _identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


@torch.no_grad()
def freeze_weight_norm(vocoder):
    """Fold original legacy WeightNorm hooks once, retaining all other hooks.

    Effective weights are recomputed from the loaded g/v parameters, because
    module.weight can still be stale immediately after load_state_dict().
    Every resulting tensor is checked byte-for-byte against that exact value.
    The function is idempotent and requires all modules to be in eval mode.
    """
    if not isinstance(vocoder, nn.Module):
        raise TypeError("Expected an already-loaded torch module")
    modules = list(vocoder.named_modules())
    if any(module.training for _, module in modules):
        raise ValueError("Weight normalization may only be frozen in eval mode")
    rng_before = torch.random.get_rng_state().clone()
    state_before = {name: _identity(value) for name, value in vocoder.state_dict().items()}
    modes_before = {name: module.training for name, module in modules}
    others_before = {}
    targets = []
    replaced_keys = set()
    for path, module in modules:
        others_before[path] = {
            "pre": {key: hook for key, hook in module._forward_pre_hooks.items()
                    if not isinstance(hook, WeightNorm)},
            "post": dict(module._forward_hooks),
        }
        for _, hook in list(module._forward_pre_hooks.items()):
            if not isinstance(hook, WeightNorm):
                continue
            weight = hook.compute_weight(module)
            if weight.dtype != torch.float32 or not torch.isfinite(weight).all():
                raise ValueError("Expected finite FP32 effective FARGAN weights")
            name = hook.name
            prefix = path + "." if path else ""
            replaced_keys.update((prefix + name + "_g", prefix + name + "_v"))
            requires_grad = (getattr(module, name + "_g").requires_grad
                             or getattr(module, name + "_v").requires_grad)
            targets.append((path, module, name, _identity(weight),
                            weight.numel(), requires_grad))
    folded = []
    for path, module, name, expected, elements, requires_grad in targets:
        remove_weight_norm(module, name=name)
        result = getattr(module, name)
        result.requires_grad_(requires_grad)
        actual = _identity(result)
        if actual != expected:
            raise RuntimeError("Effective weight changed during folding: " + path + "." + name)
        folded.append({"module": path, "parameter": name, "elements": elements,
                       "effective_weight": actual, "byte_identical": True})
    state_after = {name: _identity(value) for name, value in vocoder.state_dict().items()}
    for name, value in state_before.items():
        if name not in replaced_keys and state_after.get(name) != value:
            raise RuntimeError("Unrelated model tensor changed: " + name)
    expected_keys = set(state_before) - replaced_keys
    expected_keys.update((path + "." if path else "") + name
                         for path, _, name, _, _, _ in targets)
    if set(state_after) != expected_keys:
        raise RuntimeError("Unexpected model parameter layout after folding")
    for path, module in modules:
        if (dict(module._forward_pre_hooks) != others_before[path]["pre"]
                or dict(module._forward_hooks) != others_before[path]["post"]
                or module.training != modes_before[path]):
            raise RuntimeError("Unrelated hooks or module mode changed: " + path)
    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise RuntimeError("Weight normalization folding changed the caller CPU RNG")
    return {
        "optimization": "freeze_original_weight_norm_for_inference",
        "removed_weight_norm_hooks": len(folded),
        "effective_weight_elements": sum(item["elements"] for item in folded),
        "effective_weights_byte_identical": True,
        "unrelated_tensors_and_hooks_unchanged": True,
        "caller_cpu_rng_unchanged": True,
        "in_memory_parameterization_changed": bool(folded),
        "checkpoint_files_written": False,
        "sig_net_and_n_noise_implementation_changed": False,
        "training_updates": 0,
        "new_quantization": False,
        "folded": folded,
    }
