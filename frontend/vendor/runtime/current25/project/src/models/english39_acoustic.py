"""Explicit English linear31 / remote warped700-39 experiment contracts.

The shared network and native DSP algorithm remain unchanged. A new format
prevents a warped checkpoint from being consumed with the legacy linear basis.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch

from .universal_acoustic import UniversalAcousticModel
from ..experiments.eu_quality_spectrum_basis import (
    encode_shape, native_basis, shape_basis, spec,
)

FORMAT = "english_paired_harvest_acoustic_v1"
MODES = {"linear31": (24, None), "warped39": (32, 700.)}


def contract(mode):
    if mode not in MODES:
        raise ValueError("Unknown English representation")
    dim, knee = MODES[mode]
    return {
        "version": "english_paired_harvest_20261016_v1", "mode": mode,
        "phonetic_dim": 32, "context_dim": 27, "plan_dim": 5,
        "channels": dim + 7, "spectral": spec(dim, knee),
        "channel_names": ["logF0", "log_waveform_energy"]
                         + [f"shape_{i}" for i in range(1, dim)]
                         + [f"AP_{i}" for i in range(5)] + ["VUV_logit"],
        "control_points": 8, "frame_period_ms": 10, "sample_rate": 16000,
        "analysis": "Harvest50-400+StoneMask/CheapTrick1024/D4C1024",
        "AP_edges_Hz": [0, 500, 1500, 3000, 5000, 8000],
        "AP_encoding": "20log10(RMS amplitude ratio)/60",
        "DSP": "existing native kernel, taps128, gain1, seed1729",
        "remote_source_commit": "421570a5a42172109633da350f039baae3eb50d7",
        "postfilter": None,
    }


def spectral_projection(mode):
    """Pure basis projection of existing 23 linear shapes; no fitted calibration."""
    if mode == "linear31":
        return np.eye(23, dtype=np.float64)
    if mode != "warped39":
        raise ValueError(mode)
    # A DC offset cancels in encode_shape; unit source log shapes are sufficient.
    return encode_shape(np.exp(shape_basis(24, None)), 32, 700.).T


def transfer(source, mode, means, scales):
    """Reuse trunk/timing and project the output head into the explicit new basis."""
    if (source.config["phonetic_dim"], source.config["context_dim"],
        source.config["spectral_dim"], source.config["ap_bands"]) != (32, 27, 24, 5):
        raise ValueError("Transfer requires the frozen English P1 source")
    dim, _ = MODES[mode]
    channels = dim + 7
    config = {**source.config, "spectral_dim": dim,
              "english_representation_contract": contract(mode)}
    model = UniversalAcousticModel(config, means, scales)
    old = copy.deepcopy(source.state_dict())
    old_weight = old.pop("acoustic.2.weight").double().reshape(8, 31, -1)
    old_bias = old.pop("acoustic.2.bias").double().reshape(8, 31)
    old_mean = old.pop("means").double()
    old_scale = old.pop("scales").double()
    transform = torch.zeros(channels, 31, dtype=torch.float64)
    transform[:2, :2] = torch.eye(2, dtype=torch.float64)
    transform[2:dim+1, 2:25] = torch.from_numpy(spectral_projection(mode))
    transform[dim+1:-1, 25:30] = torch.eye(5, dtype=torch.float64)
    transform[-1, -1] = 1
    new_mean = torch.as_tensor(means, dtype=torch.float64)
    new_scale = torch.as_tensor(scales, dtype=torch.float64)
    weight = torch.einsum("dc,kch->kdh", transform,
                          old_weight * old_scale[None, :, None]) / new_scale[None, :, None]
    bias = (torch.einsum("dc,kc->kd", transform,
                         old_mean[None] + old_scale[None] * old_bias)
            - new_mean[None]) / new_scale[None]
    old.update({"acoustic.2.weight": weight.reshape(channels*8, -1).float(),
                "acoustic.2.bias": bias.flatten().float(),
                "means": model.means, "scales": model.scales})
    model.load_state_dict(old)
    return model.eval()


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    mode = state.get("mode")
    if state.get("format") != FORMAT or state.get("contract") != contract(mode):
        raise ValueError("English representation/checkpoint contract mismatch")
    config = state["config"]
    if config.get("english_representation_contract") != contract(mode):
        raise ValueError("Model config and representation differ")
    model = UniversalAcousticModel(config, state["model"]["means"], state["model"]["scales"])
    model.load_state_dict(state["model"])
    return model.eval(), state


def render_curve(curve, mode, sample_count=None):
    """Use the peer representation's native basis with the existing DSP kernel."""
    from ..experiments.representation_limit import GenericDSPAdapter
    dim, knee = MODES[mode]
    curve = np.asarray(curve, np.float32)
    if curve.ndim != 2 or curve.shape[1] != dim + 7 or not np.isfinite(curve).all():
        raise ValueError("Curve and spectral basis differ")
    physical = curve.copy()
    voiced = physical[:, -1] > 0
    physical[:, 0] = np.where(voiced, np.exp(physical[:, 0]), 0)
    physical[:, -1] = voiced
    physical[~voiced, dim+1:-1] = 0
    adapter = GenericDSPAdapter(dim, 5)
    adapter.basis = native_basis(dim, knee, 128)
    return adapter.render(physical, sample_count)
