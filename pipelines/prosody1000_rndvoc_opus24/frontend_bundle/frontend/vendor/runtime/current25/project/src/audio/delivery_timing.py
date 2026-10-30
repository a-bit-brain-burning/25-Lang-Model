"""Numeric duration/quiet-span adapter for the existing 31-channel DSP ABI.

The caller supplies unit masks and durations. No text, phone identities, tones,
or language rules enter this module; the DSP algorithm remains unchanged.
"""
import numpy as np


def apply_numeric_delivery(controls, durations, target_durations, quiet_unit_mask,
                           *, quiet_log_energy=-30.):
    x = np.asarray(controls, dtype=np.float32)
    original = np.asarray(durations)
    target = np.asarray(target_durations)
    quiet = np.asarray(quiet_unit_mask, dtype=bool)
    n = len(original)
    if x.shape != (n, 8, 31) or target.shape != (n,) or quiet.shape != (n,):
        raise ValueError('Expected controls[N,8,31], durations[N], targets[N], mask[N]')
    if not np.isfinite(x).all() or not np.isfinite(original).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite numeric delivery input')
    if (np.any(original < 0) or np.any(target < 0)
            or np.any(original != np.floor(original)) or np.any(target != np.floor(target))):
        raise ValueError('Durations must be nonnegative integer frame counts')
    if np.any((original > 0) & (target == 0) & ~quiet):
        raise ValueError('Delivery policy cannot delete an existing nonquiet unit')
    if not -40 <= quiet_log_energy <= -10:
        raise ValueError('Quiet log energy must be within [-40,-10]')
    result = x.astype(np.float32, copy=True)
    # Near-zero-energy unvoiced spans actually produce quiet audio. Merely
    # lengthening a predicted silence unit can otherwise prolong noisy controls.
    result[quiet, :, 1] = np.minimum(result[quiet, :, 1], quiet_log_energy)
    result[quiet, :, 25:30] = 0.
    result[quiet, :, 30] = -12.
    assert np.array_equal(result[~quiet], x[~quiet])
    return result, target.astype(np.int64), {
        'duration_changed_units': int(np.sum(target != original)),
        'quiet_units': int(quiet.sum()), 'quiet_log_energy_ceiling': float(quiet_log_energy),
        'nonquiet_acoustic_controls_exact': True,
        'F0_controls_exact': bool(np.array_equal(result[:, :, 0], x[:, :, 0])),
        'DSP_algorithm_changed': False}
