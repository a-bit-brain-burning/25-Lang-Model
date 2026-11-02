"""Summarize one real native FARGAN run; never convert QEMU time to M55 cycles."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
CLOCK = 200_000_000
COEFFICIENT_SOURCE = WORKSPACE / 'm55_firstspeech_packed_20261030/analyze.py'
BWE_COEFFICIENT_SOURCE = WORKSPACE / 'm55_300ms_20261030/estimate_fast.py'
EXPECTED_COEFFICIENTS = [
    ('fast_TCM_SRAM_good_overlap', True, 2, 6, 2, 12, 100, 150, 120, 1),
    ('fast_memory_limited_overlap', False, 6, 12, 6, 30, 300, 450, 360, 2),
]
NEURAL_FIELDS = ('vfma', 'activation_vec', 'dequant', 'vload', 'pack_read',
                 'pack_write', 'iterations', 'reductions', 'scalar_mac',
                 'norm_cells', 'exp', 'erf', 'tanh', 'sqrt', 'activation_scalar')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inherited_coefficients():
    """Require the real historical coefficient table, rather than invent one."""
    for node in ast.walk(ast.parse(COEFFICIENT_SOURCE.read_text())):
        if not isinstance(node, ast.For):
            continue
        try:
            value = ast.literal_eval(node.iter)
        except (ValueError, TypeError, SyntaxError):
            continue
        if value == EXPECTED_COEFFICIENTS:
            return value
    raise ValueError('Historical FP32/MVE coefficient table not found unchanged')


def neural_estimate(c, row):
    name, overlap, loop, reduction, scalar, norm, exp, erf, tanh, pack_wait = row
    fp = 2 * (c['vfma'] + c['activation_vec']) + 4 * c['dequant']
    ls = 2 * c['vload'] + 4 * c['dequant'] + (c['pack_read'] + c['pack_write']) / 8 * pack_wait
    control = c['iterations'] * loop + c['reductions'] * reduction + c['scalar_mac'] * scalar + c['norm_cells'] * norm
    library = c['exp'] * exp + c['erf'] * erf + c['tanh'] * tanh + c['sqrt'] * 16
    cycles = (max(fp, ls) if overlap else fp + ls) + control + library + c['activation_scalar'] * 4
    return dict(name=name, neural_cycles=cycles, arithmetic_cycles=fp,
                load_store_cycles=ls, control_and_scalar_cycles=control,
                counted_math_library_cycles=library,
                neural_coefficients=dict(F_and_LS_overlap=overlap,
                    loop_cycles_per_vector_iteration=loop, reduction_cycles=reduction,
                    scalar_mac_cycles=scalar, norm_cycles_per_cell=norm,
                    expf_cycles=exp, erff_cycles=erf, tanhf_cycles=tanh,
                    sqrtf_cycles=16, packing_wait_factor=pack_wait,
                    scalar_activation_cycles=4))


def sensitivity(m):
    prefix = 'chain_first_cost_'
    missing = [prefix + k for k in NEURAL_FIELDS if prefix + k not in m]
    bwe_keys = ('BWE_first_mve_vectors', 'BWE_first_i8_dots',
                'BWE_first_scalar_macs', 'BWE_first_i8_inputs',
                'first_active_BWE_input_samples', 'first_active_FIR_output_samples',
                'full_chain_first_active_audio_offset24')
    missing += [k for k in bwe_keys if k not in m]
    common = dict(clock_hz=CLOCK, guaranteed_bounds=False, calibrated_on_hardware=False,
                  QEMU_wall_time_scaling_used=False, hardware_300ms_accepted=False,
                  coefficient_source=str(COEFFICIENT_SOURCE),
                  BWE_coefficient_source=str(BWE_COEFFICIENT_SOURCE),
                  coefficients_changed=False,
                  limitations=[
                      'This is an incomplete uncalibrated sensitivity subtotal, not complete end-to-end latency.',
                      'FARGAN RNG, uncounted log/cos/pow or other nonlinear operations and FP16 expansion are not fully priced.',
                      'Neural counters use the actual 24k first-effective snapshot; no 16k snapshot is substituted.',
                      'BWE dense counters are measured. Adaptive filtering/features/nonlinear overhead uses the historical broad allowance.',
                      'Measured BWE adaptive-convolution counters are reported, but not charged again on top of that allowance.',
                      'FP64 41-tap FIR uses the historical 50/100 cycles-per-tap scenarios, not calibrated library timings.',
                      'Dynamic frontend, device drivers, external-memory waits and other uninstrumented bookkeeping are excluded.',
                      'Initial playback offset is shown separately. No leading PCM is removed.',
                  ])
    if missing:
        return dict(common, status='NOT_ESTIMATED_MISSING_ACTUAL_COUNTERS', missing=missing, scenarios=[])
    try:
        rows = inherited_coefficients()
    except (OSError, ValueError, SyntaxError) as error:
        return dict(common, status='NOT_ESTIMATED_COEFFICIENT_SOURCE_UNAVAILABLE', reason=str(error), scenarios=[])
    common['coefficient_source_sha256'] = sha(COEFFICIENT_SOURCE)
    common['BWE_coefficient_source_sha256'] = sha(BWE_COEFFICIENT_SOURCE)
    c = {k: m[prefix+k] for k in NEURAL_FIELDS}
    scenarios = []
    for row in rows:
        overlap = row[1]
        item = neural_estimate(c, row)
        vectors, dots = m['BWE_first_mve_vectors'], m['BWE_first_i8_dots']
        scalars, inputs = m['BWE_first_scalar_macs'], m['BWE_first_i8_inputs']
        bwe_fp, bwe_ls = 4*vectors + 2*dots, 2*(vectors + 1.25*dots)
        bwe_control = vectors/4*(2 if overlap else 6) + dots/4*(4 if overlap else 8)
        bwe_dense = (max(bwe_fp, bwe_ls) if overlap else bwe_fp+bwe_ls) + bwe_control
        bwe_dense += scalars*(6 if overlap else 14) + inputs*(12 if overlap else 30) + dots*(.5 if overlap else 1)
        equivalents = m['first_active_BWE_input_samples']/320
        bwe_allowance = equivalents*(450000*(2 if overlap else 6) + (100000 if overlap else 500000))
        fir = m['first_active_FIR_output_samples']*41*(50 if overlap else 100)
        cycles = item['neural_cycles'] + bwe_dense + bwe_allowance + fir
        offset = m['full_chain_first_active_audio_offset24']/24000
        item.update(BWE_counted_dense_cycles=bwe_dense,
                    BWE_remaining_legacy_allowance_cycles=bwe_allowance,
                    FP64_FIR_allowance_cycles=fir,
                    known_and_budgeted_subtotal_cycles=cycles,
                    known_and_budgeted_subtotal_seconds=cycles/CLOCK,
                    initial_audio_offset_seconds=offset,
                    subtotal_plus_initial_playback_offset_seconds=cycles/CLOCK+offset,
                    uncounted_FARGAN_RNG_and_conversion_cycles=None,
                    BWE_and_FIR_assumptions=dict(
                        signal_MAC_budget_per_320_samples=450000,
                        signal_cycles_per_MAC=2 if overlap else 6,
                        other_cycles_per_320_samples=100000 if overlap else 500000,
                        FIR_cycles_per_tap=50 if overlap else 100))
        scenarios.append(item)
    return dict(common, status='PARTIAL_UNCALIBRATED_SENSITIVITY_ONLY', scenarios=scenarios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='The one completed RESULT.json, or its directory')
    parser.add_argument('--output', type=Path, help='Default: RESULT directory/FARGAN_ANALYSIS.json')
    args = parser.parse_args()
    path = args.result / 'RESULT.json' if args.result.is_dir() else args.result
    if not path.is_file():
        parser.error('Real RESULT.json is not available; no run or estimate was started')
    result = json.loads(path.read_text())
    if result.get('status') != 'PASSED' or len(result.get('repeats', [])) != 1:
        parser.error('Expected exactly one successfully completed run; refusing to synthesize missing data')
    run = path.parent / 'run_001'
    repeat = result['repeats'][0]
    m = {k: values[0] for k, values in repeat['guest_metrics'].items() if values}
    hz = m['tick_frequency_hz']
    if hz <= 0:
        parser.error('Invalid actual guest tick frequency')
    seconds = lambda name: m[name]/hz if name in m else None
    raw16, raw24 = (run/'output.pcm').read_bytes(), (run/'output24.pcm').read_bytes()
    first24 = (run/'first24.pcm').read_bytes() if (run/'first24.pcm').is_file() else b''
    numeric = lambda p: int(re.search(r'_(\d+)\.pcm$', p.name).group(1))
    chunks16 = sorted(run.glob('stream_*.pcm'), key=numeric)
    packets24 = sorted(run.glob('pcm24_*.pcm'), key=numeric)
    frames = m['predicted_frames']
    wanted16 = (frames-4)*160
    wanted24 = (wanted16*3+1)//2
    checks = {
        'physical_frames_at_least5': frames >= 5,
        'acoustic_all_frames_completed': m.get('decoded_frames') == frames,
        'valid_center_source_start400': m.get('source_sample_start16') == 400,
        'valid_center_source_end': m.get('source_sample_end16') == frames*160-240,
        'valid_center_PCM16_length': len(raw16) == wanted16*2 == m.get('pcm_samples', -1)*2,
        'BWE24_original_length': len(raw24) == wanted24*2 == m.get('full_chain_output24_samples', -1)*2,
        'stream16_reconstructs_own_complete_PCM': bool(chunks16) and b''.join(p.read_bytes() for p in chunks16) == raw16,
        'BWE_packets_reconstruct_own_complete_PCM': bool(packets24) and b''.join(p.read_bytes() for p in packets24) == raw24,
        'BWE_packet_numbering_contiguous': [numeric(p) for p in packets24] == list(range(len(packets24))),
        'BWE_packet_lengths20ms_and_original_tail': bool(packets24) and all(p.stat().st_size == 960 for p in packets24[:-1]) and 0 < packets24[-1].stat().st_size <= 960,
        'BWE_packet_counters_match': len(packets24) == m.get('PCM24_packet_count') and len(raw24)//2 == m.get('PCM24_published_samples'),
        'BWE_packets_published_before_EOS': 0 < m.get('PCM24_packets_before_EOS', 0) <= len(packets24),
        'BWE_EOS_exactly_once': m.get('BWE_finish_calls') == 1,
        'first_effective24_is_own_complete_prefix': bool(first24) and raw24.startswith(first24),
        'first_effective24_sample_count': len(first24)//2 == m.get('first_active_FIR_output_samples'),
        'signal_threshold_detected16_and24': m.get('active_frame_detected') == m.get('BWE_active_detected') == 1,
        'arena_no_failed_requests': m.get('arena_failed_requests') == 0,
        'stack_bottom_guard_intact': m.get('stack_bottom_guard_ok') == 1,
    }
    log = (run/'qemu.log').read_text() if (run/'qemu.log').is_file() else ''
    a, b = log.find('full_chain_first_active_PCM24_host_ticks'), log.find('BWE_finish_calls')
    checks['first_effective24_observed_before_EOS'] = 0 <= a < b
    memory_keys = ('image_code_rodata_and_data_initializer_bytes', 'assets_qspi_rom_bytes',
                   'static_rw_bytes', 'arena_reserved_bytes', 'arena_peak_bytes',
                   'stack_reserved_bytes', 'stack_written_high_water_bytes', 'reserved_resident_ram_bytes')
    memory = {k: m.get(k) for k in memory_keys}
    memory['backend_code_and_assets_ROM_bytes'] = m['image_code_rodata_and_data_initializer_bytes'] + m['assets_qspi_rom_bytes']
    memory['complete_system_ROM_RAM_accepted'] = False
    memory['scope'] = 'Measured backend image/arena/stack only; native dynamic frontend and platform are not fully included'
    report = dict(
        status='NATIVE_STREAM_FUNCTIONAL_CHECKS_PASSED' if all(checks.values()) else 'NATIVE_STREAM_CHECKS_FAILED',
        result=str(path.resolve()), result_sha256=sha(path), elf_sha256=result.get('elf_sha256'),
        checks=checks, failed_checks=[k for k, value in checks.items() if not value],
        qemu_host_seconds=dict(first_PCM16=seconds('first_pcm_ready_host_ticks'),
            first_effective_PCM16=seconds('first_active_frame_ready_host_ticks'),
            first_any_PCM24=seconds('full_chain_first_any_PCM24_host_ticks'),
            first_effective_PCM24=seconds('full_chain_first_active_PCM24_host_ticks'),
            complete_firmware=seconds('firmware_total_host_ticks'),
            process_wall=repeat.get('host_process_wall_seconds')),
        QEMU_host_time_is_not_M55_cycles=True, memory=memory,
        original_valid_center=dict(physical_frames=frames, output_frames=frames-4,
            source_sample_start16=400, source_sample_end16=frames*160-240,
            samples16=len(raw16)//2, samples24=len(raw24)//2, audio_seconds=len(raw16)/32000,
            edge_extension=False, teacher_prefill_frames=0),
        clips16=m.get('clipped_samples'), clips24=m.get('BWE_quantizer_clips'),
        packet_count24=len(packets24), packets24_before_EOS=m.get('PCM24_packets_before_EOS'),
        first_effective_audio_offset24_seconds=(m['full_chain_first_active_audio_offset24']/24000
            if 'full_chain_first_active_audio_offset24' in m else None),
        complete_pcm_sha256={'16k': hashlib.sha256(raw16).hexdigest(), '24k': hashlib.sha256(raw24).hexdigest()},
        actual_first_operation_counters={k: v for k, v in m.items() if k.startswith(('first_cost_', 'chain_first_cost_', 'BWE_first_'))},
        sensitivity_200MHz=sensitivity(m), human_quality_accepted=False,
        hardware_latency_verified=False, training_updates=0, waveform_alignment_performed=False)
    output = args.output or path.parent/'FARGAN_ANALYSIS.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'failed_checks', 'qemu_host_seconds', 'memory', 'original_valid_center', 'clips16', 'clips24')}, ensure_ascii=False, indent=2))
    print('Analysis:', output)
    print('200 MHz sensitivity:', report['sensitivity_200MHz']['status'])
    if report['failed_checks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
