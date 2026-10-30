"""Current-only, uncalibrated M55 operation sensitivities for the 25-language run.

No historical execution or audio reference is invented. All coefficients and
accounting policies match opt300b's verified single-row whole-matrix estimator.
QEMU host seconds never determine target cycles.
"""
from pathlib import Path
import argparse
import copy
import inspect
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
M55 = ROOT.parent
CLOCK = 200_000_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def add_callers_current(report, m, integrated):
    """Same current-execution formulas as add_callers, without baseline fields."""
    fusion = integrated.required(m, 'chain_first_fusion_', (
        'fg_gru_groups', 'acoustic_gru_groups', 'glu_groups',
        'vector_loads', 'vector_stores', 'vector_operations'))
    fg, acoustic, glu = (fusion[n] for n in (
        'fg_gru_groups', 'acoustic_gru_groups', 'glu_groups'))
    assert fusion['vector_loads'] == 7*fg+7*acoustic+2*glu
    assert fusion['vector_stores'] == fg+2*acoustic+glu
    assert fusion['vector_operations'] == 7*fg+8*acoustic+glu
    assert not m.get('chain_first_all_i8_provided_absmax_elements', 0)
    noise = integrated.required(m, 'chain_first_noise_', (
        'groups', 'vector_loads', 'vector_stores', 'vector_operations',
        'scalar_lanes', 'MT_scalar_twists', 'draws'))
    assert noise['draws'] == 4*noise['groups']+noise['scalar_lanes']
    assert noise['scalar_lanes'] == 0
    assert noise['vector_loads'] == 2*noise['groups']
    assert noise['vector_stores'] == noise['groups']
    assert noise['vector_operations'] == 20*noise['groups']
    validations = integrated.required(m, 'chain_first_fast_math_', (
        'vector_validation_groups', 'vector_validation_fallbacks'))
    assert validations['vector_validation_groups'] == (
        m['chain_first_fast_math_vector_groups']+
        validations['vector_validation_fallbacks'])
    calls = noise['MT_scalar_twists']
    groups = m['chain_first_noise_twist_vector_groups']
    tails = m['chain_first_noise_twist_scalar_iterations']
    assert groups == 155*calls and tails == 4*calls
    for s in report['scenarios']:
        c = s['coefficients']
        resource = max if c['overlap'] else lambda a, b: a+b
        arithmetic = fusion['vector_operations']*c['vector_operation']
        memory = (fusion['vector_loads']+fusion['vector_stores'])*c['vector_load']
        control = (fg+acoustic+glu)*(2 if c['overlap'] else 6)
        caller = resource(arithmetic, memory)+control
        fallback = (validations['vector_validation_fallbacks']*
                    c['lut_group_validation_control'])
        s['component_cycles']['measured_GRU_GLU_caller'] = caller
        s['component_cycles']['LUT_fallback_group_validation'] = fallback
        s['fusion_caller_breakdown'] = dict(**fusion,
            arithmetic_cycles=arithmetic, memory_cycles=memory,
            group_control_cycles=control, cycles=caller,
            previous_LUT_epilogue_allowance_retained=True,
            possible_overlap_double_count_not_used_as_speed_credit=True)
        s['LUT_validation_breakdown'] = dict(**validations,
            successful_group_old_validation_allowance_retained=True,
            fallback_group_extra_validation_cycles=fallback)
        integrated.rebuild_total(s)
        n_arithmetic = noise['vector_operations']*c['vector_operation']
        n_memory = (noise['vector_loads']+noise['vector_stores'])*c['vector_load']
        n_control = noise['groups']*(2 if c['overlap'] else 6)
        tail_rate = 30 if c['overlap'] else 60
        word = 2 if c['overlap'] else 4
        twist_arithmetic = groups*9*c['vector_operation']
        twist_memory = groups*4*c['vector_load']
        twist_control = groups*(2 if c['overlap'] else 6)
        twist_tail = tails*tail_rate
        twist_setup = calls*(16*c['vector_operation']+32*word)
        twist = (resource(twist_arithmetic, twist_memory)+twist_control+
                 twist_tail+twist_setup)
        n_total = resource(n_arithmetic, n_memory)+n_control+twist
        s['newly_counted_noise_separate_scope'] = dict(**noise,
            vector_arithmetic_cycles=n_arithmetic, vector_memory_cycles=n_memory,
            vector_group_control_cycles=n_control,
            scalar_twist_iterations=tails,
            assumed_scalar_twist_cycles_per_iteration=tail_rate,
            scalar_twist_allowance_cycles=twist_tail,
            cycles=n_total, seconds=n_total/CLOCK,
            scalar_twist_is_visible_allowance_not_audited_cycle_count=True,
            seed_and_other_scalar_overhead_remain_incomplete=True,
            MT_vector_twist_audit=dict(calls=calls, vector_groups=groups,
                scalar_tail_iterations=tails,
                vector_arithmetic_cycles=twist_arithmetic,
                vector_memory_cycles=twist_memory,
                vector_control_allowance_cycles=twist_control,
                scalar_tail_allowance_cycles=twist_tail,
                setup_save_restore_allowance_cycles=twist_setup,
                new_twist_cycles=twist,
                scalar_tail_cycles_per_iteration=tail_rate))
        s['expanded_scope_including_noise'] = dict(
            subtotal_cycles=s['known_and_budgeted_subtotal_cycles']+n_total,
            subtotal_seconds=(s['known_and_budgeted_subtotal_cycles']+n_total)/CLOCK,
            unknown_extra_cycles=None, guaranteed_bound=False,
            scope='Current execution subtotal plus measured RNG work and explicit scalar allowances; no historical execution comparison.')
    report['fusion_snapshot'] = fusion
    report['noise_snapshot'] = noise
    return report


def apply_depthwise(s,m):
    """Replace only priced scalar tap MAC/reduction work, charge new work."""
    p='chain_first_depthwise_'
    d={k[len(p):]:v for k,v in m.items() if k.startswith(p)}
    replaced=d['original_scalar_tap_MAC_replaced']
    assert replaced==4*d['vector_multiply_groups']
    assert d['vector_add_groups']==2*d['vector_multiply_groups']
    assert replaced<=m['chain_first_cost_scalar_mac']
    assert replaced<=m['chain_first_cost_reductions']
    c=s['coefficients'];good=c['overlap'];scalar=1 if good else 2
    resource=max if good else lambda a,b:a+b
    removed=replaced*((2 if good else 6)+(6 if good else 12))
    # Actual SIMD instructions; no separate credit for fewer diagnostics,
    # scalar calls, descriptor reads or redundant zero reductions.
    fp=(d['vector_multiply_groups']+d['vector_add_groups'])*c['vector_operation']
    memory=(d['vector_loads']+d['vector_stores'])*c['vector_load']
    guard_fp=d['range_scan_vector_operations']*c['vector_operation']
    guard_memory=d['range_scan_vector_loads']*c['vector_load']
    guard_control=d['range_scan_vector_groups']*(2 if good else 6)
    guard_reductions=d['range_scan_reductions']*(4 if good else 8)
    guard_tail=d['range_scan_scalar_elements']*(10 if good else 20)
    guard_setup=d['range_scan_calls']*(12*scalar+8*(2 if good else 4))
    # Explicit retained allowances for row/tap index/address/loop work and
    # helper/channel control. They are not calibrated instruction latencies.
    helper_control=(8*d['vector_multiply_groups']+8*d['vector_groups']+
        6*d['input_frame_validity_checks']+4*d['input_frame_pointers'])*scalar
    new=resource(fp,memory)+resource(guard_fp,guard_memory)+guard_control+guard_reductions+guard_tail+guard_setup+helper_control
    assert s['component_cycles']['remaining_FP32']>=removed
    s['component_cycles']['remaining_FP32']-=removed
    s['component_cycles']['exact_depthwise_MVE_guard_and_control']=new
    s['depthwise_replacement_audit']=dict(actual_snapshot=d,
        removed_old_priced_scalar_MAC_and_reduction_cycles=removed,
        SIMD_arithmetic_cycles=fp,SIMD_load_store_cycles=memory,
        guard_vector_arithmetic_cycles=guard_fp,guard_vector_memory_cycles=guard_memory,
        guard_loop_cycles=guard_control,guard_horizontal_reduction_cycles=guard_reductions,
        guard_scalar_tail_cycles=guard_tail,guard_call_setup_allowance_cycles=guard_setup,
        helper_address_and_loop_allowance_cycles=helper_control,new_cycles=new,
        delta_cycles=new-removed,diagnostic_counter_saving_credited=False,
        unpriced_descriptor_and_name_lookup_saving_credited=False,
        helper_allowance_scalar_slots_per_tap_vector=8,
        helper_allowance_scalar_slots_per_output_vector=8,
        scalar_cycles_per_slot_assumed=scalar)
    total=sum(s['component_cycles'].values())
    s['known_and_budgeted_subtotal_cycles']=total
    s['known_and_budgeted_subtotal_seconds']=total/CLOCK
    s['ratio_to_60Mcycle_budget']=total/60_000_000
    s['subtotal_exceeds_60Mcycles']=total>60_000_000
    noise=s['newly_counted_noise_separate_scope']['cycles']
    s['expanded_scope_including_noise']['subtotal_cycles']=total+noise
    s['expanded_scope_including_noise']['subtotal_seconds']=(total+noise)/CLOCK
    s['bottleneck_ranking']=[dict(component=k,cycles=v,seconds=v/CLOCK,share=v/total)
        for k,v in sorted(s['component_cycles'].items(),key=lambda item:item[1],reverse=True)]

def analyze(path):
    path = Path(path)
    if path.is_dir():
        path = path/'RESULT.json'
    paths = dict(
        integrated=M55/'opt300_20261030/estimate_integrated.py',
        packed=M55/'opt300_20261030/packed_kernel/scripts/estimate_packed.py',
        matrix=M55/'opt300c_20261030/pair_relayout/scripts/matrix_adjust.py',
        lut=M55/'opt300b_20261030/lut_fusion/estimate_lut.py')
    integrated = module('language25_integrated', paths['integrated'])
    packed = module('language25_packed', paths['packed'])
    matrix = module('language25_matrix', paths['matrix'])
    lut = module('language25_lut', paths['lut'])
    m = integrated.metrics(path)
    prefix = 'chain_first_all_i8_'
    shared=4*m[prefix+'pair_block_iterations']
    assert m[prefix+'pair_helper_calls']==0
    assert m[prefix+'dot_instructions']==m[prefix+'weight_vector_loads']+shared
    source=inspect.getsource(packed.base.integer_budget)
    needle="if i['dot_instructions'] != lanes*i['weight_vector_loads']:"
    assert needle in source
    packed.base.PAIR_SHARED_DOTS=shared
    exec(source.replace(needle,"if i['dot_instructions'] != lanes*i['weight_vector_loads'] + PAIR_SHARED_DOTS:"),packed.base.__dict__)
    assert m[prefix+'loop_iterations'] == m[prefix+'activation_vector_loads']
    conservative = packed.analyze(path)
    actual = integrated.actual_control_sensitivity(copy.deepcopy(conservative), m)
    modes = {}
    for name, report, actual_mode in [
            ('conservative_allowance', conservative, False),
            ('actual_loop_control_sensitivity', actual, True)]:
        report = add_callers_current(report, m, integrated)
        report = matrix.adjust(report, m, actual_mode)
        for s in report['scenarios']:
            # Preserve the zero-valued component in the opt300b selected
            # no-pair implementation to keep direct formula checks explicit.
            s['component_cycles']['paired_pointer_and_wrapper_allowance'] = 0
            s['paired_audit'] = dict(matrix_pairs=m[prefix+'pair_matrix_pairs'], helper_calls=0,
                input16_block_iterations=m[prefix+'pair_block_iterations'], weight_vectors_shared=shared,
                extra_weight_pointer_adds=0, extra_cycles=0,
                selected_dispatch='Continuous group4 layout; whole single and paired matrix ASM')
        modes[name] = {}
        for lut_mode in ['old_rates_two_eliminated_ops_only',
                         'measured_vector_work_visible_scalar_allowances']:
            current = copy.deepcopy(report)
            for s in current['scenarios']:
                lut.apply_lut_budget(s, m, lut_mode)
                apply_depthwise(s,m)
                assert 'same_historical_scope_reference' not in s
            modes[name][lut_mode] = current
    paths.update(script=Path(__file__),
        integer=M55/'acceleration_20261030/estimate_integer.py',
        legacy_coefficients=M55/'native/analyze_fargan.py')
    return dict(status='CURRENT_ONLY_INCOMPLETE_UNCALIBRATED_COMBINED_SENSITIVITY',
        clock_hz=CLOCK, result=str(path.resolve()), result_sha256=sha(path),
        estimator_sources={str(p.resolve()): sha(p) for p in paths.values()},
        modes=modes, baseline_comparison_performed=False,
        audio_identity_reference=None, coefficients_fitted_to_target=False,
        guaranteed_bounds=False, unknown_extra_cycles=None,
        complete_end_to_end_latency=False, target_300ms_accepted=False,
        QEMU_time_scaling=False,
        missing_scope=[
            'This current-only estimate does not assert historical audio identity or a historical speedup for unmeasured languages.',
            'Native full text frontend, scalar identifier controls, seed/deemphasis bookkeeping, real external-memory waits, cold decompression and audio transport remain incomplete.',
            'Loop and LUT variants are sensitivities with explicit scalar/setup/spill allowances, not calibrated cycles or complete dynamic instruction traces.',
            'The shared scope excludes RNG. Expanded scope adds measured RNG work and the original vector in-place twist with 155 vector groups and four scalar tails.',
            'BWE broad remaining allowance is unchanged; no guessed speed credit for rational vector math or gain hoisting.'
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.report)
    folder = args.report if args.report.is_dir() else args.report.parent
    output = args.output or folder/'LANGUAGE_ESTIMATE_200MHZ.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name: {lut: [dict(name=s['name'],
        shared_ms=1000*s['known_and_budgeted_subtotal_seconds'],
        expanded_noise_ms=1000*s['expanded_scope_including_noise']['subtotal_seconds'])
        for s in r['scenarios']] for lut, r in variants.items()}
        for name, variants in result['modes'].items()}, indent=2))
