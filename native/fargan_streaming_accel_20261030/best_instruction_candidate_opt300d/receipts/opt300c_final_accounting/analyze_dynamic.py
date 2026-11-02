"""Reproduce the dynamic O2/O3 instruction-event accounting.

Reads only already-produced QEMU plugin CSVs plus ELF disassembly/symbols.
Pure post-processing: no new approximation, no weights, no firmware changes.
Guest instruction events are architectural instruction executions, not cycles.
"""
from pathlib import Path
import collections, csv, hashlib, json, re, subprocess, sys

ROOT = Path('/home/fomak/ffb/m55_fargan_stream_20261030/opt300c_20261030').resolve()
RT = ROOT/'release_trace'
IP = ROOT/'instruction_profile'
OUT = ROOT/'final_accounting'
CLOCK = 200_000_000
TARGET_CYCLES = 60_000_000            # 300 ms at 200 MHz
NM = str(RT/'toolchain/sysroot/usr/bin/arm-none-eabi-nm')
FILES = ['condition.f32','durations.u32','output.pcm','output24.pcm','first24.pcm','fargan.f32']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PATTERN = re.compile(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4}(?: [0-9a-f]{4})?)\s+(.+)$')

def symbols(elf):
    d = {}
    for line in subprocess.check_output([NM,'-n',str(elf)],text=True).splitlines():
        p = line.split()
        if len(p) == 3: d[p[2]] = int(p[0],16)
    return d

def load_disassembly(build):
    functions = {}; instructions = {}; owner = None
    for line in (build/'disassembly.txt').read_text().splitlines():
        m = re.match(r'^([0-9a-f]+) <([^>]+)>:', line)
        if m: owner = m[2]; functions[owner] = int(m[1],16)
        m = PATTERN.match(line)
        if m: instructions[int(m[1],16)] = dict(function=owner, raw=m[2], text=m[3])
    return functions, instructions

def classify(ins):
    """Coarse mnemonic classes. objdump aliases many MVE opcodes as coprocessor
    ops, so this classification is partial and is never used as a cycle model."""
    raw = ins['raw']; text = ins['text']; mnem = text.split()[0]
    if len(raw) == 9 and raw[:4] == 'eef0' and int(raw[5:],16) & 0xfff == 0xf22:
        return 'MVE_signed_I8_dot_opcode'
    if '.f32' in mnem or '.f64' in mnem:
        return 'VFP_vector' if re.search(r'\bq\d+', text) else 'VFP_scalar'
    if mnem.startswith(('vld','vst','vpush','vpop')):
        return 'VFP_or_MVE_data_transfer'
    if mnem.startswith(('ldr','str','ldm','stm','push','pop')):
        return 'scalar_load_store_stack'
    if mnem.startswith(('b','cb')) or mnem in ['dls','wls','le','letp','it','itt','ite','nop']:
        return 'branch_loop_control'
    if mnem in ['cdp','cdp2','ldc','ldc2','stc','stc2']:
        return 'MVE_or_coprocessor_aliased_by_objdump'
    return 'other_scalar_or_vector'

def audit(tag):
    build = RT/'build'/tag; elf = build/'firmware.elf'
    sym = symbols(elf)
    functions, instructions = load_disassembly(build)
    begin = sym['tensor_views_prepare']; end = sym['snapshot_first_active_costs']
    assert functions['tensor_views_prepare'] == begin
    assert functions['snapshot_first_active_costs'] == end
    csvp = IP/('release_'+tag+'_instruction_counts.csv')
    rows = list(csv.DictReader(csvp.open()))
    bypc = {int(r['pc'],16): int(r['request_instruction_events']) for r in rows}
    total = sum(bypc.values())
    assert bypc.get(begin) == 1, ('begin marker events', bypc.get(begin))
    assert bypc.get(end, 0) == 0, ('end marker events', bypc.get(end,0))
    unmatched = [pc for pc,n in bypc.items() if n and pc not in instructions]
    assert not unmatched, unmatched[:3]
    byfn = collections.Counter(); sem = collections.Counter()
    for pc,n in bypc.items():
        if not n: continue
        item = instructions[pc]
        byfn[item['function']] += n
        sem[classify(item)] += n
    assert sum(byfn.values()) == sum(sem.values()) == total
    return dict(tag=tag, elf=str(elf), elf_sha256=sha(elf),
        disassembly_sha256=sha(build/'disassembly.txt'),
        csv=str(csvp), csv_sha256=sha(csvp),
        begin_symbol='tensor_views_prepare', begin_pc=hex(begin),
        end_symbol='snapshot_first_active_costs', end_pc=hex(end),
        begin_marker_instruction_events=1, end_marker_instruction_events=0,
        instruction_events=total,
        uncalibrated_1ipc_floor_ms=1000*total/CLOCK,
        functions=[dict(function=f, events=n, share=n/total)
                   for f,n in byfn.most_common()],
        semantic_classes_partial=dict(sem),
        dot_opcode_events=sem['MVE_signed_I8_dot_opcode'])

def main():
    o2 = audit('ordinary_o2_v1'); o3 = audit('ordinary_o3_v1')
    # six-file identity: O3 normal run vs O2 normal run vs both frozen references
    runs = dict(
        o2=RT/'reports/ordinary_o2_v1/run_001',
        o3=RT/'reports/ordinary_o3_v1/run_001',
        integrated=ROOT/'integrated/reports/final_v1_ordinary/run_001',
        final_native=ROOT/'final_native/reports/final_v1_zh/run_001')
    identity = {}
    for n in FILES:
        b = {k: (runs['o3']/n).read_bytes() for k in runs}
        identity[n] = dict(bytes=len(b['o3']), sha256=sha(runs['o3']/n),
            o3_vs_o2=b['o3']==b['o2'], o3_vs_integrated=b['o3']==b['integrated'],
            o3_vs_final_native=b['o3']==b['final_native'])
        assert identity[n]['o3_vs_o2'] and identity[n]['o3_vs_integrated'] and identity[n]['o3_vs_final_native'], n
    # deterministic profile repeats (CSV byte identity from the two runs)
    determinism = {}
    for tag in ['ordinary_o2_v1','ordinary_o3_v1']:
        cur = sha(IP/('release_'+tag+'_instruction_counts.csv'))
        run1 = sha(IP/('release_'+tag+'_run1_instruction_counts.csv'))
        determinism[tag] = dict(run1_csv_sha256=run1, repeat_csv_sha256=cur, identical=cur==run1)
        assert determinism[tag]['identical'], tag
    profile_runs = {t: json.loads((IP/'reports'/('release_'+t)/'RESULT.json').read_text())
                    for t in ['ordinary_o2_v1','ordinary_o3_v1']}
    for t,r in profile_runs.items(): assert r['status']=='PASSED', t
    o3_delta = o3['instruction_events'] - o2['instruction_events']
    pair = 0
    for row in o3['functions']:
        if row['function'] in ('packed_matrix_pair','packed_matrix8'): pair += row['events']
    audit_out = dict(
        status='RELEASE_O2_O3_DYNAMIC_INSTRUCTION_AUDIT',
        method=('QEMU 8.2.2 mps3-an547 Cortex-M55 single-vCPU TCG; local GPL QEMU plugin '
                'counts architectural guest instruction executions per PC between the '
                'tensor_views_prepare marker entry (inclusive) and the '
                'snapshot_first_active_costs marker entry (exclusive).'),
        roi='one ordinary Chinese complete first-audio request (same sentence and six output files as the frozen native zh case)',
        clock_hz=CLOCK,
        o2=o2, o3=o3,
        o3_vs_o2=dict(delta_events=o3_delta,
            reduction_fraction=-o3_delta/o2['instruction_events'],
            o2_floor_ms=o2['uncalibrated_1ipc_floor_ms'],
            o3_floor_ms=o3['uncalibrated_1ipc_floor_ms']),
        o3_six_file_identity=identity,
        six_file_references=['opt300c/integrated/reports/final_v1_ordinary/run_001',
                             'opt300c/final_native/reports/final_v1_zh/run_001'],
        deterministic_repeat_profiles=determinism,
        profile_run_reports={t: dict(status=r['status'], elf_sha256=r['elf_sha256'],
            result=sha(IP/'reports'/('release_'+t)/'RESULT.json'))
            for t,r in profile_runs.items()},
        top_o3_functions=o3['functions'][:20],
        top_o2_functions=o2['functions'][:20],
        matrix_kernel_events_o3=dict(events=pair, share=pair/o3['instruction_events'],
            functions=['packed_matrix_pair','packed_matrix8']),
        dot_opcode_events_identical=(o2['dot_opcode_events'], o3['dot_opcode_events']),
        assertions=['begin/end marker event counts are 1/0',
                    'all event PCs map to a real ELF instruction',
                    'O3 six files byte-identical to O2 and to both frozen references',
                    'two independent profile runs produce byte-identical CSVs',
                    'MVE signed I8 dot opcode events identical in O2 and O3 (arithmetic unchanged)'],
        caveats=['Guest instruction events are not cycles. No DWT/cycle model, pipeline '
                 'calibration, memory wait, cache or external-memory cost is included.',
                 'Cortex-M55 is in-order; only specific 16-bit Thumb-1 pairs can dual issue '
                 'and MVE operations take two 64-bit beats, so real latency is above the '
                 'naive 1-instruction-per-cycle figure, not below it.',
                 'objdump aliases several MVE opcodes as coprocessor instructions, so the '
                 'mnemonic class split is partial; function attribution comes from ELF '
                 'labels and is complete.',
                 'QEMU host wall seconds in the RESULT.json files are emulation time and are '
                 'never used as chip latency.'])
    (OUT/'DYNAMIC_INSTRUCTION_AUDIT.json').write_text(json.dumps(audit_out, indent=2)+'\n')

    before = json.loads((ROOT/'delivery/BENCHMARK_POLICY.json').read_text())
    old_zh = before['historical_same_policy']['zh']['current_partial_ms']
    gap = o3['instruction_events'] - TARGET_CYCLES
    final = dict(
        status='DYNAMIC_O2_O3_PROFILED_TARGET_NOT_REACHED',
        task='continue opt300c release trace: dynamic O2/O3 instruction profiling and 300 ms accounting closure',
        training_updates=0, weights_changed=False, production_pointer_changed=False,
        artifact='opt300c_20261030/final_accounting',
        measured=dict(
            roi='one ordinary Chinese first-audio request',
            o2_instruction_events=o2['instruction_events'],
            o3_instruction_events=o3['instruction_events'],
            o2_uncalibrated_1ipc_floor_ms=o2['uncalibrated_1ipc_floor_ms'],
            o3_uncalibrated_1ipc_floor_ms=o3['uncalibrated_1ipc_floor_ms'],
            o3_event_reduction_fraction=-o3_delta/o2['instruction_events']),
        target_300ms=dict(clock_hz=CLOCK, cycle_budget=TARGET_CYCLES,
            o3_events_above_budget=int(gap),
            o3_fraction_above_budget=gap/o3['instruction_events'],
            reachable_by_o3_flags=False,
            reason=('Even at the most optimistic single-issue 1 instruction per cycle, O3 '
                    'needs %.3f ms; 300 ms needs at most %d executed instructions. MVE beats '
                    'and non-dual-issuable work make real latency higher, so 300 ms is not '
                    'met and compiler-level O2/O3 changes alone cannot meet it.'
                    % (o3['uncalibrated_1ipc_floor_ms'], TARGET_CYCLES))),
        old_partial_estimate_reconciliation=dict(
            source='opt300c_20261030/delivery/BENCHMARK_POLICY.json historical_same_policy.zh.current_partial_ms',
            old_zh_current_partial_ms=old_zh,
            measured_o3_1ipc_floor_ms=o3['uncalibrated_1ipc_floor_ms'],
            conclusion=('The old 318.58 ms figure is a *partial priced subtotal* of selected '
                        'operation groups, not a whole-request lower bound. The measured '
                        'executed-instruction floor for the same ordinary Chinese request is '
                        '344.08 ms (O3) / 363.41 ms (O2), so the old optimistic subtotal cannot '
                        'be quoted as achievable first-audio latency and 300 ms stays unaccepted.')),
        remaining_instruction_budget_targets=dict(
            needed_cut_events=int(gap),
            needed_cut_fraction=gap/o3['instruction_events'],
            largest_o3_groups=[dict(function=r['function'], events=r['events'], share=r['share'])
                               for r in o3['functions'][:10]],
            matrix_kernels=dict(events=pair, share=pair/o3['instruction_events']),
            note=('These are measurement targets only; no code was changed in this task and '
                  'no reduction is claimed.')),
        o3_candidate=dict(
            status='BYTE_IDENTICAL_CANDIDATE_NOT_ADOPTED',
            elf=str(RT/'build/ordinary_o3_v1/firmware.elf'),
            elf_sha256=sha(RT/'build/ordinary_o3_v1/firmware.elf'),
            six_files_identical=True,
            instruction_event_reduction_fraction=-o3_delta/o2['instruction_events'],
            text_bytes_o2=174032, text_bytes_o3=212016, text_delta_bytes=212016-174032,
            adopted=False,
            reason=('Byte-identical output and 5.32%% fewer executed instructions make O3 a '
                    'valid future candidate, but 300 ms is still not reached, so the default '
                    'production pointer and delivery numbers stay unchanged.')),
        not_claimed=['300 ms calibrated or hardware-verified latency',
                     'QEMU host seconds as chip latency',
                     'guaranteed cycle lower/upper bounds',
                     'complete native 25-language text frontend',
                     'complete system ROM/RAM dual 10 MB acceptance',
                     'any audio/quality change (all six files byte-identical)'],
        evidence=dict(audit=dict(path=str(OUT/'DYNAMIC_INSTRUCTION_AUDIT.json'),
                                 sha256=sha(OUT/'DYNAMIC_INSTRUCTION_AUDIT.json')),
            o2_build=str(RT/'build/ordinary_o2_v1'), o3_build=str(RT/'build/ordinary_o3_v1'),
            o2_run=str(RT/'reports/ordinary_o2_v1'), o3_run=str(RT/'reports/ordinary_o3_v1'),
            profile_csvs=[str(IP/('release_'+t+'_instruction_counts.csv'))
                          for t in ['ordinary_o2_v1','ordinary_o3_v1']]))
    (OUT/'FINAL_RESULTS.json').write_text(json.dumps(final, indent=2)+'\n')
    (OUT/'STATUS.json').write_text(json.dumps(dict(
        status=final['status'], complete=True, target_300ms_passed=False,
        o3_event_reduction_fraction=final['measured']['o3_event_reduction_fraction'],
        o3_floor_ms=final['measured']['o3_uncalibrated_1ipc_floor_ms'],
        training_updates=0, delivery_copied_to_D=False), indent=2)+'\n')
    print(json.dumps(dict(status=final['status'],
        o2_events=o2['instruction_events'], o3_events=o3['instruction_events'],
        o3_floor_ms=round(o3['uncalibrated_1ipc_floor_ms'],3),
        o2_floor_ms=round(o2['uncalibrated_1ipc_floor_ms'],3),
        gap_events=int(gap), gap_fraction=round(gap/o3['instruction_events'],4),
        matrix_kernel_share=round(pair/o3['instruction_events'],4)), indent=2))

if __name__ == '__main__':
    main()
