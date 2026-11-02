"""Read-only guest instruction-event audit. No event-to-cycle conversion."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent;M55=ROOT.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,j:p.write_text(json.dumps(j,indent=2)+'\n')
PATTERN=re.compile(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4}(?: [0-9a-f]{4})?)\s+(.+)$')

def disassembly(build):
    functions={};instructions={};owner=None
    for line in (build/'disassembly.txt').read_text().splitlines():
        label=re.match(r'^([0-9a-f]+) <([^>]+)>:',line)
        if label:
            owner=label[2];functions[owner]=int(label[1],16)
        ins=PATTERN.match(line)
        if ins:instructions[int(ins[1],16)]=dict(function=owner,raw=ins[2],text=ins[3])
    return functions,instructions

def audit(csv_name,report_name,build,begin,end,begin_symbol):
    raw=ROOT/csv_name;before=sha(raw);result=json.loads((ROOT/'reports'/report_name/'RESULT.json').read_text())
    assert result['status']=='PASSED'
    assert result['elf_sha256']==sha(build/'firmware.elf')
    functions,ins=disassembly(build);assert functions[begin_symbol]==begin
    assert functions['snapshot_first_active_costs']==end
    m={k:v[0]for k,v in result['repeats'][0]['guest_metrics'].items()if v}
    rows=list(csv.DictReader(raw.open()));request=collections.Counter();cold=collections.Counter()
    bypc={int(x['pc'],16):x for x in rows};unmatched=[];size_errors=[];dots=collections.Counter();semantics=collections.Counter()
    for row in rows:
        pc=int(row['pc'],16);n=int(row['request_instruction_events']);c=int(row['cold_instruction_events'])
        if pc not in ins:
            if n or c:unmatched.append(row)
            continue
        item=ins[pc];f=item['function'];opcode=item['raw'];text=item['text'];mnemonic=text.split()[0]
        request[f]+=n;cold[f]+=c
        actual_size=len(opcode.replace(' ',''))//2
        if int(row['size'])!=actual_size:
            size_errors.append(dict(pc=row['pc'],plugin_size=int(row['size']),ELF_instruction_bytes=actual_size,
                request_events=n,raw_opcode=opcode,function=f))
        is_dot=len(opcode)==9 and opcode[:4]=='eef0' and int(opcode[5:],16)&0xfff==0xf22
        if is_dot:
            dots[f]+=n;semantics['verified_MVE_signed_I8_dot']+=n
        elif mnemonic in ['cdp','cdp2','ldc','ldc2','stc','stc2']:
            semantics['old_objdump_coprocessor_alias_requires_MVE_decode']+=n
        elif '.f32' in mnemonic or '.f64' in mnemonic:
            semantics['decoded_FP_vector'if re.search(r'\bq\d+',text)else'decoded_FP_scalar']+=n
        elif mnemonic.startswith(('vld','vst','vpush','vpop')):
            semantics['decoded_FP_or_vector_data_transfer']+=n
        elif mnemonic.startswith(('ldr','str','ldm','stm','push','pop')):
            semantics['scalar_load_store_stack']+=n
        elif mnemonic.startswith(('b','cb'))or mnemonic in ['dls','wls','le','letp','it','itt','ite','nop']:
            semantics['branch_loop_or_control']+=n
        else:semantics['other_scalar_or_vector_operations']+=n
    assert not unmatched
    # The local QEMU metadata reports two extra bytes at these TB-boundary
    # PCs. PC hits still map to one real ELF instruction. Never use plugin
    # size to infer instruction width or beat multiplicity.
    assert all(x['plugin_size']==x['ELF_instruction_bytes']+2 and
               (int(x['pc'],16)+x['ELF_instruction_bytes'])%1024==1022 for x in size_errors)
    assert sum(dots.values())==m['chain_first_all_i8_dot_instructions']==11396864
    # Source-confirmed scalar .S loads at these exact opcodes. Verify per-PC
    # callbacks count architectural instructions once, rather than MVE beats.
    loads=collections.Counter()
    for pc,item in ins.items():
        if item['function']=='packed_matrix8' and item['raw']in ['ecb1 1e10','ecb3 3e10']:
            loads[item['raw']]+=int(bypc.get(pc,{}).get('request_instruction_events',0))
    assert loads['ecb3 3e10']==dots['packed_matrix8']
    assert loads['ecb1 1e10']*8==dots['packed_matrix8']
    # dot source cost_* increments occupy44 static instructions per call.
    # Offsets are checked against the old unchanged function's instruction
    # layout. Source counters are not inferred from generic mnemonic classes.
    dot_start=functions['dot'];hit=lambda off:int(bypc.get(dot_start+off,{}).get('request_instruction_events',0))
    profile_offsets=[pc-dot_start for pc in ins if dot_start+6<=pc<=dot_start+0x88
                     and pc-dot_start not in [0x0a,0x10,0x7a,0x86]]
    assert len(profile_offsets)==44
    assert ins[dot_start+0xb2]['text'].startswith('vfma.f32')
    assert ins[dot_start+0x108]['text'].startswith('vmul.f32')
    assert ins[dot_start+0x10c]['text'].startswith('vadd.f32')
    categories={
        'cost_counter_updates':set(profile_offsets),
        'FP_vector_FMA':{0xb2},
        'FP_horizontal_additions':{0xd0,0xd4,0xda},
        'FP_scalar_tail_arithmetic':{0x108,0x10c},
        'call_frame_save_restore':{2,0x7a,0x114,0x116},
        'vector_data_and_reduction_stack':{0,0x10,0xa8,0xac,0xbc,0xc0,0xc4,0xc8,0xcc},
        'scalar_tail_operand_loads':{0x100,0x104},
    }
    used=set();dot_breakdown={}
    for name,offsets in categories.items():
        assert not used&offsets;used|=offsets;dot_breakdown[name]=sum(hit(off)for off in offsets)
    dot_breakdown['pointer_loop_branch_other']=request['dot']-sum(dot_breakdown.values())
    assert dot_breakdown['cost_counter_updates']==44*hit(0)
    dot_summary=dict(total_events=request['dot'],entry_calls=hit(0),
        calls_entering_vector_loop=hit(0x8c),calls_skipping_vector_loop=hit(0)-hit(0x8c),
        breakdown=dot_breakdown,pure_FP_instruction_events=sum(dot_breakdown[k]for k in ['FP_vector_FMA','FP_horizontal_additions','FP_scalar_tail_arithmetic']),
        scalar_tail_multiply_events=hit(0x108),scalar_tail_add_events=hit(0x10c),
        vector_FMA_events=hit(0xb2),
        caution='Chain cost_* metrics also include gemm/decoder work; do not equate their totals with the dot function alone. Scalar-only branch means n<4; call argument distribution is not directly observed.')
    total=sum(request.values());cold_total=sum(cold.values())
    assert total==sum(int(x['request_instruction_events'])for x in rows)
    assert cold_total==sum(int(x['cold_instruction_events'])for x in rows)
    assert sha(raw)==before
    return dict(csv=str(raw),csv_sha256=before,result=str(ROOT/'reports'/report_name/'RESULT.json'),
        elf_sha256=result['elf_sha256'],begin_pc=hex(begin),begin_symbol=begin_symbol,
        end_pc=hex(end),end_symbol='snapshot_first_active_costs',
        begin_instruction_events=int(bypc[begin]['request_instruction_events']),
        end_instruction_events=int(bypc.get(end,{}).get('request_instruction_events',0)),
        request_instruction_events=total,cold_instruction_events=cold_total,
        functions=[dict(function=k,events=v,share=v/total,cold_events=cold[k])for k,v in request.most_common()],
        cold_functions=[dict(function=k,events=v,share=v/cold_total)for k,v in cold.most_common()],
        dot_opcode_events=dict(dots),expected_chain_dot_instructions=m['chain_first_all_i8_dot_instructions'],
        dot_events_ratio_to_chain_count=sum(dots.values())/m['chain_first_all_i8_dot_instructions'],
        packed_matrix_load_events=dict(loads),
        instruction_size_mismatches=size_errors,unmatched_event_PCs=unmatched,
        semantic_classes_partial=dict(semantics),dot_function_audit=dot_summary,
        no_MVE_four_beats_multiplier_in_verified_dot_and_load_PCs=True,
        complete_MVE_semantic_disassembly=False)

old=audit('ordinary_instruction_counts.csv','ordinary_v1',M55/'opt300b_20261030/final/build/ordinary_v1',
          0x100067e0,0x10007008,'acoustic_begin')
new=audit('tensor_views_instruction_counts.csv','tensor_views_v1',M55/'opt300c_20261030/tensor_views/build/ordinary_v1',
          0x100073a4,0x100071d4,'tensor_views_prepare')
oldf={x['function']:x['events']for x in old['functions']};newf={x['function']:x['events']for x in new['functions']}
comparison=dict(request_events_before=old['request_instruction_events'],request_events_after=new['request_instruction_events'],
    delta_events=new['request_instruction_events']-old['request_instruction_events'],
    reduction_fraction=1-new['request_instruction_events']/old['request_instruction_events'],
    function_deltas=[dict(function=k,before=oldf.get(k,0),after=newf.get(k,0),delta=newf.get(k,0)-oldf.get(k,0))
        for k in ['n25_value','strcmp','dot','packed_matrix8','packed_dot8','tensor_views_prepare']],
    exact_equal_dot_opcode_events=old['dot_opcode_events']==new['dot_opcode_events'],
    scope_difference='New begin includes tensor_views_prepare initialization (395392B cache) before acoustic_begin. Old begins at acoustic_begin. Different pre-acoustic clock/metric setup is also included in the new ROI; the new ROI conservatively includes added cache preparation rather than hiding it.',
    instruction_count_is_not_speedup_or_cycles=True)
result=dict(status='PC_AGGREGATION_AND_MVE_EVENT_GRANULARITY_VERIFIED',baseline=old,tensor_views=new,comparison=comparison,
    plugin_source_sha256=sha(ROOT/'profile.c'),plugin_binary_sha256=sha(ROOT/'profile.so'),
    scope_semantics='Callbacks snapshot all per-PC hits at begin/end. Observed begin PC is included once; end function entry is excluded. The call instruction leading to end and previously emitted metric/packet diagnostics remain counted.',
    cold_scope='Cold events cover startup/model+metadata decode/hash/export and all code preceding begin. They are not exclusively decompression and are separate from warm request instruction events.',
    caveats=[
        'Single-vCPU plugin adds one host-side inline U64 per guest instruction callback. No plugin machine instructions are in guest PC totals, but plugin overhead changes host wall time.',
        'The firmware guest counters/diagnostics are real guest instructions and remain in event totals. dot counter updates are explicitly separated; instrumentation elsewhere has not been completely removed.',
        'An architectural MVE instruction executes multiple lanes/beats internally; measured dot/load PCs match the source one-for-one and are not fourfold inflated.',
        'Old objdump aliases many MVE instructions as coprocessor operations; complete mnemonic-based FP/vector classification would be unsound. Raw verified dot opcodes are decoded separately.',
        'Local plugin instruction-size metadata is two bytes too large at a small set of1024-byte block-tail PCs. Every event PC still maps to a real ELF instruction. The audit uses ELF opcode widths and retains all size anomalies without claiming their internal cause.',
        'n25_value and strcmp are actual function hotspots. Their removal reduces guest instruction events; guest events alone do not imply equivalent hardware cycles or 300ms acceptance.',
        'No DWT cycle model, pipeline/memory calibration, real external-memory wait, or audio-driver timing is provided by this count.'
    ],events_are_cycles=False,target_300ms_accepted=False,raw_csv_modified=False)
write(ROOT/'INSTRUCTION_AUDIT.json',result)
print(json.dumps(dict(status=result['status'],dot_ratio=old['dot_events_ratio_to_chain_count'],dot=old['dot_function_audit'],comparison=comparison),indent=2))
