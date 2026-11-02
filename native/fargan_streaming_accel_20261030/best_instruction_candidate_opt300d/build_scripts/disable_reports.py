"""Prepare the release trace without changing model computation or controls."""
from pathlib import Path
import hashlib
import json
import re
ROOT=Path(__file__).resolve().parent
mask_re=re.compile(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
def mask(s):
    return mask_re.sub(lambda m:''.join('\n'if x=='\n'else' 'for x in m.group()),s)
disabled=[]
functions={
    'src/quality.c':{'report_first_costs':'','snapshot_first_active_costs':None,'report_front_i8':''},
    'src/fargan_stream.inc':{'report_fg_ram_cache':'','fg_noise_report':'',
        'fargan_stream_report':'if(fg_received<5||fg_output_frames!=fg_received-4)fail("FARGAN_VALID_CENTER_COUNT");'},
    'src/adapter_stream.inc':{'adapter_stream_report':''},
    'src/fast_math.inc':{'fast_math_report':''},
    'src/gru_fusion.inc':{'fusion_report':''},
    'src/depthwise_exact.inc':{'depthwise_report':''},
    'src/fir_fast.inc':{'fast_fir_report':''},
    'src/bwe_stage.c':{'bwe_activation_report':'','bwe_block16_report':''},
    'src/identifier_control.inc':{'identifier_snapshot_first':'','identifier_report':'identifier_pitch_export();'},
    'src/bench.c':{'bench_memory_report':''},
}
for name,targets in functions.items():
    path=ROOT/name;source=path.read_text()
    for function,replacement in targets.items():
        clean=mask(source)
        matches=list(re.finditer(r'\b'+function+r'\s*\([^;{}]*\)\s*\{',clean))
        assert len(matches)==1,(name,function,len(matches))
        m=matches[0];begin=m.end()-1;depth=1;end=begin+1
        while depth:
            if clean[end]=='{':depth+=1
            elif clean[end]=='}':depth-=1
            end+=1
        old=source[begin:end]
        if replacement is None:
            # Preserve the actual observer's PC boundary even after its
            # report work disappears, including under later O3 compilation.
            source=source[:begin]+'{__asm__ volatile("" ::: "memory");}'+source[end:]
            declaration='void snapshot_first_active_costs(void)'
            assert declaration in source
            source=source.replace(declaration,'__attribute__((noinline,used)) '+declaration,1)
        else:
            source=source[:begin]+'{'+replacement+'}'+source[end:]
        disabled.append(dict(file=name,function=function,
            old_body_sha256=hashlib.sha256(old.encode()).hexdigest(),
            replacement='noinline memory-barrier PC anchor'if replacement is None else replacement or'empty report body',
            retained_functional_checks=function in ['fargan_stream_report','identifier_report']))
    path.write_text(source)
header=ROOT/'src/bench.h';source=header.read_text()
assert source.rstrip().endswith('#endif')
source=source.rstrip()[:-len('#endif')]+'''/* Release-trace clients do not evaluate diagnostic metric arguments or
 * host-clock reads. These clocks were used only for debug durations/latches;
 * mathematical state, RNG, PCM counters and energy observers remain intact.
 * bench.c keeps its real semihost implementation behind the implementation
 * guard, avoiding any macro/function-definition collision. */
#ifndef TTS_BENCH_IMPLEMENTATION
#define bench_metric(name,value) ((void)0)
#define bench_now() UINT64_C(0)
#define bench_tick_frequency() UINT32_C(0)
#endif
#endif
'''
header.write_text(source)
path=ROOT/'src/bench.c';source=path.read_text();assert source.startswith('#include "bench.h"')
path.write_text('#define TTS_BENCH_IMPLEMENTATION 1\n'+source)
clock_calls=[]
base=Path(json.loads((ROOT/'BASE_SOURCE_IDENTITY.json').read_text())['source'])
for path in sorted((base/'src').rglob('*')):
    if path.suffix not in ['.c','.h','.inc']or path.name in ['bench.c','bench.h']:continue
    text=path.read_text();clean=mask(text)
    for match in re.finditer(r'\bbench_now\s*\(\s*\)',clean):
        line=clean[:match.start()].count('\n')+1
        clock_calls.append(dict(file=str(path.relative_to(base)),line=line,
            source_line=text.splitlines()[line-1],replacement='UINT64_C(0)',
            reason='Only diagnostic elapsed-time accumulation, first-time logging latch or report argument; never RNG/model/frame/cache/PCM control.'))
manifest=dict(status='RELEASE_DIAGNOSTIC_REPORTS_AND_HOST_CLOCK_CALLS_DISABLED',
    disabled_report_bodies=disabled,disabled_client_APIs=['bench_metric','bench_now','bench_tick_frequency'],
    header_implementation_guard='TTS_BENCH_IMPLEMENTATION defined in bench.c only',
    diagnostic_clock_call_audit=clock_calls,
    preserved=['actual BWE active_found power/peak detection and first2 getter',
        'snapshot_first_active_costs noinline memory-barrier PC anchor',
        'all floating-point computation and special-value guards',
        'RNG, decoder, cache, arena, PCM progress and true length invariants',
        'FARGAN_VALID_CENTER_COUNT fail check and identifier export/group completion checks',
        'original six files, streamed PCM chunks, BEGIN/PASS and error paths'],
    old_operation_estimator_compatible=False,
    diagnostic_removal_is_not_additional_saving_from_old318ms_operation_budget=True,
    model_and_FP_flags_changed=False)
(ROOT/'DISABLED_REPORTS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(report_bodies=len(disabled),diagnostic_clock_calls=len(clock_calls))))
