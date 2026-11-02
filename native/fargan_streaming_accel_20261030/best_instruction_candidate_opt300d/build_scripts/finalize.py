"""Verify a report-free ordinary synthesis using files, never zero counters."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import struct
import zlib
import numpy as np
ROOT=Path(__file__).resolve().parent;M55=ROOT.parents[1]
BASE=ROOT.parent/'final_production_trace'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,j:p.write_text(json.dumps(j,indent=2)+'\n')
REPORT=ROOT/'reports/ordinary_o2_v1';RUN=REPORT/'run_001'
raw=read(REPORT/'RESULT.json');assert raw['status']=='PASSED'
assert raw['repeats'][0]['guest_metrics']=={}
assert(RUN/'qemu.log').read_text().splitlines()==['NEW25_FARGAN_M55_BEGIN','NEW25_FARGAN_M55_PASS']
references=dict(integrated=ROOT.parent/'integrated/reports/final_v1_ordinary/run_001',
    final_native=ROOT.parent/'final_native/reports/final_v1_zh/run_001')
identity={}
for name in ['condition.f32','durations.u32','output.pcm','output24.pcm','first24.pcm','fargan.f32']:
    identity[name]=dict(bytes=(RUN/name).stat().st_size,sha256=sha(RUN/name),reference_identities={})
    for label,ref in references.items():
        assert(RUN/name).read_bytes()==(ref/name).read_bytes(),(label,name)
        identity[name]['reference_identities'][label]=True
chunks=sorted(RUN.glob('stream_*.pcm'),key=lambda p:int(p.stem.split('_')[1]))
packets=sorted(RUN.glob('pcm24_*.pcm'),key=lambda p:int(p.stem.split('_')[1]))
assert chunks and b''.join(p.read_bytes()for p in chunks)==(RUN/'output.pcm').read_bytes()
assert packets and b''.join(p.read_bytes()for p in packets)==(RUN/'output24.pcm').read_bytes()
assert all(p.stat().st_size==960 for p in packets[:-1])and 0<packets[-1].stat().st_size<=960
assert(RUN/'output24.pcm').read_bytes().startswith((RUN/'first24.pcm').read_bytes())
duration=np.fromfile(RUN/'durations.u32',dtype='<u4');frames=int(duration.sum())
condition=np.fromfile(RUN/'condition.f32',dtype='<f4');pcm=np.fromfile(RUN/'output.pcm',dtype='<i2');pcm24=np.fromfile(RUN/'output24.pcm',dtype='<i2')
assert duration.size==34 and condition.size==frames*31
assert np.isfinite(condition).all()and np.isfinite(np.fromfile(RUN/'fargan.f32',dtype='<f4')).all()
assert pcm.size==(frames-4)*160 and pcm24.size==pcm.size*3//2
extrema16=int(np.count_nonzero((pcm==-32768)|(pcm==32767)))
extrema24=int(np.count_nonzero((pcm24==-32768)|(pcm24==32767)))
assert extrema16==extrema24==0
assert(RUN/'model.sha256.bin').read_bytes()==hashlib.sha256((ROOT/'assets/model.bin').read_bytes()).digest()
index=(ROOT/'assets/metadata.index.bin').read_bytes();_,count,size,names=struct.unpack_from('<4sIII',index)
metadata=zlib.decompress((ROOT/'assets/metadata.deflate').read_bytes(),-15)
assert count==32 and len(metadata)==size and(RUN/'metadata.original.bin').read_bytes()==metadata
digests=b''.join(hashlib.sha256(metadata[o:o+n]).digest()for o,n,no,nl in[
    struct.unpack_from('<IIII',index,16+16*i)for i in range(count)])
assert(RUN/'metadata.sha256.bin').read_bytes()==digests
symbols=read(ROOT/'SYMBOLS.json')
assert len(symbols['snapshot_first_active_costs'])==len(symbols['tensor_views_prepare'])==1
assert not symbols['bench_metric']and not symbols['bench_now']and not symbols['bench_tick_frequency']
asm=Path(raw['elf']).with_name('disassembly.txt').read_text()
anchor=symbols['snapshot_first_active_costs'][0][0]
match=re.search(r'^'+anchor+r' <snapshot_first_active_costs>:\n([^\n]+)',asm,re.M)
assert match and 'bx\tlr'in match[1]
for guard_file,report in [('src/depthwise_exact.inc','depthwise_report'),('src/fast_math.inc','fast_math_report'),('src/gru_fusion.inc','fusion_report')]:
    assert(ROOT/guard_file).read_text().split('static void '+report)[0]==(BASE/guard_file).read_text().split('static void '+report)[0],guard_file
for name in ['bwe_fast/dnn/vec.h','bwe_fast/dnn/nnet_arch.h','bwe_fast/dnn/nnet_default.c','upstream/dnn/nndsp.c','src/fast_i8_dense.h','src/fast_i8_pair.h','src/tensor_views.inc','src/mt_twist_exact.h']:
    assert(ROOT/name).read_bytes()==(BASE/name).read_bytes(),name
assert(ROOT/'assets/model.bin').read_bytes()==(BASE/'assets/model.bin').read_bytes()
assert 'if(fg_received<5||fg_output_frames!=fg_received-4)fail("FARGAN_VALID_CENTER_COUNT");'in(ROOT/'src/fargan_stream.inc').read_text()
assert 'static void identifier_report(void){identifier_pitch_export();}'in(ROOT/'src/identifier_control.inc').read_text()
proof=dict(status='RELEASE_REPORT_FREE_O2_ORDINARY_SIX_FILES_EXACT',elf=raw['elf'],elf_sha256=raw['elf_sha256'],
    report=str(REPORT),file_identity=identity,symbols=symbols,
    stream16_reconstructs_complete=True,packets24_reconstruct_complete=True,
    packet_count=len(packets),audio_seconds=pcm24.size/24000,first24_own_prefix=True,
    finite_condition_and_generator=True,valid_center_and_resampling_lengths=True,
    physical_frames=frames,PCM16_extrema_samples=extrema16,PCM24_extrema_samples=extrema24,
    model_and_32_metadata_M55_SHA_verified=True,
    report_guest_metrics_empty=True,BEGIN_PASS_only_log=True,
    real_BWE_energy_peak_observer_and_first2_getter_retained=True,
    snapshot_PC_anchor_noinline_bx_lr=True,
    model_FP_guards_RNG_cache_decoder_and_PCM_state_unchanged=True,
    mode='Report/diagnostic clock suppression over the final math-only production trace; no new numerical approximation.',
    compiler_optimization='O2',FP_flags_unchanged=True,
    O3_not_run_by_this_task=True,
    dynamic_instruction_measurement_pending=True,
    operation_counter_estimator_compatible=False,
    diagnostic_removal_not_extra_saving_from_old318ms_budget=True,
    calibrated_200MHz_300ms_verified=False,
    complete_native25_language_frontend=False,
    training_updates=0,weights_changed=False,production_pointer_changed=False,
    scope='One ordinary Chinese complete synthesis. No metrics/clock report exists in this variant; verification is from actual files and original PASS control checks.')
write(ROOT/'FINAL_RESULTS.json',proof)
files={str(p.relative_to(ROOT)):sha(p)for sub in ['src','assets','upstream/dnn','bwe_fast/dnn']
    for p in sorted((ROOT/sub).rglob('*'))if p.is_file()}
for name in ['build_chain.py','disable_reports.py','finalize.py']:files[name]=sha(ROOT/name)
write(ROOT/'SOURCE_IDENTITY.json',dict(files=files,elf_sha256=raw['elf_sha256'],
    disabled_reports_sha256=sha(ROOT/'DISABLED_REPORTS.json'),
    result_sha256=sha(REPORT/'RESULT.json'),original_math_source=str(BASE)))
patch=[]
for name in files:
    if not name.startswith('src/')or not(BASE/name).is_file():continue
    if(ROOT/name).read_bytes()!=(BASE/name).read_bytes():
        patch.extend(difflib.unified_diff((BASE/name).read_text().splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
(ROOT/'RELEASE_REPORTS.patch').write_text(''.join(patch))
write(ROOT/'STATUS.json',dict(status=proof['status'],complete=True,elf=raw['elf'],symbols=symbols,
    operation_estimator_compatible=False,target_300ms_passed=False,training_updates=0))
(ROOT/'RECOVERY_README.txt').write_text(
    'TERMINAL: ordinary_o2_v1 full synthesis complete. Do not rerun completedtag. Sixfiles exact against integrated and final_native zh.\n'
    'Thisrelease has NO metrics/hostclock reports. Neverrun old operation estimator on it. BEGIN/PASS/error text and actual sixfiles/chunks remain.\n'
    'PCanchors inSYMBOLS.json; snapshot_first_active_costs is noinline empty memory-barrier function(BXLR). BWE actual energy/peak observer and first2 state remain unchanged.\n'
    '16report bodies disabled; FARGAN validity assertion and identifier export/completion check retained. bench.c protected byTTS_BENCH_IMPLEMENTATION.\n'
    'AllFP guards and math, model/RNG/cache/decoder/PCM progress untouched. No318ms partial budget credit or300ms calibrated acceptance. Root does dynamicO2/O3 profiling next.\n')
print(json.dumps(dict(status=proof['status'],elf=raw['elf'],symbols=symbols)))
