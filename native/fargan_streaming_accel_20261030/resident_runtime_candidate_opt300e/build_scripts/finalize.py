"""Finalize same-ELF native Han integration; no cycle or full25-lang signoff."""
from pathlib import Path
import hashlib,json,shutil,struct,zlib
ROOT=Path(__file__).resolve().parent;M55=ROOT.parents[1];WORK=M55.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
TAG='resident_final_v1';REP=ROOT/'reports'/TAG;RUN=REP/'run_001'
OLD=M55/'opt300b_20261030/resident/reports/resident_final_v1/run_001'
SOURCE=M55/'opt300c_20261030/resident_schedule'
raw=json.loads((REP/'RESULT.json').read_text());assert raw['status']=='PASSED'
runtime=json.loads((REP/'RUNTIME_VERIFIED.json').read_text())
m={k:v[0]for k,v in raw['repeats'][0]['guest_metrics'].items()if v}
full=raw['repeats'][0]['guest_metrics']
assert full['runtime_tensor_views_init_calls']==[1]
assert full['tensor_float_view_ready']==[1]
assert full['tensor_float_view_decoded_persistent_bytes']==[395392]
assert full['runtime_model_init_calls_final']==[1]
assert all(v==0 for v in full['BWE_quantizer_clips'])
assert(RUN/'first24.pcm').read_bytes()==(OLD/'first24.pcm').read_bytes()
first=(RUN/'first24.pcm').read_bytes()
for i in [0,1,3]:assert(RUN/f'request_{i}_output24.pcm').read_bytes().startswith(first)
assert(RUN/'model.sha256.bin').read_bytes()==hashlib.sha256((ROOT/'assets/model.bin').read_bytes()).digest()
mi=(ROOT/'assets/metadata.index.bin').read_bytes();_,count,n,namesize=struct.unpack_from('<4sIII',mi)
meta=zlib.decompress((ROOT/'assets/metadata.deflate').read_bytes(),-15)
digests=b''.join(hashlib.sha256(meta[o:o+size]).digest()for o,size,_,_ in[struct.unpack_from('<IIII',mi,16+i*16)for i in range(count)])
assert count==32 and len(meta)==n and(RUN/'metadata.original.bin').read_bytes()==meta and(RUN/'metadata.sha256.bin').read_bytes()==digests
profiles={k:v for k,v in full.items()if k.startswith('runtime_total_')or k.startswith('BWE_final_')}
assert profiles and all(len(v)==3 and len(set(v))==1 for v in profiles.values())
assert any(k.startswith('runtime_total_depthwise_')for k in profiles)
assert m['runtime_total_i8_pair_matrix_pairs']>0
old_mark=json.loads((OLD.parent/'RUNTIME_VERIFIED.json').read_text())['resident_weight_arena_bytes']
front_identity={}
for p in sorted((ROOT/'frontend_source').rglob('*')):
 if p.is_file():
  key=str(p.relative_to(ROOT));q=SOURCE/key;assert p.read_bytes()==q.read_bytes();front_identity[key]=dict(sha256=sha(p),matches_source=True)
for name in ['src/text_frontend.c','src/text_frontend.h','src/frontend_inflate.c','src/request_text.h','assets/fixed_input.h']:
 p=ROOT/name;assert p.read_bytes()==(SOURCE/name).read_bytes();front_identity[name]=dict(sha256=sha(p),matches_source=True)
rate=m['tick_frequency_hz']
result=dict(status='RESIDENT_ALL_EXACT_KERNELS_INTEGRATED_REPLAY_CANCEL_VERIFIED_TARGET_PENDING',
 report=str(REP),elf=raw['elf'],elf_sha256=raw['elf_sha256'],runtime=runtime,
 model_layout=2,model_sha256=sha(ROOT/'assets/model.bin'),
 model_weight_roundtrip_proof=str(M55/'opt300c_20261030/pair_relayout/PACK_IDENTITY.json'),
 M55_model_and_32_metadata_SHA_verified=True,
 model_and_view_init_calls=1,view_decoded_bytes=395392,
 view_actual_persistent_arena_increase_bytes=m['runtime_expected_model_mark']-old_mark,
 view_alignment_bytes=m['runtime_expected_model_mark']-old_mark-395392,
 view_init_QEMU_host_seconds=m['runtime_tensor_views_prepare_host_ticks']/rate,
 cold_model_and_views_init_QEMU_host_seconds=m['runtime_model_init_host_ticks']/rate,
 cold_first_effective_QEMU_host_seconds=m['runtime_request_0_boot_to_first_effective_host_ticks']/rate,
 views_initialized_after_model_inflate_before_persistent_arena_mark=True,
 views_and_name_index_preserved_across_requests=True,
 first24_matches_old_resident=True,first24_is_all_three_complete_request_prefix=True,
 first24_sha256=sha(RUN/'first24.pcm'),
 all_new_profile_fields_reset_and_equal_across_complete_requests=True,
 native_frontend_identity=front_identity,
 native_frontend_scope='Existing native Han/punctuation/explicit-pinyin subset; no complete number/Latin/25-language native parser.',
 PCM_quality_reference='All emitted files and final RNG/GRU match old accepted native I8 resident; no new all-text or all-language subjective acceptance.',
 training_updates=0,model_weight_values_changed=False,new_quantization=False,
 production_default_pointer_changed=False,target_300ms_accepted=False,
 complete_native_25_language_frontend=False,complete_ROM_RAM_accepted=False,
 QEMU_seconds_are_not_M55_cycles=True,cycle_estimate_owned_by_parent=True,
 diagnostic_complete_buffers_retained=True)
(ROOT/'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
files={str(p.relative_to(ROOT)):sha(p)for folder in ['src','frontend_source','assets','upstream/dnn','bwe_fast/dnn']for p in sorted((ROOT/folder).rglob('*'))if p.is_file()}
(ROOT/'SOURCE_IDENTITY.json').write_text(json.dumps(dict(files=files,source_resident=str(SOURCE),source_integrated=str(M55/'opt300c_20261030/integrated'),frontend_files_unchanged=True),indent=2)+'\n')
snap=ROOT/'snapshots'/TAG
if not snap.exists():
 shutil.copytree(ROOT/'src',snap/'src');shutil.copytree(ROOT/'assets',snap/'assets');shutil.copy2(ROOT/'build_chain.py',snap/'build_chain.py');shutil.copy2(ROOT/'SOURCE_IDENTITY.json',snap/'SOURCE_IDENTITY.json')
(ROOT/'STATUS.json').write_text(json.dumps(dict(status=result['status'],complete=True,target_300ms_accepted=False,complete_native25_frontend=False,default_pointer_changed=False),indent=2)+'\n')
(ROOT/'RECOVERY_README.txt').write_text('TERMINAL: resident_final_v1 full sameELF init1/A0/A1/cancel70/A3 completed. Do not rerun completed tag.\nAll complete request PCM/condition/duration/FARGAN/features plus RNG/GRU exactly match old accepted resident; globalfirst24 exact and a prefix of allthree complete outputs.\nLayout2 model, immutable float views initialized once before persistent arena mark, exactDW,2/16 scheduler. Per-requestDW/pair profiles reset, float/name caches persist.\nDecoded views395392B; persistent arena increase395440B includes48B alignment. Cold model/view initialization is separately reported.\nNative Han/punctuation/explicitpinyin parser retained unchanged. No complete25nativeparser/300ms/fullsystemdual10MB acceptance, no training or default switch.\nRoot owns combined cycle accounting, delivery and pointer. No remaining process.\n')
print(json.dumps(dict(status=result['status'],ROM=runtime['backend_ROM_bytes'],RAM=runtime['memory']['reserved_resident_ram_bytes'],arena_peak=runtime['memory']['arena_peak_bytes'],stack=runtime['memory']['stack_written_high_water_bytes'],mark=runtime['resident_weight_arena_bytes'],views_extra=result['view_actual_persistent_arena_increase_bytes'],view_init_host_s=result['view_init_QEMU_host_seconds'],cold_first_host_s=result['cold_first_effective_QEMU_host_seconds'])))
