"""Integrate actual G2P/resource modules and Chinese TTS in one bounded image.

This proves component coexistence and memory use; full NEW25 feature adapters
for other languages are still required and are not claimed by this probe.
"""
from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import time
from build_resources import ROOT,WORK,NATIVE,TOOLS,CC,XZ,sha,write,run,elf_memory,build_lzma,generate
from build_espeak import generate_fs,SOURCE,AUDIT,PATCHED,UNICODE,WRAPS


def main():
    global NATIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='combined_v1')
    parser.add_argument('--backend-tag',default='compact_luts_v1')
    parser.add_argument('--arena-bytes',type=int,default=5848000)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--multilang',action='store_true')
    parser.add_argument('--native-root',type=Path)
    parser.add_argument('--prewarm-g2p',action='store_true')
    parser.add_argument('--frontend-regression',action='store_true',help='Run test-only feature hashes with resident model and production heap limits; do not use this build for cold-request timing.')
    parser.add_argument('--case-manifest',type=Path,default=ROOT/'MULTILANG_SOURCE_IDENTITY.json')
    a=parser.parse_args();out=ROOT/'build'/a.tag;out.mkdir(parents=True,exist_ok=False)
    original_native=NATIVE
    if a.multilang:NATIVE=ROOT/'native_multilang'
    if a.native_root:NATIVE=a.native_root.resolve()
    generated=out/'generated';generated.mkdir()
    generate_fs()
    for name in('romfs_files.c','romfs_direct.S','espeak_cases.h','config.h','romfs_capacity.h'):
        shutil.copy2(ROOT/'generated'/name,generated/name)
    # The current native Mandarin reference uses its original zh_words/chars
    # dictionary. These older FEP Chinese dictionaries are a second frontend
    # implementation, not dependencies of that native reference.
    excluded=['zh-characters.fep','zh-shared.fep','zh-supplement-phrases.fep']
    if a.multilang:excluded.append('frontend-supplement.fep')
    extra=['espeak.fep','integration-supplement.fep','bundle.fep']
    resource_profile=NATIVE/'resource_profile.json';overrides={}
    if resource_profile.is_file():
        profile=json.loads(resource_profile.read_text());excluded+=profile.get('exclude',[])
        overrides={name:NATIVE/path for name,path in profile.get('overrides',{}).items()}
    generate(exclude=excluded,destination=generated,extra_rom=extra,overrides=overrides)
    compat=generated/'compat';compat.mkdir()
    shutil.copy2(AUDIT/'compat/dirent.h',compat/'dirent.h')
    commands=[];objects=[]
    baseline=json.loads((original_native/'build'/a.backend_tag/'BUILD.json').read_text())
    for i,row in enumerate(baseline['compile_steps']):
        command=[s.replace(str(original_native),str(NATIVE))for s in row['command']]
        source=Path(command[command.index('-c')+1])
        obj=out/('back_'+str(i)+'_'+source.stem+'.o')
        command[command.index('-o')+1]=str(obj)
        if source.name=='quality.c':
            command+=['-Dmain=n25_resident_backend_main']
            if a.frontend_regression:command+=['-DN25_FRONTEND_REGRESSION=1']
        commands.append(run(command,out/('back_'+str(i)+'.log')));objects.append(obj)
    for kernel_name in ('bwe_packed_matrix8.S','bwe_sgemv16.S','bwe_adaconv16.S'):
        extra_kernel=NATIVE/'src'/kernel_name
        if not extra_kernel.is_file():continue
        obj=out/(extra_kernel.stem+'.o')
        command=list(baseline['compile_steps'][0]['command'])
        command=[v.replace(str(original_native),str(NATIVE))for v in command]
        command[command.index('-c')+1]=str(extra_kernel)
        command[command.index('-o')+1]=str(obj)
        commands.append(run(command,out/(extra_kernel.stem+'.log')));objects.append(obj)
    flags=['-Os','-g','-ffunction-sections','-fdata-sections','-fstack-usage','-fno-unwind-tables',
        '-fno-asynchronous-unwind-tables','-fwrapv','-DLIBESPEAK_NG_EXPORT=1',
        '-DPATH_ESPEAK_DATA="/espeak-ng-data"','-D_DEFAULT_SOURCE','-D_POSIX_C_SOURCE=200809L',
        '-DROMFS_SHARED_WORKSPACE=1','-DROMFS_HEAP_BYTES='+('131072'if a.prewarm_g2p else'475136'),
        *['-I'+str(p)for p in(NATIVE/'src',generated,compat,SOURCE/'src/include',SOURCE/'src/include/compat',
            SOURCE/'src/libespeak-ng',SOURCE/'src/ucd-tools/src/include',XZ/'src/liblzma/api',NATIVE/'dictionary_runtime')]]
    inputs=[]
    for row in json.loads((AUDIT/'ARM_OBJECTS.json').read_text()):
        if row['group']not in('espeak','ucd'):continue
        source=Path(row['source'])
        if source.stem in('speech','synthdata','synthesize'):source=PATCHED/source.name
        if source.parent.name=='src'and'ucd-tools'in str(source):
            if source.stem=='case':source=UNICODE/'case_compact.c'
            if source.stem=='proplist':source=UNICODE/'properties_compact.c'
        if (NATIVE/'frontend_patched'/source.name).is_file():source=NATIVE/'frontend_patched'/source.name
        inputs.append(source)
    inputs += [NATIVE/'src'/n for n in('romfs.c','fep_runtime.c','frontend_syscalls.c','fep_probe.c','espeak_probe.c','combined_probe.c')]
    inputs += [generated/n for n in('fep_assets.c','fep_assets.S','romfs_files.c','romfs_direct.S')]
    if a.multilang:
        inputs += [NATIVE/'src/multilang_frontend.c',NATIVE/'src/four_frontend.c',
            generated/'four_tables.c',NATIVE/'assets/language_norms.S']
        shutil.copy2(ROOT/'generated/four_tables.h',generated/'four_tables.h')
        shutil.copy2(ROOT/'generated/four_tables.c',generated/'four_tables.c')
    if (NATIVE/'dictionary_runtime').is_dir():
        inputs += sorted((NATIVE/'dictionary_runtime').glob('*.c'))
    for i,source in enumerate(inputs):
        obj=out/('front_'+str(i)+'_'+source.stem+'.o');defines=[]
        if source.name=='dictionary_probe.c':defines=['-Dmain=n25_dictionary_probe_run']
        if source.name=='fep_probe.c':defines=['-Dmain=n25_fep_probe_run']
        if source.name=='espeak_probe.c':defines=['-Dmain=n25_espeak_probe_run']
        commands.append(run([CC,*flags,*defines,'-c',source,'-o',obj],out/('front_'+str(i)+'.log')));objects.append(obj)
    linker=(NATIVE/'src/m55.ld').read_text()
    linker=linker.replace('  QSPI (r)', '  FRONT_RAM (rwx) : ORIGIN = 0x31000000, LENGTH = 4M\n  QSPI (r)')
    linker=linker.replace('  .text : {', '''  .frontend_code : ALIGN(16) { *front_*.o(.text* .rodata*) } > FRONT_RAM
  .frontend_data : ALIGN(16) { *front_*.o(.data*) } > FRONT_RAM
  .frontend_assetbank : ALIGN(16) { *(.frontend_assetbank*) . = ALIGN(16); } > FRONT_RAM
  .frontend_bss (NOLOAD) : ALIGN(16) {
    __frontend_bss_start = .; *front_*.o(.bss*) *(.frontend_workspace*) . = ALIGN(16); __frontend_bss_end = .;
  } > FRONT_RAM
  .text : {''')
    (out/'m55.ld').write_text(linker)
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,-Map='+str(out/'firmware.map'),'-Wl,--defsym=ARENA_BYTES='+str(a.arena_bytes),'-Wl,--defsym=STACK_BYTES=114688',
        *['-Wl,--wrap='+name for name in WRAPS],'-T',out/'m55.ld',*objects,
        '-Wl,--start-group',build_lzma(),'-lm','-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    memory=elf_memory(elf)
    write(out/'BUILD.json',dict(elf_sha256=sha(elf),commands=commands,memory=memory,
        removed_unused_probe_archive_bytes=134483,excluded_duplicate_frontend_packages=excluded,
        assetbank_in_QEMU_SRAM_counts_as_RAM=True,
        same_image_components=['native Chinese TTS backend','23-language eSpeak G2P','13 active compressed resource packs'],
        full25_native_features_integrated=False,full25_acceptance=False,
        native_text_language_count=len(json.loads(a.case_manifest.read_text())["native_text_cases"]) if a.multilang else 1))
    print(json.dumps(dict(elf=str(elf),memory=memory),indent=2),flush=True)
    report=ROOT/'reports'/a.tag;report.mkdir(parents=True,exist_ok=False)
    command=[str(WORK/'local_qemu_20261030/bin/qemu-system-arm'),'-M','mps3-an547','-cpu','cortex-m55',
        '-accel','tcg,thread=single','-nographic','-serial','none','-monitor','none',
        '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    csv_path=None
    if a.profile:
        symbols={}
        for line in subprocess.check_output([str(TOOLS/'arm-none-eabi-nm'),'-n',str(elf)],text=True).splitlines():
            fields=line.split()
            if len(fields)==3:symbols[fields[2]]=int(fields[0],16)&~1
        (ROOT/'profile').mkdir(exist_ok=True);csv_path=ROOT/'profile'/(a.tag+'.csv')
        plugin=ROOT/'profile/multi_request.so'if a.multilang else WORK/'m55_fargan_stream_20261030/opt300c_20261030/instruction_profile/profile.so'
        begin='tts_begin_language_text_request'if a.multilang else'tts_begin_text_request'
        command+=['-plugin',str(plugin)+',begin='+hex(symbols[begin])+
            ',end='+hex(symbols['tts_first_effective_packet_delivered'])+',out='+str(csv_path)]
    start=time.monotonic();process=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=420)
    output=process.stdout+process.stderr;(report/'qemu.log').write_text(output)
    passed=process.returncode==0 and'N25_COMBINED_COMPONENTS_PASS_NOT_FULL25_TTS'in output
    metrics={}
    for line in output.splitlines():
        if line.startswith('{'):
            row=json.loads(line)
            if row.get('kind')=='metric':metrics.setdefault(row['name'],[]).append(row['value'])
    baseline=WORK/'m55_fargan_stream_20261030/opt300c_20261030/final_resident/reports/resident_final_v1/run_001'
    equal=[]
    verified_languages=[]
    if passed and a.multilang:
        cases=json.loads(a.case_manifest.read_text())['native_text_cases']
        for case in cases:
            reference=WORK/'m55_fargan_stream_20261030/language25_20261030/native/reports'/(case['language']+'_v1')/'run_001'
            for name in('condition.f32','durations.u32','output.pcm','output24.pcm','fargan.f32'):
                actual=report/('request_'+str(case['language_id'])+'_'+name)
                assert actual.read_bytes()==(reference/name).read_bytes(),(case['language'],name)
                equal.append(actual.name)
            packets=sorted(report.glob('request_'+str(case['language_id'])+'_packet_*.pcm'),key=lambda p:int(p.stem.split('_')[-1]))
            actual=report/('request_'+str(case['language_id'])+'_output24.pcm')
            assert b''.join(p.read_bytes()for p in packets)==actual.read_bytes(),case['language']
            verified_languages.append(case['language'])
    elif passed:
        for p in sorted(report.iterdir()):
            if p.name=='qemu.log':continue
            if a.backend_tag.startswith('compact_ring') and p.name in ('request_2_output.pcm','request_2_output24.pcm','request_2_fargan.f32'):
                complete=baseline/p.name.replace('request_2_','request_0_')
                assert p.read_bytes()==complete.read_bytes()[:p.stat().st_size],p
                continue
            assert p.is_file()and(baseline/p.name).is_file(),p
            assert p.read_bytes()==(baseline/p.name).read_bytes(),p
            equal.append(p.name)
        assert len(equal)==(452 if a.backend_tag.startswith('compact_ring') else 454),len(equal)
    events=None
    if csv_path and passed:
        if a.multilang:
            assert f'PROFILE_REQUESTS started={len(cases)} stopped={len(cases)} pending=0'in output
            totals={}
            for row in csv.DictReader(csv_path.open()):
                i=int(row['request_index']);totals[i]=totals.get(i,0)+int(row['request_instruction_events'])
            assert set(totals)==set(range(len(cases)))
            events=[dict(language=case['language'],events=totals[i],within_60M=totals[i]<=60000000)
                for i,case in enumerate(cases)]
        else:
            assert 'PROFILE_MARKERS started=1 stopped=1'in output
            events=sum(int(row['request_instruction_events'])for row in csv.DictReader(csv_path.open()))
    write(report/'RESULT.json',dict(status='PASSED'if passed else'FAILED',elf_sha256=sha(elf),
        returncode=process.returncode,command=command,metrics=metrics,memory=memory,
        identical_output_files=len(equal),host_wall_seconds=time.monotonic()-start,
        rom_below_10MB=memory['rom_load_bytes']<10000000,
        ram_below_10MB=memory['reserved_physical_ram_bytes']<10000000,
        chinese_full_request_delivered_packet_events=events,
        full_request_delivered_packet_events=events,
        native_text_languages_verified=verified_languages,
        profile_sha256=sha(csv_path)if csv_path and csv_path.exists() else None,
        all25_ordinary_raw_text_paths_verified=passed and len(verified_languages)==25,
        resident_frontend_regression_enabled=a.frontend_regression,
        cold_request_timing_eligible=not a.frontend_regression,
        full25_acceptance=False,full25_feature_adapters_pending=len(verified_languages)<25))
    print(json.dumps(dict(passed=passed,identical_output_files=len(equal),memory=memory,chinese_events=events),indent=2))
    if not passed:raise SystemExit(output)


if __name__=='__main__':main()
