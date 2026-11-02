"""Port the pinned phoneme-only eSpeak engine to the bounded native ROM reader."""
from pathlib import Path
import ctypes
import importlib.util
import json
import shutil
import subprocess
import sys
import time

from build_resources import ROOT,WORK,NATIVE,TOOLS,CC,XZ,sha,write,run,elf_memory,build_lzma,generate

VENDOR=WORK/'new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24/frontend_bundle/frontend/vendor'
SOURCE=WORK/'m4_frontend_4mb_20261022/upstream/espeak-ng'
AUDIT=WORK/'m4_frontend_rom_accounting_20261023'
PATCHED=WORK/'m4_frontend_phonemes_20261023/patched'
UNICODE=WORK/'m4_frontend_exact_20261023'
WRAPS='malloc calloc realloc free strdup stat fopen fclose fread fseek ftell fgets fgetc rewind feof ferror fflush getenv opendir readdir closedir'.split()


def generate_fs():
    sys.path.insert(0,str(VENDOR))
    import fe_codec as fe
    resources=ROOT/'resources/packages';generated=ROOT/'generated'
    doc=json.loads((ROOT/'resources/SECTIONS.json').read_text())
    ids={p['name']:i for i,p in enumerate(doc['packages'])}
    pack=fe.Reader(resources/'espeak.fep')
    metadata=fe.Codec([]).decode(pack.read('files'))
    files={row['name']:dict(name=row['name'],pack=ids['espeak.fep'],part=0,
        offset=row['offset'],bytes=row['bytes'])for row in metadata['files']}
    integration=fe.Reader(resources/'integration-supplement.fep')
    manifest=json.loads((ROOT/'resources/UPSTREAM_MANIFEST.json').read_text())
    for row in manifest['additional_resources']:
        name=row['path']
        if not name.startswith('espeak-ng-data/'):continue
        index=int(row['section'].split('/')[1]);blob=integration.read(row['section'])
        import hashlib
        assert hashlib.sha256(blob).hexdigest()==row['sha256']
        name=name.removeprefix('espeak-ng-data/')
        files[name]=dict(name=name,pack=ids['integration-supplement.fep'],part=index|0x80000000,
            offset=0,bytes=len(blob))
    files['ro_dict']=dict(name='ro_dict',pack=0,part=0,offset=0,
        bytes=(ROOT/'resources/extra_resources/ro_dict').stat().st_size,direct='n25_rom_ro_dict')
    code=['#include "romfs.h"','extern const unsigned char n25_rom_ro_dict[];',
        'const N25RomFile n25_rom_files[]={']
    for name,row in sorted(files.items()):
        code.append('{'+json.dumps(name)+','+','.join(str(row[k])for k in ('pack','part','offset','bytes'))+','+row.get('direct','0')+'},')
    code+=['};','const unsigned n25_rom_file_count=sizeof(n25_rom_files)/sizeof(n25_rom_files[0]);']
    (generated/'romfs_files.c').write_text('\n'.join(code)+'\n')
    (generated/'romfs_direct.S').write_text('.section .benchmark_assets.romfs,"a",%progbits\n.balign 16\n.global n25_rom_ro_dict\nn25_rom_ro_dict:\n.incbin "'+str(ROOT/'resources/extra_resources/ro_dict')+'"\n.section .note.GNU-stack,""\n')
    # Host-side comparison only: restore the exact resource bytes into an
    # independent directory and call the pinned, phoneme-only native library.
    directory=ROOT/'host_reference/espeak-ng-data';directory.mkdir(parents=True,exist_ok=True)
    part_cache={}
    for name,row in files.items():
        if row.get('direct'):blob=(ROOT/'resources/extra_resources/ro_dict').read_bytes()
        elif row['part']&0x80000000:blob=integration.read('resource/'+str(row['part']&0x7fffffff))
        else:
            at=row['offset'];left=row['bytes'];blob=bytearray()
            while left:
                i,pos=divmod(at,262144)
                if i not in part_cache:part_cache[i]=pack.read('part/'+str(i))
                chunk=part_cache[i][pos:pos+left];assert chunk
                blob.extend(chunk);left-=len(chunk);at+=len(chunk)
        assert len(blob)==row['bytes']
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    write(generated/'ROMFS_IDENTITY.json',dict(files=files,
        reconstructed_sha256={name:sha(directory/name)for name in files},
        total_uncompressed_file_bytes=sum(r['bytes']for r in files.values())))
    warm=sum((row['bytes']+15)//16*16 for row in files.values()if not row.get('direct'))
    (generated/'romfs_capacity.h').write_text('#define N25_ROMFS_FILE_COUNT '+str(len(files))+'\n#define N25_ROMFS_WARM_CAPACITY '+str(warm)+'\n')
    with (generated/'romfs_files.c').open('a')as stream:
        stream.write('\nconst unsigned char n25_rom_sha256['+str(len(files))+'][32]={\n')
        for name in sorted(files):stream.write('{'+','.join(str(x)for x in bytes.fromhex(sha(directory/name)))+'},\n')
        stream.write('};\n')
    library=ctypes.CDLL(str(VENDOR/'frontend/libs/libespeak_phonemes.so'))
    library.espeak_Initialize.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_int]
    library.espeak_SetVoiceByName.argtypes=[ctypes.c_char_p]
    library.espeak_TextToPhonemes.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.c_int]
    library.espeak_TextToPhonemes.restype=ctypes.c_char_p
    assert library.espeak_Initialize(2,0,str(directory).encode(),0)>0
    baseline=json.loads((WORK/'m55_fargan_stream_20261030/language25_20261030/FINAL_RESULTS.json').read_text())
    voices={'en':'en-us','no':'nb'}
    cases=[]
    for row in baseline['cases']:
        # Mandarin uses the native reference frontend. Spanish uses its own
        # lexicon/predictor; the pinned resource closure contains no es_dict.
        if row['language'] in ('zh','es'):continue
        voice=voices.get(row['language'],row['language'])
        assert library.espeak_SetVoiceByName(voice.encode())==0,voice
        data=ctypes.create_string_buffer(row['text'].encode());cursor=ctypes.c_void_p(ctypes.addressof(data))
        clauses=[]
        while cursor.value:
            previous=cursor.value
            raw=library.espeak_TextToPhonemes(ctypes.byref(cursor),1,2|(ord('_')<<8))
            clauses.append(raw.decode()if raw else '')
            assert cursor.value!=previous
        cases.append(dict(language=row['language'],voice=voice,text=row['text'],ipa=' '.join(clauses).strip()))
    library.espeak_Terminate()
    write(generated/'ESPEAK_CASES.json',cases)
    code=['#ifndef N25_ESPEAK_CASES_H','#define N25_ESPEAK_CASES_H',
        'typedef struct{const char *language,*voice,*text,*ipa;} EspeakCase;',
        'static const EspeakCase espeak_cases[]={']
    code.extend('{'+','.join(json.dumps(r[k],ensure_ascii=False)for k in ('language','voice','text','ipa'))+'},'for r in cases)
    code+=['};','#define ESPEAK_CASE_COUNT (sizeof(espeak_cases)/sizeof(espeak_cases[0]))','#endif']
    (generated/'espeak_cases.h').write_text('\n'.join(code)+'\n')


def main():
    global NATIVE
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='espeak_v1')
    parser.add_argument('--arena-bytes',type=int,default=16);parser.add_argument('--native-root',type=Path);parser.add_argument('--probe',default='espeak_probe.c');parser.add_argument('--extra-source',action='append',default=[])
    parser.add_argument('--stack-bytes',type=int,default=65536)
    parser.add_argument('--expect-marker',default='N25_ROMFS_ESPEAK_PASS');args=parser.parse_args()
    if args.native_root:NATIVE=args.native_root.resolve()
    archive=build_lzma();generate();generate_fs()
    out=ROOT/'build'/args.tag;out.mkdir(parents=True,exist_ok=False)
    compat=ROOT/'generated/compat';compat.mkdir(exist_ok=True)
    shutil.copy2(AUDIT/'compat/dirent.h',compat/'dirent.h')
    (ROOT/'generated/config.h').write_text('#define HAVE_MKSTEMP 0\n'+''.join('#define '+x+' 0\n'for x in
        ('USE_ASYNC','USE_KLATT','USE_LIBPCAUDIO','USE_LIBSONIC','USE_MBROLA','USE_SPEECHPLAYER'))+'#define PACKAGE_VERSION "1.52.0"\n')
    flags=['-Os','-g','-ffunction-sections','-fdata-sections','-fstack-usage','-fno-unwind-tables',
        '-fno-asynchronous-unwind-tables','-fwrapv','-DLIBESPEAK_NG_EXPORT=1',
        '-DPATH_ESPEAK_DATA="/espeak-ng-data"','-D_DEFAULT_SOURCE','-D_POSIX_C_SOURCE=200809L',
        *['-I'+str(p)for p in (NATIVE/'src',ROOT/'generated',compat,SOURCE/'src/include',
            SOURCE/'src/include/compat',SOURCE/'src/libespeak-ng',SOURCE/'src/ucd-tools/src/include',XZ/'src/liblzma/api',NATIVE/'dictionary_runtime')]]
    jobs=json.loads((AUDIT/'ARM_OBJECTS.json').read_text())
    inputs=[]
    for row in jobs:
        if row['group']not in ('espeak','ucd'):continue
        p=Path(row['source'])
        if p.stem in ('speech','synthdata','synthesize'):p=PATCHED/p.name
        if p.parent.name=='src'and 'ucd-tools'in str(p):
            if p.stem=='case':p=UNICODE/'case_compact.c'
            if p.stem=='proplist':p=UNICODE/'properties_compact.c'
        inputs.append((row['group']+'_'+p.stem,p))
    inputs += [(p.stem,p)for p in [NATIVE/'src'/n for n in ('startup.S','bench.c','syscalls.c',
        'fep_runtime.c','romfs.c','frontend_syscalls.c',args.probe)]+[ROOT/'generated'/n for n in ('fep_assets.c','fep_assets.S','romfs_files.c','romfs_direct.S')]]
    inputs += [(Path(p).stem,Path(p))for p in args.extra_source]
    commands=[];objects=[]
    for i,(name,p)in enumerate(inputs):
        obj=out/(str(i)+'_'+name+'.o');objects.append(obj)
        commands.append(run([CC,*flags,'-c',p,'-o',obj],out/(str(i)+'.log')))
    linker=(NATIVE/'src/m55.ld').read_text()
    linker=linker.replace('  .zhprewarm (NOLOAD)',
        '  .frontend_workspace (NOLOAD) : ALIGN(16) { *(.frontend_workspace*) } > SRAM\n  .zhprewarm (NOLOAD)')
    (out/'m55.ld').write_text(linker)
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,-Map='+str(out/'firmware.map'),'-Wl,--defsym=ARENA_BYTES='+str(args.arena_bytes),'-Wl,--defsym=STACK_BYTES='+str(args.stack_bytes),
        *['-Wl,--wrap='+s for s in WRAPS],'-T',out/'m55.ld',*objects,
        '-Wl,--start-group',archive,'-lm','-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    write(out/'BUILD.json',dict(elf=str(elf),elf_sha256=sha(elf),commands=commands,
        input_sha256={str(p):sha(p)for _,p in inputs},memory=elf_memory(elf),
        scope='Native compressed-resource eSpeak IPA probe, not complete NEW25 features or TTS'))
    report=ROOT/'reports'/args.tag;report.mkdir(parents=True,exist_ok=False)
    qemu=WORK/'local_qemu_20261030/bin/qemu-system-arm'
    command=[str(qemu),'-M','mps3-an547','-cpu','cortex-m55','-accel','tcg,thread=single',
        '-nographic','-serial','none','-monitor','none','-semihosting-config','enable=on,target=native',
        '-kernel',str(elf)]
    start=time.monotonic()
    process=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=180)
    output=process.stdout+process.stderr;(report/'qemu.log').write_text(output)
    metrics={}
    for line in output.splitlines():
        if line.startswith('{'):
            row=json.loads(line)
            if row.get('kind')=='metric':metrics[row['name']]=row['value']
    passed=process.returncode==0 and args.expect_marker in output
    write(report/'RESULT.json',dict(status='PASSED'if passed else 'FAILED',
        elf_sha256=sha(elf),command=command,returncode=process.returncode,
        host_wall_seconds=time.monotonic()-start,metrics=metrics,memory=elf_memory(elf),complete_tts=False))
    print(json.dumps(dict(passed=passed,metrics=metrics,memory=elf_memory(elf)),indent=2))
    if not passed:raise SystemExit(output)


if __name__=='__main__':main()
