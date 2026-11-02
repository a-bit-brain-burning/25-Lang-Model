"""Build and run the target-side compressed resource reader on Cortex-M55."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
NATIVE=ROOT/'native'
TOOLS=NATIVE/'toolchain/sysroot/usr/bin'
CC=NATIVE/'bin/arm-none-eabi-gcc'
XZ=WORK/'m4_frontend_rom_accounting_20261023/sources/xz-5.4.5'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def elf_memory(path):
    data=path.read_bytes()
    assert data[:7]==b'\x7fELF\x01\x01\x01'
    phoff=struct.unpack_from('<I',data,28)[0]
    phsize,phnum=struct.unpack_from('<HH',data,42)
    segments=[]
    for i in range(phnum):
        kind,off,vaddr,paddr,filesz,memsz,flags,align=struct.unpack_from('<8I',data,phoff+i*phsize)
        if kind!=1:continue
        # QSPI is ROM; everything else is physically writable QEMU memory,
        # including code and read-only asset overlays. Conservatively count it.
        is_qspi=0x28000000<=vaddr<0x28800000 or 0x38000000<=vaddr<0x38800000
        segments.append(dict(vaddr=vaddr,paddr=paddr,file_bytes=filesz,memory_bytes=memsz,
            physical_ram_bytes=0 if is_qspi else memsz))
    return dict(rom_load_bytes=sum(s['file_bytes'] for s in segments),
        reserved_physical_ram_bytes=sum(s['physical_ram_bytes'] for s in segments),segments=segments)


def run(command,log,**kwargs):
    command=list(map(str,command))
    with log.open('w')as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,**kwargs)
    if result.returncode:
        raise RuntimeError(str(log)+'\n'+log.read_text()[-5000:])
    return command


def build_lzma():
    out=ROOT/'lzma_m55'
    out.mkdir(exist_ok=True)
    archive=out/'src/liblzma/.libs/liblzma.a'
    if archive.exists():return archive
    env=dict(os.environ)
    env.update(CC=str(CC),AR=str(TOOLS/'arm-none-eabi-ar'),RANLIB=str(TOOLS/'arm-none-eabi-ranlib'),
        CFLAGS='-Os -ffunction-sections -fdata-sections -fno-unwind-tables -fno-asynchronous-unwind-tables',
        LDFLAGS='-nostartfiles --specs='+str(NATIVE/'toolchain/sysroot/usr/lib/arm-none-eabi/newlib/thumb/v8-m.main+dp/hard/nosys.specs'))
    commands=[]
    commands.append(run([XZ/'configure','--host=arm-none-eabi','--disable-shared','--enable-static',
        '--disable-encoders','--enable-decoders=lzma1,lzma2','--disable-threads','--disable-nls',
        '--disable-xz','--disable-xzdec','--disable-lzmadec','--disable-lzmainfo','--disable-doc',
        '--disable-scripts','--disable-sandbox'],out/'configure.log',cwd=out,env=env))
    commands.append(run(['make','-j4','-C','src/liblzma'],out/'make.log',cwd=out,env=env))
    write(out/'BUILD.json',dict(archive_sha256=sha(archive),commands=commands,
        source=str(XZ),source_configure_sha256=sha(XZ/'configure'),target='Cortex-M55'))
    return archive


def generate(exclude=(),destination=None,extra_rom=(),overrides=None):
    doc=json.loads((ROOT/'resources/SECTIONS.json').read_text())
    paths={}
    if overrides:
        import sys
        sys.path.insert(0,str(WORK/'new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24/frontend_bundle/frontend/vendor'))
        import fe_codec as fe
        for name,path in overrides.items():
            path=Path(path);pack=fe.Reader(path);paths[name]=path
            item=next(p for p in doc['packages']if p['name']==name)
            item.update(bytes=path.stat().st_size,sha256=sha(path))
            doc['sections']=[s for s in doc['sections']if s['package']!=name]
            doc['sections'] += [dict(package=name,name=section,sha256=hashlib.sha256(pack.read(section)).hexdigest())for section in pack.sections]
    out=destination or ROOT/'generated';out.mkdir(parents=True,exist_ok=True)
    selected=[s for s in doc['sections']if s['package']not in exclude]
    header=['#ifndef N25_FEP_ASSETS_H','#define N25_FEP_ASSETS_H','#include <stddef.h>',
        '#define FEP_PROBE_PACK_COUNT '+str(len(doc['packages'])),
        '#define FEP_PROBE_SECTION_COUNT '+str(len(selected)),
        '#define FEP_ACTIVE_PACK_COUNT '+str(sum(p['name']not in exclude for p in doc['packages'])),
        '#define FEP_MAX_SECTION_BYTES '+str(doc['max_output_section_bytes']),
        'typedef struct{const unsigned char *data;size_t bytes;} FepProbePack;',
        'typedef struct{unsigned pack;const char *name;unsigned char sha256[32];} FepProbeSection;',
        'extern const FepProbePack fep_probe_packs[FEP_PROBE_PACK_COUNT];',
        'extern const FepProbeSection fep_probe_sections[FEP_PROBE_SECTION_COUNT];','#endif']
    (out/'fep_assets.h').write_text('\n'.join(header)+'\n')
    asm=[]
    code=['#include "fep_assets.h"']
    ids={}
    for i,p in enumerate(doc['packages']):
        ids[p['name']]=i
        if p['name']in exclude:continue
        path=paths.get(p['name'],ROOT/'resources/packages'/p['name'])
        assert sha(path)==p['sha256']
        section='.frontend_assetbank.fep'if p['name']in extra_rom else'.benchmark_assets.fep'
        asm.extend([f'.section {section},"a",%progbits','.balign 16',f'.global n25_fep_pack_{i}',f'n25_fep_pack_{i}:',f'.incbin "{path}"'])
        code.append(f'extern const unsigned char n25_fep_pack_{i}[];')
    asm.append('.section .note.GNU-stack,""')
    (out/'fep_assets.S').write_text('\n'.join(asm)+'\n')
    code.append('const FepProbePack fep_probe_packs[FEP_PROBE_PACK_COUNT]={')
    code.extend('{0,0},'if p['name']in exclude else f'{{n25_fep_pack_{i},{p["bytes"]}}},' for i,p in enumerate(doc['packages']))
    code.append('};\nconst FepProbeSection fep_probe_sections[FEP_PROBE_SECTION_COUNT]={')
    for s in selected:
        digest=','.join(str(b) for b in bytes.fromhex(s['sha256']))
        code.append('{'+f'{ids[s["package"]]},{json.dumps(s["name"])},'+'{'+digest+'}},')
    code.append('};')
    (out/'fep_assets.c').write_text('\n'.join(code)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='resources_v1')
    args=parser.parse_args()
    out=ROOT/'build'/args.tag
    out.mkdir(parents=True,exist_ok=False)
    archive=build_lzma()
    generate()
    flags=['-O3','-g','-std=c11','-ffreestanding','-ffunction-sections','-fdata-sections',
        '-fstack-usage','-fno-unwind-tables','-fno-asynchronous-unwind-tables',
        '-I'+str(NATIVE/'src'),'-I'+str(ROOT/'generated'),'-I'+str(XZ/'src/liblzma/api')]
    inputs=[NATIVE/'src'/s for s in ('startup.S','bench.c','syscalls.c','storage_sha256.c',
        'fep_runtime.c','fep_probe.c')]+[ROOT/'generated/fep_assets.c',ROOT/'generated/fep_assets.S']
    objects=[];commands=[]
    for i,p in enumerate(inputs):
        obj=out/(str(i)+'_'+p.stem+'.o');objects.append(obj)
        commands.append(run([CC,*flags,'-c',p,'-o',obj],out/(str(i)+'.log')))
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,-Map='+str(out/'firmware.map'),'-Wl,--defsym=ARENA_BYTES=900000',
        '-Wl,--defsym=STACK_BYTES=32768','-T',NATIVE/'src/m55.ld',*objects,
        '-Wl,--start-group',archive,'-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    for tool,args_,name in [('readelf',['-lW'],'segments.txt'),('nm',['-n'],'symbols.txt')]:
        (out/name).write_text(subprocess.check_output([str(TOOLS/('arm-none-eabi-'+tool)),*args_,str(elf)],text=True))
    write(out/'BUILD.json',dict(elf=str(elf),elf_sha256=sha(elf),commands=commands,
        input_sha256={str(p):sha(p)for p in inputs},lzma_archive_sha256=sha(archive),
        memory=elf_memory(elf),scope='All25 resource reader probe; not complete TTS firmware'))
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
    passed=process.returncode==0 and 'N25_FEP_RESOURCE_PORT_PASS' in output
    write(report/'RESULT.json',dict(status='PASSED' if passed else 'FAILED',
        elf_sha256=sha(elf),command=command,returncode=process.returncode,
        host_wall_seconds=time.monotonic()-start,metrics=metrics,
        memory=elf_memory(elf),complete_tts=False))
    print(json.dumps(dict(passed=passed,metrics=metrics,memory=elf_memory(elf)),indent=2))
    if not passed:raise SystemExit(output)


if __name__=='__main__':main()
