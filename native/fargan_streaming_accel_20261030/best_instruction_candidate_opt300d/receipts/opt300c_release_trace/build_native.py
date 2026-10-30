"""Build a Cortex-M55 ELF without loading any model. Produces a timed build receipt."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def capture(command): return subprocess.check_output(command,text=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,action='append',help='C/assembly inputs; default: independent probe.c')
    p.add_argument('--output',type=Path)
    p.add_argument('--define',action='append',default=[])
    p.add_argument('--opt',choices=['0','1','2','3','s'],default='2')
    p.add_argument('--arena-bytes',type=int,default=1048576)
    p.add_argument('--stack-bytes',type=int,default=65536)
    p.add_argument('--no-latest',action='store_true',help='Keep the default run pointing at the prior build')
    a=p.parse_args()
    for value in a.define:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(=[A-Za-z_0-9+.-]+)?',value):
            p.error('Invalid preprocessor definition')
    if a.arena_bytes<0 or a.arena_bytes>8*1024*1024 or a.arena_bytes%16:
        p.error('arena-bytes must be a nonnegative multiple of16 within8MiB QEMU SDRAM')
    if not 4096<=a.stack_bytes<=512*1024 or a.stack_bytes%16:
        p.error('stack-bytes must be a multiple of16 within4KiB..512KiB')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=(a.output or ROOT/'build'/stamp).resolve()
    if out.exists() and any(out.iterdir()): p.error('Output directory must be empty; prior builds are preserved')
    out.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'src/startup.S',ROOT/'src/bench.c']+[x.resolve() for x in (a.source or [ROOT/'src/probe.c'])]
    for source in sources:
        if not source.is_file(): p.error('Missing source: '+str(source))
    cc=ROOT/'bin/arm-none-eabi-gcc'
    tools=ROOT/'toolchain/sysroot/usr/bin'
    elf=out/'firmware.elf'
    flags=['-O'+a.opt,'-std=c11','-Wall','-Wextra','-Werror','-Wno-unused-function','-fno-aggressive-loop-optimizations','-Wno-unused-variable','-Wno-unused-but-set-variable','-Wno-unused-parameter','-ffreestanding','-fno-builtin',
           '-fno-fast-math','-ffp-contract=off','-ffunction-sections','-fdata-sections','-fstack-usage',
           '-fno-unwind-tables','-fno-asynchronous-unwind-tables','-nostdlib','-nostartfiles',
           '-I',str(ROOT/'src'),'-I/home/fomak/ffb/new25_opus_bwe24_20261030/repo/new25_postprocess/assets/opus_bwe/upstream','-I/home/fomak/ffb/new25_opus_bwe24_20261030/repo/new25_postprocess/assets/opus_bwe/upstream/include','-I/home/fomak/ffb/new25_opus_bwe24_20261030/repo/new25_postprocess/assets/opus_bwe/upstream/dnn','-I/home/fomak/ffb/new25_opus_bwe24_20261030/repo/new25_postprocess/assets/opus_bwe/upstream/celt','-I/home/fomak/ffb/new25_opus_bwe24_20261030/repo/new25_postprocess/assets/opus_bwe/upstream/silk',*['-D'+x for x in a.define]]
    compile_steps=[];objects=[]
    for index,source in enumerate(sources):
        obj=out/(str(index)+'_'+source.stem+'.o')
        cmd=[str(cc),*flags,*(['-Wno-error']if source.is_relative_to(Path('/home/fomak/ffb/new25_opus_bwe24_20261030/repo/new25_postprocess/assets/opus_bwe'))or source.name in ['bbwenet_data_rowmajor.c','nnet_default.c','nndsp.c']else []),'-c',str(source),'-o',str(obj)]
        start=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True)
        row=dict(command=cmd,wall_seconds=time.perf_counter()-start,returncode=r.returncode)
        compile_steps.append(row)
        (out/(str(index)+'_compile.log')).write_text(r.stdout+r.stderr)
        if r.returncode:
            (out/'FAILED.json').write_text(json.dumps(dict(stage='compile',steps=compile_steps),indent=2)+'\n')
            raise SystemExit(r.stdout+r.stderr)
        objects.append(obj)
    cmd=[str(cc),'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
         '-Wl,-Map='+str(out/'firmware.map'),'-Wl,--defsym=ARENA_BYTES='+str(a.arena_bytes),
         '-Wl,--defsym=STACK_BYTES='+str(a.stack_bytes),'-T',str(ROOT/'src/m55.ld'),
         *map(str,objects),'-Wl,--start-group','-lm','-lc','-lgcc','-Wl,--end-group','-o',str(elf)]
    start=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True)
    link_seconds=time.perf_counter()-start
    (out/'link.log').write_text(r.stdout+r.stderr)
    if r.returncode:
        (out/'FAILED.json').write_text(json.dumps(dict(stage='link',command=cmd,compile=compile_steps),indent=2)+'\n')
        raise SystemExit(r.stdout+r.stderr)
    for binary,args,name in [('arm-none-eabi-objdump',['-d'],'disassembly.txt'),
                             ('arm-none-eabi-readelf',['-lW'],'segments.txt'),
                             ('arm-none-eabi-readelf',['-AW'],'attributes.txt'),
                             ('arm-none-eabi-size',['-A'],'sections.txt'),
                             ('arm-none-eabi-nm',['-n'],'symbols.txt')]:
        (out/name).write_text(capture([str(tools/binary),*args,str(elf)]))
    sections=[]
    for line in (out/'sections.txt').read_text().splitlines():
        fields=line.split()
        if len(fields)==3 and fields[0].startswith('.'):
            sections.append(dict(name=fields[0],bytes=int(fields[1]),address=int(fields[2])))
    identity={str(x):sha(x) for x in sources+[ROOT/'src/bench.h',ROOT/'src/m55.ld',cc,Path(__file__).resolve()]}
    result=dict(status='BUILT',target='MPS3_AN547_CORTEX_M55_MVE',elf=str(elf),elf_sha256=sha(elf),
                scope='infrastructure_probe' if a.source is None else 'user_supplied_firmware',
                compiler=capture([str(cc),'--version']).splitlines()[0],sources_sha256=identity,
                compile_steps=compile_steps,compile_wall_seconds=sum(x['wall_seconds'] for x in compile_steps),
                link_command=cmd,link_wall_seconds=link_seconds,
                total_compilation_and_link_wall_seconds=sum(x['wall_seconds'] for x in compile_steps)+link_seconds,
                sections=sections,arena_reserved_bytes=a.arena_bytes,stack_reserved_bytes=a.stack_bytes,
                host_time_is_not_inference_time=True)
    (out/'BUILD.json').write_text(json.dumps(result,indent=2)+'\n')
    if not a.no_latest:
        (ROOT/'build/LATEST.json').write_text(json.dumps(dict(directory=str(out),elf=str(elf)),indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','elf','total_compilation_and_link_wall_seconds']},indent=2))

if __name__=='__main__':main()
