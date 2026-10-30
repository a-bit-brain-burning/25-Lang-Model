"""Build real-model kernel fixtures for QEMU MPS3 AN547 / RV32 virt.

Run test_native.py first to create --assets. No Python scientific packages,
network, root privileges, or original training directories required to build.
"""
from pathlib import Path
import argparse
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['m55','rv32'],required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cc')
    a=p.parse_args();assets=a.assets.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cc=a.cc or ('arm-none-eabi-gcc'if a.target=='m55'else'riscv64-unknown-elf-gcc')
    native=ROOT/'native'
    # No architecture-specific section type syntax in this generated file.
    (out/'weights.S').write_text('.section .model_weights,"a"\n.balign 16\n.global n25_blob\nn25_blob:\n.incbin '+json.dumps(str(assets/'model.bin'))+'\n')
    flags=['-mcpu=cortex-m55','-mthumb','-mfloat-abi=hard','-DNEW25_M55=1']if a.target=='m55'else[
        '-march=rv32imafdc','-mabi=ilp32d','-mcmodel=medany','-msmall-data-limit=0','-mno-relax']
    elf=out/'kernels.elf'
    command=[cc,*flags,'-Os','-std=c11','-ffreestanding','-fno-builtin','-fno-fast-math','-ffp-contract=off',
             '-fno-unwind-tables','-fno-asynchronous-unwind-tables','-nostdlib','-I',str(native),'-I',str(assets),
             '-Wl,--build-id=none','-Wl,-Map='+str(out/'kernels.map'),'-T',str(native/(a.target+'.ld')),
             str(native/('start_'+a.target+'.S')),str(native/'platform.c'),str(native/'tensor.c'),
             str(assets/'model_index.c'),str(assets/'kernel_fixture.c'),str(out/'weights.S'),'-lgcc','-o',str(elf)]
    subprocess.run(command,check=True)
    (out/'BUILD.json').write_text(json.dumps(dict(target=a.target,command=command,elf_bytes=elf.stat().st_size),indent=2)+'\n')
    if a.target=='m55':
        qemu=['qemu-system-arm','-M','mps3-an547','-nographic','-monitor','none','-serial','none',
              '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    else:qemu=['qemu-system-riscv32','-M','virt','-m','128M','-bios','none','-nographic','-monitor','none','-kernel',str(elf)]
    print('Run: '+' '.join(qemu))
    (out/'RUN.json').write_text(json.dumps(dict(command=qemu),indent=2)+'\n')


if __name__=='__main__':main()
