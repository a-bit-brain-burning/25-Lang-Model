"""Run the shipped real-model kernel smoke on either emulator, without root.

This checks three real layers; it does not run the full text-to-waveform graph.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['m55','rv32'],required=True)
    p.add_argument('--elf',type=Path,help='Run a rebuilt ELF instead of the shipped fixture')
    p.add_argument('--qemu',help='Override QEMU executable path')
    p.add_argument('--timeout',type=float,default=60)
    a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='new25-qemu-')as tmp:
        if a.elf:elf=a.elf.resolve()
        else:
            shipped=ROOT/'simulator/prebuilt'
            manifest=json.loads((shipped/'MANIFEST.json').read_text())
            raw=gzip.decompress((shipped/(a.target+'.elf.gz')).read_bytes())
            assert hashlib.sha256(raw).hexdigest()==manifest['items'][a.target]['elf_sha256']
            elf=Path(tmp)/'kernels.elf';elf.write_bytes(raw)
        if a.target=='m55':
            command=[a.qemu or 'qemu-system-arm','-M','mps3-an547','-nographic','-monitor','none',
                     '-serial','none','-semihosting-config','enable=on,target=native','-kernel',str(elf)]
        else:
            command=[a.qemu or 'qemu-system-riscv32','-M','virt','-m','128M','-bios','none',
                     '-nographic','-monitor','none','-kernel',str(elf)]
        r=subprocess.run(command,text=True,capture_output=True,timeout=a.timeout)
        print(r.stdout+r.stderr,end='')
        if r.returncode or 'NEW25_REAL_INT8_KERNELS_PASS'not in(r.stdout+r.stderr):
            raise SystemExit('Simulator kernel check failed, exit='+str(r.returncode))


if __name__=='__main__':main()
