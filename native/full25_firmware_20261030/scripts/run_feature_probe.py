"""Build a native feature differential probe with its real dictionary reader."""
import argparse
import subprocess
import sys
from pathlib import Path
from build_resources import ROOT
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--native-root',required=True,type=Path)
    p.add_argument('--arena-bytes',default=800000,type=int);a=p.parse_args();native=a.native_root.resolve()
    command=[sys.executable,str(ROOT/'build_espeak.py'),'--tag',a.tag,'--native-root',str(native),
        '--arena-bytes',str(a.arena_bytes),'--stack-bytes','114688','--probe','four_probe.c','--expect-marker','N25_NATIVE_FOUR_FEATURES_PASS']
    sources=[native/'src/four_frontend.c',native/'src/storage_sha256.c',native/'src/frontend_inflate.c',
        native/'third_party/puff/puff.c',ROOT/'generated/four_tables.c']
    if(native/'dictionary_runtime').is_dir():sources+=sorted(x for x in(native/'dictionary_runtime').glob('*.c')if x.name!='dictionary_probe.c')
    for source in sources:command+=['--extra-source',str(source)]
    subprocess.run(command,check=True)
if __name__=='__main__':main()
