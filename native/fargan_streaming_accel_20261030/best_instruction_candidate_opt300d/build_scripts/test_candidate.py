"""QEMU six-file identity + dynamic instruction ROI test for one candidate tag."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parent
M55=ROOT.parent
IP=M55/'instruction_profile'; BENCH=Path('/home/fomak/ffb/m55_benchmark_20261030')
NMBIN=ROOT/'toolchain/sysroot/usr/bin/arm-none-eabi-nm'
FILES=['condition.f32','durations.u32','output.pcm','output24.pcm','first24.pcm','fargan.f32']
REFS=dict(integrated=M55/'integrated/reports/final_v1_ordinary/run_001',
          final_native=M55/'final_native/reports/final_v1_zh/run_001')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def symbols(elf):
    d={}
    for line in subprocess.check_output([str(NMBIN),'-n',str(elf)],text=True).splitlines():
        p=line.split()
        if len(p)==3:d[p[2]]=int(p[0],16)
    return d
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--skip-run',action='store_true');a=ap.parse_args()
    tag=a.tag;build=ROOT/'build'/tag;elf=build/'firmware.elf';run=ROOT/'reports'/tag/'run_001'
    result={"tag":tag,"elf":str(elf),"elf_sha256":sha(elf)}
    if not a.skip_run:
        out=ROOT/'reports'/tag
        if out.exists():shutil.rmtree(out)
        p=subprocess.run([sys.executable,str(BENCH/'run.py'),'--elf',str(elf),'--output',str(out),
            '--repeats','1','--timeout','300','--expect-marker','NEW25_FARGAN_M55_PASS'],capture_output=True,text=True)
        assert p.returncode==0,(p.stdout,p.stderr)
        result['qemu']=p.stdout.strip()
    ident={}
    for n in FILES:
        data=(run/n).read_bytes()
        ident[n]=dict(sha256=sha(run/n),**{k:data==(r/n).read_bytes() for k,r in REFS.items()})
    assert all(v['integrated'] and v['final_native'] for v in ident.values()),ident
    result['six_file_identity']=ident
    sym=symbols(elf);begin=sym['tensor_views_prepare'];end=sym['snapshot_first_active_costs']
    csvp=IP/('roi_'+tag+'_instruction_counts.csv')
    profout=IP/'reports'/('roi_'+tag)
    if profout.exists():shutil.rmtree(profout)
    plugin=str(IP/'profile.so')+',begin='+hex(begin)+',end='+hex(end)+',out='+str(csvp)
    p=subprocess.run([sys.executable,str(IP/'run_profile.py'),'--elf',str(elf),'--output',str(profout),
        '--repeats','1','--timeout','300','--expect-marker','NEW25_FARGAN_M55_PASS','--plugin',plugin],capture_output=True,text=True)
    assert p.returncode==0,(p.stdout,p.stderr)
    total=sum(int(r['request_instruction_events']) for r in csv.DictReader(csvp.open()))
    result.update(begin_pc=hex(begin),end_pc=hex(end),instruction_events=total,
        events_vs_release_o3=total-68816649,
        reduction_vs_release_o3=1-total/68816649.0,
        uncalibrated_1ipc_floor_ms=1000*total/200e6,
        csv=str(csvp),csv_sha256=sha(csvp))
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('tag','instruction_events','events_vs_release_o3','reduction_vs_release_o3','uncalibrated_1ipc_floor_ms')},indent=2))
if __name__=='__main__':main()
