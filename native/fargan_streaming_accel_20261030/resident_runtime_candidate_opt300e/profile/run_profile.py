"""Run an M55 ELF with timeout, exit checks and distinct build/run/guest timings."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import time

ROOT=Path('/home/fomak/ffb/m55_benchmark_20261030')
QEMU=ROOT.parent/'local_qemu_20261030/bin/qemu-system-arm'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elf',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--expect-marker',default='M55_INFRA_PASS')
    p.add_argument('--expect-exit',type=int,default=0)
    p.add_argument('--expect-timeout',action='store_true')
    p.add_argument('--plugin',required=True)
    p.add_argument('--cpu',type=int,help='Optional host logical CPU affinity; never changes guest core count')
    a=p.parse_args()
    if not 1<=a.repeats<=100 or a.timeout<=0: p.error('Invalid repeats or timeout')
    elf=(a.elf or Path(json.loads((ROOT/'build/LATEST.json').read_text())['elf'])).resolve()
    if not elf.is_file():p.error('ELF missing')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=(a.output or ROOT/'reports'/('run_'+stamp)).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    base=[str(QEMU),'-M','mps3-an547','-cpu','cortex-m55','-accel','tcg,thread=single',
          '-nographic','-monitor','none','-serial','none',
          '-semihosting-config','enable=on,target=native','-kernel',str(elf),'-plugin',a.plugin]
    if a.cpu is not None:
        import os
        if a.cpu not in os.sched_getaffinity(0):p.error('CPU outside allowed affinity')
        base=['taskset','-c',str(a.cpu),*base]
    rows=[]
    for index in range(a.repeats):
        working=out/f'run_{index+1:03d}';working.mkdir()
        start=time.perf_counter();timed_out=False
        try:
            result=subprocess.run(base,cwd=working,capture_output=True,text=True,timeout=a.timeout)
            output=result.stdout+result.stderr;code=result.returncode
        except subprocess.TimeoutExpired as error:
            timed_out=True;code=None
            def decode(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
            output=decode(error.stdout)+decode(error.stderr)
        elapsed=time.perf_counter()-start
        (working/'qemu.log').write_text(output)
        metrics={}
        for line in output.splitlines():
            if not line.startswith('{'):continue
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if item.get('kind')=='metric':metrics.setdefault(item['name'],[]).append(item['value'])
        passed=timed_out if a.expect_timeout else (not timed_out and code==a.expect_exit and (not a.expect_marker or a.expect_marker in output))
        row=dict(index=index+1,host_process_wall_seconds=elapsed,returncode=code,timed_out=timed_out,
                 expected_outcome_observed=passed,guest_metrics=metrics,
                 artifacts={x.name:dict(bytes=x.stat().st_size,sha256=sha(x)) for x in working.iterdir() if x.is_file()})
        rows.append(row)
        print(json.dumps(dict(run=index+1,wall_seconds=elapsed,exit=code,timed_out=timed_out,passed=passed)),flush=True)
        if not passed:break
    metric_summary={}
    for row in rows:
        for name,values in row['guest_metrics'].items():metric_summary.setdefault(name,[]).extend(values)
    summary={name:dict(min=min(v),median=statistics.median(v),max=max(v),samples=len(v)) for name,v in metric_summary.items()}
    build_path=elf.parent/'BUILD.json'
    build=json.loads(build_path.read_text()) if build_path.exists() else None
    if build is not None and build.get('elf_sha256')!=sha(elf):
        raise SystemExit('ELF no longer matches adjacent BUILD.json')
    report=dict(status='PASSED' if len(rows)==a.repeats and all(x['expected_outcome_observed'] for x in rows) else 'FAILED',
                elf=str(elf),elf_sha256=sha(elf),command=base,repeats=rows,
                host_process_wall_seconds_median=statistics.median(x['host_process_wall_seconds'] for x in rows),
                guest_metric_summary=summary,
                build_compilation_and_link_wall_seconds=build.get('total_compilation_and_link_wall_seconds') if build else None,
                timing_semantics={'build':'Host C/assembly compilation and linking; excludes inference',
                    'process':'Host elapsed time including QEMU launch, firmware boot, execution, logs and exit',
                    'guest_ticks':'QEMU8.2.2 SYS_ELAPSED uses host elapsed nanoseconds; includes emulation and boundary overhead, not target cycles',
                    'demo':'The vector loop is a synthetic infrastructure probe, not model inference'},
                memory_semantics={'stack':'Painted stack written high-water estimate; reserved stack is the conservative capacity',
                    'arena':'Peak allocations through bench_alloc only; external allocators must be separately instrumented',
                    'reserved_resident_ram':'Linked ITCM code/data plus reserved arena and stack; QSPI assets reported separately; no OS/model acceptance implied'},
                model_loaded=False if build and build.get('scope')=='infrastructure_probe' else None,
                model_inference_performed=False if build and build.get('scope')=='infrastructure_probe' else None,
                full_model_RAM_acceptance=False,hardware_latency_verified=False)
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Report: '+str(out/'RESULT.json'))
    if report['status']!='PASSED':raise SystemExit(1)

if __name__=='__main__':main()
