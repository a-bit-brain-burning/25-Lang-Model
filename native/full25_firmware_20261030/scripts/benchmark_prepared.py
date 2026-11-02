"""Diagnostic only: apply the optimized backend to the existing 25-language inputs.

These ELFs receive HOST-PREPARED features. Their instruction counts help locate
backend bottlenecks and can NEVER pass complete raw-text firmware acceptance.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from build_resources import ROOT,WORK,NATIVE,TOOLS,sha,write,elf_memory


def one(language):
    root=ROOT/'prepared_backend'/language
    if(root/'DIAGNOSTIC_RESULT.json').exists():return json.loads((root/'DIAGNOSTIC_RESULT.json').read_text())
    root.mkdir(parents=True,exist_ok=True)
    if not(root/'src').exists():
        shutil.copytree(NATIVE/'src',root/'src')
        (root/'assets').mkdir()
        for p in(NATIVE/'assets').iterdir():
            if p.suffix in('.h','.c','.S'):shutil.copy2(p,root/'assets'/p.name)
        for name in('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
            (root/name).symlink_to(NATIVE/name,target_is_directory=True)
        for name in('build.py','build_native.py','build_o3_resident.py'):shutil.copy2(NATIVE/name,root/name)
        prepared=WORK/'m55_fargan_stream_20261030/language25_20261030/inputs'/language
        (root/'assets/fixed_input.h').write_text('#include "../src/text_frontend.h"\n'+(prepared/'fixed_input.h').read_text())
        (root/'src/text_frontend.c').write_text('#include "text_frontend.h"\nint n25_text_frontend(const char*text,N25TextFeatures*out){(void)text;(void)out;return 0;}\n')
        p=root/'src/persistent_runtime.inc';s=p.read_text()
        s=s.replace('int prewarm_rc=native_zh_prewarm();','int prewarm_rc=0;')
        s=s.replace('bench_metric("runtime_prepared_features_only",0);bench_metric("runtime_native_Han_frontend",1);',
            'bench_metric("runtime_prepared_features_only",1);bench_metric("runtime_native_Han_frontend",0);')
        p.write_text(s)
    if not(root/'build/optimized/firmware.elf').exists():
        if(root/'build/optimized').exists():
            index=1
            while(root/'build'/('failed_'+str(index))).exists():index+=1
            (root/'build/optimized').rename(root/'build'/('failed_'+str(index)))
        with(root/'BUILD.log').open('w')as log:
            subprocess.run([sys.executable,str(root/'build_o3_resident.py'),'--tag','optimized'],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    elf=root/'build/optimized/firmware.elf'
    symbols={}
    for line in subprocess.check_output([str(TOOLS/'arm-none-eabi-nm'),'-n',str(elf)],text=True).splitlines():
        fields=line.split()
        if len(fields)==3:symbols[fields[2]]=int(fields[0],16)&~1
    csv_path=root/'instructions.csv'
    old=WORK/'m55_fargan_stream_20261030/opt300c_20261030/instruction_profile'
    plugin=str(old/'profile.so')+',begin='+hex(symbols['tts_begin_text_request'])+',end='+hex(symbols['snapshot_first_active_costs'])+',out='+str(csv_path)
    report=root/'reports/optimized'
    if not(report/'RESULT.json').exists():
        with(root/'RUN.log').open('w')as log:
            subprocess.run([sys.executable,str(old/'run_profile.py'),'--elf',str(elf),'--output',str(report),
                '--repeats','1','--timeout','240','--expect-marker','NEW25_PERSISTENT_RUNTIME_PASS',
                '--plugin',plugin],stdout=log,stderr=subprocess.STDOUT,check=True)
    events=sum(int(row['request_instruction_events'])for row in csv.DictReader(csv_path.open()))
    result=json.loads((report/'RESULT.json').read_text());assert result['status']=='PASSED'
    assert 'PROFILE_MARKERS started=1 stopped=1'in(report/'run_001/qemu.log').read_text()
    baseline=WORK/'m55_fargan_stream_20261030/language25_20261030/native/reports'/(language+'_v1')/'run_001'
    outputs={}
    for n in('condition.f32','durations.u32','output.pcm','output24.pcm','fargan.f32'):
        actual=report/'run_001'/('request_0_'+n)
        assert actual.read_bytes()==(baseline/n).read_bytes(),(language,n)
        for request in(1,3):assert (report/'run_001'/('request_'+str(request)+'_'+n)).read_bytes()==actual.read_bytes()
        outputs[n]=sha(actual)
    document=dict(language=language,status='PREPARED_BACKEND_ONLY_VERIFIED',request_instruction_events=events,
        backend_diagnostic_within_60M=events<=60000000,full_text_request_accepted=False,
        scope='Host-prepared features; this count excludes the actual native language frontend and cannot establish the user target.',
        elf_sha256=sha(elf),output_sha256=outputs,memory=elf_memory(elf))
    write(root/'DIAGNOSTIC_RESULT.json',document)
    print(json.dumps(dict(language=language,events=events,delta_to_60M=events-60000000,scope='prepared_backend_only')),flush=True)
    return document


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--languages',nargs='+',
        default='en fi hu zh bg cs da de el es et fr hr it lt lv nl no pl pt ro sk sl sv tr'.split())
    a=parser.parse_args();rows=[]
    for language in a.languages:
        row=one(language);rows.append(row)
        write(ROOT/'PREPARED_BACKEND_PROGRESS.json',dict(completed=len(rows),requested=len(a.languages),
            cases=rows,full25_acceptance=False,scope='Prepared-feature backend diagnostics only'))
    write(ROOT/'PREPARED_BACKEND_RESULTS.json',dict(cases=rows,full25_acceptance=False,
        all_outputs_byte_identical=True,scope='Prepared-feature backend diagnostics only'))


if __name__=='__main__':main()
