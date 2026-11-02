"""Measure a candidate at the original full-request anchors and compare all outputs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from build_resources import ROOT,WORK,NATIVE,TOOLS,elf_memory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--end-symbol',default='tts_first_effective_packet_delivered');parser.add_argument('--expected-files',type=int,default=456)
    parser.add_argument('--stream-captures',action='store_true');args=parser.parse_args()
    elf=NATIVE/'build'/args.tag/'firmware.elf'
    symbols={}
    for line in subprocess.check_output([str(TOOLS/'arm-none-eabi-nm'),'-n',str(elf)],text=True).splitlines():
        words=line.split()
        if len(words)==3:symbols[words[2]]=int(words[0],16)&~1
    profile=ROOT/'profile';profile.mkdir(exist_ok=True)
    csv_path=profile/(args.tag+'.csv')
    old=WORK/'m55_fargan_stream_20261030/opt300c_20261030/instruction_profile'
    plugin=str(old/'profile.so')+',begin='+hex(symbols['tts_begin_text_request'])+',end='+hex(symbols[args.end_symbol])+',out='+str(csv_path)
    output=ROOT/'reports'/args.tag
    subprocess.run([sys.executable,str(old/'run_profile.py'),'--elf',str(elf),
        '--output',str(output),'--repeats','1','--timeout','240',
        '--expect-marker','NEW25_PERSISTENT_RUNTIME_PASS','--plugin',plugin],check=True)
    data=list(csv.DictReader(csv_path.open()))
    events=sum(int(row['request_instruction_events'])for row in data)
    log=(output/'run_001/qemu.log').read_text()
    assert 'PROFILE_MARKERS started=1 stopped=1' in log
    begin=[row for row in data if int(row['pc'],16)==symbols['tts_begin_text_request']]
    end=[row for row in data if int(row['pc'],16)==symbols[args.end_symbol]]
    assert sum(int(row['request_instruction_events'])for row in begin)==1
    assert sum(int(row['request_instruction_events'])for row in end)==0
    reference=WORK/'m55_fargan_stream_20261030/opt300c_20261030/final_resident/reports/resident_final_v1/run_001'
    comparisons={}
    for p in sorted((output/'run_001').iterdir()):
        if not p.is_file()or p.name in ('qemu.log',):continue
        if p.suffix=='.wav':continue
        baseline=reference/p.name
        if args.stream_captures and p.name in ('request_2_output.pcm','request_2_output24.pcm','request_2_fargan.f32'):
            complete=reference/p.name.replace('request_2_','request_0_')
            assert p.read_bytes()==complete.read_bytes()[:p.stat().st_size],p
            continue
        assert baseline.is_file(),p
        same=p.read_bytes()==baseline.read_bytes();assert same,p
        comparisons[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(comparisons)==args.expected_files,len(comparisons)
    result=dict(status='BYTE_IDENTICAL_REQUEST_PROFILE_VERIFIED',request_instruction_events=events,
        baseline_full_request_events=59988134,delta_events=events-59988134,
        instruction_limit_passed=events<=60000000,elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
        roi_begin='tts_begin_text_request',roi_end=args.end_symbol,
        output_files_equal=len(comparisons),file_sha256=comparisons,memory=elf_memory(elf),
        scope='Chinese native-text request regression; not all25 acceptance',
        profile_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
    (output/'PROFILE_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ('status','request_instruction_events','delta_events','output_files_equal','memory')},indent=2))


if __name__=='__main__':main()
