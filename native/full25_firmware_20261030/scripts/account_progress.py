"""Record a verified intermediate checkpoint without accepting the full goal."""
import bisect
import argparse
import collections
import csv
import json
import subprocess
from pathlib import Path
from build_resources import ROOT,TOOLS,sha,write,elf_memory

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='native17_pair_interleaved_v1')
    parser.add_argument('--native-root',default='native_pair_interleaved')
    parser.add_argument('--case-manifest',default='SPANISH_SOURCE_IDENTITY.json')
    args=parser.parse_args()
    tag=args.tag;build=ROOT/'build'/tag;report=ROOT/'reports'/tag
    d=json.loads((report/'RESULT.json').read_text());elf=build/'firmware.elf'
    cases=json.loads((ROOT/args.case_manifest).read_text())['native_text_cases']
    count=len(cases);outputs=count*5
    assert d['status']=='PASSED' and d['identical_output_files']==outputs
    assert d['elf_sha256']==sha(elf) and d['memory']==elf_memory(elf)
    assert d['rom_below_10MB'] and d['ram_below_10MB']
    events=d['full_request_delivered_packet_events'];assert len(events)==count
    symbols=[]
    for line in subprocess.check_output([str(TOOLS/'arm-none-eabi-nm'),'-n','-S',str(elf)],text=True).splitlines():
        f=line.split()
        if len(f)==4 and f[2].lower()=='t':symbols.append((int(f[0],16)&~1,int(f[1],16),f[3]))
    addresses=[x[0]for x in symbols];totals=collections.defaultdict(collections.Counter)
    for row in csv.DictReader((ROOT/'profile'/(tag+'.csv')).open()):
        pc=int(row['pc'],16);i=bisect.bisect_right(addresses,pc)-1
        assert i>=0 and pc<symbols[i][0]+symbols[i][1],hex(pc)
        totals[int(row['request_index'])][symbols[i][2]]+=int(row['request_instruction_events'])
    for i,row in enumerate(events):assert sum(totals[i].values())==row['events']
    hot=[dict(language=c['language'],instruction_events=events[i]['events'],
        top_functions=[dict(function=name,events=n)for name,n in totals[i].most_common(20)])for i,c in enumerate(cases)]
    write(ROOT/'LATENCY_HOTSPOTS.json',dict(elf_sha256=sha(elf),requests=hot))
    passed=[row['language']for row in events if row['within_60M']]
    remaining=sorted(set('bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh'.split())-set(c['language']for c in cases))
    over=[row for row in events if not row['within_60M']]
    status=dict(status=f'INTERMEDIATE_{count}_LANGUAGE_TEXT_CASES_VERIFIED_FULL25_NOT_COMPLETE',complete=False,
        acceptance=dict(platform='QEMU mps3-an547 Cortex-M55, single vCPU',
            user_override='不用开发板，就用qemu验收，60M指令数以内即可',hardware_board_required=False,
            latest_priority='先把25种语言的前端全接通，有一点超时可以接受',
            instruction_limit_is_target_with_unquantified_user_tolerance=True,
            request_instruction_limit=60000000,rom_limit_exclusive=10000000,ram_limit_exclusive=10000000,
            begin='tts_begin_language_text_request entry included',
            end='tts_first_effective_packet_delivered entry excluded',
            packet='480 PCM24 samples delivered; RMS >= sqrt(0.0000316227766), peak >= 0.01',
            boot_is_separate=True,language_neutral_boot_resources_only=True),
        best_verified_intermediate_build=str(elf.relative_to(ROOT)),elf_sha256=sha(elf),
        native_source_root=args.native_root,report=str((report/'RESULT.json').relative_to(ROOT)),
        memory=d['memory'],prototype_rom_below_10MB=True,prototype_ram_below_10MB=True,
        native_raw_text_smoke_case_languages=[c['language']for c in cases],native_raw_text_smoke_case_count=count,
        all25_ordinary_raw_text_paths_same_image_verified=count==25,
        native_family_smoke_adapters_pending=bool(remaining),
        ordinary_NFC_subset_only=True,full_text_frontends_complete=False,
        native_family_adapters_remaining=remaining,
        full_request_instruction_events=events,languages_within_60M=passed,
        target_language_count=25,all25_request_instruction_limit_accepted=False,
        full25_same_image_text_acceptance=False,full_system_rom_accepted=False,full_system_ram_accepted=False,
        identical_core_output_files=outputs,zero_tail_bit_exact_edge_cases=2048,
        exact_kernel_edge_receipts=['reports/pair_quantized_edges_v1/RESULT.json',
            'reports/float_matrix_edges_v1/RESULT.json','reports/float_load_adaconv_edges_v1/RESULT.json'],
        instruction_counts_are_not_physical_cycle_measurements=True,
        original_weights_unchanged=True,training_updates=0,default_pointer_changed=False,published=False,
        explicit_pending=([f'{len(remaining)} remaining language family adapters']if remaining else[])+[
            'Full numeric/identifier/abbreviation and Unicode normalization',
            f'{len(over)} of {count} current request cases still exceed 60M; worst '+str(max(row['events']for row in events)),
            'Long-text bounded planning and broader language text regressions',
            'Recheck both 10MB limits after all remaining executable frontends are integrated'],
        rejected_experiments=[dict(name='RNG scalar alignment peel',reason='Slower; reverted before native12_prewarm_v3'),
            dict(name='native14_zero_tail_v1',reason='Bit-exact but slower because scalar finite checks called memcpy per sample; superseded by vector checks')])
    write(ROOT/'STATUS.json',status)
    write(ROOT/'PROGRESS_CHECKPOINT.json',status)
    lines=[
        'INTERMEDIATE CHECKPOINT — FULL 25-LANGUAGE GOAL NOT COMPLETE',
        '',
        'Acceptance: user explicitly chose QEMU dynamic instruction events <=60,000,000; no development board required.',
        'Both ROM and RAM must be strictly below 10,000,000 bytes in the final same image.',
        'ROI includes public language/text selection, native frontend, acoustic model, adapter, FARGAN, BWE, packet copy and effective-packet test.',
        'Language-neutral boot decoding, SHA checks, all G2P resource prewarming and voice registry initialization are separate. No sentence answers are cached.',
        '',
        'Best verified intermediate ELF: '+str(elf),
        'ELF SHA256: '+sha(elf),
        'ROM load bytes: '+str(d['memory']['rom_load_bytes']),
        'Reserved physical RAM bytes: '+str(d['memory']['reserved_physical_ram_bytes']),
        'Read-only assets in QEMU SRAM are conservatively counted toward BOTH ROM and RAM.',
        'Native raw-text smoke cases: '+', '.join(c['language']for c in cases),
        str(outputs)+' core output files exactly match their frozen per-language baselines. Full packet concatenations also match.',
        'All 25 normalization banks are in firmware; that is not completion of all 25 text adapters.',
        'Current frontend scope: ordinary NFC families plus native Han subset; full normalization remains pending.',
        'Timing passed: '+', '.join(passed),
        '',
    ]
    lines += [row['language']+': '+str(row['events'])+(' PASS'if row['within_60M']else' OVER LIMIT')for row in events]
    lines += [
        '',
        'Remaining family adapters: '+', '.join(remaining),
        'Remaining work: full numeric/identifier/abbreviation/Unicode policy, latency reduction, long-text planning, final same-image all25 memory and instruction validation.',
        'No training, new quantization, default-pointer change, GitHub push or external delivery was performed.',
        '',
        f'Recovery: use STATUS.json, reports/{tag}/RESULT.json, LATENCY_HOTSPOTS.json and build/{tag}/FROZEN_INPUTS.json.',
        'The frozen_inputs/sha256 content store preserves actual compiler dependencies and linked binary assets.',
        'Do not rerun an existing --tag: builders intentionally require a fresh directory. Use a fresh tag for any new experiment.',
        args.native_root+' owns src/assets/frontend_patched/dictionary_runtime and inherits immutable compiler/frontend/BWE sources through symlinks; preserve symlink targets.',
        'Build recipe: python3 build_combined.py --tag NEW_TAG --backend-tag compact_ring_v2 --arena-bytes 3960000 --multilang --profile --native-root '+str(ROOT/args.native_root)+' --prewarm-g2p --case-manifest '+args.case_manifest,
        'VLDM/LDM optimizations reduce architectural instruction events while retaining all arithmetic; do not infer equal hardware cycle savings.',
        'The original opt16 ELF was reconstructed byte-exactly before modifications; the published source mismatch was repaired only in this isolated workspace.',
        'Prepared-backend 25-language diagnostics are not native raw-text acceptance. Do not substitute those results for missing frontends.',
    ]
    (ROOT/'RECOVERY.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=False,languages_verified=count,timing_passed=passed,
        max_events=max(row['events']for row in events),memory=d['memory']),indent=2))

if __name__=='__main__':main()
