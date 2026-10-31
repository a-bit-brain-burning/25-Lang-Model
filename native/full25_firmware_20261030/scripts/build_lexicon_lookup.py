"""Validate original FEP key/ID/pool queries on M55 for seven full lexicons."""
import json
import re
import shutil
import subprocess
import sys
import time
from build_resources import ROOT,WORK,NATIVE,CC,run,write,sha,elf_memory,generate,build_lzma
from build_espeak import VENDOR

def main():
    sys.path.insert(0,str(VENDOR));import fe_codec as fe
    from fe_lexicon import Lexicon
    directory=ROOT/'dictionary_runtime';directory.mkdir(exist_ok=False)
    # Reuse exact, previously validated key/ID/pool primitives. The old reader's
    # host liblzma/zlib entry point is replaced by our existing M55 FEP decoder.
    original=WORK/'m4_frontend_4mb_20261022/reader.c';s=original.read_text()
    helpers=s[s.index('static uint32_t u32'):s.index('\nint fep_read')]
    primitives=s[s.index('static int utf8'):]
    (directory/'fep_query_primitives.c').write_text(s[:s.index('#include')]+'''#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include "reader.h"
'''+helpers+primitives)
    shutil.copy2(original.with_suffix('.h'),directory/'reader.h')
    packs=json.loads((ROOT/'resources/SECTIONS.json').read_text())['packages'];pack_ids={p['name']:i for i,p in enumerate(packs)}
    locales='en-CMU es-ES fr-FR it-IT lt-LT pt-PT ro-RO'.split()
    declarations=['#include "fep_dictionary.h"'];cases=[];descriptors=[];max_raw=0
    for i,locale in enumerate(locales):
        p=ROOT/'resources/packages'/(locale+'.fep');docs,pack=fe.raw_tables(p,VENDOR/'frontend/libs/libreader.so')
        records=docs[locale]['entries'];layout=fe.Codec([]).decode(pack.read('metadata'));alphabet=[]
        if layout['mode']==2:
            raw=pack.read('alphabet');at=0;cp=-1
            while at<len(raw):v,at=fe.get_varint(raw,at);cp+=v+1;alphabet.append(cp)
        first=layout['first_keys'];assert len(layout['columns'])==1
        declarations.append('static const char*const first_'+str(i)+'[]={'+','.join(json.dumps(x,ensure_ascii=False)for x in first)+'};')
        declarations.append('static const uint32_t alphabet_'+str(i)+'[]={'+','.join(map(str,alphabet or [0]))+'};')
        descriptors.append('{'+str(pack_ids[locale+'.fep'])+','+str(len(first))+',first_'+str(i)+',alphabet_'+str(i)+','+str(len(alphabet))+'},')
        words=sorted(records);chosen=set(words[::max(1,len(words)//5)]+words[-1:]+['zzzztest','zzzzzzzzzzzz'])
        language=locale.split('-')[0];source=WORK/'m55_fargan_stream_20261030/language25_20261030/inputs'/language/'job.json'
        for word in re.findall(r'[^\W\d_]+',json.loads(source.read_text())['text'].lower()):chosen.add(word)
        for word in sorted(chosen):
            value=records.get(word);cases.append(dict(lexicon=i,locale=locale,word=word,
                present=value is not None,expected_hex=value.hex()if value is not None else''))
        for name in pack.sections:
            if name.startswith(('keys/','pool/','ids/')):max_raw=max(max_raw,pack.sections[name][3])
    declarations+=['const FepLexiconDesc fep_lexicons[]={',*descriptors,'};']
    (directory/'lexicons_meta.c').write_text('\n'.join(declarations)+'\n')
    header=['#ifndef FEP_DICTIONARY_CASES_H','#define FEP_DICTIONARY_CASES_H',
        'typedef struct{unsigned lexicon;const char*word;int present;unsigned length;const unsigned char*expected;}LexiconCase;',
        'static const LexiconCase lexicon_cases[]={']
    for row in cases:
        b=bytes.fromhex(row['expected_hex']);literal=''.join('\\%03o'%x for x in b)
        header.append('{'+str(row['lexicon'])+','+json.dumps(row['word'],ensure_ascii=False)+','+str(int(row['present']))+','+str(len(b))+',(const unsigned char*)"'+literal+'"},')
    header+=['};','#endif'];(directory/'lexicon_cases.h').write_text('\n'.join(header)+'\n')
    write(directory/'IDENTITY.json',dict(locales=locales,cases=cases,max_section_bytes=max_raw,
        workspace_bytes=320000+max_raw,original_primitives_sha256=sha(original),
        complete_lexicon_packages=True,scope='Exact residual-record lookup only; feature adapters and timing acceptance are separate'))
    # The actual reader implementation is kept alongside the generic probe,
    # so it can later be linked into an integrated language feature adapter.
    for name in ('fep_dictionary.c','fep_dictionary.h','dictionary_probe.c'):
        shutil.copy2(ROOT/'dictionary_source'/name,directory/name)
    generate();out=ROOT/'build/lexicon_lookup_v1';out.mkdir(exist_ok=False)
    flags=['-O3','-g','-std=c11','-ffreestanding','-fno-builtin','-fno-fast-math','-ffp-contract=off',
        '-ffunction-sections','-fdata-sections','-fstack-usage','-fno-unwind-tables','-fno-asynchronous-unwind-tables',
        '-I'+str(directory),'-I'+str(NATIVE/'src'),'-I'+str(ROOT/'generated')]
    inputs=[NATIVE/'src'/name for name in ('startup.S','bench.c','syscalls.c','fep_runtime.c')]
    inputs += [directory/name for name in ('fep_query_primitives.c','fep_dictionary.c','dictionary_probe.c','lexicons_meta.c')]
    inputs += [ROOT/'generated/fep_assets.c',ROOT/'generated/fep_assets.S']
    flags += ['-I'+str(WORK/'m4_frontend_rom_accounting_20261023/sources/xz-5.4.5/src/liblzma/api')]
    commands=[];objects=[]
    for i,p in enumerate(inputs):
        obj=out/(str(i)+'_'+p.stem+'.o');objects.append(obj)
        commands.append(run([CC,*flags,'-c',p,'-o',obj],out/(str(i)+'.log')))
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,--defsym=ARENA_BYTES='+str(320000+max_raw),'-Wl,--defsym=STACK_BYTES=32768',
        '-T',NATIVE/'src/m55.ld',*objects,'-Wl,--start-group',build_lzma(),'-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    write(out/'BUILD.json',dict(elf_sha256=sha(elf),commands=commands,memory=elf_memory(elf),
        input_sha256={str(p):sha(p)for p in inputs}))
    report=ROOT/'reports/lexicon_lookup_v1';report.mkdir(exist_ok=False)
    command=[str(WORK/'local_qemu_20261030/bin/qemu-system-arm'),'-M','mps3-an547','-cpu','cortex-m55',
        '-accel','tcg,thread=single','-nographic','-serial','none','-monitor','none',
        '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    start=time.monotonic();result=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=240)
    output=result.stdout+result.stderr;(report/'qemu.log').write_text(output)
    passed=result.returncode==0 and'N25_SEVEN_LEXICON_LOOKUP_PASS'in output
    write(report/'RESULT.json',dict(passed=passed,elf_sha256=sha(elf),command=command,
        returncode=result.returncode,case_count=len(cases),lexicon_count=7,workspace_bytes=320000+max_raw,
        memory=elf_memory(elf),host_wall_seconds=time.monotonic()-start,full_native_text_TTS=False))
    print(output);assert passed

if __name__=='__main__':main()
