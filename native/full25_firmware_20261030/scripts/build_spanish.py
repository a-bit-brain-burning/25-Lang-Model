"""Repage the entire pinned Spanish lexicon without changing its records."""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import time
import zlib
from build_resources import ROOT,WORK,NATIVE,TOOLS,CC,sha,write,run,elf_memory
VENDOR=WORK/'new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24/frontend_bundle/frontend/vendor'


def pack_dictionary():
    sys.path.insert(0,str(VENDOR));import fe_codec as fe
    from fe_lexicon import Lexicon
    lex=Lexicon(ROOT/'resources/packages/es-ES.fep',None,VENDOR/'frontend/libs')
    reader=fe.Reader(ROOT/'resources/packages/es-ES.fep')
    words=sorted(lex.records)
    pages=[];keys=bytearray();raw_pages=[]
    rows_per_page=128
    for start in range(0,len(words),rows_per_page):
        group=words[start:start+rows_per_page];raw=bytearray()
        for word in group:
            key=word.encode();record=lex.records[word]
            assert len(key)<65536 and len(record)<65536
            raw.extend(struct.pack('<HH',len(key),len(record))+key+b'\0'+record)
        c=zlib.compressobj(9,wbits=-15);blob=c.compress(raw)+c.flush()
        pages.append(dict(first_key=group[0],key_offset=len(keys),blob=blob,
            raw_bytes=len(raw),rows=len(group),crc=zlib.crc32(raw)))
        keys.extend(group[0].encode()+b'\0');raw_pages.append(bytes(raw))
    key_offset=48+24*len(pages);payload=key_offset+len(keys)
    model=reader.read('model');model_offset=payload;cursor=model_offset+len(model)
    desc=bytearray()
    for page in pages:
        desc.extend(struct.pack('<6I',key_offset+page['key_offset'],cursor,len(page['blob']),
            page['raw_bytes'],page['rows'],page['crc']))
        cursor+=len(page['blob'])
    header=struct.pack('<8s10I',b'NVLEX1\0\0',1,len(pages),len(words),model_offset,
        len(model),key_offset,payload,0,cursor,0)
    blob=header+desc+keys+model+b''.join(p['blob']for p in pages)
    assert len(blob)==cursor
    # Independently re-read every serialized key and original residual record.
    restored={}
    for page in pages:
        raw=zlib.decompress(page['blob'],wbits=-15);position=0
        assert zlib.crc32(raw)==page['crc']
        for _ in range(page['rows']):
            kn,vn=struct.unpack_from('<HH',raw,position);position+=4
            word=raw[position:position+kn].decode();position+=kn+1
            restored[word]=raw[position:position+vn];position+=vn
        assert position==len(raw)
    assert restored==lex.records
    out=ROOT/'resources/spanish.nvl';out.write_bytes(blob)
    cases=[]
    chosen=sorted(set(words[::max(1,len(words)//80)]+words[-2:]+['hola','prueba','computadora','veintitrés','zzzztest']))
    for word in chosen:
        try:raw=lex.raw(word)
        except KeyError:continue
        cases.append(dict(word=word,expected=raw.hex(),known=word in lex.records))
    generated=ROOT/'generated'
    lines=['#ifndef SPANISH_CASES_H','#define SPANISH_CASES_H',
        '#define NVL_MAX_PAGE_BYTES '+str(max(p['raw_bytes']for p in pages)),
        'extern const unsigned char n25_spanish_nvl[];',f'#define NVL_PACK_BYTES {len(blob)}',
        'static const unsigned char spanish_default_record[]={'+','.join(str(v)for v in lex.policy['default_record'])+'};',
        'static const unsigned spanish_alphabet[]={'+','.join(str(ord(c))for c in lex.policy['alphabet'])+'};',
        'typedef struct{const char *word;unsigned length;const unsigned char *expected;} SpanishCase;',
        'static const SpanishCase spanish_cases[]={']
    for row in cases:
        raw=bytes.fromhex(row['expected']);literal=''.join('\\%03o'%b for b in raw)
        lines.append('{'+json.dumps(row['word'],ensure_ascii=False)+','+str(len(raw))+',(const unsigned char*)"'+literal+'"},')
    lines+=['};','#endif'];(generated/'spanish_cases.h').write_text('\n'.join(lines)+'\n')
    (generated/'spanish_nvl.S').write_text('.section .benchmark_assets.nvl,"a",%progbits\n.balign 16\n.global n25_spanish_nvl\nn25_spanish_nvl:\n.incbin "'+str(out)+'"\n.section .note.GNU-stack,""\n')
    write(ROOT/'resources/SPANISH_REPAGING.json',dict(source_package_sha256=sha(ROOT/'resources/packages/es-ES.fep'),
        nvl_sha256=sha(out),words=len(words),pages=len(pages),bytes=len(blob),
        max_page_output_bytes=max(p['raw_bytes']for p in pages),
        all_original_records_byte_identical=True,no_entries_removed=True,
        predictor_sha256=hashlib.sha256(model).hexdigest(),cases=cases,
        purpose='Avoid whole-dictionary LZMA/key scans for each request word; exact residual records and original OOV policy retained.'))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--tag',default='spanish_v1');a=p.parse_args()
    (ROOT/'generated').mkdir(exist_ok=True);pack_dictionary()
    out=ROOT/'build'/a.tag;out.mkdir(parents=True,exist_ok=False)
    flags=['-O3','-g','-ffunction-sections','-fdata-sections','-fstack-usage',
        '-fno-unwind-tables','-fno-asynchronous-unwind-tables',
        '-I'+str(NATIVE/'src'),'-I'+str(ROOT/'generated'),
        '-I'+str(WORK/'m4_frontend_4mb_20261022/support')]
    inputs=[NATIVE/'src'/n for n in ('startup.S','bench.c','syscalls.c','nvl.c','spanish_probe.c')]
    inputs += [NATIVE/'third_party/puff/puff.c',ROOT/'generated/spanish_nvl.S',
        WORK/'m4_frontend_4mb_20261022/support/predict_reader.c',
        WORK/'m4_frontend_lossy_20261023/open_predict.c']
    objects=[];commands=[]
    for i,source in enumerate(inputs):
        obj=out/(str(i)+'_'+source.stem+'.o');objects.append(obj)
        commands.append(run([CC,*flags,'-c',source,'-o',obj],out/(str(i)+'.log')))
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,-Map='+str(out/'firmware.map'),'-Wl,--defsym=ARENA_BYTES=65536',
        '-Wl,--defsym=STACK_BYTES=32768','-T',NATIVE/'src/m55.ld',*objects,
        '-Wl,--start-group','-lm','-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    write(out/'BUILD.json',dict(elf_sha256=sha(elf),commands=commands,
        input_sha256={str(p):sha(p)for p in inputs},memory=elf_memory(elf)))
    report=ROOT/'reports'/a.tag;report.mkdir(parents=True,exist_ok=False)
    command=[str(WORK/'local_qemu_20261030/bin/qemu-system-arm'),'-M','mps3-an547','-cpu','cortex-m55',
        '-accel','tcg,thread=single','-nographic','-serial','none','-monitor','none',
        '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    start=time.monotonic();result=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=120)
    output=result.stdout+result.stderr;(report/'qemu.log').write_text(output)
    passed=result.returncode==0 and 'N25_SPANISH_DICTIONARY_PASS'in output
    metrics={}
    for line in output.splitlines():
        if line.startswith('{'):
            row=json.loads(line)
            if row.get('kind')=='metric':metrics[row['name']]=row['value']
    write(report/'RESULT.json',dict(status='PASSED'if passed else'FAILED',elf_sha256=sha(elf),
        command=command,returncode=result.returncode,metrics=metrics,memory=elf_memory(elf),
        host_wall_seconds=time.monotonic()-start,complete_tts=False))
    print(json.dumps(dict(passed=passed,metrics=metrics),indent=2))
    if not passed:raise SystemExit(output)


if __name__=='__main__':main()
