"""Losslessly store the exact float lookup tables as boot-time DEFLATE assets."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import struct
import zlib
ROOT=Path(__file__).resolve().parent


def main():
    src=ROOT/'native/src';path=src/'fast_math_tables.h'
    before=ROOT/'snapshots/before_lut_compression';before.mkdir(parents=True,exist_ok=False)
    for name in('fast_math_tables.h','persistent_runtime.inc','m55.ld'):shutil.copy2(src/name,before/name)
    text=path.read_text();code=['#ifndef N25_COMPRESSED_FAST_TABLES','#define N25_COMPRESSED_FAST_TABLES',
        '#include "../third_party/puff/puff.h"'];rows=[]
    for name,n,body in re.findall(r'static const float (\w+)\[(\d+)\]=\{(.*?)\};',text,re.S):
        values=[v.strip().removesuffix('f')for v in body.split(',')if v.strip()]
        assert len(values)==int(n)
        raw=b''.join(struct.pack('<f',float.fromhex(value))for value in values)
        c=zlib.compressobj(9,wbits=-15);packed=c.compress(raw)+c.flush()
        assert zlib.decompress(packed,wbits=-15)==raw
        code.append(f'static float {name}[{n}] __attribute__((section(".math_luts"),aligned(16)));')
        code.append(f'static const unsigned char {name}_packed[{len(packed)}] __attribute__((section(".benchmark_assets.fast_luts")))={{'+','.join(str(v)for v in packed)+'};')
        code.append(f'static const unsigned char {name}_sha256[32]={{'+','.join(str(v)for v in hashlib.sha256(raw).digest())+'};')
        rows.append(dict(name=name,elements=int(n),raw_bytes=len(raw),packed_bytes=len(packed),sha256=hashlib.sha256(raw).hexdigest()))
    assert len(rows)==3
    code.extend(['static void fast_tables_prepare(void){',' unsigned char digest[32];unsigned long packed,raw;int rc;'])
    for row in rows:
        name=row['name']
        code.extend([f' packed=sizeof({name}_packed);raw=sizeof({name});',
            f' rc=puff((unsigned char*){name},&raw,{name}_packed,&packed);',
            f' if(rc||raw!=sizeof({name})||packed!=sizeof({name}_packed)){{bench_puts("FAST_TABLE_DECODE_FAILED\\n");bench_exit(76);}}',
            f' storage_sha256((const unsigned char*){name},sizeof({name}),digest);',
            f' if(memcmp(digest,{name}_sha256,32)){{bench_puts("FAST_TABLE_SHA_FAILED\\n");bench_exit(77);}}'])
    code.extend([' bench_metric("fast_tables_boot_SHA256_verified",3);','}','#endif'])
    path.write_text('\n'.join(code)+'\n')
    p=src/'persistent_runtime.inc';s=p.read_text();assert 'storage_initialize(start);'in s
    s=s.replace('storage_initialize(start);','storage_initialize(start);fast_tables_prepare();');p.write_text(s)
    p=src/'m55.ld';s=p.read_text();s=s.replace('  /DISCARD/',
        '  .math_luts (NOLOAD) : ALIGN(16) { *(.math_luts*) . = ALIGN(16); __math_luts_end = .; } > SRAM\n  /DISCARD/')
    s+='\nASSERT(__math_luts_end <= ORIGIN(SRAM) + LENGTH(SRAM), "Math tables exceed SRAM")\n';p.write_text(s)
    (ROOT/'LUT_COMPRESSION.json').write_text(json.dumps(dict(tables=rows,
        exact_float_bytes_preserved=True,boot_SHA256_verification=True,
        raw_bytes=sum(r['raw_bytes']for r in rows),packed_bytes=sum(r['packed_bytes']for r in rows)),indent=2)+'\n')


if __name__=='__main__':main()
