"""Store exact FP32 tables as compressed byte planes of modular bit deltas."""
import hashlib
import re
import shutil
import struct
import zlib
from build_resources import ROOT,write

def main():
    source=ROOT/'native_latvian';dest=ROOT/'native_lut_delta';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    original=(ROOT/'snapshots/before_lut_compression/fast_math_tables.h').read_text()
    rows=[];code=['#ifndef N25_COMPRESSED_FAST_TABLES','#define N25_COMPRESSED_FAST_TABLES',
        '#include "../third_party/puff/puff.h"']
    for name,n,body in re.findall(r'static const float (\w+)\[(\d+)\]=\{(.*?)\};',original,re.S):
        values=[struct.unpack('<I',struct.pack('<f',float.fromhex(v.strip().removesuffix('f'))))[0]
            for v in body.split(',')if v.strip()]
        assert len(values)==int(n)
        raw=struct.pack('<'+str(len(values))+'I',*values);delta=[];last=0
        for value in values:delta.append((value-last)&0xffffffff);last=value
        words=struct.pack('<'+str(len(delta))+'I',*delta);planes=b''.join(words[i::4]for i in range(4))
        c=zlib.compressobj(9,wbits=-15);packed=c.compress(planes)+c.flush()
        decoded=zlib.decompress(packed,wbits=-15);restored=bytearray();last=0
        for i in range(len(values)):
            difference=sum(decoded[k*len(values)+i]<<(8*k)for k in range(4))
            last=(last+difference)&0xffffffff;restored+=struct.pack('<I',last)
        assert restored==raw
        digest=hashlib.sha256(raw).digest()
        code += [f'static float {name}[{n}] __attribute__((section(".math_luts"),aligned(16)));',
            f'static const unsigned char {name}_packed[{len(packed)}] __attribute__((section(".benchmark_assets.fast_luts")))={{'+','.join(map(str,packed))+'};',
            f'static const unsigned char {name}_sha256[32]={{'+','.join(map(str,digest))+'};']
        rows.append(dict(name=name,elements=len(values),raw_bytes=len(raw),packed_bytes=len(packed),sha256=digest.hex()))
    code += ['''static void fast_restore_delta(float*out,unsigned count,const unsigned char*packed,unsigned packed_bytes,const unsigned char*expected){
 size_t mark=bench_arena_mark();unsigned char*planes=bench_alloc((size_t)count*4,16);
 if(!planes){bench_puts("FAST_TABLE_ALLOC_FAILED\\n");bench_exit(76);}
 unsigned long input=packed_bytes,output=count*4;
 if(puff(planes,&output,packed,&input)||input!=packed_bytes||output!=count*4){bench_puts("FAST_TABLE_DECODE_FAILED\\n");bench_exit(76);}
 uint32_t bits=0;unsigned char*destination=(unsigned char*)out;
 for(unsigned i=0;i<count;i++){
  uint32_t delta=(uint32_t)planes[i]|(uint32_t)planes[count+i]<<8|(uint32_t)planes[2*count+i]<<16|(uint32_t)planes[3*count+i]<<24;
  bits+=delta;
  destination[4*i]=(unsigned char)bits;destination[4*i+1]=(unsigned char)(bits>>8);
  destination[4*i+2]=(unsigned char)(bits>>16);destination[4*i+3]=(unsigned char)(bits>>24);
 }
 unsigned char digest[32];storage_sha256(destination,(size_t)count*4,digest);
 if(memcmp(digest,expected,32)){bench_puts("FAST_TABLE_SHA_FAILED\\n");bench_exit(77);}
 if(bench_arena_rewind(mark)){bench_puts("FAST_TABLE_REWIND_FAILED\\n");bench_exit(76);}
}
static void fast_tables_prepare(void){''']
    for row in rows:
        name=row['name'];code.append(f'fast_restore_delta({name},{row["elements"]},{name}_packed,sizeof({name}_packed),{name}_sha256);')
    code+=['bench_metric("fast_tables_boot_SHA256_verified",3);','}','#endif']
    (dest/'src/fast_math_tables.h').write_text('\n'.join(code)+'\n')
    write(ROOT/'LUT_DELTA_IDENTITY.json',dict(tables=rows,raw_bytes=sum(x['raw_bytes']for x in rows),
        packed_bytes=sum(x['packed_bytes']for x in rows),previous_packed_bytes=48200,
        exact_original_FP32_bytes=True,all_tables_boot_SHA256_required=True,request_lookup_arithmetic_unchanged=True))

if __name__=='__main__':main()
