"""Lossless boot-only model storage using the already linked bounded decoder."""
from pathlib import Path
import json
import lzma
import shutil
import zlib
from build_resources import ROOT,sha,write

def main():
    source=ROOT/'native_finnic';dest=ROOT/'native_lzma';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    raw=(dest/'assets/model.banked_ram.bin').read_bytes()
    packed=lzma.compress(raw,format=lzma.FORMAT_RAW,
        filters=[dict(id=lzma.FILTER_LZMA2,dict_size=262144,preset=9)])
    assert lzma.decompress(packed,format=lzma.FORMAT_RAW,
        filters=[dict(id=lzma.FILTER_LZMA2,dict_size=262144)])==raw
    (dest/'assets/model.lzma2').write_bytes(packed)
    p=dest/'assets/weights.S';s=p.read_text().replace(str(source),str(dest))
    a=s.index('.incbin "');b=s.index('.balign 16\n.global storage_tensor_rom_bank',a)
    s=s[:a]+'.incbin "'+str(dest/'assets/model.lzma2')+'"\n'+s[b:];p.write_text(s)
    for p in (dest/'assets').glob('*.S'):
        p.write_text(p.read_text().replace(str(source),str(dest)))
    p=dest/'src/storage.c';s=p.read_text().replace('#include "storage.h"','#include "storage.h"\n#include "fep_runtime.h"')
    a=s.index(' start=bench_now();const unsigned char*index=storage_model_index;')
    b=s.index(' start=bench_now();unsigned char hash[32];',a)
    s=s[:a]+''' start=bench_now();
 unsigned char*model=bench_alloc(MODEL_BYTES,16);if(!model)fail("MODEL_DECODE_ALLOC");
 size_t mark=bench_arena_mark(),peak=0;void*workspace=bench_alloc(320000,16);
 if(!workspace)fail("MODEL_LZMA_WORKSPACE_ALLOC");
 FepSection section={.codec=1,.packed=storage_model_payload,.packed_bytes='''+str(len(packed))+''',
   .raw_bytes=MODEL_BYTES,.crc32='''+str(zlib.crc32(raw))+'''u,.window=262144};
 if(fep_decode(&section,model,MODEL_BYTES,workspace,320000,&peak))fail("MODEL_LZMA_DECODE");
 if(bench_arena_rewind(mark))fail("MODEL_LZMA_REWIND");
 bench_metric("storage_model_decode_host_ticks",bench_now()-start);
 bench_metric("storage_model_LZMA2_workspace_peak",peak);
 bench_metric("storage_model_LZMA2_bytes",section.packed_bytes);
'''+s[b:];p.write_text(s)
    write(ROOT/'MODEL_LZMA_IDENTITY.json',dict(original_raw_sha256=sha(dest/'assets/model.banked_ram.bin'),
        packed_sha256=sha(dest/'assets/model.lzma2'),raw_bytes=len(raw),packed_bytes=len(packed),
        source_deflate_bytes=(source/'assets/model.banked_blocks.bin').stat().st_size,
        all_tensor_codes_and_scales_unchanged=True,boot_only=True,request_decode_calls=0))

if __name__=='__main__':main()
