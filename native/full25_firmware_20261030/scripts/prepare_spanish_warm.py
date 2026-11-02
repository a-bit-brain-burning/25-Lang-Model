"""Rebalance identical model bytes to make room for complete ES dictionary data."""
import hashlib
import json
import lzma
import re
import shutil
import zlib
from build_resources import ROOT,write,sha

def main():
    source=ROOT/'native_spanish';dest=ROOT/'native_spanish_warm';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched','dictionary_runtime'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    canonical=ROOT/'native_multilang/assets';text=(canonical/'model_index.c').read_text()
    start=text.index('const n25_tensor');end=text.index('const char')
    pattern=r'\{(\d+),(\d+),(\d+),\{(\d+),(\d+),(\d+)\},(\d+),(\d+),(\d+),(\d+)\}'
    rows=[list(map(int,row))for row in re.findall(pattern,text[start:end])]
    original=(canonical/'model.bin').read_bytes()
    plan=next(p for p in json.loads((ROOT/'ROM_BANK_CACHE_PLANS.json').read_text())if p['target']==1850000)
    selected=set(plan['chosen']);ram=bytearray();rom=bytearray();new=[];identities=[]
    for i,t in enumerate(rows):
        offset,scale,e,a,b,c,rank,dtype,axis,layout=t
        n=((a+7)//8)*((e//a+15)//16)*128 if layout else e*(1 if dtype==0 else 2 if dtype==1 else 4)
        codes=original[offset:offset+n];scales=original[scale:scale+[a,b,c][axis]*4]if dtype==0 else b''
        bank=rom if i in selected else ram;bank.extend(bytes(-len(bank)%16));off=len(bank);bank.extend(codes);sc=0
        if scales:bank.extend(bytes(-len(bank)%16));sc=len(bank);bank.extend(scales)
        assert bank[off:off+len(codes)]==codes and (not scales or bank[sc:sc+len(scales)]==scales)
        new.append([(off|0x80000000)if i in selected else off,sc,*t[2:]])
        identities.append(dict(tensor=i,code_sha256=hashlib.sha256(codes).hexdigest(),scale_sha256=hashlib.sha256(scales).hexdigest(),rom=i in selected))
    assert len(rows)==221 and len(ram)==plan['ram_model'] and len(rom)==plan['rom_raw']
    code=['const n25_tensor n25_tensors[]={']
    code += ['{%u,%u,%u,{%u,%u,%u},%u,%u,%u,%u},'%tuple(t)for t in new];code.append('};\n')
    (dest/'assets/model_index.c').write_text(text[:start]+'\n'.join(code)+text[end:])
    packed=lzma.compress(ram,format=lzma.FORMAT_RAW,filters=[dict(id=lzma.FILTER_LZMA2,dict_size=262144,preset=9)])
    assert len(packed)==plan['packed'] and lzma.decompress(packed,format=lzma.FORMAT_RAW,filters=[dict(id=lzma.FILTER_LZMA2,dict_size=262144)])==ram
    for name,data in [('model.banked_ram.bin',ram),('model.banked_rom.bin',rom),('model.lzma2',packed)]:
        (dest/'assets'/name).write_bytes(data)
    p=dest/'assets/model_index.h';s=p.read_text();s=re.sub(r'#define MODEL_BYTES \d+',f'#define MODEL_BYTES {len(ram)}',s)
    s=re.sub(r'#define MODEL_ROM_BANK_BYTES \d+u',f'#define MODEL_ROM_BANK_BYTES {len(rom)}u',s);p.write_text(s)
    p=dest/'assets/storage_identity.h';s=p.read_text()
    for name,data in [('storage_model_sha256',ram),('storage_rom_bank_sha256',rom)]:
        s=re.sub(r'static const unsigned char '+name+r'\[32\]=\{[^}]+\};',
            'static const unsigned char '+name+'[32]={'+','.join(map(str,hashlib.sha256(data).digest()))+'};',s)
    s=re.sub(r'#define STORAGE_MODEL_PACKED_BYTES \d+u',f'#define STORAGE_MODEL_PACKED_BYTES {len(packed)}u',s);p.write_text(s)
    (dest/'assets/weights.S').write_text(' .section .model_weights,"a"\n.balign 16\n.global storage_model_payload\nstorage_model_payload:\n.incbin "'+str(dest/'assets/model.lzma2')+'"\n.balign 16\n.global storage_tensor_rom_bank\nstorage_tensor_rom_bank:\n.incbin "'+str(dest/'assets/model.banked_rom.bin')+'"\n.section .note.GNU-stack,""\n')
    p=dest/'src/storage.c';s=p.read_text();s,n=re.subn(r'\.packed_bytes=\d+,\s*\.raw_bytes=MODEL_BYTES,\.crc32=\d+u',
        '.packed_bytes='+str(len(packed))+',\n   .raw_bytes=MODEL_BYTES,.crc32='+str(zlib.crc32(ram))+'u',s);assert n==1;p.write_text(s)
    p=dest/'src/persistent_runtime.inc';s=p.read_text();s='int fep_lexicon_prewarm_es(void);\n'+s
    s=s.replace('storage_initialize(start);fast_tables_prepare();','storage_initialize(start);\n if(fep_lexicon_prewarm_es())fail("SPANISH_DICTIONARY_PREWARM");\n fast_tables_prepare();');p.write_text(s)
    write(ROOT/'SPANISH_WARM_BANK_IDENTITY.json',dict(tensors=identities,canonical_model_sha256=sha(canonical/'model.bin'),
        ram_model_bytes=len(ram),rom_bank_bytes=len(rom),packed_model_bytes=len(packed),
        original_ram_model_bytes=2993672,model_RAM_released_bytes=2993672-len(ram),
        added_model_ROM_bytes=len(rom)+len(packed)-4176917,
        original_tensor_codes_scales_and_layouts_preserved=True,training_updates=0))

if __name__=='__main__':main()
