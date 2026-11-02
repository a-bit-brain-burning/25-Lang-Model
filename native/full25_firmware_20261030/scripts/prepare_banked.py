"""Keep selected exact I8 matrices in ROM to make room for G2P prewarming."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import struct
import zlib
from build_resources import ROOT,WORK,sha,write


def deflate(raw):
    c=zlib.compressobj(9,wbits=-15);return c.compress(raw)+c.flush()


def main():
    source=ROOT/'native_multilang';dest=ROOT/'native_banked';dest.mkdir(exist_ok=False)
    for name in('src','assets'):shutil.copytree(source/name,dest/name)
    for name in('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to(source/name,target_is_directory=True)
    for name in('build.py','build_native.py','build_o3_resident.py'):shutil.copy2(source/name,dest/name)
    p=dest/'assets/model_index.c';text=p.read_text();begin=text.index('const n25_tensor');end=text.index('const char')
    pattern=r'\{(\d+),(\d+),(\d+),\{(\d+),(\d+),(\d+)\},(\d+),(\d+),(\d+),(\d+)\}'
    rows=[list(map(int,row))for row in re.findall(pattern,text[begin:end])]
    names=re.findall('"([^"]+)"',text[end:]);assert len(rows)==len(names)==221
    original=(source/'assets/model.bin').read_bytes()
    def data(t):
        offset,scale,e,a,b,c,rank,dtype,axis,packed=t
        n=((a+7)//8)*((e//a+15)//16)*128 if packed else e*(1 if dtype==0 else 2 if dtype==1 else 4)
        assert offset+n<=len(original)
        codes=original[offset:offset+n]
        scales=original[scale:scale+[a,b,c][axis]*4]if dtype==0 else b''
        return codes,scales
    candidates=[]
    for i,t in enumerate(rows):
        if t[7]!=0 or t[8]!=0 or t[6]<2 or t[3]<16 or t[2]//t[3]<16:continue
        a,b=data(t);raw=a+b;candidates.append((1-len(deflate(raw))/len(raw),i,len(raw)))
    candidates.sort();chosen=set();size=0
    for _,i,n in candidates:
        chosen.add(i);size+=n
        if size>=1500000:break
    ram=bytearray();rom=bytearray();new=[]
    for i,t in enumerate(rows):
        codes,scales=data(t);bank=rom if i in chosen else ram
        bank.extend(bytes((-len(bank))%16));offset=len(bank);bank.extend(codes);scale=0
        if scales:bank.extend(bytes((-len(bank))%16));scale=len(bank);bank.extend(scales)
        updated=[(offset|0x80000000)if i in chosen else offset,scale,*t[2:]];new.append(updated)
        assert bank[offset:offset+len(codes)]==codes
        assert not scales or bank[scale:scale+len(scales)]==scales
    line=['const n25_tensor n25_tensors[]={']
    for r in new:line.append('{%u,%u,%u,{%u,%u,%u},%u,%u,%u,%u},'%tuple(r))
    line.append('};\n');p.write_text(text[:begin]+'\n'.join(line)+text[end:])
    blocks=bytearray();index=bytearray(b'DF16'+struct.pack('<3I',16384,(len(ram)+16383)//16384,len(ram)))
    for at in range(0,len(ram),16384):
        raw=ram[at:at+16384];packed=deflate(raw);flag=0
        if len(packed)>=len(raw):packed=raw;flag=0x80000000
        index+=struct.pack('<3I',len(blocks),len(packed)|flag,len(raw));blocks+=packed
    for name,value in [('model.banked_ram.bin',ram),('model.banked_rom.bin',rom),
        ('model.banked_blocks.bin',blocks),('model.banked_index.bin',index)]:
        (dest/'assets'/name).write_bytes(value)
    p=dest/'assets/model_index.h';s=p.read_text();s=re.sub(r'#define MODEL_BYTES \d+',f'#define MODEL_BYTES {len(ram)}',s)
    s+=f'\n#define MODEL_ROM_BANK_BYTES {len(rom)}u\n';p.write_text(s)
    p=dest/'assets/storage_identity.h';s=p.read_text()
    for name,n in [('STORAGE_MODEL_BLOCKS',(len(ram)+16383)//16384),('STORAGE_MODEL_PACKED_BYTES',len(blocks)),('STORAGE_MODEL_INDEX_BYTES',len(index))]:
        s=re.sub(r'#define '+name+r' \d+u',f'#define {name} {n}u',s)
    s=re.sub(r'static const unsigned char storage_model_sha256\[32\]=\{[^}]+\};',
        'static const unsigned char storage_model_sha256[32]={'+','.join(str(b)for b in hashlib.sha256(ram).digest())+'};',s)
    s+='\nstatic const unsigned char storage_rom_bank_sha256[32]={'+','.join(str(b)for b in hashlib.sha256(rom).digest())+'};\n';p.write_text(s)
    (dest/'assets/weights.S').write_text(''' .section .model_weights,"a"
.balign 16
.global storage_model_payload
storage_model_payload:
.incbin "'''+str(dest/'assets/model.banked_blocks.bin')+'''"
.balign 16
.global storage_model_index
storage_model_index:
.incbin "'''+str(dest/'assets/model.banked_index.bin')+'''"
.balign 16
.global storage_tensor_rom_bank
storage_tensor_rom_bank:
.incbin "'''+str(dest/'assets/model.banked_rom.bin')+'''"
.section .note.GNU-stack,""
''')
    p=dest/'src/tensor.h';s=p.read_text();s=s.replace('float n25_value(','''extern const unsigned char storage_tensor_rom_bank[];
static inline const unsigned char*n25_tensor_data(const unsigned char*blob,const n25_tensor*t){
 return t->offset&0x80000000u?storage_tensor_rom_bank+(t->offset&0x7fffffffu):blob+t->offset;
}
static inline const unsigned char*n25_tensor_scales(const unsigned char*blob,const n25_tensor*t){
 return (t->offset&0x80000000u?storage_tensor_rom_bank:blob)+t->scale_offset;
}
float n25_value(''');p.write_text(s)
    for name in('tensor.c','quality.c','tensor_views.inc','fargan_stream.inc'):
        p=dest/'src'/name;s=p.read_text()
        for var in('t','w'):
            s=s.replace('n25_blob+'+var+'->offset','n25_tensor_data(n25_blob,'+var+')')
            s=s.replace('n25_blob+'+var+'->scale_offset','n25_tensor_scales(n25_blob,'+var+')')
            s=s.replace('blob+'+var+'->offset','n25_tensor_data(blob,'+var+')')
            s=s.replace('blob+'+var+'->scale_offset','n25_tensor_scales(blob,'+var+')')
        p.write_text(s)
    p=dest/'src/fargan_stream.inc';s=p.read_text()
    old='if(t->axis||t->rank<2||t->scale_offset%4||t->offset>MODEL_BYTES||fg_storage_bytes(t)>MODEL_BYTES-t->offset||t->scale_offset>MODEL_BYTES||t->dims[0]*4>MODEL_BYTES-t->scale_offset)fail("FARGAN_I8_RAM_LAYOUT");'
    assert old in s
    s=s.replace(old,'''unsigned offset=t->offset&0x7fffffffu,capacity=t->offset&0x80000000u?MODEL_ROM_BANK_BYTES:MODEL_BYTES;
  if(t->axis||t->rank<2||t->scale_offset%4||offset>capacity||fg_storage_bytes(t)>capacity-offset||t->scale_offset>capacity||t->dims[0]*4>capacity-t->scale_offset)fail("FARGAN_I8_BANK_LAYOUT");''')
    s=s.replace('bench_metric("FARGAN_model_RAM_zero_copy",1);','bench_metric("FARGAN_model_banked_zero_copy",1);')
    p.write_text(s)
    p=dest/'src/storage.c';s=p.read_text();point='n25_blob=model;'
    assert point in s;s=s.replace(point,'''unsigned char rom_hash[32];storage_sha256(storage_tensor_rom_bank,MODEL_ROM_BANK_BYTES,rom_hash);
 if(memcmp(rom_hash,storage_rom_bank_sha256,32))fail("MODEL_ROM_BANK_SHA256");
 bench_metric("storage_exact_weight_ROM_bank_bytes",MODEL_ROM_BANK_BYTES);
 '''+point);p.write_text(s)
    # The native feature result is temporary; copy it into the request input
    # and rewind before neural inference, instead of keeping a second array.
    p=dest/'src/multilang_frontend.c';s=p.read_text()
    s=s.replace('static N25FourFeatures four_features __attribute__((section(".frontend_workspace"),aligned(16)));','')
    s=s.replace('int rc=n25_four_frontend(language,text,&four_features);',
        'N25FourFeatures*four_features=bench_alloc(sizeof(*four_features),16);\n if(!four_features)return -95;\n int rc=n25_four_frontend(language,text,four_features);')
    s=s.replace('four_features.','four_features->')
    # Warm file reads no longer require a per-request LZMA workspace.
    old='size_t mark=bench_arena_mark();void*work=bench_alloc(582144,16);\n if(!work||n25_romfs_set_workspace(work,582144))return -98;'
    assert old in s;s=s.replace(old,'size_t mark=bench_arena_mark();')
    p.write_text(s)
    write(ROOT/'BANKED_MODEL_IDENTITY.json',dict(status='PREPARED_NOT_YET_TARGET_VERIFIED',
        canonical_model_sha256=hashlib.sha256(original).hexdigest(),
        ram_model_bytes=len(ram),ram_model_sha256=hashlib.sha256(ram).hexdigest(),
        rom_bank_bytes=len(rom),rom_bank_sha256=hashlib.sha256(rom).hexdigest(),
        compressed_ram_model_bytes=len(blocks),index_bytes=len(index),
        rom_delta_bytes=len(rom)+len(blocks)+len(index)-(4141408+3328),
        all_221_tensor_code_and_scale_payloads_byte_identical=True,
        selected_tensors=[names[i]for i in sorted(chosen)],training_updates=0,arithmetic_changes=False))
    print('Prepared',len(ram),'RAM model bytes and',len(rom),'exact ROM weight bytes')


if __name__=='__main__':main()
