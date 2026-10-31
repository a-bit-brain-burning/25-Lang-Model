"""Make the pinned G2P filesystem resident; borrow its unchanged file bytes."""
from pathlib import Path
import shutil
from build_resources import ROOT,WORK


def main():
    native=ROOT/'native_banked';src=native/'src'
    p=src/'romfs.h';s=p.read_text();s+='\nint n25_romfs_prewarm_files(void);\nconst void*n25_romfs_direct_file(const char*,size_t*);\n';p.write_text(s)
    p=src/'romfs.c';s=p.read_text()
    s=s.replace('#include "bench.h"','#include "bench.h"\n#include "storage.h"\n#include "romfs_capacity.h"')
    point='static size_t cached_bytes;'
    s=s.replace(point,point+'''
static unsigned char warm_image[N25_ROMFS_WARM_CAPACITY] __attribute__((section(".frontend_workspace"),aligned(16)));
static const unsigned char*warm_files[N25_ROMFS_FILE_COUNT];
static unsigned warm_ready;
extern const unsigned char n25_rom_sha256[N25_ROMFS_FILE_COUNT][32];
''')
    s=s.replace('if(!p)return;\n    HeapBlock *b=', '''if(!p)return;
    if((uintptr_t)p>=(uintptr_t)warm_image&&(uintptr_t)p<(uintptr_t)warm_image+sizeof(warm_image))return;
    for(unsigned i=0;i<n25_rom_file_count;i++)if(n25_rom_files[i].direct==p)return;
    HeapBlock *b=''')
    point='static int directory_exists(const char *name){'
    s=s.replace(point,'''const void*n25_romfs_direct_file(const char*path,size_t*bytes){
    const N25RomFile*f=file_by_name(relative_path(path));
    if(!warm_ready||!f)return NULL;
    if(bytes)*bytes=f->bytes;
    return warm_files[f-n25_rom_files];
}
'''+point)
    point='if(n>f->bytes-h->pos)n=f->bytes-h->pos;'
    assert point in s;s=s.replace(point,point+'''
    if(warm_ready){memcpy(out,warm_files[f-n25_rom_files]+h->pos,n);h->pos+=n;return n;}
''')
    point='size_t __wrap_fread('
    assert point in s;s=s.replace(point,'''int n25_romfs_prewarm_files(void){
    if(warm_ready)return 0;
    if(n25_rom_file_count!=N25_ROMFS_FILE_COUNT)return -1;
    size_t used=0;
    for(unsigned i=0;i<n25_rom_file_count;i++){
        const N25RomFile*f=n25_rom_files+i;
        if(f->direct)warm_files[i]=f->direct;
        else{
            used=(used+15)&~(size_t)15;
            if(used>sizeof(warm_image)||f->bytes>sizeof(warm_image)-used)return -1;
            warm_files[i]=warm_image+used;RomHandle handle={f,0,1};
            if(read_bytes(warm_image+used,f->bytes,&handle)!=f->bytes)return -1;
            used+=f->bytes;
        }
        unsigned char hash[32];storage_sha256(warm_files[i],f->bytes,hash);
        if(memcmp(hash,n25_rom_sha256[i],32))return -2;
    }
    warm_ready=1;
    bench_metric("all_language_G2P_files_prewarmed_SHA256",n25_rom_file_count);
    bench_metric("all_language_G2P_file_image_RAM_bytes",used);
    return 0;
}
'''+point);p.write_text(s)
    p=src/'combined_probe.c';s=p.read_text();point='if(!workspace||n25_romfs_set_workspace(workspace,320000+262144))return 22;'
    assert point in s;s=s.replace(point,point+'\n    if(n25_romfs_prewarm_files())return 25;');p.write_text(s)
    p=src/'fargan_stream.inc';s=p.read_text().replace('"FARGAN_borrowed_model_RAM_bytes"','"FARGAN_borrowed_model_banks_bytes"');p.write_text(s)
    patch_engine_files(native)


def patch_engine_files(native):
    patches=native/'frontend_patched';patches.mkdir(exist_ok=True)
    # Preserve the upstream GPL notices and the prior phoneme-only guard patch.
    original=WORK/'m4_frontend_4mb_20261022/upstream/espeak-ng/src/libespeak-ng/dictionary.c'
    s=original.read_text();s='#include "romfs.h"\n'+s
    old='''if ((tr->data_dictlist = malloc(size)) == NULL) {
		fclose(f);
		return 3;
	}
	size = fread(tr->data_dictlist, 1, size, f);
	fclose(f);'''
    assert old in s
    new='''size_t borrowed_bytes=0;
	const void*borrowed=n25_romfs_direct_file(fname,&borrowed_bytes);
	if(borrowed){
		tr->data_dictlist=(char*)borrowed;size=(int)borrowed_bytes;fclose(f);
	}else{
		'''+old+'''\n	}'''
    s=s.replace(old,new);(patches/'dictionary.c').write_text(s)
    original=WORK/'m4_frontend_phonemes_20261023/patched/synthdata.c'
    s=original.read_text();s='#include "romfs.h"\n'+s
    point='\tif ((f_in = fopen(buf, "rb")) == NULL)'
    assert point in s;s=s.replace(point,'''    size_t borrowed_bytes=0;
    const void*borrowed=n25_romfs_direct_file(buf,&borrowed_bytes);
    if(borrowed){
        if(*ptr)free(*ptr);*ptr=(void*)borrowed;if(size)*size=(int)borrowed_bytes;return ENS_OK;
    }
'''+point);(patches/'synthdata.c').write_text(s)
    print('Prepared SHA-verified G2P prewarm and zero-copy dictionary/phoneme files')


if __name__=='__main__':main()
