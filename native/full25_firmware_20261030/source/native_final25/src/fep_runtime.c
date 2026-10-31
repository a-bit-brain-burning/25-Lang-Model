/* Bounded FEPACK1 decoder for the pinned 25-language resource packages.
 * The raw LZMA2 decoder is XZ Utils liblzma; original notices accompany its source.
 * No host filesystem, Python, libc heap or persistent per-language allocation.
 */
#include "fep_runtime.h"
#include <lzma.h>
#include <string.h>
#include <stdint.h>
typedef struct { unsigned char *base; size_t capacity, used; } FepArena;
static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
static int span(size_t start,size_t bytes,size_t capacity) {
    return start<=capacity && bytes<=capacity-start;
}
static void *arena_alloc(void *opaque,size_t n,size_t bytes) {
    FepArena *a=opaque;
    if(bytes && n>SIZE_MAX/bytes)return NULL;
    uintptr_t address=(uintptr_t)a->base;
    if(a->used>SIZE_MAX-15 || address>UINTPTR_MAX-a->used-15)return NULL;
    uintptr_t aligned=(address+a->used+15)&~(uintptr_t)15;
    size_t at=aligned-address;
    if(!span(at,n*bytes,a->capacity))return NULL;
    a->used=at+n*bytes;
    return (void *)aligned;
}
static void arena_free(void *opaque,void *allocation) {
    (void)opaque;(void)allocation;
}
/* Nibble table avoids a 1 KiB table while remaining much faster than bit loops. */
static uint32_t checksum(const unsigned char *p,size_t n) {
    static const uint32_t table[16]={0,0x1db71064,0x3b6e20c8,0x26d930ac,
        0x76dc4190,0x6b6b51f4,0x4db26158,0x5005713c,0xedb88320,0xf00f9344,
        0xd6d6a3e8,0xcb61b38c,0x9b64c2b0,0x86d3d2d4,0xa00ae278,0xbdbdf21c};
    uint32_t crc=~(uint32_t)0;
    for(size_t i=0;i<n;i++){
        crc^=p[i];crc=(crc>>4)^table[crc&15];crc=(crc>>4)^table[crc&15];
    }
    return ~crc;
}
int fep_open(FepPack *pack,const void *rom,size_t bytes) {
    if(!pack)return FEP_INVALID;
    memset(pack,0,sizeof(*pack));
    if(!rom||bytes<32||bytes>UINT32_MAX)return FEP_INVALID;
    const unsigned char *p=rom;
    if(memcmp(p,"FEPACK1\0",8)||u32(p+8)!=1||u32(p+16)!=32||u32(p+24)!=bytes||u32(p+28))return FEP_INVALID;
    uint32_t n=u32(p+12),payload=u32(p+20);
    if(n>(bytes-32)/56||payload!=32+(size_t)n*56)return FEP_INVALID;
    for(uint32_t i=0;i<n;i++){
        const unsigned char *d=p+32+(size_t)i*56;
        const unsigned char *end=memchr(d,0,32);
        if(!end||end==d||u32(d+32)>1||u32(d+36)<payload||
           !span(u32(d+36),u32(d+40),bytes)||u32(d+52)!=262144)return FEP_INVALID;
        if(!u32(d+32)&&u32(d+40)!=u32(d+44))return FEP_INVALID;
    }
    pack->data=p;pack->size=bytes;pack->sections=n;pack->payload=payload;
    return FEP_OK;
}
int fep_find(const FepPack *pack,const char *name,FepSection *section) {
    if(!pack||!pack->data||!name||!section)return FEP_INVALID;
    memset(section,0,sizeof(*section));
    for(uint32_t i=0;i<pack->sections;i++){
        const unsigned char *d=pack->data+32+(size_t)i*56;
        if(strcmp((const char *)d,name))continue;
        section->codec=u32(d+32);section->packed=pack->data+u32(d+36);
        section->packed_bytes=u32(d+40);section->raw_bytes=u32(d+44);
        section->crc32=u32(d+48);section->window=u32(d+52);
        return FEP_OK;
    }
    return FEP_ABSENT;
}
int fep_decode(const FepSection *s,void *out,size_t cap,void *workspace,size_t workcap,size_t *peak) {
    if(!peak)return FEP_INVALID;
    *peak=0;
    if(!s||!s->packed||(!out&&s->raw_bytes)||s->codec>1||s->window!=262144)return FEP_INVALID;
    if(s->raw_bytes>cap)return FEP_CAPACITY;
    if(!s->codec){
        if(s->raw_bytes!=s->packed_bytes)return FEP_INVALID;
        if(s->raw_bytes)memcpy(out,s->packed,s->raw_bytes);
    }else{
        if(!workspace)return FEP_CAPACITY;
        FepArena arena={workspace,workcap,0};
        lzma_allocator allocator={arena_alloc,arena_free,&arena};
        lzma_options_lzma options;memset(&options,0,sizeof(options));options.dict_size=s->window;
        lzma_filter filters[2]={{LZMA_FILTER_LZMA2,&options},{LZMA_VLI_UNKNOWN,NULL}};
        size_t input=0,output=0;
        lzma_ret rc=lzma_raw_buffer_decode(filters,&allocator,s->packed,&input,s->packed_bytes,
            out,&output,s->raw_bytes);
        *peak=arena.used;
        if(rc==LZMA_MEM_ERROR)return FEP_CAPACITY;
        if(rc!=LZMA_OK||input!=s->packed_bytes||output!=s->raw_bytes)return FEP_DECODE;
    }
    return checksum(out,s->raw_bytes)==s->crc32?FEP_OK:FEP_CRC;
}
