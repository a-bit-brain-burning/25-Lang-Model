/* Bounded FEPACK1 section decoder and exact lookup primitives.
 * Host tests use liblzma. This source does not assert an ARM link/code budget.
 * Caller owns the ROM, output buffers and allocator arena. No system malloc.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <lzma.h>
#include <zlib.h>
#include "reader.h"

typedef struct { unsigned char *p; size_t cap, used; } Arena;
static void *alloc(void *opaque,size_t n,size_t size) {
    Arena *a=opaque;size_t offset=(a->used+15u)&~(size_t)15u;
    if ((size && n>SIZE_MAX/size) || offset>a->cap || n*size>a->cap-offset) return NULL;
    void *p=a->p+offset;a->used=offset+n*size;return p;
}
static void release(void *opaque,void *p) {(void)opaque;(void)p;}
static uint32_t u32(const unsigned char *p) {return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static int span(size_t start,size_t n,size_t cap) {return start<=cap && n<=cap-start;}
static int vi(const unsigned char **p,const unsigned char *end,uint32_t *out) {
    uint32_t v=0;
    for(unsigned s=0;s<=28;s+=7) {
        if(*p>=end)return -1;unsigned c=*(*p)++;
        if(s==28 && (c&0xf0))return -1;v|=(uint32_t)(c&127)<<s;
        if(!(c&128)){*out=v;return 0;}
    }return -1;
}

int fep_read(const unsigned char *rom,size_t size,const char *name,unsigned char *out,size_t cap,
             void *workspace,size_t workcap,size_t *used,size_t *written) {
    if(!rom||!name||!workspace||!used||!written||size<32)return -1;
    *used=*written=0;
    if(memcmp(rom,"FEPACK1\0",8)||u32(rom+8)!=1||u32(rom+16)!=32||u32(rom+24)!=size||u32(rom+28))return -1;
    uint32_t n=u32(rom+12),pay=u32(rom+20);
    if(n>(size-32)/56 || pay!=32+(size_t)n*56)return -1;
    for(uint32_t i=0;i<n;i++) {
        const unsigned char *d=rom+32+(size_t)i*56;
        if(!memchr(d,0,32))return -1;
        if(strcmp((const char *)d,name))continue;
        uint32_t codec=u32(d+32),start=u32(d+36),packed=u32(d+40),raw=u32(d+44),crc=u32(d+48),window=u32(d+52);
        if(codec>1||start<pay||!span(start,packed,size)||window!=262144||raw>cap||(!out&&raw))return -2;
        if(!codec) {if(raw!=packed)return -1;if(raw)memcpy(out,rom+start,raw);}
        else {
            Arena a={(unsigned char *)workspace,workcap,0};lzma_allocator allocator={alloc,release,&a};
            lzma_options_lzma options;memset(&options,0,sizeof(options));options.dict_size=window;
            lzma_filter filters[2]={{LZMA_FILTER_LZMA2,&options},{LZMA_VLI_UNKNOWN,NULL}};
            size_t inpos=0,outpos=0;
            lzma_ret rc=lzma_raw_buffer_decode(filters,&allocator,rom+start,&inpos,packed,out,&outpos,raw);
            *used=a.used;
            if(rc!=LZMA_OK||inpos!=packed||outpos!=raw)return -3;
        }
        if(crc32(0,out,raw)!=crc)return -4;*written=raw;return 0;
    }return -5;
}

static int utf8(const unsigned char **p,const unsigned char *end,uint32_t *out) {
    if(*p>=end)return -1;unsigned c=*(*p)++;unsigned n;uint32_t v,min;
    if(c<128){if(!c)return -1;*out=c;return 0;}
    if(c>=194&&c<224){n=1;v=c&31;min=128;}
    else if(c>=224&&c<240){n=2;v=c&15;min=2048;}
    else if(c>=240&&c<245){n=3;v=c&7;min=65536;}
    else return -1;
    for(unsigned i=0;i<n;i++){if(*p>=end)return -1;c=*(*p)++;if((c&192)!=128)return -1;v=v<<6|(c&63);}
    if(v<min||v>0x10ffff||(v>=0xd800&&v<=0xdfff))return -1;*out=v;return 0;
}
static int emit(uint32_t cp,unsigned char *out,size_t *n,size_t cap) {
    unsigned count=cp<128?1:cp<2048?2:cp<65536?3:4;
    if(!cp||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff)||count>cap-*n)return -1;
    if(count==1)out[(*n)++]=(unsigned char)cp;
    else {
        out[(*n)++]=(unsigned char)((count==2?192:count==3?224:240)|(cp>>(6*(count-1))));
        for(unsigned i=count-1;i;i--)out[(*n)++]=(unsigned char)(128|((cp>>(6*(i-1)))&63));
    }return 0;
}

/* query != NULL: exact row index, -5 absent. Otherwise NUL-separated enumeration.
 * The returned row index is within this block, not a hash or probabilistic hit.
 */
int fep_keys(const unsigned char *raw,size_t size,const uint32_t *alphabet,size_t an,const char *query,
             unsigned char *output,size_t cap,size_t *written) {
    if(!raw||!written||size<32||memcmp(raw,"FEKEY1\0\0",8))return -1;*written=0;
    uint32_t mode=u32(raw+8),n=u32(raw+12),alpha=u32(raw+16),pn=u32(raw+20),sn=u32(raw+24),fn=u32(raw+28);
    if(mode>2||n>65536||(mode==2?(alpha!=an||!alphabet):alpha!=0))return -1;
    if(!span(32,pn,size)||!span(32+(size_t)pn,sn,size)||!span(32+(size_t)pn+sn,fn,size))return -1;
    const unsigned char *p=raw+32,*pe=p+pn,*s=pe,*se=s+sn,*f=se,*fe=f+fn,*t=fe,*end=raw+size;
    unsigned char prev[1024]={0},key[1024];uint32_t prev_ids[256];size_t prev_n=0,prev_c=0,written_n=0;
    int found=-5;
    for(uint32_t i=0;i<n;i++) {
        uint32_t prefix,suffix;size_t key_n=0;
        if(vi(&p,pe,&prefix)||vi(&s,se,&suffix)||!suffix)return -1;
        if(mode==2) {
            uint32_t delta;
            if(prefix>prev_c||suffix>256-prefix||vi(&f,fe,&delta))return -1;
            uint32_t base=prefix<prev_c?prev_ids[prefix]+1:0;
            if(delta>UINT32_MAX-base)return -1;prev_ids[prefix]=base+delta;
            for(uint32_t j=1;j<suffix;j++)if(vi(&t,end,&prev_ids[prefix+j]))return -1;
            prev_c=prefix+suffix;
            for(size_t j=0;j<prev_c;j++)if(prev_ids[j]>=an||emit(alphabet[prev_ids[j]],key,&key_n,sizeof(key)-1))return -1;
        } else {
            size_t prefix_bytes=prefix;
            if(mode==1) {
                const unsigned char *q=prev;uint32_t cp;
                for(uint32_t j=0;j<prefix;j++)if(utf8(&q,prev+prev_n,&cp))return -1;
                prefix_bytes=(size_t)(q-prev);
            }
            if(prefix_bytes>prev_n)return -1;
            memcpy(key,prev,prefix_bytes);key_n=prefix_bytes;
            const unsigned char *start=t;
            if(mode==1){uint32_t cp;for(uint32_t j=0;j<suffix;j++)if(utf8(&t,end,&cp))return -1;}
            else {if(suffix>(size_t)(end-t))return -1;t+=suffix;}
            size_t tail=(size_t)(t-start);
            if(tail>sizeof(key)-1-key_n)return -2;memcpy(key+key_n,start,tail);key_n+=tail;
            const unsigned char *q=key;uint32_t cp;while(q<key+key_n)if(utf8(&q,key+key_n,&cp))return -1;
        }
        key[key_n]=0;
        if(i&&strcmp((char *)prev,(char *)key)>=0)return -1;
        if(query&&!strcmp(query,(char *)key))found=(int)i;
        if(!query){if(!output||key_n+1>cap-written_n)return -2;memcpy(output+written_n,key,key_n+1);written_n+=key_n+1;}
        memcpy(prev,key,key_n+1);prev_n=key_n;
    }
    if(p!=pe||s!=se||f!=fe||t!=end)return -1;
    *written=written_n;return query?found:0;
}

int fep_id(const unsigned char *raw,size_t size,uint32_t index,uint32_t *out) {
    if(!raw||!out||index>=65536)return -1;const unsigned char *p=raw,*end=raw+size;
    for(uint32_t i=0;i<=index;i++)if(vi(&p,end,out))return -1;return 0;
}
int fep_pool(const unsigned char *raw,size_t size,uint32_t index,const unsigned char **value,size_t *length) {
    if(!raw||!value||!length||size<12)return -1;
    uint32_t n=u32(raw),group=u32(raw+4),ng=u32(raw+8);
    if(group!=256||ng!=n/256u+(n%256u!=0)||ng>(size-12)/4||index>=n)return -1;
    size_t base=12+(size_t)ng*4;uint32_t g=index/256,a=u32(raw+12+4*g),b=g+1<ng?u32(raw+12+4*(g+1)):(uint32_t)(size-base);
    if(a>b||b>size-base)return -1;const unsigned char *p=raw+base+a,*end=raw+base+b;uint32_t len;
    for(uint32_t i=g*256;i<=index;i++){
        if(vi(&p,end,&len)||len>(size_t)(end-p))return -1;
        if(i==index){*value=p;*length=len;return 0;}p+=len;
    }return -1;
}
