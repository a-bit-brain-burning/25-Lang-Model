#include "table_reader.h"
#include <limits.h>
#include <string.h>
#include <zlib.h>

#define HEAD 96u
#define SEC 40u
#define BLK 24u
static uint32_t rd32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint32_t rd16(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8;}
static int span(uint32_t offset,uint32_t length,uint32_t bound){return offset<=bound&&length<=bound-offset;}
static const unsigned char *descriptor(const lex_view *v,uint32_t b){return v->rom+v->block_offset+(size_t)b*BLK;}
static int compare(const unsigned char *a,size_t na,const unsigned char *b,size_t nb){size_t n=na<nb?na:nb;int c=memcmp(a,b,n);return c?c:(na>nb)-(na<nb);}
static uint32_t flags(const lex_view *v){return v->rom[2]=='C'?rd32(v->rom+92):0;}

int lex_open(lex_view *v,const void *rom,size_t bytes){
    lex_view t;const unsigned char *p=rom;uint32_t i,expected_block=0;
    if(!v)return LEX_INVALID;
    memset(v,0,sizeof(*v));
    if(!p||bytes<HEAD||bytes>UINT32_MAX||(memcmp(p,"DEVPK1\0\0",8)&&memcmp(p,"LXBND1\0\0",8)&&memcmp(p,"LXCMP1\0\0",8))||rd32(p+8)!=1)return LEX_INVALID;
    t.rom=p;t.sections=rd32(p+12);t.section_offset=rd32(p+16);t.blocks=rd32(p+20);t.block_offset=rd32(p+24);
    t.key_offset=rd32(p+28);t.data_offset=rd32(p+32);t.size=rd32(p+36);t.max_raw=rd32(p+40);t.max_key=rd32(p+44);
    if(t.size!=bytes||t.section_offset!=HEAD||t.block_offset<t.section_offset||t.key_offset<t.block_offset||t.data_offset<t.key_offset||t.data_offset>t.size)return LEX_INVALID;
    if((t.block_offset-t.section_offset)/SEC!=t.sections||(t.block_offset-t.section_offset)%SEC||
       (t.key_offset-t.block_offset)/BLK!=t.blocks||(t.key_offset-t.block_offset)%BLK)return LEX_INVALID;
    if(flags(&t)){
        uint32_t f=flags(&t),a=rd32(p+84),n=rd32(p+88),j;
        if(f>3 || a<t.key_offset || a>t.data_offset || n>(t.data_offset-a)/4 || a+4*n!=t.data_offset || ((f&1)!=0)!=(n!=0))return LEX_INVALID;
        for(j=0;j<n;j++){uint32_t cp=rd32(p+a+(size_t)j*4);if(cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff))return LEX_INVALID;}
    }else if(!memcmp(p,"LXCMP1\0\0",8))return LEX_INVALID;
    for(i=0;i<t.sections;i++){
        const unsigned char *s=p+t.section_offset+(size_t)i*SEC;
        uint32_t first=rd32(s+24),count=rd32(s+28),j,total=0;
        if(!memchr(s,0,24)||first!=expected_block||first>t.blocks||count>t.blocks-first)return LEX_INVALID;
        for(j=0;j<count;j++){
            const unsigned char *b=descriptor(&t,first+j);uint32_t k=rd32(b),n=rd16(b+4),nr=rd16(b+6),o=rd32(b+8),z=rd32(b+12),r=rd32(b+16);
            if(!n||!nr||n>t.max_key||r>t.max_raw||!span(k,n,t.data_offset-t.key_offset)||!span(o,z,t.size-t.data_offset)||!z)return LEX_INVALID;
            if(j){const unsigned char *q=descriptor(&t,first+j-1);if(compare(p+t.key_offset+rd32(q),rd16(q+4),p+t.key_offset+k,n)>=0)return LEX_INVALID;}
            if(nr>UINT32_MAX-total)return LEX_INVALID;
            total+=nr;
        }
        if(total!=rd32(s+32))return LEX_INVALID;
        expected_block=first+count;
    }
    if(expected_block!=t.blocks)return LEX_INVALID;
    *v=t;return LEX_OK;
}

int lex_section(const lex_view *v,const char *name,size_t n,uint32_t *section){
    uint32_t i;
    if(!v||!v->rom||!name||!section||n>=24)return LEX_INVALID;
    for(i=0;i<v->sections;i++){
        const unsigned char *s=v->rom+v->section_offset+(size_t)i*SEC;
        if(!memcmp(s,name,n)&&s[n]==0){*section=i;return LEX_OK;}
    }
    return LEX_NOT_FOUND;
}

typedef struct{unsigned char *base;size_t capacity,used;} arena_state;
static voidpf take(voidpf opaque,uInt items,uInt bytes){
    arena_state *a=opaque;size_t n,at,pad;
    if(items&&bytes>SIZE_MAX/items)return Z_NULL;
    n=(size_t)items*bytes;
    pad=(8u-(((uintptr_t)a->base+a->used)&7u))&7u;
    if(pad>a->capacity-a->used)return Z_NULL;
    at=a->used+pad;
    if(n>a->capacity-at)return Z_NULL;
    a->used=at+n;memset(a->base+at,0,n);return a->base+at;
}
static void release(voidpf opaque,voidpf address){(void)opaque;(void)address;}

int lex_unpack(const lex_view *v,uint32_t block,void *raw,size_t cap,void *arena,size_t acap,size_t *used){
    const unsigned char *b;uint32_t offset,packed,size;z_stream stream;arena_state a;int code;
    if(used)*used=0;
    if(!v||!v->rom||!raw||!arena||!used||block>=v->blocks)return LEX_INVALID;
    b=descriptor(v,block);offset=rd32(b+8);packed=rd32(b+12);size=rd32(b+16);
    if(size>cap||acap<65536)return LEX_CAPACITY;
    memset(&stream,0,sizeof(stream));a.base=arena;a.capacity=acap;a.used=0;
    stream.zalloc=take;stream.zfree=release;stream.opaque=&a;
    stream.next_in=(Bytef *)(v->rom+v->data_offset+offset);stream.avail_in=packed;
    stream.next_out=raw;stream.avail_out=size;
    code=inflateInit2(&stream,-15);
    if(code!=Z_OK)return LEX_CORRUPT;
    code=inflate(&stream,Z_FINISH);
    if(code!=Z_STREAM_END||stream.total_out!=size||stream.total_in!=packed){inflateEnd(&stream);return LEX_CORRUPT;}
    inflateEnd(&stream);
    if((uint32_t)crc32(0,raw,size)!=rd32(b+20))return LEX_CORRUPT;
    *used=size;return LEX_OK;
}

static int vint(const unsigned char *raw,size_t size,size_t *pos,uint32_t *value){
    unsigned int shift;uint32_t result=0;
    for(shift=0;shift<35;shift+=7){unsigned char b;if(*pos>=size)return 0;b=raw[(*pos)++];if(shift==28&&(b&0xf0))return 0;result|=(uint32_t)(b&127)<<shift;if(!(b&128)){*value=result;return 1;}}
    return 0;
}
static int glyph(const lex_view *v,uint32_t identifier,unsigned char *out,size_t capacity,size_t *position){
    uint32_t cp;unsigned char bytes[4];size_t n;
    if(identifier>=rd32(v->rom+88))return 0;
    cp=rd32(v->rom+rd32(v->rom+84)+(size_t)identifier*4);
    if(cp<128){bytes[0]=(unsigned char)cp;n=1;}
    else if(cp<2048){bytes[0]=(unsigned char)(0xc0|(cp>>6));bytes[1]=(unsigned char)(0x80|(cp&63));n=2;}
    else if(cp<65536){bytes[0]=(unsigned char)(0xe0|(cp>>12));bytes[1]=(unsigned char)(0x80|((cp>>6)&63));bytes[2]=(unsigned char)(0x80|(cp&63));n=3;}
    else {bytes[0]=(unsigned char)(0xf0|(cp>>18));bytes[1]=(unsigned char)(0x80|((cp>>12)&63));bytes[2]=(unsigned char)(0x80|((cp>>6)&63));bytes[3]=(unsigned char)(0x80|(cp&63));n=4;}
    if(n>capacity-*position)return 0;
    memcpy(out+*position,bytes,n);*position+=n;return 1;
}

int lex_lookup(const lex_view *v,uint32_t section,const void *key,size_t key_n,
               void *raw,size_t cap,void *key_buffer,size_t key_cap,
               void *arena,size_t acap,const unsigned char **value,size_t *value_n){
    const unsigned char *s,*b,*r=raw,*wanted=key;unsigned char *current=key_buffer;
    uint32_t first,count,lo,hi,chosen,i,nr,f;size_t used,pos=0,previous=0,end,value_pos=0;int status;
    if(value)*value=NULL;
    if(value_n)*value_n=0;
    if(!v||!v->rom||!key||!key_buffer||!value||!value_n||section>=v->sections)return LEX_INVALID;
    if(key_cap<v->max_key)return LEX_CAPACITY;
    s=v->rom+v->section_offset+(size_t)section*SEC;first=rd32(s+24);count=rd32(s+28);lo=first;hi=first+count;
    while(lo<hi){uint32_t mid=lo+(hi-lo)/2;b=descriptor(v,mid);if(compare(v->rom+v->key_offset+rd32(b),rd16(b+4),wanted,key_n)<=0)lo=mid+1;else hi=mid;}
    if(lo==first)return LEX_NOT_FOUND;
    chosen=lo-1;b=descriptor(v,chosen);nr=rd16(b+6);
    status=lex_unpack(v,chosen,raw,cap,arena,acap,&used);if(status)return status;
    f=flags(v);end=used;
    if(f&2){if(used<4 || rd32(r)>used-4)return LEX_CORRUPT;end=4+rd32(r);pos=4;value_pos=end;}
    for(i=0;i<nr;i++){
        uint32_t prefix,suffix,n;size_t start;int c;
        if(v->rom[0]=='D'){if(used-pos<4)return LEX_CORRUPT;pos+=4;}
        if(!vint(r,end,&pos,&prefix)||!vint(r,end,&pos,&suffix)||prefix>previous||prefix>key_cap)return LEX_CORRUPT;
        if(f&1){uint32_t j;previous=prefix;
            if(suffix>end-pos)return LEX_CORRUPT;
            for(j=0;j<suffix;j++){uint32_t identifier;if(!vint(r,end,&pos,&identifier)||!glyph(v,identifier,current,key_cap,&previous))return LEX_CORRUPT;}
        }else{
            if(suffix>key_cap-prefix||suffix>end-pos)return LEX_CORRUPT;
            memcpy(current+prefix,r+pos,suffix);previous=(size_t)prefix+suffix;pos+=suffix;
        }
        if(!vint(r,end,&pos,&n))return LEX_CORRUPT;
        start=(f&2)?value_pos:pos;if(start>used || n>used-start)return LEX_CORRUPT;
        c=compare(current,previous,wanted,key_n);
        if(c==0){*value=r+start;*value_n=n;return LEX_OK;}
        if(c>0)return LEX_NOT_FOUND;
        if(f&2)value_pos+=n;else pos+=n;
    }
    return pos==end&&(!(f&2)||value_pos==used)?LEX_NOT_FOUND:LEX_CORRUPT;
}
