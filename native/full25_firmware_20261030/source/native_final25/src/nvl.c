/* NVLEX1 finite dictionary paging. Original predictor/residual bytes preserved.
 * Raw DEFLATE decoding uses the existing, caller-buffered zlib puff source.
 */
#include "nvl.h"
#include <string.h>
#include "../third_party/puff/puff.h"
static uint32_t nvl_u32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint16_t nvl_u16(const unsigned char *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static int nvl_span(size_t at,size_t n,size_t cap){return at<=cap&&n<=cap-at;}
static uint32_t nvl_crc(const unsigned char*p,size_t n){
 static const uint32_t t[16]={0,0x1db71064,0x3b6e20c8,0x26d930ac,0x76dc4190,0x6b6b51f4,0x4db26158,0x5005713c,
  0xedb88320,0xf00f9344,0xd6d6a3e8,0xcb61b38c,0x9b64c2b0,0x86d3d2d4,0xa00ae278,0xbdbdf21c};
 uint32_t c=~0u;for(size_t i=0;i<n;i++){c^=p[i];c=(c>>4)^t[c&15];c=(c>>4)^t[c&15];}return ~c;
}
int nvl_open(NvlPack *pack,const void *rom,size_t bytes){
 if(!pack)return -1;memset(pack,0,sizeof(*pack));
 if(!rom||bytes<48||bytes>UINT32_MAX)return -1;const unsigned char*p=rom;
 if(memcmp(p,"NVLEX1\0\0",8)||nvl_u32(p+8)!=1||nvl_u32(p+40)!=bytes||nvl_u32(p+44))return -1;
 unsigned pages=nvl_u32(p+12),words=nvl_u32(p+16),mo=nvl_u32(p+20),mn=nvl_u32(p+24);
 unsigned key_start=nvl_u32(p+28),payload=nvl_u32(p+32);
 if(!pages||pages>(bytes-48)/24||key_start!=48+(size_t)pages*24||payload<key_start||mo<payload||
    !nvl_span(mo,mn,bytes)||nvl_u32(p+36))return -1;
 unsigned count=0;const char *previous=NULL;
 for(unsigned i=0;i<pages;i++){
  const unsigned char*d=p+48+(size_t)i*24;unsigned key=nvl_u32(d),off=nvl_u32(d+4),n=nvl_u32(d+8),rows=nvl_u32(d+16);
  if(key<key_start||key>=payload||!memchr(p+key,0,payload-key)||off<mo+mn||!nvl_span(off,n,bytes)||!rows||rows>256||rows>UINT32_MAX-count)return -1;
  if(previous&&strcmp(previous,(const char*)p+key)>=0)return -1;previous=(const char*)p+key;count+=rows;
 }
 if(count!=words)return -1;
 pack->data=p;pack->bytes=bytes;pack->model=p+mo;pack->model_bytes=mn;pack->pages=pages;pack->words=words;return 0;
}
int nvl_page(const NvlPack *pack,NvlCache *cache,unsigned index){
 if(!pack||!pack->data||!cache||!cache->page||index>=pack->pages)return -1;
 if(cache->owner==pack->data&&cache->index==index)return 0;
 cache->owner=NULL;cache->used=0;
 const unsigned char*d=pack->data+48+(size_t)index*24;unsigned raw=nvl_u32(d+12);
 if(raw>cache->capacity)return -2;
 unsigned long output=raw,input=nvl_u32(d+8);unsigned long expected=input;
 int rc=puff(cache->page,&output,pack->data+nvl_u32(d+4),&input);
 if(rc||output!=raw||input!=expected)return -3;
 if(nvl_crc(cache->page,raw)!=nvl_u32(d+20))return -4;
 cache->owner=pack->data;cache->index=index;cache->used=raw;return 0;
}
int nvl_lookup(const NvlPack *pack,NvlCache *cache,const char *word,unsigned char *out,size_t cap,size_t *written){
 if(!pack||!pack->data||!cache||!word||!written)return -1;*written=0;
 unsigned lo=0,hi=pack->pages;
 while(lo<hi){unsigned mid=lo+(hi-lo)/2;const unsigned char*d=pack->data+48+(size_t)mid*24;
  if(strcmp((const char*)pack->data+nvl_u32(d),word)<=0)lo=mid+1;else hi=mid;}
 if(!lo)return -5;unsigned index=lo-1;int rc=nvl_page(pack,cache,index);if(rc)return rc;
 const unsigned char*p=cache->page,*end=p+cache->used;
 unsigned rows=nvl_u32(pack->data+48+(size_t)index*24+16);
 for(unsigned row=0;row<rows;row++){
  if((size_t)(end-p)<4)return -1;unsigned kn=nvl_u16(p),vn=nvl_u16(p+2);p+=4;
  if(!kn||!vn||(size_t)(end-p)<(size_t)kn+1+vn||p[kn])return -1;
  int cmp=strcmp((const char*)p,word);
  if(!cmp){if(vn>cap||!out)return -2;memcpy(out,p+kn+1,vn);*written=vn;return 0;}
  if(cmp>0)return -5;p+=kn+1+vn;
 }
 return p==end?-5:-1;
}
