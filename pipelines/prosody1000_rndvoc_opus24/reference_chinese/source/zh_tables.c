#if defined(__GNUC__)
#pragma GCC optimize("O3")
#endif
#include "zh_tables.h"
#include "unicode_native.h"
#include <string.h>
#include "generated/zh_tables.h"
#include "generated/zh_properties.h"
extern int native_deflate_block(const unsigned char*,size_t,unsigned char*,size_t);
extern uint32_t native_frontend_crc32(const unsigned char*,size_t);
static unsigned char cache[64][ZH_DB_PAGE_RAM] __attribute__((section(".lexcache"),aligned(8)));
static int cached_table[64],cached_page[64],valid[64];static unsigned ages[64],tick;
static uint32_t u32(const unsigned char*p){return(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
int native_zh_varint(const unsigned char**p,const unsigned char*end,uint32_t*out){uint32_t n=0;for(unsigned shift=0;shift<=28;shift+=7){if(*p>=end)return -1;unsigned c=*(*p)++;if(shift==28&&(c&240))return -1;n|=(uint32_t)(c&127)<<shift;if(!(c&128)){*out=n;return 0;}}return -1;}
static int cmp(const uint32_t*a,int an,const uint32_t*b,int bn){int n=an<bn?an:bn;for(int i=0;i<n;i++)if(a[i]!=b[i])return a[i]<b[i]?-1:1;return an<bn?-1:an>bn;}
int native_zh_lookup(int table,const uint32_t*query,int length,unsigned char*out,size_t cap,size_t*used,int*prefix){
 if(table<0||table>1||length<1||length>ZH_MAX_WORD)return 1;
 if(table==1){*used=0;if(prefix)*prefix=0;if(length!=1||ZH_CHARS_BYTES<24||memcmp(zh_chars_rom,"ZHCH1\0\0\0",8))return -1;uint32_t first=u32(zh_chars_rom+8),last=u32(zh_chars_rom+12),pool=u32(zh_chars_rom+16);if(last<first||(last-first)>(ZH_CHARS_BYTES-24)/2||pool!=ZH_CHARS_BYTES-24-(last-first)*2)return -1;if(query[0]<first||query[0]>=last)return 1;const unsigned char*at=zh_chars_rom+24+(query[0]-first)*2;uint32_t offset=at[0]|(unsigned)at[1]<<8;if(offset==65535)return 1;if(offset>=pool)return -1;const unsigned char*start=zh_chars_rom+24+(last-first)*2+offset,*p=start,*end=zh_chars_rom+ZH_CHARS_BYTES;uint32_t n;if(native_zh_varint(&p,end,&n)||n>(size_t)(end-p)/2)return -1;size_t bytes=(size_t)(p-start)+n*2;if(bytes>cap)return -3;memcpy(out,start,bytes);*used=bytes;if(prefix)*prefix=1;return 0;}
 const unsigned char*b=table?zh_chars_rom:zh_words_rom;size_t bytes=table?ZH_CHARS_BYTES:ZH_WORDS_BYTES;*used=0;if(prefix)*prefix=0;if(bytes<28||memcmp(b,"ZHDB3\0\0\0",8))return -1;
 uint32_t pages=u32(b+12),strings=u32(b+16),data=u32(b+20);if(strings<28||data<strings||data>bytes||pages>(strings-28)/24)return -1;
 char key[ZH_MAX_WORD*4+1];int keybytes=0;for(int i=0;i<length;i++){int n=native_utf8_write(query[i],key+keybytes,sizeof(key)-keybytes-1);if(n<1)return -1;keybytes+=n;}key[keybytes]=0;
 uint32_t lo=0,hi=pages;while(lo<hi){uint32_t mid=lo+(hi-lo)/2,at=u32(b+28+mid*24);if(at>=data-strings||!memchr(b+strings+at,0,data-strings-at))return -1;if(strcmp((const char*)b+strings+at,key)<=0)lo=mid+1;else hi=mid;}
 if(prefix&&lo<pages){uint32_t at=u32(b+28+lo*24);if(at>=data-strings)return -1;if(!strncmp((const char*)b+strings+at,key,keybytes))*prefix=1;}
 if(!lo)return 1;uint32_t page=lo-1;const unsigned char*index=b+28+page*24;uint32_t offset=u32(index+4),packed=u32(index+8),raw=u32(index+12),crc=u32(index+16),rows=u32(index+20);if(raw>ZH_DB_PAGE_RAM||offset>bytes-data||packed>bytes-data-offset||rows>256)return -1;
 int slot=-1;unsigned oldest=~0u;for(int i=0;i<64;i++){if(valid[i]&&cached_table[i]==table&&cached_page[i]==(int)page){slot=i;break;}if(!valid[i]){if(oldest){slot=i;oldest=0;}}else if(ages[i]<oldest){slot=i;oldest=ages[i];}}
 if(slot<0)return -1;if(!valid[slot]||cached_table[slot]!=table||cached_page[slot]!=(int)page){valid[slot]=0;int rc=native_deflate_block(b+data+offset,packed,cache[slot],raw);if(rc)return -2;if(native_frontend_crc32(cache[slot],raw)!=crc)return -2;cached_table[slot]=table;cached_page[slot]=page;valid[slot]=1;}
 ages[slot]=++tick;const unsigned char*p=cache[slot],*end=p+raw;uint32_t previous[ZH_MAX_WORD];int pn=0;
 for(unsigned row=0;row<rows;row++){uint32_t common,suffix,delta,n;if(native_zh_varint(&p,end,&common)||native_zh_varint(&p,end,&suffix)||!suffix||common>(unsigned)pn||common+suffix>ZH_MAX_WORD||native_zh_varint(&p,end,&delta))return -2;uint32_t base=common<(unsigned)pn?previous[common]+1:0x3400;if(delta>0x9fff-base)return -2;previous[common]=base+delta;for(unsigned j=1;j<suffix;j++){if(native_zh_varint(&p,end,&delta)||delta>0x9fff-0x3400)return -2;previous[common+j]=0x3400+delta;}pn=common+suffix;
  if(native_zh_varint(&p,end,&n)||n>(size_t)(end-p))return -2;int comparison=cmp(previous,pn,query,length);if(!comparison){if(n>cap)return -3;memcpy(out,p,n);*used=n;if(prefix)*prefix=1;return 0;}if(comparison>0){if(prefix&&pn>=length&&!memcmp(previous,query,length*4))*prefix=1;return 1;}p+=n;
 }if(p!=end)return -2;return 1;
}
