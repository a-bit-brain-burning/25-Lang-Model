/* Complete functional lexical resources, independently compressed bounded pages. */
#include "paged_lexicon.h"
#include "bench.h"
#include "storage.h"
#include <stdint.h>
#include <string.h>
int native_deflate_block(const unsigned char*,size_t,unsigned char*,size_t);
static int paged_vint(const unsigned char**p,const unsigned char*end,unsigned*out){
 unsigned value=0;for(unsigned shift=0;shift<=28;shift+=7){if(*p>=end)return -1;unsigned c=*(*p)++;if(shift==28&&(c&240))return -1;value|=(c&127)<<shift;if(!(c&128)){*out=value;return 0;}}return -1;
}
int n25_paged_lexicon_initialize(void){
 for(unsigned i=0;i<n25_paged_lexicon_count;i++){unsigned char digest[32];const N25PagedLexicon*d=n25_paged_lexicons+i;
  storage_sha256(d->data,d->bytes,digest);if(memcmp(digest,d->sha256,32))return -1;
 }bench_metric("native_complete_paged_lexicon_SHA256_passed",n25_paged_lexicon_count);return 0;
}
int n25_paged_lexicon_record(const char*language,const char*word,unsigned char*out,size_t cap,size_t*length){
 const N25PagedLexicon*d=NULL;for(unsigned i=0;i<n25_paged_lexicon_count;i++)if(!strcmp(language,n25_paged_lexicons[i].language))d=n25_paged_lexicons+i;
 if(!d||!word||!out||!length)return -1;*length=0;unsigned lo=0,hi=d->count;
 while(lo<hi){unsigned m=lo+(hi-lo)/2;if(strcmp(d->pages[m].first,word)<=0)lo=m+1;else hi=m;}if(!lo)return -5;
 const N25LexiconPage*page=d->pages+lo-1;if(page->offset>d->bytes||page->packed>d->bytes-page->offset||page->raw>8192)return -2;
 size_t mark=bench_arena_mark();unsigned char*data=bench_alloc(page->raw,16);int rc=-98;
 if(data&&!native_deflate_block(d->data+page->offset,page->packed,data,page->raw)){
  const unsigned char*p=data,*end=p+page->raw;unsigned rows;char key[512]={0};unsigned bytes=0;rc=-2;
  if(!paged_vint(&p,end,&rows)&&rows<=64)for(unsigned i=0;i<rows;i++){
   unsigned prefix,suffix,n;if(paged_vint(&p,end,&prefix)||paged_vint(&p,end,&suffix)||prefix>bytes||suffix>=sizeof(key)-prefix||suffix>(size_t)(end-p))break;
   memcpy(key+prefix,p,suffix);p+=suffix;bytes=prefix+suffix;key[bytes]=0;
   if(paged_vint(&p,end,&n)||n>(size_t)(end-p))break;int cmp=strcmp(key,word);
   if(!cmp){if(n>cap)rc=-6;else{memcpy(out,p,n);*length=n;rc=0;}break;}
   if(cmp>0||i+1==rows){rc=-5;break;}p+=n;
  }
 }if(bench_arena_rewind(mark))return -96;return rc;
}
