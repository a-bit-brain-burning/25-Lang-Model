/* Original compressed dictionaries, without materializing a host word map. */
#include "fep_dictionary.h"
#include "fep_runtime.h"
#include "fep_assets.h"
#include "reader.h"
#include <string.h>
#include "bench.h"
typedef struct{const char*name;const unsigned char*data;size_t bytes;}WarmSection;
static WarmSection spanish_warm[4];
static FepKeyIndex spanish_key_index;
static unsigned spanish_ready,spanish_rows;
static uint32_t*spanish_id_index;
const void*fep_lexicon_cached_section(unsigned lexicon,const char*name,size_t*bytes){
 if(!spanish_ready||lexicon!=1||!name)return NULL;
 for(unsigned i=0;i<4;i++)if(!strcmp(name,spanish_warm[i].name)){if(bytes)*bytes=spanish_warm[i].bytes;return spanish_warm[i].data;}
 return NULL;
}
int fep_lexicon_prewarm_es(void){
 if(spanish_ready)return 0;
 static const char*names[]={"keys/0","ids/0/0","pool/0","model"};
 const FepProbePack*image=fep_probe_packs+fep_lexicons[1].pack;FepPack pack;FepSection sections[4];
 if(fep_open(&pack,image->data,image->bytes))return -1;
 size_t persistent=bench_arena_mark();
 for(unsigned i=0;i<4;i++){
  if(fep_find(&pack,names[i],sections+i))return -2;
  void*out=bench_alloc(sections[i].raw_bytes,16);if(!out)return -3;
  spanish_warm[i]=(WarmSection){names[i],out,sections[i].raw_bytes};
 }
 void*index=bench_alloc(14668,16);if(!index)return -3;
 const unsigned char*header=spanish_warm[0].data;
 /* Row count is read after decoding below. Reserve the bounded 65536/64 directory. */
 spanish_id_index=bench_alloc(496*4,16);if(!spanish_id_index)return -3;
 size_t temporary=bench_arena_mark();void*work=bench_alloc(320000,16);if(!work)return -3;
 for(unsigned i=0;i<4;i++){size_t peak=0;
  if(fep_decode(sections+i,(void*)spanish_warm[i].data,spanish_warm[i].bytes,work,320000,&peak))return -4;
 }
 if(fep_keys_index_build(&spanish_key_index,spanish_warm[0].data,spanish_warm[0].bytes,
    fep_lexicons[1].alphabet,fep_lexicons[1].alphabet_count,index,14668))return -5;
 spanish_rows=(unsigned)header[12]|(unsigned)header[13]<<8|(unsigned)header[14]<<16|(unsigned)header[15]<<24;
 if(fep_ids_index_build(spanish_warm[1].data,spanish_warm[1].bytes,spanish_rows,spanish_id_index,496))return -7;
 if(bench_arena_rewind(temporary))return -6;spanish_ready=1;
 bench_metric("spanish_full_dictionary_persistent_bytes",temporary-persistent);
 bench_metric("spanish_full_dictionary_index_bytes_used",spanish_key_index.used);
 bench_metric("spanish_full_dictionary_boot_CRC_passed",4);return 0;
}
static int read_section(const FepPack*pack,const char*name,void*out,size_t cap,void*work,size_t*bytes){
 FepSection section;size_t peak=0;
 int rc=fep_find(pack,name,&section);if(rc)return -10;
 rc=fep_decode(&section,out,cap,work,320000,&peak);if(rc)return -11;
 *bytes=section.raw_bytes;return 0;
}
static int section_view(const FepPack*pack,const char*name,void*buffer,size_t cap,void*work,size_t*bytes,const unsigned char**value){
 const FepProbePack*es=fep_probe_packs+fep_lexicons[1].pack;
 if(pack->data==es->data){const void*p=fep_lexicon_cached_section(1,name,bytes);if(p){*value=p;return 0;}}
 int rc=read_section(pack,name,buffer,cap,work,bytes);if(!rc)*value=buffer;return rc;
}
static void section_name(char*out,const char*prefix,unsigned block){
 size_t n=strlen(prefix);memcpy(out,prefix,n);char digits[10];unsigned k=0;
 do{digits[k++]=(char)('0'+block%10);block/=10;}while(block);
 while(k)out[n++]=digits[--k];out[n]=0;
}
int fep_lexicon_record(const FepLexiconDesc*d,const char*word,unsigned char*out,size_t cap,size_t*written,void*workspace,size_t workcap){
 if(!d||!word||!out||!written||d->pack>=FEP_PROBE_PACK_COUNT||!d->blocks)return -1;
 if((!workspace||workcap<320000)&&!(spanish_ready&&d->pack==fep_lexicons[1].pack))return -1;
 *written=0;if(!*word)return -5;
 unsigned lo=0,hi=d->blocks;
 while(lo<hi){unsigned m=lo+(hi-lo)/2;if(strcmp(d->first_keys[m],word)<=0)lo=m+1;else hi=m;}
 if(!lo)return -5;unsigned block=lo-1;
 const FepProbePack*image=fep_probe_packs+d->pack;FepPack pack;
 if(fep_open(&pack,image->data,image->bytes))return -2;
 unsigned char*buffer=workspace?(unsigned char*)workspace+320000:NULL;size_t capacity=workcap>=320000?workcap-320000:0,bytes=0,unused=0;
 char name[32];section_name(name,"keys/",block);
 const unsigned char*value;
 int rc=section_view(&pack,name,buffer,capacity,workspace,&bytes,&value);if(rc)return rc;
 int row=spanish_ready&&d->pack==fep_lexicons[1].pack?fep_keys_index_query(&spanish_key_index,word):
   fep_keys(value,bytes,d->alphabet,d->alphabet_count,word,NULL,0,&unused);
 if(row<0)return row;
 section_name(name,"ids/0/",block);rc=section_view(&pack,name,buffer,capacity,workspace,&bytes,&value);if(rc)return rc;
 uint32_t id;
 if(spanish_ready&&d->pack==fep_lexicons[1].pack){if(fep_id_index_query(value,bytes,spanish_id_index,spanish_rows,(unsigned)row,&id))return -3;}
 else if(fep_id(value,bytes,(unsigned)row,&id))return -3;
 rc=section_view(&pack,"pool/0",buffer,capacity,workspace,&bytes,&value);if(rc)return rc;
 size_t length;
 if(fep_pool(value,bytes,id,&value,&length))return -4;
 if(!length)return -5;if(length>cap)return -6;
 memcpy(out,value,length);*written=length;return 0;
}
