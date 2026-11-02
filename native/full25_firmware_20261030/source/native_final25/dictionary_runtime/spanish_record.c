/* Exact original FEP residual + bounded SPR model + stress overlay decoding. */
#include "spanish_record.h"
#include "spanish_tables.h"
#include "fep_dictionary.h"
#include "fep_assets.h"
#include "fep_runtime.h"
#include <string.h>
int open_word_allowed(const void*,size_t,const uint32_t*,size_t);
int open_predict(const spr_view*,const void*,size_t,const uint32_t*,size_t,const void*,size_t,void*,size_t,void*,size_t,size_t*);
static int variable(const unsigned char**p,const unsigned char*end,uint32_t*out){
 uint32_t value=0;for(unsigned shift=0;shift<=28;shift+=7){if(*p>=end)return -1;unsigned c=*(*p)++;
  if(shift==28&&(c&240))return -1;value|=(uint32_t)(c&127)<<shift;if(!(c&128)){*out=value;return 0;}}
 return -1;
}
static int integer(const unsigned char**p,const unsigned char*end,int*out){
 if(*p>=end||*(*p)++!=3)return -1;uint32_t v;if(variable(p,end,&v))return -1;
 *out=v&1?-(int)(v/2)-1:(int)(v/2);return 0;
}
static int skip(const unsigned char**p,const unsigned char*end,unsigned depth){
 if(*p>=end||depth>32)return -1;unsigned tag=*(*p)++;if(tag<3)return 0;
 if(tag==7){if((size_t)(end-*p)<8)return -1;*p+=8;return 0;}
 uint32_t count;if(variable(p,end,&count))return -1;
 if(tag==3||tag==4)return 0;
 if(tag==8||tag==9){if(count>(size_t)(end-*p))return -1;*p+=count;return 0;}
 if(tag!=5&&tag!=6)return -1;if(count>(size_t)(end-*p))return -1;
 for(uint32_t i=0;i<count;i++){if(skip(p,end,depth+1))return -1;if(tag==6&&skip(p,end,depth+1))return -1;}
 return 0;
}
int spanish_context_init(SpanishContext*c,void*model,size_t modelcap,void*workspace,size_t workcap){
 if(!c)return -1;
 size_t cached_length=0;const void*cached_model=fep_lexicon_cached_section(1,"model",&cached_length);
 if(cached_model){
  if(cached_length!=SPANISH_MODEL_BYTES||spr_open(&c->model,cached_model,cached_length))return -3;
  c->workspace=NULL;c->workspace_bytes=0;return 0;
 }
 if(!model||!workspace||workcap<664098||modelcap<SPANISH_MODEL_BYTES)return -1;
 const FepProbePack*image=fep_probe_packs+fep_lexicons[1].pack;FepPack pack;FepSection section;size_t peak=0;
 if(fep_open(&pack,image->data,image->bytes)||fep_find(&pack,"model",&section)||section.raw_bytes!=SPANISH_MODEL_BYTES)return -2;
 size_t cached_bytes=0;const void*cached=fep_lexicon_cached_section(1,"model",&cached_bytes);
 if(cached){if(cached_bytes!=SPANISH_MODEL_BYTES||spr_open(&c->model,cached,cached_bytes))return -3;}
 else if(fep_decode(&section,model,modelcap,workspace,320000,&peak)||spr_open(&c->model,model,SPANISH_MODEL_BYTES))return -3;
 c->workspace=workspace;c->workspace_bytes=workcap;return 0;
}
int spanish_word(SpanishContext*c,const char*word,unsigned char*phones,signed char*stress,unsigned capacity,unsigned*count){
 if(!c||!word||!phones||!stress||!count)return -1;*count=0;
 unsigned char record[4096],prediction[1024],raw[4096];size_t rn=0,pn=0,n=0;
 int rc=fep_lexicon_record(fep_lexicons+1,word,record,sizeof(record),&rn,c->workspace,c->workspace_bytes);
 if(rc==-5){
  if(!open_word_allowed(word,strlen(word),spanish_open_alphabet,sizeof(spanish_open_alphabet)/4))return -5;
  rc=open_predict(&c->model,word,strlen(word),spanish_open_alphabet,sizeof(spanish_open_alphabet)/4,
   spanish_default_record,sizeof(spanish_default_record),prediction,sizeof(prediction),raw,sizeof(raw),&n);
 }else if(!rc){
  if(!rn||record[0]>>4)return -6;
  rc=spr_predict(&c->model,word,strlen(word),prediction,sizeof(prediction),&pn);
  if(!rc)rc=spr_apply(prediction,pn,record,rn,raw,sizeof(raw),&n);
 }
 if(rc)return rc;
 unsigned length=0;while(length<n&&raw[length]!=255){if(raw[length]>=SPANISH_SYMBOL_COUNT||length>=capacity)return -7;
  phones[length]=raw[length];stress[length]=-1;length++;}
 if(!length||length>=n)return -7;
 const unsigned char*p=raw+length+1,*end=raw+n;uint32_t fields;int template_id,mask;
 if(p>=end||*p++!=5||variable(&p,end,&fields)||integer(&p,end,&template_id)||template_id||p>=end||*p++!=0||integer(&p,end,&mask)||mask<0||mask>3)return -8;
 if(fields!=3+(unsigned)((mask&1)!=0)+(unsigned)((mask&2)!=0))return -8;
 if(mask&1){uint32_t size;if(p>=end||*p++!=5||variable(&p,end,&size)||size!=length)return -8;
  for(unsigned i=0;i<length;i++){int s;if(integer(&p,end,&s)||s< -1||s>2)return -8;stress[i]=(signed char)s;}}
 if((mask&2)&&skip(&p,end,0))return -8;if(p!=end)return -8;
 *count=length;return 0;
}
