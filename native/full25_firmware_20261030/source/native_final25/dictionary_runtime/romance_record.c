/* Original full lexicon residuals and SPR predictions, with explicit overlays. */
#include "romance_record.h"
#include "fep_dictionary.h"
#include "fep_runtime.h"
#include "fep_assets.h"
#include "value_decoders.h"
#include <espeak-ng/speak_lib.h>
#include <string.h>
int open_word_allowed(const void*,size_t,const uint32_t*,size_t);
int open_predict(const spr_view*,const void*,size_t,const uint32_t*,size_t,const void*,size_t,void*,size_t,void*,size_t,size_t*);
static int rv(const unsigned char**p,const unsigned char*end,uint32_t*out){
 uint32_t v=0;for(unsigned shift=0;shift<=28;shift+=7){if(*p>=end)return -1;unsigned c=*(*p)++;if(shift==28&&(c&240))return -1;v|=(uint32_t)(c&127)<<shift;if(!(c&128)){*out=v;return 0;}}return -1;
}
static int ri(const unsigned char**p,const unsigned char*end,int*out){
 uint32_t n;if(*p>=end||*(*p)++!=3||rv(p,end,&n))return -1;*out=n&1?-(int)(n/2)-1:(int)(n/2);return 0;
}
static int rskip(const unsigned char**p,const unsigned char*end,unsigned depth){
 if(*p>=end||depth>32)return -1;unsigned tag=*(*p)++;if(tag<3)return 0;if(tag==7){if(end-*p<8)return -1;*p+=8;return 0;}
 uint32_t n;if(rv(p,end,&n))return -1;if(tag==3||tag==4)return 0;
 if(tag==8||tag==9){if(n>(size_t)(end-*p))return -1;*p+=n;return 0;}if((tag!=5&&tag!=6)||n>(size_t)(end-*p))return -1;
 for(unsigned i=0;i<n;i++)if(rskip(p,end,depth+1)||(tag==6&&rskip(p,end,depth+1)))return -1;return 0;
}
static int decode_section(unsigned lexicon,const char*name,void*out,size_t bytes,void*work){
 const FepProbePack*image=fep_probe_packs+fep_lexicons[lexicon].pack;FepPack pack;FepSection s;size_t peak=0;
 return fep_open(&pack,image->data,image->bytes)||fep_find(&pack,name,&s)||s.raw_bytes!=bytes||fep_decode(&s,out,bytes,work,320000,&peak)?-12:0;
}
int romance_record_init(RomanceRecordContext*c,const RomanceDictionary*d,void*work,size_t work_bytes,void*model,void*aux){
 if(!c||!d||!work||work_bytes<d->workspace_bytes||!model||(d->aux_bytes&&!aux))return -1;
 if(decode_section(d->lexicon,"model",model,d->model_bytes,work)||spr_open(&c->model,model,d->model_bytes))return -2;
 if(d->aux_bytes&&decode_section(d->lexicon,"auxiliary",aux,d->aux_bytes,work))return -2;
 c->d=d;c->workspace=work;c->workspace_bytes=work_bytes;c->aux=aux;return 0;
}
static int french_prediction(const RomanceDictionary*d,const char*word,unsigned mode,unsigned char*out,size_t capacity,size_t*length){
 if(espeak_SetVoiceByName("fr")!=EE_OK)return -5;const void*cursor=word;unsigned clauses=0,n=0;
 while(cursor&&clauses++<100){const void*old=cursor;const char*p=espeak_TextToPhonemes(&cursor,espeakCHARS_UTF8,2|('_'<<8));if(old==cursor)return -5;if(!p)p="";
  while(*p){while(*p=='_'||*p==' '||*p=='\t'||*p=='\n')p++;if(!*p)break;char name[64];unsigned used=0;
   while(*p&&*p!='_'&&*p!=' '&&*p!='\t'&&*p!='\n'){
    if((unsigned char)p[0]==0xcb&&((unsigned char)p[1]==0x88||(unsigned char)p[1]==0x8c||(mode==1&&(unsigned char)p[1]==0x90))){p+=2;continue;}
    if(used+1>=sizeof(name))return -5;name[used++]=*p++;
   }name[used]=0;unsigned symbol=254;for(unsigned j=0;j<d->symbol_count;j++)if(!strcmp(name,d->symbols[j])){symbol=j;break;}
   if(n>=capacity)return -6;out[n++]=(unsigned char)symbol;
  }
 }*length=n;return 0;
}
int romance_record_word(RomanceRecordContext*c,const char*word,unsigned char*phones,signed char*stress,unsigned capacity,unsigned*count){
 if(!c||!word||!phones||!stress||!count)return -1;const RomanceDictionary*d=c->d;*count=0;
 unsigned char record[4096],prediction[1024],raw[4096],expanded[4096];size_t rn=0,pn=0,n=0;
 int rc=fep_lexicon_record(fep_lexicons+d->lexicon,word,record,sizeof(record),&rn,c->workspace,c->workspace_bytes);
 if(rc==-5&&d->default_bytes){
  if(!open_word_allowed(word,strlen(word),d->alphabet,d->alphabet_count))return -5;
  rc=open_predict(&c->model,word,strlen(word),d->alphabet,d->alphabet_count,d->default_record,d->default_bytes,prediction,sizeof(prediction),raw,sizeof(raw),&n);
 }else if(!rc){
  if(!rn)return -6;unsigned mode=record[0]>>4;
  if(mode){if(strcmp(d->language,"fr")||mode>2)return -6;rc=french_prediction(d,word,mode,prediction,sizeof(prediction),&pn);record[0]&=15;}
  else rc=spr_predict(&c->model,word,strlen(word),prediction,sizeof(prediction),&pn);
  if(!rc)rc=spr_apply(prediction,pn,record,rn,raw,sizeof(raw),&n);
 }if(rc)return rc;
 const unsigned char*value=raw;
 if(d->aux_bytes){rc=aux_expand(c->aux,d->aux_bytes,raw,n,expanded,sizeof(expanded),&n);if(rc)return rc;value=expanded;}
 unsigned length=0;while(length<n&&value[length]!=255){if(value[length]>=d->symbol_count||length>=capacity)return -7;phones[length]=value[length];stress[length]=-1;length++;}
 if(!length||length>=n)return -7;
 const unsigned char*p=value+length+1,*end=value+n;uint32_t fields;int tid,mask;
 if(p>=end||*p++!=5||rv(&p,end,&fields)||fields<3||ri(&p,end,&tid)||tid||p>=end||*p++!=0||ri(&p,end,&mask)||mask<0||mask>3)return -8;
 if(fields!=3+(unsigned)((mask&1)!=0)+(unsigned)((mask&2)!=0))return -8;
 if(mask&1){uint32_t size;if(p>=end||*p++!=5||rv(&p,end,&size)||size!=length)return -8;
  for(unsigned i=0;i<length;i++){int s;if(ri(&p,end,&s)||s< -1||s>2)return -8;stress[i]=(signed char)s;}}
 if((mask&2)&&rskip(&p,end,0))return -8;if(p!=end)return -8;*count=length;return 0;
}
