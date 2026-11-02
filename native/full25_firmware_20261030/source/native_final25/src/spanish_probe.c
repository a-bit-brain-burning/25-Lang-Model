#include "bench.h"
#include "nvl.h"
#include "spanish_cases.h"
#include "predict_reader.h"
#include <string.h>
int open_predict(const spr_view*,const void*,size_t,const uint32_t*,size_t,const void*,size_t,
    void*,size_t,void*,size_t,size_t*);
int main(void){
 NvlPack pack;unsigned char *page=bench_alloc(NVL_MAX_PAGE_BYTES,16);
 if(!page||nvl_open(&pack,n25_spanish_nvl,NVL_PACK_BYTES)){bench_puts("NVL_OPEN_FAILED\n");return 1;}
 NvlCache cache={page,NVL_MAX_PAGE_BYTES,0,0,0};spr_view model;
 if(spr_open(&model,pack.model,pack.model_bytes))return 2;
 for(unsigned i=0;i<pack.pages;i++)if(nvl_page(&pack,&cache,i))return 3;
 for(unsigned i=0;i<sizeof(spanish_cases)/sizeof(spanish_cases[0]);i++){
  const SpanishCase *row=spanish_cases+i;
  unsigned char record[4096],prediction[1024],output[4096];size_t rn=0,pn=0,n=0;
  int rc=nvl_lookup(&pack,&cache,row->word,record,sizeof(record),&rn);
  if(rc==-5)rc=open_predict(&model,row->word,strlen(row->word),spanish_alphabet,
      sizeof(spanish_alphabet)/sizeof(spanish_alphabet[0]),spanish_default_record,sizeof(spanish_default_record),
      prediction,sizeof(prediction),output,sizeof(output),&n);
  else if(!rc){
   rc=spr_predict(&model,row->word,strlen(row->word),prediction,sizeof(prediction),&pn);
   if(!rc)rc=spr_apply(prediction,pn,record,rn,output,sizeof(output),&n);
  }
  if(rc||n!=row->length||memcmp(output,row->expected,n)){
   bench_puts("SPANISH_WORD_MISMATCH\n");bench_puts(row->word);bench_metric("case_index",i);return 4;
  }
 }
 bench_metric("spanish_dictionary_words",pack.words);bench_metric("spanish_dictionary_pages",pack.pages);
 bench_metric("spanish_exact_pronunciation_cases",sizeof(spanish_cases)/sizeof(spanish_cases[0]));
 bench_metric("spanish_query_page_buffer_bytes",NVL_MAX_PAGE_BYTES);
 bench_memory_report();bench_puts("N25_SPANISH_DICTIONARY_PASS\n");return 0;
}
