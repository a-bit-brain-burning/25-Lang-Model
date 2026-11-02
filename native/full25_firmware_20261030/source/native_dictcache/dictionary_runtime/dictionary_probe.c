#include "bench.h"
#include "fep_dictionary.h"
#include "lexicon_cases.h"
#include <string.h>
int main(void){
 size_t mark=bench_arena_mark();size_t size=664098;void*workspace=bench_alloc(size,16);if(!workspace)return 1;
 unsigned char out[4096];unsigned matched=0,absent=0;
 for(unsigned i=0;i<sizeof(lexicon_cases)/sizeof(lexicon_cases[0]);i++){
  const LexiconCase*c=lexicon_cases+i;size_t written=0;
  int rc=fep_lexicon_record(fep_lexicons+c->lexicon,c->word,out,sizeof(out),&written,workspace,size);
  if((c->present&&(rc||written!=c->length||memcmp(out,c->expected,written)))||(!c->present&&(rc!=-5||written))){
   bench_metric("failed_case",i);bench_metric("return_code",(unsigned)-rc);bench_puts(c->word);bench_puts("\n");return 2;
  }
  if(c->present)matched++;else absent++;
 }
 bench_metric("lexicon_records_exact",matched);bench_metric("lexicon_absent_exact",absent);
 if(bench_arena_rewind(mark))return 3;bench_metric("lookup_workspace_bytes",size);bench_memory_report();bench_puts("N25_SEVEN_LEXICON_LOOKUP_PASS\n");return 0;
}

int n25_dictionary_warm_es_probe(void){
 unsigned char out[4096];unsigned matched=0;
 for(unsigned i=0;i<sizeof(lexicon_cases)/sizeof(lexicon_cases[0]);i++){
  const LexiconCase*c=lexicon_cases+i;if(c->lexicon!=1)continue;size_t n=0;
  int rc=fep_lexicon_record(fep_lexicons+1,c->word,out,sizeof(out),&n,NULL,0);
  if((c->present&&(rc||n!=c->length||memcmp(out,c->expected,n)))||(!c->present&&(rc!=-5||n)))return 1;
  matched++;
 }
 bench_metric("resident_spanish_cached_queries_exact",matched);return 0;
}
