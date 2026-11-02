#include "bench.h"
#include "romfs.h"
#include "storage.h"
#include "four_frontend.h"
#include "four_cases.h"
#include <espeak-ng/speak_lib.h>
#include <string.h>
static N25FourFeatures features;
int main(void){
 n25_romfs_initialize();
 if(espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,0,"/espeak-ng-data",0)<=0)return 1;
 for(unsigned i=0;i<sizeof(four_cases)/sizeof(four_cases[0]);i++){
  const FourCase*row=four_cases+i;int rc=n25_four_frontend(row->language,row->text,&features);
  if(rc||features.count!=row->count||features.phonetic_dim!=row->dim){
   bench_metric("case_index",i);bench_metric("expected_count",row->count);
   bench_puts("NATIVE_FOUR_FRONTEND_FAILED\n");bench_puts(row->language);bench_puts("\n");bench_metric("code",(unsigned)-rc);bench_metric("count",features.count);return 2;
  }
  const void*data[4]={features.phones,features.pause,features.phonetic,features.context};
  unsigned bytes[4]={row->count*4,row->count*4,row->count*row->dim*4,row->count*row->context_dim*4};
  for(unsigned k=0;k<4;k++){
   unsigned char digest[32];storage_sha256(data[k],bytes[k],digest);
   if(memcmp(digest,row->sha256[k],32)){
    bench_metric("case_index",i);
    bench_puts("NATIVE_FOUR_ARRAY_MISMATCH\n");bench_puts(row->language);bench_puts("\n");bench_metric("array_index",k);
    char name[32]="four_mismatch_0.bin";name[14]=(char)('0'+k);bench_write_file(name,data[k],bytes[k]);return 3;
   }
  }
 }
 espeak_Terminate();bench_metric("native_four_feature_cases",sizeof(four_cases)/sizeof(four_cases[0]));
 bench_metric("native_four_heap_peak_bytes",n25_romfs_heap_peak());bench_memory_report();
 bench_puts("N25_NATIVE_FOUR_FEATURES_PASS\n");return 0;
}
