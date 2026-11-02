/* Resource-port acceptance probe; it does not claim text-to-speech completion. */
#include "bench.h"
#include "fep_runtime.h"
#include "storage.h"
#include "fep_assets.h"
#include <string.h>
__attribute__((noinline)) void fep_probe_begin(void){__asm__ volatile("" ::: "memory");}
__attribute__((noinline)) void fep_probe_end(void){__asm__ volatile("" ::: "memory");}
int main(void){
    unsigned char *work=bench_alloc(320000,16),*out=bench_alloc(FEP_MAX_SECTION_BYTES,16);
    if(!work||!out)bench_exit(1);
    size_t peak=0;
    fep_probe_begin();
    for(unsigned i=0;i<FEP_PROBE_SECTION_COUNT;i++){
        const FepProbeSection *row=fep_probe_sections+i;
        FepPack pack;FepSection section;
        const FepProbePack *resource=fep_probe_packs+row->pack;
        if(fep_open(&pack,resource->data,resource->bytes)||fep_find(&pack,row->name,&section)){
            bench_puts("FEP_OPEN_FIND_FAILED\n");bench_metric("section_index",i);bench_exit(2);
        }
        size_t used=0;
        int rc=fep_decode(&section,out,FEP_MAX_SECTION_BYTES,work,320000,&used);
        if(rc){bench_puts("FEP_DECODE_FAILED\n");bench_metric("section_index",i);bench_metric("error",(unsigned)-rc);bench_exit(3);}
        if(used>peak)peak=used;
        unsigned char digest[32];storage_sha256(out,section.raw_bytes,digest);
        if(memcmp(digest,row->sha256,32)){bench_puts("FEP_SHA_FAILED\n");bench_exit(4);}
    }
    fep_probe_end();
    bench_metric("fep_packages_verified",FEP_ACTIVE_PACK_COUNT);
    bench_metric("fep_sections_sha256_verified",FEP_PROBE_SECTION_COUNT);
    bench_metric("fep_workspace_peak_bytes",peak);
    bench_metric("fep_output_reserved_bytes",FEP_MAX_SECTION_BYTES);
    bench_memory_report();
    bench_puts("N25_FEP_RESOURCE_PORT_PASS\n");
    return 0;
}
