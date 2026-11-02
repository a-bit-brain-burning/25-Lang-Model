/* One linked image exercises resources, G2P and the accepted native Chinese TTS.
 * The other language-specific NEW25 feature adapters are NOT yet integrated.
 */
#include "bench.h"
#include "romfs.h"
int n25_fep_probe_run(void);
int n25_espeak_probe_run(void);
int n25_resident_backend_main(void);
int n25_dictionary_probe_run(void);
void tts_runtime_init(void);
int main(void){
    extern unsigned char __frontend_bss_start[],__frontend_bss_end[];
    for(unsigned char*p=__frontend_bss_start;p<__frontend_bss_end;p++)*p=0;
    size_t mark=bench_arena_mark();
    if(n25_fep_probe_run())return 20;
    if(bench_arena_rewind(mark))return 21;
    if(n25_dictionary_probe_run())return 26; /* Boot-only cold-reader regression; no persistent results. */
    tts_runtime_init();
    mark=bench_arena_mark(); /* Prove native G2P while model weights stay resident. */
    void *workspace=bench_alloc(320000+262144,16);
    if(!workspace||n25_romfs_set_workspace(workspace,320000+262144))return 22;
    if(n25_romfs_prewarm_files())return 25;
    if(n25_espeak_probe_run())return 23;
    if(n25_romfs_set_workspace(NULL,0)||bench_arena_rewind(mark))return 24;
    bench_puts("N25_COMBINED_RESOURCES_G2P_AND_ZH_BACKEND_BEGIN\n");
    int rc=n25_resident_backend_main();
    if(rc)return rc;
    bench_puts("N25_COMBINED_COMPONENTS_PASS_NOT_FULL25_TTS\n");
    return 0;
}
