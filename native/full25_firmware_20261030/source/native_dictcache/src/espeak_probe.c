/* Validate device-side raw-text G2P against the pinned host eSpeak backend.
 * NEW25 language-specific feature adapters remain a separate integration step.
 */
#include "bench.h"
#include "romfs.h"
#include "espeak_cases.h"
#include <espeak-ng/speak_lib.h>
#include <string.h>
int main(void){
    n25_romfs_initialize();
    int rate=espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,0,"/espeak-ng-data",0);
    if(rate<=0){bench_puts("ESPEAK_INITIALIZE_FAILED\n");return 1;}
    for(unsigned pass=0;pass<2;pass++)for(unsigned i=0;i<ESPEAK_CASE_COUNT;i++){
        const EspeakCase *row=espeak_cases+i;
        if(espeak_SetVoiceByName(row->voice)!=EE_OK){bench_metric("voice_failed_index",i);bench_puts(row->voice);return 2;}
        const void *text=row->text;char output[4096]={0};size_t bytes=0;
        for(unsigned clause=0;text&&clause<100;clause++){
            const void *old=text;
            const char *ipa=espeak_TextToPhonemes(&text,espeakCHARS_UTF8,2|('_'<<8));
            if(text==old){bench_puts("ESPEAK_CURSOR_STUCK\n");return 3;}
            if(!ipa)ipa="";
            size_t n=strlen(ipa);
            if(bytes+n+2>=sizeof(output))return 4;
            if(bytes)output[bytes++]=' ';
            memcpy(output+bytes,ipa,n);bytes+=n;output[bytes]=0;
        }
        size_t begin=0;while(output[begin]==' ')begin++;
        while(bytes>begin&&output[bytes-1]==' ')output[--bytes]=0;
        if(strcmp(output+begin,row->ipa)){
            bench_puts("ESPEAK_IPA_MISMATCH\n");bench_puts(row->language);bench_puts("\n");
            bench_puts(output+begin);bench_puts("\nEXPECTED:\n");bench_puts(row->ipa);bench_puts("\n");return 5;
        }
    }
    espeak_Terminate();
    bench_metric("native_espeak_language_cases",ESPEAK_CASE_COUNT);
    bench_metric("native_espeak_switch_passes",2);
    bench_metric("native_espeak_heap_peak_bytes",n25_romfs_heap_peak());
    bench_metric("native_espeak_decode_workspace_peak_bytes",n25_romfs_decode_workspace_peak());
    bench_metric("native_espeak_decoded_blocks",n25_romfs_decode_count());
    bench_memory_report();bench_puts("N25_ROMFS_ESPEAK_PASS\n");return 0;
}
