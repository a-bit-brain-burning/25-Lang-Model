"""Bind all normalization banks and native text feature families to one backend."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import struct
from build_resources import ROOT,WORK,NATIVE,sha,write
LANGUAGES='bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh'.split()
SUPPORTED='zh cs da de el hu nl no pl sk sv tr'.split()


def main():
    dest=ROOT/'native_multilang';dest.mkdir(exist_ok=False)
    for name in('src','assets'):shutil.copytree(NATIVE/name,dest/name)
    for name in('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to(NATIVE/name,target_is_directory=True)
    for name in('build.py','build_native.py','build_o3_resident.py'):shutil.copy2(NATIVE/name,dest/name)
    prepared=WORK/'m55_fargan_stream_20261030/language25_20261030/inputs'
    blob=bytearray();identities=[];cases=[]
    for i,language in enumerate(LANGUAGES):
        header=(prepared/language/'fixed_input.h').read_text()
        receipt=json.loads((prepared/language/'PREPARED.json').read_text())
        assert receipt['language_id']==i
        arrays={}
        for name,body in re.findall(r'static const float (\w+)\[\d+\]=\{(.*?)\};',header,re.S):
            if name not in('norm_phonetic_mean','norm_phonetic_std','norm_context_mean','norm_context_std','norm_mean','norm_std'):continue
            arrays[name]=b''.join(struct.pack('<f',float.fromhex(v.strip().removesuffix('f')))
                for v in body.split(',')if v.strip())
        dims=[receipt['phonetic_dim'],receipt['context_dim'],receipt['acoustic_dim']]
        raw=bytearray(struct.pack('<4I',i,*dims))
        for name,cap in [('norm_phonetic_mean',36),('norm_phonetic_std',36),
            ('norm_context_mean',34),('norm_context_std',34),('norm_mean',39),('norm_std',39)]:
            assert len(arrays[name])<=cap*4
            raw+=arrays[name]+bytes(cap*4-len(arrays[name]))
        assert len(raw)==888;raw+=bytes(8);blob+=raw
        identities.append(dict(language=language,language_id=i,dimensions=dims,
            source_header_sha256=sha(prepared/language/'fixed_input.h'),record_sha256=hashlib.sha256(raw).hexdigest()))
        if language in SUPPORTED:
            text=json.loads((prepared/language/'job.json').read_text())['text']
            cases.append(dict(language=language,language_id=i,text=text))
    assert len(blob)==25*896
    (dest/'assets/language_norms.bin').write_bytes(blob)
    (dest/'assets/language_norms.S').write_text('.section .benchmark_assets.lang_norms,"a",%progbits\n.balign 16\n.global n25_language_norms\nn25_language_norms:\n.incbin "'+str(dest/'assets/language_norms.bin')+'"\n.section .note.GNU-stack,""\n')
    (dest/'src/language_norms.h').write_text('''#ifndef N25_LANGUAGE_NORMS_H
#define N25_LANGUAGE_NORMS_H
#include <stdint.h>
typedef struct __attribute__((aligned(16))) {
 uint32_t language_id,phonetic_dim,context_dim,acoustic_dim;
 float phonetic_mean[36],phonetic_std[36],context_mean[34],context_std[34],mean[39],std[39];
} N25LanguageNorm;
_Static_assert(sizeof(N25LanguageNorm)==896,"Language normalization ABI");
extern const N25LanguageNorm n25_language_norms[25];
extern const N25LanguageNorm *n25_current_norm;
void n25_frontend_runtime_init(void);
#endif
''')
    (dest/'assets/fixed_input.h').write_text('''#ifndef FIXED_INPUT_H
#define FIXED_INPUT_H
#include "../src/text_frontend.h"
#include "../src/language_norms.h"
extern N25TextFeatures n25_text_input;
#define PHONE_COUNT n25_text_input.count
#define input_phones n25_text_input.phones
#define input_pause n25_text_input.pause
#define input_phonetic n25_text_input.phonetic
#define input_context n25_text_input.context
#define LANGUAGE_ID n25_current_norm->language_id
#define PHONETIC_DIM n25_current_norm->phonetic_dim
#define CONTEXT_DIM n25_current_norm->context_dim
#define ACOUSTIC_DIM n25_current_norm->acoustic_dim
#define norm_phonetic_mean n25_current_norm->phonetic_mean
#define norm_phonetic_std n25_current_norm->phonetic_std
#define norm_context_mean n25_current_norm->context_mean
#define norm_context_std n25_current_norm->context_std
#define norm_mean n25_current_norm->mean
#define norm_std n25_current_norm->std
/* The legacy HiFi helpers are unreferenced in this FARGAN-only image. */
static const float resample_filter[1]={0};
#define RESAMPLE_TAPS 1
#define RESAMPLE_REMOVE 0
#endif
''')
    p=dest/'src/text_frontend.h';s=p.read_text().replace('phonetic[N25_TEXT_PHONE_CAP*32]','phonetic[N25_TEXT_PHONE_CAP*36]');p.write_text(s)
    p=dest/'src/text_frontend.c';s=p.read_text().replace('int n25_text_frontend(','int n25_chinese_text_frontend(')
    p.write_text(s)
    digest=','.join(str(x)for x in hashlib.sha256(blob).digest())
    bindings='\n'.join(' {'+str(i)+',"'+l+'"},'for i,l in enumerate(LANGUAGES)if l in SUPPORTED and l!='zh')
    (dest/'src/multilang_frontend.c').write_text('''#include "bench.h"
#include "romfs.h"
#include "storage.h"
#include "language_norms.h"
#include "text_frontend.h"
#include "four_frontend.h"
#include <espeak-ng/speak_lib.h>
#include <string.h>
const N25LanguageNorm *n25_current_norm=n25_language_norms+24;
static N25FourFeatures four_features __attribute__((section(".frontend_workspace"),aligned(16)));
static const unsigned char norm_sha256[32]={'''+digest+'''};
static unsigned frontend_initialized;
int n25_chinese_text_frontend(const char*,N25TextFeatures*);
void n25_frontend_runtime_init(void){
 if(frontend_initialized)return;
 unsigned char digest[32];storage_sha256((const unsigned char*)n25_language_norms,sizeof(n25_language_norms),digest);
 if(memcmp(digest,norm_sha256,32))bench_exit(81);
 size_t mark=bench_arena_mark();void*work=bench_alloc(582144,16);
 if(!work||n25_romfs_set_workspace(work,582144))bench_exit(82);
 n25_romfs_initialize();
 if(espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,0,"/espeak-ng-data",0)<=0)bench_exit(83);
 if(n25_romfs_set_workspace(NULL,0)||bench_arena_rewind(mark))bench_exit(84);
 frontend_initialized=1;bench_metric("all25_normalization_banks_SHA256_passed",25);
}
int n25_text_frontend(const char*text,N25TextFeatures*out){
 if(n25_current_norm->language_id==24)return n25_chinese_text_frontend(text,out);
 static const struct {unsigned id;const char*name;} languages[]={
'''+bindings+'''
 };
 const char*language=NULL;
 for(unsigned i=0;i<sizeof(languages)/sizeof(languages[0]);i++)if(languages[i].id==n25_current_norm->language_id)language=languages[i].name;
 if(!language||!frontend_initialized)return -99;
 size_t mark=bench_arena_mark();void*work=bench_alloc(582144,16);
 if(!work||n25_romfs_set_workspace(work,582144))return -98;
 int rc=n25_four_frontend(language,text,&four_features);
 if(!rc){
  if(four_features.phonetic_dim!=n25_current_norm->phonetic_dim||n25_current_norm->context_dim!=22)rc=-97;
  else{
   unsigned n=four_features.count;out->count=n;
   memcpy(out->phones,four_features.phones,n*4);memcpy(out->pause,four_features.pause,n*4);
   memcpy(out->phonetic,four_features.phonetic,(size_t)n*four_features.phonetic_dim*4);
   memcpy(out->context,four_features.context,(size_t)n*22*4);
  }
 }
 if(n25_romfs_set_workspace(NULL,0)||bench_arena_rewind(mark))return -96;
 return rc;
}
''')
    p=dest/'src/quality.c';s=p.read_text();s=s.replace('static LayerOp adapter_input_op,adapter_norm_op,adapter_head_op,acoustic_head_op;',
        'static LayerOp adapter_input_op,adapter_norm_op,adapter_head_op,acoustic_head_op;\nstatic LayerOp adapter_input_ops[2],acoustic_head_ops[2];')
    s=s.replace('#include "persistent_runtime.inc"','#include "persistent_runtime.inc"\n#include "multilang_driver.inc"');p.write_text(s)
    p=dest/'src/layer_views.inc';s=p.read_text()
    s=s.replace('namefmt(name,sizeof(name),"physical_to_mel.inputs.%d",ACOUSTIC_DIM);adapter_input_op=layer_op(name);',
        'for(unsigned i=0;i<2;i++){unsigned dim=i?39:31;namefmt(name,sizeof(name),"physical_to_mel.inputs.%u",dim);adapter_input_ops[i]=layer_op(name);namefmt(name,sizeof(name),"acoustic.heads.%u",dim);acoustic_head_ops[i]=layer_op(name);}\n adapter_input_op=adapter_input_ops[ACOUSTIC_DIM==39];')
    s=s.replace('namefmt(name,sizeof(name),"acoustic.heads.%d",ACOUSTIC_DIM);acoustic_head_op=layer_op(name);','acoustic_head_op=acoustic_head_ops[ACOUSTIC_DIM==39];');p.write_text(s)
    p=dest/'src/persistent_runtime.inc';s=p.read_text().replace('int main(void){','static int legacy_runtime_main(void){');p.write_text(s)
    rows='\n'.join(' {'+str(c['language_id'])+','+json.dumps(c['text'],ensure_ascii=False)+'},'for c in cases)
    (dest/'src/multilang_driver.inc').write_text('''/* Native text requests. No phone/features/audio fixtures are read by the device. */
__attribute__((noinline)) void tts_begin_language_text_request(unsigned request_id,unsigned language_id,const char*text){
 if(language_id>=25)fail("BAD_LANGUAGE_ID");
 n25_current_norm=n25_language_norms+language_id;
 adapter_input_op=adapter_input_ops[ACOUSTIC_DIM==39];acoustic_head_op=acoustic_head_ops[ACOUSTIC_DIM==39];
 tts_begin_text_request(request_id,text);
}
int main(void){
 tts_boot_start=bench_now();tts_runtime_init();n25_frontend_runtime_init();
 static const struct {unsigned language;const char *text;} cases[]={
'''+rows+'''
 };
 for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){
  unsigned language=cases[i].language;
  tts_begin_language_text_request(language,language,cases[i].text);
  int16_t packet[480];unsigned index=0,count;
  while((count=tts_pull20ms(packet))){
   char name[100];namefmt(name,sizeof(name),"request_%u_packet_%u.pcm",language,index++);
   if(bench_write_file(name,packet,count*2))fail("NATIVE_MULTILANG_PACKET_CAPTURE");
  }
  tts_verify_complete();tts_reset_request();
 }
 bench_metric("native_text_languages_completed",sizeof(cases)/sizeof(cases[0]));
 bench_memory_report();bench_puts("N25_NATIVE_MULTILANG_SUBSET_PASS_NOT_FULL25\\n");return 0;
}
''')
    write(ROOT/'MULTILANG_SOURCE_IDENTITY.json',dict(normalizations=identities,
        blob_sha256=sha(dest/'assets/language_norms.bin'),native_text_cases=cases,
        native_text_language_count=len(cases),full25_native_text=False,
        scope='Native Han subset plus 11 eu_four ordinary NFC language families; numeric/identifier and remaining 13 families pending.'))
    print('Prepared one runtime backend for',len(cases),'native text languages and all25 normalization banks')


if __name__=='__main__':main()
