"""Bound PCM/FARGAN diagnostic memory while retaining optional stream captures."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent


def main():
    src=ROOT/'native/src';snapshot=ROOT/'snapshots/before_bounded_diagnostics';snapshot.mkdir(parents=True,exist_ok=False)
    for name in('quality.c','bwe_stage.c','persistent_runtime.inc'):shutil.copy2(src/name,snapshot/name)
    p=src/'quality.c';s=p.read_text()
    marker='static void fargan_emit(const float *values,unsigned count){'
    start=s.index(marker);end=s.index('\n#include "fargan_stream.inc"',start)
    original=s[start:end]
    replacement='''static unsigned tts_capture_request_id;
static int16_t fg_observer_ring[512];
static void fargan_capture(const char*suffix,const void*data,unsigned bytes){
 char name[96];namefmt(name,sizeof(name),"request_%u_%s",tts_capture_request_id,suffix);
 if(bench_append_file(name,data,bytes))fail("STREAM_CAPTURE_FAILED");
}
#if TTS_RETAIN_WHOLE_DIAGNOSTICS
'''+original+'''\n#else
static void fargan_emit(const float *values,unsigned count){
 if(count>160||fg_samples+count>fg_expected)fail("FARGAN_BOUNDED_CHUNK_SHAPE");
 int16_t pcm[160];
 for(unsigned i=0;i<count;i++){
  float value=values[i];if(!isfinite(value))fail("FARGAN_NONFINITE");
  if(fabsf(value)>fg_peak)fg_peak=fabsf(value);
  if(value>=1.f||value< -1.f)fg_clips++;
  float q=nearbyintf(value*32768.f);if(q>32767)q=32767;if(q< -32768)q=-32768;
  pcm[i]=(int16_t)q;fg_observer_ring[fg_samples++&511]=pcm[i];
 }
 if(!stream_part){first_pcm_ticks=bench_now()-inference_start_ticks;bench_metric("first_pcm_ready_host_ticks",first_pcm_ticks);bench_metric("first_pcm_samples",count);}
 stream_part++;
 fargan_capture("output.pcm",pcm,count*2);fargan_capture("fargan.f32",values,count*4);
 while(!active_found&&fg_scan+320<=fg_samples){
  float energy=0,peak=0;for(unsigned i=0;i<320;i++){float v=(float)fg_observer_ring[(fg_scan+i)&511]/32768.f;energy+=v*v;if(fabsf(v)>peak)peak=fabsf(v);}
  if(energy/320>=.0000316227766f&&peak>=.01f){active_found=1;first_active_ticks=bench_now()-inference_start_ticks;bench_metric("first_active_frame_ready_host_ticks",first_active_ticks);bench_metric("first_active_frame_start_sample",fg_scan);report_first_costs();break;}
  fg_scan+=160;
 }
 bwe_push(pcm,count);stream_samples=fg_samples;
}
#endif
'''
    s=s[:start]+replacement+s[end:];p.write_text(s)
    p=src/'persistent_runtime.inc';s=p.read_text()
    old='memcpy(tts_pcm_ring[tts_ring_tail],pcm,count*2);tts_pcm_lengths[tts_ring_tail]=count;'
    s=s.replace(old,'''#if !TTS_RETAIN_WHOLE_DIAGNOSTICS
 fargan_capture("output24.pcm",pcm,count*2);
#endif
 '''+old)
    s=s.replace('tts_reset_request();tts_request_id=request_id;','tts_reset_request();tts_request_id=request_id;tts_capture_request_id=request_id;')
    old='fg_pcm=bench_alloc(fg_expected*2,16);fg_float=alloc(fg_expected);\n if(!fg_pcm)fail("RUNTIME_DIAGNOSTIC_OUTPUT_ALLOC");'
    assert old in s;s=s.replace(old,'#if TTS_RETAIN_WHOLE_DIAGNOSTICS\n '+old+'\n#else\n fg_pcm=0;fg_float=0;\n#endif')
    old='tts_export("output24.pcm",diagnostic,samples*2);tts_export("output.pcm",fg_pcm,fg_samples*2);\n tts_export("fargan.f32",fg_float,fg_samples*4);tts_export("condition.f32",condition_cache,frames*ACOUSTIC_DIM*4);'
    assert old in s;s=s.replace(old,'''#if TTS_RETAIN_WHOLE_DIAGNOSTICS
 tts_export("output24.pcm",diagnostic,samples*2);tts_export("output.pcm",fg_pcm,fg_samples*2);
 tts_export("fargan.f32",fg_float,fg_samples*4);
#else
 (void)diagnostic;
#endif
 tts_export("condition.f32",condition_cache,frames*ACOUSTIC_DIM*4);''')
    p.write_text(s)
    p=src/'bwe_stage.c';s=p.read_text()
    s=s.replace('static int16_t pending[320],*pcm;','static int16_t pending[320],*pcm;\n#if TTS_RETAIN_WHOLE_DIAGNOSTICS\n#define BWE_PCM_INDEX(n) (n)\n#else\nstatic int16_t bounded_pcm[960];\n#define BWE_PCM_INDEX(n) ((n)%960)\n#endif')
    s=s.replace('pcm+packet_samples','pcm+BWE_PCM_INDEX(packet_samples)')
    s=s.replace('pcm[produced++]=(int16_t)q;','pcm[BWE_PCM_INDEX(produced)]=(int16_t)q;produced++;')
    s=s.replace('pcm[j]/32768','pcm[BWE_PCM_INDEX(j)]/32768')
    old='if(bench_write_file("first24.pcm",pcm,produced*2))fail("FIRST24_WRITE");}'
    assert old in s;s=s.replace(old,'#if TTS_RETAIN_WHOLE_DIAGNOSTICS\n    '+old[:-1]+'\n#endif\n    }')
    old='pcm=bench_alloc(((n*3+1)/2)*2,16);if(!pcm)fail("BWE_OUTPUT_ALLOC");'
    assert old in s;s=s.replace(old,'#if TTS_RETAIN_WHOLE_DIAGNOSTICS\n '+old+'\n#else\n pcm=bounded_pcm;\n#endif')
    old='if(bench_write_file("output24.pcm",pcm,produced*2))fail("BWE_WRITE");'
    assert old in s;s=s.replace(old,'#if TTS_RETAIN_WHOLE_DIAGNOSTICS\n '+old+'\n#endif')
    p.write_text(s)


if __name__=='__main__':main()
