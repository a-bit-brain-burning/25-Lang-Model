#include <stdio.h>
#include "bench.h"
#include "bwe_stage.h"
#include "../assets/bwe_fir.h"
#include "osce.h"
#include <stdint.h>
#include <math.h>
#include <string.h>
extern void snapshot_first_active_costs(void);
extern unsigned long long bwe_quant_groups;
extern unsigned long long bwe_activation_calls[8],bwe_activation_elements[8];
extern unsigned long long bwe_act_guard_vectors,bwe_act_fallback_vectors,bwe_act_fallback_elements;
extern unsigned long long bwe_act_rational_vectors,bwe_act_scalar_divisions,bwe_act_scalar_tail_elements,bwe_act_vector_operations,bwe_act_vector_loads,bwe_act_vector_stores;
extern unsigned long long bwe_exp_guard_vectors,bwe_exp_fast_vectors,bwe_exp_fallback_vectors,bwe_exp_scalar_elements;
extern unsigned long long bwe_exp_vector_arithmetic_operations,bwe_exp_guard_operations,bwe_exp_explicit_vector_loads,bwe_exp_explicit_vector_stores;
static void bwe_activation_report(int first){}

extern unsigned long long bwe_mve_vectors,bwe_scalar_macs,bwe_i8_dots,bwe_i8_inputs;
extern unsigned long long bwe_adaconv_mve_groups,bwe_adaconv_scalar_macs,bwe_adaconv_calls,bwe_adaconv_output4_stores;

extern unsigned long long bwe_adaconv_outer16_groups,bwe_adaconv_outer4_groups,bwe_adaconv_scalar_outputs;
extern unsigned long long bwe_adaconv_x_scalar_loads,bwe_adaconv_shared_x_loads_saved,bwe_adaconv_outer_iterations_saved;
static void bwe_block16_report(int first){}
static OSCEModel model;
static silk_OSCE_BWE_struct state;
#ifndef FAST_BWE_FIR
#define FAST_BWE_FIR 0
#endif
#if FAST_BWE_FIR
static float history[41];
#include "fir_fast.inc"
#else
static double history[41];
#endif
static int16_t pending[320],*pcm;
static unsigned total,provided,held,raw_skip,received,produced,clips,frames,active_boundary,active_found;
static uint64_t started,model_ticks,filter_ticks;
static unsigned finish_calls;
static unsigned packet_samples,packet_count;
static void fail(const char*s){bench_puts(s);bench_puts("\n");bench_exit(70);}
/* Semihost files stand in for the device PCM callback. Publish a packet as
 * soon as its samples exist, rather than waiting for the completed waveform. */
static void publish_packet(unsigned count){
 char name[40]="pcm24_",digits[12];unsigned at=6,n=packet_count,k=0;
 do{digits[k++]=(char)('0'+n%10);n/=10;}while(n);
 while(k)name[at++]=digits[--k];
 memcpy(name+at,".pcm",5);
 if(bench_write_file(name,pcm+packet_samples,count*2))fail("PCM24_PACKET_WRITE");
 packet_samples+=count;packet_count++;
}
static void consume(const int16_t *native){
 for(unsigned i=0;i<960;i++){
  if(raw_skip){raw_skip--;continue;}
  #if FAST_BWE_FIR
  history[received%41]=(float)native[i]/32768.f;
#else
  history[received%41]=(double)native[i]/32768.;
#endif
  received++;
  while(produced<(total*3+1)/2&&2*produced+20<received){
   #if FAST_BWE_FIR
   double value=(double)fast_fir_value(2*produced+20);
#else
   double value=0;for(unsigned tap=0;tap<41;tap++){int index=(int)(2*produced+20)-(int)tap;if(index>=0)value+=fir[tap]*history[(unsigned)index%41];}
#endif
   if(!isfinite(value)){fail("BWE_NONFINITE");}
   double q=nearbyint(value*32768.);
   if(q>32767){q=32767;clips++;}if(q< -32768){q=-32768;clips++;}pcm[produced++]=(int16_t)q;
   if(!active_found&&produced>=480&&(produced-480)%240==0&&produced-480>=active_boundary){
    float power=0,peak=0;for(unsigned j=produced-480;j<produced;j++){float v=(float)pcm[j]/32768;power+=v*v;if(fabsf(v)>peak)peak=fabsf(v);}
    if(power/480>=.0000316227766f&&peak>=.01f){active_found=1;bench_metric("full_chain_first_active_PCM24_host_ticks",bench_now()-started);bench_metric("full_chain_first_active_audio_offset24",produced-480);bench_metric("first_active_BWE_frames",frames+1);bench_metric("first_active_BWE_input_samples",320*(frames+1));bench_metric("BWE_first_mve_vectors",bwe_mve_vectors);bench_metric("BWE_first_adaconv_mve_groups",bwe_adaconv_mve_groups);bench_metric("BWE_first_adaconv_scalar_macs",bwe_adaconv_scalar_macs);bench_metric("BWE_first_adaconv_calls",bwe_adaconv_calls);bench_metric("BWE_first_adaconv_output4_stores",bwe_adaconv_output4_stores);bench_metric("BWE_first_scalar_macs",bwe_scalar_macs);bench_metric("BWE_first_i8_dots",bwe_i8_dots);bench_metric("BWE_first_i8_inputs",bwe_i8_inputs);bench_metric("BWE_first_quant_groups",bwe_quant_groups);bench_metric("first_active_FIR_output_samples",produced);bwe_block16_report(1);snapshot_first_active_costs();
#if FAST_BWE_FIR
    fast_fir_report(1);bwe_activation_report(1);
#endif
    if(bench_write_file("first24.pcm",pcm,produced*2))fail("FIRST24_WRITE");}
   }
   if(produced-packet_samples==480)publish_packet(480);
  }
 }
}
static void frame(void){
 int16_t out[960];uint64_t t=bench_now();osce_bwe(&model,&state,out,pending,320,0);model_ticks+=bench_now()-t;
 t=bench_now();consume(out);filter_ticks+=bench_now()-t;frames++;held=0;
 if(frames==1){bench_metric("full_chain_first_any_PCM24_host_ticks",bench_now()-started);bench_metric("full_chain_first_any_PCM24_samples",produced);}
}
void bwe_start(unsigned n,unsigned nonpause,uint64_t start){
 provided=held=received=produced=clips=frames=active_found=finish_calls=0;
 packet_samples=packet_count=0;model_ticks=filter_ticks=0;memset(history,0,sizeof(history));
 #if FAST_BWE_FIR
 fast_fir_init();
#endif
 bench_metric("BWE_FIR_FP32_MVE",FAST_BWE_FIR);
 total=n;started=start;active_boundary=(nonpause*3+1)/2;raw_skip=34;
 pcm=bench_alloc(((n*3+1)/2)*2,16);if(!pcm)fail("BWE_OUTPUT_ALLOC");
 if(osce_load_models(&model,0,0)){fail("BWE_MODEL_INIT");}
 osce_bwe_reset(&state);
 bench_metric("BWE_context_bytes",sizeof(model)+sizeof(state));
}
void bwe_push(const int16_t*x,unsigned n){
 if(provided+n>total){fail("BWE_TOO_MANY_SAMPLES");}
 provided+=n;
 while(n){unsigned count=n<320-held?n:320-held;memcpy(pending+held,x,count*2);x+=count;n-=count;held+=count;if(held==320)frame();}
}
void bwe_preview(void){
 bench_metric("BWE_preview_input_samples",provided);bench_metric("BWE_preview_output_samples",produced);
 if(!active_found)fail("PREVIEW_HAS_NO_ACTIVE_BWE_SPEECH");
 if(bench_write_file("preview24.pcm",pcm,produced*2))fail("BWE_PREVIEW_WRITE");
}
void bwe_finish(void){
 if(finish_calls++)fail("BWE_FINISH_TWICE");
 bench_metric("PCM24_packets_before_EOS",packet_count);
 bench_metric("BWE_finish_calls",finish_calls);
 if(provided!=total)fail("BWE_MISSING_INPUT");
 if(held){memset(pending+held,0,(320-held)*2);frame();}
 memset(pending,0,sizeof(pending));frame();frame();
 if(produced!=(total*3+1)/2)fail("BWE_OUTPUT_COUNT");
 if(packet_samples<produced)publish_packet(produced-packet_samples);
 bench_metric("PCM24_packet_count",packet_count);bench_metric("PCM24_published_samples",packet_samples);
 if(bench_write_file("output24.pcm",pcm,produced*2))fail("BWE_WRITE");
 bench_metric("full_chain_output24_samples",produced);bench_metric("BWE_frames_processed",frames);bench_metric("BWE_final_mve_vectors",bwe_mve_vectors);bench_metric("BWE_final_adaconv_mve_groups",bwe_adaconv_mve_groups);bench_metric("BWE_final_adaconv_scalar_macs",bwe_adaconv_scalar_macs);bench_metric("BWE_final_adaconv_calls",bwe_adaconv_calls);bench_metric("BWE_final_adaconv_output4_stores",bwe_adaconv_output4_stores);bench_metric("BWE_final_scalar_macs",bwe_scalar_macs);bench_metric("BWE_final_i8_dots",bwe_i8_dots);bench_metric("BWE_final_i8_inputs",bwe_i8_inputs);bench_metric("BWE_final_quant_groups",bwe_quant_groups);bench_metric("BWE_quantizer_clips",clips);bench_metric("BWE_active_detected",active_found);
 #if FAST_BWE_FIR
 fast_fir_report(0);bwe_activation_report(0);bwe_block16_report(0);
#endif
 bench_metric("BWE_model_host_ticks",model_ticks);bench_metric("BWE_FIR_host_ticks",filter_ticks);
}

int bwe_has_active_speech(void){return active_found!=0;}
