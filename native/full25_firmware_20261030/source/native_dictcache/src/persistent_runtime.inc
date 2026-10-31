int fep_lexicon_prewarm_es(void);
/* Model lifetime is separate from request arena and 20ms PCM queue. */
#include "stream_runtime.h"
#define TTS_RING_PACKETS 4
static int16_t tts_pcm_ring[TTS_RING_PACKETS][480];
static unsigned tts_pcm_lengths[TTS_RING_PACKETS],tts_ring_head,tts_ring_tail,tts_ring_count,tts_ring_peak;
static unsigned tts_initialized,tts_model_init_calls,tts_request_id,tts_request_active,tts_request_done;
static unsigned tts_next_feature,tts_prefetch_end,tts_first_features,tts_pulled_packets,tts_pulled_samples;
static unsigned tts_prefetch_first_batches,tts_prefetch_pre_active_batches,tts_prefetch_post_active_batches;
static unsigned tts_effective_packet_delivered;
__attribute__((noinline)) void tts_first_effective_packet_delivered(void){__asm__ volatile("" ::: "memory");}
static uint64_t tts_boot_start;
static uint64_t tts_model_init_ticks,tts_first_pull_ticks,tts_request_start,tts_request_finished_ticks;
static size_t tts_model_mark;

static void tts_metric(const char*field,uint64_t value){
 char name[100];namefmt(name,sizeof(name),"runtime_request_%u_%s",tts_request_id,field);bench_metric(name,value);
}
static void tts_publish(const int16_t*pcm,unsigned count){
 if(!tts_request_active||!count||count>480||tts_ring_count==TTS_RING_PACKETS)fail("RUNTIME_PCM_RING_OVERFLOW");
 #if !TTS_RETAIN_WHOLE_DIAGNOSTICS
 fargan_capture("output24.pcm",pcm,count*2);
#endif
 memcpy(tts_pcm_ring[tts_ring_tail],pcm,count*2);tts_pcm_lengths[tts_ring_tail]=count;
 tts_ring_tail=(tts_ring_tail+1)%TTS_RING_PACKETS;tts_ring_count++;
 if(tts_ring_count>tts_ring_peak)tts_ring_peak=tts_ring_count;
}
static void tts_clear_profile(void){
 cost_mac=cost_vfma=cost_vload=cost_iterations=cost_reductions=cost_scalar_mac=cost_pack_read=cost_pack_write=cost_dequant=cost_activation_vec=cost_activation_scalar=cost_norm_cells=0;
 cost_exp=cost_erf=cost_tanh=cost_sqrt=0;
 fast_sigmoid_lut=fast_tanh_lut=fast_erf_lut=fast_sigmoid_fallback=fast_tanh_fallback=fast_erf_fallback=0;
 fast_tanh_saturated=fast_erf_saturated=fast_lut_vector_groups=fast_lut_vector_gathers=0;
 memset(&fast_i8_stats,0,sizeof(fast_i8_stats));
 memset(&fast_i8_pair_stats,0,sizeof(fast_i8_pair_stats));
 memset(&fast_pair_matrix_stats,0,sizeof(fast_pair_matrix_stats));
 memset(&depthwise_stats,0,sizeof(depthwise_stats));
 fast_lut_vector_validation_groups=fast_lut_vector_validation_fallbacks=0;
 fast_lut_inline_invocations=fast_lut_fallback_calls=fast_lut_fast_vector_operations=0;
 fast_lut_validation_vector_operations=fast_lut_validation_reductions=fast_lut_table_words=0;
 fusion_fg_gru_groups=fusion_acoustic_gru_groups=fusion_glu_groups=fusion_vector_loads=fusion_vector_stores=fusion_vector_operations=0;
 gru_ih_pairs=gru_ih_singles=gru_hh_frames=linear_pair_calls=linear_single_rows=0;
 generator_ticks=adapter_ticks=acoustic_ticks=first_pcm_ticks=first_nonpause_ticks=first_active_ticks=0;
 acoustic_gru_cache_uses=0;
}
void tts_runtime_init(void){
 if(tts_initialized)return;
 uint64_t start=bench_now();
 storage_initialize(start);
 if(fep_lexicon_prewarm_es())fail("SPANISH_DICTIONARY_PREWARM");
 fast_tables_prepare();
 uint64_t views_start=bench_now();tensor_views_prepare();layer_views_prepare();
 bench_metric("runtime_tensor_views_prepare_host_ticks",bench_now()-views_start);
 bench_metric("runtime_tensor_views_init_calls",1);
 uint64_t prewarm_start=bench_now();int prewarm_rc=native_zh_prewarm();
 bench_metric("runtime_frontend_dict_prewarm_host_ticks",bench_now()-prewarm_start);
 bench_metric("runtime_frontend_dict_prewarm_status",prewarm_rc==0);
 acoustic_gru_cache_init();fargan_stream_init();bwe_model_init();
 tts_model_mark=bench_arena_mark();tts_model_init_ticks=bench_now()-start;
 tts_initialized=1;tts_model_init_calls++;bwe_set_pcm_callback(tts_publish);
 bench_metric("runtime_model_init_calls",tts_model_init_calls);
 bench_metric("runtime_model_init_host_ticks",tts_model_init_ticks);
 bench_metric("runtime_resident_weight_arena_bytes",tts_model_mark);
 bench_metric("runtime_PCM_ring_capacity_bytes",sizeof(tts_pcm_ring));
 bench_metric("runtime_prepared_features_only",0);bench_metric("runtime_native_Han_frontend",1);
}
void tts_reset_request(void){
 if(!tts_initialized)fail("RUNTIME_NOT_INITIALIZED");
 bwe_cancel();if(bwe_has_active_speech())fail("RUNTIME_BWE_ACTIVE_NOT_RESET");rewind_to(tts_model_mark);
 tts_ring_head=tts_ring_tail=tts_ring_count=0;tts_request_active=tts_request_done=0;
 condition_cache=encoded_cache=prosody_cache=0;fg_pcm=0;fg_float=0;
 for(unsigned i=0;i<5;i++)decoder_history[i]=0;
 if(bench_arena_mark()!=tts_model_mark)fail("RUNTIME_ARENA_NOT_RESET");
}
void tts_cancel_request(void){
 if(!tts_request_active||tts_request_done)fail("RUNTIME_CANCEL_STATE");
 tts_metric("cancelled_packets",tts_pulled_packets);tts_metric("cancelled_samples",tts_pulled_samples);
 if(bwe_finish_count())fail("RUNTIME_CANCEL_CALLED_EOS");
 tts_reset_request();
 tts_metric("cancel_arena_mark",bench_arena_mark());
}
void tts_begin_text_request(unsigned request_id,const char*text){
 if(!tts_initialized)tts_runtime_init();
 tts_reset_request();tts_request_id=request_id;tts_capture_request_id=request_id;tts_clear_profile();fargan_reset_request();
 frames=mel_frames=decoded_frames=stream_part=stream_samples=active_found=active_scan_sample=0;
 first_nonpause_sample=0;memset(gru_hidden,0,sizeof(gru_hidden));
 fg_samples=fg_expected=fg_scan=fg_clips=0;fg_peak=0;
 tts_next_feature=tts_prefetch_end=tts_ring_peak=tts_pulled_packets=tts_pulled_samples=0;
 tts_prefetch_first_batches=tts_prefetch_pre_active_batches=tts_prefetch_post_active_batches=0;
 tts_first_pull_ticks=tts_request_finished_ticks=0;tts_request_active=1;
 tts_effective_packet_delivered=0;
 tts_request_start=inference_start_ticks=bench_now();
 uint64_t start=bench_now();int front_rc=n25_text_frontend(text,&n25_text_input);tts_metric("frontend_host_ticks",bench_now()-start);if(front_rc)fail("RUNTIME_NATIVE_TEXT_REJECTED");start=bench_now();acoustic_begin();acoustic_ticks=bench_now()-start;
 if(frames<5)fail("RUNTIME_SHORT_FARGAN_UNIT");
 mel_frames=frames;fg_expected=(frames-4)*160;
 #if TTS_RETAIN_WHOLE_DIAGNOSTICS
 fg_pcm=bench_alloc(fg_expected*2,16);fg_float=alloc(fg_expected);
 if(!fg_pcm)fail("RUNTIME_DIAGNOSTIC_OUTPUT_ALLOC");
#else
 fg_pcm=0;fg_float=0;
#endif
 unsigned nonpause=first_nonpause_sample>400?first_nonpause_sample-400:0;fg_scan=nonpause;
 adapter_stream_init();bwe_start(fg_expected,nonpause,tts_request_start);
 unsigned boundary24=(nonpause*3+1)/2;
 unsigned needed24=((boundary24+239)/240)*240+480;
 unsigned needed16=((needed24+27+479)/480)*320;
 tts_first_features=(needed16+159)/160+4;if(tts_first_features>frames)tts_first_features=frames;
 tts_metric("BWE_active_at_begin",bwe_has_active_speech());
 if(bwe_has_active_speech())fail("RUNTIME_BWE_ACTIVE_AT_BEGIN");
 tts_metric("begin_model_arena_mark",tts_model_mark);
 tts_metric("physical_frames",frames);tts_metric("first_prefetch_features",tts_first_features);
}
void tts_begin_request(unsigned request_id){tts_begin_text_request(request_id,N25_REQUEST_TEXT);}
void tts_finish_request(void){
 if(!tts_request_active||tts_request_done||tts_next_feature!=frames)fail("RUNTIME_FINISH_STATE");
 if(fg_samples!=fg_expected)fail("RUNTIME_FARGAN_LENGTH");
 ensure_condition(frames);bwe_finish();tts_request_done=1;
 tts_request_finished_ticks=bench_now()-tts_request_start;
}
static void tts_step_until_packet(void){
 while(!tts_ring_count&&!tts_request_done){
  if(tts_next_feature==frames){tts_finish_request();break;}
  if(tts_next_feature==tts_prefetch_end){
   unsigned end;
   if(!tts_next_feature){end=tts_first_features;tts_prefetch_first_batches++;}
   /* Keep the two-frame lookahead until an effective complete packet is
    * actually delivered. A partial overlapping BWE window is not sufficient. */
   else if(tts_effective_packet_delivered){end=tts_next_feature+16;tts_prefetch_post_active_batches++;}
   else{end=tts_next_feature+2;tts_prefetch_pre_active_batches++;}
   if(end>frames)end=frames;
   (void)adapter_fargan_frame(end-1);tts_prefetch_end=end;
  }
  fargan_frame(adapter_fargan_frame(tts_next_feature),tts_next_feature);tts_next_feature++;
 }
}
unsigned tts_pull20ms(int16_t output[480]){
 if(!tts_request_active)fail("RUNTIME_PULL_WITHOUT_REQUEST");
 if(!tts_ring_count)tts_step_until_packet();
 if(!tts_ring_count)return 0;
 unsigned count=tts_pcm_lengths[tts_ring_head];memcpy(output,tts_pcm_ring[tts_ring_head],count*2);
 tts_ring_head=(tts_ring_head+1)%TTS_RING_PACKETS;tts_ring_count--;
 tts_pulled_packets++;tts_pulled_samples+=count;
 if(!tts_first_pull_ticks)tts_first_pull_ticks=bench_now()-tts_request_start;
 if(!tts_effective_packet_delivered&&count==480&&bwe_has_active_speech()){
  float power=0,peak=0;
  for(unsigned i=0;i<480;i++){float v=(float)output[i]/32768.f;power+=v*v;if(fabsf(v)>peak)peak=fabsf(v);}
  if(power/480>=.0000316227766f&&peak>=.01f){tts_effective_packet_delivered=1;tts_first_effective_packet_delivered();}
 }
 return count;
}
static void tts_export(const char*suffix,const void*data,unsigned bytes){
 char name[100];namefmt(name,sizeof(name),"request_%u_%s",tts_request_id,suffix);
 if(bench_write_file(name,data,bytes))fail("RUNTIME_EXPORT_FAILED");
}
static void tts_verify_complete(void){
 if(!tts_request_done||tts_ring_count||tts_pulled_samples!=(frames-4)*240||bwe_finish_count()!=1)fail("RUNTIME_COMPLETE_STATE");
 if(!active_found||!bwe_first_active_ticks())fail("RUNTIME_NO_EFFECTIVE_SPEECH");
 if(fg_received!=frames||fg_output_frames!=frames-4)fail("RUNTIME_MODEL_PROGRESS");
 unsigned samples;const int16_t*diagnostic=bwe_diagnostic_pcm(&samples);
 if(samples!=tts_pulled_samples)fail("RUNTIME_RING_DIAGNOSTIC_LENGTH");
 tts_export("features.bin",&n25_text_input,sizeof(n25_text_input));
 #if TTS_RETAIN_WHOLE_DIAGNOSTICS
 tts_export("output24.pcm",diagnostic,samples*2);tts_export("output.pcm",fg_pcm,fg_samples*2);
 tts_export("fargan.f32",fg_float,fg_samples*4);
#else
 (void)diagnostic;
#endif
 tts_export("condition.f32",condition_cache,frames*ACOUSTIC_DIM*4);
 tts_export("durations.u32",predicted_durations,PHONE_COUNT*4);tts_export("rng.u32",fg_mt,sizeof(fg_mt));
 tts_export("gru.f32",gru_hidden,sizeof(gru_hidden));
 depthwise_report("runtime_total");report_front_i8("runtime_total_i8");fusion_report("runtime_total");fg_noise_report("runtime_total");fast_math_report("runtime_total");
 tts_metric("first_packet_host_ticks",tts_first_pull_ticks);
 tts_metric("first_effective_host_ticks",bwe_first_active_ticks());
 tts_metric("boot_to_first_effective_host_ticks",tts_request_start-tts_boot_start+bwe_first_active_ticks());
 tts_metric("request_finished_host_ticks",tts_request_finished_ticks);
 tts_metric("packets",tts_pulled_packets);tts_metric("samples24",tts_pulled_samples);
 tts_metric("PCM_ring_peak_packets",tts_ring_peak);tts_metric("request_arena_end",bench_arena_mark());
 tts_metric("arena_peak_bytes",bench_arena_peak());tts_metric("RNG_draws",fg_random_draws);tts_metric("RNG_next",fg_mt_next);
 tts_metric("EOS_calls",bwe_finish_count());tts_metric("model_init_calls",tts_model_init_calls);
 tts_metric("weight_ROM_copy_bytes_lifetime",fg_i8_source_copy_bytes);tts_metric("clips16",fg_clips);
 tts_metric("prefetch_first_batches",tts_prefetch_first_batches);
 tts_metric("prefetch_pre_active_batches",tts_prefetch_pre_active_batches);
 tts_metric("prefetch_post_active_batches",tts_prefetch_post_active_batches);
 tts_metric("prefetch_pre_active_frames",2);tts_metric("prefetch_post_active_frames",16);
 tts_metric("BWE_active_at_complete",bwe_has_active_speech());
}
static void tts_run_complete(unsigned id){
 tts_begin_request(id);int16_t packet[480];unsigned index=0,count;
 while((count=tts_pull20ms(packet))){
  char name[100];namefmt(name,sizeof(name),"request_%u_packet_%u.pcm",id,index++);
  if(bench_write_file(name,packet,count*2))fail("RUNTIME_PULL_PACKET_EXPORT");
 }
 tts_verify_complete();
}
static int legacy_runtime_main(void){
 tts_boot_start=bench_now();
 bench_puts("NEW25_PERSISTENT_RUNTIME_BEGIN\n");bench_metric("tick_frequency_hz",bench_tick_frequency());
 tts_runtime_init();
 tts_run_complete(0);tts_run_complete(1);
 tts_begin_request(2);int16_t packet[480];
 for(unsigned i=0;i<70;i++)if(tts_pull20ms(packet)!=480)fail("RUNTIME_EARLY_CANCEL_POINT");
 tts_cancel_request();tts_run_complete(3);
 tts_reset_request();
 bench_metric("runtime_final_arena_mark",bench_arena_mark());
 bench_metric("runtime_expected_model_mark",tts_model_mark);bench_metric("runtime_model_init_calls_final",tts_model_init_calls);
 bench_metric("runtime_full_requests",3);bench_metric("runtime_cancel_requests",1);
 bench_memory_report();bench_puts("NEW25_PERSISTENT_RUNTIME_PASS\n");return 0;
}
