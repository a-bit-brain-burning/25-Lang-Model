/* FARGAN4750 integer candidate: original state equations with PTQ MVE dense.
 * Parent provides fargan_emit(float*,count); no teacher prefill or edge padding.
 * The original n(x) noise is retained with PyTorch CPU MT19937/uniform mapping.
 * Native libm/MVE arithmetic is not claimed byte-identical to host PyTorch.
 */
#ifndef FARGAN_SEED
#define FARGAN_SEED 0
#endif
typedef struct {const float*w,*b,*scale;const int8_t*q;unsigned ci,co;} FGLinear;
static FGLinear fg_fdense1,fg_fconv1,fg_fdense2,fg_cond_gain,fg_fwc,fg_fwc_glu,fg_pitch_gain;
static FGLinear fg_gru_ih[3],fg_gru_hh[3],fg_gru_glu[3],fg_skip,fg_skip_glu,fg_signal;
static const float*fg_embedding;
static float fg_zero_bias[480],fg_hidden1[160],fg_hidden2[128],fg_hidden3[128];
static float fg_fwc_state[164],fg_excitation[256],fg_previous_prediction[256];
static float fg_encoded[3][64];static unsigned fg_period[3];
static unsigned fg_received,fg_output_frames,fg_initialized;
static uint64_t fg_subframes,fg_random_draws,fg_weight_ram_bytes,fg_weight_rom_bytes;
static double fg_deemphasis_zi;
/* Original I8 bytes/scales are copied once, inside the request timer.
 * The arena is mapped to QEMU SDRAM, not a claim of hardware TCM placement. */
static const int8_t*fg_i8_ram_codes[N25_TENSOR_COUNT];
static const float*fg_i8_ram_scales[N25_TENSOR_COUNT];
static unsigned fg_storage_bytes(const n25_tensor*t){return t->reserved?((t->dims[0]+7)/8)*((t->elements/t->dims[0]+15)/16)*128:t->elements;}
static unsigned fg_i8_ram_bytes,fg_i8_ram_tensors,fg_i8_borrowed_bytes;
static uint64_t fg_i8_source_copy_bytes,fg_i8_ram_weight_reads,fg_i8_ram_scale_reads,fg_i8_ram_hits,fg_i8_ram_init_ticks;
static void fg_i8_ram_init(void){
 uint64_t start=bench_now();
 extern unsigned char __arena_start[],__arena_end[];
 if((uintptr_t)n25_blob<(uintptr_t)__arena_start||(uintptr_t)n25_blob+MODEL_BYTES>(uintptr_t)__arena_end)fail("FARGAN_MODEL_NOT_IN_ARENA_RAM");
 for(unsigned i=0;i<N25_TENSOR_COUNT;i++){
  const n25_tensor*t=n25_tensors+i;
  if(strncmp(n25_names[i],"generator.",10)||t->dtype!=0)continue;
  if(t->axis||t->rank<2||t->scale_offset%4||t->offset>MODEL_BYTES||fg_storage_bytes(t)>MODEL_BYTES-t->offset||t->scale_offset>MODEL_BYTES||t->dims[0]*4>MODEL_BYTES-t->scale_offset)fail("FARGAN_I8_RAM_LAYOUT");
  fg_i8_ram_codes[i]=(const int8_t*)(n25_blob+t->offset);
  fg_i8_ram_scales[i]=(const float*)(n25_blob+t->scale_offset);
  fg_i8_borrowed_bytes+=fg_storage_bytes(t)+t->dims[0]*4;fg_i8_ram_tensors++;
 }
 if(fg_i8_ram_tensors!=19)fail("FARGAN_I8_RAM_COUNT");
 fg_i8_ram_bytes=0;fg_i8_source_copy_bytes=0;fg_i8_ram_init_ticks=bench_now()-start;
 bench_metric("FARGAN_model_RAM_zero_copy",1);bench_metric("FARGAN_model_RAM_base_address",(uintptr_t)n25_blob);
 bench_metric("FARGAN_borrowed_model_RAM_bytes",fg_i8_borrowed_bytes);
}
static void report_fg_ram_cache(const char*prefix){}

/* MT19937 algorithm: Copyright (C)1997-2002 Makoto Matsumoto and Takuji
 * Nishimura, all rights reserved. Redistribution/use with or without changes
 * is permitted provided this copyright, conditions and disclaimer are kept;
 * binary redistributions reproduce them in their documentation. Contributor
 * names may not endorse derived products without specific written permission.
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT
 * SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING PROCUREMENT
 * OF SUBSTITUTE GOODS/SERVICES, LOSS OF USE/DATA/PROFITS, OR BUSINESS INTERRUPTION),
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER CONTRACT, STRICT
 * LIABILITY OR TORT (INCLUDING NEGLIGENCE), ARISING IN ANY WAY FROM ITS USE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * PyTorch's MT19937RNGEngine.h uses this seed/twist/temper sequence; uniform
 * FP32 keeps the low24 bits, unlike the usual std::uniform_real_distribution.
 */
static uint32_t fg_mt[624];static unsigned fg_mt_next=624;
static uint64_t fg_noise_vector_groups,fg_noise_vector_loads,fg_noise_vector_stores,fg_noise_vector_operations,fg_noise_scalar_lanes,fg_mt_twists;
static void fg_seed(uint64_t seed){
 fg_mt[0]=(uint32_t)seed;
 for(unsigned i=1;i<624;i++)fg_mt[i]=1812433253u*(fg_mt[i-1]^(fg_mt[i-1]>>30))+i;
 fg_mt_next=624;
}
#include "mt_twist_exact.h"
static uint64_t fg_twist_vector_groups,fg_twist_scalar_iterations;
static void fg_twist(void){
 ((void)0);
  mt_twist_exact(fg_mt);((void)0);((void)0);
  fg_mt_next=0;
}
static uint32_t fg_random_u32(void){
 if(fg_mt_next==624)fg_twist();
 uint32_t y=fg_mt[fg_mt_next++];y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;y^=y>>18;
 fg_random_draws++;return y;
}
static float fg_uniform(void){return (float)(fg_random_u32()&0x00ffffffu)*(1.f/16777216.f);}
/* Same MT draw order and per-lane rounded arithmetic; twist stays scalar. */
static void fg_noise(float*x,unsigned n){
 unsigned i=0;
 if(!(fg_mt_next%4))for(;i+4<=n;i+=4){
  if(fg_mt_next==624)fg_twist();
  uint32x4_t y=vld1q_u32(fg_mt+fg_mt_next);fg_mt_next+=4;fg_random_draws+=4;
  y=veorq_u32(y,vshrq_n_u32(y,11));
  y=veorq_u32(y,vandq_u32(vshlq_n_u32(y,7),vdupq_n_u32(0x9d2c5680u)));
  y=veorq_u32(y,vandq_u32(vshlq_n_u32(y,15),vdupq_n_u32(0xefc60000u)));
  y=veorq_u32(y,vshrq_n_u32(y,18));
  float32x4_t uniform=vmulq_n_f32(vcvtq_f32_u32(vandq_u32(y,vdupq_n_u32(0x00ffffffu))),1.f/16777216.f);
  float32x4_t delta=vmulq_n_f32(vsubq_f32(uniform,vdupq_n_f32(.5f)),1.f/127.f);
  float32x4_t v=vaddq_f32(vld1q_f32(x+i),delta);
  v=vpselq_f32(vdupq_n_f32(-1.f),v,vcmpltq_n_f32(v,-1.f));
  v=vpselq_f32(vdupq_n_f32(1.f),v,vcmpgtq_n_f32(v,1.f));
  vst1q_f32(x+i,v);
  ((void)0);((void)0);((void)0);((void)0);
 }
 for(;i<n;i++){
  float delta=(fg_uniform()-.5f)*(1.f/127.f);float v=x[i]+delta;
  x[i]=v< -1.f?-1.f:v>1.f?1.f:v;((void)0);
 }
}
static void fg_noise_report(const char*prefix){}
static const n25_tensor*fg_tryfind(const char*name){
 for(unsigned i=0;i<N25_TENSOR_COUNT;i++)if(!strcmp(name,n25_names[i]))return n25_tensors+i;
 return 0;
}
static const float*fg_weight(const char*name,unsigned expected){
 const n25_tensor*t=fg_tryfind(name);if(!t){fail(name);return 0;}
 if(t->elements!=expected||t->dtype==0)fail("FARGAN_WEIGHT_SHAPE_OR_NEW_QUANTIZATION");
 if(t->dtype==2){if(t->offset%4)fail("FARGAN_FP32_ALIGNMENT");fg_weight_rom_bytes+=(uint64_t)t->elements*4;return(const float*)(n25_blob+t->offset);}
 if(t->dtype!=1)fail("FARGAN_WEIGHT_DTYPE");
 float*p=alloc(t->elements);n25_unpack(n25_blob,t,p);fg_weight_ram_bytes+=(uint64_t)t->elements*4;return p;
}
static void fg_layer(FGLinear*l,const char*name,const char*bias,unsigned ci,unsigned co){
 const n25_tensor*t=fg_tryfind(name);
 if(!t||t->elements!=ci*co||t->dims[0]!=co)fail("FARGAN_INTEGER_SHAPE");
 l->ci=ci;l->co=co;l->b=bias?fg_weight(bias,co):fg_zero_bias;
 if(t->dtype==0){
  if(t->axis||t->scale_offset%4)fail("FARGAN_INTEGER_SCALE_LAYOUT");
  unsigned index=(unsigned)(t-n25_tensors);
  l->q=fg_i8_ram_codes[index];l->scale=fg_i8_ram_scales[index];
  if(!l->q||!l->scale)fail("FARGAN_I8_RAM_MISSING_WEIGHT");
  fg_weight_rom_bytes+=(uint64_t)fg_storage_bytes(t)+4*co;
 }else l->w=fg_weight(name,ci*co);
}
static void fg_dense(const FGLinear*l,const float*x,float*y){
 if(l->q){
  uint64_t before=fast_i8_stats.weight_vector_loads;
  if(fast_i8_dense(x,l->q,l->scale,l->b,y,l->ci,l->co,0))fail("FARGAN_INTEGER_DENSE_INPUT");
  uint64_t reads=(fast_i8_stats.weight_vector_loads-before)*16;((void)0);
  if(reads)((void)0);
  ((void)0);
 }
 else gemm4(x,l->w,l->b,y,l->ci,l->co,0);
}
static void fg_tanh(float*x,unsigned n){unsigned i=0;for(;i+4<=n;i+=4)vst1q_f32(x+i,fast_nonlinear4(vld1q_f32(x+i),1));for(;i<n;i++)x[i]=tanhf(x[i]);}
static void fg_glu(const FGLinear*gate,float*x,unsigned n){
 float gates[192];if(n>192||gate->ci!=n||gate->co!=n)fail("FARGAN_GLU_SHAPE");
 fg_dense(gate,x,gates);fusion_glu(x,gates,n);
}
static void fg_gru(unsigned which,const float*x,float*hidden,unsigned width){
 float ig[480],hg[480];fg_dense(&fg_gru_ih[which],x,ig);fg_dense(&fg_gru_hh[which],hidden,hg);
 fusion_fg_gru(ig,hg,hidden,width);
}
static void fg_subframe(const float*condition,unsigned period,float*out){
 float cond[80],pred[44],previous[40],tmp[164],fwc_input[328],fwc[192],pitch_gain[4];
 float gru_input[272],gru1[160],gru2[128],gru3[128],skip_input[688],skip[128],gain_value[1];
 memcpy(cond,condition,sizeof(cond));fg_noise(cond,80);fg_dense(&fg_cond_gain,cond,gain_value);
 float gain=expf(gain_value[0]),denominator=1e-5f+gain;
 if(!isfinite(gain))fail("FARGAN_NONFINITE_GAIN");
 for(unsigned i=0;i<44;i++){
  int at=256-(int)period+(int)i-2;if(at>=256)at-=(int)period;
  if(at<0||at>=256)fail("FARGAN_PITCH_GATHER_RANGE");
  pred[i]=fg_excitation[at]/denominator;
 }
 fg_noise(pred,44);
 for(unsigned i=0;i<40;i++)previous[i]=fg_excitation[216+i]/denominator;
 fg_noise(previous,40);
 memcpy(tmp,cond,80*4);memcpy(tmp+80,pred,44*4);memcpy(tmp+124,previous,40*4);
 memcpy(fwc_input,fg_fwc_state,164*4);memcpy(fwc_input+164,tmp,164*4);
 fg_dense(&fg_fwc,fwc_input,fwc);fg_tanh(fwc,192);fg_glu(&fg_fwc_glu,fwc,192);memcpy(fg_fwc_state,tmp,164*4);fg_noise(fwc,192);
 fg_dense(&fg_pitch_gain,fwc,pitch_gain);for(unsigned i=0;i<4;i++)pitch_gain[i]=sigmoid(pitch_gain[i]);
 memcpy(gru_input,fwc,192*4);for(unsigned i=0;i<40;i++)gru_input[192+i]=pitch_gain[0]*pred[i+2];memcpy(gru_input+232,previous,40*4);
 fg_gru(0,gru_input,fg_hidden1,160);memcpy(gru1,fg_hidden1,160*4);fg_noise(gru1,160);fg_glu(&fg_gru_glu[0],gru1,160);fg_noise(gru1,160);
 memcpy(gru_input,gru1,160*4);for(unsigned i=0;i<40;i++)gru_input[160+i]=pitch_gain[1]*pred[i+2];memcpy(gru_input+200,previous,40*4);
 fg_gru(1,gru_input,fg_hidden2,128);memcpy(gru2,fg_hidden2,128*4);fg_noise(gru2,128);fg_glu(&fg_gru_glu[1],gru2,128);fg_noise(gru2,128);
 memcpy(gru_input,gru2,128*4);for(unsigned i=0;i<40;i++)gru_input[128+i]=pitch_gain[2]*pred[i+2];memcpy(gru_input+168,previous,40*4);
 fg_gru(2,gru_input,fg_hidden3,128);memcpy(gru3,fg_hidden3,128*4);fg_noise(gru3,128);fg_glu(&fg_gru_glu[2],gru3,128);fg_noise(gru3,128);
 memcpy(skip_input,gru1,160*4);memcpy(skip_input+160,gru2,128*4);memcpy(skip_input+288,gru3,128*4);memcpy(skip_input+416,fwc,192*4);
 for(unsigned i=0;i<40;i++){skip_input[608+i]=pitch_gain[3]*pred[i+2];}memcpy(skip_input+648,previous,40*4);
 fg_dense(&fg_skip,skip_input,skip);fg_tanh(skip,128);fg_noise(skip,128);fg_glu(&fg_skip_glu,skip,128);
 fg_dense(&fg_signal,skip,out);fg_tanh(out,40);for(unsigned i=0;i<40;i++)out[i]*=gain;
 memmove(fg_excitation,fg_excitation+40,216*4);memcpy(fg_excitation+216,out,40*4);
 memmove(fg_previous_prediction,fg_previous_prediction+40,216*4);memcpy(fg_previous_prediction+216,pred+2,40*4);fg_subframes++;
}
static void fargan_stream_init(void){
 if(fg_initialized++){fail("FARGAN_STREAM_INIT_TWICE");}fg_seed((uint64_t)FARGAN_SEED);
 fg_i8_ram_init();
 fg_embedding=fg_weight("generator.cond_net.pembed.weight",224*12);
 fg_layer(&fg_fdense1,"generator.cond_net.fdense1.weight",0,32,64);
 fg_layer(&fg_fconv1,"generator.cond_net.fconv1.weight",0,3*64,128);
 fg_layer(&fg_fdense2,"generator.cond_net.fdense2.weight",0,128,320);
 fg_layer(&fg_cond_gain,"generator.sig_net.cond_gain_dense.weight","generator.sig_net.cond_gain_dense.bias",80,1);
 fg_layer(&fg_fwc,"generator.sig_net.fwc0.conv.weight",0,328,192);
 fg_layer(&fg_fwc_glu,"generator.sig_net.fwc0.glu.gate.weight",0,192,192);
 fg_layer(&fg_pitch_gain,"generator.sig_net.gain_dense_out.weight","generator.sig_net.gain_dense_out.bias",192,4);
 char name[160];unsigned widths[]={160,128,128},inputs[]={272,240,208};
 for(unsigned i=0;i<3;i++){
  namefmt(name,sizeof(name),"generator.sig_net.gru%u.weight_ih",i+1);fg_layer(&fg_gru_ih[i],name,0,inputs[i],3*widths[i]);
  namefmt(name,sizeof(name),"generator.sig_net.gru%u.weight_hh",i+1);fg_layer(&fg_gru_hh[i],name,0,widths[i],3*widths[i]);
  namefmt(name,sizeof(name),"generator.sig_net.gru%u_glu.gate.weight",i+1);fg_layer(&fg_gru_glu[i],name,0,widths[i],widths[i]);
 }
 fg_layer(&fg_skip,"generator.sig_net.skip_dense.weight",0,688,128);
 fg_layer(&fg_skip_glu,"generator.sig_net.skip_glu.gate.weight",0,128,128);
 fg_layer(&fg_signal,"generator.sig_net.sig_dense_out.weight",0,128,40);
 bench_metric("FARGAN_seed",FARGAN_SEED);bench_metric("FARGAN_weight_unpack_ram_bytes",fg_weight_ram_bytes);bench_metric("FARGAN_weight_direct_rom_bytes",fg_weight_rom_bytes);
 bench_metric("FARGAN_matrix_weight_bits",8);bench_metric("FARGAN_activation_split_I16",FAST_I8_ACT_SPLIT16);
 bench_metric("FARGAN_noise_enabled",1);bench_metric("FARGAN_state_float32",1);bench_metric("FARGAN_teacher_prefill_frames",0);
}
static void fargan_frame(const float features[20],unsigned index){
 if(!fg_initialized||index!=fg_received)fail("FARGAN_FEATURE_ORDER");
 for(unsigned i=0;i<20;i++)if(!isfinite(features[i]))fail("FARGAN_NONFINITE_FEATURE");
 fg_received++;if(index<2)return;
 float period_value=256.f/exp2f(features[18]+1.5f);if(period_value<32)period_value=32;if(period_value>255)period_value=255;
 unsigned period=(unsigned)nearbyintf(period_value),slot=(index-2)%3;fg_period[slot]=period;
 float cond_input[32];memcpy(cond_input,features,20*4);memcpy(cond_input+20,fg_embedding+(period-32)*12,12*4);
 fg_dense(&fg_fdense1,cond_input,fg_encoded[slot]);fg_tanh(fg_encoded[slot],64);
 if(index<4)return;
 float window[192],conv[128],conditioning[320],raw[160],pcm[160];
 for(unsigned i=0;i<3;i++)memcpy(window+i*64,fg_encoded[(index-4+i)%3],64*4);
 fg_dense(&fg_fconv1,window,conv);fg_tanh(conv,128);fg_dense(&fg_fdense2,conv,conditioning);fg_tanh(conditioning,320);
 unsigned center_period=fg_period[(index-3)%3];
 for(unsigned sub=0;sub<4;sub++)fg_subframe(conditioning+sub*80,center_period,raw+sub*40);
 for(unsigned i=0;i<160;i++){
  double y=(double)raw[i]+fg_deemphasis_zi;fg_deemphasis_zi=.85*y;pcm[i]=(float)y;
  if(!isfinite(pcm[i]))fail("FARGAN_NONFINITE_PCM");
 }
 fg_output_frames++;fargan_emit(pcm,160);
}
static void fargan_stream_report(void){if(fg_received<5||fg_output_frames!=fg_received-4)fail("FARGAN_VALID_CENTER_COUNT");}

static void fargan_reset_request(void){
 if(!fg_initialized)fail("FARGAN_MODEL_NOT_READY");
 memset(fg_hidden1,0,sizeof(fg_hidden1));memset(fg_hidden2,0,sizeof(fg_hidden2));memset(fg_hidden3,0,sizeof(fg_hidden3));
 memset(fg_fwc_state,0,sizeof(fg_fwc_state));memset(fg_excitation,0,sizeof(fg_excitation));memset(fg_previous_prediction,0,sizeof(fg_previous_prediction));
 memset(fg_encoded,0,sizeof(fg_encoded));memset(fg_period,0,sizeof(fg_period));
 fg_noise_vector_groups=fg_noise_vector_loads=fg_noise_vector_stores=fg_noise_vector_operations=fg_noise_scalar_lanes=fg_mt_twists=0;
 fg_twist_vector_groups=fg_twist_scalar_iterations=0;
 fg_received=fg_output_frames=0;fg_subframes=fg_random_draws=0;fg_deemphasis_zi=0;
 fg_i8_ram_hits=fg_i8_ram_weight_reads=fg_i8_ram_scale_reads=0;fg_seed((uint64_t)FARGAN_SEED);
}
