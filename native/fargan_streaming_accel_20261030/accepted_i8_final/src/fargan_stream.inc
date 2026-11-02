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
static unsigned char*fg_i8_ram_blob;
static unsigned fg_i8_ram_bytes,fg_i8_ram_tensors;
static uint64_t fg_i8_source_copy_bytes,fg_i8_ram_weight_reads,fg_i8_ram_scale_reads,fg_i8_ram_hits,fg_i8_ram_init_ticks;
static unsigned fg_cache_align(unsigned n){return(n+15u)&~15u;}
static void fg_i8_ram_init(void){
 uint64_t start=bench_now();unsigned bytes=0;
 for(unsigned i=0;i<N25_TENSOR_COUNT;i++){
  const n25_tensor*t=n25_tensors+i;
  if(strncmp(n25_names[i],"generator.",10)||t->dtype!=0)continue;
  if(t->axis||t->rank<2)fail("FARGAN_I8_RAM_LAYOUT");
  bytes+=fg_cache_align(t->elements)+fg_cache_align(t->dims[0]*4);
 }
 fg_i8_ram_blob=bench_alloc(bytes,16);if(!fg_i8_ram_blob)fail("FARGAN_I8_RAM_ALLOC");
 memset(fg_i8_ram_blob,0,bytes);unsigned at=0;
 for(unsigned i=0;i<N25_TENSOR_COUNT;i++){
  const n25_tensor*t=n25_tensors+i;
  if(strncmp(n25_names[i],"generator.",10)||t->dtype!=0)continue;
  fg_i8_ram_codes[i]=(const int8_t*)(fg_i8_ram_blob+at);
  memcpy(fg_i8_ram_blob+at,n25_blob+t->offset,t->elements);at+=fg_cache_align(t->elements);
  fg_i8_ram_scales[i]=(const float*)(fg_i8_ram_blob+at);
  memcpy(fg_i8_ram_blob+at,n25_blob+t->scale_offset,t->dims[0]*4);at+=fg_cache_align(t->dims[0]*4);
  fg_i8_source_copy_bytes+=(uint64_t)t->elements+t->dims[0]*4;fg_i8_ram_tensors++;
 }
 if(at!=bytes||fg_i8_ram_tensors!=19)fail("FARGAN_I8_RAM_COUNT");
 fg_i8_ram_bytes=bytes;fg_i8_ram_init_ticks=bench_now()-start;
}
static void report_fg_ram_cache(const char*prefix){
 char name[100];
#define FG_CACHE_METRIC(label,value) namefmt(name,sizeof(name),"%s_%s",prefix,label);bench_metric(name,value)
 FG_CACHE_METRIC("bytes",fg_i8_ram_bytes);FG_CACHE_METRIC("tensors",fg_i8_ram_tensors);
 FG_CACHE_METRIC("source_ROM_copy_bytes",fg_i8_source_copy_bytes);
 FG_CACHE_METRIC("runtime_I8_source_ROM_bytes",0);
 FG_CACHE_METRIC("runtime_RAM_weight_bytes",fg_i8_ram_weight_reads);FG_CACHE_METRIC("runtime_RAM_scale_bytes",fg_i8_ram_scale_reads);
 FG_CACHE_METRIC("matrix_hits",fg_i8_ram_hits);FG_CACHE_METRIC("init_host_ticks",fg_i8_ram_init_ticks);
#undef FG_CACHE_METRIC
}

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
static void fg_seed(uint64_t seed){
 fg_mt[0]=(uint32_t)seed;
 for(unsigned i=1;i<624;i++)fg_mt[i]=1812433253u*(fg_mt[i-1]^(fg_mt[i-1]>>30))+i;
 fg_mt_next=624;
}
static uint32_t fg_random_u32(void){
 if(fg_mt_next==624){
  for(unsigned i=0;i<624;i++){
   uint32_t y=(fg_mt[i]&0x80000000u)|(fg_mt[(i+1)%624]&0x7fffffffu);
   fg_mt[i]=fg_mt[(i+397)%624]^(y>>1)^((y&1u)?0x9908b0dfu:0u);
  }
  fg_mt_next=0;
 }
 uint32_t y=fg_mt[fg_mt_next++];y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;y^=y>>18;
 fg_random_draws++;return y;
}
static float fg_uniform(void){return (float)(fg_random_u32()&0x00ffffffu)*(1.f/16777216.f);}
static void fg_noise(float*x,unsigned n){
 for(unsigned i=0;i<n;i++){
  float delta=(fg_uniform()-.5f)*(1.f/127.f);float v=x[i]+delta;
  x[i]=v< -1.f?-1.f:v>1.f?1.f:v;
 }
}
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
  fg_weight_rom_bytes+=(uint64_t)t->elements+4*co;
 }else l->w=fg_weight(name,ci*co);
}
static void fg_dense(const FGLinear*l,const float*x,float*y){
 if(l->q){
  uint64_t before=fast_i8_stats.logical_mac;
  if(fast_i8_dense(x,l->q,l->scale,l->b,y,l->ci,l->co,0))fail("FARGAN_INTEGER_DENSE_INPUT");
  uint64_t reads=fast_i8_stats.logical_mac-before;fg_i8_ram_weight_reads+=reads;
  if(reads)fg_i8_ram_scale_reads+=(uint64_t)l->co*4;
  fg_i8_ram_hits++;
 }
 else gemm4(x,l->w,l->b,y,l->ci,l->co,0);
}
static void fg_tanh(float*x,unsigned n){unsigned i=0;for(;i+4<=n;i+=4)vst1q_f32(x+i,fast_nonlinear4(vld1q_f32(x+i),1));for(;i<n;i++)x[i]=tanhf(x[i]);}
static void fg_glu(const FGLinear*gate,float*x,unsigned n){
 float gates[192];if(n>192||gate->ci!=n||gate->co!=n)fail("FARGAN_GLU_SHAPE");
 fg_dense(gate,x,gates);for(unsigned i=0;i<n;i++)x[i]*=sigmoid(gates[i]);
}
static void fg_gru(unsigned which,const float*x,float*hidden,unsigned width){
 float ig[480],hg[480];fg_dense(&fg_gru_ih[which],x,ig);fg_dense(&fg_gru_hh[which],hidden,hg);
 for(unsigned j=0;j<width;j++){
  float r=sigmoid(ig[j]+hg[j]),z=sigmoid(ig[width+j]+hg[width+j]);
  float candidate=tanhf(ig[2*width+j]+r*hg[2*width+j]);
  hidden[j]=(hidden[j]-candidate)*z+candidate;
 }
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
static void fargan_stream_report(void){
 report_fg_ram_cache("total_FARGAN_I8_cache");
 if(bench_write_file("fargan_i8_cache.bin",fg_i8_ram_blob,fg_i8_ram_bytes))fail("FARGAN_I8_RAM_EXPORT");
 if(fg_received<5||fg_output_frames!=fg_received-4)fail("FARGAN_VALID_CENTER_COUNT");
 bench_metric("FARGAN_input_feature_frames",fg_received);bench_metric("FARGAN_output_frames",fg_output_frames);bench_metric("FARGAN_output_samples",fg_output_frames*160);
 bench_metric("FARGAN_subframe_calls",fg_subframes);bench_metric("FARGAN_random_float_draws",fg_random_draws);bench_metric("FARGAN_source_sample_start",400);bench_metric("FARGAN_source_sample_end",fg_received*160-240);
}
