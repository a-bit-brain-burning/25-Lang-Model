/* Complete fixed-frontend baseline. Storage INT8; FP32 MVE arithmetic.
 * All dynamic storage uses the benchmark arena. No teacher durations. */
#include "bench.h"
#include "storage.h"
#include "bwe_stage.h"
#include "../assets/model_index.h"
#include "../assets/cache_index.h"
#ifndef GENERATOR_CACHE_BUDGET
#define GENERATOR_CACHE_BUDGET GENERATOR_CACHE_FULL_BYTES
#endif
static unsigned char*generator_weight_cache;
static unsigned char generator_cache_valid[N25_TENSOR_COUNT];
static uint64_t generator_cache_hits,generator_cache_misses,generator_cache_uncached,generator_cache_filled_bytes,generator_source_weight_read_bytes;
static unsigned generator_cached_tensors;
#include "../assets/fixed_input.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <arm_mve.h>
#include "fast_i8_dense.h"
#include "fast_i8_pair.h"
#ifndef FRONT_I8_ENABLED
#define FRONT_I8_ENABLED 1
#endif
static void fg_noise_report(const char*prefix);
static void depthwise_report(const char*prefix);
static void report_front_i8(const char*prefix);
static void report_fg_ram_cache(const char*prefix);
#ifndef GRU_INPUT_PAIR
#define GRU_INPUT_PAIR 0
#endif
#ifndef ACOUSTIC_GRU_WEIGHT_CACHE
#define ACOUSTIC_GRU_WEIGHT_CACHE 1
#endif
#ifndef FARGAN_HEAD_PREFETCH
#define FARGAN_HEAD_PREFETCH 1
#endif
static uint64_t gru_ih_pairs,gru_ih_singles,gru_hh_frames;
static float *acoustic_gru_ih_cache,*acoustic_gru_hh_cache;
static unsigned acoustic_gru_cache_fills,acoustic_gru_cache_uses;
static uint64_t acoustic_gru_cache_init_ticks;
static uint64_t linear_pair_calls,linear_single_rows;
#define MAX_FRAMES 1024
#define MEL_CORE 48
#define FIRST_MEL_CORE 2
/* Cost counters describe the executed model operations, excluding instrumentation itself. */
static uint64_t cost_mac,cost_vfma,cost_vload,cost_iterations,cost_reductions,cost_scalar_mac,cost_pack_read,cost_pack_write,cost_dequant,cost_activation_vec,cost_activation_scalar,cost_norm_cells;
static uint64_t cost_exp,cost_erf,cost_tanh,cost_sqrt;
void profile_dequant(unsigned n){((void)0);}
#include "fast_math.inc"
#include "gru_fusion.inc"
static float measured_exp(float x){((void)0);return expf(x);}
static float measured_erf(float x){return fast_erf(x);}
static float measured_tanh(float x){return fast_tanh(x);}
static float measured_sqrt(float x){((void)0);return sqrtf(x);}
#define expf measured_exp
#define erff measured_erf
#define tanhf measured_tanh
#define sqrtf measured_sqrt
static void report_first_costs(void){}
static void identifier_snapshot_first(void);
__attribute__((noinline,used)) void snapshot_first_active_costs(void){__asm__ volatile("" ::: "memory");}
static void namefmt(char*out,size_t cap,const char*fmt,...){
 va_list args;va_start(args,fmt);size_t n=0;
 while(*fmt&&n+1<cap){
  if(*fmt!='%'){out[n++]=*fmt++;continue;}fmt++;
  if(*fmt=='s'){const char*p=va_arg(args,const char*);while(*p&&n+1<cap)out[n++]=*p++;}
  else{unsigned v=va_arg(args,unsigned);char digits[12];unsigned k=0;do{digits[k++]=(char)('0'+v%10);v/=10;}while(v);while(k&&n+1<cap)out[n++]=digits[--k];}
  fmt++;
 }out[n]=0;va_end(args);
}
static void report_front_i8(const char*prefix){}
static uint64_t generator_ticks,adapter_ticks,inference_start_ticks,first_pcm_ticks;
static unsigned stream_part,stream_samples;
static unsigned frames,mel_frames;
static float *condition_cache,*encoded_cache,*prosody_cache,*decoder_history[5];
static unsigned predicted_durations[PHONE_COUNT],decoded_frames,frame_owner[MAX_FRAMES],frame_offset[MAX_FRAMES],first_nonpause_sample;
static float gru_hidden[192];
static uint64_t acoustic_ticks,first_nonpause_ticks,first_active_ticks;
static unsigned active_found,active_scan_sample;
static void fail(const char*s){bench_puts(s);bench_puts("\n");bench_exit(70);}
static float *alloc(unsigned n){float*p=bench_alloc((size_t)n*4,16);if(!p)fail("ARENA_EXHAUSTED");return p;}
static void rewind_to(size_t m){if(bench_arena_rewind(m))fail("ARENA_REWIND_ERROR");}
#include "tensor_views.inc"
static const n25_tensor *field(const char *prefix,const char *suffix){char name[160];namefmt(name,sizeof(name),"%s.%s",prefix,suffix);return find(name);}
static int fast_front_eligible(const n25_tensor*w){return FRONT_I8_ENABLED&&w->dtype==0&&w->axis==0&&w->rank>=2&&w->dims[0]>=16&&w->elements/w->dims[0]>=16;}
static void fast_front_matrix(const float*x,const n25_tensor*w,const float*b,float*y,unsigned ci,unsigned co,int accumulate){
 if(fast_i8_dense(x,(const int8_t*)(n25_blob+w->offset),(const float*)(n25_blob+w->scale_offset),b,y,ci,co,accumulate))fail("FRONT_INTEGER_KERNEL_INPUT");
}
static void fast_front_pair(const float*x0,const float*x1,const n25_tensor*w,const float*b,float*y0,float*y1,unsigned ci,unsigned co,int accumulate){
 if(co%8){fast_front_matrix(x0,w,b,y0,ci,co,accumulate);fast_front_matrix(x1,w,b,y1,ci,co,accumulate);return;}
 if(!w->reserved||fast_i8_dense_pair(x0,x1,(const int8_t*)(n25_blob+w->offset),(const float*)(n25_blob+w->scale_offset),b,y0,y1,ci,co,accumulate))fail("FRONT_INTEGER_PAIR_INPUT");
}
static void acoustic_gru_cache_init(void){
#if ACOUSTIC_GRU_WEIGHT_CACHE
 const n25_tensor*ih=find("acoustic.rnn.ih.weight"),*hh=find("acoustic.rnn.hh.weight");
 if(fast_front_eligible(ih)&&fast_front_eligible(hh))return;
 if(acoustic_gru_cache_fills||ih->rank!=2||hh->rank!=2||ih->dims[0]!=576||hh->dims[0]!=576||ih->dims[1]!=192||hh->dims[1]!=192)fail("ACOUSTIC_GRU_CACHE_SHAPE_OR_LIFETIME");
 uint64_t start=bench_now();
 /* Request-owned allocations precede all decoder/adapter scratch marks. */
 acoustic_gru_ih_cache=alloc(576*192);acoustic_gru_hh_cache=alloc(576*192);
 n25_unpack(n25_blob,ih,acoustic_gru_ih_cache);n25_unpack(n25_blob,hh,acoustic_gru_hh_cache);
 acoustic_gru_cache_fills=1;acoustic_gru_cache_init_ticks=bench_now()-start;
#endif
}

static float sigmoid(float x){return fast_sigmoid(x);}
static float dot(const float*a,const float*b,unsigned n){
 ((void)0);((void)0);((void)0);((void)0);((void)0);((void)0);
 float32x4_t v=vdupq_n_f32(0);unsigned i=0;
 for(;i+4<=n;i+=4)v=vfmaq_f32(v,vld1q_f32(a+i),vld1q_f32(b+i));
 float q[4];vst1q_f32(q,v);float s=(q[0]+q[1])+(q[2]+q[3]);
 for(;i<n;i++){s+=a[i]*b[i];}
 return s;
}
#include "depthwise_exact.inc"
static void gemm2(const float*,const float*,const float*,const float*,float*,float*,unsigned,unsigned,int);
static void linear(const char *prefix,const float*x,float*y,unsigned n){
 const n25_tensor*w=field(prefix,"weight"),*b=field(prefix,"bias");
 if(fast_front_eligible(w)){
  unsigned co=w->dims[0],ci=w->dims[1];size_t mark=bench_arena_mark();float*bias=alloc(co);
  for(unsigned o=0;o<co;o++)bias[o]=value(b,o);
  unsigned t=0;for(;t+1<n;t+=2)fast_front_pair(x+t*ci,x+(t+1)*ci,w,bias,y+t*co,y+(t+1)*co,ci,co,0);
  if(t<n)fast_front_matrix(x+t*ci,w,bias,y+t*co,ci,co,0);
  rewind_to(mark);return;
 }
 unsigned co=w->dims[0],ci=w->dims[1];size_t mark=bench_arena_mark();float*weights=alloc(w->elements);n25_unpack(n25_blob,w,weights);
 float*bias=alloc(co);for(unsigned o=0;o<co;o++)bias[o]=value(b,o);
 unsigned t=0;for(;t+1<n;t+=2){gemm2(x+t*ci,x+(t+1)*ci,weights,bias,y+t*co,y+(t+1)*co,ci,co,0);((void)0);}
 if(t<n){for(unsigned o=0;o<co;o++)y[t*co+o]=dot(x+t*ci,weights+o*ci,ci)+bias[o];((void)0);}
 rewind_to(mark);
}
/* Time-major convolution; transpose weights are [in,out,k], normal [out,in,k]. */
static void conv_original(const char*p,const float*x,float*y,unsigned n,unsigned dil,unsigned pad,unsigned groups,unsigned stride,int transpose){
 const n25_tensor*w=field(p,"weight"),*b=field(p,"bias");
 unsigned ci=transpose?w->dims[0]:w->dims[1]*groups,co=transpose?w->dims[1]:w->dims[0],k=w->dims[2],cg=ci/groups,og=co/groups;
 unsigned nt=transpose?(n-1)*stride-2*pad+k:n+2*pad-dil*(k-1);
 if(cg==1&&og==1&&!transpose&&k<=31){
  size_t mark=bench_arena_mark();float*weights=alloc(co*k),*bias=alloc(co);
  for(unsigned j=0;j<k;j++)for(unsigned o=0;o<co;o++)weights[j*co+o]=value(w,o*k+j);
  for(unsigned o=0;o<co;o++)bias[o]=value(b,o);
  int safe=depthwise_range(weights,co*k,0x26800000u,0x58800000u);safe&=depthwise_range(bias,co,0x0d800000u,0x71800000u);safe&=depthwise_range(x,n*ci,0x26800000u,0x58800000u);
  uint64_t valid=0;
  for(unsigned t=0;t<nt;t++){
   const float*row[31];unsigned tap[31],count=0;
   for(unsigned j=0;j<k;j++){int at=(int)(t+j*dil)-(int)pad;if(at>=0&&(unsigned)at<n){row[count]=x+(unsigned)at*ci;tap[count++]=j;}}
   depthwise_channels(row,tap,count,weights,bias,y+t*co,co,safe);valid+=count;
  }
  depthwise_record(co,nt,valid,k,0,safe);rewind_to(mark);return;
 }
 size_t mark=bench_arena_mark();float*weights=alloc(w->elements);
 for(unsigned o=0;o<co;o++)for(unsigned j=0;j<k;j++)for(unsigned i=0;i<cg;i++)weights[(o*k+j)*cg+i]=value(w,transpose?(i*co+o)*k+j:(o*cg+i)*k+j);
 for(unsigned t=0;t<nt;t++)for(unsigned o=0;o<co;o++){
  float sum=value(b,o);unsigned group=o/og;
  for(unsigned j=0;j<k;j++){
   int at=transpose?(int)(t+pad)-(int)j:(int)(t+j*dil)-(int)pad;
   if(transpose){if(at<0||(unsigned)at%stride)continue;at/=(int)stride;}
   if(at>=0&&(unsigned)at<n)sum+=dot(x+(unsigned)at*ci+group*cg,weights+(o*k+j)*cg,cg);
  }y[t*co+o]=sum;
 }
 rewind_to(mark);
}
/* Four output channels share each input-vector load. Scalar tail handles conv_post. */
static void gemm4(const float*x,const float*w,const float*b,float*y,unsigned ci,unsigned co,int accumulate){
 unsigned o=0;
 ((void)0);((void)0);((void)0);((void)0);((void)0);((void)0);
 for(;o+4<=co;o+=4){
  float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0;unsigned j=0;
  for(;j+4<=ci;j+=4){
   float32x4_t v=vld1q_f32(x+j);
   a0=vfmaq_f32(a0,v,vld1q_f32(w+(o+0)*ci+j));
   a1=vfmaq_f32(a1,v,vld1q_f32(w+(o+1)*ci+j));
   a2=vfmaq_f32(a2,v,vld1q_f32(w+(o+2)*ci+j));
   a3=vfmaq_f32(a3,v,vld1q_f32(w+(o+3)*ci+j));
  }
  float q[16];vst1q_f32(q,a0);vst1q_f32(q+4,a1);vst1q_f32(q+8,a2);vst1q_f32(q+12,a3);
  for(unsigned k=0;k<4;k++){
   float v=(q[k*4]+q[k*4+1])+(q[k*4+2]+q[k*4+3]);
   for(unsigned tail=j;tail<ci;tail++)v+=x[tail]*w[(o+k)*ci+tail];
   y[o+k]=v+b[o+k]+(accumulate?y[o+k]:0.f);
  }
 }
 for(;o<co;o++)y[o]=dot(x,w+o*ci,ci)+b[o]+(accumulate?y[o]:0.f);
}
/* 2x2 output tile: two input rows and two weight rows share four vector FMAs. */
static void gemm2(const float*x0,const float*x1,const float*w,const float*b,float*y0,float*y1,unsigned ci,unsigned co,int accumulate){
 unsigned o=0;((void)0);((void)0);((void)0);((void)0);((void)0);((void)0);
 for(;o+2<=co;o+=2){
  float32x4_t a00=vdupq_n_f32(0),a01=a00,a10=a00,a11=a00;unsigned j=0;
  for(;j+4<=ci;j+=4){
   float32x4_t v0=vld1q_f32(x0+j),v1=vld1q_f32(x1+j),ww=vld1q_f32(w+o*ci+j);
   a00=vfmaq_f32(a00,v0,ww);a10=vfmaq_f32(a10,v1,ww);ww=vld1q_f32(w+(o+1)*ci+j);
   a01=vfmaq_f32(a01,v0,ww);a11=vfmaq_f32(a11,v1,ww);
  }
  float q[16];vst1q_f32(q,a00);vst1q_f32(q+4,a01);vst1q_f32(q+8,a10);vst1q_f32(q+12,a11);
  float sums[4];for(unsigned k=0;k<4;k++)sums[k]=(q[k*4]+q[k*4+1])+(q[k*4+2]+q[k*4+3]);
  for(unsigned tail=j;tail<ci;tail++){sums[0]+=x0[tail]*w[o*ci+tail];sums[1]+=x0[tail]*w[(o+1)*ci+tail];sums[2]+=x1[tail]*w[o*ci+tail];sums[3]+=x1[tail]*w[(o+1)*ci+tail];}
  for(unsigned k=0;k<2;k++){y0[o+k]=sums[k]+b[o+k]+(accumulate?y0[o+k]:0.f);y1[o+k]=sums[2+k]+b[o+k]+(accumulate?y1[o+k]:0.f);}
 }
 for(;o<co;o++){y0[o]=dot(x0,w+o*ci,ci)+b[o]+(accumulate?y0[o]:0.f);y1[o]=dot(x1,w+o*ci,ci)+b[o]+(accumulate?y1[o]:0.f);}
}
static void conv_range(const char*p,const float*x,float*y,unsigned n,unsigned dil,unsigned pad,unsigned groups,unsigned stride,int transpose,float input_slope,int accumulate,unsigned requested){
 if(groups!=1){conv_original(p,x,y,n,dil,pad,groups,stride,transpose);return;}
 const n25_tensor*w=field(p,"weight"),*b=field(p,"bias");
 unsigned ci=transpose?w->dims[0]:w->dims[1],co=transpose?w->dims[1]:w->dims[0],k=w->dims[2];
 unsigned nt=transpose?(n-1)*stride-2*pad+k:n+2*pad-dil*(k-1);
 if(requested&&requested<nt)nt=requested;
 if(!transpose&&fast_front_eligible(w)){
  unsigned inner=ci*k;size_t integer_mark=bench_arena_mark();float*patch=alloc(2*inner),*bias=alloc(co);
  for(unsigned o=0;o<co;o++)bias[o]=value(b,o);
  for(unsigned t=0;t<nt;t+=2){
   unsigned count=t+1<nt?2:1;
   for(unsigned lane=0;lane<count;lane++)for(unsigned j=0;j<k;j++){
    int at=(int)(t+lane+j*dil)-(int)pad;float*dest=patch+lane*inner+j*ci;
    if(at>=0&&(unsigned)at<n){for(unsigned i=0;i<ci;i++){float v=x[(unsigned)at*ci+i];dest[i]=input_slope>0&&v<0?v*input_slope:v;}}
    else memset(dest,0,ci*4);
   }
   if(count==2)fast_front_pair(patch,patch+inner,w,bias,y+t*co,y+(t+1)*co,inner,co,accumulate);
   else fast_front_matrix(patch,w,bias,y+t*co,inner,co,accumulate);
  }
  rewind_to(integer_mark);return;
 }
 unsigned phases=transpose?stride:1,kp=transpose?k/stride:k,inner=ci*kp;
 if(transpose&&k%stride)fail("UNSUPPORTED_TRANSPOSE_PHASE_LAYOUT");
 size_t mark=bench_arena_mark();unsigned index=(unsigned)(w-n25_tensors),offset=generator_cache_offsets[index];
 int is_generator=offset!=0xffffffffu;
 int resident=is_generator&&offset<=GENERATOR_CACHE_BUDGET&&w->elements*4<=GENERATOR_CACHE_BUDGET-offset;
 int hit=resident&&generator_cache_valid[index];
 float*weights=resident?(float*)(generator_weight_cache+offset):alloc(w->elements),*bias=alloc(co),*patch=alloc(16*inner);
 if(is_generator){if(hit)generator_cache_hits++;else if(resident)generator_cache_misses++;else generator_cache_uncached++;}
 for(unsigned o=0;o<co;o++)bias[o]=value(b,o);
 for(unsigned phase=0;phase<phases;phase++){
  unsigned first=transpose?(phase+pad)%stride:0;
  float*pw=weights+phase*co*inner;
  if(w->dtype!=0&&transpose)fail("PACKED_FLOAT_TRANSPOSE_UNSUPPORTED");
  const int8_t*codes=(const int8_t*)(n25_blob+w->offset)+phase*co*inner;
  const float*scales=(const float*)(n25_blob+w->scale_offset);
  if(!hit){
  if(is_generator)generator_source_weight_read_bytes+=(uint64_t)co*inner+(uint64_t)co*4;
  if(w->dtype!=0){
   /* FARGAN adapter tensors retain original FP16/F32 values, with the
    * same output/kernel/input layout used by the INT8 packed convolution. */
   for(unsigned i=0;i<co*inner;i++)pw[i]=value(w,phase*co*inner+i);
  }else{
  ((void)0);
  for(unsigned o=0;o<co;o++){
   unsigned i=0;
   for(;i+4<=inner;i+=4){int32x4_t q=vldrbq_s32(codes+o*inner+i);vst1q_f32(pw+o*inner+i,vmulq_n_f32(vcvtq_f32_s32(q),scales[o]));}
   for(;i<inner;i++)pw[o*inner+i]=(float)codes[o*inner+i]*scales[o];
  }
  }
  }
  unsigned count=(nt-1-phase)/phases+1;
  for(unsigned begin=0;begin<count;begin+=16){
   unsigned rows=count-begin;if(rows>16)rows=16;
   for(unsigned row=0;row<rows;row++){
    unsigned t=phase+(begin+row)*phases;
    for(unsigned j=0;j<kp;j++){
     int at=transpose?((int)(t+pad)-(int)(first+j*stride))/(int)stride:(int)(t+j*dil)-(int)pad;
     float*dest=patch+row*inner+j*ci;((void)0);
     if(at>=0&&(unsigned)at<n){
      const float*source=x+(unsigned)at*ci;((void)0);
      if(input_slope>0){
       ((void)0);((void)0);
       unsigned i=0;
       for(;i+4<=ci;i+=4){float32x4_t v=vld1q_f32(source+i);vst1q_f32(dest+i,vmaxnmq_f32(v,vmulq_n_f32(v,input_slope)));}
       for(;i<ci;i++)dest[i]=source[i]<0?source[i]*input_slope:source[i];
      }else memcpy(dest,source,ci*4);
     }else memset(dest,0,ci*4);
    }
   }
   unsigned row=0;
   for(;row+2<=rows;row+=2)gemm2(patch+row*inner,patch+(row+1)*inner,pw,bias,y+(phase+(begin+row)*phases)*co,y+(phase+(begin+row+1)*phases)*co,inner,co,accumulate);
   if(row<rows)gemm4(patch+row*inner,pw,bias,y+(phase+(begin+row)*phases)*co,inner,co,accumulate);
  }
 }
 if(resident&&!hit){generator_cache_valid[index]=1;generator_cache_filled_bytes+=(uint64_t)w->elements*4;generator_cached_tensors++;}
 rewind_to(mark);
}
static void conv_fused(const char*p,const float*x,float*y,unsigned n,unsigned dil,unsigned pad,unsigned groups,unsigned stride,int transpose,float input_slope,int accumulate){conv_range(p,x,y,n,dil,pad,groups,stride,transpose,input_slope,accumulate,0);}
static void conv(const char*p,const float*x,float*y,unsigned n,unsigned dil,unsigned pad,unsigned groups,unsigned stride,int transpose){conv_fused(p,x,y,n,dil,pad,groups,stride,transpose,0,0);}
static void norm(const char*p,float*x,unsigned n,unsigned c,float eps){
 ((void)0);
 const n25_tensor*w=field(p,"weight"),*b=field(p,"bias");
 const float*wv=tensor_view(w),*bv=tensor_view(b);if(!wv||!bv)fail("NORM_FLOAT_VIEW_MISSING");
 for(unsigned t=0;t<n;t++){
  float mean=0,var=0;for(unsigned j=0;j<c;j++)mean+=x[t*c+j];mean/=c;
  for(unsigned j=0;j<c;j++){float a=x[t*c+j]-mean;var+=a*a;}float inv=1.f/sqrtf(var/c+eps);
  for(unsigned j=0;j<c;j++)x[t*c+j]=(x[t*c+j]-mean)*inv*wv[j]+bv[j];
 }
}
static void activation(float*x,unsigned n,int type){
 unsigned i=0;
 if(type<2)for(;i+4<=n;i+=4){
  float32x4_t v=vld1q_f32(x+i),y;
  if(type==0)y=vmulq_f32(v,fast_nonlinear4(v,0));
  else{float32x4_t e=fast_nonlinear4(vmulq_n_f32(v,.7071067811865475f),2);y=vmulq_f32(vmulq_n_f32(v,.5f),vaddq_f32(vdupq_n_f32(1.f),e));}
  vst1q_f32(x+i,y);
 }
 if(type>=2)((void)0);
 for(;i<n;i++){float v=x[i];x[i]=type==0?v*sigmoid(v):type==1?.5f*v*(1.f+erff(v*.7071067811865475f)):v<0?v*(type==2?.1f:.01f):v;}
}
static void local(const char*p,float*x,unsigned n,unsigned c,unsigned d,int causal,int adapter){
 size_t mark=bench_arena_mark();float*z=alloc(n*c),*up=alloc(n*c*(adapter?4:2)),*down=alloc(n*c);char q[160];
 namefmt(q,sizeof(q),"%s.dw",p);
 if(causal){
  /* Causal left pad4*d; preserve timeline size without a right tail. */
  const n25_tensor*w=field(q,"weight"),*b=field(q,"bias");
  for(unsigned t=0;t<n;t++)for(unsigned j=0;j<c;j++){float s=value(b,j);for(unsigned k=0;k<5;k++){int at=(int)t+(int)(k*d)-(int)(4*d);if(at>=0)s+=x[(unsigned)at*c+j]*value(w,j*5+k);}z[t*c+j]=s;}
 }else conv(q,x,z,n,d,adapter?3:2*d,c,1,0);
 namefmt(q,sizeof(q),"%s.norm",p);norm(q,z,n,c,adapter?1e-6f:1e-5f);
 namefmt(q,sizeof(q),"%s.up",p);linear(q,z,up,n);activation(up,n*c*(adapter?4:2),adapter?1:0);
 namefmt(q,sizeof(q),"%s.down",p);linear(q,up,down,n);
 const n25_tensor*g=adapter?field(p,"gamma"):0;
 for(unsigned i=0;i<n*c;i++)x[i]+=down[i]*(adapter?value(g,i%c):1.f);
 rewind_to(mark);
}
static void identifier_adjust_durations(float*,unsigned*);
static void identifier_pitch_prepare(void);
static void identifier_report(void);
static void identifier_observe_physical(unsigned,unsigned,float);
static void acoustic_begin(void){
 condition_cache=alloc(MAX_FRAMES*ACOUSTIC_DIM);encoded_cache=alloc(PHONE_COUNT*128);prosody_cache=alloc(PHONE_COUNT*5);
 for(unsigned i=0;i<5;i++){unsigned ds[]={1,2,4,8,16};decoder_history[i]=alloc(4*ds[i]*192);memset(decoder_history[i],0,4*ds[i]*192*4);}
 size_t base=bench_arena_mark();unsigned n=PHONE_COUNT;
 float*x=alloc(n*128),*ph=alloc(n*PHONETIC_DIM),*ct=alloc(n*CONTEXT_DIM),*temp=alloc(n*128),*prosody=alloc(n*5);
 for(unsigned i=0;i<n*PHONETIC_DIM;i++)ph[i]=(input_phonetic[i]-norm_phonetic_mean[i%PHONETIC_DIM])/norm_phonetic_std[i%PHONETIC_DIM];
 for(unsigned i=0;i<n*CONTEXT_DIM;i++)ct[i]=(input_context[i]-norm_context_mean[i%CONTEXT_DIM])/norm_context_std[i%CONTEXT_DIM];
 const n25_tensor*phone=find("acoustic.phone.weight"),*lang=find("acoustic.language.weight");
 const float*phone_v=tensor_view(phone),*language_v=tensor_view(lang);if(!phone_v||!language_v)fail("EMBEDDING_VIEWS_MISSING");
 for(unsigned t=0;t<n;t++)for(unsigned j=0;j<128;j++)x[t*128+j]=phone_v[input_phones[t]*128+j]+language_v[LANGUAGE_ID*128+j];
 char name[160];namefmt(name,sizeof(name),"acoustic.phonetic.%d",PHONETIC_DIM);linear(name,ph,temp,n);for(unsigned i=0;i<n*128;i++)x[i]+=temp[i];
 namefmt(name,sizeof(name),"acoustic.context.%d",CONTEXT_DIM);linear(name,ct,temp,n);for(unsigned i=0;i<n*128;i++)x[i]+=temp[i];
 const unsigned enc_d[]={1,1,2,2,1,1,4,8};for(unsigned i=0;i<8;i++){namefmt(name,sizeof(name),"acoustic.encoder.%u",i);local(name,x,n,128,enc_d[i],0,0);}
 linear("acoustic.prosody.0",x,temp,n);activation(temp,n*128,0);linear("acoustic.prosody.2",temp,prosody,n);
 const n25_tensor*lp=find("acoustic.language_prosody.weight");for(unsigned i=0;i<n*5;i++)prosody[i]+=value(lp,LANGUAGE_ID*5+i%5);
 /* Added trained prosody residual. Full text-unit mean is preserved. */
 size_t pm=bench_arena_mark();float*summary_in=alloc(n*256),*rh=alloc(n*48),*localh=alloc(n*48),*delta=alloc(n*5);float summary[128]={0};
 for(unsigned t=0;t<n;t++)for(unsigned j=0;j<128;j++)summary[j]+=x[t*128+j];
 for(unsigned j=0;j<128;j++)summary[j]/=n;
 for(unsigned t=0;t<n;t++){memcpy(summary_in+t*256,x+t*128,128*4);memcpy(summary_in+t*256+128,summary,128*4);}
 linear("acoustic.prosody_learning.input",summary_in,rh,n);const n25_tensor*pl=find("acoustic.prosody_learning.language.weight");
 for(unsigned i=0;i<n*48;i++)rh[i]+=value(pl,LANGUAGE_ID*48+i%48);
 activation(rh,n*48,0);conv("acoustic.prosody_learning.local",rh,localh,n,1,2,48,1,0);
 for(unsigned i=0;i<n*48;i++){rh[i]+=localh[i];}
 activation(rh,n*48,0);
 linear("acoustic.prosody_learning.output",rh,delta,n);for(unsigned i=0;i<n*5;i++)prosody[i]+=delta[i];rewind_to(pm);
 unsigned *durations=predicted_durations;frames=0;
 for(unsigned t=0;t<n;t++){
  float v=prosody[t*5+(input_pause[t]?1:0)];
  /* exp(softplus(v))-1 equals exp(v); softplus threshold20 has negligible correction. */
  float d=nearbyintf(expf(v));if(d<1)d=1;if(d>400)d=400;
  durations[t]=(input_pause[t]&&prosody[t*5+2]<0)?0:(unsigned)d;frames+=durations[t];
 }
 identifier_adjust_durations(prosody,durations);frames=0;for(unsigned i=0;i<n;i++)frames+=durations[i];
 if(!frames||frames>MAX_FRAMES)fail("FRAME_CAPACITY_EXCEEDED");
 bench_metric("predicted_frames",frames);if(bench_write_file("durations.u32",durations,PHONE_COUNT*sizeof(unsigned)))fail("DURATION_EXPORT_FAILED");
 memcpy(encoded_cache,x,n*128*4);memcpy(prosody_cache,prosody,n*5*4);
 unsigned cursor=0;first_nonpause_sample=0;int found=0;
 for(unsigned t=0;t<n;t++){
  if(!found&&!input_pause[t]&&durations[t]){first_nonpause_sample=cursor*160;found=1;}
  for(unsigned j=0;j<durations[t];j++,cursor++){frame_owner[cursor]=t;frame_offset[cursor]=j;}
 }
 bench_metric("first_nonpause_sample",first_nonpause_sample);rewind_to(base);
}
/* Transposed depthwise taps [k][c] so four channels advance as one MVE
 * vector. Each lane still accumulates k=0..4 in the original order, so every
 * output float is bit-identical to the scalar loop. */
static float decoder_dw_taps[5][5*192];
static unsigned decoder_dw_taps_ready[5];
static void decode_local(unsigned stage,const char*p,float*x,unsigned n,unsigned d,float*history){
 ((void)0);((void)0);
 size_t mark=bench_arena_mark();unsigned c=192,hist=4*d;float*z=alloc(n*c),*up=alloc(n*c*2),*down=alloc(n*c);char q[160];
 namefmt(q,sizeof(q),"%s.dw",p);const n25_tensor*w=field(q,"weight"),*b=field(q,"bias");
 const float*wv=tensor_view(w),*bv=tensor_view(b);if(!wv||!bv)fail("DECODER_DW_VIEWS_MISSING");
 if(!decoder_dw_taps_ready[stage]){
  for(unsigned j=0;j<c;j++)for(unsigned k=0;k<5;k++)decoder_dw_taps[stage][k*c+j]=wv[j*5+k];
  decoder_dw_taps_ready[stage]=1;((void)0);
 }
 const float*wt=decoder_dw_taps[stage];
 for(unsigned t=0;t<n;t++){
  const float*s0,*s1,*s2,*s3,*s4;int at;
  at=(int)t-(int)hist;s0=at<0?history+(unsigned)(at+(int)hist)*c:x+(unsigned)at*c;
  at=(int)t+(int)d-(int)hist;s1=at<0?history+(unsigned)(at+(int)hist)*c:x+(unsigned)at*c;
  at=(int)t+2*(int)d-(int)hist;s2=at<0?history+(unsigned)(at+(int)hist)*c:x+(unsigned)at*c;
  at=(int)t+3*(int)d-(int)hist;s3=at<0?history+(unsigned)(at+(int)hist)*c:x+(unsigned)at*c;
  at=(int)t+4*(int)d-(int)hist;s4=at<0?history+(unsigned)(at+(int)hist)*c:x+(unsigned)at*c;
  for(unsigned j=0;j<c;j+=4){
   float32x4_t sum=vld1q_f32(bv+j);
   sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(s0+j),vld1q_f32(wt+0*c+j)));
   sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(s1+j),vld1q_f32(wt+1*c+j)));
   sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(s2+j),vld1q_f32(wt+2*c+j)));
   sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(s3+j),vld1q_f32(wt+3*c+j)));
   sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(s4+j),vld1q_f32(wt+4*c+j)));
   vst1q_f32(z+t*c+j,sum);
  }
 }
 namefmt(q,sizeof(q),"%s.norm",p);norm(q,z,n,c,1e-5f);namefmt(q,sizeof(q),"%s.up",p);linear(q,z,up,n);activation(up,n*c*2,0);namefmt(q,sizeof(q),"%s.down",p);linear(q,up,down,n);
 if(n>=hist)memcpy(history,x+(n-hist)*c,hist*c*4);else{memmove(history,history+n*c,(hist-n)*c*4);memcpy(history+(hist-n)*c,x,n*c*4);}
 for(unsigned i=0;i<n*c;i++){x[i]+=down[i];}
 rewind_to(mark);
}
static void ensure_raw_condition_chunk(unsigned end){
 if(end>frames){end=frames;}
 if(end<=decoded_frames)return;
 uint64_t begin_time=bench_now();unsigned n=end-decoded_frames;size_t base=bench_arena_mark();
 float*in=alloc(n*135),*seq=alloc(n*192);
 for(unsigned t=0;t<n;t++){
  unsigned f=decoded_frames+t,owner=frame_owner[f],j=frame_offset[f],duration=predicted_durations[owner];
  memcpy(in+t*135,encoded_cache+owner*128,128*4);for(unsigned k=0;k<3;k++)in[t*135+128+k]=prosody_cache[owner*5+2+k];
  in[t*135+131]=(float)j/(duration>1?duration-1:1);in[t*135+132]=1.f/duration;in[t*135+133]=j==0;in[t*135+134]=j+1==duration;
 }
 linear("acoustic.frame_in",in,seq,n);activation(seq,n*192,0);
 size_t gm=bench_arena_mark();
 const n25_tensor*integer_ih=find("acoustic.rnn.ih.weight"),*integer_hh=find("acoustic.rnn.hh.weight");
 int use_integer=fast_front_eligible(integer_ih)&&fast_front_eligible(integer_hh);
 const float *wi=0,*wh=0;
 if(!use_integer){
#if ACOUSTIC_GRU_WEIGHT_CACHE
 if(!acoustic_gru_cache_fills)fail("ACOUSTIC_GRU_CACHE_NOT_INITIALIZED");
 wi=acoustic_gru_ih_cache;wh=acoustic_gru_hh_cache;((void)0);
#else
 float*wi_mut=alloc(576*192),*wh_mut=alloc(576*192);n25_unpack(n25_blob,integer_ih,wi_mut);n25_unpack(n25_blob,integer_hh,wh_mut);wi=wi_mut;wh=wh_mut;
#endif
 }
 if(fast_front_eligible(integer_ih)&&fast_front_eligible(integer_hh)){
  float ig[2*576],hg[576],bias[576];const n25_tensor*bi=find("acoustic.rnn.ih.bias");
  for(unsigned j=0;j<576;j++)bias[j]=value(bi,j);
  for(unsigned t=0;t<n;t+=2){
   unsigned count=t+1<n?2:1;
   if(count==2)fast_front_pair(seq+t*192,seq+(t+1)*192,integer_ih,bias,ig,ig+576,192,576,0);
   else fast_front_matrix(seq+t*192,integer_ih,bias,ig,192,576,0);
   for(unsigned lane=0;lane<count;lane++){
    fast_front_matrix(gru_hidden,integer_hh,0,hg,192,576,0);
    fusion_acoustic_gru(ig+lane*576,hg,gru_hidden,seq+(t+lane)*192);((void)0);
   }
  }
 }else{
#if GRU_INPUT_PAIR
 /* Input rows are known. Hidden-to-hidden remains strictly sequential. */
 float ig[2*576],hg[576],bias[576];const n25_tensor*bi=find("acoustic.rnn.ih.bias");
 for(unsigned j=0;j<576;j++)bias[j]=value(bi,j);
 for(unsigned t=0;t<n;t+=2){
  unsigned count=t+1<n?2:1;
  if(count==2){gemm2(seq+t*192,seq+(t+1)*192,wi,bias,ig,ig+576,192,576,0);((void)0);}
  else{for(unsigned j=0;j<576;j++)ig[j]=dot(seq+t*192,wi+j*192,192)+bias[j];((void)0);}
  for(unsigned lane=0;lane<count;lane++){
   for(unsigned j=0;j<576;j++)hg[j]=dot(gru_hidden,wh+j*192,192);
   const float*g=ig+lane*576;
   fusion_acoustic_gru(g,hg,gru_hidden,seq+(t+lane)*192);
   ((void)0);
  }
 }
#else
 float ig[576],hg[576];const n25_tensor*bi=find("acoustic.rnn.ih.bias");
 for(unsigned t=0;t<n;t++){((void)0);((void)0);
  for(unsigned j=0;j<576;j++){ig[j]=dot(seq+t*192,wi+j*192,192)+value(bi,j);hg[j]=dot(gru_hidden,wh+j*192,192);}
  fusion_acoustic_gru(ig,hg,gru_hidden,seq+t*192);
 }
#endif
 }
 rewind_to(gm);char name[160];const unsigned ds[]={1,2,4,8,16};
 for(unsigned i=0;i<5;i++){namefmt(name,sizeof(name),"acoustic.decoder.%u",i);decode_local(i,name,seq,n,ds[i],decoder_history[i]);}
 float*out=condition_cache+decoded_frames*ACOUSTIC_DIM;namefmt(name,sizeof(name),"acoustic.heads.%d",ACOUSTIC_DIM);linear(name,seq,out,n);const n25_tensor*la=find("acoustic.language_acoustic.weight");const float*la_v=tensor_view(la);if(!la_v)fail("ACOUSTIC_LANGUAGE_VIEW_MISSING");
 for(unsigned t=0;t<n;t++)for(unsigned j=0;j<ACOUSTIC_DIM;j++){
  float v=out[t*ACOUSTIC_DIM+j]+la_v[LANGUAGE_ID*39+j];float physical=j==ACOUSTIC_DIM-1?sigmoid(v):v*norm_std[j]+norm_mean[j];identifier_observe_physical(decoded_frames+t,j,physical);out[t*ACOUSTIC_DIM+j]=(physical-norm_mean[j])/norm_std[j];
 }
 decoded_frames=end;rewind_to(base);acoustic_ticks+=bench_now()-begin_time;
}
static void ensure_condition(unsigned end);
#include "identifier_control.inc"

static float *adapter_window(unsigned begin,unsigned end){
 float pfirst=((float)begin*256+128)*100/22050-.5f,plast=((float)(end-1)*256+128)*100/22050-.5f;
 int lo=(int)floorf(pfirst),hi=(int)floorf(plast);if(lo<0)lo=0;if(hi<0)hi=0;if(hi>=(int)frames)hi=frames-1;
 unsigned maxhi=(unsigned)hi+1;if(maxhi>=frames)maxhi=frames-1;
 unsigned left=lo>21?(unsigned)lo-21:0,right=maxhi+22;if(right>frames)right=frames;
 ensure_condition(right);uint64_t start=bench_now();unsigned n=right-left;
 float*mel=alloc((end-begin)*80);size_t mark=bench_arena_mark();float*z=alloc(n*192),*head=alloc(n*80);char name[160];
 namefmt(name,sizeof(name),"physical_to_mel.inputs.%d",ACOUSTIC_DIM);conv(name,condition_cache+left*ACOUSTIC_DIM,z,n,1,3,1,1,0);
 const n25_tensor*l=find("physical_to_mel.language.weight");for(unsigned i=0;i<n*192;i++)z[i]+=value(l,LANGUAGE_ID*192+i%192);norm("physical_to_mel.norm",z,n,192,1e-6f);
 for(unsigned i=0;i<6;i++){namefmt(name,sizeof(name),"physical_to_mel.blocks.%u",i);local(name,z,n,192,1,0,1);}
 linear("physical_to_mel.head",z,head,n);
 for(unsigned t=begin;t<end;t++){
  float pos=((float)t*256+128)*100/22050-.5f;int lidx=(int)floorf(pos);float a=pos-floorf(pos);if(lidx<0)lidx=0;if(lidx>=(int)frames)lidx=frames-1;
  unsigned hidx=(unsigned)lidx+1;if(hidx>=frames)hidx=frames-1;
  for(unsigned j=0;j<80;j++)mel[(t-begin)*80+j]=head[((unsigned)lidx-left)*80+j]*(1-a)+head[(hidx-left)*80+j]*a;
 }
 rewind_to(mark);adapter_ticks+=bench_now()-start;return mel;
}
/* Fused activation/packing and residual add use four reusable activation buffers. */
static void resblock(const char*p,const float*x,float*y,float*z,unsigned n,unsigned c,unsigned k){
 memcpy(y,x,n*c*4);char name[160];unsigned dil[]={1,3,5};
 for(unsigned i=0;i<3;i++){
  namefmt(name,sizeof(name),"%s.convs1.%u",p,i);conv_fused(name,y,z,n,dil[i],(k-1)*dil[i]/2,1,1,0,.1f,0);
  namefmt(name,sizeof(name),"%s.convs2.%u",p,i);conv_fused(name,z,y,n,1,(k-1)/2,1,1,0,.1f,1);
 }
}
static void generator(const float*mel,unsigned m,float*out){
 size_t base=bench_arena_mark();unsigned capacity=m*2048;float*x=alloc(capacity),*up=alloc(capacity),*branch=alloc(capacity),*z=alloc(capacity);
 conv("generator.conv_pre",mel,x,m,1,3,1,1,0);unsigned n=m,c=128;char name[160];unsigned rates[]={8,8,2,2},kernels[]={16,16,4,4},rk[]={3,7,11};
 for(unsigned stage=0;stage<4;stage++){
  activation(x,n*c,2);unsigned nt=n*rates[stage],co=c/2;
  namefmt(name,sizeof(name),"generator.ups.%u",stage);conv(name,x,up,n,1,(kernels[stage]-rates[stage])/2,1,rates[stage],1);
  memset(x,0,nt*co*4);
  for(unsigned j=0;j<3;j++){
   namefmt(name,sizeof(name),"generator.resblocks.%u",stage*3+j);resblock(name,up,branch,z,nt,co,rk[j]);for(unsigned i=0;i<nt*co;i++)x[i]+=branch[i];
  }
  for(unsigned i=0;i<nt*co;i++){x[i]/=3;}
  n=nt;c=co;
 }
 activation(x,n*c,3);conv("generator.conv_post",x,out,n,1,3,1,1,0);for(unsigned i=0;i<n;i++)out[i]=tanhf(out[i]);rewind_to(base);
}
static unsigned minimum(unsigned a,unsigned b){return a<b?a:b;}
/* First prefix: propagate exact required lengths backward through the unchanged graph.
 * Later outputs that cannot affect this prefix are never evaluated. */
static void generator_prefix(const float*mel,unsigned m,unsigned wanted,float*out){
 unsigned strides[]={8,8,2,2},kernels[]={16,16,4,4},rk[]={3,7,11};
 unsigned full[4],need[4],upneed[4],total=mel_frames;
 for(unsigned i=0;i<4;i++){total*=strides[i];full[i]=total;}
 need[3]=minimum(wanted+3,full[3]);unsigned pre=0;
 for(int i=3;i>=0;i--){
  upneed[i]=minimum(need[i]+60,full[i]);unsigned pad=(kernels[i]-strides[i])/2;
  unsigned prev=(upneed[i]-1+pad)/strides[i]+1;
  prev=minimum(prev,i?full[i-1]:mel_frames);
  if(i)need[i-1]=prev;else pre=prev;
 }
 size_t base=bench_arena_mark();unsigned capacity=pre*128;
 for(unsigned i=0;i<4;i++){unsigned size=upneed[i]*(64>>i);if(size>capacity)capacity=size;}
 float*x=alloc(capacity),*up=alloc(capacity),*branch=alloc(capacity),*z=alloc(capacity);char name[160];
 conv_range("generator.conv_pre",mel,x,m,1,3,1,1,0,0,0,pre);unsigned n=pre,c=128;
 for(unsigned stage=0;stage<4;stage++){
  activation(x,n*c,2);unsigned co=c/2,target=need[stage];
  namefmt(name,sizeof(name),"generator.ups.%u",stage);conv_range(name,x,up,n,1,(kernels[stage]-strides[stage])/2,1,strides[stage],1,0,0,upneed[stage]);
  memset(x,0,target*co*4);
  for(unsigned j=0;j<3;j++){
   unsigned dilation[]={1,3,5},radius=(rk[j]-1)/2,cur=upneed[stage];memcpy(branch,up,cur*co*4);
   for(unsigned pair=0;pair<3;pair++){
    unsigned extra=0;for(unsigned later=pair+1;later<3;later++)extra+=radius*(dilation[later]+1);
    unsigned result_n=minimum(target+extra,full[stage]),mid=minimum(result_n+radius,full[stage]);
    namefmt(name,sizeof(name),"generator.resblocks.%u.convs1.%u",stage*3+j,pair);conv_range(name,branch,z,cur,dilation[pair],radius*dilation[pair],1,1,0,.1f,0,mid);
    namefmt(name,sizeof(name),"generator.resblocks.%u.convs2.%u",stage*3+j,pair);conv_range(name,z,branch,mid,1,radius,1,1,0,.1f,1,result_n);cur=result_n;
   }
   for(unsigned q=0;q<target*co;q++)x[q]+=branch[q];
  }
  for(unsigned q=0;q<target*co;q++)x[q]/=3;
  n=target;c=co;
 }
 activation(x,n*c,3);conv_range("generator.conv_post",x,out,n,1,3,1,1,0,0,0,wanted);for(unsigned q=0;q<wanted;q++)out[q]=tanhf(out[q]);rewind_to(base);
}
static int ceildiv(int a,int b){return a>=0?(a+b-1)/b:a/b;}
static void gen_interval(int begin,int end,int*left,int*right){
 int lo=begin-3,hi=end+3;int rates[]={8,8,2,2},kernels[]={16,16,4,4};
 for(int i=3;i>=0;i--){lo-=60;hi+=60;int pad=(kernels[i]-rates[i])/2;lo=ceildiv(lo+pad-(kernels[i]-1),rates[i]);hi=(hi-1+pad)/rates[i]+1;}
 *left=lo-3;*right=hi+3;
}
static void stream_ready(const float*x,unsigned produced,int final);
static float *vocode(void){
 float*native=alloc(mel_frames*256);unsigned max_mel=0;
 for(unsigned begin=0;begin<mel_frames;){
  unsigned first_core=1;
  unsigned required24=(first_nonpause_sample*3+1)/2+480;
  unsigned required16=((required24+27+479)/480)*320;
  while(((first_core*256-1)*320/441)-(RESAMPLE_REMOVE-1)<required16)first_core++;
  if(!begin){bench_metric("BWE_first_required_input16",required16);bench_metric("BWE_first_core_mel",first_core);bench_metric("BWE_one_less_mel_stable16",(((first_core-1)*256-1)*320/441)-(RESAMPLE_REMOVE-1));}
  unsigned end=begin+(begin?(active_found?MEL_CORE:4):first_core);if(end>mel_frames)end=mel_frames;int left,right;gen_interval(begin*256,end*256,&left,&right);if(left<0)left=0;if(right>(int)mel_frames)right=mel_frames;
  unsigned m=(unsigned)(right-left);if(m>max_mel)max_mel=m;size_t mark=bench_arena_mark();float*mel=adapter_window((unsigned)left,(unsigned)right);float*chunk=alloc(m*256);uint64_t start=bench_now();if(!begin)generator_prefix(mel,m,end*256,chunk);else generator(mel,m,chunk);generator_ticks+=bench_now()-start;
  memcpy(native+begin*256,chunk+(begin-(unsigned)left)*256,(end-begin)*256*4);rewind_to(mark);
  stream_ready(native,end*256,end==mel_frames);
  bench_metric("generator_completed_mel_frames",end);begin=end;
 }
 bench_metric("max_generator_tile_mel_frames",max_mel);return native;
}
static void pcm_range(const float*x,unsigned n,unsigned offset,unsigned count,int16_t*pcm,unsigned*clips,float*peak){
 for(unsigned t=offset;t<offset+count;t++){
  unsigned at=(t+RESAMPLE_REMOVE)*441;int first=ceildiv((int)at-(RESAMPLE_TAPS-1),320),last=at/320;if(first<0)first=0;if(last>=(int)n)last=n-1;float y=0;
  for(int i=first;i<=last;i++){y+=x[i]*resample_filter[at-(unsigned)i*320];}
  if(!isfinite(y)){fail("NONFINITE_AUDIO");}
  if(fabsf(y)>*peak){*peak=fabsf(y);}
  if(y>=1||y< -1){(*clips)++;}
  float q=nearbyintf(y*32768);if(q>32767)q=32767;if(q< -32768)q=-32768;pcm[t-offset]=(int16_t)q;
 }
}
static void stream_ready(const float*x,unsigned produced,int final){
 unsigned ready=final?frames*160:((produced-1)*320/441)-(RESAMPLE_REMOVE-1);
 if(ready>frames*160)ready=frames*160;
 if(ready<=stream_samples)return;
 size_t mark=bench_arena_mark();unsigned count=ready-stream_samples;int16_t*pcm=bench_alloc(count*2,16);if(!pcm)fail("STREAM_PCM_ALLOCATION_FAILED");
 unsigned clips=0;float peak=0;pcm_range(x,produced,stream_samples,count,pcm,&clips,&peak);
 if(!stream_part){first_pcm_ticks=bench_now()-inference_start_ticks;bench_metric("first_pcm_ready_host_ticks",first_pcm_ticks);bench_metric("first_pcm_samples",count);}
 char filename[64];namefmt(filename,sizeof(filename),"stream_%u.pcm",stream_part++);
 if(bench_write_file(filename,pcm,count*2))fail("STREAM_PCM_EXPORT_FAILED");
 if(!first_nonpause_ticks&&ready>=first_nonpause_sample+320){first_nonpause_ticks=bench_now()-inference_start_ticks;bench_metric("first_nonpause_frame_ready_host_ticks",first_nonpause_ticks);}
 if(!active_found){
  /* This observer owns its absolute cursor. Native generation may publish
   * fewer than 320 PCM16 samples at a time; scan across those boundaries. */
  int16_t window[320];
  while(active_scan_sample+320<=ready){
   unsigned observer_clips=0;float observer_peak=0;
   pcm_range(x,produced,active_scan_sample,320,window,&observer_clips,&observer_peak);
   float energy=0,maximum=0;for(unsigned k=0;k<320;k++){float v=(float)window[k]/32768.f;energy+=v*v;if(fabsf(v)>maximum)maximum=fabsf(v);}
   if(energy/320>=.0000316227766f&&maximum>=.01f){active_found=1;first_active_ticks=bench_now()-inference_start_ticks;bench_metric("first_active_frame_ready_host_ticks",first_active_ticks);bench_metric("first_active_frame_start_sample",active_scan_sample);report_first_costs();break;}
   active_scan_sample+=160;
  }
 }
 bwe_push(pcm,count);
 stream_samples=ready;bench_metric("streamed_pcm_samples",stream_samples);rewind_to(mark);
}
static void export_pcm(const float*x,unsigned n){
 size_t mark=bench_arena_mark();unsigned samples=frames*160;int16_t*pcm=bench_alloc(samples*2,16);if(!pcm)fail("PCM_CAPACITY_EXCEEDED");unsigned clips=0;float peak=0;
 pcm_range(x,n,0,samples,pcm,&clips,&peak);
 bench_metric("audio_peak_millionths",(uint64_t)(peak*1000000));bench_metric("clipped_samples",clips);bench_metric("pcm_samples",samples);
 if(bench_write_file("output.pcm",pcm,samples*2)){fail("PCM_EXPORT_FAILED");}
 rewind_to(mark);
}
#include "adapter_stream.inc"

static int16_t *fg_pcm;
static float *fg_float;
static unsigned fg_samples,fg_expected,fg_scan,fg_clips;
static float fg_peak;
static void fargan_emit(const float *values,unsigned count){
 if(fg_samples+count>fg_expected)fail("FARGAN_TOO_MANY_SAMPLES");
 unsigned begin=fg_samples;
 for(unsigned i=0;i<count;i++){
  float value=values[i];if(!isfinite(value))fail("FARGAN_NONFINITE");
  if(fabsf(value)>fg_peak)fg_peak=fabsf(value);
  if(value>=1.f||value< -1.f)fg_clips++;
  float q=nearbyintf(value*32768.f);if(q>32767)q=32767;if(q< -32768)q=-32768;
  fg_float[fg_samples]=value;fg_pcm[fg_samples++]=(int16_t)q;
 }
 if(!stream_part){first_pcm_ticks=bench_now()-inference_start_ticks;bench_metric("first_pcm_ready_host_ticks",first_pcm_ticks);bench_metric("first_pcm_samples",count);}
 char name[64];namefmt(name,sizeof(name),"stream_%u.pcm",stream_part++);
 if(bench_write_file(name,fg_pcm+begin,count*2))fail("FARGAN_STREAM_WRITE");
 while(!active_found&&fg_scan+320<=fg_samples){
  float energy=0,peak=0;for(unsigned i=0;i<320;i++){float v=(float)fg_pcm[fg_scan+i]/32768.f;energy+=v*v;if(fabsf(v)>peak)peak=fabsf(v);}
  if(energy/320>=.0000316227766f&&peak>=.01f){active_found=1;first_active_ticks=bench_now()-inference_start_ticks;bench_metric("first_active_frame_ready_host_ticks",first_active_ticks);bench_metric("first_active_frame_start_sample",fg_scan);report_first_costs();break;}
  fg_scan+=160;
 }
 bwe_push(fg_pcm+begin,count);stream_samples=fg_samples;
}

#include "fargan_stream.inc"

int main(void){
 bench_puts("NEW25_FARGAN_M55_BEGIN\n");bench_metric("weights_bytes",MODEL_BYTES);bench_metric("acoustic_step",3000);bench_metric("acoustic_added_stage_step",1000);bench_metric("FARGAN_step",4750);bench_metric("gru_input_pair_enabled",GRU_INPUT_PAIR);
 uint64_t boot=bench_now();storage_initialize(boot);
 uint64_t all=bench_now(),start=bench_now();inference_start_ticks=all;bench_metric("storage_before_request_host_ticks",all-boot);
 tensor_views_prepare();
 acoustic_begin();acoustic_ticks=bench_now()-start;
 acoustic_gru_cache_init();
 identifier_pitch_prepare();
 if(frames<5)fail("FARGAN_REQUIRES_FIVE_PHYSICAL_FRAMES");
 mel_frames=frames;fg_expected=(frames-4)*160;fg_pcm=bench_alloc(fg_expected*2,16);fg_float=alloc(fg_expected);if(!fg_pcm)fail("FARGAN_OUTPUT_ALLOC");
 unsigned nonpause=first_nonpause_sample>400?first_nonpause_sample-400:0;fg_scan=nonpause;
 adapter_stream_init();fargan_stream_init();
 bwe_start(fg_expected,nonpause,all);
#if FARGAN_HEAD_PREFETCH
 /* Preserve the original valid center and predicted leading pause. BWE emits
  * 480*k-27 stable24k samples after k real320-sample input frames. Its speech
  * observer tests480-sample windows on a240-sample lattice. Prepare just the
  * features needed for the first eligible complete window, then batches16. */
 unsigned boundary24=(nonpause*3+1)/2;
 unsigned first_window_end24=((boundary24+239)/240)*240+480;
 unsigned first_required16=((first_window_end24+27+479)/480)*320;
 unsigned first_features=(first_required16+159)/160+4;
 if(first_features>frames)first_features=frames;
 bench_metric("FARGAN_prefetch_first_required16",first_required16);
 bench_metric("FARGAN_prefetch_first_feature_frames",first_features);
 bench_metric("FARGAN_prefetch_subsequent_feature_frames",16);
 bench_metric("FARGAN_prefetch_pre_active_feature_frames",2);
 bench_metric("FARGAN_prefetch_post_active_feature_frames",16);
 unsigned batches=0;
 for(unsigned begin=0;begin<frames;){
  unsigned end=begin?begin+(bwe_has_active_speech()?16:2):first_features;if(end>frames)end=frames;
  (void)adapter_fargan_frame(end-1); /* head storage is persistent, safe to reread */
  for(unsigned i=begin;i<end;i++)fargan_frame(adapter_fargan_frame(i),i);
  batches++;begin=end;
 }
 bench_metric("FARGAN_prefetch_batches",batches);
#else
 for(unsigned i=0;i<frames;i++)fargan_frame(adapter_fargan_frame(i),i);
#endif
 if(fg_samples!=fg_expected)fail("FARGAN_OUTPUT_COUNT");
 ensure_condition(frames);bwe_finish();
 if(bench_write_file("condition.f32",condition_cache,frames*ACOUSTIC_DIM*4))fail("CONDITION_EXPORT_FAILED");
 if(bench_write_file("output.pcm",fg_pcm,fg_samples*2))fail("FARGAN_PCM_WRITE");
 if(bench_write_file("fargan.f32",fg_float,fg_samples*4))fail("FARGAN_FLOAT_WRITE");
 bench_metric("source_sample_start16",400);bench_metric("source_sample_end16",frames*160-240);bench_metric("pcm_samples",fg_samples);bench_metric("clipped_samples",fg_clips);bench_metric("audio_peak_millionths",(uint64_t)(fg_peak*1000000));
 bench_metric("decoded_frames",decoded_frames);bench_metric("active_frame_detected",active_found);bench_metric("tick_frequency_hz",bench_tick_frequency());bench_metric("diagnostic_operation_counters_disabled",1);bench_metric("operation_estimator_must_not_read_zero_counters",1);bench_metric("acoustic_host_ticks",acoustic_ticks);bench_metric("adapter_host_ticks",adapter_ticks);bench_metric("firmware_total_host_ticks",bench_now()-all);
 depthwise_report("total");bench_metric("total_cost_mac",cost_mac);bench_metric("total_cost_vfma",cost_vfma);bench_metric("total_cost_vload",cost_vload);bench_metric("total_cost_iterations",cost_iterations);bench_metric("total_cost_dequant",cost_dequant);
 bench_metric("acoustic_gru_cache_enabled",ACOUSTIC_GRU_WEIGHT_CACHE);bench_metric("acoustic_gru_cache_bytes",ACOUSTIC_GRU_WEIGHT_CACHE?2*576*192*4:0);bench_metric("acoustic_gru_cache_fills",acoustic_gru_cache_fills);bench_metric("acoustic_gru_cache_uses",acoustic_gru_cache_uses);bench_metric("acoustic_gru_cache_init_host_ticks",acoustic_gru_cache_init_ticks);bench_metric("FARGAN_head_prefetch_enabled",FARGAN_HEAD_PREFETCH);
 report_front_i8("total_all_i8");bench_metric("front_integer_enabled",FRONT_I8_ENABLED);bench_metric("front_integer_activation_split16",FAST_I8_ACT_SPLIT16);
 identifier_report();fargan_stream_report();fast_math_report("total");fusion_report("total");fg_noise_report("total");adapter_stream_report();bench_memory_report();bench_puts("NEW25_FARGAN_M55_PASS\n");return 0;
}
