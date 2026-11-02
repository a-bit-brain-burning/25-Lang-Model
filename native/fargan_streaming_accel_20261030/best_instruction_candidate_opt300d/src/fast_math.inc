/* Independent numerical candidate: piecewise-linear interpolation on a 1/512
 * grid. Gain/duration exp, sqrt and FARGAN RNG/de-emphasis remain unchanged. */
#include "fast_math_tables.h"
static uint64_t fast_sigmoid_lut,fast_tanh_lut,fast_erf_lut;
static uint64_t fast_sigmoid_fallback,fast_tanh_fallback,fast_erf_fallback;
static uint64_t fast_tanh_saturated,fast_erf_saturated;
static uint64_t fast_lut_vector_groups,fast_lut_vector_gathers;
static uint64_t fast_lut_vector_validation_groups,fast_lut_vector_validation_fallbacks;
static uint64_t fast_lut_inline_invocations,fast_lut_fallback_calls,fast_lut_fast_vector_operations;
static uint64_t fast_lut_validation_vector_operations,fast_lut_validation_reductions,fast_lut_table_words;
static void namefmt(char*,size_t,const char*,...);
static float fast_table(const float*table,float a){
 float at=a*512.f;unsigned i=(unsigned)at;float f=at-(float)i;
 return table[i]+f*(table[i+1]-table[i]);
}
static float fast_sigmoid(float x){
 float a=fabsf(x);
 if(!isfinite(a)||a>=18.f){((void)0);((void)0);return 1.f/(1.f+expf(-x));}
 ((void)0);float q=fast_table(fast_sigmoid_tail_table,a);
 return x>=0.f?1.f-q:q;
}
static float fast_tanh(float x){
 float a=fabsf(x);
 if(!isfinite(a)){((void)0);((void)0);return tanhf(x);}
 if(a>=10.f){((void)0);return copysignf(1.f,x);}
 ((void)0);return copysignf(fast_table(fast_tanh_table,a),x);
}
static float fast_erf(float x){
 float a=fabsf(x);
 if(!isfinite(a)){((void)0);((void)0);return erff(x);}
 if(a>=6.f){((void)0);return copysignf(1.f,x);}
 ((void)0);return copysignf(fast_table(fast_erf_table,a),x);
}
/* Rare path is out of line, retaining exactly four scalar lane evaluations. */
static __attribute__((noinline,cold)) float32x4_t fast_nonlinear4_fallback(float32x4_t x,unsigned kind){
 float lane[4];vst1q_f32(lane,x);((void)0);((void)0);
 for(unsigned k=0;k<4;k++)lane[k]=kind==0?fast_sigmoid(lane[k]):kind==1?fast_tanh(lane[k]):fast_erf(lane[k]);
 return vld1q_f32(lane);
}
/* Same LUTs, interpolant and rounded order. Constant kinds fold at callers. */
static __attribute__((always_inline)) inline float32x4_t fast_nonlinear4(float32x4_t x,unsigned kind){
 union{float f;uint32_t u;}limit={.f=kind==0?18.f:kind==1?10.f:6.f};
 uint32x4_t absbits=vandq_u32(vreinterpretq_u32_f32(x),vdupq_n_u32(0x7fffffffu));
 ((void)0);((void)0);
 ((void)0);((void)0);
 if(vmaxvq_u32(0,absbits)>=limit.u)return fast_nonlinear4_fallback(x,kind);
 const float*table=kind==0?fast_sigmoid_tail_table:kind==1?fast_tanh_table:fast_erf_table;
 float32x4_t at=vmulq_n_f32(vreinterpretq_f32_u32(absbits),512.f);
 uint32x4_t index=vcvtq_u32_f32(at);
 float32x4_t fraction=vsubq_f32(at,vcvtq_f32_u32(index));
 float32x4_t a=vldrwq_gather_shifted_offset_f32(table,index);
 float32x4_t b=vldrwq_gather_shifted_offset_f32(table+1,index);
 float32x4_t y=vaddq_f32(a,vmulq_f32(fraction,vsubq_f32(b,a)));
 if(kind==0){((void)0);y=vpselq_f32(vsubq_f32(vdupq_n_f32(1.f),y),y,vcmpgeq_n_f32(x,0.f));}
 else{
  if(kind==1)((void)0);else ((void)0);
  uint32x4_t sign=vandq_u32(vreinterpretq_u32_f32(x),vdupq_n_u32(0x80000000u));
  y=vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(y),sign));
 }
 ((void)0);((void)0);((void)0);
 ((void)0);
 return y;
}
static void fast_math_report(const char*prefix){}
