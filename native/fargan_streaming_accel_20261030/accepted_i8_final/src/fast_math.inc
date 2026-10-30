/* Independent numerical candidate: piecewise-linear interpolation on a 1/512
 * grid. Gain/duration exp, sqrt and FARGAN RNG/de-emphasis remain unchanged. */
#include "fast_math_tables.h"
static uint64_t fast_sigmoid_lut,fast_tanh_lut,fast_erf_lut;
static uint64_t fast_sigmoid_fallback,fast_tanh_fallback,fast_erf_fallback;
static uint64_t fast_tanh_saturated,fast_erf_saturated;
static uint64_t fast_lut_vector_groups,fast_lut_vector_gathers;
static void namefmt(char*,size_t,const char*,...);
static float fast_table(const float*table,float a){
 float at=a*512.f;unsigned i=(unsigned)at;float f=at-(float)i;
 return table[i]+f*(table[i+1]-table[i]);
}
static float fast_sigmoid(float x){
 float a=fabsf(x);
 if(!isfinite(a)||a>=18.f){fast_sigmoid_fallback++;cost_exp++;return 1.f/(1.f+expf(-x));}
 fast_sigmoid_lut++;float q=fast_table(fast_sigmoid_tail_table,a);
 return x>=0.f?1.f-q:q;
}
static float fast_tanh(float x){
 float a=fabsf(x);
 if(!isfinite(a)){fast_tanh_fallback++;cost_tanh++;return tanhf(x);}
 if(a>=10.f){fast_tanh_saturated++;return copysignf(1.f,x);}
 fast_tanh_lut++;return copysignf(fast_table(fast_tanh_table,a),x);
}
static float fast_erf(float x){
 float a=fabsf(x);
 if(!isfinite(a)){fast_erf_fallback++;cost_erf++;return erff(x);}
 if(a>=6.f){fast_erf_saturated++;return copysignf(1.f,x);}
 fast_erf_lut++;return copysignf(fast_table(fast_erf_table,a),x);
}
/* Same interpolant, four independent lanes; no FMA contraction. */
static float32x4_t fast_nonlinear4(float32x4_t x,unsigned kind){
 float lane[4];vst1q_f32(lane,x);float limit=kind==0?18.f:kind==1?10.f:6.f;
 for(unsigned j=0;j<4;j++)if(!isfinite(lane[j])||fabsf(lane[j])>=limit){
  for(unsigned k=0;k<4;k++)lane[k]=kind==0?fast_sigmoid(lane[k]):kind==1?fast_tanh(lane[k]):fast_erf(lane[k]);
  return vld1q_f32(lane);
 }
 const float*table=kind==0?fast_sigmoid_tail_table:kind==1?fast_tanh_table:fast_erf_table;
 float32x4_t at=vmulq_n_f32(vabsq_f32(x),512.f);
 uint32x4_t index=vcvtq_u32_f32(at);
 float32x4_t fraction=vsubq_f32(at,vcvtq_f32_u32(index));
 float32x4_t a=vldrwq_gather_shifted_offset_f32(table,index);
 float32x4_t b=vldrwq_gather_shifted_offset_f32(table,vaddq_n_u32(index,1));
 float32x4_t y=vaddq_f32(a,vmulq_f32(fraction,vsubq_f32(b,a)));
 if(kind==0){fast_sigmoid_lut+=4;y=vpselq_f32(vsubq_f32(vdupq_n_f32(1.f),y),y,vcmpgeq_n_f32(x,0.f));}
 else{
  if(kind==1)fast_tanh_lut+=4;else fast_erf_lut+=4;
  uint32x4_t sign=vandq_u32(vreinterpretq_u32_f32(x),vdupq_n_u32(0x80000000u));
  y=vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(y),sign));
 }
 fast_lut_vector_groups++;fast_lut_vector_gathers+=2;return y;
}
static void fast_math_report(const char*prefix){
 const char*names[]={"sigmoid_lut","tanh_lut","erf_lut","sigmoid_fallback","tanh_fallback","erf_fallback","tanh_saturated","erf_saturated","vector_groups","vector_gathers"};
 uint64_t values[]={fast_sigmoid_lut,fast_tanh_lut,fast_erf_lut,fast_sigmoid_fallback,fast_tanh_fallback,fast_erf_fallback,fast_tanh_saturated,fast_erf_saturated,fast_lut_vector_groups,fast_lut_vector_gathers};
 for(unsigned i=0;i<10;i++){char key[100];namefmt(key,sizeof(key),"%s_fast_math_%s",prefix,names[i]);bench_metric(key,values[i]);}
}
