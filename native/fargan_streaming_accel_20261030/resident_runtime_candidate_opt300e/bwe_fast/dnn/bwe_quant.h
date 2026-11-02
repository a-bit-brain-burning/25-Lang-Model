#ifndef BWE_QUANT_H
#define BWE_QUANT_H
#include <arm_mve.h>
#include <math.h>
/* Original rounds float product p by floor((double)p + .5). RNE plus
 * positive-halfway correction avoids a float .5 addition rounding p. */
static inline void bwe_quantize_exact(signed char *out,const float *x,int n){
 int j=0;
 for(;j+4<=n;j+=4){
  float32x4_t p=vmulq_n_f32(vld1q_f32(x+j),127.f);
  /* Original conversion is defined only in int32 range. Fallback keeps
   * scalar behavior for unusual large values; normal network path is SIMD. */
  if(vmaxvq_u32(0,vreinterpretq_u32_f32(vabsq_f32(p)))>0x4a800000u){
   for(int k=0;k<4;k++)out[j+k]=(int)floor(.5+127*x[j+k]);
   continue;
  }
  int32x4_t q=vcvtnq_s32_f32(p);
  mve_pred16_t tie=vcmpeqq_n_f32(vsubq_f32(p,vcvtq_f32_s32(q)),.5f);
  q=vaddq_s32(q,vpselq_s32(vdupq_n_s32(1),vdupq_n_s32(0),tie));
  vstrbq_s32(out+j,q);
 }
 for(;j<n;j++)out[j]=(int)floor(.5+127*x[j]);
}
#endif
