/* INT8 row-major weights with dynamic, optionally split-I16 activations.
 * q = 256*hi+lo, both signed I8; q is clamped to +/-32511 so high fits.
 * Matrix rows contain exactly ci elements, with predicated final loads.
 */
#ifndef FAST_I8_DENSE_H
#define FAST_I8_DENSE_H
#include <arm_mve.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifndef FAST_I8_ACT_SPLIT16
#define FAST_I8_ACT_SPLIT16 1
#endif
#define FAST_I8_MAX_INPUTS 2048
typedef struct {
 uint64_t matrix_calls,quantized_activation_elements,integer_mac,logical_mac;
 uint64_t dot_instructions,weight_vector_loads,activation_vector_loads;
 uint64_t quantized_rows,output_elements,tail_vectors,zero_rows,loop_iterations;
} FastI8Stats;
static FastI8Stats fast_i8_stats;
/* Exact conversion fast path; retain full I64 fallback for unusually large
 * dot products, rather than overflowing or changing the final rounding. */
static float fast_i8_dot_to_float(int64_t value){
 if(value>=INT32_MIN&&value<=INT32_MAX)return(float)(int32_t)value;
 return(float)value;
}
static int fast_i8_dense(const float*x,const int8_t*w,const float*scales,
 const float*b,float*y,unsigned ci,unsigned co,int accumulate){
 if(!ci||ci>FAST_I8_MAX_INPUTS||!co)return -1;
 int8_t hi[FAST_I8_MAX_INPUTS],lo[FAST_I8_MAX_INPUTS];float maximum=0;
 for(unsigned i=0;i<ci;i++){if(!isfinite(x[i]))return -1;float a=fabsf(x[i]);if(a>maximum)maximum=a;}
 fast_i8_stats.matrix_calls++;fast_i8_stats.quantized_rows++;
 fast_i8_stats.quantized_activation_elements+=ci;fast_i8_stats.output_elements+=co;
 if(maximum==0){for(unsigned o=0;o<co;o++)y[o]=(b?b[o]:0)+(accumulate?y[o]:0.f);fast_i8_stats.zero_rows++;return 0;}
 const float qmax=FAST_I8_ACT_SPLIT16?32511.f:127.f;
 float inverse=qmax/maximum,scale=maximum/qmax;
 unsigned quant_at=0;
 for(;quant_at+4<=ci;quant_at+=4){
  int32x4_t q=vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+quant_at),inverse));
  q=vmaxq_s32(vdupq_n_s32(-(int)qmax),vminq_s32(vdupq_n_s32((int)qmax),q));
  if(FAST_I8_ACT_SPLIT16){
   int32x4_t h=vshrq_n_s32(vaddq_n_s32(q,128),8);
   int32x4_t l=vsubq_s32(q,vshlq_n_s32(h,8));
   vstrbq_s32(hi+quant_at,h);vstrbq_s32(lo+quant_at,l);
  }else vstrbq_s32(hi+quant_at,q);
 }
 for(unsigned i=quant_at;i<ci;i++){
  int q=(int)nearbyintf(x[i]*inverse);
  if(q>(int)qmax)q=(int)qmax;
  if(q<-(int)qmax)q=-(int)qmax;
  if(FAST_I8_ACT_SPLIT16){int h=q/256,l=q-h*256;if(l>127){h++;l-=256;}else if(l< -128){h--;l+=256;}hi[i]=(int8_t)h;lo[i]=(int8_t)l;}
  else{hi[i]=(int8_t)q;lo[i]=0;}
 }
 unsigned blocks=(ci+15)/16,lanes=FAST_I8_ACT_SPLIT16?2:1;
 unsigned tiles=co/4+((co%4)+1)/2;
 fast_i8_stats.logical_mac+=(uint64_t)ci*co;
 fast_i8_stats.integer_mac+=(uint64_t)blocks*16*co*lanes;
 fast_i8_stats.dot_instructions+=(uint64_t)blocks*co*lanes;
 fast_i8_stats.weight_vector_loads+=(uint64_t)blocks*co;
 fast_i8_stats.activation_vector_loads+=(uint64_t)blocks*tiles*lanes;
 fast_i8_stats.loop_iterations+=(uint64_t)blocks*tiles;
 if(ci%16)fast_i8_stats.tail_vectors+=(uint64_t)co;
 unsigned o=0;
 /* Full four-output tiles: each high/low activation vector is loaded once.
  * Integer accumulators and final scalar float expression are unchanged. */
 for(;o+4<=co;o+=4){
  int32_t h0=0,l0=0,h1=0,l1=0,h2=0,l2=0,h3=0,l3=0;
  unsigned i=0;
  for(;i+16<=ci;i+=16){
   int8x16_t a=vld1q_s8(hi+i),z;
   if(FAST_I8_ACT_SPLIT16)z=vld1q_s8(lo+i);
   int8x16_t w0=vld1q_s8(w+(size_t)(o+0)*ci+i);
   h0=vmladavaq_s8(h0,a,w0);if(FAST_I8_ACT_SPLIT16)l0=vmladavaq_s8(l0,z,w0);
   int8x16_t w1=vld1q_s8(w+(size_t)(o+1)*ci+i);
   h1=vmladavaq_s8(h1,a,w1);if(FAST_I8_ACT_SPLIT16)l1=vmladavaq_s8(l1,z,w1);
   int8x16_t w2=vld1q_s8(w+(size_t)(o+2)*ci+i);
   h2=vmladavaq_s8(h2,a,w2);if(FAST_I8_ACT_SPLIT16)l2=vmladavaq_s8(l2,z,w2);
   int8x16_t w3=vld1q_s8(w+(size_t)(o+3)*ci+i);
   h3=vmladavaq_s8(h3,a,w3);if(FAST_I8_ACT_SPLIT16)l3=vmladavaq_s8(l3,z,w3);
  }
  if(i<ci){
   mve_pred16_t mask=vctp8q(ci-i);
   int8x16_t a=vldrbq_z_s8(hi+i,mask),z;
   if(FAST_I8_ACT_SPLIT16)z=vldrbq_z_s8(lo+i,mask);
   int8x16_t w0=vldrbq_z_s8(w+(size_t)(o+0)*ci+i,mask);
   h0=vmladavaq_s8(h0,a,w0);if(FAST_I8_ACT_SPLIT16)l0=vmladavaq_s8(l0,z,w0);
   int8x16_t w1=vldrbq_z_s8(w+(size_t)(o+1)*ci+i,mask);
   h1=vmladavaq_s8(h1,a,w1);if(FAST_I8_ACT_SPLIT16)l1=vmladavaq_s8(l1,z,w1);
   int8x16_t w2=vldrbq_z_s8(w+(size_t)(o+2)*ci+i,mask);
   h2=vmladavaq_s8(h2,a,w2);if(FAST_I8_ACT_SPLIT16)l2=vmladavaq_s8(l2,z,w2);
   int8x16_t w3=vldrbq_z_s8(w+(size_t)(o+3)*ci+i,mask);
   h3=vmladavaq_s8(h3,a,w3);if(FAST_I8_ACT_SPLIT16)l3=vmladavaq_s8(l3,z,w3);
  }
  int64_t q0=FAST_I8_ACT_SPLIT16?(int64_t)h0*256+l0:h0;
  y[o+0]=(fast_i8_dot_to_float(q0)*scale)*scales[o+0]+(b?b[o+0]:0.f)+(accumulate?y[o+0]:0.f);
  int64_t q1=FAST_I8_ACT_SPLIT16?(int64_t)h1*256+l1:h1;
  y[o+1]=(fast_i8_dot_to_float(q1)*scale)*scales[o+1]+(b?b[o+1]:0.f)+(accumulate?y[o+1]:0.f);
  int64_t q2=FAST_I8_ACT_SPLIT16?(int64_t)h2*256+l2:h2;
  y[o+2]=(fast_i8_dot_to_float(q2)*scale)*scales[o+2]+(b?b[o+2]:0.f)+(accumulate?y[o+2]:0.f);
  int64_t q3=FAST_I8_ACT_SPLIT16?(int64_t)h3*256+l3:h3;
  y[o+3]=(fast_i8_dot_to_float(q3)*scale)*scales[o+3]+(b?b[o+3]:0.f)+(accumulate?y[o+3]:0.f);
 }
 /* At most three remaining outputs keep the exact original pair path. */
 for(;o<co;o+=2){
  int32_t h0=0,l0=0,h1=0,l1=0;unsigned pair=o+1<co;
  unsigned i=0;
  for(;i+16<=ci;i+=16){
   int8x16_t a=vld1q_s8(hi+i),z;
   if(FAST_I8_ACT_SPLIT16)z=vld1q_s8(lo+i);
   int8x16_t weight=vld1q_s8(w+(size_t)o*ci+i);
   h0=vmladavaq_s8(h0,a,weight);if(FAST_I8_ACT_SPLIT16)l0=vmladavaq_s8(l0,z,weight);
   if(pair){weight=vld1q_s8(w+(size_t)(o+1)*ci+i);h1=vmladavaq_s8(h1,a,weight);if(FAST_I8_ACT_SPLIT16)l1=vmladavaq_s8(l1,z,weight);}
  }
  if(i<ci){
   mve_pred16_t mask=vctp8q(ci-i);
   int8x16_t a=vldrbq_z_s8(hi+i,mask),z;
   if(FAST_I8_ACT_SPLIT16)z=vldrbq_z_s8(lo+i,mask);
   int8x16_t weight=vldrbq_z_s8(w+(size_t)o*ci+i,mask);
   h0=vmladavaq_s8(h0,a,weight);if(FAST_I8_ACT_SPLIT16)l0=vmladavaq_s8(l0,z,weight);
   if(pair){weight=vldrbq_z_s8(w+(size_t)(o+1)*ci+i,mask);h1=vmladavaq_s8(h1,a,weight);if(FAST_I8_ACT_SPLIT16)l1=vmladavaq_s8(l1,z,weight);}
  }
  int64_t q0=FAST_I8_ACT_SPLIT16?(int64_t)h0*256+l0:h0;
  y[o]=(fast_i8_dot_to_float(q0)*scale)*scales[o]+(b?b[o]:0.f)+(accumulate?y[o]:0.f);
  if(pair){int64_t q1=FAST_I8_ACT_SPLIT16?(int64_t)h1*256+l1:h1;y[o+1]=(fast_i8_dot_to_float(q1)*scale)*scales[o+1]+(b?b[o+1]:0.f)+(accumulate?y[o+1]:0.f);}
 }
 return 0;
}
#endif
