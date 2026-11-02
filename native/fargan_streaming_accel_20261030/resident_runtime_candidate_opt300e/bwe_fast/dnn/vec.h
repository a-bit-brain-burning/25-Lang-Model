/* Copyright (c) 2018 Mozilla
                 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef VEC_H
#define VEC_H

#include "opus_types.h"
#include <math.h>
#include "arch.h"
#include "x86/x86_arch_macros.h"


#if (defined(__AVX__) || defined(__SSE2__)) && !defined(OPUS_BWE_SCALAR)
#include "vec_avx.h"
#elif (defined(__ARM_NEON__) || defined(__ARM_NEON)) && !defined(DISABLE_NEON) && !defined(OPUS_BWE_SCALAR)
#include "vec_neon.h"
#else

#include "os_support.h"

#define MAX_INPUTS (2048)

#define NO_OPTIMIZATIONS

#include <arm_mve.h>
#include "bwe_quant.h"
extern unsigned long long bwe_quant_groups;
extern unsigned long long bwe_mve_vectors,bwe_scalar_macs,bwe_i8_dots,bwe_i8_inputs;
static inline void sgemv(float *out,const float *weights,int rows,int cols,int stride,const float*x){
 int i=0;
 for(;i+16<=rows;i+=16){
  float32x4_t a=vdupq_n_f32(0),b=a,c=a,d=a;
  const float*restrict wp=weights+i;
  for(int j=0;j<cols;j++,wp+=stride){
   float32x4_t v=vdupq_n_f32(x[j]);
   a=vaddq_f32(a,vmulq_f32(vld1q_f32(wp),v));b=vaddq_f32(b,vmulq_f32(vld1q_f32(wp+4),v));
   c=vaddq_f32(c,vmulq_f32(vld1q_f32(wp+8),v));d=vaddq_f32(d,vmulq_f32(vld1q_f32(wp+12),v));
  }
  vst1q_f32(out+i,a);vst1q_f32(out+i+4,b);vst1q_f32(out+i+8,c);vst1q_f32(out+i+12,d);((void)0);
 }
 for(;i+4<=rows;i+=4){float32x4_t a=vdupq_n_f32(0);const float*restrict wp=weights+i;for(int j=0;j<cols;j++,wp+=stride)a=vaddq_f32(a,vmulq_n_f32(vld1q_f32(wp),x[j]));vst1q_f32(out+i,a);((void)0);}
 for(;i<rows;i++){float a=0;for(int j=0;j<cols;j++)a+=weights[j*stride+i]*x[j];out[i]=a;((void)0);}
}

static inline void sparse_sgemv8x4(float *out, const float *w, const int *idx, int rows, const float *x)
{
   int i, j;
   OPUS_CLEAR(out, rows);
   for (i=0;i<rows;i+=8)
   {
      int cols;
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         int pos;
         float * restrict y;
         float xj0, xj1, xj2, xj3;
         pos = (*idx++);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += w[0]*xj0;
         y[1] += w[1]*xj0;
         y[2] += w[2]*xj0;
         y[3] += w[3]*xj0;
         y[4] += w[4]*xj0;
         y[5] += w[5]*xj0;
         y[6] += w[6]*xj0;
         y[7] += w[7]*xj0;

         y[0] += w[8]*xj1;
         y[1] += w[9]*xj1;
         y[2] += w[10]*xj1;
         y[3] += w[11]*xj1;
         y[4] += w[12]*xj1;
         y[5] += w[13]*xj1;
         y[6] += w[14]*xj1;
         y[7] += w[15]*xj1;

         y[0] += w[16]*xj2;
         y[1] += w[17]*xj2;
         y[2] += w[18]*xj2;
         y[3] += w[19]*xj2;
         y[4] += w[20]*xj2;
         y[5] += w[21]*xj2;
         y[6] += w[22]*xj2;
         y[7] += w[23]*xj2;

         y[0] += w[24]*xj3;
         y[1] += w[25]*xj3;
         y[2] += w[26]*xj3;
         y[3] += w[27]*xj3;
         y[4] += w[28]*xj3;
         y[5] += w[29]*xj3;
         y[6] += w[30]*xj3;
         y[7] += w[31]*xj3;
         w += 32;
      }
   }
}

#ifdef USE_SU_BIAS
static inline void sparse_cgemv8x4(float *out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   unsigned char x[MAX_INPUTS];
   for (i=0;i<rows;i++) out[i] = 0;
   for (i=0;i<cols;i++) x[i] = 127+floor(.5+127*_x[i]);
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      colblocks = *idx++;
      for (j=0;j<colblocks;j++)
      {
         int pos;
         float * restrict y;
         int xj0, xj1, xj2, xj3;
         pos = (*idx++);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
      }
   }
   for (i=0;i<rows;i++) out[i] *= scale[i];
}
static inline void cgemv8x4(float*out,const opus_int8*w,const float*scale,int rows,int cols,const float*_x){
 opus_int8 x[MAX_INPUTS];bwe_quantize_exact(x,_x,cols);((void)0);((void)0);
 for(int i=0;i<rows;i+=4){
  int32_t a=0,b=0,c=0,d=0;
  const opus_int8*row0=w+(size_t)i*cols;const opus_int8*row1=row0+cols;
  const opus_int8*row2=row1+cols;const opus_int8*row3=row2+cols;
  const opus_int8*xp=x;int j=0;
  for(;j+16<=cols;j+=16){
   int8x16_t v=vldrbq_s8(xp);
   a=vmladavaq_s8(a,v,vldrbq_s8(row0));
   b=vmladavaq_s8(b,v,vldrbq_s8(row1));
   c=vmladavaq_s8(c,v,vldrbq_s8(row2));
   d=vmladavaq_s8(d,v,vldrbq_s8(row3));
   xp+=16;row0+=16;row1+=16;row2+=16;row3+=16;
  }
  if(j<cols){
   mve_pred16_t p=vctp8q(cols-j);int8x16_t v=vldrbq_z_s8(xp,p);
   a=vmladavaq_s8(a,v,vldrbq_z_s8(row0,p));
   b=vmladavaq_s8(b,v,vldrbq_z_s8(row1,p));
   c=vmladavaq_s8(c,v,vldrbq_z_s8(row2,p));
   d=vmladavaq_s8(d,v,vldrbq_z_s8(row3,p));
  }
  int32_t sums[4]={a,b,c,d};vst1q_f32(out+i,vmulq_f32(vcvtq_f32_s32(vld1q_s32(sums)),vld1q_f32(scale+i)));((void)0);
 }
}
#else
static inline void sparse_cgemv8x4(float *out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   opus_int8 x[MAX_INPUTS];
   for (i=0;i<rows;i++) out[i] = 0;
   for (i=0;i<cols;i++) x[i] = (int)floor(.5+127*_x[i]);
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      colblocks = *idx++;
      for (j=0;j<colblocks;j++)
      {
         int pos;
         float * restrict y;
         int xj0, xj1, xj2, xj3;
         pos = (*idx++);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
      }
   }
   for (i=0;i<rows;i++) out[i] *= scale[i];
}
static inline void cgemv8x4(float*out,const opus_int8*w,const float*scale,int rows,int cols,const float*_x){
 opus_int8 x[MAX_INPUTS];bwe_quantize_exact(x,_x,cols);((void)0);((void)0);
 for(int i=0;i<rows;i+=4){
  int32_t a=0,b=0,c=0,d=0;
  const opus_int8*row0=w+(size_t)i*cols;const opus_int8*row1=row0+cols;
  const opus_int8*row2=row1+cols;const opus_int8*row3=row2+cols;
  const opus_int8*xp=x;int j=0;
  for(;j+16<=cols;j+=16){
   int8x16_t v=vldrbq_s8(xp);
   a=vmladavaq_s8(a,v,vldrbq_s8(row0));
   b=vmladavaq_s8(b,v,vldrbq_s8(row1));
   c=vmladavaq_s8(c,v,vldrbq_s8(row2));
   d=vmladavaq_s8(d,v,vldrbq_s8(row3));
   xp+=16;row0+=16;row1+=16;row2+=16;row3+=16;
  }
  if(j<cols){
   mve_pred16_t p=vctp8q(cols-j);int8x16_t v=vldrbq_z_s8(xp,p);
   a=vmladavaq_s8(a,v,vldrbq_z_s8(row0,p));
   b=vmladavaq_s8(b,v,vldrbq_z_s8(row1,p));
   c=vmladavaq_s8(c,v,vldrbq_z_s8(row2,p));
   d=vmladavaq_s8(d,v,vldrbq_z_s8(row3,p));
  }
  int32_t sums[4]={a,b,c,d};vst1q_f32(out+i,vmulq_f32(vcvtq_f32_s32(vld1q_s32(sums)),vld1q_f32(scale+i)));((void)0);
 }
}
#endif

/* No AVX2/FMA support */
#ifndef LPCNET_TEST
static inline float lpcnet_exp2(float x)
{
   int integer;
   float frac;
   union {
      float f;
      opus_uint32 i;
   } res;
   integer = floor(x);
   if (integer < -50)
      return 0;
   frac = x-integer;
   /* K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2 */
   res.f = 0.99992522f + frac * (0.69583354f
           + frac * (0.22606716f + 0.078024523f*frac));
   res.i = (res.i + (integer<<23)) & 0x7fffffff;
   return res.f;
}
#define lpcnet_exp(x) lpcnet_exp2((x)*1.44269504f)

#define fmadd(a, b, c) ((a)*(b)+(c))
static OPUS_INLINE float tanh_approx(float x)
{
    const float N0 = 952.52801514f;
    const float N1 = 96.39235687f;
    const float N2 = 0.60863042f;
    const float D0 = 952.72399902f;
    const float D1 = 413.36801147f;
    const float D2 = 11.88600922f;
    float X2, num, den;
    X2 = x*x;
    num = fmadd(fmadd(N2, X2, N1), X2, N0);
    den = fmadd(fmadd(D2, X2, D1), X2, D0);
    num = num*x/den;
    return MAX32(-1.f, MIN32(1.f, num));
}

static inline float sigmoid_approx(float x)
{
   return .5f + .5f*tanh_approx(.5f*x);
}

extern unsigned long long bwe_exp_guard_vectors,bwe_exp_fast_vectors,bwe_exp_fallback_vectors,bwe_exp_scalar_elements;
extern unsigned long long bwe_exp_vector_arithmetic_operations,bwe_exp_guard_operations,bwe_exp_explicit_vector_loads,bwe_exp_explicit_vector_stores;
/* The ORIGINAL lpcnet_exp2 polynomial, not a replacement exp approximation.
 * Raw-bit guard permits0 or|input| in[2^-30,32]. Scaled floor is[-47,46],
 * fractions and Horner products are normal or zero, and exponent addition
 * cannot overflow. Domain exceptions preserve the original scalar routine. */
static inline float32x4_t bwe_exp4_exact_checked(float32x4_t x,unsigned*fallback){
 uint32x4_t bits=vandq_u32(vreinterpretq_u32_f32(x),vdupq_n_u32(0x7fffffffu));
 uint32x4_t nonzero=vpselq_u32(vdupq_n_u32(0x30800000u),bits,vcmpeqq_n_u32(bits,0));
 uint32x4_t range=vsubq_n_u32(nonzero,0x30800000u);
 if(vcmphiq_n_u32(range,0x11800000u)){
  float a[4],b[4];vst1q_f32(a,x);
  for(unsigned k=0;k<4;k++)b[k]=lpcnet_exp(a[k]);
  (*fallback)++;return vld1q_f32(b);
 }
 float32x4_t scaled=vmulq_n_f32(x,1.44269504f);
 int32x4_t integer=vcvtmq_s32_f32(scaled);
 float32x4_t frac=vsubq_f32(scaled,vcvtq_f32_s32(integer));
 float32x4_t value=vmulq_n_f32(frac,0.078024523f);
 value=vaddq_f32(vdupq_n_f32(0.22606716f),value);
 value=vmulq_f32(frac,value);
 value=vaddq_f32(vdupq_n_f32(0.69583354f),value);
 value=vmulq_f32(frac,value);
 value=vaddq_f32(vdupq_n_f32(0.99992522f),value);
 uint32x4_t exponent=vshlq_n_u32(vreinterpretq_u32_s32(integer),23);
 return vreinterpretq_f32_u32(vandq_u32(vaddq_u32(vreinterpretq_u32_f32(value),exponent),vdupq_n_u32(0x7fffffffu)));
}
static inline void softmax(float *y, const float *x, int N)
{
 if(N<=0)return;
 int i=0;unsigned fallback=0;
 for(;i+4<=N;i+=4)vst1q_f32(y+i,bwe_exp4_exact_checked(vld1q_f32(x+i),&fallback));
 unsigned groups=i/4,tail=N-i;
 for(;i<N;i++)y[i]=lpcnet_exp(x[i]);
 ((void)0);((void)0);((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);((void)0);
}

extern unsigned long long bwe_activation_calls[8],bwe_activation_elements[8];
extern unsigned long long bwe_act_guard_vectors,bwe_act_fallback_vectors,bwe_act_fallback_elements;
extern unsigned long long bwe_act_rational_vectors,bwe_act_scalar_divisions,bwe_act_scalar_tail_elements,bwe_act_vector_operations,bwe_act_vector_loads,bwe_act_vector_stores;
/* Same Horner order and explicit multiply/add as scalar tanh_approx.
 * MVE has no vector divide: retain FOUR exact FP32 scalar VDIV operations.
 * Comparison/select clamp matches MIN32/MAX32 for NaN and signed zero too. */
static inline float32x4_t bwe_tanh4_exact(float32x4_t x){
 float32x4_t x2=vmulq_f32(x,x);
 float32x4_t num=vaddq_f32(vmulq_f32(vaddq_f32(vmulq_n_f32(x2,.60863042f),vdupq_n_f32(96.39235687f)),x2),vdupq_n_f32(952.52801514f));
 float32x4_t den=vaddq_f32(vmulq_f32(vaddq_f32(vmulq_n_f32(x2,11.88600922f),vdupq_n_f32(413.36801147f)),x2),vdupq_n_f32(952.72399902f));
 float a[4],b[4],q[4];vst1q_f32(a,vmulq_f32(num,x));vst1q_f32(b,den);
 q[0]=a[0]/b[0];q[1]=a[1]/b[1];q[2]=a[2]/b[2];q[3]=a[3]/b[3];
 float32x4_t out=vld1q_f32(q),hi=vdupq_n_f32(1),lo=vdupq_n_f32(-1);
 out=vpselq_f32(hi,out,vcmpgtq_f32(out,hi));out=vpselq_f32(lo,out,vcmpltq_f32(out,lo));

 return out;
}
/* MVE FP math flushes subnormal inputs and vector LT is unordered for NaN.
 * Guard raw bits BEFORE any FP operation. Zero/subnormal/large/nonfinite groups
 * use the exact original scalar function; common normal groups retain SIMD. */
static inline float32x4_t bwe_activation4_checked(float32x4_t x,int sigmoid_,unsigned*fallbacks){
 uint32x4_t bits=vandq_u32(vreinterpretq_u32_f32(x),vdupq_n_u32(0x7fffffffu));
 uint32x4_t range=vsubq_n_u32(bits,0x00800000u);

 if(vcmphiq_n_u32(range,0x49000000u)){
  float a[4],b[4];vst1q_f32(a,x);
  for(unsigned k=0;k<4;k++)b[k]=sigmoid_?sigmoid_approx(a[k]):tanh_approx(a[k]);
  (void)fallbacks; /* observation count disabled */
  return vld1q_f32(b);
 }
 if(sigmoid_)x=vmulq_n_f32(x,.5f);
 float32x4_t q=bwe_tanh4_exact(x);
 if(sigmoid_)q=vaddq_f32(vdupq_n_f32(.5f),vmulq_n_f32(q,.5f));
 return q;
}
/* Aggregate observation once per activation call, never per vector group. */
static inline void bwe_count_activation_vectors(unsigned groups,unsigned fallback,unsigned tail,unsigned sigmoid_){
 unsigned normal=groups-fallback;
 ((void)0);((void)0);((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);
 ((void)0);((void)0);
}
static inline void vec_tanh(float *y, const float *x, int N)
{
 int i=0;unsigned fallback=0;
 for(;i+4<=N;i+=4){float32x4_t q=bwe_activation4_checked(vld1q_f32(x+i),0,&fallback);vst1q_f32(y+i,q);}
 unsigned groups=i/4;
 for(;i<N;i++)y[i]=tanh_approx(x[i]);
 bwe_count_activation_vectors(groups,fallback,N%4,0);
}
static inline void vec_sigmoid(float *y, const float *x, int N)
{
 int i=0;unsigned fallback=0;
 for(;i+4<=N;i+=4){float32x4_t q=bwe_activation4_checked(vld1q_f32(x+i),1,&fallback);vst1q_f32(y+i,q);}
 unsigned groups=i/4;
 for(;i<N;i++)y[i]=sigmoid_approx(x[i]);
 bwe_count_activation_vectors(groups,fallback,N%4,1);
}

#endif

#define SCALE (128.f*127.f)
#define SCALE_1 (1.f/128.f/127.f)

#endif /*no optimizations*/
#endif /*VEC_H*/
