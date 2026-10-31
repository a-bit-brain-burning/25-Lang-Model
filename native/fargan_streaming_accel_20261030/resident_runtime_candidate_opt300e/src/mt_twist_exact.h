#ifndef MT_TWIST_EXACT_H
#define MT_TWIST_EXACT_H
#include <stdint.h>
#include <arm_mve.h>
/* The original twist is IN PLACE. Split at227 so its lagged read observes
 * updated first-segment words exactly as in the scalar624-iteration loop. */
static inline void mt_twist_group(uint32_t*m,unsigned i,unsigned lag){
 uint32x4_t a=vld1q_u32(m+i),b=vld1q_u32(m+i+1),c=vld1q_u32(m+lag);
 uint32x4_t y=vorrq_u32(vandq_u32(a,vdupq_n_u32(0x80000000u)),vandq_u32(b,vdupq_n_u32(0x7fffffffu)));
 uint32x4_t mask=vsubq_u32(vdupq_n_u32(0),vandq_u32(y,vdupq_n_u32(1)));
 uint32x4_t value=veorq_u32(veorq_u32(c,vshrq_n_u32(y,1)),vandq_u32(mask,vdupq_n_u32(0x9908b0dfu)));
 vst1q_u32(m+i,value);
}
static void mt_twist_exact(uint32_t*m){
 for(unsigned i=0;i<224;i+=4)mt_twist_group(m,i,i+397);
 for(unsigned i=224;i<227;i++){uint32_t y=(m[i]&0x80000000u)|(m[i+1]&0x7fffffffu);m[i]=m[i+397]^(y>>1)^((y&1u)?0x9908b0dfu:0u);}
 for(unsigned i=227;i<623;i+=4)mt_twist_group(m,i,i-227);
 uint32_t y=(m[623]&0x80000000u)|(m[0]&0x7fffffffu);m[623]=m[396]^(y>>1)^((y&1u)?0x9908b0dfu:0u);
}
#endif
