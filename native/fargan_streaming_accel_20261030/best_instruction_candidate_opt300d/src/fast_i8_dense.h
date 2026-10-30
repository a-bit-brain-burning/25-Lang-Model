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
 uint64_t no_clamp_groups,no_clamp_rows;
 uint64_t matrix_asm_calls,matrix_asm_tiles,matrix_context_word_loads,matrix_context_word_stores,matrix_context_initialization_words,matrix_direct_register_moves,matrix_extra_vector_operations;
 uint64_t packed_control_iterations,packed_tile_calls,activation_padding_bytes,asm_saved_words,asm_restored_words,asm_result_store_words;
 uint64_t matrix_calls,quantized_activation_elements,integer_mac,logical_mac;
 uint64_t dot_instructions,weight_vector_loads,activation_vector_loads;
 uint64_t quantized_rows,output_elements,tail_vectors,zero_rows,loop_iterations;
 uint64_t absmax_vector_groups,absmax_reductions,absmax_scalar_tail;
 uint64_t tile8_full_iterations,tile8_tail_iterations;
 uint64_t epilogue_vector_groups,epilogue_vector_loads,epilogue_vector_stores,epilogue_vector_operations,epilogue_scalar_outputs;
} FastI8Stats;
static FastI8Stats fast_i8_stats;
/* Exact conversion fast path; retain full I64 fallback for unusually large
 * dot products, rather than overflowing or changing the final rounding. */
static float fast_i8_dot_to_float(int64_t value){
 if(value>=INT32_MIN&&value<=INT32_MAX)return(float)(int32_t)value;
 return(float)value;
}
typedef struct {const int8_t*x,*w;const float*scales,*bias;float*y;unsigned blocks,tiles;float scale;unsigned accumulate;} PackedMatrix8;
extern void packed_matrix8(PackedMatrix8*);
extern void packed_dot8(const int8_t*,const int8_t*,unsigned,int32_t*);
static int fast_i8_dense(const float*x,const int8_t*w,const float*scales,
 const float*b,float*y,unsigned ci,unsigned co,int accumulate){
 if(!ci||ci>FAST_I8_MAX_INPUTS||!co)return -1;
 int8_t hi[FAST_I8_MAX_INPUTS],lo[FAST_I8_MAX_INPUTS];float maximum=0;
 /* Positive IEEE-754 bit patterns have the same ordering as finite abs(x).
  * Integer maximum also preserves NaN/Inf detection: either exceeds the
  * largest finite encoding. No floating reduction may silently ignore NaN. */
 uint32x4_t maxbits4=vdupq_n_u32(0);unsigned max_at=0;
 const uint32x4_t absmask=vdupq_n_u32(0x7fffffffu);
 for(;max_at+8<=ci;max_at+=8){
  uint32x4_t bits0=vreinterpretq_u32_f32(vld1q_f32(x+max_at));
  uint32x4_t bits1=vreinterpretq_u32_f32(vld1q_f32(x+max_at+4));
  maxbits4=vmaxq_u32(maxbits4,vandq_u32(bits0,absmask));
  maxbits4=vmaxq_u32(maxbits4,vandq_u32(bits1,absmask));
 }
 for(;max_at+4<=ci;max_at+=4){
  uint32x4_t bits=vreinterpretq_u32_f32(vld1q_f32(x+max_at));
  maxbits4=vmaxq_u32(maxbits4,vandq_u32(bits,absmask));
 }
 uint32_t maxbits=vmaxvq_u32(0,maxbits4);
 for(unsigned i=max_at;i<ci;i++){
  union{float f;uint32_t u;}bits={.f=x[i]};bits.u&=0x7fffffffu;
  if(bits.u>maxbits)maxbits=bits.u;
 }
 if(maxbits>=0x7f800000u)return -1;
 union{float f;uint32_t u;}maximum_bits={.u=maxbits};maximum=maximum_bits.f;
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);((void)0);
 ((void)0);((void)0);
 if(maximum==0){for(unsigned o=0;o<co;o++)y[o]=(b?b[o]:0)+(accumulate?y[o]:0.f);((void)0);((void)0);return 0;}
 const float qmax=FAST_I8_ACT_SPLIT16?32511.f:127.f;
 float inverse=qmax/maximum,scale=maximum/qmax;
 unsigned quant_at=0;
 /* Safe bound:127(1+2^-24)^2 <127.5. The unchanged RNE
  * conversion therefore already produces[-127,127]. Outside this normal
  * maximum range, preserve the complete original clamped vector loop. */
 if(!FAST_I8_ACT_SPLIT16&&maxbits>=0x0d800000u&&maxbits<=0x71800000u){
  for(;quant_at+4<=ci;quant_at+=4){
   int32x4_t q=vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+quant_at),inverse));
   vstrbq_s32(hi+quant_at,q);
  }
  ((void)0);
  ((void)0);
 }else{
 for(;quant_at+4<=ci;quant_at+=4){
  int32x4_t q=vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+quant_at),inverse));
  q=vmaxq_s32(vdupq_n_s32(-(int)qmax),vminq_s32(vdupq_n_s32((int)qmax),q));
  if(FAST_I8_ACT_SPLIT16){
   int32x4_t h=vshrq_n_s32(vaddq_n_s32(q,128),8);
   int32x4_t l=vsubq_s32(q,vshlq_n_s32(h,8));
   vstrbq_s32(hi+quant_at,h);vstrbq_s32(lo+quant_at,l);
  }else vstrbq_s32(hi+quant_at,q);
 }
 }
 for(unsigned i=quant_at;i<ci;i++){
  int q=(int)nearbyintf(x[i]*inverse);
  if(q>(int)qmax)q=(int)qmax;
  if(q<-(int)qmax)q=-(int)qmax;
  if(FAST_I8_ACT_SPLIT16){int h=q/256,l=q-h*256;if(l>127){h++;l-=256;}else if(l< -128){h--;l+=256;}hi[i]=(int8_t)h;lo[i]=(int8_t)l;}
  else{hi[i]=(int8_t)q;lo[i]=0;}
 }
 if(FAST_I8_ACT_SPLIT16)return -1;
 unsigned blocks=(ci+15)/16,tiles=(co+7)/8,padded_co=tiles*8;
 for(unsigned i=ci;i<blocks*16;i++)hi[i]=0;
 ((void)0);
 ((void)0);((void)0);unsigned matrix_tiles=co/8,tail_tiles=(co%8)!=0;
 ((void)0);((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 unsigned output_groups=co/4;
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 if(ci%16)((void)0);
 if(matrix_tiles){PackedMatrix8 context={hi,w,scales,b,y,blocks,matrix_tiles,scale,(unsigned)accumulate};packed_matrix8(&context);}
 for(unsigned o=matrix_tiles*8;o<co;o+=8){
  int32_t dots[8];packed_dot8(hi,w+(size_t)(o/8)*blocks*128,blocks,dots);
  unsigned count=co-o<8?co-o:8,j=0;
  for(;j+4<=count;j+=4){
   float32x4_t value=vcvtq_f32_s32(vld1q_s32(dots+j));
   value=vmulq_n_f32(value,scale);
   value=vmulq_f32(value,vld1q_f32(scales+o+j));
   value=vaddq_f32(value,b?vld1q_f32(b+o+j):vdupq_n_f32(0.f));
   value=vaddq_f32(value,accumulate?vld1q_f32(y+o+j):vdupq_n_f32(0.f));
   vst1q_f32(y+o+j,value);
  }
  for(;j<count;j++)y[o+j]=((float)dots[j]*scale)*scales[o+j]+(b?b[o+j]:0.f)+(accumulate?y[o+j]:0.f);
 }
 return 0;
}
#endif
