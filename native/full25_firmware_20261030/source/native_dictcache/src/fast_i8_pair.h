#ifndef FAST_I8_PAIR_H
#define FAST_I8_PAIR_H
#include <string.h>
extern void packed_dot4_pair(const int8_t*,const int8_t*,const int8_t*,unsigned,int32_t*);
typedef struct {uint64_t matrix_pairs,helper_calls,block_iterations,extra_control_instructions,zero_rows;} FastI8PairStats;
static FastI8PairStats fast_i8_pair_stats;
typedef struct {uint64_t calls,tiles,context_load_words,context_store_words,context_initialization_words,direct_register_moves,extra_vector_operations,outer_scalar_instructions;} FastPairMatrixStats;
static FastPairMatrixStats fast_pair_matrix_stats;
typedef struct {const int8_t*x0,*x1,*w;const float*scales,*bias;float*y0,*y1;unsigned blocks,tiles;float scale0,scale1;unsigned accumulate;} PackedMatrixPair;
extern void packed_matrix_pair(PackedMatrixPair*);

static int pair_prepare(const float*x,unsigned ci,unsigned co,int8_t*hi,float*scale_out){
 float maximum=0;int8_t lo[FAST_I8_MAX_INPUTS];
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
 if(maximum==0){*scale_out=0;for(unsigned at=0;at<((ci+15u)&~15u);at+=16)memset(hi+at*2,0,16);return 1;}
 const float qmax=FAST_I8_ACT_SPLIT16?32511.f:127.f;
 float inverse=qmax/maximum,scale=maximum/qmax;
 unsigned quant_at=0;
 /* Safe bound:127(1+2^-24)^2 <127.5. The unchanged RNE
  * conversion therefore already produces[-127,127]. Outside this normal
  * maximum range, preserve the complete original clamped vector loop. */
 if(!FAST_I8_ACT_SPLIT16&&maxbits>=0x0d800000u&&maxbits<=0x71800000u){
  for(;quant_at+8<=ci;quant_at+=8){
   int32x4_t q0=vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+quant_at),inverse));
   int32x4_t q1=vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+quant_at+4),inverse));
   vstrbq_s32(hi+quant_at+(quant_at&~15u),q0);vstrbq_s32(hi+quant_at+(quant_at&~15u)+4,q1);
  }
  for(;quant_at+4<=ci;quant_at+=4){
   int32x4_t q=vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+quant_at),inverse));
   vstrbq_s32(hi+quant_at+(quant_at&~15u),q);
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
   vstrbq_s32(hi+quant_at+(quant_at&~15u),h);vstrbq_s32(lo+quant_at,l);
  }else vstrbq_s32(hi+quant_at+(quant_at&~15u),q);
 }
 }
 for(unsigned i=quant_at;i<ci;i++){
  int q=(int)nearbyintf(x[i]*inverse);
  if(q>(int)qmax)q=(int)qmax;
  if(q<-(int)qmax)q=-(int)qmax;
  if(FAST_I8_ACT_SPLIT16){int h=q/256,l=q-h*256;if(l>127){h++;l-=256;}else if(l< -128){h--;l+=256;}hi[i+(i&~15u)]=(int8_t)h;lo[i]=(int8_t)l;}
  else{hi[i+(i&~15u)]=(int8_t)q;lo[i]=0;}
 }
 *scale_out=scale;return 0;
}
static int fast_i8_dense_pair(const float*x0,const float*x1,const int8_t*w,const float*scales,const float*b,float*y0,float*y1,unsigned ci,unsigned co,int accumulate){
 if(!ci||ci>FAST_I8_MAX_INPUTS||!co||co%8||FAST_I8_ACT_SPLIT16)return -1;
 int8_t interleaved[FAST_I8_MAX_INPUTS*2];int8_t*a=interleaved,*c=interleaved+16;float scale0,scale1;
 int za=pair_prepare(x0,ci,co,a,&scale0),zc=pair_prepare(x1,ci,co,c,&scale1);if(za<0||zc<0)return -1;
 unsigned blocks=(ci+15)/16,tiles=co/4;
 for(unsigned i=ci;i<blocks*16;i++)a[i+(i&~15u)]=c[i+(i&~15u)]=0;
 ((void)0);
 ((void)0);((void)0);
 int use_matrix=za!=1&&zc!=1;
 ((void)0);((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 if(ci%16)((void)0);
 ((void)0);((void)0);((void)0);((void)0);
 ((void)0);
 if(use_matrix){
  ((void)0);((void)0);
  ((void)0);
  ((void)0);
  ((void)0);
  ((void)0);
  ((void)0);
  /* Tile phase branch/weight-start update and scalar setup/control, counted
   * separately from actual subs/bne inner control and per-block pointer add. */
  ((void)0);
  ((void)0);
  ((void)0);
  ((void)0);
  ((void)0);
  PackedMatrixPair context={a,c,w,scales,b,y0,y1,blocks,tiles,scale0,scale1,(unsigned)accumulate};
  packed_matrix_pair(&context);return 0;
 }
 float*outputs[2]={y0,y1};int zero[2]={za,zc};float acts[2]={scale0,scale1};
 for(unsigned o=0;o<co;o+=4){
  int32_t dots[8];packed_dot4_pair(a,c,w+(size_t)(o/4)*blocks*64,blocks,dots);
  for(unsigned row=0;row<2;row++){
   float*y=outputs[row];
   if(zero[row]){for(unsigned j=0;j<4;j++)y[o+j]=(b?b[o+j]:0)+(accumulate?y[o+j]:0.f);((void)0);continue;}
   float32x4_t value=vcvtq_f32_s32(vld1q_s32(dots+4*row));
   value=vmulq_n_f32(value,acts[row]);value=vmulq_f32(value,vld1q_f32(scales+o));
   value=vaddq_f32(value,b?vld1q_f32(b+o):vdupq_n_f32(0.f));
   value=vaddq_f32(value,accumulate?vld1q_f32(y+o):vdupq_n_f32(0.f));vst1q_f32(y+o,value);
   ((void)0);((void)0);((void)0);((void)0);
  }
 }
 return 0;
}
#endif
