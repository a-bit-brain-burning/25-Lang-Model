#include "bench.h"
#include <math.h>
#include <string.h>
#include "fast_i8_dense.h"
#include "fast_i8_pair.h"
static int fast_i8_dense_pair_baseline(const float*x0,const float*x1,const int8_t*w,const float*scales,const float*b,float*y0,float*y1,unsigned ci,unsigned co,int accumulate){
 if(!ci||ci>FAST_I8_MAX_INPUTS||!co||co%8||FAST_I8_ACT_SPLIT16)return -1;
 int8_t a[FAST_I8_MAX_INPUTS],c[FAST_I8_MAX_INPUTS];float scale0,scale1;
 int za=pair_prepare(x0,ci,co,a,&scale0),zc=pair_prepare(x1,ci,co,c,&scale1);if(za<0||zc<0)return -1;
 unsigned blocks=(ci+15)/16,tiles=co/4;
 for(unsigned i=ci;i<blocks*16;i++)a[i]=c[i]=0;
 ((void)0);
 ((void)0);((void)0);
 ((void)0);((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 if(ci%16)((void)0);
 ((void)0);((void)0);((void)0);((void)0);
 ((void)0);
 float*outputs[2]={y0,y1};int zero[2]={za,zc};float acts[2]={scale0,scale1};
 for(unsigned o=0;o<co;o+=4){
  int32_t dots[8];packed_dot4_pair(a,c,w+(size_t)(o/8)*blocks*128+(o%8)*16,blocks,dots);
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

static float x[2][2048],scale[16],bias[16],old[2][16],got[2][16];
static int8_t weights[32768];
int main(void){
 if(fast_i8_dot_to_float(1)!=1.f)return 1;
 unsigned inputs[]={1,15,16,17,135,192,217,273,328,688,1536,2048},state=12345,cases=0;
 for(unsigned ci_index=0;ci_index<12;ci_index++)for(unsigned co=8;co<=16;co+=8)for(unsigned pattern=0;pattern<4;pattern++)for(int bias_on=0;bias_on<2;bias_on++)for(int accum=0;accum<2;accum++){
  unsigned ci=inputs[ci_index],blocks=(ci+15)/16;
  memset(weights,0,sizeof(weights));
  for(unsigned r=0;r<2;r++)for(unsigned j=0;j<ci;j++){state=state*1664525u+1013904223u;x[r][j]=pattern==0?0.f:pattern==1?((j+r)%2?1.f:-1.f):pattern==2?(j%5==0?.5f:0.f):(float)((int)(state>>8)-8388608)/8388608.f;}
  for(unsigned o=0;o<co;o++){
   scale[o]=.001f*(o+1);bias[o]=o%2?-.125f:.25f;
   for(unsigned j=0;j<ci;j++){state=state*1664525u+1013904223u;weights[(o/8)*blocks*128+(j/16)*128+(o%8)*16+j%16]=(int8_t)(state>>24);}
   for(unsigned r=0;r<2;r++)old[r][o]=got[r][o]=(r+o)%2?-.5f:.125f;
  }
  if(fast_i8_dense_pair_baseline(x[0],x[1],weights,scale,bias_on?bias:0,old[0],old[1],ci,co,accum))return 2;
  if(fast_i8_dense_pair(x[0],x[1],weights,scale,bias_on?bias:0,got[0],got[1],ci,co,accum))return 3;
  for(unsigned r=0;r<2;r++)if(memcmp(old[r],got[r],co*4)){bench_metric("failed_ci",ci);bench_metric("failed_pattern",pattern);bench_exit(7);}
  cases++;
 }
 bench_metric("pair_matrix_cases",cases);bench_puts("PAIR_ROWS_PROBE_PASS\n");return 0;
}
