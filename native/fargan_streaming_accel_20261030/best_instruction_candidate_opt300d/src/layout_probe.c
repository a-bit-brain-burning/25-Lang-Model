#include "bench.h"
#include <math.h>
#include <string.h>
#include "fast_i8_dense.h"
#include "fast_i8_pair.h"
static float x[2][2048],scales[24],bias[24],expected[2][24],got[2][24],pairgot[2][24];
static int8_t weights[49152],raw[49152],quantized[2][2048];
int main(void){
 if(fast_i8_dot_to_float(1)!=1.f)return 9;
 unsigned widths[]={1,15,16,17,135,192,217,273,328,688,1536,2048};
 unsigned outputs[]={1,4,5,7,8,9,16,17};unsigned state=12345,cases=0,pairs=0;
 for(unsigned wi=0;wi<12;wi++)for(unsigned oi=0;oi<8;oi++)for(unsigned pattern=0;pattern<4;pattern++)for(unsigned usebias=0;usebias<2;usebias++)for(unsigned acc=0;acc<2;acc++){
  unsigned ci=widths[wi],co=outputs[oi],blocks=(ci+15)/16;memset(weights,0,sizeof(weights));
  float act[2];int zero[2];
  for(unsigned r=0;r<2;r++){
   for(unsigned j=0;j<ci;j++){state=1664525u*state+1013904223u;x[r][j]=pattern==0?0.f:pattern==1?((j+r)%2?1.f:-1.f):pattern==2?(j%5==0?.5f:0.f):(float)((int)(state>>8)-8388608)/8388608.f;}
   zero[r]=pair_prepare(x[r],ci,co,quantized[r],act+r);if(zero[r]<0)return 1;
  }
  for(unsigned o=0;o<co;o++){
   scales[o]=.001f*(o+1);bias[o]=o%2?-.125f:.25f;
   for(unsigned j=0;j<ci;j++){state=1664525u*state+1013904223u;int8_t v=(int8_t)(state>>24);raw[o*ci+j]=v;weights[((o/4*blocks+j/16)*4+o%4)*16+j%16]=v;}
   for(unsigned r=0;r<2;r++){
    float prior=(r+o)%2?-.5f:.125f;got[r][o]=pairgot[r][o]=prior;
    int64_t dot=0;for(unsigned j=0;j<ci;j++)dot+=(int)quantized[r][j]*(int)raw[o*ci+j];
    expected[r][o]=zero[r]?(usebias?bias[o]:0.f)+(acc?prior:0.f):((float)dot*act[r])*scales[o]+(usebias?bias[o]:0.f)+(acc?prior:0.f);
   }
  }
  for(unsigned r=0;r<2;r++){
   if(fast_i8_dense(x[r],weights,scales,usebias?bias:0,got[r],ci,co,acc))return 2;
   if(memcmp(expected[r],got[r],4*co)){bench_metric("failed_ci",ci);bench_metric("failed_co",co);bench_metric("failed_pattern",pattern);bench_exit(7);}
  }
  if(co%8==0){
   if(fast_i8_dense_pair(x[0],x[1],weights,scales,usebias?bias:0,pairgot[0],pairgot[1],ci,co,acc))return 3;
   for(unsigned r=0;r<2;r++)if(memcmp(expected[r],pairgot[r],4*co)){bench_metric("pair_failed_ci",ci);bench_metric("pair_failed_co",co);bench_metric("pair_failed_pattern",pattern);bench_exit(8);}
   pairs++;
  }
  cases++;
 }
 bench_metric("layout_single_matrix_cases",cases);bench_metric("layout_pair_matrix_cases",pairs);bench_puts("PAIR_LAYOUT_PROBE_PASS\n");return 0;
}
