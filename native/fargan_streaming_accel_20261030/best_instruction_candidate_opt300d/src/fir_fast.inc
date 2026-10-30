/* Numerical candidate: same 41 coefficients rounded once to FP32; same sample
 * clock, FIR center, circular history and EOS limits. De-emphasis stays FP64. */
#include <arm_mve.h>
static float fir32[41];
static uint32_t fir_history_indices[41][41];
static uint64_t fir_mve_groups,fir_vector_loads,fir_scalar_taps,fir_reductions,fir_outputs;
static void fast_fir_init(void){
 for(unsigned i=0;i<41;i++)fir32[i]=(float)fir[i];
 for(unsigned p=0;p<41;p++)for(unsigned k=0;k<41;k++)fir_history_indices[p][k]=(p+41-k)%41;
 fir_mve_groups=fir_vector_loads=fir_scalar_taps=fir_reductions=fir_outputs=0;
}
static float fast_fir_value(unsigned center){
 const uint32_t*indices=fir_history_indices[center%41];float32x4_t acc=vdupq_n_f32(0.f);
 for(unsigned k=0;k<40;k+=4){
  uint32x4_t at=vld1q_u32(indices+k);
  float32x4_t h=vldrwq_gather_shifted_offset_f32(history,at);
  acc=vaddq_f32(acc,vmulq_f32(vld1q_f32(fir32+k),h));
 }
 float lane[4];vst1q_f32(lane,acc);
 float value=(lane[0]+lane[1])+(lane[2]+lane[3]);
 value+=fir32[40]*history[indices[40]];
 ((void)0);((void)0);((void)0);((void)0);((void)0);
 return value;
}
static void fast_fir_report(int first){}
