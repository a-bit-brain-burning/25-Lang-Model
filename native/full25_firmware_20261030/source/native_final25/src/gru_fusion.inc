static void fail(const char*);
/* Four independent lanes; preserve the two original GRU update formulas.
 * Explicit counters exclude fast_nonlinear4 internals, counted by fast_math. */
static uint64_t fusion_fg_gru_groups,fusion_acoustic_gru_groups,fusion_glu_groups;
static uint64_t fusion_vector_loads,fusion_vector_stores,fusion_vector_operations;
static void fusion_fg_gru(const float*ig,const float*hg,float*hidden,unsigned width){
 if(width%4)fail("FUSION_GRU_WIDTH");
 for(unsigned j=0;j<width;j+=4){
  float32x4_t r=fast_nonlinear4(vaddq_f32(vld1q_f32(ig+j),vld1q_f32(hg+j)),0);
  float32x4_t z=fast_nonlinear4(vaddq_f32(vld1q_f32(ig+width+j),vld1q_f32(hg+width+j)),0);
  float32x4_t candidate=fast_nonlinear4(vaddq_f32(vld1q_f32(ig+2*width+j),vmulq_f32(r,vld1q_f32(hg+2*width+j))),1);
  float32x4_t state=vaddq_f32(vmulq_f32(vsubq_f32(vld1q_f32(hidden+j),candidate),z),candidate);
  vst1q_f32(hidden+j,state);
 }
 unsigned groups=width/4;((void)0);((void)0);((void)0);((void)0);
}
static void fusion_acoustic_gru(const float*ig,const float*hg,float*hidden,float*out){
 for(unsigned j=0;j<192;j+=4){
  float32x4_t r=fast_nonlinear4(vaddq_f32(vld1q_f32(ig+j),vld1q_f32(hg+j)),0);
  float32x4_t z=fast_nonlinear4(vaddq_f32(vld1q_f32(ig+192+j),vld1q_f32(hg+192+j)),0);
  float32x4_t candidate=fast_nonlinear4(vaddq_f32(vld1q_f32(ig+384+j),vmulq_f32(r,vld1q_f32(hg+384+j))),1);
  float32x4_t state=vaddq_f32(vmulq_f32(vsubq_f32(vdupq_n_f32(1.f),z),candidate),vmulq_f32(z,vld1q_f32(hidden+j)));
  vst1q_f32(hidden+j,state);vst1q_f32(out+j,state);
 }
 ((void)0);((void)0);((void)0);((void)0);
}
static void fusion_glu(float*x,const float*gates,unsigned width){
 if(width%4)fail("FUSION_GLU_WIDTH");
 for(unsigned j=0;j<width;j+=4)vst1q_f32(x+j,vmulq_f32(vld1q_f32(x+j),fast_nonlinear4(vld1q_f32(gates+j),0)));
 unsigned groups=width/4;((void)0);((void)0);((void)0);((void)0);
}
static void fusion_report(const char*prefix){}
