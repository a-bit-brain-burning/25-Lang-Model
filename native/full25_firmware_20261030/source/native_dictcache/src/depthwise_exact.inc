/* Original dot(1) is (0+0)+(0+0), then +input*weight. Retain both
 * positive-zero addition and the later sum addition; never contract FMA. */
typedef struct {
 uint64_t generic_calls,adapter_calls,logical_dot1_calls_removed;
 uint64_t vector_groups,scalar_tail_outputs,vector_multiply_groups,vector_math_operations;
 uint64_t vector_loads,vector_stores,scalar_tail_taps,bias_value_reads,weight_value_reads;
 uint64_t input_frame_validity_checks,input_frame_pointers;
 uint64_t original_scalar_tap_MAC_replaced,vector_add_groups,scalar_fallback_calls,scalar_fallback_taps;
 uint64_t range_scan_calls,range_scan_vector_groups,range_scan_vector_loads,range_scan_vector_operations;
 uint64_t range_scan_reductions,range_scan_scalar_elements;
} DepthwiseStats;
static DepthwiseStats depthwise_stats;
/* MVE may flush subnormal operations that scalar VFP preserves. The fast
 * domain has input/weight |v| in[2^-50,2^50] or zero, bias in[2^-100,2^100]
 * or zero. Every rounded product/bias is a multiple of2^-123; <=31tap sums
 * cannot overflow and nonzero cancellation results remain normal. Thus no
 * operation in this domain can produce a subnormal or a NaN/Inf. Exceptional
 * batches retain original scalar arithmetic. Scan integers, preserving flags. */
static int depthwise_range(const float*x,unsigned n,uint32_t lower,uint32_t upper){
 uint32x4_t least=vdupq_n_u32(UINT32_MAX),greatest=vdupq_n_u32(0);unsigned i=0;
 for(;i+4<=n;i+=4){
  uint32x4_t bits=vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x+i)),vdupq_n_u32(0x7fffffffu));
  uint32x4_t nonzero=vpselq_u32(vdupq_n_u32(lower),bits,vcmpeqq_n_u32(bits,0));
  least=vminq_u32(least,nonzero);greatest=vmaxq_u32(greatest,bits);
 }
 uint32_t lo=vminvq_u32(UINT32_MAX,least),hi=vmaxvq_u32(0,greatest);
 for(unsigned j=i;j<n;j++){union{float f;uint32_t u;}a={.f=x[j]};uint32_t bits=a.u&0x7fffffffu;if(bits&&bits<lo)lo=bits;if(bits>hi)hi=bits;}
 ((void)0);((void)0);
 ((void)0);((void)0);
 ((void)0);((void)0);
 return lo>=lower&&hi<=upper;
}
static inline float depthwise_dot1(float input,float weight){
 float product=input*weight;return 0.f+product;
}
static void depthwise_channels(const float*const *row,const unsigned*tap,unsigned count,
 const float*weights,const float*bias,float*out,unsigned channels,int vector_safe){
 const float*wptr[8];
 for(unsigned v=0;v<count;v++)wptr[v]=weights+tap[v]*channels;
 unsigned j=0;
 if(vector_safe)for(;j+4<=channels;j+=4){
  float32x4_t sum=vld1q_f32(bias+j);
  for(unsigned v=0;v<count;v++){
   float32x4_t product=vmulq_f32(vld1q_f32(row[v]+j),vld1q_f32(wptr[v]+j));
   product=vaddq_f32(vdupq_n_f32(0.f),product);
   sum=vaddq_f32(sum,product);
  }
  vst1q_f32(out+j,sum);
 }
 for(;j<channels;j++){
  float sum=bias[j];
  for(unsigned v=0;v<count;v++)sum+=vector_safe?depthwise_dot1(row[v][j],wptr[v][j]):dot(row[v]+j,wptr[v]+j,1);
  out[j]=sum;
 }
}
static void depthwise_record(unsigned channels,unsigned rows,uint64_t valid_taps,unsigned kernel,unsigned adapter,int vector_safe){
 uint64_t logical=valid_taps*channels,groups=vector_safe?channels/4:0,tails=vector_safe?channels%4:channels;
 /* Exactly the old logical dot(1) counters. Profiling itself is not counted
  * as model work, and its elimination is never a target-cycle speed credit. */
 if(vector_safe){((void)0);((void)0);((void)0);}
 ((void)0);((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
 ((void)0);
}
static void depthwise_report(const char*prefix){}
