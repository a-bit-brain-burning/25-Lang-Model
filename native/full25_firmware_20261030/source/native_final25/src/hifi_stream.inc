/* Original HiFi graph evaluated once per output position, with bounded queues.
 * Include after the original helpers and adapter_stream.inc, before main(). */
#ifndef HFS_RING_ROWS
#define HFS_RING_ROWS 128
#endif
#define HFS_MAX_NODES 128
#define HFS_BATCH 16
enum {HFS_MEL,HFS_CONV,HFS_UP,HFS_ACT,HFS_MEAN};
typedef struct {
 unsigned type,channels,full,capacity,base,end;
 int parent[3];unsigned needed[3],parents;
 unsigned child[8],edge[8],children;
 float *ring,*bias;const n25_tensor *weight;
 unsigned ci,kernel,pad,dilation,stride,accumulate;
 float slope;unsigned vector_activation,output_tanh;
 uint64_t computed;
} HFSNode;
static HFSNode hfs_nodes[HFS_MAX_NODES];
static unsigned hfs_count,hfs_output,hfs_initialized,hfs_callback_samples;
static uint64_t hfs_ring_bytes,hfs_output_rows,hfs_conv_output_rows;

static float *hfs_slot(HFSNode*n,unsigned at){return n->ring+(at%n->capacity)*n->channels;}
static const float *hfs_read(unsigned index,unsigned at){
 HFSNode*n=&hfs_nodes[index];
 if(at<n->base||at>=n->end){bench_metric("hfs_missing_node",index);bench_metric("hfs_missing_at",at);bench_metric("hfs_available_begin",n->base);bench_metric("hfs_available_end",n->end);fail("HIFI_STREAM_MISSING_DEPENDENCY");}
 return hfs_slot(n,at);
}
static void hfs_reclaim(unsigned index){
 HFSNode*n=&hfs_nodes[index];if(!n->children)return;
 unsigned keep=n->end;
 for(unsigned i=0;i<n->children;i++){
  unsigned q=hfs_nodes[n->child[i]].needed[n->edge[i]];if(q<keep)keep=q;
 }
 if(keep>n->base)n->base=keep;
}
static unsigned hfs_new(unsigned type,unsigned channels,unsigned full){
 if(hfs_count==HFS_MAX_NODES||!full)fail("HIFI_STREAM_GRAPH_CAPACITY");
 unsigned id=hfs_count++;HFSNode*n=&hfs_nodes[id];n->type=type;n->channels=channels;n->full=full;
 n->capacity=full<HFS_RING_ROWS?full:HFS_RING_ROWS;n->ring=alloc(n->capacity*channels);
 hfs_ring_bytes+=(uint64_t)n->capacity*channels*4;return id;
}
static void hfs_link(unsigned child,unsigned parent){
 HFSNode*c=&hfs_nodes[child],*p=&hfs_nodes[parent];
 if(c->parents==3||p->children==8)fail("HIFI_STREAM_EDGE_CAPACITY");
 unsigned edge=c->parents++;c->parent[edge]=(int)parent;p->child[p->children]=child;p->edge[p->children++]=edge;
}
static unsigned hfs_activation(unsigned parent,float slope){
 unsigned id=hfs_new(HFS_ACT,hfs_nodes[parent].channels,hfs_nodes[parent].full);hfs_nodes[id].slope=slope;hfs_link(id,parent);return id;
}
static unsigned hfs_convolution(const char*prefix,unsigned parent,unsigned dilation,unsigned stride,int transpose,float slope,int skip){
 const n25_tensor*w=field(prefix,"weight"),*b=field(prefix,"bias");
 unsigned ci=transpose?w->dims[0]:w->dims[1],co=transpose?w->dims[1]:w->dims[0],k=w->dims[2];
 if(ci!=hfs_nodes[parent].channels||w->dtype!=0||k%stride)fail("HIFI_STREAM_CONV_SHAPE");
 unsigned id=hfs_new(transpose?HFS_UP:HFS_CONV,co,hfs_nodes[parent].full*stride);HFSNode*n=&hfs_nodes[id];
 n->weight=w;n->bias=alloc(co);for(unsigned j=0;j<co;j++)n->bias[j]=value(b,j);
 n->ci=ci;n->kernel=k;n->dilation=dilation;n->stride=stride;n->pad=transpose?(k-stride)/2:((k-1)*dilation)/2;
 n->slope=slope;n->vector_activation=slope>0;n->accumulate=skip>=0;
 hfs_link(id,parent);if(skip>=0){if(hfs_nodes[skip].channels!=co||hfs_nodes[skip].full!=n->full)fail("HIFI_STREAM_SKIP_SHAPE");hfs_link(id,(unsigned)skip);}
 return id;
}
static float *hfs_weights(HFSNode*n){
 const n25_tensor*w=n->weight;unsigned index=(unsigned)(w-n25_tensors),offset=generator_cache_offsets[index];
 if(offset==0xffffffffu||offset>GENERATOR_CACHE_BUDGET||w->elements*4>GENERATOR_CACHE_BUDGET-offset)fail("HIFI_STREAM_REQUIRES_GENERATOR_WEIGHT_CACHE");
 float*weights=(float*)(generator_weight_cache+offset);
 if(generator_cache_valid[index]){generator_cache_hits++;return weights;}
 generator_cache_misses++;unsigned phases=n->type==HFS_UP?n->stride:1,inner=n->ci*(n->kernel/phases),co=n->channels;
 const float*scales=(const float*)(n25_blob+w->scale_offset);
 for(unsigned phase=0;phase<phases;phase++){
  const int8_t*codes=(const int8_t*)(n25_blob+w->offset)+phase*co*inner;float*pw=weights+phase*co*inner;
  generator_source_weight_read_bytes+=(uint64_t)co*inner+(uint64_t)co*4;cost_dequant+=(uint64_t)co*(inner/4);
  for(unsigned o=0;o<co;o++){
   unsigned i=0;for(;i+4<=inner;i+=4){int32x4_t q=vldrbq_s32(codes+o*inner+i);vst1q_f32(pw+o*inner+i,vmulq_n_f32(vcvtq_f32_s32(q),scales[o]));}
   for(;i<inner;i++)pw[o*inner+i]=(float)codes[o*inner+i]*scales[o];
  }
 }
 generator_cache_valid[index]=1;generator_cache_filled_bytes+=(uint64_t)w->elements*4;generator_cached_tensors++;return weights;
}
static unsigned hfs_left_needed(const HFSNode*n,unsigned at){
 if(n->type==HFS_CONV)return at>n->pad?at-n->pad:0;
 if(n->type==HFS_UP){int v=ceildiv((int)at+(int)n->pad-(int)n->kernel+1,(int)n->stride);return v>0?(unsigned)v:0;}
 return at;
}
static unsigned hfs_right_needed(const HFSNode*n,unsigned end){
 unsigned wanted=n->type==HFS_UP?(end-1+n->pad)/n->stride+1:n->type==HFS_CONV?end+n->pad:end;
 unsigned full=hfs_nodes[n->parent[0]].full;return wanted<full?wanted:full;
}
static void hfs_patch_row(HFSNode*n,unsigned t,unsigned phase,float*patch){
 unsigned phases=n->type==HFS_UP?n->stride:1,kp=n->kernel/phases,first=n->type==HFS_UP?(phase+n->pad)%n->stride:0;
 HFSNode*p=&hfs_nodes[n->parent[0]];
 for(unsigned j=0;j<kp;j++){
  int at=n->type==HFS_UP?((int)(t+n->pad)-(int)(first+j*n->stride))/(int)n->stride:(int)t+(int)(j*n->dilation)-(int)n->pad;
  float*dest=patch+j*n->ci;cost_pack_write+=(uint64_t)n->ci*4;
  if(at<0||(unsigned)at>=p->full){memset(dest,0,n->ci*4);continue;}
  const float*source=hfs_read((unsigned)n->parent[0],(unsigned)at);cost_pack_read+=(uint64_t)n->ci*4;
  if(n->vector_activation){
   cost_activation_vec+=2*(n->ci/4);cost_activation_scalar+=n->ci%4;unsigned i=0;
   for(;i+4<=n->ci;i+=4){float32x4_t v=vld1q_f32(source+i);vst1q_f32(dest+i,vmaxnmq_f32(v,vmulq_n_f32(v,n->slope)));}
   for(;i<n->ci;i++)dest[i]=source[i]<0?source[i]*n->slope:source[i];
  }else memcpy(dest,source,n->ci*4);
 }
}
static void hfs_compute_conv(HFSNode*n,unsigned begin,unsigned end){
 unsigned phases=n->type==HFS_UP?n->stride:1,inner=n->ci*(n->kernel/phases),co=n->channels;
 float*weights=hfs_weights(n);size_t mark=bench_arena_mark();float*patch=alloc(2*inner);
 for(unsigned phase=0;phase<phases;phase++){
  unsigned t=begin+(phase+phases-begin%phases)%phases;const float*pw=weights+phase*co*inner;
  for(;t<end;){
   unsigned count=t+phases<end?2:1;float*y0=hfs_slot(n,t),*y1=count==2?hfs_slot(n,t+phases):0;
   hfs_patch_row(n,t,phase,patch);if(count==2)hfs_patch_row(n,t+phases,phase,patch+inner);
   if(n->accumulate){memcpy(y0,hfs_read((unsigned)n->parent[1],t),co*4);if(count==2)memcpy(y1,hfs_read((unsigned)n->parent[1],t+phases),co*4);}
   if(count==2)gemm2(patch,patch+inner,pw,n->bias,y0,y1,inner,co,n->accumulate);
   else gemm4(patch,pw,n->bias,y0,inner,co,n->accumulate);
   t+=count*phases;
  }
 }
 if(n->output_tanh)for(unsigned t=begin;t<end;t++)hfs_slot(n,t)[0]=tanhf(hfs_slot(n,t)[0]);
 rewind_to(mark);hfs_conv_output_rows+=end-begin;
}
static void hfs_ensure(unsigned index,unsigned target){
 HFSNode*n=&hfs_nodes[index];if(target>n->full)fail("HIFI_STREAM_OUTPUT_RANGE");
 while(n->end<target){
  unsigned begin=n->end,end=target;if(end-begin>HFS_BATCH)end=begin+HFS_BATCH;
  /* MEL batches are deliberately small; only true dependency rows are obtained. */
  if(n->type==HFS_MEL&&end-begin>8)end=begin+8;
  if(n->parents){
   hfs_ensure((unsigned)n->parent[0],hfs_right_needed(n,end));
   for(unsigned edge=1;edge<n->parents;edge++)hfs_ensure((unsigned)n->parent[edge],end);
  }
  hfs_reclaim(index);
  if(end-n->base>n->capacity){bench_metric("hfs_overflow_node",index);bench_metric("hfs_queue_begin",n->base);bench_metric("hfs_queue_end",end);fail("HIFI_STREAM_FIFO_OVERFLOW");}
  if(n->type==HFS_MEL){
   size_t mark=bench_arena_mark();float*x=adapter_stream_window(begin,end);
   for(unsigned t=begin;t<end;t++){memcpy(hfs_slot(n,t),x+(t-begin)*80,80*4);}rewind_to(mark);
  }else if(n->type==HFS_CONV||n->type==HFS_UP){
   uint64_t stamp=bench_now();hfs_compute_conv(n,begin,end);generator_ticks+=bench_now()-stamp;
  }else if(n->type==HFS_ACT){
   cost_activation_scalar+=(uint64_t)(end-begin)*n->channels;
   for(unsigned t=begin;t<end;t++){const float*x=hfs_read((unsigned)n->parent[0],t);float*y=hfs_slot(n,t);for(unsigned j=0;j<n->channels;j++)y[j]=x[j]<0?x[j]*n->slope:x[j];}
  }else if(n->type==HFS_MEAN){
   for(unsigned t=begin;t<end;t++){
    const float*a=hfs_read((unsigned)n->parent[0],t),*b=hfs_read((unsigned)n->parent[1],t),*c=hfs_read((unsigned)n->parent[2],t);float*y=hfs_slot(n,t);
    for(unsigned j=0;j<n->channels;j++){float v=0;v+=a[j];v+=b[j];v+=c[j];y[j]=v/3;}
   }
  }else fail("HIFI_STREAM_UNKNOWN_NODE");
  n->end=end;n->computed+=end-begin;hfs_output_rows+=end-begin;
  for(unsigned edge=0;edge<n->parents;edge++){
   n->needed[edge]=edge==0?hfs_left_needed(n,end):end;
   HFSNode*p=&hfs_nodes[n->parent[edge]];if(n->needed[edge]>p->full)n->needed[edge]=p->full;
   hfs_reclaim((unsigned)n->parent[edge]);
  }
 }
}
static void hifi_stream_init(void){
 if(hfs_initialized++)fail("HIFI_STREAM_INIT_TWICE");
 unsigned x=hfs_new(HFS_MEL,80,mel_frames);x=hfs_convolution("generator.conv_pre",x,1,1,0,0,-1);
 unsigned rates[]={8,8,2,2},kernels[]={3,7,11},dilations[]={1,3,5};char name[160];
 for(unsigned stage=0;stage<4;stage++){
  x=hfs_activation(x,.1f);namefmt(name,sizeof(name),"generator.ups.%u",stage);x=hfs_convolution(name,x,1,rates[stage],1,0,-1);
  unsigned branches[3];
  for(unsigned branch=0;branch<3;branch++){
   unsigned y=x;
   for(unsigned pair=0;pair<3;pair++){
    unsigned skip=y;namefmt(name,sizeof(name),"generator.resblocks.%u.convs1.%u",stage*3+branch,pair);y=hfs_convolution(name,y,dilations[pair],1,0,.1f,-1);
    if(hfs_nodes[y].kernel!=kernels[branch])fail("HIFI_STREAM_RESIDUAL_KERNEL");
    namefmt(name,sizeof(name),"generator.resblocks.%u.convs2.%u",stage*3+branch,pair);y=hfs_convolution(name,y,1,1,0,.1f,(int)skip);
   }
   branches[branch]=y;
  }
  unsigned mean=hfs_new(HFS_MEAN,hfs_nodes[x].channels,hfs_nodes[x].full);for(unsigned branch=0;branch<3;branch++)hfs_link(mean,branches[branch]);x=mean;
 }
 x=hfs_activation(x,.01f);x=hfs_convolution("generator.conv_post",x,1,1,0,0,-1);hfs_nodes[x].output_tanh=1;hfs_output=x;
 bench_metric("hifi_stream_nodes",hfs_count);bench_metric("hifi_stream_ring_bytes",hfs_ring_bytes);bench_metric("hifi_stream_ring_rows",HFS_RING_ROWS);
}
static float *vocode_stateful(void){
 if(!hfs_initialized)fail("HIFI_STREAM_NOT_INITIALIZED");
 unsigned total=mel_frames*256;float*native=alloc(total);HFSNode*out=&hfs_nodes[hfs_output];
 if(out->channels!=1||out->full!=total)fail("HIFI_STREAM_FINAL_SHAPE");
 for(unsigned begin=0;begin<total;){
  unsigned end=begin+64;if(end>total)end=total;hfs_ensure(hfs_output,end);
  for(unsigned t=begin;t<end;t++){native[t]=hfs_read(hfs_output,t)[0];}out->base=end;
  /* Initial FIR lookahead is not yet stable before sixteen native samples. */
  if(end>=16||end==total){stream_ready(native,end,end==total);hfs_callback_samples=end;}
  begin=end;
 }
 for(unsigned i=0;i<hfs_count;i++)if(hfs_nodes[i].computed!=hfs_nodes[i].end||hfs_nodes[i].end!=hfs_nodes[i].full)fail("HIFI_STREAM_INCOMPLETE_LAYER");
 bench_metric("hifi_stream_unique_output_rows",hfs_output_rows);bench_metric("hifi_stream_unique_conv_output_rows",hfs_conv_output_rows);bench_metric("hifi_stream_final_native_samples",hfs_callback_samples);bench_metric("hifi_stream_recomputed_rows",0);
 return native;
}
