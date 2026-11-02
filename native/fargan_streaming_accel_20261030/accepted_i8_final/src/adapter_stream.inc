/* Persistent physical -> FARGAN20 adapter. Include after the E2 helpers.
 * Same six residual blocks, centered k7 convolution and per-frame norms.
 * Only finalized physical frames enter a layer ring. Padding is at real EOS.
 * No HiFi mel-rate conversion: output remains on the physical 100Hz clock.
 */
#define ADAPTER_STREAM_LAYERS 7
#define ADAPTER_STREAM_CHANNELS 192
#define ADAPTER_STREAM_RING 32
#define ADAPTER_STREAM_BATCH 16
#define ADAPTER_STREAM_OUTPUT 20
static float *adapter_stream_ring[ADAPTER_STREAM_LAYERS];
static unsigned adapter_stream_done[ADAPTER_STREAM_LAYERS];
static float *adapter_stream_head;
static unsigned adapter_stream_head_done,adapter_stream_initialized;
static uint64_t adapter_stream_requests,adapter_stream_reused_frames;

static void adapter_stream_init(void){
 if(!frames||frames>MAX_FRAMES)fail("ADAPTER_STREAM_BAD_FRAME_COUNT");
 const n25_tensor*head=field("physical_to_mel.head","weight");
 if(head->rank!=2||head->dims[0]!=ADAPTER_STREAM_OUTPUT||head->dims[1]!=ADAPTER_STREAM_CHANNELS)fail("ADAPTER_FARGAN_HEAD_SHAPE");
 for(unsigned i=0;i<ADAPTER_STREAM_LAYERS;i++){
  adapter_stream_ring[i]=alloc(ADAPTER_STREAM_RING*ADAPTER_STREAM_CHANNELS);
  adapter_stream_done[i]=0;
 }
 adapter_stream_head=alloc(frames*ADAPTER_STREAM_OUTPUT);adapter_stream_head_done=0;
 adapter_stream_requests=adapter_stream_reused_frames=0;
 adapter_stream_initialized=1;
}

static const float *adapter_stream_frame(unsigned layer,unsigned t){
 unsigned done=adapter_stream_done[layer];
 if(t>=done||done-t>ADAPTER_STREAM_RING)fail("ADAPTER_STREAM_HISTORY_LOST");
 return adapter_stream_ring[layer]+(t%ADAPTER_STREAM_RING)*ADAPTER_STREAM_CHANNELS;
}

/* Existing packed convolution order is [out,k,input]. Build only requested
 * output patches; gemm2/gemm4 retain the original MVE reduction arithmetic. */
static void adapter_stream_input(unsigned begin,unsigned rows,float *out){
 char name[160];namefmt(name,sizeof(name),"physical_to_mel.inputs.%d",ACOUSTIC_DIM);
 const n25_tensor*w=field(name,"weight"),*b=field(name,"bias");
 unsigned inner=ACOUSTIC_DIM*7,co=ADAPTER_STREAM_CHANNELS;
 if(w->dtype>2||w->rank!=3||w->dims[0]!=co||w->dims[1]!=ACOUSTIC_DIM||w->dims[2]!=7||w->elements!=co*inner)fail("ADAPTER_STREAM_INPUT_SHAPE");
 int integer=fast_front_eligible(w);
 size_t mark=bench_arena_mark();float *weights=integer?0:alloc(w->elements),*bias=alloc(co),*patch=alloc(rows*inner);
 if(!integer)n25_unpack(n25_blob,w,weights);
 for(unsigned j=0;j<co;j++)bias[j]=value(b,j);
 for(unsigned t=0;t<rows;t++)for(unsigned k=0;k<7;k++){
  int at=(int)(begin+t+k)-3;float*dest=patch+t*inner+k*ACOUSTIC_DIM;
  cost_pack_write+=ACOUSTIC_DIM*4;
  if(at>=0&&(unsigned)at<frames){
   if((unsigned)at>=decoded_frames)fail("ADAPTER_STREAM_CONDITION_NOT_READY");
   memcpy(dest,condition_cache+(unsigned)at*ACOUSTIC_DIM,ACOUSTIC_DIM*4);cost_pack_read+=ACOUSTIC_DIM*4;
  }else memset(dest,0,ACOUSTIC_DIM*4);
 }
 if(integer){for(unsigned t=0;t<rows;t++)fast_front_matrix(patch+t*inner,w,bias,out+t*co,inner,co,0);}
 else{
  unsigned t=0;for(;t+1<rows;t+=2)gemm2(patch+t*inner,patch+(t+1)*inner,weights,bias,out+t*co,out+(t+1)*co,inner,co,0);
  if(t<rows)gemm4(patch+t*inner,weights,bias,out+t*co,inner,co,0);
 }
 rewind_to(mark);
 const n25_tensor*language= find("physical_to_mel.language.weight");
 for(unsigned j=0;j<rows*co;j++)out[j]+=value(language,LANGUAGE_ID*co+j%co);
 norm("physical_to_mel.norm",out,rows,co,1e-6f);
}

static void adapter_stream_need_layer(unsigned layer,unsigned end){
 if(end>frames)end=frames;
 while(adapter_stream_done[layer]<end){
  unsigned begin=adapter_stream_done[layer],rows=end-begin;
  if(rows>ADAPTER_STREAM_BATCH)rows=ADAPTER_STREAM_BATCH;
  unsigned upstream=begin+rows+3;if(upstream>frames)upstream=frames;
  if(layer)adapter_stream_need_layer(layer-1,upstream);else ensure_condition(upstream);
  size_t mark=bench_arena_mark();unsigned c=ADAPTER_STREAM_CHANNELS;
  float*z=alloc(rows*c);
  if(!layer)adapter_stream_input(begin,rows,z);
  else{
   char prefix[160],name[160];namefmt(prefix,sizeof(prefix),"physical_to_mel.blocks.%u",layer-1);
   namefmt(name,sizeof(name),"%s.dw",prefix);const n25_tensor*w=field(name,"weight"),*b=field(name,"bias");
   if(w->dims[0]!=c||w->dims[1]!=1||w->dims[2]!=7)fail("ADAPTER_STREAM_DW_SHAPE");
   float*weights=alloc(c*7);for(unsigned j=0;j<c*7;j++)weights[j]=value(w,j);
   for(unsigned t=0;t<rows;t++)for(unsigned j=0;j<c;j++){
    float sum=value(b,j);
    for(unsigned k=0;k<7;k++){
     int at=(int)(begin+t+k)-3;
     if(at>=0&&(unsigned)at<frames)sum+=dot(adapter_stream_frame(layer-1,(unsigned)at)+j,weights+j*7+k,1);
    }
    z[t*c+j]=sum;
   }
   namefmt(name,sizeof(name),"%s.norm",prefix);norm(name,z,rows,c,1e-6f);
   float*up=alloc(rows*c*4),*down=alloc(rows*c);
   namefmt(name,sizeof(name),"%s.up",prefix);linear(name,z,up,rows);activation(up,rows*c*4,1);
   namefmt(name,sizeof(name),"%s.down",prefix);linear(name,up,down,rows);
   const n25_tensor*gamma=field(prefix,"gamma");
   for(unsigned t=0;t<rows;t++){
    const float*residual=adapter_stream_frame(layer-1,begin+t);
    for(unsigned j=0;j<c;j++)z[t*c+j]=residual[j]+down[t*c+j]*value(gamma,j);
   }
  }
  for(unsigned t=0;t<rows;t++)memcpy(adapter_stream_ring[layer]+((begin+t)%ADAPTER_STREAM_RING)*c,z+t*c,c*4);
  adapter_stream_done[layer]=begin+rows;rewind_to(mark);
 }
}

static void adapter_stream_need_head(unsigned end){
 if(end>frames)end=frames;
 while(adapter_stream_head_done<end){
  unsigned begin=adapter_stream_head_done,rows=end-begin;
  if(rows>ADAPTER_STREAM_BATCH)rows=ADAPTER_STREAM_BATCH;
  adapter_stream_need_layer(ADAPTER_STREAM_LAYERS-1,begin+rows);
  size_t mark=bench_arena_mark();float*x=alloc(rows*ADAPTER_STREAM_CHANNELS);
  for(unsigned t=0;t<rows;t++)memcpy(x+t*ADAPTER_STREAM_CHANNELS,adapter_stream_frame(ADAPTER_STREAM_LAYERS-1,begin+t),ADAPTER_STREAM_CHANNELS*4);
  linear("physical_to_mel.head",x,adapter_stream_head+begin*ADAPTER_STREAM_OUTPUT,rows);
  adapter_stream_head_done=begin+rows;rewind_to(mark);
 }
}

static const float *adapter_fargan_frame(unsigned index){
 if(!adapter_stream_initialized||index>=frames)fail("ADAPTER_FARGAN_BAD_FRAME");
 adapter_stream_requests++;if(index<adapter_stream_head_done)adapter_stream_reused_frames++;
 /* Keep acoustic time in its original counter rather than charging it to E2. */
 uint64_t acoustic_before=acoustic_ticks,start=bench_now();
 adapter_stream_need_head(index+1);
 adapter_ticks+=bench_now()-start-(acoustic_ticks-acoustic_before);
 return adapter_stream_head+index*ADAPTER_STREAM_OUTPUT;
}

static void adapter_stream_report(void){
 bench_metric("adapter_fargan_frame_requests",adapter_stream_requests);
 bench_metric("adapter_fargan_reused_frames",adapter_stream_reused_frames);
 bench_metric("adapter_stream_head_frames",adapter_stream_head_done);
 bench_metric("adapter_stream_ring_bytes",ADAPTER_STREAM_LAYERS*ADAPTER_STREAM_RING*ADAPTER_STREAM_CHANNELS*4);
 bench_metric("adapter_stream_head_bytes",frames*ADAPTER_STREAM_OUTPUT*4);
 bench_metric("adapter_stream_output_dimensions",ADAPTER_STREAM_OUTPUT);
 for(unsigned i=0;i<ADAPTER_STREAM_LAYERS;i++){
  char name[80];namefmt(name,sizeof(name),"adapter_stream_layer_%u_frames",i);bench_metric(name,adapter_stream_done[i]);
 }
}
