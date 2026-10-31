/* Immutable model-scoped float views. These allocations precede request
 * scratch marks. No activation-pointer or request-data cache is involved. */
#define TENSOR_NAME_BUCKETS 512
static uint16_t tensor_name_index[TENSOR_NAME_BUCKETS];
static const float*tensor_float_views[N25_TENSOR_COUNT];
static unsigned tensor_views_initialized,tensor_view_decoded_bytes,tensor_view_borrowed_bytes;
static uint32_t tensor_name_hash(const char*s){
 uint32_t hash=2166136261u;
 for(;*s;s++)hash=(hash^(unsigned char)*s)*16777619u;
 return hash;
}
static const n25_tensor*find(const char*name){
 if(tensor_views_initialized){
  unsigned at=tensor_name_hash(name)&(TENSOR_NAME_BUCKETS-1);
  while(tensor_name_index[at]){
   unsigned index=tensor_name_index[at]-1;
   if(!strcmp(name,n25_names[index]))return n25_tensors+index;
   at=(at+1)&(TENSOR_NAME_BUCKETS-1);
  }
 }else for(unsigned i=0;i<N25_TENSOR_COUNT;i++)if(!strcmp(name,n25_names[i]))return n25_tensors+i;
 fail(name);return 0;
}
static const float*tensor_view(const n25_tensor*t){return tensor_float_views[t-n25_tensors];}
static float value(const n25_tensor*t,unsigned index){
 const float*p=tensor_view(t);
 return p?p[index]:n25_value(n25_blob,t,index);
}
__attribute__((noinline)) void tensor_views_prepare(void){
 if(tensor_views_initialized)fail("TENSOR_VIEW_INIT_TWICE");
 for(unsigned i=0;i<N25_TENSOR_COUNT;i++){
  unsigned at=tensor_name_hash(n25_names[i])&(TENSOR_NAME_BUCKETS-1);
  while(tensor_name_index[at])at=(at+1)&(TENSOR_NAME_BUCKETS-1);
  tensor_name_index[at]=(uint16_t)(i+1);
  const n25_tensor*t=n25_tensors+i;
  int front=!strncmp(n25_names[i],"acoustic.",9)||!strncmp(n25_names[i],"physical_to_mel.",16);
  int integer=t->dtype==0&&t->axis==0&&t->rank>=2&&t->dims[0]>=16&&t->elements/t->dims[0]>=16;
  int embedding=strstr(n25_names[i],"language")!=0||!strcmp(n25_names[i],"acoustic.phone.weight");
  if(!front||(!embedding&&integer))continue;
  if(t->dtype==2){
   if(t->offset%4)fail("TENSOR_VIEW_F32_ALIGNMENT");
   tensor_float_views[i]=(const float*)(n25_blob+t->offset);
   tensor_view_borrowed_bytes+=4*t->elements;
  }else{
   float*out=alloc(t->elements);n25_unpack(n25_blob,t,out);
   tensor_float_views[i]=out;tensor_view_decoded_bytes+=4*t->elements;
  }
 }
 tensor_views_initialized=1;
 bench_metric("tensor_float_view_decoded_persistent_bytes",tensor_view_decoded_bytes);
 bench_metric("tensor_float_view_borrowed_model_bytes",tensor_view_borrowed_bytes);
 bench_metric("tensor_float_view_ready",1);
}
