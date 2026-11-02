/* Immutable per-layer tensor references. Never caches activation pointers. */
static LayerOp layer_op(const char *name){LayerOp op={field(name,"weight"),field(name,"bias")};return op;}
static LayerBlock layer_block(const char *name){
 LayerBlock block={0};char child[160];
 namefmt(child,sizeof(child),"%s.dw",name);block.dw=layer_op(child);
 namefmt(child,sizeof(child),"%s.norm",name);block.norm=layer_op(child);
 namefmt(child,sizeof(child),"%s.up",name);block.up=layer_op(child);
 namefmt(child,sizeof(child),"%s.down",name);block.down=layer_op(child);
 return block;
}
static void layer_views_prepare(void){
 char name[160];
 for(unsigned i=0;i<5;i++){
  namefmt(name,sizeof(name),"acoustic.decoder.%u",i);decoder_layers[i]=layer_block(name);
  const float*w=tensor_view(decoder_layers[i].dw.weight);
  if(!w)fail("DECODER_IMMUTABLE_VIEW_MISSING");
  for(unsigned j=0;j<192;j++)for(unsigned k=0;k<5;k++)decoder_dw_taps[i][k*192+j]=w[j*5+k];
  decoder_dw_taps_ready[i]=1;
 }
 for(unsigned i=0;i<6;i++){
  namefmt(name,sizeof(name),"physical_to_mel.blocks.%u",i);adapter_layers[i]=layer_block(name);
  LayerBlock *block=adapter_layers+i;
  block->gamma=tensor_view(field(name,"gamma"));block->dw_bias=tensor_view(block->dw.bias);
  const float*w=tensor_view(block->dw.weight);block->dw_taps=adapter_dw_taps[i];
  if(!w||!block->gamma||!block->dw_bias)fail("ADAPTER_IMMUTABLE_VIEW_MISSING");
  for(unsigned j=0;j<192;j++)for(unsigned k=0;k<7;k++)adapter_dw_taps[i][k*192+j]=w[j*7+k];
  block->dw_safe=depthwise_range(block->dw_taps,7*192,0x26800000u,0x58800000u);
  block->dw_safe&=depthwise_range(block->dw_bias,192,0x0d800000u,0x71800000u);
 }
 for(unsigned i=0;i<2;i++){unsigned dim=i?39:31;namefmt(name,sizeof(name),"physical_to_mel.inputs.%u",dim);adapter_input_ops[i]=layer_op(name);namefmt(name,sizeof(name),"acoustic.heads.%u",dim);acoustic_head_ops[i]=layer_op(name);}
 adapter_input_op=adapter_input_ops[ACOUSTIC_DIM==39];
 adapter_norm_op=layer_op("physical_to_mel.norm");adapter_head_op=layer_op("physical_to_mel.head");
 acoustic_head_op=acoustic_head_ops[ACOUSTIC_DIM==39];
 adapter_language_view=tensor_view(find("physical_to_mel.language.weight"));
 acoustic_language_view=tensor_view(find("acoustic.language_acoustic.weight"));
 if(!adapter_language_view||!acoustic_language_view)fail("LANGUAGE_IMMUTABLE_VIEW_MISSING");
}
