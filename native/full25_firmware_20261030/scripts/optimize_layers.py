"""Resolve immutable per-layer tensors at boot; retain every model operation."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'native/src'


def main():
    snapshot=ROOT/'snapshots/before_layer_views'
    snapshot.mkdir(parents=True,exist_ok=False)
    names=['quality.c','adapter_stream.inc','persistent_runtime.inc']
    before={}
    for name in names:
        p=SRC/name;shutil.copy2(p,snapshot/name)
        before[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=SRC/'quality.c';s=p.read_text()
    old='static void linear(const char *prefix,const float*x,float*y,unsigned n){\n const n25_tensor*w=field(prefix,"weight"),*b=field(prefix,"bias");'
    new='''typedef struct {const n25_tensor *weight,*bias;} LayerOp;
typedef struct {LayerOp dw,norm,up,down;const float *gamma,*dw_taps,*dw_bias;int dw_safe;} LayerBlock;
static LayerBlock decoder_layers[5],adapter_layers[6];
static LayerOp adapter_input_op,adapter_norm_op,adapter_head_op,acoustic_head_op;
static const float *adapter_language_view,*acoustic_language_view;
static float adapter_dw_taps[6][7*192];
static void linear_resolved(const n25_tensor*w,const n25_tensor*b,const float*x,float*y,unsigned n){'''
    assert s.count(old)==1;s=s.replace(old,new)
    point='/* Time-major convolution; transpose weights are [in,out,k], normal [out,in,k]. */'
    s=s.replace(point,'static void linear(const char*p,const float*x,float*y,unsigned n){linear_resolved(field(p,"weight"),field(p,"bias"),x,y,n);}\n'+point)
    old='static void norm(const char*p,float*x,unsigned n,unsigned c,float eps){\n ((void)0);\n const n25_tensor*w=field(p,"weight"),*b=field(p,"bias");'
    new='static void norm_resolved(const n25_tensor*w,const n25_tensor*b,float*x,unsigned n,unsigned c,float eps){\n ((void)0);'
    assert s.count(old)==1;s=s.replace(old,new)
    point='static void activation(float*x,unsigned n,int type){'
    s=s.replace(point,'static void norm(const char*p,float*x,unsigned n,unsigned c,float eps){norm_resolved(field(p,"weight"),field(p,"bias"),x,n,c,eps);}\n'+point)
    old='namefmt(q,sizeof(q),"%s.dw",p);const n25_tensor*w=field(q,"weight"),*b=field(q,"bias");'
    assert s.count(old)==1;s=s.replace(old,'const LayerBlock *layer=decoder_layers+stage;const n25_tensor*w=layer->dw.weight,*b=layer->dw.bias;')
    old='namefmt(q,sizeof(q),"%s.norm",p);norm(q,z,n,c,1e-5f);namefmt(q,sizeof(q),"%s.up",p);linear(q,z,up,n);activation(up,n*c*2,0);namefmt(q,sizeof(q),"%s.down",p);linear(q,up,down,n);'
    new='norm_resolved(layer->norm.weight,layer->norm.bias,z,n,c,1e-5f);linear_resolved(layer->up.weight,layer->up.bias,z,up,n);activation(up,n*c*2,0);linear_resolved(layer->down.weight,layer->down.bias,up,down,n);'
    assert s.count(old)==1;s=s.replace(old,new)
    old='for(unsigned i=0;i<5;i++){namefmt(name,sizeof(name),"acoustic.decoder.%u",i);decode_local(i,name,seq,n,ds[i],decoder_history[i]);}'
    assert old in s;s=s.replace(old,'for(unsigned i=0;i<5;i++)decode_local(i,0,seq,n,ds[i],decoder_history[i]);')
    old='float*out=condition_cache+decoded_frames*ACOUSTIC_DIM;namefmt(name,sizeof(name),"acoustic.heads.%d",ACOUSTIC_DIM);linear(name,seq,out,n);const n25_tensor*la=find("acoustic.language_acoustic.weight");const float*la_v=tensor_view(la);if(!la_v)fail("ACOUSTIC_LANGUAGE_VIEW_MISSING");'
    new='float*out=condition_cache+decoded_frames*ACOUSTIC_DIM;linear_resolved(acoustic_head_op.weight,acoustic_head_op.bias,seq,out,n);const float*la_v=acoustic_language_view;'
    assert old in s;s=s.replace(old,new)
    s=s.replace('#include "persistent_runtime.inc"','#include "layer_views.inc"\n#include "persistent_runtime.inc"')
    p.write_text(s)
    p=SRC/'adapter_stream.inc';s=p.read_text()
    s=s.replace('const n25_tensor*head=field("physical_to_mel.head","weight");','const n25_tensor*head=adapter_head_op.weight;')
    old='char name[160];namefmt(name,sizeof(name),"physical_to_mel.inputs.%d",ACOUSTIC_DIM);\n const n25_tensor*w=field(name,"weight"),*b=field(name,"bias");'
    assert old in s;s=s.replace(old,'const n25_tensor*w=adapter_input_op.weight,*b=adapter_input_op.bias;')
    old='const n25_tensor*language= find("physical_to_mel.language.weight");\n const float*language_v=tensor_view(language);if(!language_v)fail("ADAPTER_LANGUAGE_VIEW_MISSING");'
    assert old in s;s=s.replace(old,'const float*language_v=adapter_language_view;')
    s=s.replace('norm("physical_to_mel.norm",out,rows,co,1e-6f);','norm_resolved(adapter_norm_op.weight,adapter_norm_op.bias,out,rows,co,1e-6f);')
    old='''char prefix[160],name[160];namefmt(prefix,sizeof(prefix),"physical_to_mel.blocks.%u",layer-1);
   namefmt(name,sizeof(name),"%s.dw",prefix);const n25_tensor*w=field(name,"weight"),*b=field(name,"bias");'''
    new='const LayerBlock*resolved=adapter_layers+layer-1;const n25_tensor*w=resolved->dw.weight,*b=resolved->dw.bias;'
    assert old in s;s=s.replace(old,new)
    old='''float*weights=alloc(c*7),*bias=alloc(c);
   for(unsigned k=0;k<7;k++)for(unsigned j=0;j<c;j++)weights[k*c+j]=value(w,j*7+k);
   for(unsigned j=0;j<c;j++)bias[j]=value(b,j);
   int safe=depthwise_range(weights,c*7,0x26800000u,0x58800000u);safe&=depthwise_range(bias,c,0x0d800000u,0x71800000u);'''
    new='const float*weights=resolved->dw_taps,*bias=resolved->dw_bias;\n   int safe=resolved->dw_safe;'
    assert old in s;s=s.replace(old,new)
    s=s.replace('namefmt(name,sizeof(name),"%s.norm",prefix);norm(name,z,rows,c,1e-6f);',
        'norm_resolved(resolved->norm.weight,resolved->norm.bias,z,rows,c,1e-6f);')
    s=s.replace('namefmt(name,sizeof(name),"%s.up",prefix);linear(name,z,up,rows);activation(up,rows*c*4,1);',
        'linear_resolved(resolved->up.weight,resolved->up.bias,z,up,rows);activation(up,rows*c*4,1);')
    s=s.replace('namefmt(name,sizeof(name),"%s.down",prefix);linear(name,up,down,rows);',
        'linear_resolved(resolved->down.weight,resolved->down.bias,up,down,rows);')
    s=s.replace('const n25_tensor*gamma=field(prefix,"gamma");const float*gamma_v=tensor_view(gamma);if(!gamma_v)fail("ADAPTER_GAMMA_VIEW_MISSING");',
        'const float*gamma_v=resolved->gamma;')
    s=s.replace('linear("physical_to_mel.head",x,adapter_stream_head+begin*ADAPTER_STREAM_OUTPUT,rows);',
        'linear_resolved(adapter_head_op.weight,adapter_head_op.bias,x,adapter_stream_head+begin*ADAPTER_STREAM_OUTPUT,rows);')
    p.write_text(s)
    p=SRC/'persistent_runtime.inc';s=p.read_text()
    s=s.replace('uint64_t views_start=bench_now();tensor_views_prepare();','uint64_t views_start=bench_now();tensor_views_prepare();layer_views_prepare();')
    p.write_text(s)
    after={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest()for n in names}
    (ROOT/'LAYER_VIEWS_PATCH.json').write_text(json.dumps(dict(before=before,after=after,
        policy='Per-layer tensor pointers, transposed immutable depthwise tables, resolved at boot. No global field-name cache; model arithmetic and input values unchanged.'),indent=2)+'\n')


if __name__=='__main__':main()
