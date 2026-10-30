"""Check all native tensor decodes and matrix/conv kernels against real weights.

Also generates three real-layer numerical fixtures for both simulators. The
fixtures are operator tests, not ordinary-text TTS or quality measurements.
"""
from pathlib import Path
import argparse
import ctypes as C
import json
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'int8'))
from codec import unpack
from export_native import export


class Tensor(C.Structure):
    _fields_=[('offset',C.c_uint32),('scale_offset',C.c_uint32),('elements',C.c_uint32),
              ('dims',C.c_uint32*3),('rank',C.c_uint8),('dtype',C.c_uint8),('axis',C.c_uint8),('reserved',C.c_uint8)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cc',default='cc')
    args=p.parse_args();out=args.output.resolve();receipt=export(out)
    subprocess.run([args.cc,'-O2','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',
                    str(ROOT/'native/tensor.c'),'-o',str(out/'libnew25_tensor.so')],check=True)
    lib=C.CDLL(str(out/'libnew25_tensor.so'));ptr=C.POINTER(C.c_float)
    lib.n25_unpack.argtypes=[C.c_void_p,C.POINTER(Tensor),ptr]
    lib.n25_linear.argtypes=[C.c_void_p,C.POINTER(Tensor),C.POINTER(Tensor),ptr,ptr,C.c_uint]
    lib.n25_conv1d.argtypes=[C.c_void_p,C.POINTER(Tensor),C.POINTER(Tensor),ptr,ptr,C.c_uint,C.c_uint,C.c_uint,C.c_uint]
    lib.n25_conv_transpose1d.argtypes=[C.c_void_p,C.POINTER(Tensor),C.POINTER(Tensor),ptr,ptr,C.c_uint,C.c_uint,C.c_uint]
    blob=C.create_string_buffer((out/'model.bin').read_bytes())
    arrays=unpack(ROOT/'int8');descs={}
    for d in receipt['descriptors']:
        t=Tensor(d['offset'],d['scale_offset'],d['elements'],(C.c_uint32*3)(*d['dims']),d['rank'],d['dtype'],d['axis'],0)
        v=np.empty(d['elements'],dtype='f4');lib.n25_unpack(blob,C.byref(t),v.ctypes.data_as(ptr))
        assert np.array_equal(v,arrays[d['name']].reshape(-1)),d['name']
        descs[d['name']]=t
    selected=['acoustic.phonetic.36.weight','physical_to_mel.inputs.31.weight','generator.ups.0.weight']
    quant=json.loads((ROOT/'int8/quantization.json').read_text())
    rows=[];fixtures=[]
    torch.set_num_threads(2)
    for name,info in quant['tensors'].items():
        if 'quantization' not in info:continue
        w=arrays[name];weight=torch.from_numpy(w)
        bias_name=name.removesuffix('weight')+'bias'
        b=arrays.get(bias_name);bias=torch.from_numpy(b)if b is not None else None
        wt=C.byref(descs[name]);bt=C.byref(descs[bias_name])if b is not None else None
        # Bounded deterministic inputs; no corpus features or audio.
        if w.ndim==2:
            x=((np.arange(2*w.shape[1])%17-8)/16).astype('f4').reshape(2,w.shape[1])
            gold=F.linear(torch.from_numpy(x),weight,bias).numpy()
            y=np.empty_like(gold);lib.n25_linear(blob,wt,bt,x.ctypes.data_as(ptr),y.ctypes.data_as(ptr),2)
            mode='linear';params=[2,0,0,0]
        elif name.startswith('generator.ups.'):
            x=((np.arange(w.shape[0]*3)%17-8)/16).astype('f4').reshape(w.shape[0],3)
            rate=[8,8,2,2][int(name.split('.')[2])];pad=(w.shape[2]-rate)//2
            gold=F.conv_transpose1d(torch.from_numpy(x)[None],weight,bias,stride=rate,padding=pad)[0].numpy()
            y=np.empty_like(gold);lib.n25_conv_transpose1d(blob,wt,bt,x.ctypes.data_as(ptr),y.ctypes.data_as(ptr),3,rate,pad)
            mode='transpose';params=[3,rate,pad,0]
        else:
            co,ci,k=w.shape;groups=co if ci==1 and '.dw.'in name else 1
            x=((np.arange(ci*groups*7)%17-8)/16).astype('f4').reshape(ci*groups,7)
            pad=(k-1)//2
            gold=F.conv1d(torch.from_numpy(x)[None],weight,bias,padding=pad,groups=groups)[0].numpy()
            y=np.empty_like(gold);lib.n25_conv1d(blob,wt,bt,x.ctypes.data_as(ptr),y.ctypes.data_as(ptr),7,1,pad,groups)
            mode='conv';params=[7,1,pad,groups]
        err=float(np.max(np.abs(y-gold)))
        assert np.allclose(y,gold,rtol=2e-4,atol=2e-4),(name,err)
        row=dict(name=name,mode=mode,max_absolute_error=err);rows.append(row)
        if name in selected:
            fixtures.append(dict(name=name,bias=bias_name if b is not None else None,mode=mode,params=params,
                                 x=x.reshape(-1),gold=gold.reshape(-1)))
    lines=['#include "model_index.h"','#include "platform.h"',
           'static float output[8192];']
    for i,f in enumerate(fixtures):
        for n in ['x','gold']:
            values=','.join(float(v).hex()+'f'for v in f[n])
            lines.append(f'static const float {n}{i}[] = '+'{'+values+'};')
    lines.append('int main(void) {')
    names=list(quant['tensors'])
    for i,f in enumerate(fixtures):
        weight='&n25_tensors[%d]'%names.index(f['name'])
        bias='&n25_tensors[%d]'%names.index(f['bias'])if f['bias']else '0'
        params=','.join(map(str,f['params'][:{'linear':1,'conv':4,'transpose':3}[f['mode']]]))
        fn={'linear':'n25_linear','conv':'n25_conv1d','transpose':'n25_conv_transpose1d'}[f['mode']]
        lines.append(f'{fn}(n25_blob,{weight},{bias},x{i},output,{params});')
        lines.append(f'for(unsigned j=0;j<{len(f["gold"])};j++) '+'{'+f'float d=output[j]-gold{i}[j];if(!(d>=-0.0003f&&d<=0.0003f))n25_exit(1);'+'}')
    lines+=['n25_puts("NEW25_REAL_INT8_KERNELS_PASS\\n");','n25_exit(0);return 0;}']
    (out/'kernel_fixture.c').write_text('\n'.join(lines)+'\n')
    result=dict(passed=True,all_316_tensors_decode_exact=True,real_weight_kernel_cases=len(rows),
                maximum_absolute_error=max(r['max_absolute_error']for r in rows),
                simulator_fixtures=[f['name']for f in fixtures],rows=rows)
    (out/'KERNEL_TEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))


if __name__=='__main__':main()
