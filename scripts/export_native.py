"""Losslessly export ALL current INT8/FP16 tensors to a freestanding C interface.

No quantization or training. Generated files go under --output, not the source
release. Only numpy is required; reference PyTorch is needed for kernel tests.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'int8'))
from codec import read_tensors,unpack


def export(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    package=ROOT/'int8'
    spec=json.loads((package/'quantization.json').read_text())
    restored=unpack(package)  # also checks source SHA and shapes
    values=read_tensors(package/'weights.safetensors')
    blob=bytearray();offsets={};descriptors=[]
    for name,array in sorted(values.items()):
        blob.extend(b'\0'*((-len(blob))%4));offsets[name]=len(blob)
        blob.extend(array.tobytes())
    (output/'model.bin').write_bytes(blob)
    dtype={np.dtype('int8'):0,np.dtype('<f2'):1,np.dtype('<f4'):2}
    for name,entry in spec['tensors'].items():
        v=values[name];q=entry.get('quantization')
        desc=dict(name=name,offset=offsets[name],scale_offset=offsets[q['scale']]if q else 0,
                  elements=v.size,dims=list(v.shape)+[1]*(3-v.ndim),rank=v.ndim,
                  dtype=dtype[v.dtype],axis=q['axis']if q else 0)
        assert v.ndim<=3
        # Validate all native descriptor offsets and scale indexing before export.
        stored=np.frombuffer(blob,dtype=v.dtype,count=v.size,offset=desc['offset']).reshape(v.shape)
        if q:
            s=np.frombuffer(blob,dtype='<f4',count=v.shape[q['axis']],offset=desc['scale_offset'])
            shape=[1]*v.ndim;shape[q['axis']]=len(s)
            actual=stored.astype('f4')*s.reshape(shape)
        else:actual=stored.astype('f4')
        assert np.array_equal(actual,restored[name]),name
        descriptors.append(desc)
    header=['#ifndef NEW25_MODEL_INDEX_H','#define NEW25_MODEL_INDEX_H','#include "tensor.h"',
            'extern const unsigned char n25_blob[];',f'#define N25_TENSOR_COUNT {len(descriptors)}',
            'extern const n25_tensor n25_tensors[N25_TENSOR_COUNT];',
            'extern const char *const n25_names[N25_TENSOR_COUNT];','#endif']
    (output/'model_index.h').write_text('\n'.join(header)+'\n')
    lines=['#include "model_index.h"','const n25_tensor n25_tensors[N25_TENSOR_COUNT] = {']
    for d in descriptors:
        dims=','.join(map(str,d['dims']))
        lines.append('{%d,%d,%d,{%s},%d,%d,%d,0},' %(d['offset'],d['scale_offset'],d['elements'],dims,d['rank'],d['dtype'],d['axis']))
    lines+=['};','const char *const n25_names[N25_TENSOR_COUNT] = {']
    lines +=[json.dumps(d['name'])+',' for d in descriptors]
    lines+=['};']
    (output/'model_index.c').write_text('\n'.join(lines)+'\n')
    (output/'weights.S').write_text('.section .model_weights,"a"\n.balign 16\n.global n25_blob\nn25_blob:\n.incbin '+json.dumps(str(output/'model.bin'))+'\n.section .note.GNU-stack,""\n')
    receipt=dict(format='new25_native_storage_v1',source_weights_sha256=spec['weights_sha256'],
                 model_bin_bytes=len(blob),model_bin_sha256=hashlib.sha256(blob).hexdigest(),
                 tensors=len(descriptors),all_restored_tensors_exact=True,descriptors=descriptors)
    (output/'MODEL.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=export(a.output)
    print(json.dumps({k:v for k,v in r.items()if k!='descriptors'},indent=2))
