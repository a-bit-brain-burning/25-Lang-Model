"""Dense safetensors plus symmetric per-output-channel INT8 weight storage.

This is a host inference storage codec. It reconstructs FP32 tensors and does not
claim a native INT8 kernel, MCU firmware, or bounded MCU memory implementation.
"""
from pathlib import Path
import hashlib
import json
import math
import struct

import numpy as np

DTYPES = {'F16':np.dtype('<f2'), 'F32':np.dtype('<f4'), 'I8':np.dtype('i1')}


def read_tensors(path):
    raw=Path(path).read_bytes()
    if len(raw)<10:
        raise ValueError('Truncated tensor file')
    count=struct.unpack('<Q',raw[:8])[0]
    if count>1024*1024 or 8+count>len(raw):
        raise ValueError('Invalid header size')
    header=json.loads(raw[8:8+count]);body=memoryview(raw)[8+count:]
    tensors={};ranges=[]
    for name,spec in header.items():
        if name=='__metadata__':continue
        dtype=DTYPES[spec['dtype']];shape=spec['shape'];start,end=spec['data_offsets']
        if not 0<=start<=end<=len(body) or end-start!=math.prod(shape)*dtype.itemsize:
            raise ValueError('Invalid tensor range: '+name)
        tensors[name]=np.frombuffer(body[start:end],dtype=dtype).reshape(shape).copy()
        ranges.append((start,end))
    cursor=0
    for start,end in sorted(ranges):
        if start!=cursor:raise ValueError('Holes or overlapping buffers')
        cursor=end
    if cursor!=len(body):raise ValueError('Unreferenced data')
    return tensors


def write_tensors(path,tensors,metadata):
    header={'__metadata__':metadata};blocks=[];offset=0
    reverse={v.str:k for k,v in DTYPES.items()}
    for name,value in sorted(tensors.items()):
        array=np.ascontiguousarray(value);block=array.tobytes()
        header[name]=dict(dtype=reverse[array.dtype.str],shape=list(array.shape),data_offsets=[offset,offset+len(block)])
        blocks.append(block);offset+=len(block)
    raw=json.dumps(header,separators=(',',':'),ensure_ascii=True).encode()
    raw+=b' '*((-len(raw))%8)
    with Path(path).open('wb')as f:
        f.write(struct.pack('<Q',len(raw)));f.write(raw)
        for block in blocks:f.write(block)


def quantize(tensors,mode):
    packed={};specs={};errors=[]
    for name,value in tensors.items():
        selected=mode=='all_int8' or (mode=='adapter_int8' and name.startswith('physical_to_mel.'))
        if selected and value.ndim>=2:
            axis=1 if name.startswith('generator.ups.') else 0
            view=np.moveaxis(value.astype(np.float32),axis,0)
            flat=view.reshape(view.shape[0],-1)
            maximum=np.max(np.abs(flat),axis=1)
            scales=np.where(maximum>0,maximum/127.,1.).astype('<f4')
            shape=(len(scales),)+(1,)*(view.ndim-1)
            codes=np.rint(view/scales.reshape(shape)).clip(-127,127).astype('i1')
            packed[name]=np.moveaxis(codes,0,axis).copy()
            scale_name='__scale__.'+name;packed[scale_name]=scales
            specs[name]=dict(axis=axis,scale=scale_name,scheme='symmetric_per_output_channel',zero_point=0)
            reconstructed=np.moveaxis(codes.astype(np.float32)*scales.reshape(shape),0,axis)
            delta=reconstructed-value.astype(np.float32)
            errors.append(dict(name=name,max_absolute_error=float(np.max(np.abs(delta))),
                               relative_L2_error=float(np.linalg.norm(delta)/max(float(np.linalg.norm(value.astype(np.float32))),1e-20))))
        else:
            # Preserve original FP16 constants, biases, normalization and gamma.
            packed[name]=value.copy()
    return packed,specs,errors


def unpack(directory):
    directory=Path(directory);spec=json.loads((directory/'quantization.json').read_text())
    path=directory/'weights.safetensors'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['weights_sha256']:
        raise ValueError('Compressed model SHA256 mismatch')
    values=read_tensors(path);state={}
    for name,info in spec['tensors'].items():
        value=values[name]
        if 'quantization' in info:
            q=info['quantization'];axis=q['axis'];scales=values[q['scale']]
            if value.dtype!=np.int8 or scales.dtype!=np.float32 or not np.isfinite(scales).all() or not (scales>0).all():
                raise ValueError('Invalid quantized tensor '+name)
            shape=[1]*value.ndim;shape[axis]=len(scales)
            out=value.astype(np.float32)*scales.reshape(shape)
        else:
            out=value.astype(np.float32)
        if list(out.shape)!=info['shape']or not np.isfinite(out).all():
            raise ValueError('Invalid restored state '+name)
        state[name]=out.copy()
    return state
