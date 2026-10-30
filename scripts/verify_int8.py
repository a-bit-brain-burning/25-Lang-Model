"""Verify INT8 storage layout and bundled frontend integrity without PyTorch."""
from pathlib import Path
import hashlib
import json
import math
import struct

ROOT=Path(__file__).resolve().parents[1]


def main():
    package=ROOT/'int8'
    spec=json.loads((package/'quantization.json').read_text())
    raw=(package/'weights.safetensors').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==spec['weights_sha256']
    n=struct.unpack('<Q',raw[:8])[0]
    assert 2<=n<=1024*1024 and n+8<=len(raw)
    header=json.loads(raw[8:8+n]);header.pop('__metadata__',None)
    spans=[]
    for name,t in header.items():
        width={'I8':1,'F16':2,'F32':4}[t['dtype']]
        lo,hi=t['data_offsets']
        assert 0<=lo<=hi<=len(raw)-8-n and hi-lo==math.prod(t['shape'])*width,name
        spans.append((lo,hi))
    cursor=0
    for lo,hi in sorted(spans):assert lo==cursor;cursor=hi
    assert cursor==len(raw)-8-n
    quantized=0
    for name,t in spec['tensors'].items():
        assert header[name]['shape']==t['shape'],name
        if 'quantization'in t:
            q=t['quantization'];axis=1 if name.startswith('generator.ups.')else 0
            assert header[name]['dtype']=='I8'and q['axis']==axis and q['zero_point']==0,name
            scale=header[q['scale']]
            assert scale['dtype']=='F32'and scale['shape']==[t['shape'][axis]],name
            lo,hi=scale['data_offsets'];scales=struct.unpack('<'+'f'*((hi-lo)//4),raw[8+n+lo:8+n+hi])
            assert all(math.isfinite(v)and v>0 for v in scales),name
            quantized+=1
        else:assert header[name]['dtype']=='F16',name
    assert len(spec['tensors'])==316 and quantized==142 and len(header)==458
    assert sum(math.prod(t['shape'])for t in spec['tensors'].values())==4015048
    vendor=ROOT/'frontend/vendor';frontend=json.loads((vendor/'frontend/MANIFEST.json').read_text())
    for kind,folder in [('packages','packages'),('libraries','libs')]:
        for name,digest in frontend[kind].items():
            assert hashlib.sha256((vendor/'frontend'/folder/name).read_bytes()).hexdigest()==digest,name
    langs=set(json.loads((package/'assets/normalization.json').read_text()))
    entries=json.loads((vendor/'FRONTEND_INDEX.json').read_text())['models']
    assert langs=={e['short']for e in entries}and len(langs)==25
    print(json.dumps(dict(passed=True,model_parameters=4015048,tensors=316,quantized_weight_tensors=142,
                         weights_bytes=len(raw),frontend_languages=25,full_firmware_acceptance=False),indent=2))


if __name__=='__main__':main()
