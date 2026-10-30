"""Dump a fresh ordinary-text full-graph trace for native inference development.

Each .npy is plain numeric, pickle-free and named by its module invocation.
GRU cell calls are excluded by default to keep whole-utterance traces small.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'int8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--language',required=True)
    p.add_argument('--text',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-gru',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from frontend import TextFrontend
    from runtime import load
    from new25_hifigan.inference import write_pcm16
    acoustic,vocoder=load(ROOT/'int8')
    with TextFrontend()as frontend:features,metadata=frontend.encode(a.text,a.language)
    np.savez_compressed(a.output/'frontend.npz',**features)
    rows=[];handles=[];counters={}
    def save(value,name):
        if isinstance(value,(torch.Tensor,np.ndarray)):
            array=value.detach().cpu().numpy()if isinstance(value,torch.Tensor)else value
            np.save(a.output/(name+'.npy'),array,allow_pickle=False)
            rows.append(dict(file=name+'.npy',shape=list(array.shape),dtype=str(array.dtype)))
        elif isinstance(value,(list,tuple)):
            for i,item in enumerate(value):save(item,name+'_'+str(i))
        elif isinstance(value,dict):
            for k,item in value.items():save(item,name+'_'+str(k))
    def hook(name):
        def capture(module,inputs,outputs):
            i=counters.get(name,0);counters[name]=i+1
            save(inputs,name+'_'+str(i)+'_in');save(outputs,name+'_'+str(i)+'_out')
        return capture
    for prefix,model in [('acoustic',acoustic.model),('vocoder',vocoder)]:
        for name,module in model.named_modules():
            if not name:continue
            if not a.include_gru and(name=='rnn'or name.startswith('rnn.')):continue
            handles.append(module.register_forward_hook(hook(prefix+'.'+name)))
    with torch.inference_mode():
        result=acoustic.predict(language=a.language,**features)
        save(result.normalized,'acoustic_normalized')
        save(result.physical,'acoustic_physical')
        save(result.frame_phone,'frame_phone')
        save(result.position,'frame_position')
        np.save(a.output/'durations.npy',result.durations,allow_pickle=False)
        audio=vocoder.synthesize(result.normalized[None],a.language)
        save(audio,'audio_16k');write_pcm16(a.output/'audio.wav',audio)
    for h in handles:h.remove()
    (a.output/'TRACE.json').write_text(json.dumps(dict(input=metadata,arrays=rows,
        reference_audio_or_durations_used=False,execution='FP32 from current INT8 storage'),ensure_ascii=False,indent=2)+'\n')
    print('Saved',len(rows),'numeric arrays to',a.output)


if __name__=='__main__':main()
