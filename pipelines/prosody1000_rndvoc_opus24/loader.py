"""Strict portable loading of the exact deployed acoustic/prosody and RND models."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'neural'))
import numpy as np
import torch
from storage import unpack,read_tensors
from acoustic_model import UnifiedAcoustic
from new25_acoustic.inference import AcousticPredictor

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load():
    torch.set_num_threads(2);torch.manual_seed(20261030)
    package=ROOT/'models/acoustic'
    assert sha(package/'weights.safetensors')=='e2b5342c3d116693de58fe3a8cf8e55cf1aacaad5816fe0ebb38aa26b456fc74'
    values=unpack(package);vocabulary=read(package/'assets/vocabulary.json')
    model=UnifiedAcoustic(len(vocabulary))
    model.load_state_dict({k[len('acoustic.'):]:torch.from_numpy(v) for k,v in values.items() if k.startswith('acoustic.')},strict=True)
    acoustic=AcousticPredictor(model,read(package/'assets/normalization.json'),vocabulary)
    import new25_rndvoc.inference as interface
    # Preserve the stored F16 parameters/F32 projection buffers. The published
    # RND loader retains all original SHA/name/shape/dtype validation.
    interface.load_safetensors=lambda p:{k:torch.from_numpy(v)for k,v in read_tensors(p).items()}
    vocoder=interface.RNDVoCVocoder.from_pretrained(ROOT/'models/rndvoc')
    for part,file in [(vocoder.generator,'vocoder_fp16.safetensors'),(vocoder.adapter,'adapter_fp16.safetensors')]:
        reference=read_tensors(ROOT/'models/rndvoc'/file)
        assert all(torch.equal(part.state_dict()[k],torch.from_numpy(v).float())for k,v in reference.items())
    assert sum(p.numel()for p in model.parameters())==1637644
    assert sum(p.numel()for p in model.prosody_learning.parameters())==14069
    assert sum(p.numel()for p in vocoder.adapter.parameters())==1902992
    assert sum(p.numel()for p in vocoder.generator.parameters())==713929
    assert len(values)==136
    return acoustic,vocoder
