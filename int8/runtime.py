"""Load a real compressed package, then run the unchanged reference networks."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

from codec import unpack


def load(directory,threads=2):
    directory=Path(directory).resolve()
    sys.path.insert(0,str(directory/'runtime'))
    from new25_acoustic.model import Acoustic
    from new25_acoustic.inference import AcousticPredictor
    from new25_hifigan.inference import HiFiGANVocoder,validate_metadata
    from new25_hifigan.model import Generator,PhysicalMelAdapter,validate_generator_config
    torch.set_num_threads(threads)
    vocab=json.loads((directory/'assets/vocabulary.json').read_text())
    normalization=json.loads((directory/'assets/normalization.json').read_text())
    config=validate_generator_config(json.loads((directory/'assets/hifigan_config.json').read_text()))
    metadata=validate_metadata(json.loads((directory/'assets/hifigan_model_config.json').read_text()))
    arrays=unpack(directory)
    acoustic=Acoustic(len(vocab))
    vocoder=HiFiGANVocoder(Generator(config),PhysicalMelAdapter(),device='cpu')
    a={k.removeprefix('acoustic.'):torch.from_numpy(v)for k,v in arrays.items()if k.startswith('acoustic.')}
    v={k:torch.from_numpy(x)for k,x in arrays.items()if not k.startswith('acoustic.')}
    acoustic.load_state_dict(a,strict=True);vocoder.load_state_dict(v,strict=True)
    assert sum(p.numel()for p in acoustic.parameters())==1186071
    assert sum(p.numel()for p in vocoder.parameters())==2828977
    vocoder.metadata=metadata
    return AcousticPredictor(acoustic,normalization,vocab,device='cpu'),vocoder.eval()
