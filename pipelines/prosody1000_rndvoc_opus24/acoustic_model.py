import torch
from torch import nn
from torch.nn import functional as F
from new25_acoustic.model import Acoustic,LocalBlock

class ExpandedAcoustic(Acoustic):
 def __init__(self,vocab_size):
  super().__init__(vocab_size)
  for dilation in [4,8]:self.encoder.append(LocalBlock(128,dilation=dilation))
  for dilation in [8,16]:self.decoder.append(LocalBlock(192,dilation=dilation,causal=True))
  for block in list(self.encoder)[6:]+list(self.decoder)[3:]:
   nn.init.zeros_(block.down.weight);nn.init.zeros_(block.down.bias)


class ProsodyLearningLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(256,48)
        self.language=nn.Embedding(25,48)
        self.local=nn.Conv1d(48,48,5,padding=2,groups=48)
        self.output=nn.Linear(48,5)
        # Preserve the selected parent before any learning takes place.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x,lang,mask=None):
        weight=torch.ones_like(x[:,:,:1]) if mask is None else mask[:,:,None].to(x.dtype)
        summary=(x*weight).sum(1,keepdim=True)/weight.sum(1,keepdim=True).clamp_min(1)
        h=F.silu(self.input(torch.cat([x,summary.expand_as(x)],-1))+self.language(lang)[:,None])
        h=F.silu(h+self.local((h*weight).transpose(1,2)).transpose(1,2))*weight
        return self.output(h)*weight

class UnifiedAcoustic(ExpandedAcoustic):
    def __init__(self,vocab_size):
        super().__init__(vocab_size)
        self.prosody_learning=ProsodyLearningLayer()

    def encode(self,phones,phonetic,context,lang,mask=None,prosody_control=None):
        x,p=super().encode(phones,phonetic,context,lang,mask)
        p=p+self.prosody_learning(x,lang,mask)
        # Optional residual controls share the learned head's five coordinates:
        # regular/pause log-duration, pause logit, normalized F0, log-energy.
        if prosody_control is not None:
            if prosody_control.shape!=p.shape or not torch.isfinite(prosody_control).all():
                raise ValueError('Expected finite [B,phones,5] prosody controls')
            p=p+prosody_control*(1 if mask is None else mask[:,:,None])
        return x,p
