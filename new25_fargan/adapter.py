"""Independent31/39 physical timeline -> official20D feature regression."""
import torch
from torch import nn
from torch.nn import functional as F
class Block(nn.Module):
 def __init__(self):
  super().__init__();w=192;self.dw=nn.Conv1d(w,w,7,padding=3,groups=w);self.norm=nn.LayerNorm(w,eps=1e-6);self.up=nn.Linear(w,768);self.down=nn.Linear(768,w);self.gamma=nn.Parameter(torch.full((w,),1/6))
 def forward(self,x,mask):return (x+self.gamma*self.down(F.gelu(self.up(self.norm(self.dw(x.transpose(1,2)).transpose(1,2))))))*mask
class PhysicalFarganAdapter(nn.Module):
 def __init__(self):
  super().__init__();w=192;self.inputs=nn.ModuleDict({str(d):nn.Conv1d(d,w,7,padding=3) for d in (31,39)});self.language=nn.Embedding(25,w);self.norm=nn.LayerNorm(w,eps=1e-6);self.blocks=nn.ModuleList([Block() for _ in range(6)]);self.head=nn.Linear(w,20)
 def forward(self,x,language,mask=None,context_frames=0,frames=None):
  if mask is None:mask=torch.ones(x.shape[:2],device=x.device,dtype=x.dtype)
  mask=mask[:,:,None];z=self.norm(self.inputs[str(x.shape[-1])](x.transpose(1,2)).transpose(1,2)+self.language(language)[:,None])*mask
  for block in self.blocks:z=block(z,mask)
  y=self.head(z).float()
  if frames is None:frames=x.shape[1]-2*context_frames
  return y[:,context_frames:context_frames+frames]
 def expected(self,d):
  inactive='inputs.'+('31' if d==39 else '39')+'.';return {n for n,p in self.named_parameters() if not n.startswith(inactive)}
