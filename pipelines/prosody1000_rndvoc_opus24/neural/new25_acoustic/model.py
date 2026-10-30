"""Acoustic network extracted from the trained New25 reference implementation.

No waveform model is included. Parameter names and forward mathematics match the
original acoustic module. Public inference uses FP32 operations with FP16-rounded
export weights loaded as FP32 tensors.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LocalBlock(nn.Module):
    def __init__(self, width, expansion=2, dilation=1, causal=False):
        super().__init__()
        self.radius = dilation*2
        self.causal = causal
        self.dw = nn.Conv1d(width,width,5,groups=width,dilation=dilation)
        self.up = nn.Linear(width,width*expansion)
        self.down = nn.Linear(width*expansion,width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, state=None):
        if self.causal:
            if state is None: state=x.new_zeros(x.shape[0],self.radius*2,x.shape[2])
            z=torch.cat([state,x],1)
            state=z[:,-self.radius*2:]
        else:
            z=F.pad(x.transpose(1,2),(self.radius,self.radius)).transpose(1,2)
        h=self.dw(z.transpose(1,2)).transpose(1,2)
        h=self.down(F.silu(self.up(self.norm(h))))
        return x+h, state


class Acoustic(nn.Module):
    def __init__(self, vocab_size, phone_width=128, frame_width=192):
        super().__init__()
        self.phone=nn.Embedding(vocab_size,phone_width,padding_idx=0)
        with torch.no_grad():self.phone.weight[:2].zero_()
        self.language=nn.Embedding(25,phone_width)
        self.phonetic=nn.ModuleDict({str(d):nn.Linear(d,phone_width) for d in (32,36)})
        self.context=nn.ModuleDict({str(d):nn.Linear(d,phone_width) for d in (22,34)})
        self.encoder=nn.ModuleList([LocalBlock(phone_width,dilation=d) for d in (1,1,2,2,1,1)])
        self.prosody=nn.Sequential(nn.Linear(phone_width,128),nn.SiLU(),nn.Linear(128,5))
        self.language_prosody=nn.Embedding(25,5)
        nn.init.zeros_(self.language_prosody.weight)
        self.frame_in=nn.Linear(phone_width+4+3,frame_width)
        self.rnn=FP32GRUCell(frame_width,frame_width)
        self.decoder=nn.ModuleList([LocalBlock(frame_width,causal=True,dilation=d) for d in (1,2,4)])
        self.heads=nn.ModuleDict({str(d):nn.Linear(frame_width,d) for d in (31,39)})
        self.language_acoustic=nn.Embedding(25,39)
        nn.init.zeros_(self.language_acoustic.weight)

    def encode(self, phones, phonetic, context, lang, mask=None):
        x=self.phone(phones)+self.language(lang)[:,None]
        x=x+self.phonetic[str(phonetic.shape[-1])](phonetic)
        x=x+self.context[str(context.shape[-1])](context)
        if mask is not None: x=x*mask[:,:,None]
        for block in self.encoder:
            x,_=block(x)
            if mask is not None: x=x*mask[:,:,None]
        p=self.prosody(x)+self.language_prosody(lang)[:,None]
        # log1p duration for regular phones / pauses, pause occurrence, F0, energy.
        return x,p

    def decode(self, encoded, prosody, frame_phone, position, lang, dimension, state=None):
        h=encoded.gather(1,frame_phone[:,:,None].expand(-1,-1,encoded.shape[-1]))
        p=prosody.gather(1,frame_phone[:,:,None].expand(-1,-1,5))[:,:,[2,3,4]]
        x=F.silu(self.frame_in(torch.cat([h,p,position],-1)))
        rs=x.new_zeros(x.shape[0],self.rnn.hidden,dtype=torch.float32) if state is None else state['gru']
        sequence=[]
        for t in range(x.shape[1]):
            rs=self.rnn(x[:,t],rs);sequence.append(rs)
        x=torch.stack(sequence,1)
        states=[]
        for i,block in enumerate(self.decoder):
            x,s=block(x,None if state is None else state['conv'][i]);states.append(s)
        y=self.heads[str(dimension)](x)+self.language_acoustic(lang)[:,:dimension,None].transpose(1,2)
        return y,{'gru':rs,'conv':states}


class FP32GRUCell(nn.Module):
    """Affine autocast, gate math/long-lived state FP32; r,z,n convention."""
    def __init__(self, inputs, hidden):
        super().__init__();self.hidden=hidden;self.fused_execution=False
        self.ih=nn.Linear(inputs,3*hidden)
        self.hh=nn.Linear(hidden,3*hidden,bias=False)
    def forward(self,x,h):
        if self.fused_execution and x.is_cuda:
            # Preserve original FP16 affine rounding, including input bias.
            # Native GRU gate and persistent-state math must remain FP32:
            # this ATen operator otherwise has an AMP lower-precision policy.
            ig=self.ih(x).float();hg=self.hh(h).float()
            with torch.autocast('cuda',enabled=False):
                return torch.ops.aten._thnn_fused_gru_cell.default(
                    ig,hg,h.float(),None,None)[0]
        ar,az,an=self.ih(x).float().chunk(3,-1)
        br,bz,bn=self.hh(h).float().chunk(3,-1)
        r=torch.sigmoid(ar+br);z=torch.sigmoid(az+bz)
        return (1-z)*torch.tanh(an+r*bn)+z*h.float()
