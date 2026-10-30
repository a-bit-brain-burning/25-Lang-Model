"""Language-balanced acoustic minibatches on the real duration timeline."""
import numpy as np
import torch
from .contracts import LANGUAGES
from .data import frame_layout

def collate(corpus,indices,device):
    items=[corpus.item(i) for i in indices]
    langs={a['row']['language'] for a in items}
    if len(langs)!=1:raise ValueError('A microbatch shares language; scheduler rotates all 25')
    lang=langs.pop();n=corpus.norm[lang]
    ps=[len(a['phones']) for a in items];ts=[len(a['curve']) for a in items]
    p=max(ps);t=max(ts)
    fields={}
    phonekeys=('phones','phonetic','context','pause','durations','phone_valid','phone_prosody','phone_prosody_weight')
    framekeys=('curve','f0_valid','voice_weight','silence')
    for key in phonekeys+framekeys:
        length=p if key in phonekeys else t
        value=np.zeros((len(items),length,*items[0][key].shape[1:]),dtype=items[0][key].dtype)
        for j,a in enumerate(items):value[j,:len(a[key])]=a[key]
        fields[key]=value
    fields['phone_mask']=np.arange(p)[None]<np.asarray(ps)[:,None]
    fields['frame_mask']=np.arange(t)[None]<np.asarray(ts)[:,None]
    fields['frame_phone']=np.zeros((len(items),t),np.int64)
    fields['position']=np.zeros((len(items),t,4),np.float32)
    fields['target']=np.zeros_like(fields['curve'])
    for j,a in enumerate(items):
        fi,po=frame_layout(a['durations']);fields['frame_phone'][j,:ts[j]]=fi;fields['position'][j,:ts[j]]=po
        fields['target'][j,:ts[j]]=corpus.normalized(a)
    for key in ('phonetic','context'):
        fields[key]=(fields[key]-np.asarray(n[key+'_mean'],np.float32))/np.asarray(n[key+'_std'],np.float32)
        fields[key]*=fields['phone_mask'][:,:,None]
    # Phone F0 / log-energy use the same per-language train-only normalization.
    fields['phone_prosody']=(fields['phone_prosody']-np.asarray(n['mean'][:2],np.float32))/np.asarray(n['std'][:2],np.float32)
    b={k:torch.from_numpy(v.copy()).to(device) for k,v in fields.items()}
    b.update(lang=torch.full((len(items),),LANGUAGES.index(lang),device=device,dtype=torch.long),
             norm=n,rows=[a['row'] for a in items],lengths=ts)
    return b

def acoustic_forward(model,b):
    e,p=model.encode(b['phones'],b['phonetic'],b['context'],b['lang'],b['phone_mask'])
    out,_=model.decode(e,p,b['frame_phone'],b['position'],b['lang'],b['curve'].shape[-1])
    return out,p
