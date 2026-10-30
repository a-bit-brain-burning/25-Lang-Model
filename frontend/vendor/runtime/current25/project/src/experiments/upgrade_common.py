"""Dynamic representation contracts and bounded-memory record access."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from ..common import PROJECT, save_json
from ..training.train_acoustic import TRAIN_FIELDS

OUT = PROJECT / 'outputs/representation_upgrade_20261014'

def semantic_indices(dim, bands):
    return list(range(dim+1)) + list(range(33,33+bands)) + [38]

class LazyRecords:
    def __init__(self, root, split, dim=32, bands=5, cache_size=24):
        if split not in ('train','validation','test'): raise ValueError(split)
        self.root=Path(root); self.dim=dim; self.bands=bands; self.limit=cache_size; self.cache=OrderedDict()
        self.rows=[json.loads(x) for x in (self.root/'manifest.jsonl').read_text().splitlines() if x.strip()]
        self.rows=[x for x in self.rows if x['split']==split]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        i=int(i)
        if i<0 or i>=len(self): raise IndexError(i)
        if i not in self.cache:
            row=self.rows[i]
            with np.load(self.root/row['record_path'],allow_pickle=False) as z:
                a={k:z[k] for k in TRAIN_FIELDS if k in z.files}
                for dest,full,ap3 in [('target_controls','controls32ap5','controls_ap3'),('control_mask','mask32ap5','mask_ap3'),('target_curve','curve32ap5','curve_ap3')]:
                    v=z[full]
                    ap=z[ap3] if self.bands==3 else v[...,33:38]
                    a[dest]=np.concatenate([v[...,:self.dim+1],ap,v[...,-1:]],axis=-1).astype(np.float32)
            self.cache[i]=(row,a)
            if len(self.cache)>self.limit:self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]


def shared_normalization(root):
    """One train-only pass for all shared channels; AP3 cannot derive from AP5."""
    path=OUT/'shared_normalization.json'
    if path.exists():return json.loads(path.read_text())
    records=LazyRecords(root,'train',32,5)
    total=np.zeros(42);square=total.copy();count=total.copy();voiced=units=0.
    for i,row in enumerate(records.rows):
        with np.load(Path(root)/row['record_path'],allow_pickle=False) as z:
            t=np.concatenate([z['controls32ap5'],z['controls_ap3']],-1).reshape(-1,42)
            m=np.concatenate([z['mask32ap5'],z['mask_ap3']],-1).reshape(-1,42)
            total+=(t*m).sum(0);square+=(t*t*m).sum(0);count+=m.sum(0)
            voice=1/(1+np.exp(-np.clip(t[:,38],-30,30)))
            voiced+=(voice*m[:,38]).sum();units+=m[:,38].sum()
    mean=total/np.maximum(count,1);scale=np.sqrt(np.maximum(square/np.maximum(count,1)-mean**2,0))
    floor=np.array([.1,.5]+[.05]*36+[1.]+[.05]*3)
    scale=np.maximum(scale,floor);mean[38]=0;scale[38]=1
    value={'means':mean.tolist(),'scales':scale.tolist(),'prior':float(voiced/units),'train_records':len(records),'source':'all 7296 train only; shared spectral/F0/VUV and independent AP3/AP5 statistics','test_loaded':False}
    save_json(path,value);return value


def variant_stats(stats,dim,bands):
    indices=list(range(dim+1))+(list(range(39,42)) if bands==3 else list(range(33,38)))+[38]
    return np.array(stats['means'],np.float32)[indices],np.array(stats['scales'],np.float32)[indices],stats['prior']
