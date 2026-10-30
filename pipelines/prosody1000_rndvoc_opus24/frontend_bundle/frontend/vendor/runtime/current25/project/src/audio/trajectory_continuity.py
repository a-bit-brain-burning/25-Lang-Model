"""Conservative numeric boundary bridges; no text, phone names or language rules.

The caller decides eligible boundaries from its frontend. Only a bounded
neighbourhood of existing frames is modified: no duration, AP or VUV changes.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class ContinuityConfig:
    half_window_frames:int=1
    minimum_segment_frames:int=4
    max_f0_change_cents:float=150.
    max_energy_log_change:float=.3
    max_spectral_vector_change:float=.6
    f0_index:int=0
    energy_index:int=1
    spectral_start:int=2
    spectral_stop:int=25


def bridge_trajectory(curve,durations,boundary_allowed,voiced_mask,*,f0=False,spectral_energy=False,config=None):
    """Return independent curve and audit. boundary_allowed[j] means before j.

    Zero-duration units are skipped; the caller must approve the actual next
    positive-duration unit. Nonzero pauses are ordinary segments and are never
    skipped. Both whole windows must be acoustically voiced. At most a quarter
    of either segment can change, keeping segment/tone interiors untouched.
    """
    config=config or ContinuityConfig();x=np.asarray(curve,dtype=np.float32)
    d=np.asarray(durations,dtype=np.int64);allowed=np.asarray(boundary_allowed,dtype=bool);v=np.asarray(voiced_mask,dtype=bool)
    if x.ndim!=2 or len(x)!=int(d.sum()) or allowed.shape!=d.shape or v.shape!=(len(x),) or (d<0).any() or not np.isfinite(x).all():raise ValueError('Invalid numeric trajectory/boundary contract')
    if config.half_window_frames<1 or config.minimum_segment_frames<4:raise ValueError('Need conservative window and >=4-frame segments')
    y=x.copy();modified=np.zeros(len(x),bool);events=[];offset=np.concatenate(([0],d.cumsum()));active=np.flatnonzero(d>0)
    for left,right in zip(active[:-1],active[1:]):
        if not allowed[right]:continue
        boundary=int(offset[right]);half=min(config.half_window_frames,int(d[left]//4),int(d[right]//4))
        if min(d[left],d[right])<config.minimum_segment_frames or not half:continue
        lo,hi=boundary-half,boundary+half
        if not v[lo-1:hi+1].all():continue
        columns=([config.f0_index] if f0 else [])+([config.energy_index]+list(range(config.spectral_start,config.spectral_stop)) if spectral_energy else [])
        if not columns:continue
        a=x[lo-1,columns];b=x[hi,columns];alpha=np.arange(1,hi-lo+1,dtype=np.float32)[:,None]/(hi-lo+1)
        target=a[None]+alpha*(b-a)[None];delta=target-x[lo:hi][:,columns]
        cursor=0
        if f0:
            cap=config.max_f0_change_cents*np.log(2)/1200
            delta[:,0]=np.clip(delta[:,0],-cap,cap);cursor=1
        if spectral_energy:
            delta[:,cursor]=np.clip(delta[:,cursor],-config.max_energy_log_change,config.max_energy_log_change)
            shape=delta[:,cursor+1:];norm=np.linalg.norm(shape,axis=-1,keepdims=True)
            shape*=np.minimum(1.,config.max_spectral_vector_change/np.maximum(norm,1e-12))
        y[lo:hi,columns]=x[lo:hi][:,columns]+delta
        changed=np.any(delta!=0,axis=-1);modified[lo:hi]|=changed
        events.append({'left_segment':int(left),'right_segment':int(right),'boundary_frame':boundary,'changed_start':lo,'changed_stop':hi,'columns':columns,'changed_frames':int(changed.sum())})
    return y,{'config':asdict(config),'events':events,'changed_frames':int(modified.sum()),'modified_mask':modified,'durations_unchanged':True,'AP_and_VUV_unchanged':True,'scope':'local numeric interpolation; not evidence of perceptual improvement'}
