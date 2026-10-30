"""Numeric continuity with protected spans and baseline-direction safeguards.

No text, tone classes, language names or phone identities are consumed.
"""
import numpy as np
from .trajectory_continuity import bridge_trajectory,ContinuityConfig

def guarded_bridge(curve,durations,boundary_allowed,voiced_mask,protected_frame_mask,span_ranges_frames,*,f0=False,spectral_energy=False,config=None,min_direction_cents=25.):
    x=np.asarray(curve,dtype=np.float32);v=np.asarray(voiced_mask,dtype=bool);protected=np.asarray(protected_frame_mask,dtype=bool);spans=np.asarray(span_ranges_frames,dtype=np.int64).reshape(-1,2)
    if protected.shape!=(len(x),) or (spans<0).any() or (spans>len(x)).any() or (spans[:,0]>spans[:,1]).any():raise ValueError('Invalid numeric protected/span contract')
    y,meta=bridge_trajectory(x,durations,boundary_allowed,v,f0=f0,spectral_energy=spectral_energy,config=config)
    original_changed=meta['modified_mask'].copy();y[protected]=x[protected];guard_events=[];f0index=(config or ContinuityConfig()).f0_index
    # Every reversion restores baseline entries; modifications only decrease.
    for iteration in range(len(spans)+1):
        reverted=False
        for span_index,(a,b) in enumerate(spans):
            indices=np.flatnonzero(v[a:b])+a
            if len(indices)<2:continue
            start,end=int(indices[0]),int(indices[-1]);baseline=float((x[end,f0index]-x[start,f0index])*1200/np.log(2));candidate=float((y[end,f0index]-y[start,f0index])*1200/np.log(2))
            if abs(baseline)>=min_direction_cents and baseline*candidate<0:
                changed=bool(np.any(y[a:b,f0index]!=x[a:b,f0index]));y[a:b,f0index]=x[a:b,f0index];reverted|=changed
                guard_events.append({'span_index':span_index,'start_frame':int(a),'stop_frame':int(b),'voiced_start':start,'voiced_end':end,'baseline_excursion_cents':baseline,'candidate_excursion_cents':candidate,'action':'revert_local_F0_to_baseline','iteration':iteration})
        if not reverted:break
    for a,b in spans:
        indices=np.flatnonzero(v[a:b])+a
        if len(indices)>1:
            base=float(x[indices[-1],f0index]-x[indices[0],f0index]);new=float(y[indices[-1],f0index]-y[indices[0],f0index])
            assert abs(base)*1200/np.log(2)<min_direction_cents or base*new>=0
    changed=np.any(y!=x,axis=-1);meta.update(modified_mask=changed,changed_frames=int(changed.sum()),candidate_changed_frames=int(original_changed.sum()),protected_changed_frames_reverted=int((original_changed&protected).sum()),guard_events=guard_events,guard_min_baseline_excursion_cents=float(min_direction_cents),protected_frames=int(protected.sum()),direction_guard_scope='numeric unit and supplied span voiced endpoints; baseline preservation is not tone correctness')
    assert np.array_equal(y[protected],x[protected]);return y,meta
