"""Isolated group-local CEA pitch scheduling, with the real AcousticCursor state.

Full text-unit encoding is retained. No main runtime/model/weights are changed.
"""
from types import SimpleNamespace
import numpy as np
import torch
from acoustic_cursor import AcousticCursor
from sync_cea342e_20261030 import cea_controls

class ObservedAcousticCursor(AcousticCursor):
    """Same pull arithmetic as AcousticCursor, additionally retain exact physical.

    Inverting normalized output would round raw logF0/VUV and change CEA pitch.
    """
    @torch.inference_mode()
    def pull(self,count):
        from new25_acoustic.inference import denormalize,waveform_condition
        if self.closed or self.cancelled:raise RuntimeError('Acoustic request is closed/cancelled')
        if type(count)is not int or count<1:raise ValueError('Positive pull required')
        if self.offset==self.frames:return np.empty((0,self.dimension),np.float32)
        end=min(self.frames,self.offset+count);owners,position=self._layout(self.offset,end)
        with torch.autocast(self.device.type,enabled=False):
            prediction,state=self.predictor.model.decode(self.encoded,self.prosody,owners,position,self.lang,self.dimension,self.state)
            physical=denormalize(prediction,self.norm);result=waveform_condition(physical,self.norm)
        if not torch.isfinite(result).all()or not torch.isfinite(state['gru']).all()or any(not torch.isfinite(x).all()for x in state['conv']):raise ValueError('Nonfinite decoder')
        self.state={'gru':state['gru'].detach().clone(),'conv':[x.detach().clone()for x in state['conv']]}
        self.last_physical=physical[0].detach().cpu().numpy().copy();output=result[0].detach().cpu().numpy().copy()
        self.offset=end;self.calls+=1
        return output

class LazyConditionCursor:
    def __init__(self,predictor,features,meta,language,*,active=True,raw_chunk=16):
        self.raw=ObservedAcousticCursor(predictor,features,language);self.frames=self.raw.frames;self.dimension=self.raw.dimension;self.durations=self.raw.durations
        self.features,self.meta,self.norm=features,meta,self.raw.norm;self.offset=self.calls=0;self.raw_chunk=raw_chunk;self.first_pcm_decoded_frames=None
        self.normalized=np.zeros((self.frames,self.dimension),np.float32);self.physical=np.zeros_like(self.normalized);self.pitch_records=[]
        self.frame_phone=np.repeat(np.arange(len(self.durations)),self.durations);ends=np.r_[0,np.cumsum(self.durations)];self.groups=[]
        if active:
            for gid in sorted({u['code_id']for u in meta.get('identifier_units',[])}):
                units=[u for u in meta['identifier_units']if u['code_id']==gid];a=int(ends[units[0]['start']]);b=int(ends[units[-1]['end']])
                if self.groups and a<self.groups[-1]['end']:raise ValueError('Identifier groups overlap/out of order')
                self.groups.append(dict(begin=a,end=b,units=units,done=False))
    def _raw_through(self,end):
        while self.raw.offset<end:
            a=self.raw.offset;x=self.raw.pull(self.raw_chunk);b=self.raw.offset
            self.normalized[a:b]=x;self.physical[a:b]=self.raw.last_physical
    def _pitch(self,g):
        self._raw_through(g['end'])
        # Exact raw physical for the complete group; unrelated future rows are
        # not selected by original CEA group medians. Copy only this group's
        # corrected rows back, retaining raw physical for every later group.
        result=SimpleNamespace(physical=self.physical.copy(),normalized=self.normalized.copy(),prediction=np.zeros_like(self.normalized),durations=self.durations,frame_phone=self.frame_phone)
        meta=dict(self.meta,identifier_units=g['units']);records=cea_controls.apply_pitch(result,self.features,meta,self.norm)
        a,b=g['begin'],g['end'];self.normalized[a:b]=result.normalized[a:b]
        self.pitch_records.extend(records);g['done']=True
    def pull(self,count):
        if type(count)is not int or count<1:raise ValueError('Positive pull required')
        end=min(self.frames,self.offset+count)
        for g in self.groups:
            if not g['done']and g['begin']<end:self._pitch(g)
        self._raw_through(end);out=self.normalized[self.offset:end].copy();self.offset=end;self.calls+=1
        return out
    def report(self):
        return dict(self.raw.report(),emitted_frames=self.offset,decoded_frames=self.raw.offset,lazy_identifier_groups=len(self.groups),finalized_identifier_groups=sum(g['done']for g in self.groups),full_acoustic_unit_materialized_before_first_PCM=(self.first_pcm_decoded_frames==self.frames if self.first_pcm_decoded_frames is not None else None),first_PCM_decoded_frames=self.first_pcm_decoded_frames,condition_cache_bytes=self.normalized.nbytes+self.physical.nbytes,raw_chunk=self.raw_chunk,training_updates=0)
