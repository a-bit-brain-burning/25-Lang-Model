"""Softer yiu1 U and a sustained C vowel, before whole-utterance synthesis."""
import types
import numpy as np
import torch
from identifier_u_natural_control import install as natural_install

def install(predictor,features,meta,profile='natural'):
 from new25_acoustic.inference import predict_durations
 regular,u_evidence=natural_install(predictor,features,meta,'natural')
 cs=[u for u in meta.get('identifier_units',[]) if u['symbol']=='C'];evidence=[]
 if not cs:return regular,u_evidence,evidence
 original=predictor.model.encode
 def controlled(self,phones,phonetic,context,lang,mask=None):
  encoded,prosody=original(phones,phonetic,context,lang,mask)
  pause=torch.as_tensor(features['pause'],device=prosody.device)[None]
  before=predict_durations(prosody,pause)[0].cpu().numpy();desired=before.copy();p=prosody.clone()
  for unit in cs:
   start,end=unit['start'],unit['end']
   assert meta['ipa'][start:end]==['zh_letter:s','zh_letter:i']
   desired[start]=min(9,int(before[start]));desired[start+1]=max(18,int(before[start+1])+6)
   evidence.append(dict(symbol='C',code=unit['code'],phone_range=[start,end],before_total_ms=int(before[start:end].sum())*10,after_total_ms=int(desired[start:end].sum())*10,before_vowel_ms=int(before[start+1])*10,after_vowel_ms=int(desired[start+1])*10,consonant_ms=int(desired[start])*10))
   for item in regular:
    if item['code']==unit['code'] and item['phone_range']==[start,end]:item['after_ms']=int(desired[start:end].sum())*10
  for i,(old,new) in enumerate(zip(before,desired)):
   if old!=new:p[0,i,0]=float(np.log(int(new)))
  assert np.array_equal(predict_durations(p,pause)[0].cpu().numpy(),desired)
  return encoded,p
 predictor.model.encode=types.MethodType(controlled,predictor.model)
 return regular,u_evidence,evidence

def install_vocoder_energy(vocoder,result,evidence,profile='natural'):
 """Lighten only U's mel conditioning, leaving generated PCM untouched."""
 if not evidence:return []
 original=vocoder.adapter.forward;records=[]
 def controlled(self,condition,language,mel_frames,context_frames=0):
  mel=original(condition,language,mel_frames,context_frames=context_frames)
  time=(128+256*torch.arange(mel_frames,device=mel.device,dtype=mel.dtype))/22050.;offset=torch.zeros_like(time)
  for item in evidence:
   start_phone,end_phone=item['phone_range'];start=float(result.durations[:start_phone].sum())*.01;duration=float(result.durations[start_phone:end_phone].sum())*.01
   phase=(time-start)/duration;gain=((phase>=0)&(phase<=1)).to(mel.dtype)
   gain=torch.where((phase>=0)&(phase<.2),.5-.5*torch.cos(torch.pi*phase/.2),gain)
   gain=torch.where((phase>.8)&(phase<=1),.5-.5*torch.cos(torch.pi*(1-phase)/.2),gain)
   offset+=gain*(-.14)
   records.append(dict(symbol='U',code=item['code'],start_seconds=start,duration_seconds=duration,peak_logmel_offset=-.14,PCM_gain_or_splicing=False))
  return mel+offset[None,None]
 vocoder.adapter.forward=types.MethodType(controlled,vocoder.adapter)
 return records
