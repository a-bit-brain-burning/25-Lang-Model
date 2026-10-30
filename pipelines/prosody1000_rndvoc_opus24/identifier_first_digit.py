"""Emphasize each digit run's first character before decoding and vocoding."""
import types
import numpy as np
import torch
from identifier_light_u_long_c import install as install_previous,install_vocoder_energy as previous_energy

def install(predictor,features,meta,profile='natural',enable_first_digit=True):
 from new25_acoustic.inference import predict_durations
 regular,u_evidence,c_evidence=install_previous(predictor,features,meta,profile)
 units=meta.get('identifier_units',[]);starts=[];records=[]
 for i,u in enumerate(units):
  if u['symbol'].isdigit() and (i==0 or units[i-1]['code_id']!=u['code_id'] or not units[i-1]['symbol'].isdigit()):starts.append(u)
 if not starts or not enable_first_digit:return regular,u_evidence,c_evidence,records
 original=predictor.model.encode
 def controlled(self,phones,phonetic,context,lang,mask=None):
  encoded,prosody=original(phones,phonetic,context,lang,mask);p=prosody.clone()
  pause=torch.as_tensor(features['pause'],device=prosody.device)[None];before=predict_durations(prosody,pause)[0].cpu().numpy();target=before.copy()
  for u in starts:
   start,end=u['start'],u['end'];assert end>start and not np.any(features['pause'][start:end])
   # The final phone is the native Mandarin syllable final, including 二/ɚ/
   # and 四's syllabic final, which an ASCII-vowel classifier would miss.
   core=end-1;target[core]+=5;p[0,core,4]+=.25
   gap=[];i=start-1
   while i>=0 and features['pause'][i]:gap.insert(0,i);i-=1
   assert gap,'The first digit requires an explicit frontend silence boundary'
   old_gap=int(before[gap].sum());target[gap]=0;target[gap[0]]=max(10,old_gap+5) if i>=0 else max(10,old_gap)
   records.append(dict(code=u['code'],symbol=u['symbol'],symbol_index=u['symbol_index'],phone_range=[start,end],before_ms=int(before[start:end].sum())*10,after_ms=int(target[start:end].sum())*10,before_gap_ms=old_gap*10,after_gap_ms=int(target[gap].sum())*10,energy_condition_delta=.25))
   for prior in u_evidence:
    if prior['phone_range'][1]==gap[0]:
     prior['natural_following_gap_ms']=prior['following_gap_ms'];prior['following_gap_ms']=int(target[gap].sum())*10
   for item in regular:
    if item['code']==u['code'] and item['phone_range']==[start,end]:item['after_ms']=int(target[start:end].sum())*10
  for i,(a,b) in enumerate(zip(before,target)):
   if a==b:continue
   if features['pause'][i]:p[0,i,1]=float(np.log(max(1,int(b))));p[0,i,2]=1. if b else -1.
   else:p[0,i,0]=float(np.log(int(b)))
  assert np.array_equal(predict_durations(p,pause)[0].cpu().numpy(),target)
  return encoded,p
 predictor.model.encode=types.MethodType(controlled,predictor.model)
 return regular,u_evidence,c_evidence,records

def install_vocoder_energy(vocoder,result,u_evidence,digit_evidence,profile='natural'):
 u_records=previous_energy(vocoder,result,u_evidence,profile);records=[]
 if not digit_evidence:return u_records,records
 original=vocoder.adapter.forward
 def controlled(self,condition,language,mel_frames,context_frames=0):
  mel=original(condition,language,mel_frames,context_frames=context_frames);time=(128+256*torch.arange(mel_frames,device=mel.device,dtype=mel.dtype))/22050.;offset=torch.zeros_like(time)
  for item in digit_evidence:
   lo,hi=item['phone_range'];start=float(result.durations[:lo].sum())*.01;duration=float(result.durations[lo:hi].sum())*.01;phase=(time-start)/duration;gain=((phase>=0)&(phase<=1)).to(mel.dtype)
   gain=torch.where((phase>=0)&(phase<.2),.5-.5*torch.cos(torch.pi*phase/.2),gain);gain=torch.where((phase>.8)&(phase<=1),.5-.5*torch.cos(torch.pi*(1-phase)/.2),gain)
   offset+=.2*gain;records.append(dict(code=item['code'],symbol=item['symbol'],start_seconds=start,duration_seconds=duration,peak_logmel_offset=.2,PCM_gain_or_splicing=False))
  return mel+offset[None,None]
 vocoder.adapter.forward=types.MethodType(controlled,vocoder.adapter)
 return u_records,records
