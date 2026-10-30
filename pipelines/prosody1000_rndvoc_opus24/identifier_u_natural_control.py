"""U prominence via pre-decoder duration, energy conditioning and a boundary."""
import types
import numpy as np
import torch
from identifier_rhythm import install as install_rhythm

PROFILES={
 'natural':dict(extra_vowel_ms=0,energy_control_delta=0.,following_gap_ms=20,mel_log_gain=0.),
 'moderate':dict(extra_vowel_ms=60,energy_control_delta=.5,following_gap_ms=60,mel_log_gain=.35),
 'strong':dict(extra_vowel_ms=100,energy_control_delta=.8,following_gap_ms=80,mel_log_gain=.55),
}

def install(predictor,features,meta,profile='moderate'):
 from new25_acoustic.inference import predict_durations
 regular=install_rhythm(predictor,features,meta)
 units=meta.get('identifier_units',[]);us=[u for u in units if u['symbol']=='U']
 evidence=[]
 if not us:return regular,evidence
 config=PROFILES[profile];original=predictor.model.encode
 def emphasized(self,phones,phonetic,context,lang,mask=None):
  encoded,prosody=original(phones,phonetic,context,lang,mask)
  pause=torch.as_tensor(features['pause'],device=prosody.device)[None]
  durations=predict_durations(prosody,pause)[0].cpu().numpy();target=durations.copy();p=prosody.clone()
  for unit in us:
   start,end=unit['start'],unit['end']
   expected_phone='u' if meta.get('U_reading') in ['wu1','yiu1'] else 'joʊ'
   core=[i for i in range(start,end) if meta['ipa'][i]==expected_phone]
   assert len(core)==1,'The stored U pronunciation must have one explicit Mandarin you1 nucleus'
   index=core[0]
   if meta.get('U_reading')=='yiu1':
    assert meta['ipa'][start:end]==['i','u']
    # One syllable: a brief i transition, a sustained u, no interior pause.
    total=int(durations[start:end].sum())+config['extra_vowel_ms']//10
    target[start]=4 if profile=='natural' else 5;target[index]=total-int(target[start])
   else:target[index]+=config['extra_vowel_ms']//10
   p[0,index,4]+=config['energy_control_delta']
   preceding=[u for u in units if u['code_id']==unit['code_id'] and u['symbol_index']==unit['symbol_index']-1]
   before_gap=[]
   if profile=='natural' and preceding:
    before_gap=list(range(preceding[0]['end'],start));assert before_gap
    target[before_gap]=0;target[before_gap[0]]=1 if preceding[0]['symbol'].isalpha() else 2
   following=[u for u in units if u['code_id']==unit['code_id'] and u['symbol_index']==unit['symbol_index']+1]
   gap=[]
   if following:
    gap=list(range(end,following[0]['start']));assert gap and all(features['pause'][i] for i in gap)
    target[gap]=0;target[gap[0]]=1 if profile=='natural' and following[0]['symbol'].isalpha() else config['following_gap_ms']//10
   evidence.append(dict(symbol='U',code=unit['code'],profile=profile,
    previous_unit_ms=int(durations[start:end].sum())*10,new_unit_ms=int(target[start:end].sum())*10,
    previous_nucleus_ms=int(durations[index])*10,new_nucleus_ms=int(target[index])*10,
    energy_condition_delta=config['energy_control_delta'],preceding_gap_ms=int(target[before_gap].sum())*10,following_gap_ms=int(target[gap].sum())*10,
    nucleus_phone_index=index,phone_range=[start,end]))
   for item in regular:
    if item['code']==unit['code'] and item['phone_range']==[start,end]:
     item['balanced_unit_ms']=item['after_ms'];item['after_ms']=int(target[start:end].sum())*10
  for i,(old,new) in enumerate(zip(durations,target)):
   if old==new:continue
   if features['pause'][i]:p[0,i,1]=float(np.log(max(1,int(new))));p[0,i,2]=1. if new else -1.
   else:p[0,i,0]=float(np.log(int(new)))
  assert np.array_equal(predict_durations(p,pause)[0].cpu().numpy(),target)
  return encoded,p
 predictor.model.encode=types.MethodType(emphasized,predictor.model)
 return regular,evidence

def install_vocoder_energy(vocoder,result,evidence,profile='moderate'):
 """Smooth U emphasis on raw mel conditioning; no generated PCM gain/edit."""
 if not evidence:return []
 config=PROFILES[profile];original=vocoder.adapter.forward;records=[]
 def controlled(self,condition,language,mel_frames,context_frames=0):
  mel=original(condition,language,mel_frames,context_frames=context_frames)
  time=(128+256*torch.arange(mel_frames,device=mel.device,dtype=mel.dtype))/22050.
  envelope=torch.zeros_like(time)
  for item in evidence:
   i=item['nucleus_phone_index'];start=float(result.durations[:i].sum())*.01;duration=float(result.durations[i])*.01
   phase=(time-start)/duration
   # 20% raised-cosine onset/offset, constant emphasis in the middle.
   gain=torch.where((phase>=0)&(phase<=1),torch.ones_like(time),torch.zeros_like(time))
   gain=torch.where((phase>=0)&(phase<.2),.5-.5*torch.cos(torch.pi*phase/.2),gain)
   gain=torch.where((phase>.8)&(phase<=1),.5-.5*torch.cos(torch.pi*(1-phase)/.2),gain)
   envelope=torch.maximum(envelope,gain*config['mel_log_gain'])
   records.append(dict(code=item['code'],symbol='U',start_seconds=start,nucleus_seconds=duration,
    peak_logmel_offset=config['mel_log_gain'],control_before_RNDVoC=True,PCM_gain_or_splicing=False))
  return mel+envelope[None,None]
 vocoder.adapter.forward=types.MethodType(controlled,vocoder.adapter)
 return records
