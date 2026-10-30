"""Reference-free identifier character rhythm, applied before acoustic decoding."""
from pathlib import Path
import sys,re,json,importlib.util,types
import numpy as np
import torch
VOWELS=set('aeiouɛɑæʌɒɜɔɤəɪʊyɯɨøœɐɚɝ')

def install(predictor,features,meta,minimum_character_ms=200,gap_ms=30):
 from new25_acoustic.inference import predict_durations
 units=meta.get('identifier_units',[]);evidence=[]
 if not units:return evidence
 original=predictor.model.encode
 def controlled(self,phones,phonetic,context,lang,mask=None):
  encoded,prosody=original(phones,phonetic,context,lang,mask)
  pause=torch.tensor(features['pause'],device=prosody.device)[None]
  predicted=predict_durations(prosody,pause)[0].detach().cpu().numpy();desired=predicted.copy()
  for code_id in sorted({u['code_id'] for u in units}):
   group=[u for u in units if u['code_id']==code_id]
   estimates=[int(predicted[u['start']:u['end']].sum()) for u in group if u['symbol']!='W']
   target=int(np.clip(round(np.median(estimates)),minimum_character_ms/10,28))
   for u in group:
    start,end=u['start'],u['end'];old=predicted[start:end].copy();budget=target*(3 if u['symbol']=='W' else 1)
    core=np.array([any(c in VOWELS for c in meta['ipa'][i].removeprefix('zh_letter:')) for i in range(start,end)])
    if not core.any():core[:]=True
    consonants=int(old[~core].sum());budget=max(budget,consonants+int(core.sum()))
    scaled=old.copy();weights=old[core]/max(old[core].sum(),1);floats=weights*(budget-consonants)
    values=np.maximum(1,np.floor(floats).astype(int))
    while values.sum()<budget-consonants:values[np.argmax(floats-values)]+=1
    while values.sum()>budget-consonants:
     j=int(np.argmax(np.where(values>1,values-floats,-np.inf)));assert values[j]>1;values[j]-=1
    scaled[core]=values;desired[start:end]=scaled
    evidence.append(dict(symbol=u['symbol'],code=u['code'],before_ms=int(old.sum())*10,after_ms=int(scaled.sum())*10,phone_range=[start,end]))
   for left,right in zip(group,group[1:]):
    gap=list(range(left['end'],right['start']));assert gap
    desired[gap]=0;desired[gap[0]]=int(gap_ms/10)
  adjusted=prosody.clone()
  for i,(before,after) in enumerate(zip(predicted,desired)):
   if before==after:continue
   if features['pause'][i]:
    adjusted[0,i,1]=float(np.log(max(int(after),1)));adjusted[0,i,2]=1. if after else -1.
   else:adjusted[0,i,0]=float(np.log(int(after)))
  assert np.array_equal(predict_durations(adjusted,pause)[0].cpu().numpy(),desired)
  return encoded,adjusted
 predictor.model.encode=types.MethodType(controlled,predictor.model)
 return evidence
