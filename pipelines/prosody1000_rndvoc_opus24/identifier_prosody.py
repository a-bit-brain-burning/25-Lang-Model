"""Reference-free identifier pitch continuity before the unchanged E2/RND.
Physical log-F0 conditioning only; no PCM pitch effect, stretching or splicing.
"""
import types
import numpy as np
import torch
TONE=np.asarray([[0,0,0],[1,1,1],[-.5,0,1],[-.5,-1,.5],[1,0,-1],[0,0,0]],np.float32)
def install_connection(predictor,features,meta):
 from new25_acoustic.inference import predict_durations
 units=meta.get('identifier_units',[]);records=[];original=predictor.model.encode
 def controlled(self,phones,phonetic,context,lang,mask=None):
  encoded,p=original(phones,phonetic,context,lang,mask);pause=torch.as_tensor(features['pause'],device=p.device)[None];before=predict_durations(p,pause)[0].cpu().numpy();desired=before.copy();p=p.clone()
  for left,right in zip(units,units[1:]):
   if left['code_id']!=right['code_id']:continue
   positions=[i for i,c in enumerate(left['code'])if c!='-']
   if '-'in left['code'][positions[left['symbol_index']]+1:positions[right['symbol_index']]]:continue
   gap=list(range(left['end'],right['start']));assert gap and np.all(features['pause'][gap])
   first_digit=right['symbol'].isdigit()and not left['symbol'].isdigit()
   desired[gap]=0
   if first_digit:desired[gap[0]]=6
   records.append(dict(code=left['code'],left=left['symbol'],right=right['symbol'],before_ms=int(before[gap].sum())*10,after_ms=int(desired[gap].sum())*10))
  for i,(a,b)in enumerate(zip(before,desired)):
   if a!=b:p[0,i,1]=float(np.log(max(1,int(b))));p[0,i,2]=-1 if b==0 else 1
  assert np.array_equal(predict_durations(p,pause)[0].cpu().numpy(),desired)
  return encoded,p
 predictor.model.encode=types.MethodType(controlled,predictor.model);return records

def apply_pitch(result,features,meta,normalization,*,style='soft'):
 units=meta.get('identifier_units',[])
 if not units:return []
 before=result.physical[:,0].copy();after=before.copy();ends=np.r_[0,np.cumsum(result.durations)];records=[]
 for group in sorted({u['code_id']for u in units}):
  members=[u for u in units if u['code_id']==group];begin=int(ends[members[0]['start']]);end=int(ends[members[-1]['end']])
  frame_phone=result.frame_phone;core=features['phonetic'][frame_phone,31]>.5;voiced=result.physical[:,-1]>.55
  selected=before[begin:end][core[begin:end]&voiced[begin:end]]
  if len(selected)<3:selected=before[begin:end][voiced[begin:end]]
  base=float(np.median(selected))if len(selected)else float(normalization['mean'][0]);mu=float(normalization['mean'][0]);sd=float(normalization['std'][0]);base=float(np.clip(base,mu-1.1*sd,mu+1.1*sd))
  last_end=None
  for j,u in enumerate(members):
   a,b=int(ends[u['start']]),int(ends[u['end']]);assert b>a
   level=base+.012-.025*j/max(len(members)-1,1);phase=np.linspace(0,1,b-a,dtype=np.float32)
   if u['symbol'].isalpha():
    target=np.full(b-a,level,np.float32)
    if u['symbol']!='U':target+=.018*np.sin(np.pi*phase)
    tone='level1_custom_yiu'if u['symbol']=='U'else'soft_letter_stress'
   else:
    contour=features['context'][u['end']-1,14:17];matches=[k for k in range(1,6)if np.allclose(contour,TONE[k])];assert len(matches)==1,(u,contour)
    tone=matches[0]
    # Keep the model's local Mandarin shape and compress its register, rather
    # than imposing a new contour on a rhotic/creaky vowel.
    local_core=features['phonetic'][result.frame_phone[a:b],31]>.5
    local_voice=result.physical[a:b,-1]>.55
    local=before[a:b][local_core&local_voice]
    center=float(np.median(local))if len(local)else float(np.median(before[a:b]))
    # Compress between-character register differences while retaining 85%
    # of the original within-syllable rise/fall (Mandarin tone identity).
    target=level+.4*(center-base)+.85*(before[a:b]-center)
   # A bounded change avoids pushing the E2/vocoder to a remote pitch register.
   target=np.clip(target,before[a:b]-.18,before[a:b]+.18)
   if len(target)>=5:target=np.convolve(np.pad(target,(2,2),mode='edge'),np.asarray([1,2,3,2,1])/9,'valid')
   confidence=np.clip((result.physical[a:b,-1]-.3)/.5,0,1)
   after[a:b]=before[a:b]+confidence*(target-before[a:b])
   if last_end is not None and a>last_end:after[last_end:a]=np.linspace(after[last_end-1],after[a],a-last_end+2)[1:-1]
   records.append(dict(code=u['code'],symbol=u['symbol'],first_frame=a,last_frame=b,base_Hz=float(np.exp(base)),before_median_Hz=float(np.exp(np.median(before[a:b]))),after_median_Hz=float(np.exp(np.median(after[a:b]))),tone=tone,max_requested_logF0_change=.18))
   last_end=b
  fade=min(3,(end-begin)//3)
  if fade:
   weight=.5-.5*np.cos(np.pi*np.arange(1,fade+1)/(fade+1));after[begin:begin+fade]=before[begin:begin+fade]*(1-weight)+after[begin:begin+fade]*weight;after[end-fade:end]=after[end-fade:end]*(1-weight)+before[end-fade:end]*weight
 result.physical[:,0]=after;normalized=(after-normalization['mean'][0])/normalization['std'][0];result.normalized[:,0]=normalized;result.prediction[:,0]=normalized
 return records
