"""Bounded sentence/clause inference; Chinese identifiers never change voice."""
from pathlib import Path
import sys,importlib.util,json,hashlib,types,wave,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'neural'))
from loader import load
from identifier_first_digit import install,install_vocoder_energy
from identifier_prosody import apply_pitch,install_connection
from new25_acoustic.inference import predict_durations
import torch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def install_flow(predictor,features,meta):
    """Accepted continuous letter flow; retain explicit user stress controls."""
    units=meta.get('identifier_units',[]);records=[]
    if not units:return records
    original=predictor.model.encode
    def controlled(self,phones,phonetic,context,lang,mask=None):
        encoded,prosody=original(phones,phonetic,context,lang,mask)
        pause=torch.as_tensor(features['pause'],device=prosody.device)[None]
        before=predict_durations(prosody,pause)[0].cpu().numpy();target=before.copy();p=prosody.clone()
        for left,right in zip(units,units[1:]):
            if left['code_id']!=right['code_id']:continue
            # The first digit retains its requested longer preceding boundary.
            if right['symbol'].isdigit() and not left['symbol'].isdigit():continue
            positions=[i for i,c in enumerate(left['code']) if c!='-']
            if '-' in left['code'][positions[left['symbol_index']]+1:positions[right['symbol_index']]]:continue
            gap=list(range(left['end'],right['start']))
            assert gap and all(features['pause'][i] for i in gap)
            old=int(before[gap].sum());target[gap]=0;target[gap[0]]=1
            records.append(dict(code=left['code'],left=left['symbol'],right=right['symbol'],before_ms=old*10,after_ms=10))
        for i,(a,b) in enumerate(zip(before,target)):
            if a!=b:p[0,i,1]=float(np.log(max(1,int(b))));p[0,i,2]=1. if b else -1.
        assert np.array_equal(predict_durations(p,pause)[0].cpu().numpy(),target)
        return encoded,p
    predictor.model.encode=types.MethodType(controlled,predictor.model)
    return records
def join_phrases(pieces,texts,sample_rate):
    """Old accepted quiet-edge joining, adjusted for the actual sample rate.

    Only natural sentence/clause boundaries are joined. No identifier is split
    into generated letter/digit clips. Do not time-stretch or normalize gain.
    """
    pieces=[p.astype(np.float32,copy=True) for p in pieces];records=[]
    def quiet(p,side,threshold):
        p=p if side=='start' else p[::-1];count=0;hop=round(sample_rate*.005)
        while count+hop<=min(len(p),2*sample_rate):
            if float(np.sqrt(np.mean(p[count:count+hop]**2)))>threshold:break
            count+=hop
        return count
    for i in range(len(pieces)-1):
        a,b=pieces[i],pieces[i+1]
        threshold=max(.0003,min(.0015,float(np.sqrt((np.sum(a*a)+np.sum(b*b))/(len(a)+len(b))))*.025))
        left=quiet(a,'end',threshold);right=quiet(b,'start',threshold)
        ending=texts[i].rstrip(' \n”’\"」』）)')[-1:]
        gap=.24 if ending in '。.!?！？；;' else .12 if ending in '，,、：:' else .06
        removable_left=max(0,left-round(.02*sample_rate));removable_right=max(0,right-round(.02*sample_rate))
        remove=min(max(0,left+right-round(gap*sample_rate)),removable_left+removable_right)
        cut_left=min(removable_left,round(remove*removable_left/max(1,removable_left+removable_right)))
        cut_right=min(removable_right,remove-cut_left);cut_left+=min(removable_left-cut_left,remove-cut_left-cut_right)
        if cut_left:pieces[i]=a[:-cut_left]
        if cut_right:pieces[i+1]=b[cut_right:]
        records.append(dict(after_segment=i,ending=ending,quiet_ms_before=1000*(left+right)/sample_rate,
          removed_left_samples=cut_left,removed_right_samples=cut_right,quiet_ms_after=1000*(left+right-cut_left-cut_right)/sample_rate,
          target_max_ms=1000*gap,threshold=threshold,voiced_samples_removed=0,added_silence_samples=0))
    if len(pieces)>1:
        for p in pieces:
            fade=min(round(.003*sample_rate),len(p)//2)
            p[:fade]*=np.linspace(0,1,fade,dtype=np.float32);p[-fade:]*=np.linspace(1,0,fade,dtype=np.float32)
    return np.concatenate(pieces),records
class Engine:
    def __init__(self,identifier_prosody='natural'):
        if identifier_prosody not in ['original','natural']:raise ValueError('Unknown identifier prosody mode')
        self.identifier_prosody=identifier_prosody
        self.acoustic,self.vocoder=load();self.base_encode=self.acoustic.model.encode;self.base_adapter=self.vocoder.adapter.forward
        from new25_postprocess.bwe24 import OpusBWE24
        self.bwe=OpusBWE24(cache=None)
    def acoustic_piece(self,entry,language,flow=True):
        path=ROOT/'inputs'/entry['file'];assert sha(path)==entry['sha256']
        mp=ROOT/'inputs'/entry['metadata_file'];assert sha(mp)==entry['metadata_sha256']
        with np.load(path,allow_pickle=False) as z:features={k:z[k].copy() for k in z.files}
        meta=read(mp);self.acoustic.model.encode=self.base_encode;self.vocoder.adapter.forward=self.base_adapter
        regular,u,c,digit=install(self.acoustic,features,meta,'natural') if language=='zh' else ([],[],[],[])
        continuous=install_flow(self.acoustic,features,meta) if flow and language=='zh' else []
        active=language=='zh' and self.identifier_prosody!='original' and bool(meta.get('identifier_units'))
        connection=install_connection(self.acoustic,features,meta) if active else []
        with torch.inference_mode():result=self.acoustic.predict(language=language,**features)
        pitch=apply_pitch(result,features,meta,self.acoustic.normalization[language]) if active else []
        # Record final durations after the connection override, as well as the
        # earlier emphasis stage's request. Energy controls use actual results.
        for d in digit:
            start=d['phone_range'][0];gap=[];i=start-1
            while i>=0 and features['pause'][i]:gap.append(i);i-=1
            actual=int(result.durations[gap].sum())*10
            if actual!=d['after_gap_ms']:d['emphasis_stage_gap_ms']=d['after_gap_ms'];d['after_gap_ms']=actual
        for record in u:
            unit=next(v for v in meta['identifier_units'] if v['symbol']=='U' and [v['start'],v['end']]==record['phone_range'])
            for direction,key in [(-1,'preceding_gap_ms'),(1,'following_gap_ms')]:
                neighbor=next((v for v in meta['identifier_units'] if v['code_id']==unit['code_id'] and v['symbol_index']==unit['symbol_index']+direction),None)
                if neighbor:
                    a,b=(neighbor['end'],unit['start']) if direction<0 else (unit['end'],neighbor['start'])
                    actual=int(result.durations[a:b].sum())*10
                    if actual!=record[key]:record['previous_stage_'+key]=record[key];record[key]=actual
        return result,dict(frontend=meta,regular=regular,U=u,C=c,first_digit=digit,continuous_code_flow=continuous,
            identifier_prosody_mode=self.identifier_prosody,connected_identifier_gaps=connection,identifier_pitch=pitch)
    def voice(self,result,language,controls):
        self.vocoder.adapter.forward=self.base_adapter
        with torch.inference_mode():
            uc,dc=install_vocoder_energy(self.vocoder,result,controls.get('U',[]),controls.get('first_digit',[]),'natural')
            audio=self.vocoder.synthesize(result.normalized[None],language)[0].numpy()
        assert np.isfinite(audio).all() and len(audio)==len(result.normalized)*160
        return audio,dict(U_mel=uc,first_digit_mel=dc)
    def process(self,row):
        from new25_postprocess.bwe24 import save_wav
        raw=[];output=[];details=[];start=time.monotonic()
        for i,entry in enumerate(row['chunks']):
            result,controls=self.acoustic_piece(entry,row['language'])
            assert len(result.normalized)<=1200,(row['id'],i,'Local duration budget exceeded')
            pcm,mel_control=self.voice(result,row['language'],controls)
            q=np.rint(np.clip(pcm.astype(np.float64),-1,1)*32768).clip(-32768,32767).astype(np.int16)
            final,report=self.bwe.process_pcm16(q,sample_rate=16000)
            raw.append(q.astype(np.float32)/32768);output.append(final.astype(np.float32)/32768)
            details.append(dict(**entry,segment=i,acoustic_frames=len(result.normalized),duration_seconds=len(q)/16000,
              phone_durations=result.durations.tolist(),controls=controls,mel_control=mel_control,bwe=report,
              raw_pcm16_sha256=hashlib.sha256(q.astype('<i2').tobytes()).hexdigest(),
              final_pcm24_sha256=hashlib.sha256(final.astype('<i2').tobytes()).hexdigest(),
              rndvoc_clipped_samples=int(np.count_nonzero(np.abs(pcm)>=1)),
              acoustic_calls=1,English_acoustic_calls=1 if row['language']=='en' else 0))
        audio16,joins16=join_phrases(raw,[c['text'] for c in row['chunks']],16000)
        audio24,joins24=join_phrases(output,[c['text'] for c in row['chunks']],24000)
        q16=np.rint(audio16.astype(np.float64)*32768).clip(-32768,32767).astype(np.int16)
        q24=np.rint(audio24.astype(np.float64)*32768).clip(-32768,32767).astype(np.int16)
        self.acoustic.model.encode=self.base_encode;self.vocoder.adapter.forward=self.base_adapter
        return q16,q24,dict(id=row['id'],text=row['text'],language=row['language'],segments=details,joins16=joins16,joins24=joins24,
          sample_rate=24000,seconds=time.monotonic()-start,acoustic_additional_updates=1000,identifier_prosody_mode=self.identifier_prosody,vocoder='RNDVoC25000',postprocessor='Opus BWE24',
          acoustic_state_reset_per_segment=True,RND_global_statistics_bounded=True,BWE_state_reset_per_segment=True,
          identifiers_split=False,English_acoustic_calls=len(details) if row['language']=='en' else 0,
          identifier_waveform_splicing=False,natural_text_boundary_joins=len(details)-1,training_updates=0,human_quality_accepted=False)
