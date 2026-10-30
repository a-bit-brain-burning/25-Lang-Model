#!/usr/bin/env python3
"""New independent cache; original corpus stays read-only. Run on Linux data host."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from new25_acoustic.contracts import LANGUAGES,SCHEMA,safe_path,sha256,json_write,load_arrays,validate_arrays,frontend
from new25_acoustic.data import read_pcm

class Moments:
    def __init__(self,d):self.n=np.zeros(d);self.s=np.zeros(d);self.q=np.zeros(d)
    def add(self,x,weight=None):
        x=np.asarray(x,dtype='float64');w=np.ones_like(x) if weight is None else weight
        self.n+=w.sum(0);self.s+=(x*w).sum(0);self.q+=(x*x*w).sum(0)
    def value(self,floor=.02):
        m=self.s/np.maximum(self.n,1)
        return m,np.sqrt(np.maximum(self.q/np.maximum(self.n,1)-m*m,floor*floor))

def check_split_identities(input_rows):
    seen={}
    for l,r in input_rows:
        for key in ('id','text_key_sha256','wav_sha256','source_group'):
            if not r.get(key):raise ValueError('Missing split identity '+key)
            scope=l if key in ('source_group','text_key_sha256') else '*'
            k=(key,scope,r[key]);old=seen.get(k)
            if old and old!=r['split']:raise ValueError(f'Cross-split leakage {key}: {l}/{r["id"]}')
            if old and key in ('id','wav_sha256'):raise ValueError(f'Duplicate {key}: {l}/{r["id"]}')
            seen[k]=r['split']

def reserve_test(input_rows,seconds):
    """Keep connected source groups/text identities together before any fit."""
    if not np.isfinite(seconds) or seconds<0:raise ValueError('Invalid heldout duration')
    heldout={}
    if not seconds:return heldout
    for l in LANGUAGES:
        train=[r for lang,r in input_rows if lang==l and r['split']=='train']
        parent={r['source_group']:r['source_group'] for r in train}
        def find(g):
            while parent[g]!=g:
                parent[g]=parent[parent[g]];g=parent[g]
            return g
        texts={}
        for r in train:
            group=r['source_group'];text=r['text_key_sha256']
            if text in texts:
                a,b=find(group),find(texts[text]);parent[max(a,b)]=min(a,b)
            else:texts[text]=group
        components={}
        for r in train:
            group=find(r['source_group'])
            c=components.setdefault(group,{'seconds':0.,'groups':set()})
            c['seconds']+=r['frames']/100;c['groups'].add(r['source_group'])
        total=sum(c['seconds'] for c in components.values());chosen=set();duration=0.
        for group in sorted(components,key=lambda s:hashlib.sha256(('newtts_test_v1/'+l+'/'+s).encode()).hexdigest()):
            c=components[group]
            if duration+c['seconds']>.10*total:continue
            chosen.update(c['groups']);duration+=c['seconds']
            if duration>=seconds:break
        if duration<seconds or len(chosen)==len(parent):
            raise ValueError('Cannot reserve requested connected-group test while retaining >=90% train: '+l)
        for r in train:
            if r['source_group'] in chosen:r['original_split']='train';r['split']='test'
        heldout[l]={'seconds':duration,'source_groups':sorted(chosen),
                    'selection':'metadata-only SHA ordering; source groups connected by shared text kept together; <=10%'}
    check_split_identities(input_rows)
    return heldout

def prepare(root,out,reserve_test_seconds=0):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.is_relative_to(root):raise ValueError('Cache cannot be inside original corpus')
    out.mkdir(parents=True,exist_ok=False);(out/'records').mkdir()
    started=time.time();json_write(out/'IN_PROGRESS.json',{'started':started,'schema':SCHEMA})
    rows=[];vocab={'<pad>':0,'<unk>':1};mom={};identities={};seconds={}
    input_rows=[]
    # Freeze split identities before opening waveform payloads; no validation fit.
    for l in LANGUAGES:
        mf=root/'languages'/l/'manifest.jsonl';identities[str(mf.relative_to(root))]=sha256(mf)
        with mf.open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['language']!=l or r['split'] not in ('train','validation'):raise ValueError('Split/language')
                input_rows.append((l,r))
    check_split_identities(input_rows)
    heldout=reserve_test(input_rows,reserve_test_seconds)
    if heldout:json_write(out/'TEST_SELECTION.json',heldout)
    # First pass over actual train labels builds vocabulary; original integer IDs
    # may have incompatible language-specific vocabularies, so never reuse them.
    for l,r in input_rows:
        if r['split']=='train':
            a=load_arrays(root/'languages'/l,r,verify=True)
            for phone in a['ipa'].astype(str):
                if phone not in vocab:vocab[phone]=len(vocab)
    if len(vocab)>1024:raise ValueError('Vocabulary exceeds documented 1024-token memory bound')
    unknown={'validation':0,'test':0}
    with (out/'manifest.jsonl').open('x') as mf, (out/'sealed_test.jsonl').open('x') as tf:
        for count,(l,r) in enumerate(input_rows,1):
            if shutil.disk_usage(out).free<4*1024**3:raise ValueError('4 GiB preparation free-space floor')
            directory=root/'languages'/l
            a=load_arrays(directory,r,verify=True)
            c,duration=validate_arrays(a,r);ph,ct,pause=frontend(a)
            wav=safe_path(directory,r['wav']);tg=safe_path(directory,r['textgrid'])
            if sha256(wav)!=r['wav_sha256'] or sha256(tg)!=r['textgrid_sha256']:
                raise ValueError('WAV/MFA SHA mismatch '+r['id'])
            pcm=read_pcm(wav,len(c))
            phones=np.array([vocab.get(p,1) for p in a['ipa'].astype(str)],'int64')
            if r['split']!='train':unknown[r['split']]+=int((phones==1).sum())
            if l not in mom:mom[l]=[Moments(c.shape[1]),Moments(ph.shape[1]),Moments(ct.shape[1])]
            if r['split']=='train':
                w=np.ones_like(c);w[:,0]=a['frame_f0_valid'];w[:,-1]=a['frame_voice_weight']
                mom[l][0].add(c,w);mom[l][1].add(ph);mom[l][2].add(ct)
                seconds[l]=seconds.get(l,0)+len(c)/100
            frame_phone=np.repeat(np.arange(len(duration)),duration)
            pvalid=np.asarray(a.get('valid_mask',np.ones(len(duration))),bool)
            # Measured phone-level F0/energy; valid pitch has an explicit mask.
            pt=np.zeros((len(duration),2),np.float32);pw=np.zeros_like(pt)
            for p in range(len(duration)):
                mask=frame_phone==p;voiced=mask&a['frame_f0_valid'].astype(bool)
                if voiced.any():pt[p,0]=c[voiced,0].mean();pw[p,0]=1
                if mask.any():pt[p,1]=c[mask,1].mean();pw[p,1]=1
            cache=f'records/{l}_{count:07d}.npz'
            np.savez(out/cache,phones=phones,phonetic=ph,context=ct,pause=pause,
                     durations=duration,curve=c,f0_valid=a['frame_f0_valid'].astype(bool),
                     voice_weight=a['frame_voice_weight'].astype('float32'),
                     silence=a['confirmed_silence'].astype(bool),phone_valid=pvalid,
                     phone_prosody=pt,phone_prosody_weight=pw)
            row={k:r[k] for k in ('id','language','split','text','frames','phones','source_group','wav_sha256')}
            row.update(cache=cache,cache_sha256=sha256(out/cache),wav=str(wav.relative_to(root)),
                       dimension=c.shape[1],phonetic_dim=ph.shape[1],context_dim=ct.shape[1],
                       record_sha256=r['sha256'],strict_sha256=r['strict_overlay_sha256'],
                       frontend_signature=r.get('frontend_signature'),textgrid_sha256=r['textgrid_sha256'])
            if r['split']=='test':tf.write(json.dumps(row,ensure_ascii=False)+'\n')
            else:mf.write(json.dumps(row,ensure_ascii=False)+'\n');rows.append(row)
            if count%500==0:json_write(out/'progress.json',{'records':count,'total':len(input_rows),'elapsed':time.time()-started})
    norm={}
    for l,ms in mom.items():
        mean,std=ms[0].value();mean[-1]=0;std[-1]=1
        pm,ps=ms[1].value(.1);cm,cs=ms[2].value(.1)
        norm[l]={'mean':mean.tolist(),'std':std.tolist(),'phonetic_mean':pm.tolist(),
                 'phonetic_std':ps.tolist(),'context_mean':cm.tolist(),'context_std':cs.tolist()}
    json_write(out/'normalization.json',norm);json_write(out/'vocabulary.json',vocab)
    json_write(out/'SOURCE_MANIFESTS.json',identities)
    # Deterministic dev set: multiple complete utterances/language, never test.
    dev={}
    for l in LANGUAGES:
        selected=[];groups=set();texts=set()
        for i in sorted([i for i,r in enumerate(rows) if r['language']==l and r['split']=='validation'],key=lambda i:rows[i]['id']):
            r=rows[i]
            if r['source_group'] in groups or r['text'] in texts:continue
            selected.append(i);groups.add(r['source_group']);texts.add(r['text'])
            if len(selected)==3:break
        dev[l]=selected
    if any(len(v)<3 for v in dev.values()):raise ValueError('Need three full dev utterances/language')
    json_write(out/'development.json',dev)
    metadata=['manifest.jsonl','normalization.json','vocabulary.json','SOURCE_MANIFESTS.json','development.json','sealed_test.jsonl']
    if heldout:metadata.append('TEST_SELECTION.json')
    json_write(out/'COMPLETE.json',{'status':'PREPARED','schema':SCHEMA,'records':len(rows),
        'train_seconds_by_language':seconds,'vocabulary_size':len(vocab),'unknown_phones_by_non_train_split':unknown,
        'all_payloads_sha_verified':True,'heldout_for_this_run_available':bool(heldout),'heldout_seconds_by_language':{l:v['seconds'] for l,v in heldout.items()},
        'historically_unseen_test_claim':False,
        'metadata_sha256':{n:sha256(out/n) for n in metadata},'elapsed_seconds':time.time()-started})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reserve-test-seconds',type=float,default=300)
    a=p.parse_args()
    try:prepare(a.corpus,a.out,a.reserve_test_seconds)
    except Exception as e:
        if Path(a.out).exists():json_write(Path(a.out)/'FAILED.json',{'error':repr(e),'complete':False})
        raise
