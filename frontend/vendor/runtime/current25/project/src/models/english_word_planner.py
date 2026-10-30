"""Small English word BiGRU and train-only automatic plan supervision.

CLI prepare/train/export separates text-only inference from reference-driven
weak labels. The existing frontend/acoustic model/DSP are never modified.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from ..common import PROJECT,save_json,sha256
from ..frontend.english import from_manifest_row,PHONE_TO_ID
from ..frontend.english_word_prosody import (FEATURE_DIM,FEATURE_NAMES,PLAN_NAMES,PLAN_VERSION,VOWELS,
    encode_frontend,encode_text,expanded_frontend,plan_for_units,validate_record)

FORMAT='tiny_english_word_planner_v1'
DEFAULT_DATA=PROJECT/'outputs/representation_upgrade_20261014/data'


class EnglishWordPlanner(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config={'feature_dim':FEATURE_DIM,'width':48,'hidden':32,'layers':2,'dropout':.1,**(config or {})}
        c=self.config
        if c['feature_dim']!=FEATURE_DIM:raise ValueError('Word feature schema differs')
        self.input=nn.Linear(FEATURE_DIM,c['width']);self.norm=nn.LayerNorm(c['width'])
        self.gru=nn.GRU(c['width'],c['hidden'],num_layers=c['layers'],batch_first=True,bidirectional=True,
                        dropout=c['dropout']if c['layers']>1 else 0.)
        self.head=nn.Sequential(nn.Linear(2*c['hidden'],32),nn.GELU(),nn.Linear(32,5))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[3]=-3.

    def forward(self,features):
        numeric=features['numeric'];boundary=features['boundary'];single=numeric.ndim==2
        if single:numeric,boundary=numeric[None],boundary[None]
        if numeric.shape[-1]!=FEATURE_DIM or boundary.shape!=numeric.shape[:2]:raise ValueError('Word input shape mismatch')
        valid=features.get('valid_mask',torch.ones(numeric.shape[:2],dtype=torch.bool,device=numeric.device))
        if valid.ndim==1:valid=valid[None]
        lengths=valid.sum(1)
        if torch.any(lengths<1):raise ValueError('Planner needs at least one word per record')
        if not torch.equal(valid,torch.arange(valid.shape[1],device=valid.device)[None]<lengths[:,None]):raise ValueError('Word masks must be prefix contiguous')
        hidden=F.gelu(self.norm(self.input(numeric)))
        packed=pack_padded_sequence(hidden,lengths.cpu(),batch_first=True,enforce_sorted=False)
        hidden,_=self.gru(packed);hidden,_=pad_packed_sequence(hidden,batch_first=True,total_length=numeric.shape[1])
        raw=self.head(hidden);continuous=3*torch.tanh(raw/3)
        has_next=torch.arange(valid.shape[1],device=valid.device)[None]<lengths[:,None]-1
        plan=torch.cat((continuous[...,:3],raw[...,3:4].sigmoid()*has_next[...,None],continuous[...,4:5]*has_next[...,None]),-1)*valid[...,None]
        result={'plan':plan,'pause_logits':raw[...,3],'valid_mask':valid}
        return {k:v[0]for k,v in result.items()}if single else result

    @torch.inference_mode()
    def predict_text(self,text):
        front,a=encode_text(text);device=next(self.parameters()).device;self.eval()
        f={k:torch.from_numpy(a[k]).to(device)for k in ('numeric','boundary')}
        plan=self(f)['plan'].cpu().numpy().astype(np.float32)
        return {'frontend':front,'word_plan':plan,'unit_plan':plan_for_units(plan,a['word_indices']),
                'word_indices':a['word_indices'],'word_phone_ranges':a['word_phone_ranges'],'words':a['words'],
                'plan_version':PLAN_VERSION,'plan_source':'PREDICTED_TEXT_ONLY',
                'pause_source':'PREDICTED_FROM_TEXT_TRAINED_ON_EXPLICIT_AUTOMATIC_ALIGNMENT_GAPS'}


def load_word_planner(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!=FORMAT or saved.get('plan_version')!=PLAN_VERSION:raise ValueError('Unknown English word planner format')
    model=EnglishWordPlanner(saved['config']);model.load_state_dict(saved['model']);model.eval()
    model.normalization=saved.get('normalization',{});model.checkpoint_step=saved.get('step',0)
    return model


def _key(phone,stress):return str(int(phone))+':'+str(int(stress))


def reference_measurements(front,record):
    """Native time only, with vowel/voicing/energy masks; no closure→pause rule."""
    duration=np.asarray(record['target_duration'],np.int64);curve=np.asarray(record['target_curve'],np.float32)
    if curve.ndim!=2 or curve.shape[1]!=31 or int(duration.sum())!=len(curve):raise ValueError('Reference axis/schema mismatch')
    if np.any(duration<0)or not np.isfinite(curve).all():raise ValueError('Invalid native target')
    a=encode_frontend(front);offsets=np.r_[0,duration.cumsum()];n=len(duration);w=len(front['words'])
    vowel=np.array([p in VOWELS for p in front['phones']])&~np.asarray(record['internal_pause_mask'],bool);vowel_frame=np.repeat(vowel,duration)
    energy_valid=vowel_frame&(curve[:,30]>.5)&(curve[:,1]*10/np.log(10)>=-45)
    f0_valid=energy_valid&(curve[:,0]>=math.log(50))&(curve[:,0]<=math.log(500))
    sentence_pitch=float(np.median(curve[f0_valid,0]))if f0_valid.sum()>=10 else np.nan
    e_sum=np.zeros(n,np.float64);e_count=np.zeros(n,np.int64)
    for i in np.flatnonzero(vowel):
        start,stop=offsets[i:i+2];keep=energy_valid[start:stop];e_count[i]=int(keep.sum());e_sum[i]=curve[start:stop,1][keep].sum(dtype=np.float64)
    word_duration=np.zeros(w,np.float64);word_energy=np.full(w,np.nan);word_pitch=np.full(w,np.nan);pause_frames=np.zeros(w,np.int64);pause_available=np.zeros(w,bool)
    for unit in np.flatnonzero(record['internal_pause_mask']):
        word=int(record['pause_after_word'][unit]);pause_available[word]=True;pause_frames[word]=duration[unit]
    for i,(first,last)in enumerate(a['word_phone_ranges']):
        word_duration[i]=duration[first:last].sum();count=e_count[first:last].sum()
        if count>=3:word_energy[i]=e_sum[first:last].sum()/count
        start,stop=offsets[first],offsets[last];keep=f0_valid[start:stop]
        if keep.sum()>=3 and np.isfinite(sentence_pitch):word_pitch[i]=float(np.median(curve[start:stop,0][keep]))-sentence_pitch
    return {'encoded':a,'phone_ids':np.asarray(front['phone_ids'],np.int64),'stress':np.asarray(front['stress'],np.int64),
            'duration':duration,'word_duration':word_duration,'word_energy':word_energy,'word_pitch_relative':word_pitch,
            'energy_sum':e_sum,'energy_count':e_count,'pause_frames':pause_frames,'pause_available':pause_available}


def fit_phone_baselines(measurements):
    stats=defaultdict(lambda:np.zeros(4,np.float64))
    for m in measurements:
        for phone,stress,duration,energy,count in zip(m['phone_ids'],m['stress'],m['duration'],m['energy_sum'],m['energy_count']):
            if phone<=3:continue
            for key in (_key(phone,stress),'phone:'+str(int(phone)),'global'):
                stats[key]+=np.array([float(duration),float(duration>0),energy,count])if duration>0 else np.array([0,0,energy,count])
    return {k:{'duration_mean_frames':float(v[0]/max(1,v[1])),'duration_support':int(v[1]),
               'energy_mean_log_power':float(v[2]/max(1,v[3])),'energy_frame_support':int(v[3])}for k,v in stats.items()}


def _baseline(baselines,phone,stress,field):
    support='duration_support'if field=='duration_mean_frames'else'energy_frame_support'
    for key in (_key(phone,stress),'phone:'+str(int(phone)),'global'):
        candidate=baselines.get(key)
        if candidate and candidate[support]>=(20 if key!='global'else 1):return candidate[field]
    raise ValueError('Missing train-only phonetic baseline')


def raw_plan(m,baselines):
    result=np.zeros((len(m['word_duration']),5),np.float32);mask=np.zeros_like(result,dtype=bool)
    for i,(first,last)in enumerate(m['encoded']['word_phone_ranges']):
        expected=sum(_baseline(baselines,p,s,'duration_mean_frames')for p,s in zip(m['phone_ids'][first:last],m['stress'][first:last]))
        if m['word_duration'][i]>0 and expected>0:
            result[i,0]=math.log(m['word_duration'][i]/expected);mask[i,0]=True
        if np.isfinite(m['word_energy'][i]):
            counts=m['energy_count'][first:last];predicted=[_baseline(baselines,p,s,'energy_mean_log_power')for p,s in zip(m['phone_ids'][first:last],m['stress'][first:last])]
            result[i,1]=m['word_energy'][i]-np.dot(counts,predicted)/max(1,counts.sum());mask[i,1]=True
        if np.isfinite(m['word_pitch_relative'][i]):result[i,2]=m['word_pitch_relative'][i];mask[i,2]=True
        if m['pause_available'][i]:result[i,3]=float(m['pause_frames'][i]>0);mask[i,3]=True
        if m['pause_frames'][i]>0:result[i,4]=np.log1p(m['pause_frames'][i]);mask[i,4]=True
    return result,mask


def fit_plan_normalization(raw):
    values=np.concatenate([p for p,m in raw]);mask=np.concatenate([m for p,m in raw]);means=[];scales=[];counts=[]
    for j,floor in zip((0,1,2,4),(.1,.25,.1,.25)):
        valid=values[mask[:,j],j]
        if not len(valid):raise ValueError('No valid training labels for '+PLAN_NAMES[j])
        means.append(float(valid.mean()));scales.append(max(floor,float(valid.std())));counts.append(len(valid))
    return {'continuous_indices':[0,1,2,4],'mean':means,'scale':scales,'valid_word_count':counts,'fit_split':'train','clip_z':3.,
            'raw_units':['log(actual_speech_word_frames/sum_train_expected_phone_frames)','log_wave_power_residual','logHz_relative_to_recording_voiced_vowel_median','log1p_positive_explicit_pause_frames']}


def normalized_plan(raw,mask,normalization,boundary):
    plan=np.zeros((len(raw),5),np.float32);indices=np.array(normalization['continuous_indices'])
    z=(raw[:,indices]-np.array(normalization['mean']))/np.array(normalization['scale']);plan[:,indices]=np.where(mask[:,indices],np.clip(z,-3,3),0)
    plan[:,3]=raw[:,3]
    return plan,mask.copy()


def prepare(root,data=None):
    root=Path(root);data=Path(data or root/'data/pause_v2');out=root/'data/plans';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():raise FileExistsError('Completed plan cache is frozen')
    receipt=json.loads((data/'completed.json').read_text())
    if receipt['status'] not in ('COMPLETED','complete'):raise ValueError('Wait for atomically completed pause target cache')
    source=data/'manifest.jsonl';splitpath=DEFAULT_DATA/'split.json';sourcehash=sha256(source);split=json.loads(splitpath.read_text())
    rows=[json.loads(line)for line in source.read_text().splitlines()if line.strip()];selected={s:[r for r in rows if r['split']==s]for s in ('train','validation')}
    for s in selected:
        if set(split[s])!={r['id']for r in selected[s]}:raise ValueError('Frozen split identity differs')
    save_json(out/'status.json',{'status':'RUNNING'});started=time.perf_counter();audit=[]
    def collect(rows):
        measurements=[]
        for i,row in enumerate(rows):
            path=Path(row['record_path']);path=path if path.is_absolute()else data/path
            if sha256(path)!=row['record_sha256']:raise ValueError('Reference SHA mismatch '+row['id'])
            keys=('phone_ids','text_features','stress','target_duration','target_curve','word_indices','internal_pause_mask','pause_after_word','source_indices')
            with np.load(path,allow_pickle=False)as z:record={k:z[k]for k in keys}
            # The source publisher text/lexicon remains the old speech topology.
            legacy=dict(row);legacy.pop('phonemes',None)
            front=expanded_frontend(from_manifest_row(legacy));checks=validate_record(front,record);m=reference_measurements(front,record);measurements.append(m)
            audit.append({'id':row['id'],'split':row['split'],'record_path':str(path),'record_sha256':row['record_sha256'],
                          'identity_checks':checks,'words':len(front['words']),'phones':len(front['phones']),'OOV_words':front['oov_words']})
            if (i+1)%1000==0:print(json.dumps({'stage':'reference_plan_measurements','split':row['split'],'records':i+1}),flush=True)
        return measurements
    train=collect(selected['train']);baselines=fit_phone_baselines(train);train_raw=[raw_plan(m,baselines)for m in train]
    normalization=fit_plan_normalization(train_raw);save_json(out/'normalization.json',{'word':normalization,'phone_baselines':baselines,
        'fit_split':'train','records':len(train),'test_targets_loaded':False})
    normalization_sha=sha256(out/'normalization.json');validation=collect(selected['validation'])
    aggregate={}
    for name,measurements in [('train',train),('validation',validation)]:
        numeric=[];boundary=[];plans=[];masks=[];offsets=[0];ids=[];metadata=[]
        for row,m in zip(selected[name],measurements):
            raw,valid=raw_plan(m,baselines);a=m['encoded'];plan,mask=normalized_plan(raw,valid,normalization,a['boundary']);uid=row['id'];ids.append(uid)
            folder=out/'reference';folder.mkdir(exist_ok=True);unit=plan_for_units(plan,a['word_indices']);unitmask=plan_for_units(mask.astype(np.float32),a['word_indices'])
            np.savez_compressed(folder/(uid+'.npz'),numeric=a['numeric'],boundary=a['boundary'],word_plan=plan,word_mask=mask,
                                raw_plan=raw,word_indices=a['word_indices'],word_phone_ranges=a['word_phone_ranges'],words=np.asarray(a['words']))
            units=out/'reference_units';units.mkdir(exist_ok=True);np.save(units/(uid+'.npy'),unit)
            masksdir=out/'reference_unit_masks';masksdir.mkdir(exist_ok=True);np.save(masksdir/(uid+'.npy'),unitmask)
            numeric.append(a['numeric']);boundary.append(a['boundary']);plans.append(plan);masks.append(mask);offsets.append(offsets[-1]+len(plan))
            metadata.append({'id':uid,'words':a['words'],'source_record_sha256':row['record_sha256'],'source':'REFERENCE_AUTOMATIC_WEAK_PLAN',
                             'pause_source':'SOURCE_ALIGNER_EXPLICIT_GAP_NOT_LOW_ENERGY_GUESS','normalization_sha256':normalization_sha})
        target=out/(name+'.npz');temporary=out/(name+'.tmp.npz')
        np.savez_compressed(temporary,numeric=np.concatenate(numeric),boundary=np.concatenate(boundary),plan=np.concatenate(plans),
                            mask=np.concatenate(masks),offsets=np.array(offsets,np.int64));temporary.replace(target)
        save_json(out/(name+'_records.json'),metadata);aggregate[name]={'records':len(ids),'words':offsets[-1],
            'valid_labels':np.concatenate(masks).sum(0).tolist(),'mask_coverage':np.concatenate(masks).mean(0).tolist(),'cache_sha256':sha256(target)}
    if sha256(source)!=sourcehash:raise ValueError('Manifest changed during prepare')
    save_json(out/'identity_audit.json',{'status':'COMPLETED','records':audit,'all_exact':True,'test_target_loaded':False})
    code=[Path(__file__),PROJECT/'src/frontend/english_word_prosody.py',PROJECT/'src/frontend/english.py']
    protocol={'status':'COMPLETED','plan_version':PLAN_VERSION,'feature_dim':FEATURE_DIM,'feature_names':FEATURE_NAMES,'plan_names':PLAN_NAMES,
              'manifest':str(source),'manifest_sha256':sourcehash,'split_sha256':sha256(splitpath),'normalization_sha256':normalization_sha,
              'source_sha256':{str(p):sha256(p)for p in code},'counts':aggregate,'wall_seconds':time.perf_counter()-started,
              'field_sources':['REFERENCE_SPEECH_WORD_DURATION_MINUS_TRAIN_PHONE_BASELINE','REFERENCE_VOICED_VOWEL_ENERGY_MINUS_TRAIN_PHONE_BASELINE',
                               'REFERENCE_VOICED_VOWEL_PITCH_MINUS_RECORDING_MEDIAN','SOURCE_ALIGNER_EXPLICIT_WORD_GAP_EXISTS','LOG1P_POSITIVE_SOURCE_ALIGNER_WORD_GAP_FRAMES'],
              'reference_confidence':'automatic unreviewed phone alignment/WORLD VUV; >=3 voiced vowel frames, energy>=-45dBFS; pitch50..500Hz and >=10 sentence frames; no human prosody gold',
              'pause_policy':'source aligner explicit interword gap, every boundary has a zero-or-positive optional SIL slot; no stop closure/low-energy→pause inference; first3 plan stats exclude internal pauses',
              'normalization_fit_split':'train','test_audio_or_targets_loaded':False,'ordinary_inference_reads_reference':False,
              'validation_role':'previously reused validation monitoring, not independent confirmation','acoustic_model_or_DSP_modified':False}
    save_json(out/'completed.json',protocol);save_json(out/'status.json',{'status':'COMPLETED'})
    print(json.dumps({'status':'COMPLETED','counts':aggregate,'seconds':protocol['wall_seconds']}),flush=True)


def load_cache(root,split):
    folder=Path(root)/'data/plans';receipt=json.loads((folder/'completed.json').read_text())
    path=folder/(split+'.npz')
    if receipt['status']!='COMPLETED'or sha256(path)!=receipt['counts'][split]['cache_sha256']:raise ValueError('Plan cache identity mismatch')
    with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in z.files}


def batch(cache,indices,device):
    spans=[cache['offsets'][i:i+2]for i in indices];length=max(int(b-a)for a,b in spans);count=len(spans)
    f={'numeric':torch.zeros(count,length,FEATURE_DIM),'boundary':torch.zeros(count,length),'valid_mask':torch.zeros(count,length,dtype=torch.bool)}
    target=torch.zeros(count,length,5);mask=torch.zeros(count,length,5,dtype=torch.bool)
    for i,(a,b)in enumerate(spans):
        n=int(b-a);f['numeric'][i,:n]=torch.from_numpy(cache['numeric'][a:b]);f['boundary'][i,:n]=torch.from_numpy(cache['boundary'][a:b]);f['valid_mask'][i,:n]=True
        target[i,:n]=torch.from_numpy(cache['plan'][a:b]);mask[i,:n]=torch.from_numpy(cache['mask'][a:b])
    return {k:v.to(device)for k,v in f.items()},target.to(device),mask.to(device)


def objective(output,target,mask):
    error=F.smooth_l1_loss(output['plan'].float(),target.float(),reduction='none')
    error=error.clone();error[...,3]=F.binary_cross_entropy_with_logits(output['pause_logits'].float(),target[...,3].float(),reduction='none')
    denom=mask.sum((0,1)).clamp_min(1)
    fields=(error*mask).sum((0,1))/denom;return fields.mean()


@torch.inference_mode()
def evaluate(model,cache,device):
    model.eval();total=np.zeros(5);absolute=np.zeros(5);count=np.zeros(5);pause_bce=0.;pause_errors=0.;tp=fp=fn=0
    for start in range(0,len(cache['offsets'])-1,32):
        f,target,mask=batch(cache,range(start,min(start+32,len(cache['offsets'])-1)),device)
        output=model(f);difference=(output['plan']-target).float();total+=((difference**2)*mask).sum((0,1)).cpu().numpy();absolute+=(difference.abs()*mask).sum((0,1)).cpu().numpy();count+=mask.sum((0,1)).cpu().numpy()
        pause_bce+=float((F.binary_cross_entropy_with_logits(output['pause_logits'].float(),target[...,3],reduction='none')*mask[...,3]).sum())
        pause_errors+=float((((output['plan'][...,3]>=.5)!=(target[...,3]>.5))*mask[...,3]).sum())
        pv=output['plan'][...,3]>=.5;rv=target[...,3]>.5;mv=mask[...,3]
        tp+=int((pv&rv&mv).sum());fp+=int((pv&~rv&mv).sum());fn+=int((~pv&rv&mv).sum())
    rmse=np.sqrt(total/np.maximum(count,1));mae=absolute/np.maximum(count,1)
    scores=total/np.maximum(count,1);scores[3]=pause_bce/max(1,count[3])
    return {'fields':{name:{'RMSE':float(rmse[i]),'MAE':float(mae[i]),'units':'probability'if i==3 else'z','words':int(count[i])}for i,name in enumerate(PLAN_NAMES)},
            'pause_BCE':float(scores[3]),'pause_classification_error_at_0_5':float(pause_errors/max(1,count[3])),
            'pause_counts_at_0_5':{'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)},
            'selection_mean_MSE_z':float(np.mean(scores)),
            'selection_definition':'equal mean of four continuous normalized MSE and explicit-gap existence BCE; positive pause duration regression only',
            'pause_labels':'original aligner explicit interword gaps, automatic not human gold'}


def train(root,device='cpu',steps=2000,batch_records=32,learning_rate=3e-4,seed=20261016,resume=False):
    root=Path(root);out=root/'planner';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():raise FileExistsError('Completed planner is immutable')
    if (out/'latest.pt').exists()and not resume:raise FileExistsError('Use --resume for existing optimizer/RNG state')
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed);device=torch.device(device)
    if device.type=='cuda'and not torch.cuda.is_available():raise RuntimeError('Requested GPU unavailable; no CPU fallback disguised as GPU')
    training=load_cache(root,'train');validation=load_cache(root,'validation');norm=json.loads((root/'data/plans/normalization.json').read_text())
    model=EnglishWordPlanner().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    pause_prior=float(training['plan'][:,3].sum()/max(1,training['mask'][:,3].sum()))
    with torch.no_grad():model.head[-1].bias[3]=math.log(max(1e-6,pause_prior)/max(1e-6,1-pause_prior))
    config={'steps':steps,'batch_records':batch_records,'learning_rate':learning_rate,'seed':seed,'device':str(device),
            'parameters':sum(p.numel()for p in model.parameters()),'feature_dim':FEATURE_DIM,'plan_version':PLAN_VERSION,'initial_pause_probability_train_prior':pause_prior}
    start=0;best=float('inf');best_step=0
    if resume:
        previous=json.loads((out/'config.json').read_text())
        if any(previous[k]!=config[k]for k in ('batch_records','learning_rate','seed','feature_dim')):raise ValueError('Resume must preserve sampler/optimizer configuration')
        state=torch.load(out/'latest.pt',map_location=device,weights_only=True)
        if state['plan_cache_sha256']!=sha256(root/'data/plans/completed.json'):raise ValueError('Resume plan cache changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];best=state['best_score'];best_step=state['best_step'];rng.bit_generator.state=state['sampler_rng'];torch.set_rng_state(state['torch_rng'].cpu())
        if steps<=start:raise ValueError('Resume budget must exceed saved step')
        if device.type=='cuda'and state.get('cuda_rng')is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    save_json(out/'config.json',config);shutil.copyfile(__file__,out/'source_snapshot.py');started=time.perf_counter()
    def checkpoint(step,metrics):
        nonlocal best,best_step
        improved=metrics['selection_mean_MSE_z']<best
        if improved:best=metrics['selection_mean_MSE_z'];best_step=step
        state={'format':FORMAT,'plan_version':PLAN_VERSION,'config':model.config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'step':step,'best_score':best,'best_step':best_step,'validation':metrics,'normalization':norm,'sampler_rng':rng.bit_generator.state,
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()if device.type=='cuda'else None,
               'plan_cache_sha256':sha256(root/'data/plans/completed.json')}
        temporary=out/'latest.tmp.pt';torch.save(state,temporary);temporary.replace(out/'latest.pt')
        if improved:shutil.copyfile(out/'latest.pt',out/'best.pt')
        save_json(out/f'validation_step{step:06}.json',metrics)
    if start==0:checkpoint(0,evaluate(model,validation,device))
    with (out/'training.jsonl').open('a')as log:
        for step in range(start+1,steps+1):
            model.train();indices=rng.choice(len(training['offsets'])-1,min(batch_records,len(training['offsets'])-1),replace=False)
            f,target,mask=batch(training,indices,device);loss=objective(model(f),target,mask)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite word planner loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            record={'status':'RUNNING','step':step,'loss':float(loss.detach().cpu()),'sample_ids_sha256':__import__('hashlib').sha256(indices.astype('<i8').tobytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
            log.write(json.dumps(record)+'\n')
            if step%100==0:log.flush();save_json(out/'status.json',record);print(json.dumps(record),flush=True)
            if step%250==0 or step==steps:checkpoint(step,evaluate(model,validation,device))
    final={'status':'COMPLETED','steps':steps,'best_step':best_step,'best_validation_mean_MSE_z':best,'parameters':config['parameters'],
           'device':str(device),'record_exposures':steps*min(batch_records,len(training['offsets'])-1),'wall_seconds':time.perf_counter()-started,
           'latest_sha256':sha256(out/'latest.pt'),'best_sha256':sha256(out/'best.pt'),'human_listening':'NOT_RUN','acoustic_model_changed':False}
    save_json(out/'completed.json',final);save_json(out/'status.json',final);print(json.dumps(final),flush=True)


def export(root,checkpoint=None):
    root=Path(root);out=root/'data/plans';checkpoint=Path(checkpoint or root/'planner/best.pt');model=load_word_planner(checkpoint);digest=sha256(checkpoint)
    marker=out/'predicted_completed.json'
    if marker.exists():raise FileExistsError('Completed prediction cache is frozen')
    paths={};torch.set_num_threads(1)
    with torch.inference_mode():
        for split in ('train','validation'):
            rows=json.loads((out/(split+'_records.json')).read_text())
            for row in rows:
                with np.load(out/'reference'/(row['id']+'.npz'),allow_pickle=False)as z:
                    # Only text-derived arrays/mapping; never read reference word_plan/raw/mask.
                    f={k:torch.from_numpy(z[k])for k in ('numeric','boundary')};mapping=z['word_indices']
                plan=model(f)['plan'].numpy().astype(np.float32);unit=plan_for_units(plan,mapping)
                folder=out/'predicted_units';folder.mkdir(exist_ok=True);path=folder/(row['id']+'.npy');np.save(path,unit);paths[row['id']]=sha256(path)
    save_json(marker,{'status':'COMPLETED','checkpoint':str(checkpoint),'checkpoint_sha256':digest,'source':'PREDICTED_TEXT_ONLY',
        'pause_source':'PREDICTED_TEXT_ONLY_TRAINED_ON_EXPLICIT_ALIGNMENT_GAPS','reference_labels_read_in_prediction':False,'output_sha256':paths})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','train','export'));p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-records',type=int,default=32);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--seed',type=int,default=20261016);p.add_argument('--resume',action='store_true');p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.data)
    elif a.action=='train':train(a.root,a.device,a.steps,a.batch_records,a.learning_rate,a.seed,a.resume)
    else:export(a.root,a.checkpoint)


if __name__=='__main__':main()
