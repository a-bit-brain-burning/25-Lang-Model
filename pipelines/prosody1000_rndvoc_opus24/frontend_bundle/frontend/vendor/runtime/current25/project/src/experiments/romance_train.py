"""Recoverable GPU training using the English39 objective and tiny network.

Separate language weights, generic numeric IPA frontend and text-only plan5.
No PCM enhancer; source recordings and already released models are immutable.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_k,'1')
import argparse,copy,json,math,random,resource,time,traceback
from pathlib import Path
import numpy as np
import torch
from ..common import save_json,sha256
from ..frontend.romance_dispatch import RomanceFrontend
from ..models import romance_word_planner as wp
from ..models.universal_acoustic import UniversalAcousticModel,expand_controls
from ..experiments.eu_quality_spectrum_basis import PhysicalSpectrum
from ..training.english39_objective import objective

FORMAT='tiny_romance_acoustic39_v1'
WEIGHTS={'relative_duration':.2,'vowel_midpoint':.15,'vowel_contrast':.1,'consonant_transition':.15,'frame_vuv':.15}


def setup(seed):
    torch.set_num_threads(1);torch.manual_seed(seed);random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


class Bank:
    def __init__(self,root,lexicon,language):
        self.root=Path(root); receipt=json.loads((self.root/'completed.json').read_text())
        manifest=self.root/'manifest.jsonl'
        if receipt['status']!='COMPLETED' or receipt['manifest_sha256']!=sha256(manifest):raise ValueError('Unpublished/changed data')
        overlay=json.loads((self.root/'strict/completed.json').read_text())
        if (overlay['status']!='COMPLETED' or overlay['source_manifest_sha256']!=sha256(manifest)
            or overlay['statistics_sha256']!=sha256(self.root/'strict/statistics.json')):raise ValueError('Strict overlay source changed')
        overlay_rows={r['id']:r for r in json.loads((self.root/'strict/statistics.json').read_text())}
        self.frontend=RomanceFrontend(language,lexicon);self.rows=[json.loads(s) for s in manifest.read_text().splitlines()]
        self.arrays={};self.fronts={};self.measurements={};self.encoded={}
        for r in self.rows:
            if r['split'] not in ('train','validation'):raise ValueError('Training must not read test')
            p=Path(r['record_path'])
            if sha256(p)!=r['sha256']:raise ValueError('Data changed')
            with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            strict=self.root/'strict'/(r['id']+'.npz')
            if sha256(strict)!=overlay_rows[r['id']]['sha256']:raise ValueError('Overlay changed')
            with np.load(strict,allow_pickle=False) as z:a.update({k:z[k] for k in z.files})
            f=self.frontend.encode(r['text'])
            for k in ('phonetic_features','context_features','plan_owner','lexical_word_indices','internal_pause_mask'):
                if not np.array_equal(f[k],a[k]):raise ValueError('Training/inference frontend differs: '+k)
            self.arrays[r['id']]=a;self.fronts[r['id']]=f;self.encoded[r['id']]=wp.encode_frontend(f)
            self.measurements[r['id']]=wp.reference_measurements(f,{**a,'split':r['split']})
        self.train=[r for r in self.rows if r['split']=='train'];self.validation=[r for r in self.rows if r['split']=='validation']
        if not self.train or not self.validation:raise ValueError('Nonempty train/validation required')
        self.baselines=wp.fit_phone_baselines([self.measurements[r['id']] for r in self.train])
        raw={r['id']:wp.raw_plan(self.measurements[r['id']],self.baselines) for r in self.rows}
        self.normalization=wp.fit_plan_normalization([raw[r['id']] for r in self.train])
        self.targets={k:wp.normalized_plan(*v,self.normalization) for k,v in raw.items()}
        self.lineage={'manifest_sha256':sha256(manifest),'strict_receipt_sha256':sha256(self.root/'strict/completed.json'),
                      'lexicon_sha256':sha256(lexicon),'frontend_signature':self.frontend.signature,'language':language,
                      'test_opened':False,'reference_plan_is_deployment_input':False}


def planner_batch(bank,rows,device):
    b,n=len(rows),max(len(bank.encoded[r['id']]['numeric']) for r in rows)
    f={'numeric':torch.zeros(b,n,wp.FEATURE_DIM),'boundary':torch.zeros(b,n),'valid_mask':torch.zeros(b,n,dtype=torch.bool)}
    t=torch.zeros(b,n,5);m=torch.zeros(b,n,5,dtype=torch.bool)
    for i,r in enumerate(rows):
        e=bank.encoded[r['id']];k=len(e['numeric']);f['numeric'][i,:k]=torch.from_numpy(e['numeric']);f['boundary'][i,:k]=torch.from_numpy(e['boundary']);f['valid_mask'][i,:k]=True
        target,mask=bank.targets[r['id']];t[i,:k]=torch.from_numpy(target);m[i,:k]=torch.from_numpy(mask)
    return {k:v.to(device) for k,v in f.items()},t.to(device),m.to(device)


@torch.inference_mode()
def planner_validate(model,bank,device):
    model.eval();total=0.;error=np.zeros(5);count=np.zeros(5);pause_correct=0
    for r in bank.validation:
        f,t,m=planner_batch(bank,[r],device);o=model(f);total+=float(wp.objective(o,t,m))
        err=(o['plan']-t).abs();error+=(err*m).sum((0,1)).cpu().numpy();count+=m.sum((0,1)).cpu().numpy()
        pause_correct+=int((((o['plan'][...,3]>=.5)==(t[...,3]>=.5))*m[...,3]).sum())
    return {'loss':total/len(bank.validation),'field_MAE':(error/np.maximum(count,1)).tolist(),'field_valid_words':count.tolist(),
            'pause_accuracy':float(pause_correct/max(count[3],1)),'recordings':len(bank.validation)}


def atomic_checkpoint(path,payload):
    def portable(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
        if isinstance(value,list):return [portable(v) for v in value]
        if isinstance(value,tuple):return tuple(portable(v) for v in value)
        return value
    path=Path(path);tmp=path.with_suffix('.tmp.pt');torch.save(portable(payload),tmp);tmp.replace(path)


def resume_state(out,config,lineage,resume):
    """Validate before touching run metadata; interruption saves are resumable."""
    out=Path(out);paths=[out/name for name in ('latest.pt','interrupted.pt') if (out/name).exists()]
    if paths and not resume:raise ValueError('Existing unfinished run requires --resume')
    if not resume:return None
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    for state in states:
        if state['lineage']!=lineage or state['training_config']!=config:raise ValueError('Resume lineage/config differs')
    return max(states,key=lambda s:s['step']) if states else None


def train_planner(bank,out,device,steps=3000,batch_size=24,seed=20261025,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);setup(seed)
    config={'steps':steps,'batch':batch_size,'seed':seed,'learning_rate':.001,'weight_decay':.0001,'precision':'FP32','device':str(device)}
    if (out/'completed.json').exists():return wp.load_word_planner(out/'best.pt').to(device)
    state=resume_state(out,config,bank.lineage,resume)
    save_json(out/'config.json',config);save_json(out/'baselines.json',bank.baselines);save_json(out/'normalization.json',bank.normalization)
    model=wp.RomanceWordPlanner().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    sampling=np.stack([rng.choice(len(bank.train),batch_size,replace=False) for _ in range(steps)]);np.save(out/'sampling.npy',sampling)
    start=0;best=float('inf')
    if state is not None:
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];best=state['best_score'];torch.set_rng_state(state['torch_rng'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    started=time.perf_counter();last=start
    def save(name,step,metrics):
        payload=wp.checkpoint_payload(model,bank.normalization,step)
        payload.update(optimizer=optimizer.state_dict(),training_config=config,lineage=bank.lineage,best_score=best,metrics=metrics,
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [])
        atomic_checkpoint(out/name,payload)
    try:
        with (out/'training.jsonl').open('a') as log:
            for step in range(start+1,steps+1):
                rows=[bank.train[i] for i in sampling[step-1]];f,t,m=planner_batch(bank,rows,device)
                model.train();optimizer.zero_grad(set_to_none=True);loss=wp.objective(model(f),t,m)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite planner loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite planner gradient')
                optimizer.step();last=step
                for g in optimizer.param_groups:g['lr']=.001*(.15+.85*.5*(1+math.cos(math.pi*step/steps)))
                if step==1 or step%100==0:
                    status={'status':'RUNNING','step':step,'loss':float(loss),'seconds':time.perf_counter()-started}
                    log.write(json.dumps(status)+'\n');log.flush();save_json(out/'status.json',status);print('planner',json.dumps(status),flush=True)
                if step%250==0 or step==steps:
                    metrics=planner_validate(model,bank,device);improved=metrics['loss']<best;best=min(best,metrics['loss'])
                    save('latest.pt',step,metrics)
                    if improved:save('best.pt',step,metrics)
                    save_json(out/f'validation_{step:06}.json',metrics)
        result={'status':'COMPLETED','steps':steps,'seconds':time.perf_counter()-started,'best_score':best,'parameters':sum(p.numel() for p in model.parameters()),'best_sha256':sha256(out/'best.pt')}
        save_json(out/'completed.json',result);save_json(out/'status.json',result)
        return wp.load_word_planner(out/'best.pt').to(device)
    except BaseException:
        save('interrupted.pt',last,{});save_json(out/'status.json',{'status':'FAILED','step':last,'traceback':traceback.format_exc()});raise


def normalization(bank):
    total=np.zeros(39);sq=np.zeros(39);count=np.zeros(39)
    for r in bank.train:
        a=bank.arrays[r['id']];t=a['target_controls'].astype(np.float64);m=a['control_mask'].astype(np.float64)
        total+=(t*m).sum((0,1));sq+=(t*t*m).sum((0,1));count+=m.sum((0,1))
    means=total/np.maximum(count,1);scales=np.sqrt(np.maximum(sq/np.maximum(count,1)-means**2,0))
    scales=np.maximum(scales,np.r_[.1,.5,np.full(36,.05),1.]);means[-1]=0;scales[-1]=1
    return means.astype(np.float32),scales.astype(np.float32)


def transfer(source,means,scales,language):
    """Keep physical output exactly before phonetic contract change; add4 zero columns.

    Context is newly defined language frontend numeric22+same plan5, so cross-
    language exact audio equivalence is not claimed. Only parameter conversion
    equivalence for matched artificial vectors is tested by smoke().
    """
    state=torch.load(source,map_location='cpu',weights_only=True);c=state['config']
    if (c['phonetic_dim'],c['context_dim'],c['spectral_dim'],c['ap_bands'])!=(32,27,32,5):raise ValueError('Expected frozen native39 English parent')
    config={**c,'phonetic_dim':36,'language':language,'romance_contract':'numeric36_context22_plan5_warped39_v1'}
    config.pop('english_representation_contract',None)
    model=UniversalAcousticModel(config,means,scales);old=copy.deepcopy(state['model'])
    inp=torch.zeros_like(model.input.weight);inp[:,:32]=old['input.weight'][:,:32];inp[:,36:]=old['input.weight'][:,32:];old['input.weight']=inp
    om,oscale=old['means'],old['scales'];ratio=oscale/model.scales
    w=old['acoustic.2.weight'].reshape(8,39,-1);b=old['acoustic.2.bias'].reshape(8,39)
    w=w*ratio[None,:,None];b=(om[None]+b*oscale[None]-model.means[None])/model.scales[None]
    # VUV is an unnormalised logit and does not consume means/scales.
    w[:,-1]=old['acoustic.2.weight'].reshape(8,39,-1)[:,-1];b[:,-1]=old['acoustic.2.bias'].reshape(8,39)[:,-1]
    old['acoustic.2.weight']=w.reshape(8*39,-1);old['acoustic.2.bias']=b.flatten();old['means']=model.means;old['scales']=model.scales
    model.load_state_dict(old);return model


def acoustic_batch(bank,rows,plans,device):
    b,n=len(rows),max(len(bank.arrays[r['id']]['durations']) for r in rows)
    f={'phonetic_features':torch.zeros(b,n,36),'context_features':torch.zeros(b,n,27),'valid_mask':torch.zeros(b,n,dtype=torch.bool),
       'edge_sil_mask':torch.zeros(b,n,dtype=torch.bool),'_internal_pause_mask':torch.zeros(b,n,dtype=torch.bool)}
    t=torch.zeros(b,n,8,39);m=torch.zeros_like(t);d=torch.zeros(b,n)
    groups={k:torch.zeros(b,n,dtype=torch.float32 if k=='duration_weight' else torch.bool) for k in ('vowel','obstruent','lexical','stressed_vowel','final_obstruent','duration_weight')}
    indices=[];fractions=[];voice=[];confidence=[];left=[];right=[]
    for j,r in enumerate(rows):
        uid=r['id'];a=bank.arrays[uid];front=bank.fronts[uid];k=len(a['durations']);counts=a['durations'].astype(np.int64)
        f['phonetic_features'][j,:k]=torch.from_numpy(a['phonetic_features']);f['context_features'][j,:k]=torch.from_numpy(np.column_stack((a['context_features'],plans[uid])))
        f['valid_mask'][j,:k]=True;f['edge_sil_mask'][j,:k]=torch.from_numpy(a['edge_sil_mask']);f['_internal_pause_mask'][j,:k]=torch.from_numpy(a['internal_pause_mask'])
        t[j,:k]=torch.from_numpy(a['target_controls']);m[j,:k]=torch.from_numpy(a['control_mask']);d[j,:k]=torch.from_numpy(counts)
        vowel=front['vowel_mask'];lexical=front['lexical_word_indices']>=0;ph=a['phonetic_features'];obstruent=((ph[:,2:5].sum(1)>.5)&lexical)
        gv={'vowel':vowel,'lexical':lexical,'obstruent':obstruent,'stressed_vowel':vowel&(front['stress']==1),
            'final_obstruent':obstruent&(a['context_features'][:,1]>.5),'duration_weight':np.where(obstruent,1.5,1.).astype(np.float32)}
        for key,v in gv.items():groups[key][j,:k]=torch.from_numpy(v)
        vv=np.flatnonzero(vowel)
        if len(vv)>1:left.extend(vv[:-1]+j*n);right.extend(vv[1:]+j*n)
        indices.append(np.repeat(np.arange(k)+j*n,counts));fractions.extend(np.arange(c,dtype=np.float32)/c for c in counts if c)
        voice.append(a['frame_voice']);confidence.append(a['frame_voice_weight'])
    f.update(_frame_indices=torch.from_numpy(np.concatenate(indices).astype(np.int64)),_frame_fractions=torch.from_numpy(np.concatenate(fractions)),
             _frame_voice=torch.from_numpy(np.concatenate(voice)),_frame_voice_weight=torch.from_numpy(np.concatenate(confidence)),_strict_physical_uniform=False)
    groups.update(pair_left=torch.tensor(left,dtype=torch.long),pair_right=torch.tensor(right,dtype=torch.long))
    return tuple({k:v.to(device) if torch.is_tensor(v) else v for k,v in a.items()} if isinstance(a,dict) else a.to(device) for a in (f,t,m,d,groups))


@torch.inference_mode()
def acoustic_validate(model,bank,plans,device,config):
    model.eval();decoder=PhysicalSpectrum(32,700).to(device);sums={};started=time.perf_counter()
    for first in range(0,len(bank.validation),config['batch']):
        rows=bank.validation[first:first+config['batch']];loss,terms=objective(model,acoustic_batch(bank,rows,plans,device),decoder,config,True)
        for k,v in terms.items():sums[k]=sums.get(k,0.)+float(v)*len(rows)
    return {**{k:v/len(bank.validation) for k,v in sums.items()},'recordings':len(bank.validation),'seconds':time.perf_counter()-started,'axis':'REFERENCE_TIME_PREDICTED_TEXT_PLAN'}


def train_acoustic(bank,planner,source,out,device,steps=8000,batch_size=24,seed=20261025,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);setup(seed)
    means,scales=normalization(bank);model=transfer(source,means,scales,bank.lineage['language']).to(device)
    config={'steps':steps,'batch':batch_size,'seed':seed,'learning_rate':.00015,'weights':WEIGHTS,'physical_samples':768,
            'weight_decay':.0001,'precision':'FP32','device':str(device),'selection':'minimum common validation total_with_aux,1000step cadence',
            'reference_plan_probability':0.,'effective_samples':steps*batch_size,'postfilter':None}
    lineage={**bank.lineage,'parent_sha256':sha256(source),'planner_sha256':sha256(out.parent/'planner/best.pt'),
             'source_sha256':sha256(__file__),'objective_sha256':sha256(Path(__file__).parents[1]/'training/english39_objective.py')}
    if (out/'completed.json').exists():return
    state=resume_state(out,config,lineage,resume)
    save_json(out/'config.json',config);save_json(out/'lineage.json',lineage)
    plans={r['id']:planner.predict_frontend(bank.fronts[r['id']])['unit_plan'] for r in bank.rows}
    planroot=out/'predicted_plans';planroot.mkdir(exist_ok=True)
    for uid,a in plans.items():
        p=planroot/(uid+'.npy')
        if p.exists():
            if not np.array_equal(np.load(p),a):raise ValueError('Frozen predicted text plan changed')
        else:np.save(p,a)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.0001);decoder=PhysicalSpectrum(32,700).to(device)
    rng=np.random.default_rng(seed);sampling=np.stack([rng.choice(len(bank.train),batch_size,replace=False) for _ in range(steps)]);np.save(out/'sampling.npy',sampling)
    start=0;best=float('inf');last=0
    if state is not None:
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];best=state['best_score'];torch.set_rng_state(state['torch_rng'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    last=start
    started=time.perf_counter();data_seconds=compute_seconds=validation_seconds=0.
    def save(name,step,metrics):
        atomic_checkpoint(out/name,{'format':FORMAT,'config':model.config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'step':step,'best_score':best,'metrics':metrics,'training_config':config,'lineage':lineage,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []})
    if not start:
        initial=acoustic_validate(model,bank,plans,device,config);save_json(out/'initial_validation.json',initial);best=initial['total_with_aux'];save('initial.pt',0,initial);save('best.pt',0,initial)
    try:
        with (out/'training.jsonl').open('a') as log:
            for step in range(start+1,steps+1):
                tick=time.perf_counter();rows=[bank.train[i] for i in sampling[step-1]];packed=acoustic_batch(bank,rows,plans,device)
                if str(device).startswith('cuda'):torch.cuda.synchronize()
                data_seconds+=time.perf_counter()-tick;tick=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
                loss,terms=objective(model,packed,decoder,config)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite acoustic loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite acoustic gradient')
                optimizer.step();last=step
                if str(device).startswith('cuda'):torch.cuda.synchronize()
                compute_seconds+=time.perf_counter()-tick
                for g in optimizer.param_groups:g['lr']=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*step/steps)))
                if step==1 or step%100==0:
                    status={'status':'RUNNING','step':step,'loss':float(loss),'seconds':time.perf_counter()-started,'data_seconds':data_seconds,'compute_seconds':compute_seconds}
                    log.write(json.dumps(status)+'\n');log.flush();save_json(out/'status.json',status);print('acoustic',json.dumps(status),flush=True)
                if step%1000==0 or step==steps:
                    tick=time.perf_counter();metrics=acoustic_validate(model,bank,plans,device,config);validation_seconds+=time.perf_counter()-tick
                    improved=metrics['total_with_aux']<best;best=min(best,metrics['total_with_aux']);save('latest.pt',step,metrics)
                    if improved:save('best.pt',step,metrics)
                    save_json(out/f'validation_{step:06}.json',metrics)
        result={'status':'COMPLETED','steps':steps,'seconds':time.perf_counter()-started,'data_seconds':data_seconds,'compute_seconds':compute_seconds,'validation_seconds':validation_seconds,
                'best_score':best,'best_sha256':sha256(out/'best.pt'),'parameters':sum(p.numel() for p in model.parameters()),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,'natural_inference_evaluated':False,'lineage':lineage}
        save_json(out/'completed.json',result);save_json(out/'status.json',result)
    except BaseException:
        save('interrupted.pt',last,{});save_json(out/'status.json',{'status':'FAILED','step':last,'traceback':traceback.format_exc()});raise


def smoke(bank,source,out):
    """True model update, CUDA architecture, CPU/GPU numerical and throughput test."""
    setup(1729);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    planner=wp.RomanceWordPlanner();plans={r['id']:planner.predict_frontend(bank.fronts[r['id']])['unit_plan'] for r in bank.rows}
    means,scales=normalization(bank);base=transfer(source,means,scales,bank.lineage['language']);config={'weights':WEIGHTS,'physical_samples':768}
    rows=bank.train[:min(24,len(bank.train))];records={};outputs={}
    for device in ('cpu','cuda'):
        model=copy.deepcopy(base).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.00015);decoder=PhysicalSpectrum(32,700).to(device)
        packed=acoustic_batch(bank,rows,plans,device);model.eval()
        with torch.no_grad():outputs[device]=model(packed[0])['controls'].cpu().numpy()
        model.train();times=[];compute=[];waiting=[]
        for step in range(15):
            start=time.perf_counter();packed=acoustic_batch(bank,rows,plans,device)
            if device=='cuda':torch.cuda.synchronize()
            compute_start=time.perf_counter();optimizer.zero_grad(set_to_none=True);torch.manual_seed(100+step)
            loss,_=objective(model,packed,decoder,config);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Smoke nonfinite')
            if device=='cuda':torch.cuda.synchronize()
            if step>=5:
                end=time.perf_counter();times.append(end-start);compute.append(end-compute_start);waiting.append(compute_start-start)
        path=out/(device+'_resume.pt');atomic_checkpoint(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict()})
        restored=torch.load(path,map_location=device,weights_only=True);model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
        loss,_=objective(model,packed,decoder,config);loss.backward();optimizer.step()
        records[device]={'batch':len(rows),'step_mean_seconds':float(np.mean(times)),'loss':float(loss),'saved_restored_extra_update':True,
                         'compute_mean_seconds':float(np.mean(compute)),'data_mean_seconds':float(np.mean(waiting)),
                         'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
        del model,optimizer,packed,decoder
        if device=='cuda':torch.cuda.empty_cache()
    maximum=float(np.max(np.abs(outputs['cpu']-outputs['cuda'])))
    if maximum>.002:raise ValueError('CPU/GPU forward differs >.002')
    result={'status':'COMPLETED','torch':str(torch.__version__),'cuda':str(torch.version.cuda),'GPU':torch.cuda.get_device_name(0),'devices':records,
            'CPU_GPU_controls_max_abs':maximum,'parameters':sum(p.numel() for p in base.parameters()),'selected_training_device':'cuda' if records['cuda']['step_mean_seconds']<records['cpu']['step_mean_seconds'] else 'cpu'}
    save_json(out/'smoke.json',result);print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--lexicon',required=True);p.add_argument('--language',default='es');p.add_argument('--parent',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--planner-steps',type=int,default=3000);p.add_argument('--steps',type=int,default=8000);p.add_argument('--batch',type=int,default=24);p.add_argument('--seed',type=int,default=20261025);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    bank=Bank(a.data,a.lexicon,a.language);out=Path(a.output)
    if a.smoke:smoke(bank,a.parent,out/'smoke')
    else:
        planner=train_planner(bank,out/'planner',a.device,a.planner_steps,a.batch,a.seed,a.resume)
        train_acoustic(bank,planner,a.parent,out/'acoustic',a.device,a.steps,a.batch,a.seed,a.resume)
