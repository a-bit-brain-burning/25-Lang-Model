"""Matched continuation / numeric prosody-extension training, isolated artifacts.

The shared acoustic implementation is unchanged. Chinese interpretation is in
ChineseProsodyFrontend; this experiment supervises actual numeric trajectories.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse,hashlib,json,math,random,shutil,subprocess,sys,time
from collections import OrderedDict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..common import PROJECT,save_json,sha256
from ..models.universal_acoustic import UniversalAcousticModel,load_acoustic,FORMAT,resource_estimate
from . import chinese_train as base
OUT=PROJECT/'outputs/chinese_prosody_20261015'
TEXT_KEYS=base.TEXT_INPUT_KEYS


def encode_for_model(frontend,text,model,pinyin=None):
    """Text inference adapter; optional pinyin is for corpus diagnostics only."""
    f=frontend.encode(text,pinyin=pinyin);f=dict(f)
    context=np.asarray(f['context_features'],np.float32)
    dim=model.config['context_dim']
    if dim==34:
        extra=np.asarray(f['prosody_features'],np.float32)
        if context.shape[-1]!=22 or extra.shape!=(len(context),12):raise ValueError('Expected22+12 contract')
        f['context_features']=np.concatenate((context,extra),axis=-1)
    elif dim!=22:raise ValueError('Unsupported model numeric context size')
    f['metadata']={**f.get('metadata',{}),'model_context_dim':dim,'prosody_input_contract':'context22_plus_prosody12_v1' if dim==34 else 'retained_context22','publisher_pinyin_input':pinyin is not None}
    return f


def prepare_cache(config):
    """Train/validation only, never edit paired source NPZ or load test audio."""
    from ..frontend.chinese_prosody import ChineseProsodyFrontend,PROSODY_FEATURE_NAMES
    root=Path(config['data_dir']);cache=Path(config['feature_cache']);cache.mkdir(parents=True,exist_ok=True)
    source=PROJECT/'src/frontend/chinese_prosody.py';source_sha=sha256(source)
    ready=cache/'completed.json'
    if ready.exists():
        value=json.loads(ready.read_text())
        if value['frontend_sha256']==source_sha and value['manifest_sha256']==sha256(root/'manifest.jsonl'):return value
        raise ValueError('Feature cache identity mismatch: use a new cache directory')
    front=ChineseProsodyFrontend();records={split:base.NumericRecords(root,split) for split in ('train','validation')};counts={};started=time.monotonic();examples=[]
    for split,view in records.items():
        counts[split]=len(view)
        for index,(row,a) in enumerate(view):
            f=front.encode(row['text'],pinyin=row['publisher_pinyin'])
            for key in ('phonetic_features','context_features','edge_sil_mask'):
                if not np.array_equal(f[key],a[key]):raise ValueError(f'Frozen input drift {row["id"]} {key}')
            x=np.asarray(f['prosody_features'],np.float32)
            if x.shape!=(len(a['phonetic_features']),12) or not np.isfinite(x).all():raise ValueError('Invalid12D feature contract')
            np.save(cache/(row['id']+'.npy'),x,allow_pickle=False)
            if index<3:examples.append({'id':row['id'],'split':split,'shape':list(x.shape),'active_values':int(np.count_nonzero(x))})
            if index%500==0:print(json.dumps({'preparing':split,'index':index,'total':len(view)}),flush=True)
    assert sha256(source)==source_sha,'Frontend changed while building cache: rebuild before publication'
    value={'status':'complete','frontend_sha256':source_sha,'manifest_sha256':sha256(root/'manifest.jsonl'),'counts':counts,'feature_names':list(PROSODY_FEATURE_NAMES),'feature_dim':12,'base_input_byte_exact':True,'test_loaded':False,'publisher_pinyin_source':'manifest','examples':examples,'wall_seconds':time.monotonic()-started}
    save_json(ready,value);return value


class ProsodyRecords(base.NumericRecords):
    def __init__(self,root,split,cache):
        if split not in ('train','validation'):raise ValueError('No test in training view')
        super().__init__(root,split);self.feature_cache=Path(cache)
    def __getitem__(self,index):
        row,a=super().__getitem__(index)
        if 'prosody_features' not in a:a['prosody_features']=np.load(self.feature_cache/(row['id']+'.npy'),allow_pickle=False)
        return row,a


def batch(records,indices,extension):
    f,t,m,d=base.batch_records(records,indices);chosen=[records[int(i)][1] for i in indices]
    if extension:
        context=torch.zeros(*f['context_features'].shape[:-1],34);context[...,:22]=f['context_features']
        for i,a in enumerate(chosen):context[i,:len(a['prosody_features']),22:]=torch.from_numpy(a['prosody_features'])
        f['context_features']=context
    target_log=np.concatenate([a['target_curve'][:,0] for a in chosen]).astype(np.float32)
    f['_target_frame_logf0']=torch.from_numpy(target_log)
    return f,t,m,d


def trajectory_objectives(out,features):
    """Fit reference frame slopes and voiced inter-segment deltas, not zero."""
    idx=features['_frame_indices'];frac=features['_frame_fractions'];width=features['valid_mask'].shape[1]
    position=frac*7;left=position.long().clamp(0,6);alpha=position-left
    controls=out['controls'].reshape(-1,8,31)[idx,:,0];r=torch.arange(len(idx))
    predicted=controls[r,left]*(1-alpha)+controls[r,left+1]*alpha
    target=features['_target_frame_logf0'];voice=features['_frame_voice']>.5
    phonetic=features['phonetic_features'].reshape(-1,32);context=features['context_features'].reshape(-1,features['context_features'].shape[-1])
    numeric_sonorant=(phonetic[:,17]>.5)&(phonetic[:,[1,5,6]].sum(-1)>.5)
    safe=voice[:-1]&voice[1:]&numeric_sonorant[idx[:-1]]&numeric_sonorant[idx[1:]]&(idx[:-1]//width==idx[1:]//width)
    within=safe&(idx[:-1]==idx[1:])
    boundaries=safe&(idx[:-1]!=idx[1:])&(context[idx[:-1],21]>.5)&(context[idx[1:],20]>.5)&(context[idx[:-1],8]==0)&(context[idx[1:],9]==0)
    # A real positive-duration silence contains frames and fails numeric_sonorant;
    # omitted zero-duration units cannot create a source-time pause.
    scale=100*np.log(2)/1200
    error=((predicted[1:]-predicted[:-1])-(target[1:]-target[:-1]))/scale
    huber=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none',beta=1.)
    mean=lambda mask:(huber*mask).sum()/mask.sum().clamp_min(1)
    return {'reference_f0_slope_huber_100c':mean(within),'reference_boundary_delta_huber_100c':mean(boundaries)}, {'slope_pairs':int(within.sum()),'boundary_pairs':int(boundaries.sum())}


def initialize(checkpoint,extension):
    old=load_acoustic(checkpoint);config={**old.config,'context_dim':34 if extension else 22,'prosody_contract':'context22_plus_prosody12_v1' if extension else 'retained_context22'}
    model=UniversalAcousticModel(config,old.means.numpy(),old.scales.numpy());source=old.state_dict();dest=model.state_dict()
    for key in dest:
        if key=='input.weight' and extension:
            dest[key].zero_();dest[key][:,:54]=source[key]
        else:dest[key]=source[key].clone()
    model.load_state_dict(dest);model.eval();return model


@torch.inference_mode()
def evaluate(model,records,extension,decoder,config):
    model.eval();sums={};duration_error=phones=0.;pairs={'slope_pairs':0,'boundary_pairs':0}
    for start in range(0,len(records),8):
        indices=list(range(start,min(start+8,len(records))));f,t,m,d=batch(records,indices,extension);out=model({k:f[k] for k in TEXT_KEYS})
        original,terms=base.objective(out,t,m,d,f,model.scales,decoder,10000000,model.voiced_log_prior,'frame');aux,counts=trajectory_objectives(out,f);values={**terms,**aux};values['combined_objective']=original+config['slope_weight']*aux['reference_f0_slope_huber_100c']+config['boundary_weight']*aux['reference_boundary_delta_huber_100c']
        for k,v in values.items():sums[k]=sums.get(k,0.)+float(v)*len(indices)
        valid=f['valid_mask'];duration_error+=float(((out['duration_frames']-d).abs()*valid).sum());phones+=int(valid.sum())
        for k,v in counts.items():pairs[k]+=v
    return {**{k:v/len(records) for k,v in sums.items()},'duration_mae_ms':10*duration_error/phones,'recordings':len(records),'auxiliary_pairs':pairs,'selection_metric':'total original31D loss; auxiliary reported separately'}


def train(config,arm,resume=None):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);out=Path(config['output'])/'training'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        if not resume:return
        raise ValueError('Completed frozen run cannot be resumed in place: use new output namespace')
    cache_identity=json.loads((Path(config['feature_cache'])/'completed.json').read_text())
    if cache_identity['manifest_sha256']!=sha256(Path(config['data_dir'])/'manifest.jsonl') or cache_identity['frontend_sha256']!=sha256(PROJECT/'src/frontend/chinese_prosody.py') or cache_identity.get('test_loaded') is not False:raise ValueError('Feature cache identity mismatch')
    extension=arm=='prosody';seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    trainset=ProsodyRecords(config['data_dir'],'train',config['feature_cache']);val=ProsodyRecords(config['data_dir'],'validation',config['feature_cache']);model=initialize(config['checkpoint'],extension)
    # Same weights and physical normalization;12 added input columns are zero.
    original=load_acoustic(config['checkpoint']);max_control=0.;max_duration=0.;vuv_flips=0
    for start in range(0,24,8):
        ix=list(range(start,min(start+8,len(val))));f,t,m,d=batch(val,ix,extension);f0,_,_,_=batch(val,ix,False)
        with torch.inference_mode():a=original({k:f0[k] for k in TEXT_KEYS});b=model({k:f[k] for k in TEXT_KEYS})
        max_control=max(max_control,float((a['controls']-b['controls']).abs().max()));max_duration=max(max_duration,float((a['duration_frames']-b['duration_frames']).abs().max()));vuv_flips+=int(((a['controls'][...,-1]>0)!=(b['controls'][...,-1]>0)).sum())
    assert max_control<1e-4 and max_duration<1e-4 and vuv_flips==0
    save_json(out/'initial_parity.json',{'max_control_absolute_difference':max_control,'max_duration_difference':max_duration,'VUV_gate_flips':vuv_flips,'validation_records':24,'means_scales_retained':True,'new12_input_weights_zero':extension})
    torch.manual_seed(seed+1);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4);decoder=base.PhysicalSpectrum();start_step=0
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['config']['context_dim']!=model.config['context_dim'] or state['lineage']['manifest_sha256']!=sha256(Path(config['data_dir'])/'manifest.jsonl'):raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start_step=state['step'];torch.set_rng_state(state['rng']['torch']);rng.bit_generator.state=state['rng']['sampler'];random.setstate(state['rng']['python'])
    src=[Path(__file__),PROJECT/'src/frontend/chinese_prosody.py',PROJECT/'src/frontend/chinese_training.py',PROJECT/'src/models/universal_acoustic.py',PROJECT/'src/experiments/chinese_train.py'];snap=out/'source_snapshot';snap.mkdir(exist_ok=True);hashes={str(p):sha256(p) for p in src}
    for p in src:shutil.copy2(p,snap/(p.stem+'_'+hashes[str(p)][:12]+p.suffix))
    lineage={'checkpoint':config['checkpoint'],'checkpoint_sha256':sha256(Path(config['checkpoint'])),'manifest_sha256':sha256(Path(config['data_dir'])/'manifest.jsonl'),'source_sha256':hashes,'train_records':len(trainset),'validation_records':len(val),'test_loaded':False,'normalization':'retained source buffers','feature_cache':json.loads((Path(config['feature_cache'])/'completed.json').read_text()),'changed_architecture':'context22→34 only' if extension else 'none','selection':'validation original loss only','fresh_optimizer_both_arms':not bool(resume),'resume':str(resume) if resume else None,'resume_sha256':sha256(Path(resume)) if resume else None,'sampler_digest_scope':f'new updates {start_step+1}..{config["steps"]}'}
    save_json(out/'config.json',{**config,'arm':arm});save_json(out/'lineage.json',lineage);save_json(out/'resources.json',resource_estimate(model));
    if not resume:base.save_checkpoint(out/'initial.pt',model,optimizer,start_step,config,{},lineage,rng)
    started=time.monotonic();digest=hashlib.sha256();sanity=None
    initial_validation=evaluate(model,val,extension,decoder,config);save_json(out/'initial_validation.json',initial_validation);best=initial_validation['total'];beststep=start_step;best_validation=initial_validation
    if (out/'best.pt').exists() and resume:
        previous=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        if previous['lineage']['manifest_sha256']!=lineage['manifest_sha256'] or previous['config']['context_dim']!=model.config['context_dim']:raise ValueError('Existing best checkpoint identity mismatch')
        if previous['validation']['total']<best:best,beststep,best_validation=previous['validation']['total'],previous['step'],previous['validation']
        else:base.save_checkpoint(out/'best.pt',model,optimizer,start_step,config,initial_validation,lineage,rng)
    if not (out/'best.pt').exists() or not resume:base.save_checkpoint(out/'best.pt',model,optimizer,start_step,config,initial_validation,lineage,rng)
    with (out/'training.jsonl').open('a') as log:
        for step in range(start_step+1,config['steps']+1):
            model.train();indices=rng.choice(len(trainset),config['batch_records'],replace=False);digest.update(np.asarray(indices,dtype='<i8').tobytes());f,t,m,d=batch(trainset,indices,extension);prediction=model({k:f[k] for k in TEXT_KEYS});original_loss,terms=base.objective(prediction,t,m,d,f,model.scales,decoder,config['physical_samples'],model.voiced_log_prior,'frame');aux,counts=trajectory_objectives(prediction,f)
            extra=config['slope_weight']*aux['reference_f0_slope_huber_100c']+config['boundary_weight']*aux['reference_boundary_delta_huber_100c'];loss=original_loss+extra if extension else original_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            aux_gradient_norm=float(torch.autograd.grad(extra,model.acoustic[-1].weight,retain_graph=True)[0].norm()) if sanity is None else None
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if sanity is None:
                sanity={'auxiliary_F0_head_gradient_norm':aux_gradient_norm,'auxiliary_pairs':counts,'finite_loss':True,'gradient_norm':float(grad),'input12_gradient_norm':float(model.input.weight.grad[:,54:].norm()) if extension else None,'auxiliary_loss_terms':{k:float(v.detach()) for k,v in aux.items()},'fixed_weights':{'slope':config['slope_weight'],'boundary':config['boundary_weight']},'test_not_used_to_choose_weights':True};save_json(out/'train_gradient_sanity.json',sanity)
            optimizer.step();progress=step/config['steps'];lr=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:group['lr']=lr
            if step==1 or step%100==0:
                row={'step':step,'wall_seconds':time.monotonic()-started,'original_loss':float(original_loss.detach()),'training_objective':float(loss.detach()),'auxiliary':{k:float(v.detach()) for k,v in aux.items()},'learning_rate':lr,'gradient_norm':float(grad)};log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step==1 or step%config['eval_every']==0 or step==config['steps']:
                v=evaluate(model,val,extension,decoder,config);save_json(out/f'validation_step{step:06d}.json',v);base.save_checkpoint(out/'latest.pt',model,optimizer,step,config,v,lineage,rng)
                if v['total']<best:best,beststep,best_validation=v['total'],step,v;base.save_checkpoint(out/'best.pt',model,optimizer,step,config,v,lineage,rng)
                print(json.dumps({'step':step,'validation':v,'best_step':beststep}),flush=True)
    loaded=load_acoustic(out/'best.pt');assert sum(p.numel() for p in loaded.parameters())==(107577 if extension else 106617)
    save_json(out/'completed.json',{'status':'complete','steps':config['steps'],'best_step':beststep,'best_original_validation_loss':best,'best_validation':best_validation,'wall_seconds':time.monotonic()-started,'parameters':resource_estimate(model)['parameters'],'sampler_sha256':digest.hexdigest(),'sampler_digest_scope':f'new updates {start_step+1}..{config["steps"]}','checkpoint_sha256':sha256(out/'best.pt'),'checkpoint_reload':True,'test_loaded':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--arm',choices=['baseline','prosody']);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args();config=json.loads(a.config.read_text());config['portable_config_sha256']=sha256(a.config)
    for key in ('data_dir','checkpoint','output','feature_cache'):config[key]=str((PROJECT.parent/Path(config[key])).resolve())
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if a.arm:train(config,a.arm,a.resume);return
    prepare_cache(config)
    if a.prepare_only:return
    processes=[];logs=[]
    for arm in ['baseline','prosody']:
        folder=out/'training'/arm;folder.mkdir(parents=True,exist_ok=True);log=(folder/'stdout.log').open('w');logs.append(log);processes.append((arm,subprocess.Popen([sys.executable,'-u','-B','-m','project.src.experiments.chinese_prosody_train','--config',str(a.config),'--arm',arm],stdout=log,stderr=subprocess.STDOUT)))
    failures=[]
    for arm,proc in processes:
        if proc.wait():failures.append(arm)
    for log in logs:log.close()
    if failures:save_json(out/'failed.json',{'arms':failures});raise RuntimeError(str(failures))
    frozen={arm:json.loads((out/'training'/arm/'completed.json').read_text()) for arm in ['baseline','prosody']};assert frozen['baseline']['sampler_sha256']==frozen['prosody']['sampler_sha256']
    save_json(out/'selection_ready.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':{arm:{'checkpoint':str(out/'training'/arm/'best.pt'),'sha256':frozen[arm]['checkpoint_sha256'],'completed':frozen[arm]} for arm in frozen},'selection':'validation original loss only; no test','test_loaded':False,'sampler_pair_exact':True});save_json(out/'completed.json',{'status':'complete','arms':frozen,'test_loaded':False})
    print(out,flush=True)
if __name__=='__main__':main()
