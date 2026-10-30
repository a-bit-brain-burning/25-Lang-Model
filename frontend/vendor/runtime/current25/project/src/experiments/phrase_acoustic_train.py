"""Isolated numeric plan8 conditioning, matched warmstarted acoustic training."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from ..common import PROJECT,save_json,sha256
from ..frontend.phrase_plan import PLAN_DIM,PLAN_NAMES
from ..models.universal_acoustic import UniversalAcousticModel,load_acoustic,resource_estimate
from . import chinese_train as base
from . import chinese_prosody_train as previous


def initialize(source):
    if source.config['context_dim']!=34:raise ValueError('Expected context34 source')
    model=UniversalAcousticModel({**source.config,'context_dim':42,'phrase_plan_names':list(PLAN_NAMES),
        'phrase_plan_dim':PLAN_DIM},source.means.numpy(),source.scales.numpy())
    old=source.state_dict();state=model.state_dict()
    for name in state:
        if name=='input.weight':
            state[name].zero_();state[name][:,:66]=old[name]
        else:state[name]=old[name].clone()
    model.load_state_dict(state)
    if sum(p.numel() for p in model.parameters())!=563098:raise ValueError('Unexpected acoustic parameter count')
    return model


class PlanRecords(previous.ProsodyRecords):
    def __init__(self,root,split,feature_cache,labels,mode):
        super().__init__(root,split,feature_cache);self.labels=Path(labels);self.mode=mode
    def __getitem__(self,index):
        row,a=super().__getitem__(index)
        if '_phrase_plan8' not in a:
            plan=np.zeros((len(a['phonetic_features']),8),np.float32) if self.mode=='zero' else np.load(self.labels/self.mode/(row['id']+'.npy'),allow_pickle=False)
            if plan.shape!=(len(a['phonetic_features']),8) or plan.dtype!=np.float32 or not np.isfinite(plan).all():raise ValueError('Invalid phrase cache '+row['id'])
            a['_phrase_plan8']=plan
        return row,a


def batch(records,indices):
    features,target,mask,duration=previous.batch(records,indices,True)
    plan=torch.zeros(*features['context_features'].shape[:-1],8)
    spans=[]
    for i,index in enumerate(indices):
        row,a=records[int(index)];plan[i,:len(a['_phrase_plan8'])]=torch.from_numpy(a['_phrase_plan8'])
        spans.append([(s['phone_start'],s['phone_end']) for s in row['frontend_metadata']['syllables']])
    features['context_features']=torch.cat((features['context_features'],plan),-1)
    features['_syllable_spans']=spans
    return features,target,mask,duration


@torch.inference_mode()
def evaluate(model,records,decoder):
    model.eval();sums={};phones=duration_error=f0_square=f0_count=0.;prosody={'predicted_f0_std_cents':[],'reference_f0_std_cents':[],
        'predicted_syllable_duration_std_ms':[],'reference_syllable_duration_std_ms':[]}
    for start in range(0,len(records),8):
        indices=list(range(start,min(start+8,len(records))));f,t,m,d=batch(records,indices);out=model({k:f[k] for k in base.TEXT_INPUT_KEYS})
        _,terms=base.objective(out,t,m,d,f,model.scales,decoder,10000000,model.voiced_log_prior,'frame')
        aux,_=previous.trajectory_objectives(out,f)
        for key,value in {**terms,**aux}.items():sums[key]=sums.get(key,0.)+float(value)*len(indices)
        valid=f['valid_mask'];duration_error+=float(((out['duration_frames']-d).abs()*valid).sum());phones+=int(valid.sum())
        voiced=m[...,0]>0;error=(out['controls'][...,0]-t[...,0])*1200/math.log(2)
        f0_square+=float(error[voiced].square().sum());f0_count+=int(voiced.sum())
        for i,spans in enumerate(f['_syllable_spans']):
            voice=voiced[i]
            if voice.sum()>1:
                prosody['predicted_f0_std_cents'].append(float(out['controls'][i,...,0][voice].std(unbiased=False))*1200/math.log(2))
                prosody['reference_f0_std_cents'].append(float(t[i,...,0][voice].std(unbiased=False))*1200/math.log(2))
            p=torch.stack([out['duration_frames'][i,a:b].sum() for a,b in spans]);r=torch.stack([d[i,a:b].sum() for a,b in spans])
            prosody['predicted_syllable_duration_std_ms'].append(float(p.std(unbiased=False))*10)
            prosody['reference_syllable_duration_std_ms'].append(float(r.std(unbiased=False))*10)
    return {**{k:v/len(records) for k,v in sums.items()},'duration_mae_ms':10*duration_error/phones,
        'f0_control_RMSE_cents':math.sqrt(f0_square/max(f0_count,1)),
        'prosody':{k:float(np.mean(v)) if v else None for k,v in prosody.items()},'recordings':len(records),
        'selection_metric':'common original31D total; no auxiliary selection',
        'plan_mode':records.mode,'reference_plan_is_oracle_diagnostic':records.mode=='reference',
        'reference_phonetic_and_true_timing_used_for_parameter_scoring':True}


def cache_identity(config,arm):
    if config['parameters']+config['planner_parameter_budget']>config['parameter_budget']:
        raise ValueError('Declared acoustic plus planner budget exceeds1M')
    labels=Path(config['labels_dir']);ready=json.loads((labels/'completed.json').read_text())
    if ready['manifest_sha256']!=sha256(Path(config['data_dir'])/'manifest.jsonl') or ready['test_loaded'] is not False or ready['plan_dim']!=8 or tuple(ready['plan_names'])!=PLAN_NAMES:raise ValueError('Plan labels schema/data identity mismatch')
    result={'reference_labels':ready,'reference_ready_sha256':sha256(labels/'completed.json'),
        'normalization_sha256':sha256(labels/'normalization.json'),'planner_parameters':0,'planner_checkpoint':None}
    if result['normalization_sha256']!=ready['normalization_sha256']:raise ValueError('Plan normalization changed')
    feature_identity=json.loads((Path(config['feature_cache'])/'completed.json').read_text())
    if feature_identity['manifest_sha256']!=ready['manifest_sha256'] or feature_identity['test_loaded'] is not False or feature_identity['frontend_sha256']!=sha256(PROJECT/'src/frontend/chinese_prosody.py'):
        raise ValueError('Frozen prosody12 feature cache mismatch')
    result['prosody12_cache_identity']=feature_identity
    if arm=='predicted_condition':
        pred=json.loads((labels/'predicted_completed.json').read_text());result['predicted_cache']=pred
        result['predicted_ready_sha256']=sha256(labels/'predicted_completed.json')
        if pred['manifest_sha256']!=ready['manifest_sha256'] or pred['normalization_sha256']!=ready['normalization_sha256'] or pred.get('test_loaded') is not False or pred.get('pure_text_only') is not True:raise ValueError('Predicted cache is not frozen pure-text plan')
        result['planner_parameters']=int(pred['parameters']);result['planner_checkpoint']=pred['planner_checkpoint']
        if sha256(Path(pred['planner_checkpoint']))!=pred['planner_checkpoint_sha256']:raise ValueError('Planner checkpoint changed')
        from ..models.phrase_prosody import load_planner
        planner=load_planner(pred['planner_checkpoint'])
        if sum(p.numel() for p in planner.parameters())!=result['planner_parameters']:
            raise ValueError('Actual planner parameter count differs from cache metadata')
    if config['parameters']+result['planner_parameters']>config['parameter_budget'] or result['planner_parameters']>config['planner_parameter_budget']:raise ValueError('Total deployment parameter budget exceeded')
    return result


def expected_sampler(seed,steps,records,batch_size,start=0):
    rng=np.random.default_rng(seed);digest=hashlib.sha256()
    for step in range(1,steps+1):
        values=rng.choice(records,batch_size,replace=False)
        if step>start:digest.update(np.asarray(values,dtype='<i8').tobytes())
    return digest.hexdigest()


def train(config,arm,resume=None):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=Path(config['output'])/'training'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        done=json.loads((out/'completed.json').read_text())
        if resume or done['portable_config_sha256']!=config['portable_config_sha256'] or sha256(out/'best.pt')!=done['checkpoint_sha256']:raise ValueError('Frozen completed run differs or cannot resume')
        return
    if (out/'initial.pt').exists() and not resume:raise ValueError('Unfinished run requires explicit resume')
    if sha256(Path(config['checkpoint']))!=config['checkpoint_sha256']:raise ValueError('Source checkpoint changed')
    identity=cache_identity(config,arm);mode={'zero_control':'zero','reference_condition':'reference','predicted_condition':'predicted'}[arm]
    seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    source=load_acoustic(config['checkpoint']);model=initialize(source)
    trainset=PlanRecords(config['data_dir'],'train',config['feature_cache'],config['labels_dir'],mode)
    val=PlanRecords(config['data_dir'],'validation',config['feature_cache'],config['labels_dir'],mode)
    if len(trainset)!=8932 or len(val)!=520:raise ValueError('Frozen split changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4);decoder=base.PhysicalSpectrum();start=0
    files=[Path(__file__),PROJECT/'src/models/universal_acoustic.py',PROJECT/'src/frontend/phrase_plan.py',
        PROJECT/'src/experiments/chinese_train.py',PROJECT/'src/experiments/chinese_prosody_train.py']
    hashes={str(p.relative_to(PROJECT.parent)):sha256(p) for p in files}
    lineage={'source_checkpoint_sha256':config['checkpoint_sha256'],'manifest_sha256':sha256(Path(config['data_dir'])/'manifest.jsonl'),
        'source_sha256':hashes,'plan_identity':identity,'plan_mode':mode,'arm':arm,'test_loaded':False,
        'normalization':'source acoustic train-derived means/scales retained; plan train-derived normalization',
        'train_records':len(trainset),'validation_records':len(val),'source34_to42_new8_input_weights_zero':True,
        'selection':config['selection'],'reference_arm_not_ordinary_text':mode=='reference'}
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['training_config']['portable_config_sha256']!=config['portable_config_sha256'] or state['lineage']['source_sha256']!=hashes or state['lineage']['plan_identity']!=identity or state['lineage']['arm']!=arm:raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];torch.set_rng_state(state['rng']['torch']);rng.bit_generator.state=state['rng']['sampler'];random.setstate(state['rng']['python'])
    else:
        snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
        for p in files:shutil.copy2(p,snap/(p.stem+'_'+sha256(p)[:12]+p.suffix))
        f,_,_,_=batch(val,range(8));sourcef={k:f[k] for k in base.TEXT_INPUT_KEYS};sourcef['context_features']=sourcef['context_features'][...,:34]
        model.eval();source.eval()
        with torch.inference_mode():
            a=source(sourcef);b=model({k:f[k] for k in base.TEXT_INPUT_KEYS})
        parity={k:float((a[k]-b[k]).abs().max()) for k in a}
        if max(parity.values())>5e-5:raise ValueError('Extended source parity failed '+str(parity))
        save_json(out/'initial_parity.json',{'outputs':parity,'new8_weights_zero':True,'plan_mode':mode,'source_physical_output_retained':True})
        torch.manual_seed(seed+1)
    save_json(out/'config.json',{**config,'arm':arm});save_json(out/'lineage.json',lineage);save_json(out/'resources.json',resource_estimate(model))
    initial=evaluate(model,val,decoder);best=initial['total'];best_step=start;best_validation=initial
    if resume:
        previous_best=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        if previous_best['validation']['total']<=best:best,best_step,best_validation=previous_best['validation']['total'],previous_best['step'],previous_best['validation']
    else:
        save_json(out/'initial_validation.json',initial);base.save_checkpoint(out/'initial.pt',model,optimizer,0,config,initial,lineage,rng);base.save_checkpoint(out/'best.pt',model,optimizer,0,config,initial,lineage,rng)
    digest=hashlib.sha256();started=time.monotonic()
    with (out/'training.jsonl').open('a') as log:
        for step in range(start+1,config['steps']+1):
            model.train();ix=rng.choice(len(trainset),config['batch_records'],replace=False);digest.update(np.asarray(ix,dtype='<i8').tobytes());f,t,m,d=batch(trainset,ix)
            prediction=model({k:f[k] for k in base.TEXT_INPUT_KEYS});loss,terms=base.objective(prediction,t,m,d,f,model.scales,decoder,config['physical_samples'],model.voiced_log_prior,'frame')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if step==start+1:save_json(out/'gradient_sanity.json',{'finite_loss':True,'gradient_norm':float(gradient),'plan8_gradient_norm':float(model.input.weight.grad[:,66:].norm()),'zero_mode_gradient_expected_zero':mode=='zero'})
            optimizer.step();lr=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*step/config['steps'])))
            for group in optimizer.param_groups:group['lr']=lr
            if step==1 or step%100==0:
                row={'step':step,'wall_seconds':time.monotonic()-started,'training_loss':float(loss.detach()),'learning_rate':lr,'gradient_norm':float(gradient)};log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step%config['eval_every']==0 or step==config['steps']:
                validation=evaluate(model,val,decoder);save_json(out/f'validation_step{step:06d}.json',validation);base.save_checkpoint(out/'latest.pt',model,optimizer,step,config,validation,lineage,rng)
                if validation['total']<best:best,best_step,best_validation=validation['total'],step,validation;base.save_checkpoint(out/'best.pt',model,optimizer,step,config,validation,lineage,rng)
                save_json(out/'running.json',{'step':step,'best_step':best_step,'best_loss':best,'status':'training'})
    actual_segment=digest.hexdigest()
    if actual_segment!=expected_sampler(seed,config['steps'],len(trainset),config['batch_records'],start):raise ValueError('Actual sampler sequence mismatch')
    loaded=load_acoustic(out/'best.pt')
    if not torch.equal(loaded.means,source.means) or not torch.equal(loaded.scales,source.scales):raise ValueError('Acoustic normalization changed')
    save_json(out/'completed.json',{'status':'complete','steps':config['steps'],'best_step':best_step,'best_original_validation_loss':best,'best_validation':best_validation,
        'parameters':config['parameters'],'planner_parameters':identity['planner_parameters'],'total_parameters':config['parameters']+identity['planner_parameters'],
        'checkpoint_sha256':sha256(out/'best.pt'),'source_checkpoint_sha256':config['checkpoint_sha256'],'portable_config_sha256':config['portable_config_sha256'],
        'sampler_sha256':expected_sampler(seed,config['steps'],len(trainset),config['batch_records']),
        'actual_segment_sampler_sha256':actual_segment,'actual_segment_scope':f'updates {start+1}..{config["steps"]}',
        'sampler_digest_scope':f'updates 1..{config["steps"]}','plan_mode':mode,'test_loaded':False,'wall_seconds':time.monotonic()-started})


def freeze_completed(config,names,path):
    output=Path(config['output']);completed={name:json.loads((output/'training'/name/'completed.json').read_text()) for name in names}
    if len({v['sampler_sha256'] for v in completed.values()})!=1:raise ValueError('Matched sampler differs')
    target=output/path
    if target.exists():
        frozen=json.loads(target.read_text())
        if set(frozen['checkpoints'])!=set(names):raise ValueError('Frozen selection arm set differs')
        for name,value in completed.items():
            saved=frozen['checkpoints'][name]
            if value['portable_config_sha256']!=config['portable_config_sha256'] or saved['sha256']!=value['checkpoint_sha256'] or sha256(Path(saved['checkpoint']))!=saved['sha256']:
                raise ValueError('Frozen selection/config/checkpoint identity mismatch')
        return
    save_json(output/path,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'test_loaded':False,'same_sampler':True,
        'selection':config['selection'],'checkpoints':{name:{'checkpoint':str(output/'training'/name/'best.pt'),
            'sha256':value['checkpoint_sha256'],'completed':value} for name,value in completed.items()},'reference_condition_is_diagnostic_only':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=PROJECT/'configs/phrase_acoustic_20261015.json');p.add_argument('--arm',choices=('zero_control','reference_condition','predicted_condition'));p.add_argument('--resume',type=Path);p.add_argument('--group',choices=('first_two','predicted','freeze_all'),default='first_two');a=p.parse_args()
    config=json.loads(a.config.read_text());config['portable_config_sha256']=sha256(a.config)
    for key in ('checkpoint','data_dir','feature_cache','labels_dir','output'):config[key]=str((PROJECT.parent/config[key]).resolve())
    if a.arm:train(config,a.arm,a.resume);return
    if a.resume:p.error('--resume requires --arm')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    if a.group=='freeze_all':freeze_completed(config,config['arms'],'selection_ready.json');return
    names=['zero_control','reference_condition'] if a.group=='first_two' else ['predicted_condition']
    frozen_path='zero_reference_ready.json' if a.group=='first_two' else 'predicted_ready.json'
    if (output/frozen_path).exists():
        freeze_completed(config,names,frozen_path)
        if all((output/'training'/name/'completed.json').exists() for name in config['arms']):freeze_completed(config,config['arms'],'selection_ready.json')
        print('Completed frozen group verified; files preserved',flush=True);return
    ready=Path(config['labels_dir'])/('completed.json' if a.group=='first_two' else 'predicted_completed.json')
    while not ready.exists():print('Waiting for '+str(ready),flush=True);time.sleep(20)
    jobs=[];logs=[]
    for name in names:
        folder=output/'training'/name;folder.mkdir(parents=True,exist_ok=True);log=(folder/'stdout.log').open('ab',buffering=0);logs.append(log)
        jobs.append((name,subprocess.Popen([sys.executable,'-u','-B','-m','project.src.experiments.phrase_acoustic_train','--config',str(a.config),'--arm',name],stdout=log,stderr=subprocess.STDOUT)))
    save_json(output/('processes_'+a.group+'.json'),{'jobs':{n:process.pid for n,process in jobs},'UTC':datetime.now(timezone.utc).isoformat()})
    failures=[name for name,process in jobs if process.wait()!=0]
    for log in logs:log.close()
    if failures:save_json(output/('failed_'+a.group+'.json'),{'arms':failures});raise RuntimeError(str(failures))
    freeze_completed(config,names,'zero_reference_ready.json' if a.group=='first_two' else 'predicted_ready.json')
    if all((output/'training'/name/'completed.json').exists() for name in config['arms']):freeze_completed(config,config['arms'],'selection_ready.json')


if __name__=='__main__':main()
