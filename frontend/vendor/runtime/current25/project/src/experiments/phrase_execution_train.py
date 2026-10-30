"""Matched utterance-level plan exposure, fixed pretrained planner and acoustic model."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import random,shutil,subprocess,sys,time
import numpy as np
import torch
from ..common import PROJECT,save_json,sha256
from ..models.universal_acoustic import load_acoustic,resource_estimate
from . import phrase_acoustic_train as previous
base=previous.base


def sampling_protocol(seed,steps,records,batch_size):
    batches=np.empty((steps,batch_size),np.int64)
    rng=np.random.default_rng(seed)
    for step in range(steps):batches[step]=rng.choice(records,batch_size,replace=False)
    uniforms=np.random.default_rng(seed+1).random((steps,batch_size))
    return batches,uniforms


def select_plans(predicted,reference,uniforms,probability):
    if not 0<=probability<=1:raise ValueError('Reference probability outside0..1')
    u=np.asarray(uniforms)
    if u.shape!=(len(predicted),) or len(reference)!=len(predicted) or ((u<0)|(u>=1)).any():raise ValueError('Invalid utterance mixing uniforms')
    selected=u<probability;result=[]
    for i,(p,r) in enumerate(zip(predicted,reference)):
        p,r=np.asarray(p,np.float32),np.asarray(r,np.float32)
        if p.shape!=r.shape or p.ndim!=2 or p.shape[1]!=8 or not np.isfinite(p).all() or not np.isfinite(r).all():raise ValueError('Invalid paired native plans')
        result.append(r if selected[i] else p)
    return result,selected


class MixingRecords(previous.previous.ProsodyRecords):
    def __init__(self,config,split):
        if split!='train':raise ValueError('Reference mixing is training-only')
        super().__init__(config['data_dir'],split,config['feature_cache']);self.labels=Path(config['labels_dir'])
    def __getitem__(self,index):
        row,a=super().__getitem__(index)
        if '_mix_predicted' not in a:
            for key,folder in (('_mix_predicted','predicted'),('_mix_reference','reference')):
                plan=np.load(self.labels/folder/(row['id']+'.npy'),allow_pickle=False)
                if plan.shape!=(len(a['phonetic_features']),8) or plan.dtype!=np.float32 or not np.isfinite(plan).all():raise ValueError('Native plan mismatch '+row['id'])
                a[key]=plan
        return row,a


def batch(records,indices,uniforms,probability):
    features,target,mask,duration=previous.previous.batch(records,indices,True)
    chosen=[records[int(i)][1] for i in indices]
    plans,reference_used=select_plans([a['_mix_predicted'] for a in chosen],[a['_mix_reference'] for a in chosen],uniforms,probability)
    packed=torch.zeros(*features['context_features'].shape[:-1],8)
    for i,plan in enumerate(plans):packed[i,:len(plan)]=torch.from_numpy(plan)
    features['context_features']=torch.cat((features['context_features'],packed),-1)
    return features,target,mask,duration,reference_used


def validation_view(config):
    return previous.PlanRecords(config['data_dir'],'validation',config['feature_cache'],config['labels_dir'],'predicted')


def validate(model,records,decoder):
    if records.mode!='predicted':raise ValueError('Validation must always use pure predicted plans')
    return previous.evaluate(model,records,decoder)


def load_source(checkpoint,expected_sha):
    if sha256(Path(checkpoint))!=expected_sha:raise ValueError('Frozen source checkpoint changed')
    model=load_acoustic(checkpoint)
    if model.config['context_dim']!=42 or model.channels!=31 or sum(p.numel() for p in model.parameters())!=563098:raise ValueError('Wrong conditioned acoustic source')
    return model


def prepare_focus_protocol(config,identity):
    """Freeze the same six ordinary-text explicit spans before any training."""
    from ..models.phrase_prosody import load_planner
    from ..frontend.phrase_plan import plan_for_units
    from .phrase_acoustic_inference import append_plan
    from .phrase_prosody_evaluate import mask_focus_span,counterfactual_plans
    for directory in (PROJECT/'outputs/chinese_training_20261014/deps',PROJECT/'outputs/universal_tts_20261014/deps'):
        if str(directory) not in sys.path:sys.path.append(str(directory))
    output=Path(config['output']);destination=output/'focus_protocol.npz';metadata=output/'focus_protocol.json'
    if destination.exists():
        saved=json.loads(metadata.read_text())
        if saved['npz_sha256']!=sha256(destination) or saved['registration_sha256']!=sha256(Path(config['focus_registration'])) or saved['planner_checkpoint_sha256']!=identity['predicted_cache']['planner_checkpoint_sha256']:raise ValueError('Frozen focus protocol identity mismatch')
        return
    registered=json.loads(Path(config['focus_registration']).read_text())['focus_counterfactual']
    if len(registered)!=6:raise ValueError('Expected unchanged six registered focus spans')
    planner=load_planner(identity['planner_checkpoint']);arrays={};rows=[]
    for case in registered:
        predicted=planner.predict_text(case['text']);front=predicted['frontend'];mask=mask_focus_span(front,case['text'],case['character_start'],case['character_end'])
        variants=counterfactual_plans(predicted['syllable_plan'],mask,-1.5,1.5);low=append_plan(front,plan_for_units(front,variants['low']));high=append_plan(front,plan_for_units(front,variants['high']))
        n=len(front['phonetic_features']);target=np.zeros(n,bool)
        for syllable,selected in zip(front['metadata']['syllables'],mask):
            if selected:target[syllable['phone_start']:syllable['phone_end']]=True
        key=case['id'];arrays[key+'_phonetic']=np.asarray(low['phonetic_features'],np.float32);arrays[key+'_low']=low['context_features'];arrays[key+'_high']=high['context_features'];arrays[key+'_edge']=np.asarray(low['edge_sil_mask'],bool);arrays[key+'_target']=target
        changed=low['context_features']!=high['context_features'];allowed=np.zeros_like(changed);allowed[:,38]=True
        if (changed&~allowed).any():raise ValueError('Focus changes more than prominence input')
        rows.append({k:case[k] for k in ('id','text','target','character_start','character_end')})
    np.savez_compressed(destination,**arrays)
    save_json(metadata,{'cases':rows,'npz_sha256':sha256(destination),'registration_sha256':sha256(Path(config['focus_registration'])),
        'planner_checkpoint_sha256':identity['predicted_cache']['planner_checkpoint_sha256'],'normalization_sha256':identity['normalization_sha256'],
        'mode':'manually specified focus intent, not inferred focus correctness','input_change':'only PLAN8 prominence at explicit target ±1.5',
        'no_reference_pinyin_audio_timing_used':True,'nonzero_response_is_not_audible_quality_evidence':True})


def response_audit(model,output):
    """Same6 explicit-focus ordinary-text control responses; never modify outputs."""
    output=Path(output);metadata=json.loads((output/'focus_protocol.json').read_text());result=[];model.eval()
    with np.load(output/'focus_protocol.npz',allow_pickle=False) as cache:
        for case in metadata['cases']:
            key=case['id'];phonetic=torch.from_numpy(cache[key+'_phonetic']);edge=torch.from_numpy(cache[key+'_edge']);target=torch.from_numpy(cache[key+'_target']);speech=phonetic[:,0]<.5
            low={'phonetic_features':phonetic,'context_features':torch.from_numpy(cache[key+'_low']),'edge_sil_mask':edge}
            high={**low,'context_features':torch.from_numpy(cache[key+'_high'])}
            with torch.inference_mode():a=model(high);b=model(low)
            pitch=(a['controls'][...,0]-b['controls'][...,0])*1200/math.log(2);energy=(a['controls'][...,1]-b['controls'][...,1])*10/math.log(10)
            shape=a['controls'][...,2:25]-b['controls'][...,2:25];voice=(a['controls'][...,-1]>0)!=(b['controls'][...,-1]>0);voiced=(a['controls'][...,-1]>0)&(b['controls'][...,-1]>0)
            groups={}
            for name,mask in (('target',target&speech),('non_target',~target&speech)):
                ctrl=mask[:,None].expand_as(pitch);v=ctrl&voiced;duration=(a['duration_frames']-b['duration_frames'])[mask]
                groups[name]={'F0_RMS_difference_cents':float(pitch[v].square().mean().sqrt()) if v.any() else None,'F0_signed_mean_difference_cents':float(pitch[v].mean()) if v.any() else None,
                    'energy_RMS_difference_dB':float(energy[ctrl].square().mean().sqrt()),'energy_signed_mean_difference_dB':float(energy[ctrl].mean()),
                    'shape_RMS_difference':float(shape[mask].square().mean().sqrt()),'duration_mean_absolute_difference_ms':float(duration.abs().mean())*10,
                    'duration_signed_total_difference_ms':float(duration.sum())*10,'VUV_control_gate_flips':int(voice[mask].sum()),'units':int(mask.sum())}
            result.append({**case,'responses':groups})
    return {'scope':'same six ordinary-text explicit-focus numeric probes before runtime D; not automatic focus or listening evidence',
        'protocol_sha256':metadata['npz_sha256'],'input_change':'target prominence only +1.5 versus-1.5; phonetics and other7 plan columns exact',
        'output_F0_energy_duration_not_manually_modified':True,'nonzero_response_is_not_audible_quality_evidence':True,'cases':result,'MOS_measured':False}


def train(config,arm,resume=None):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(config['output']);out=output/'training'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        done=json.loads((out/'completed.json').read_text())
        if resume or done['portable_config_sha256']!=config['portable_config_sha256'] or sha256(out/'best.pt')!=done['checkpoint_sha256'] or sha256(out/'latest.pt')!=done['final_checkpoint_sha256']:raise ValueError('Frozen completed run identity differs')
        return
    if (out/'initial.pt').exists() and not resume:raise ValueError('Unfinished run requires explicit resume')
    identity=previous.cache_identity(config,'predicted_condition')
    if identity['planner_parameters']!=176408:raise ValueError('Fixed176408 planner required')
    seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model=load_source(config['checkpoint'],config['checkpoint_sha256']);means=model.means.clone();scales=model.scales.clone();prior=model.voiced_log_prior.clone()
    trainset=MixingRecords(config,'train');val=validation_view(config)
    if len(trainset)!=8932 or len(val)!=520:raise ValueError('Frozen split mismatch')
    with np.load(output/'sampling.npz',allow_pickle=False) as z:batches=z['indices'];uniforms=z['uniforms']
    if batches.shape!=(config['steps'],config['batch_records']) or uniforms.shape!=batches.shape:raise ValueError('Sampling protocol shape mismatch')
    probability=config['reference_probabilities'][arm];optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4);decoder=base.PhysicalSpectrum();start=0
    sources=[Path(__file__),PROJECT/'src/experiments/phrase_acoustic_train.py',PROJECT/'src/models/universal_acoustic.py',
        PROJECT/'src/experiments/chinese_train.py',PROJECT/'src/experiments/chinese_prosody_train.py',PROJECT/'src/frontend/phrase_plan.py',PROJECT/'src/experiments/phrase_prosody_evaluate.py']
    hashes={str(p.relative_to(PROJECT.parent)):sha256(p) for p in sources}
    lineage={'source_checkpoint_sha256':config['checkpoint_sha256'],'source_sha256':hashes,'plan_identity':identity,'arm':arm,
        'reference_probability':probability,'validation_plan_mode':'predicted','sampling_protocol_sha256':sha256(output/'sampling.npz'),
        'manifest_sha256':sha256(Path(config['data_dir'])/'manifest.jsonl'),'train_records':8932,'validation_records':520,'test_loaded':False,
        'normalization':'all acoustic and fixed planner normalization unchanged','selection':config['selection'],'preregistered_final_step':4000}
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['training_config']['portable_config_sha256']!=config['portable_config_sha256'] or state['lineage']!=lineage:raise ValueError('Resume identity/source mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];torch.set_rng_state(state['rng']['torch']);rng.bit_generator.state=state['rng']['sampler'];random.setstate(state['rng']['python'])
    else:
        snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
        for p in sources:shutil.copy2(p,snap/(p.stem+'_'+sha256(p)[:12]+p.suffix))
        torch.manual_seed(seed+1)
    save_json(out/'config.json',{**config,'arm':arm});save_json(out/'lineage.json',lineage);save_json(out/'resources.json',resource_estimate(model))
    initial=validate(model,val,decoder);best=initial['total'];best_step=start;best_metrics=initial
    if resume:
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        if saved['validation']['total']<=best:best,best_step,best_metrics=saved['validation']['total'],saved['step'],saved['validation']
    else:
        save_json(out/'initial_validation.json',initial);base.save_checkpoint(out/'initial.pt',model,optimizer,0,config,initial,lineage,rng);base.save_checkpoint(out/'best.pt',model,optimizer,0,config,initial,lineage,rng)
        save_json(out/'response_initial.json',response_audit(model,output))
    digest=hashlib.sha256();mix_digest=hashlib.sha256();selected_count=0;started=time.monotonic()
    with (out/'training.jsonl').open('a') as log:
        for step in range(start+1,config['steps']+1):
            indices=batches[step-1];u=uniforms[step-1];digest.update(np.asarray(indices,dtype='<i8').tobytes());mix_digest.update(np.asarray(u,dtype='<f8').tobytes())
            f,t,m,d,used=batch(trainset,indices,u,probability);selected_count+=int(used.sum());model.train();prediction=model({k:f[k] for k in base.TEXT_INPUT_KEYS})
            loss,_=base.objective(prediction,t,m,d,f,model.scales,decoder,config['physical_samples'],model.voiced_log_prior,'frame')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if step==start+1:save_json(out/'gradient_sanity.json',{'finite':True,'gradient_norm':float(grad),'plan8_gradient_norm':float(model.input.weight.grad[:,66:].norm())})
            optimizer.step();lr=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*step/config['steps'])))
            for group in optimizer.param_groups:group['lr']=lr
            if step==1 or step%100==0:
                row={'step':step,'wall_seconds':time.monotonic()-started,'training_loss':float(loss.detach()),'learning_rate':lr,'gradient_norm':float(grad),'reference_utterances_in_batch':int(used.sum())};log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step%config['eval_every']==0 or step==config['steps']:
                v=validate(model,val,decoder);save_json(out/f'validation_step{step:06d}.json',v);save_json(out/f'response_step{step:06d}.json',response_audit(model,output));base.save_checkpoint(out/'latest.pt',model,optimizer,step,config,v,lineage,rng)
                if v['total']<best:best,best_step,best_metrics=v['total'],step,v;base.save_checkpoint(out/'best.pt',model,optimizer,step,config,v,lineage,rng)
                save_json(out/'running.json',{'step':step,'best_step':best_step,'best_loss':best,'status':'training'})
    expected=hashlib.sha256(np.asarray(batches[start:],dtype='<i8').tobytes()).hexdigest();expected_mix=hashlib.sha256(np.asarray(uniforms[start:],dtype='<f8').tobytes()).hexdigest()
    if digest.hexdigest()!=expected or mix_digest.hexdigest()!=expected_mix:raise ValueError('Actual batch/U sampler mismatch')
    for label,path in (('best',out/'best.pt'),('final4000',out/'latest.pt')):
        loaded=load_acoustic(path)
        if not torch.equal(loaded.means,means) or not torch.equal(loaded.scales,scales) or not torch.equal(loaded.voiced_log_prior,prior):raise ValueError('Normalization changed')
        save_json(out/('response_'+label+'.json'),response_audit(loaded,output))
    if sha256(Path(config['checkpoint']))!=config['checkpoint_sha256']:raise ValueError('Source checkpoint bytes changed')
    save_json(out/'completed.json',{'status':'complete','steps':config['steps'],'best_step':best_step,'best_original_validation_loss':best,'best_validation':best_metrics,
        'checkpoint_sha256':sha256(out/'best.pt'),'final_checkpoint_sha256':sha256(out/'latest.pt'),'final4000_role':config['final_step_role'],
        'parameters':563098,'planner_parameters':176408,'total_parameters':739506,'reference_probability':probability,
        'sampler_sha256':hashlib.sha256(np.asarray(batches,dtype='<i8').tobytes()).hexdigest(),'uniforms_sha256':hashlib.sha256(np.asarray(uniforms,dtype='<f8').tobytes()).hexdigest(),
        'actual_segment_sampler_sha256':digest.hexdigest(),'actual_segment_uniforms_sha256':mix_digest.hexdigest(),'actual_segment_scope':f'updates {start+1}..{config["steps"]}',
        'reference_selected_segment_count':selected_count,'validation_plan_mode':'predicted','source_checkpoint_unchanged':True,
        'portable_config_sha256':config['portable_config_sha256'],'test_loaded':False,'wall_seconds':time.monotonic()-started})


def freeze(config):
    output=Path(config['output']);names=list(config['reference_probabilities']);completed={n:json.loads((output/'training'/n/'completed.json').read_text()) for n in names}
    if len({d['sampler_sha256'] for d in completed.values()})!=1 or len({d['uniforms_sha256'] for d in completed.values()})!=1:raise ValueError('Matched batch/U protocol differs')
    target=output/'selection_ready.json'
    if target.exists():
        old=json.loads(target.read_text())
        for n,d in completed.items():
            if old['checkpoints'][n]['sha256']!=d['checkpoint_sha256'] or sha256(output/'training'/n/'best.pt')!=d['checkpoint_sha256'] or sha256(output/'training'/n/'latest.pt')!=d['final_checkpoint_sha256'] or d['portable_config_sha256']!=config['portable_config_sha256']:raise ValueError('Frozen selection identity mismatch')
        return
    save_json(target,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'same_sampler_and_uniforms':True,'test_loaded':False,'validation_plan_mode':'predicted',
        'source_checkpoint_sha256':config['checkpoint_sha256'],'checkpoints':{n:{'checkpoint':str(output/'training'/n/'best.pt'),'sha256':d['checkpoint_sha256'],'completed':d} for n,d in completed.items()},
        'fixed_final4000':{n:{'checkpoint':str(output/'training'/n/'latest.pt'),'sha256':d['final_checkpoint_sha256'],'role':config['final_step_role']} for n,d in completed.items()},'default_promoted':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=PROJECT/'configs/phrase_execution_20261015.json');p.add_argument('--arm');p.add_argument('--resume',type=Path);a=p.parse_args();config=json.loads(a.config.read_text());config['portable_config_sha256']=sha256(a.config)
    for key in ('checkpoint','data_dir','feature_cache','labels_dir','focus_registration','output'):config[key]=str((PROJECT.parent/config[key]).resolve())
    if config['steps']!=4000 or config['preregistered_final_step']!=4000 or config['validation_plan_mode']!='predicted':raise ValueError('Frozen experiment protocol mismatch')
    if a.arm:
        if a.arm not in config['reference_probabilities']:p.error('Unknown arm')
        train(config,a.arm,a.resume);return
    if a.resume:p.error('--resume requires --arm')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    if (output/'selection_ready.json').exists():freeze(config);print('Completed frozen experiment verified; preserved');return
    batches,uniforms=sampling_protocol(config['seed'],config['steps'],8932,config['batch_records']);sample=output/'sampling.npz'
    if sample.exists():
        with np.load(sample,allow_pickle=False) as z:
            if not np.array_equal(z['indices'],batches) or not np.array_equal(z['uniforms'],uniforms):raise ValueError('Frozen sampling differs')
    else:
        np.savez_compressed(sample,indices=batches,uniforms=uniforms)
        save_json(output/'sampling.json',{'steps':4000,'batch_size':12,'seed':config['seed'],'uniform_seed':config['seed']+1,'sampling_sha256':sha256(sample),
            'uniforms_are_precomputed_shared_across_all_arms':True,'mixing_unit':'whole utterance Nx8','final4000_preregistered':True,'test_loaded':False})
    identity=previous.cache_identity(config,'predicted_condition');prepare_focus_protocol(config,identity)
    jobs=[];logs=[]
    for name in config['reference_probabilities']:
        folder=output/'training'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'initial.pt').exists() and not (folder/'completed.json').exists():raise ValueError('Unfinished arm requires explicit resume')
        log=(folder/'stdout.log').open('ab',buffering=0);logs.append(log);jobs.append((name,subprocess.Popen([sys.executable,'-u','-B','-m','project.src.experiments.phrase_execution_train','--config',str(a.config),'--arm',name],stdout=log,stderr=subprocess.STDOUT)))
    save_json(output/'processes.json',{'UTC':datetime.now(timezone.utc).isoformat(),'jobs':{n:proc.pid for n,proc in jobs}})
    failed=[n for n,proc in jobs if proc.wait()!=0]
    for log in logs:log.close()
    if failed:save_json(output/'failed.json',{'arms':failed});raise RuntimeError(str(failed))
    freeze(config)


if __name__=='__main__':main()
