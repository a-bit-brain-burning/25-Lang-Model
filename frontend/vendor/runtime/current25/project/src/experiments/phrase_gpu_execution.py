"""Separate FP32 CUDA runtime experiment for the frozen R2 mix50 protocol.

CPU formula/gradient parity is tested before CUDA execution. This module never
changes the CPU experiment, old objective, planner, DSP or source checkpoint.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_k,'1')
import argparse
from datetime import datetime,timezone
import json,math,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..common import PROJECT,save_json,sha256
from ..models.universal_acoustic import FORMAT,UniversalAcousticModel,load_acoustic
from . import phrase_execution_train as cpu


def device_objective(out,target,mask,duration,features,scales,decoder,physical_samples=768,
                     voiced_log_prior=0.,vuv_supervision='frame'):
    """Exactly the old objective, with constructed indices on p.device."""
    p=out['controls'];device=p.device;norm_error=(p-target)/scales
    avg=lambda x,m:(x*m).sum()/m.sum().clamp_min(1)
    spectral=avg(norm_error[...,2:25].square(),mask[...,2:25]);energy=avg(norm_error[...,1].square(),mask[...,1])
    pitch=avg(F.smooth_l1_loss(norm_error[...,0],torch.zeros_like(norm_error[...,0]),reduction='none'),mask[...,0])
    ap=avg(norm_error[...,25:-1].square(),mask[...,25:-1])
    if vuv_supervision=='frame':
        positions=features['_frame_fractions']*7;left=positions.long().clamp(0,6);alpha=positions-left
        c=p.reshape(-1,8,31)[features['_frame_indices'],:,-1];row=torch.arange(len(left),device=device)
        logits=c[row,left]*(1-alpha)+c[row,left+1]*alpha-voiced_log_prior
        vuv=F.binary_cross_entropy_with_logits(logits,features['_frame_voice'],reduction='none').mean()
    else:
        voice=torch.sigmoid(target[...,-1]);bce=F.binary_cross_entropy_with_logits(p[...,-1]-voiced_log_prior,voice,reduction='none')
        vuv=.5*avg(bce,mask[...,-1]*voice)+.5*avg(bce,mask[...,-1]*(1-voice))
    active=(mask[...,1]>0).flatten().nonzero().flatten()
    if len(active)>physical_samples:active=active[torch.randperm(len(active),device=device)[:physical_samples]]
    pp,tt=p.reshape(-1,31)[active],target.reshape(-1,31)[active]
    db_error=(decoder(pp[:,1:25])-decoder(tt[:,1:25]))/math.log(10);physical=db_error.square().mean()
    valid,edge=features['valid_mask'],features['edge_sil_mask']
    logduration=F.smooth_l1_loss(torch.log1p(out['duration_frames']),torch.log1p(duration),reduction='none',beta=.5)
    local=avg(logduration,valid);delta=out['duration_frames']-duration;count=valid.sum(-1)
    position=torch.arange(1,duration.shape[1]+1,device=device)[None];inner=valid&(position<count[:,None])
    boundary=avg(F.smooth_l1_loss(delta.cumsum(-1)/(5*position.sqrt()),torch.zeros_like(delta),reduction='none'),inner)
    sentence=F.smooth_l1_loss(delta.sum(-1)/(5*count.sqrt()),torch.zeros_like(count,dtype=torch.float32))
    edge_bce=avg(F.binary_cross_entropy_with_logits(out['edge_nonzero_logits'],(duration>0).float(),reduction='none'),edge&valid)
    timing=local+.3*boundary+.3*sentence+.1*edge_bce
    total=physical+.15*spectral+.1*energy+.3*pitch+.2*ap+.7*vuv+.3*timing
    return total,{'total':total,'physical_logpower_mse_db_div10':physical,'shape_normalized_mse':spectral,
        'energy_normalized_mse':energy,'f0_normalized_huber':pitch,'ap_normalized_mse':ap,'vuv_balanced_bce':vuv,
        'duration':timing,'duration_log_huber':local,'duration_boundary':boundary,'duration_sentence':sentence,'edge_nonzero_bce':edge_bce}


def prepare(config_path):
    config=json.loads(Path(config_path).read_text())
    for key in ('checkpoint','data_dir','feature_cache','labels_dir','focus_registration'):
        config[key]=str((PROJECT.parent/config[key]).resolve())
    config.update(output=str(PROJECT/'outputs/phrase_prosody_r2_20261015/gpu_execution'),
        arm='reference_mix50',reference_probability=.5,device='cuda:0',precision='float32',TF32=False,
        runtime_comparison='separate CUDA/torch runtime variable; not bitwise matched to CPU',
        source_CPU_config_sha256=sha256(Path(config_path)))
    if config['steps']!=4000 or config['seed']!=20261023 or config['batch_records']!=12 or config['learning_rate']!=.0001:raise ValueError('GPU experiment must match frozen R2 mix50 recipe')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    path=output/'config.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise ValueError('Frozen GPU config differs')
    else:save_json(path,config)
    source=PROJECT/'outputs/phrase_prosody_r2_20261015/execution'
    for name in ('sampling.npz','sampling.json','focus_protocol.npz','focus_protocol.json'):
        target=output/name;origin=source/name
        if target.exists():
            if sha256(target)!=sha256(origin):raise ValueError('CPU/GPU frozen protocol differs')
        else:shutil.copy2(origin,target)
    return config


def cpu_copy(model):
    # Creating a validation copy must not consume the CPU training RNG stream.
    with torch.random.fork_rng(devices=[]):
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        clone=UniversalAcousticModel(model.config,state['means'],state['scales']);clone.load_state_dict(state);clone.eval()
    return clone


def save_checkpoint(path,model,optimizer,step,config,validation,lineage,rng):
    state={'format':FORMAT,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
        'config':model.config,'training_config':config,'validation':validation,'lineage':lineage,
        'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'sampler':rng.bit_generator.state,'python':random.getstate()},
        'saved_utc':datetime.now(timezone.utc).isoformat()}
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)


def execute(config,resume=None):
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this runtime; root must execute in GPU environment with approved access')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);np.random.seed(config['seed']);random.seed(config['seed']);rng=np.random.default_rng(config['seed'])
    output=Path(config['output']);folder=output/'training';folder.mkdir(parents=True,exist_ok=True)
    if (output/'completed.json').exists():
        done=json.loads((output/'completed.json').read_text())
        if resume or done['config_sha256']!=sha256(output/'config.json') or done['best_checkpoint_sha256']!=sha256(folder/'best.pt') or done['final_checkpoint_sha256']!=sha256(folder/'latest.pt'):raise ValueError('Frozen CUDA output differs')
        print('Already complete; frozen CUDA experiment verified');return
    if (folder/'initial.pt').exists() and not resume:raise ValueError('Unfinished CUDA run requires --resume')
    identity=cpu.previous.cache_identity(config,'predicted_condition')
    if identity['planner_parameters']!=176408:raise ValueError('Fixed original planner required')
    model=cpu.load_source(config['checkpoint'],config['checkpoint_sha256']).to(config['device'])
    if next(model.parameters()).dtype!=torch.float32:raise ValueError('FP32 required')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    decoder=cpu.base.PhysicalSpectrum().to(config['device']);cpu_decoder=cpu.base.PhysicalSpectrum()
    training=cpu.MixingRecords(config,'train');validation=cpu.validation_view(config)
    with np.load(output/'sampling.npz',allow_pickle=False) as z:indices=z['indices'];uniforms=z['uniforms']
    runtime={'torch':str(torch.__version__),'CUDA_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(),
        'capability':list(torch.cuda.get_device_capability()),'device_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'CUDA_devices':torch.cuda.device_count(),'TF32':False,'precision':'float32','deterministic_algorithms':True,
        'CPU_threads':1,'cuDNN_version':torch.backends.cudnn.version(),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}
    sources=[Path(__file__),PROJECT/'src/experiments/phrase_execution_train.py',PROJECT/'src/experiments/phrase_acoustic_train.py',PROJECT/'src/experiments/chinese_train.py',PROJECT/'src/models/universal_acoustic.py']
    hashes={str(p.relative_to(PROJECT.parent)):sha256(p) for p in sources}
    lineage={'source_sha256':hashes,'source_checkpoint_sha256':config['checkpoint_sha256'],'runtime':runtime,
        'sampling_protocol_sha256':sha256(output/'sampling.npz'),'focus_protocol_sha256':sha256(output/'focus_protocol.npz'),
        'plan_identity':identity,'test_loaded':False,'selection':'pure predicted full520 common CPU validation loss; step0 eligible',
        'runtime_axis_is_separate_from_CPU_recipe':True,'config_sha256':sha256(output/'config.json')}
    start=0
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['lineage']!=lineage:raise ValueError('CUDA resume identity/runtime/source mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);rng.bit_generator.state=state['rng']['sampler'];random.setstate(state['rng']['python'])
    else:
        snap=output/'source_snapshot';snap.mkdir(exist_ok=True)
        for p in sources:shutil.copy2(p,snap/(p.stem+'_'+sha256(p)[:12]+p.suffix))
        torch.manual_seed(config['seed']+1);torch.cuda.manual_seed_all(config['seed']+1)
    save_json(output/'lineage.json',lineage);save_json(output/'runtime.json',runtime)
    clone=cpu_copy(model);initial=cpu.validate(clone,validation,cpu_decoder);best,best_step,best_metrics=initial['total'],start,initial
    if resume:
        old=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        if old['validation']['total']<=best:best,best_step,best_metrics=old['validation']['total'],old['step'],old['validation']
    else:
        save_json(output/'initial_validation.json',initial);save_checkpoint(folder/'initial.pt',model,optimizer,0,config,initial,lineage,rng);save_checkpoint(folder/'best.pt',model,optimizer,0,config,initial,lineage,rng)
        save_json(output/'response_initial.json',cpu.response_audit(clone,output))
    del clone
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();training_seconds=compute_ms=0.;postwarm_seconds=0.;mixture_count=0
    event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True)
    with (output/'training.jsonl').open('a') as log:
        for step in range(start+1,config['steps']+1):
            tick=time.perf_counter();f,t,m,d,used=cpu.batch(training,indices[step-1],uniforms[step-1],.5);mixture_count+=int(used.sum())
            f={k:v.to(config['device']) if torch.is_tensor(v) else v for k,v in f.items()};t,m,d=(v.to(config['device']) for v in (t,m,d))
            model.train();event_start.record();out=model({k:f[k] for k in cpu.base.TEXT_INPUT_KEYS})
            loss,terms=device_objective(out,t,m,d,f,model.scales,decoder,config['physical_samples'],model.voiced_log_prior,'frame')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CUDA loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();event_end.record();event_end.synchronize()
            elapsed=time.perf_counter()-tick;training_seconds+=elapsed;compute_ms+=event_start.elapsed_time(event_end)
            if step>50:postwarm_seconds+=elapsed
            lr=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*step/config['steps'])))
            for group in optimizer.param_groups:group['lr']=lr
            if step==1 or step%100==0:
                row={'step':step,'wall_seconds':time.monotonic()-started,'training_loss':float(loss.detach()),'gradient_norm':float(gradient),'learning_rate':lr,
                    'training_steps_per_second':(step-start)/training_seconds,'CUDA_compute_ms_total':compute_ms,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                save_json(output/'running.json',{**row,'status':'training_cuda'})
            if step%config['eval_every']==0 or step==config['steps']:
                clone=cpu_copy(model);metrics=cpu.validate(clone,validation,cpu_decoder)
                save_json(output/f'validation_step{step:06d}.json',metrics);save_json(output/f'response_step{step:06d}.json',cpu.response_audit(clone,output));del clone
                save_checkpoint(folder/'latest.pt',model,optimizer,step,config,metrics,lineage,rng)
                if metrics['total']<best:best,best_step,best_metrics=metrics['total'],step,metrics;save_checkpoint(folder/'best.pt',model,optimizer,step,config,metrics,lineage,rng)
    if sha256(Path(config['checkpoint']))!=config['checkpoint_sha256']:raise ValueError('Source checkpoint changed')
    for name in ('best','latest'):
        clone=load_acoustic(folder/(name+'.pt'));save_json(output/('response_'+name+'.json'),cpu.response_audit(clone,output))
    result={'status':'complete','steps':config['steps'],'best_step':best_step,'best_validation':best_metrics,'best_checkpoint':str(folder/'best.pt'),
        'best_checkpoint_sha256':sha256(folder/'best.pt'),'final_checkpoint':str(folder/'latest.pt'),'final_checkpoint_sha256':sha256(folder/'latest.pt'),
        'config_sha256':sha256(output/'config.json'),'runtime':runtime,'parameters':563098,'planner_parameters':176408,'total_parameters':739506,
        'training_steps_per_second':(config['steps']-start)/training_seconds,'postwarm_steps_per_second':max(0,config['steps']-max(start,50))/max(postwarm_seconds,1e-9),
        'training_seconds_excludes_validation':training_seconds,'CUDA_event_compute_ms_total':compute_ms,'wall_seconds':time.monotonic()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'reference_selected_segment_count':mixture_count,'segment_scope':f'{start+1}..{config["steps"]}',
        'test_loaded':False,'CPU_GPU_not_bitwise_or_single_torch_version_comparison':True,'default_promoted':False}
    save_json(output/'selection_ready.json',result);save_json(output/'completed.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=PROJECT/'configs/phrase_execution_20261015.json');p.add_argument('--execute-training',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
    config=prepare(a.config)
    if not a.execute_training:print('CUDA experiment prepared only; root approval/execution required');return
    execute(config,a.resume)


if __name__=='__main__':main()
