"""Matched continuation with optional, confidence-weighted supervision overlay.

The original objective remains untouched. Q0 calls it directly; Q1 changes
only target/mask/frame-confidence inputs and confidence weighting. No model,
frontend, duration, planner, representation or DSP transformation is added.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_key,'1')
import argparse,copy,hashlib,json,math,random,resource,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..common import PROJECT,save_json,sha256
from ..models.universal_acoustic import FORMAT,load_acoustic
from ..training.clarity_objective import clarity_terms,weighted_clarity,frame_logits
from . import english_word_train as old


class Plans(old.Plans):
    """Explicit frozen old plan_root; never reinterpret the new output root."""
    def __init__(self,config):
        root=Path(config['plan_root']);receipt=json.loads((root/'predicted_completed.json').read_text())
        if receipt['status']!='COMPLETED' or receipt['checkpoint_sha256']!=config['planner_sha256']:
            raise ValueError('Frozen text planner/cache differs')
        self.root=root;self.cache={};self.receipt=receipt


class Overlay:
    def __init__(self,root,views,allow_provisional=False):
        root=Path(root);receipt=json.loads((root/'completed.json').read_text())
        if receipt['status']!='COMPLETED':raise ValueError('Strict overlay is not atomically complete')
        if receipt.get('provisional',False)and not allow_provisional:raise ValueError('Provisional overlay cannot train a formal model')
        manifest=root/'manifest.jsonl'
        if receipt.get('manifest_sha256')and receipt['manifest_sha256']!=sha256(manifest):raise ValueError('Strict manifest changed')
        rows=[json.loads(line)for line in manifest.read_text().splitlines()if line.strip()]
        expected={r['id']:(r,a)for view in views for r,a in zip(view.rows,view.arrays)}
        if len(rows)!=len(expected)or {r['id']for r in rows}!=set(expected):raise ValueError('Strict overlay must preserve every source item')
        self.cache={};self.receipt=receipt;self.manifest_sha256=sha256(manifest)
        for row in rows:
            source,original=expected[row['id']]
            if row['split']!=source['split']:raise ValueError('Overlay split differs')
            path=Path(row['overlay_path']);digest=row.get('overlay_sha256',row.get('sha256'))
            if sha256(path)!=digest:raise ValueError('Overlay record changed '+row['id'])
            if Path(row['source_v2_path']).resolve()!=Path(source['record_path']).resolve():raise ValueError('Overlay source path differs')
            expected_source=source.get('record_sha256',source.get('sha256'))
            if expected_source and row['source_v2_sha256']!=expected_source:raise ValueError('Overlay source SHA differs')
            with np.load(path,allow_pickle=False)as z:
                target=z['target_controls'];mask=z['control_mask'];voice=z['frame_voice'];weight=z['frame_voice_weight']
            if target.shape!=original['target_controls'].shape or mask.shape!=target.shape:raise ValueError('Overlay control topology changed')
            if not np.array_equal(target[...,1:-1],original['target_controls'][...,1:-1]):raise ValueError('Only F0/VUV target values may change')
            if not np.array_equal(mask[...,1],original['control_mask'][...,1]):raise ValueError('Independent energy supervision must stay unchanged')
            frames=int(original['target_duration'].sum())
            if voice.shape!=(frames,)or weight.shape!=(frames,):raise ValueError('Overlay frame axis changed')
            if not all(np.isfinite(x).all()for x in (target,mask,voice,weight)):raise ValueError('Nonfinite overlay')
            if np.any((mask<0)|(mask>1))or np.any((weight<0)|(weight>1))or np.any((voice<0)|(voice>1)):
                raise ValueError('Confidence and voice targets must be in [0,1]')
            # Retain only changed values plus full FP32 confidence masks.
            self.cache[row['id']]={'control_f0_vuv':target[...,[0,-1]].copy(),'control_mask':mask,
                                   'frame_voice':voice,'frame_voice_weight':weight,'sha256':digest}
    def get(self,uid):return self.cache[uid]


def to_device(packed,device):
    return tuple({k:v.to(device)if torch.is_tensor(v)else v for k,v in x.items()}if isinstance(x,dict)else x.to(device)for x in packed)


def batch(view,indices,device,plans=None,reference_rows=None,overlay=None):
    indices=list(map(int,indices));packed=old.batch(view,indices,'cpu',plans,reference_rows)
    if overlay is None:return to_device(packed,device)
    f,target,mask,duration,groups=packed;width=f['valid_mask'].shape[1];voices=[];weights=[];expected_indices=[];fractions=[]
    for j,index in enumerate(indices):
        row,a=view[index];item=overlay.get(row['id']);counts=np.asarray(a['target_duration'],np.int64);n=len(counts)
        values=item['control_f0_vuv'];target[j,:n,:,0]=torch.from_numpy(values[...,0]);target[j,:n,:,-1]=torch.from_numpy(values[...,1]);mask[j,:n]=torch.from_numpy(item['control_mask'])
        # Base flattened controls are indexed in padded B×N coordinates.
        # Reconstruct the exact corresponding per-record source frame order;
        # raw flattening a padded frame matrix would silently mix recordings.
        expected_indices.append(np.repeat(np.arange(n,dtype=np.int64)+j*width,counts))
        fractions.extend([np.arange(c,dtype=np.float32)/c for c in counts if c>0])
        expected_shape=(int(counts.sum()),)
        if item['frame_voice'].shape!=expected_shape or item['frame_voice_weight'].shape!=expected_shape:
            raise ValueError('Frame voice/confidence shape differs from unchanged duration')
        voices.append(item['frame_voice']);weights.append(item['frame_voice_weight'])
    if not np.array_equal(f['_frame_indices'].numpy(),np.concatenate(expected_indices)):raise ValueError('Padded control/frame index mapping changed')
    if not np.array_equal(f['_frame_fractions'].numpy(),np.concatenate(fractions)):raise ValueError('Within-phone frame positions changed')
    f['_frame_voice']=torch.from_numpy(np.concatenate(voices).astype(np.float32,copy=False))
    f['_frame_voice_weight']=torch.from_numpy(np.concatenate(weights).astype(np.float32,copy=False))
    physical_weight=mask[...,1]*mask[...,2:25].amin(-1);active=physical_weight>0
    f['_strict_physical_uniform']=bool(torch.all(physical_weight[active]==1))
    return to_device((f,target,mask,duration,groups),device)


def strict_common(out,target,mask,duration,features,scales,decoder,physical_samples=768,voiced_log_prior=0.):
    """Original common terms, with correctly weighted physical spectrum only.

    Spectrum size comes from decoder metadata; this does not assert that a
    different representation's targets or old clarity terms are compatible.
    """
    p=out['controls'];device=p.device;channels=p.shape[-1];spec=decoder.basis.shape[0]+1;stop=1+spec
    error=(p-target)/scales;avg=lambda x,m:(x*m).sum()/m.sum().clamp_min(1)
    spectral=avg(error[...,2:stop].square(),mask[...,2:stop]);energy=avg(error[...,1].square(),mask[...,1])
    pitch=avg(F.smooth_l1_loss(error[...,0],torch.zeros_like(error[...,0]),reduction='none'),mask[...,0])
    ap=avg(error[...,stop:-1].square(),mask[...,stop:-1])
    voice=target[...,-1].sigmoid();bce=F.binary_cross_entropy_with_logits(p[...,-1]-voiced_log_prior,voice,reduction='none')
    vuv=.5*avg(bce,mask[...,-1]*voice)+.5*avg(bce,mask[...,-1]*(1-voice))
    physical_weight=(mask[...,1]*mask[...,2:stop].amin(-1)).flatten();active=(physical_weight>0).nonzero().flatten()
    if len(active)>physical_samples:active=active[torch.randperm(len(active),device=device)[:physical_samples]]
    if len(active):
        pp,tt=p.reshape(-1,channels)[active],target.reshape(-1,channels)[active]
        db=(decoder(pp[:,1:stop])-decoder(tt[:,1:stop]))/math.log(10)
        if features.get('_strict_physical_uniform',False):physical=db.square().mean()
        else:
            weight=physical_weight[active];physical=(db.square().mean(-1)*weight).sum()/weight.sum().clamp_min(1e-12)
    else:physical=p.sum()*0
    valid,edge=features['valid_mask'],features['edge_sil_mask'];delta=out['duration_frames']-duration;count=valid.sum(-1)
    local=avg(F.smooth_l1_loss(torch.log1p(out['duration_frames']),torch.log1p(duration),reduction='none',beta=.5),valid)
    position=torch.arange(1,duration.shape[1]+1,device=device)[None];inner=valid&(position<count[:,None])
    boundary=avg(F.smooth_l1_loss(delta.cumsum(-1)/(5*position.sqrt()),torch.zeros_like(delta),reduction='none'),inner)
    sentence=F.smooth_l1_loss(delta.sum(-1)/(5*count.sqrt()),torch.zeros_like(count,dtype=torch.float32))
    edge_bce=avg(F.binary_cross_entropy_with_logits(out['edge_nonzero_logits'],(duration>0).float(),reduction='none'),edge&valid)
    timing=local+.3*boundary+.3*sentence+.1*edge_bce
    total=physical+.15*spectral+.1*energy+.3*pitch+.2*ap+.7*vuv+.3*timing
    return total,{'total':total,'physical_logpower_mse_db_div10':physical,'shape_normalized_mse':spectral,
                  'energy_normalized_mse':energy,'f0_normalized_huber':pitch,'ap_normalized_mse':ap,'vuv_balanced_bce':vuv,
                  'duration':timing,'duration_log_huber':local,'duration_boundary':boundary,'duration_sentence':sentence,'edge_nonzero_bce':edge_bce}


def objective(model,packed,decoder,config,all_physical=False,strict=False):
    if not strict:return old.objective(model,packed,decoder,config,all_physical)
    f,target,mask,duration,groups=packed;out=model({k:f[k]for k in old.prior.base.TEXT_INPUT_KEYS})
    common,terms=strict_common(out,target,mask,duration,f,model.scales,decoder,10000000 if all_physical else config['physical_samples'],model.voiced_log_prior)
    clarity=clarity_terms(out,target,mask,duration,f,model.scales,groups)
    logits=frame_logits(out['controls'],f);lexical=groups['lexical'].flatten()[f['_frame_indices']]
    weight=lexical*f['_frame_voice_weight'];frame_bce=F.binary_cross_entropy_with_logits(logits,f['_frame_voice'],reduction='none')
    clarity['frame_vuv']=(frame_bce*weight).sum()/weight.sum().clamp_min(1)
    loss=common+weighted_clarity(clarity,config['weights'])
    pause=f['_internal_pause_mask']&f['valid_mask'];positive=pause&(duration>0);negative=pause&~positive
    avg=lambda value,mask:(value*mask).sum()/mask.sum().clamp_min(1)
    gate=F.binary_cross_entropy_with_logits(out['edge_nonzero_logits'],(duration>0).float(),reduction='none')
    gate_loss=.5*avg(gate,positive)+.5*avg(gate,negative)
    pd=F.smooth_l1_loss(torch.log1p(out['positive_frames']),torch.log1p(duration),reduction='none',beta=.5);pause_duration=avg(pd,positive)
    loss=loss+.05*gate_loss+.2*pause_duration
    return loss,{**terms,**{'clarity_'+k:v for k,v in clarity.items()},'pause_gate':gate_loss,'positive_pause_duration':pause_duration,
                 'pause_positive_recall':avg((out['edge_nonzero_logits']>=0).float(),positive),
                 'pause_false_positive':avg((out['edge_nonzero_logits']>=0).float(),negative),
                 'positive_pause_MAE_ms':10*avg((out['positive_frames']-duration).abs(),positive),'total_with_aux':loss}


def sampling(config,count):
    rng=np.random.default_rng(config['seed']);samples=np.stack([rng.choice(count,config['batch_records'],replace=False)for _ in range(config['steps'])])
    mix=np.random.default_rng(config['seed']+1).random((config['steps'],config['batch_records']))<config['reference_plan_probability']
    return samples,mix


@torch.inference_mode()
def validate(model,view,device,decoder,config,plans,overlay=None):
    model.eval();sums={};started=time.perf_counter()
    for start in range(0,len(view),config['batch_records']):
        ix=range(start,min(start+config['batch_records'],len(view)));packed=batch(view,ix,device,plans,overlay=overlay)
        _,metrics=objective(model,packed,decoder,config,True,overlay is not None)
        for name,value in metrics.items():sums[name]=sums.get(name,0.)+float(value)*len(ix)
    return {**{k:v/len(view)for k,v in sums.items()},'recordings':len(view),'axis':'REFERENCE_TIME_PREDICTED_TEXT_PLAN',
            'supervision':'STRICT_CONFIDENCE_OVERLAY'if overlay else'COMMON_ORIGINAL_PAUSE_V2','seconds':time.perf_counter()-started}


def save_checkpoint(path,model,opt,step,config,lineage,validation,progress,best_score,best_step):
    state={'format':FORMAT,'config':model.config,'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'training_config':config,
           'lineage':lineage,'validation':validation,'progress':progress,'best_score':best_score,'best_step':best_step,
           'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
           'cuda_rng':torch.cuda.get_rng_state_all()if next(model.parameters()).is_cuda else []}
    path=Path(path);temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)


def restore_rng(state,device):
    torch.set_rng_state(state['torch_rng'].cpu());random.setstate(state['python_rng'])
    if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])


def source_model(config):
    if sha256(config['parent'])!=config['parent_sha256']:raise ValueError('Frozen source checkpoint changed')
    model=load_acoustic(config['parent'])
    if model.config['context_dim']!=27 or model.channels!=31:raise ValueError('This frozen strict31 experiment needs the existing context27 source, no upgrade')
    return model.eval()


def train(config,arms,device='cuda:0',resume=None):
    old.setup();root=Path(config['output']);root.joinpath('training').mkdir(parents=True,exist_ok=True)
    source=source_model(config);tick=time.perf_counter();training=old.Records(config,'train',True);validation=old.Records(config,'validation',True);plans=Plans(config)
    overlay=Overlay(config['strict_data'],(training,validation))if 'Q1'in arms else None
    cache_seconds=time.perf_counter()-tick;samples,mix=sampling(config,len(training))
    files=[Path(__file__),Path(old.__file__),PROJECT/'src/experiments/phrase_gpu_execution.py',PROJECT/'src/experiments/universal_train.py',
           PROJECT/'src/training/clarity_objective.py',PROJECT/'src/models/universal_acoustic.py']
    identity={'source_checkpoint':config['parent_sha256'],'planner_checkpoint':config['planner_sha256'],
              'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'sources':{str(p):sha256(p)for p in files},'sample_sha256':hashlib.sha256(samples.astype('<i8').tobytes()).hexdigest(),
              'mix_sha256':hashlib.sha256(mix.astype(np.uint8).tobytes()).hexdigest(),
              'record_ids_sha256':hashlib.sha256('\n'.join(r['id']for r in training.rows).encode()).hexdigest(),
              'pause_manifest_sha256':sha256(Path(config['pause_data'])/'manifest.jsonl'),
              'plans_receipt_sha256':sha256(Path(config['plan_root'])/'predicted_completed.json'),
              # Identity must not depend on whether Q0 was launched alone or
              # together with Q1; a single-arm resume must stay compatible.
              'strict_manifest_sha256':sha256(Path(config['strict_data'])/'manifest.jsonl'),
              'strict_receipt_sha256':sha256(Path(config['strict_data'])/'completed.json'),
              'test_loaded':False,'selection':'fixed final step primary; best uses COMMON ORIGINAL total_with_aux only'}
    for arm in arms:
        if arm not in ('Q0','Q1'):raise ValueError('Unknown strict supervision arm')
        folder=root/'training'/arm;folder.mkdir(exist_ok=True);lineage={**identity,'arm':arm}
        if (folder/'completed.json').exists():
            completed=json.loads((folder/'completed.json').read_text())
            if completed['lineage']!=lineage or completed['latest_sha256']!=sha256(folder/'latest.pt'):
                raise ValueError('Frozen completed run identity/checkpoint differs')
            continue
        if (folder/'initial.pt').exists()and not resume:raise ValueError('Unfinished run needs explicit --resume')
        torch.manual_seed(config['seed']);random.seed(config['seed']);model=copy.deepcopy(source).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4);decoder=old.prior.base.PhysicalSpectrum().to(device)
        save_json(folder/'config.json',{**config,'arm':arm,'device':str(device)});save_json(folder/'lineage.json',lineage)
        np.save(folder/'sampling.npy',samples);np.save(folder/'reference_mix.npy',mix);snapshot=folder/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for path in files:shutil.copyfile(path,snapshot/path.name)
        start=0;progress={'data_seconds':0.,'compute_seconds':0.,'validation_seconds':0.,'prior_wall_seconds':0.};best=float('inf');best_step=0;metrics={};done_step=0
        if resume:
            state=torch.load(resume,map_location='cpu',weights_only=True)
            if state['lineage']!=lineage:raise ValueError('Resume identity/source/sampling changed')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];progress=state['progress'];best=state['best_score'];best_step=state['best_step'];metrics=state['validation'];restore_rng(state,device)
        else:
            metrics={'common_original':validate(model,validation,device,decoder,config,plans)}
            if overlay:metrics['strict_monitor']=validate(model,validation,device,decoder,config,plans,overlay)
            best=metrics['common_original']['total_with_aux'];save_json(folder/'validation_000000.json',metrics)
            torch.manual_seed(config['seed']+2)
            if str(device).startswith('cuda'):torch.cuda.manual_seed_all(config['seed']+2)
            save_checkpoint(folder/'initial.pt',model,opt,0,config,lineage,metrics,progress,best,0);shutil.copyfile(folder/'initial.pt',folder/'best.pt')
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        begun=time.perf_counter();prior_wall=progress.get('prior_wall_seconds',0.);done_step=start
        try:
            with (folder/'training.jsonl').open('a')as log:
                for step in range(start+1,config['steps']+1):
                    tick=time.perf_counter();packed=batch(training,samples[step-1],device,plans,mix[step-1],overlay if arm=='Q1'else None)
                    if str(device).startswith('cuda'):torch.cuda.synchronize()
                    progress['data_seconds']+=time.perf_counter()-tick;tick=time.perf_counter();model.train();opt.zero_grad(set_to_none=True)
                    loss,terms=objective(model,packed,decoder,config,strict=arm=='Q1')
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite strict loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite strict gradient')
                    opt.step();done_step=step
                    if str(device).startswith('cuda'):torch.cuda.synchronize()
                    progress['compute_seconds']+=time.perf_counter()-tick
                    for group in opt.param_groups:group['lr']=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*step/config['steps'])))
                    if step==1 or step%100==0:
                        record={'status':'RUNNING','arm':arm,'step':step,'loss':float(loss.detach()),**progress,'wall_seconds':prior_wall+time.perf_counter()-begun}
                        log.write(json.dumps(record)+'\n');log.flush();save_json(folder/'status.json',record);print(json.dumps(record),flush=True)
                    if step%config['eval_every']==0 or step==config['steps']:
                        tick=time.perf_counter();metrics={'common_original':validate(model,validation,device,decoder,config,plans)}
                        if overlay:metrics['strict_monitor']=validate(model,validation,device,decoder,config,plans,overlay)
                        progress['validation_seconds']+=time.perf_counter()-tick;progress['prior_wall_seconds']=prior_wall+time.perf_counter()-begun
                        improved=metrics['common_original']['total_with_aux']<best
                        if improved:best=metrics['common_original']['total_with_aux'];best_step=step
                        save_json(folder/f'validation_{step:06}.json',metrics);save_checkpoint(folder/'latest.pt',model,opt,step,config,lineage,metrics,progress,best,best_step)
                        if improved:shutil.copyfile(folder/'latest.pt',folder/'best.pt')
            result={'status':'COMPLETED','arm':arm,'steps':config['steps'],'seed':config['seed'],'record_exposures':config['steps']*config['batch_records'],
                    'acoustic_parameters':sum(p.numel()for p in model.parameters()),'fixed_planner_parameters':46309,'total_parameters':sum(p.numel()for p in model.parameters())+46309,
                    'final_primary_checkpoint':str(folder/'latest.pt'),'latest_sha256':sha256(folder/'latest.pt'),'best_secondary_step':best_step,'best_sha256':sha256(folder/'best.pt'),
                    'common_original_selection_score':best,'wall_seconds':prior_wall+time.perf_counter()-begun,'cache_seconds_shared_once':cache_seconds,**progress,
                    'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if str(device).startswith('cuda')else 0,
                    'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()if str(device).startswith('cuda')else 0,
                    'host_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'device':str(device),'lineage':lineage,'human_listening':'NOT_RUN'}
            save_json(folder/'completed.json',result);save_json(folder/'status.json',{'status':'COMPLETED','step':config['steps']})
        except BaseException:
            progress['prior_wall_seconds']=prior_wall+time.perf_counter()-begun;save_checkpoint(folder/'recovery.pt',model,opt,done_step,config,lineage,metrics,progress,best,best_step)
            (folder/'FAILED_STEP.md').write_text(traceback.format_exc());save_json(folder/'status.json',{'status':'FAILED','step':done_step});raise
        del model,opt,decoder
        if str(device).startswith('cuda'):torch.cuda.empty_cache()


def smoke(config,label='CPU',strict=False,device='cpu'):
    """Small real-data update and exact save/restore, never formal training."""
    old.setup();root=Path(config['output']);folder=root/'results'/'trainer_review';folder.mkdir(parents=True,exist_ok=True)
    model=source_model(config).to(device);training=old.Records(config,'train',True);plans=Plans(config);overlay=None
    if strict:
        validation=old.Records(config,'validation',True);overlay=Overlay(config['strict_data'],(training,validation))
    torch.manual_seed(config['seed']);random.seed(config['seed']);opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4);decoder=old.prior.base.PhysicalSpectrum().to(device)
    ix=list(range(min(4,len(training))));packed=batch(training,ix,device,plans,[False,True,False,True][:len(ix)],overlay)
    def update(m,o):
        m.train();o.zero_grad(set_to_none=True);loss,terms=objective(m,packed,decoder,config,strict=strict);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);o.step();return float(loss.detach())
    if str(device).startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter();first=update(model,opt)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-tick
    checkpoint=folder/f'{label}_smoke_checkpoint.pt';save_checkpoint(checkpoint,model,opt,1,config,{'smoke':True},{},{},0.,0)
    second=update(model,opt);expected=copy.deepcopy(model.state_dict());state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    restored=load_acoustic(checkpoint).to(device);restored_opt=torch.optim.AdamW(restored.parameters(),lr=config['learning_rate'],weight_decay=1e-4);restored_opt.load_state_dict(state['optimizer']);restore_rng(state,device);actual=update(restored,restored_opt)
    maximum=max(float((expected[k]-v).abs().max())for k,v in restored.state_dict().items());assert maximum==0 and second==actual
    result={'status':'COMPLETED','strict':strict,'device':str(device),'threads':1,'real_records':len(ix),'first_loss':first,'first_update_seconds':elapsed,
            'save_restore_next_loss_difference':abs(second-actual),'save_restore_state_maxdiff':maximum,'finite_gradients':all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters()),
            'acoustic_parameters':sum(p.numel()for p in model.parameters()),'total_with_fixed_planner':sum(p.numel()for p in model.parameters())+46309,
            'checkpoint':str(checkpoint),'scope':'FUNCTIONAL_SMOKE_NOT_FORMAL_TRAINING_OR_QUALITY_CANDIDATE','checkpoint_sha256':sha256(checkpoint),'source_sha256':sha256(Path(__file__)),
            'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if str(device).startswith('cuda')else 0,
            'torch':str(torch.__version__),'CUDA_runtime':torch.version.cuda,'GPU_name':torch.cuda.get_device_name()if str(device).startswith('cuda')else None}
    save_json(folder/f'{label}_smoke.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--arms',default='Q0,Q1');p.add_argument('--device',default='cuda:0');p.add_argument('--resume');p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-label',default='CPU');p.add_argument('--smoke-device',default='cpu');a=p.parse_args();c=json.loads(a.config.read_text())
    if a.smoke:smoke(c,a.smoke_label,a.arms=='Q1',a.smoke_device)
    else:train(c,a.arms.split(','),a.device,a.resume)
