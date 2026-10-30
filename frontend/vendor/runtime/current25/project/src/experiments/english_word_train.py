"""Matched GPU English continuation, explicit pauses, and word-plan guidance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(_key,'1')
import argparse,copy,hashlib,json,math,random,resource,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..common import save_json,sha256,PROJECT
from ..models.universal_acoustic import UniversalAcousticModel,load_acoustic,FORMAT
from ..frontend.english_clarity import supervision_groups
from ..frontend.english_word_prosody import PLAN_VERSION,PLAN_NAMES,plan_for_units,expanded_frontend
from ..training.clarity_objective import clarity_terms,weighted_clarity
from . import english_clarity_train as prior
from .phrase_gpu_execution import device_objective


def setup():
    prior.configure();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def upgrade(source):
    model=UniversalAcousticModel({**source.config,'context_dim':27,'english_word_plan_version':PLAN_VERSION,
                                  'english_topology':'english_optional_word_pause_v1'},source.means,source.scales)
    state=copy.deepcopy(source.state_dict());weight=torch.zeros_like(model.input.weight);weight[:,:54]=state['input.weight'];state['input.weight']=weight
    model.load_state_dict(state);return model.eval()


class Records:
    """Precache once; retain frame VUV only for training, not redundant spectra."""
    def __init__(self,config,split,pause=False):
        self.rows=[];self.arrays=[];self.pause=pause
        if pause:
            root=Path(config['pause_data']);receipt=json.loads((root/'completed.json').read_text())
            if receipt['status']!='COMPLETED':raise ValueError('Pause data incomplete')
            rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines() if s.strip()]
            rows=[r for r in rows if r['split']==split]
            for row in rows:
                path=Path(row['record_path']);path=path if path.is_absolute() else root/path
                if row.get('sha256') and sha256(path)!=row['sha256']:raise ValueError('Pause data changed')
                with np.load(path,allow_pickle=False) as z:
                    a={k:z[k] for k in ('phone_ids','text_features','stress','phonetic_features','context_features','edge_sil_mask',
                        'target_controls','control_mask','target_duration','word_indices','internal_pause_mask')}
                    a['target_curve']=z['target_curve'][:,-1:].copy()
                a['_groups']=supervision_groups(a['phone_ids'],a['text_features'],a['stress']);self.rows.append(row);self.arrays.append(a)
        else:
            view=prior.Records(config['data_dir'],split)
            for i in range(len(view)):
                row,a=view[i];keep={k:a[k] for k in ('phone_ids','text_features','stress','phonetic_features','context_features','target_controls','control_mask','target_duration','_groups')}
                keep['edge_sil_mask']=a.get('edge_sil_mask',np.r_[True,np.zeros(len(a['phone_ids'])-2,bool),True]);keep['target_curve']=a['target_curve'][:,-1:].copy()
                keep['internal_pause_mask']=np.zeros(len(a['phone_ids']),bool);self.rows.append(row);self.arrays.append(keep)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[int(i)],self.arrays[int(i)]


class Plans:
    def __init__(self,config):
        r=Path(config['output'])/'data/plans';receipt=json.loads((r/'predicted_completed.json').read_text())
        if receipt['status']!='COMPLETED':raise ValueError('Predicted plans incomplete')
        self.root=r;self.cache={};self.receipt=receipt
    def get(self,uid):
        if uid not in self.cache:
            pred=np.load(self.root/'predicted_units'/(uid+'.npy'))
            ref=np.load(self.root/'reference_units'/(uid+'.npy'));mask=np.load(self.root/'reference_unit_masks'/(uid+'.npy'))
            self.cache[uid]=(pred,ref,mask)
        return self.cache[uid]


def batch(view,indices,device,plans=None,reference_rows=None):
    indices=list(indices);f,t,m,d,g=prior.batch(view,indices)
    width=f['valid_mask'].shape[1];pause=torch.zeros_like(f['valid_mask'])
    if plans:extra=torch.zeros(*f['valid_mask'].shape,5)
    for j,i in enumerate(indices):
        a=view[i][1];n=len(a['phone_ids']);pause[j,:n]=torch.from_numpy(a['internal_pause_mask'])
        if plans:
            pred,ref,mask=plans.get(view.rows[i]['id']);p=np.where(mask>.5,ref,pred) if reference_rows is not None and reference_rows[j] else pred
            if p.shape!=(n,5):raise ValueError('Plan topology differs')
            extra[j,:n]=torch.from_numpy(p)
    f['_internal_pause_mask']=pause
    if plans:f['context_features']=torch.cat((f['context_features'],extra),-1)
    return tuple({k:v.to(device) for k,v in x.items()} if isinstance(x,dict) else x.to(device) for x in (f,t,m,d,g))


def objective(model,packed,decoder,config,all_physical=False):
    f,t,m,d,g=packed;out=model({k:f[k] for k in prior.base.TEXT_INPUT_KEYS})
    common,terms=device_objective(out,t,m,d,f,model.scales,decoder,10000000 if all_physical else config['physical_samples'],model.voiced_log_prior,'control')
    clarity=clarity_terms(out,t,m,d,f,model.scales,g);loss=common+weighted_clarity(clarity,config['weights'])
    pause=f['_internal_pause_mask']&f['valid_mask'];positive=pause&(d>0);negative=pause&~positive
    avg=lambda value,mask:(value*mask).sum()/mask.sum().clamp_min(1)
    gate=F.binary_cross_entropy_with_logits(out['edge_nonzero_logits'],(d>0).float(),reduction='none')
    gate_loss=.5*avg(gate,positive)+.5*avg(gate,negative)
    duration=F.smooth_l1_loss(torch.log1p(out['positive_frames']),torch.log1p(d),reduction='none',beta=.5)
    pause_duration=avg(duration,positive)
    loss=loss+.05*gate_loss+.2*pause_duration
    metrics={**terms,**{'clarity_'+k:v for k,v in clarity.items()},'pause_gate':gate_loss,'positive_pause_duration':pause_duration,
        'pause_positive_recall':avg((out['edge_nonzero_logits']>=0).float(),positive),'pause_false_positive':avg((out['edge_nonzero_logits']>=0).float(),negative),
        'positive_pause_MAE_ms':10*avg((out['positive_frames']-d).abs(),positive),'total_with_aux':loss}
    return loss,metrics


@torch.inference_mode()
def validate(model,view,device,decoder,config,plans):
    model.eval();sums={};start=time.perf_counter()
    for first in range(0,len(view),config['batch_records']):
        ix=list(range(first,min(len(view),first+config['batch_records'])));loss,metrics=objective(model,batch(view,ix,device,plans),decoder,config,True)
        for k,v in metrics.items():sums[k]=sums.get(k,0.)+float(v)*len(ix)
    return {**{k:v/len(view) for k,v in sums.items()},'recordings':len(view),'axis':'reference-time controls; predicted text plan only','validation_seconds':time.perf_counter()-start}


def save(path,model,opt,step,config,lineage,metrics):
    state={'format':FORMAT,'config':model.config,'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'training_config':config,'lineage':lineage,'validation':metrics,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python_rng':random.getstate()}
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def train(config,arms,device='cuda:0',resume=None):
    setup();root=Path(config['output']);source=load_acoustic(config['parent'])
    if sha256(config['parent'])!=config['parent_sha256']:raise ValueError('Source checkpoint changed')
    views={};plans=None
    for arm in arms:
        pause=arm!='L0'
        if pause not in views:
            tick=time.perf_counter();views[pause]=(Records(config,'train',pause),Records(config,'validation',pause));print('cached',pause,len(views[pause][0]),time.perf_counter()-tick,flush=True)
        training,validation=views[pause]
        if arm=='P1' and plans is None:plans=Plans(config)
        active_plans=plans if arm=='P1' else None
        rng=np.random.default_rng(config['seed']);samples=np.stack([rng.choice(len(training),config['batch_records'],replace=False) for _ in range(config['steps'])]);mix=np.random.default_rng(config['seed']+1).random((config['steps'],config['batch_records']))<config['reference_plan_probability']
        folder=root/'training'/arm;folder.mkdir(exist_ok=True)
        files=[Path(__file__),PROJECT/'src/frontend/english_pause_units.py',PROJECT/'src/frontend/english_word_prosody.py',PROJECT/'src/models/english_word_planner.py',PROJECT/'src/training/english_pause_targets.py',PROJECT/'src/training/clarity_objective.py',PROJECT/'src/models/universal_acoustic.py',PROJECT/'src/experiments/phrase_gpu_execution.py']
        lineage={'source':config['parent_sha256'],'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'sources':{str(p):sha256(p) for p in files},'sample_sha256':hashlib.sha256(samples.astype('<i8').tobytes()).hexdigest(),'record_ids_sha256':hashlib.sha256('\n'.join(r['id'] for r in training.rows).encode()).hexdigest(),'pause_data_sha256':sha256(Path(config['pause_data'])/'completed.json') if pause else None,'plans':plans.receipt if arm=='P1' else None,'arm':arm,'test_loaded':False}
        if (folder/'completed.json').exists():continue
        if (folder/'initial.pt').exists() and not resume:raise ValueError('Existing run requires explicit resume')
        torch.manual_seed(config['seed']);random.seed(config['seed']);model=(upgrade(source) if arm=='P1' else copy.deepcopy(source)).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4);decoder=prior.base.PhysicalSpectrum().to(device);start=0
        if resume:
            state=torch.load(resume,map_location='cpu',weights_only=True)
            if state['lineage']!=lineage:raise ValueError('Resume lineage changed')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
            if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.save(folder/'sampling.npy',samples);save_json(folder/'config.json',{**config,'arm':arm,'device':device});save_json(folder/'lineage.json',lineage)
        snap=folder/'source_snapshot';snap.mkdir(exist_ok=True)
        for p in files:shutil.copy2(p,snap/p.name)
        metrics=validate(model,validation,device,decoder,config,active_plans);best=metrics['total_with_aux'];best_step=start
        if not resume:save(folder/'initial.pt',model,opt,0,config,lineage,metrics);save(folder/'best.pt',model,opt,0,config,lineage,metrics)
        if resume:
            prior_best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            best=prior_best['validation']['total_with_aux'];best_step=prior_best['step']
        else:
            torch.manual_seed(config['seed']+2)
            if str(device).startswith('cuda'):torch.cuda.manual_seed_all(config['seed']+2)
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        begun=time.perf_counter();data_time=compute_time=validation_time=0.;done_step=start
        save_json(folder/'status.json',{'status':'RUNNING','step':start})
        try:
            with (folder/'training.jsonl').open('a') as log:
                for step in range(start+1,config['steps']+1):
                    tick=time.perf_counter();packed=batch(training,samples[step-1],device,active_plans,mix[step-1] if active_plans else None)
                    if str(device).startswith('cuda'):torch.cuda.synchronize()
                    data_time+=time.perf_counter()-tick;tick=time.perf_counter();model.train();opt.zero_grad(set_to_none=True);loss,metrics_step=objective(model,packed,decoder,config)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                    opt.step();done_step=step
                    if str(device).startswith('cuda'):torch.cuda.synchronize()
                    compute_time+=time.perf_counter()-tick
                    for group in opt.param_groups:group['lr']=config['learning_rate']*(.15+.85*.5*(1+math.cos(math.pi*step/config['steps'])))
                    if step==1 or step%100==0:
                        record={'step':step,'loss':float(loss.detach()),'data_seconds':data_time,'compute_seconds':compute_time,'wall_seconds':time.perf_counter()-begun}
                        log.write(json.dumps(record)+'\n');log.flush();print(arm,record,flush=True);save_json(folder/'status.json',{'status':'RUNNING',**record})
                    if step%config['eval_every']==0 or step==config['steps']:
                        tick=time.perf_counter();metrics=validate(model,validation,device,decoder,config,active_plans);validation_time+=time.perf_counter()-tick;save_json(folder/f'validation_{step:06d}.json',metrics);save(folder/'latest.pt',model,opt,step,config,lineage,metrics)
                        if metrics['total_with_aux']<best:best=metrics['total_with_aux'];best_step=step;save(folder/'best.pt',model,opt,step,config,lineage,metrics)
            receipt={'status':'COMPLETED','steps':config['steps'],'seed':config['seed'],'parameters':sum(p.numel() for p in model.parameters()),'record_exposures':config['steps']*config['batch_records'],'latest_sha256':sha256(folder/'latest.pt'),'best_step':best_step,'wall_seconds':time.perf_counter()-begun,'data_seconds':data_time,'compute_seconds':compute_time,'validation_seconds':validation_time,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,'host_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'lineage':lineage,'device':device}
            save_json(folder/'completed.json',receipt);save_json(folder/'status.json',{'status':'COMPLETED','step':config['steps']})
        except BaseException:
            save(folder/'recovery.pt',model,opt,done_step,config,lineage,metrics);(folder/'FAILED_STEP.md').write_text(traceback.format_exc());save_json(folder/'status.json',{'status':'FAILED','step':done_step});raise
        del model,opt,decoder
        if str(device).startswith('cuda'):torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--arms',default='L0,P0,P1');p.add_argument('--device',default='cuda:0');p.add_argument('--resume');args=p.parse_args();train(json.loads(args.config.read_text()),args.arms.split(','),args.device,args.resume)
