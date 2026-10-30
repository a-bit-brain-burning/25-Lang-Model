"""Explicit routed39 loader, independent-language evaluation and listening.

Reference parameters are read only after ordinary text prediction, exclusively
for matched reference-duration diagnostics. Exported checkpoints reject the old
unrouted loader. Source frontend/planner/DSP and weights remain immutable.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse,html,json,math,random,shutil,time,resource
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from ..common import save_json,sha256
from ..frontend.romance_dispatch import RomanceFrontend
from ..models import romance_word_planner as wp
from ..models.romance_routed_acoustic import load_routed
from ..models.universal_acoustic import expand_controls
from ..models.english39_acoustic import render_curve,contract
from ..audio.dsp import write_wav
from .romance_evaluate import (physical_logpower,trajectory_metrics,boundary_regions,
    duration_metrics,parameter_metrics,natural_statistics,write_csv)
from .romance_listen import CHALLENGES
from .eu_quality_spectrum_basis import native_basis

CHALLENGES = {**CHALLENGES, 'ro':['Casa este aproape de noi.', 'Astăzi este o zi bună.']}

FEATURE_KEYS=('phonetic_features','context_features','ipa','words','plan_owner',
              'lexical_word_indices','internal_pause_mask','word_phone_ranges')


def load_system(language,lexicon,training,*,deploy_lexicon=None):
    training=Path(training)
    model,state=load_routed(training/'acoustic/best.pt')
    if state['config'].get('language')!=language:raise ValueError('Acoustic configuration language differs')
    if tuple(state['config'].get(k) for k in ('phonetic_dim','context_dim','spectral_dim','ap_bands','control_points','frame_period_ms'))!=(36,27,32,5,8,10):
        raise ValueError('Expected the frozen36/context27/8x39/10ms acoustic output')
    planner_path=training/'planner/best.pt';planner=wp.load_word_planner(planner_path)
    front=RomanceFrontend(language,lexicon)
    for s in (state,torch.load(planner_path,map_location='cpu',weights_only=True)):
        line=s['lineage']
        if (line['language']!=language or line['lexicon_sha256']!=front.lexicon_sha256
            or line['frontend_signature']!=front.signature):raise ValueError('Frozen frontend identity differs')
    if state['lineage']['planner_sha256']!=sha256(planner_path):raise ValueError('Wrong bound planner')
    trained=front
    if deploy_lexicon is not None:
        expanded=RomanceFrontend(language,deploy_lexicon)
        for word,phones in trained.entries.items():
            if expanded.entries.get(word)!=phones:raise ValueError('Extended lexicon changes trained pronunciation: '+word)
        front=expanded
    return {'frontend':front,'training_frontend':trained,'planner':planner,'model':model,
        'identity':{'language':language,'format':state['format'],'acoustic_sha256':sha256(training/'acoustic/best.pt'),
            'planner_sha256':sha256(planner_path),'training_lexicon_sha256':trained.lexicon_sha256,
            'runtime_lexicon_sha256':front.lexicon_sha256,'frontend_signature':front.signature,
            'acoustic_step':state['step'],'planner_step':planner.checkpoint_step,
            'duration_route':'rate_pause_only_v1','encoder_evaluations':2,
            'acoustic_parameters':sum(p.numel() for p in model.parameters()),
            'planner_parameters':sum(p.numel() for p in planner.parameters()),
            'reference_inputs_used':False,'postfilter':None,'DSP':contract('warped39')}}


@torch.inference_mode()
def predict(system,text):
    f=system['frontend'].encode(text);plan=system['planner'].predict_frontend(f)
    ff={**f,'context_features':np.column_stack((f['context_features'],plan['unit_plan'])).astype(np.float32)}
    p=system['model'].predict_features(ff)
    return f,plan,p


def quiet_declared_pauses(front,prediction):
    """Numerical SIL contract: no added gaps or lexical waveform processing.

    Optional interword pause durations stay model-predicted. Existing edge SIL
    durations are bounded by the existing450ms pause ceiling. Only explicit
    silence units become quiet/unvoiced; lexical parameters remain exact.
    Raw network outputs are saved independently for evaluation and attribution.
    """
    p={**prediction};c=p['controls'].copy();d=p['durations'].copy()
    sil=np.asarray(front['lexical_word_indices'])<0
    if np.any(np.asarray(front['phonetic_features'])[sil,0]<.5):raise ValueError('Nonlexical unit is not SIL')
    c[sil,:,1]=np.minimum(c[sil,:,1],-80*math.log(10)/10)
    c[sil,:,33:38]=0;c[sil,:,-1]=-12.
    if len(d):d[[0,-1]]=np.minimum(d[[0,-1]],45)
    p.update(controls=c,durations=d,curve=expand_controls(c,d))
    return p


def evaluate(system,data,output):
    data=Path(data);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    manifest=data/'manifest.jsonl';receipt=json.loads((data/'completed.json').read_text())
    if receipt['manifest_sha256']!=sha256(manifest):raise ValueError('Changed data')
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    values=[];details=[];plans=[]
    for row in rows:
        if row['split']!='validation':continue
        f,plan,p=predict(system,row['text'])  # Ordinary text first; no reference read.
        if sha256(row['record_path'])!=row['sha256']:raise ValueError('Reference cache changed')
        with np.load(row['record_path'],allow_pickle=False) as z:
            ref={k:z[k] for k in ('curve','target_controls','durations','phonetic_features','context_features','ipa')}
        if not np.array_equal(f['phonetic_features'],ref['phonetic_features']) or list(f['ipa'])!=ref['ipa'].tolist():
            raise ValueError('Validation phonetic topology changed')
        d=ref['durations'];o8=expand_controls(ref['target_controls'],d)
        aligned=expand_controls(p['controls'],d)
        regions,counts,_=boundary_regions(d,f);rpower=physical_logpower(ref['curve'])
        metrics={**duration_metrics(p['durations'],d,f),**natural_statistics(p['curve'])}
        delivered=quiet_declared_pauses(f,p)
        metrics.update({'delivered_'+k:v for k,v in duration_metrics(delivered['durations'],d,f).items()})
        with np.load(data/'strict'/(row['id']+'.npz')) as z:
            strict={k:z[k] for k in ('frame_voice','frame_voice_weight','frame_f0_valid')}
        metrics.update(parameter_metrics(aligned,ref['curve'],d,f,strict))
        for name,curve in (('P8_REFERENCE_DURATION',aligned),('O8_REFERENCE_DURATION',o8)):
            power=physical_logpower(curve)
            for region,mask in {'whole':np.ones(int(d.sum()),bool),**regions}.items():
                metric=trajectory_metrics(power,rpower,curve[:,1],ref['curve'][:,1],mask)
                details.append({'id':row['id'],'system':name,'region':region,'boundaries':counts.get(region),**metric})
                if name=='P8_REFERENCE_DURATION' and region=='whole':metrics.update({'whole_'+k:v for k,v in metric.items()})
        values.append({'id':row['id'],**metrics})
        measurement=wp.reference_measurements(f,{'durations':d,'curve':ref['curve'],'frame_f0_valid':strict['frame_f0_valid'],'split':'validation'})
        # Training baselines/normalization are frozen, not fitted from validation.
        root=Path(system['training'])
        raw,mask=wp.raw_plan(measurement,json.loads((root/'planner/baselines.json').read_text()))
        target,mask=wp.normalized_plan(raw,mask,system['planner'].normalization)
        for j,name in enumerate(wp.PLAN_NAMES):
            good=mask[:,j]>0
            plans.append({'id':row['id'],'field':name,'words':int(good.sum()),
                'MAE':float(abs(plan['word_plan'][good,j]-target[good,j]).mean()) if good.any() else None})
    def mean_table(rows,key):
        keys=[k for k in rows[0] if isinstance(rows[0][k],(float,int)) and not isinstance(rows[0][k],bool)]
        return {k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys if any(r.get(k) is not None for r in rows)}
    summary={'status':'COMPLETED','validation_recordings':len(values),'mean_sentence_metrics':mean_table(values,'id'),
        'plan_fields':{name:{'mean_sentence_MAE':float(np.mean([r['MAE'] for r in plans if r['field']==name and r['MAE'] is not None])),
                           'valid_words':sum(r['words'] for r in plans if r['field']==name)} for name in wp.PLAN_NAMES},
        'axes':'reference-duration frame reconstruction vs independent natural-duration error; raw network, no pause adapter',
        'test':'NOT_RUN','human_listening':'NOT_RUN','MOS':None,'identity':system['identity']}
    write_csv(out/'per_sentence.csv',values);write_csv(out/'transition_by_type.csv',details);write_csv(out/'plan_fields.csv',plans)
    save_json(out/'completed.json',summary);return summary


def export(system,training,lexicon,output,deploy_lexicon=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);training=Path(training)
    for kind in ('acoustic','planner'):
        source=training/kind/'best.pt';s=torch.load(source,map_location='cpu',weights_only=True)
        compact={k:v for k,v in s.items() if k in ('format','config','model','step','lineage','normalization','plan_version','feature_schema_sha256')}
        compact['source_training_checkpoint_sha256']=sha256(source)
        torch.save(compact,out/(kind+'.pt'))
    shutil.copyfile(lexicon,out/'lexicon_train.json')
    shutil.copyfile(deploy_lexicon or lexicon,out/'lexicon.json')
    np.save(out/'basis.npy',native_basis(32,700))
    assets=[{'path':p.name,'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())]
    save_json(out/'manifest.json',{'status':'COMPLETED','format':'tiny_romance_routed_bundle_v1','language':system['identity']['language'],
        'identity':system['identity'],'assets':assets,'duration_route':'rate_pause_only_v1',
        'pause_adapter':'quiet_declared_SIL_and_edge450ms_v1','added_parameters':0,
        'parameters_total':system['identity']['acoustic_parameters']+system['identity']['planner_parameters'],
        'DSP':'fixed existing C native warped700/32; no PCM postfilter',
        'source_training':str(training.resolve()),'human_listening':'NOT_RUN'})


def load_bundle(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest['format']!='tiny_romance_routed_bundle_v1':raise ValueError('Wrong bundle')
    for asset in manifest['assets']:
        if sha256(path/asset['path'])!=asset['sha256']:raise ValueError('Bundle asset changed')
    model,state=load_routed(path/'acoustic.pt');planner=wp.load_word_planner(path/'planner.pt')
    if tuple(state['config'].get(k) for k in ('phonetic_dim','context_dim','spectral_dim','ap_bands','control_points','frame_period_ms'))!=(36,27,32,5,8,10):
        raise ValueError('Exported acoustic shape/frame contract differs')
    planner_state=torch.load(path/'planner.pt',map_location='cpu',weights_only=True)
    if state['lineage']['planner_sha256']!=planner_state['source_training_checkpoint_sha256']:
        raise ValueError('Export binds a different planner')
    front=RomanceFrontend(manifest['language'],path/'lexicon.json')
    trained=RomanceFrontend(manifest['language'],path/'lexicon_train.json')
    identity=manifest['identity']
    if (state['config'].get('language')!=manifest['language']
        or state['source_training_checkpoint_sha256']!=identity['acoustic_sha256']
        or planner_state['source_training_checkpoint_sha256']!=identity['planner_sha256']):
        raise ValueError('Exported source checkpoint/language binding differs')
    for component in (state,planner_state):
        line=component['lineage']
        if (line['language']!=manifest['language'] or line['lexicon_sha256']!=trained.lexicon_sha256
            or line['frontend_signature']!=trained.signature):raise ValueError('Exported linguistic contract differs')
    if (front.lexicon_sha256!=identity['runtime_lexicon_sha256']
        or trained.lexicon_sha256!=identity['training_lexicon_sha256']
        or front.signature!=identity['frontend_signature']):raise ValueError('Runtime frontend identity differs')
    if not np.array_equal(np.load(path/'basis.npy',allow_pickle=False),native_basis(32,700)):
        raise ValueError('Wrong fixed native spectral basis')
    for word,p in trained.entries.items():
        if front.entries.get(word)!=p:raise ValueError('Runtime dictionary changes trained entry')
    if trained.lexicon_sha256!=state['lineage']['lexicon_sha256']:raise ValueError('Wrong training dictionary')
    return {'frontend':front,'training_frontend':trained,'model':model,'planner':planner,'identity':manifest['identity']}


def create_listen(system,data,output,*,synthetic_teacher=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir();(out/'reference').mkdir();(out/'network_raw').mkdir()
    rows=[json.loads(s) for s in (Path(data)/'manifest.jsonl').read_text().splitlines()]
    chosen=[r for r in rows if r['split']=='validation'][:6]
    cases=[{'id':f'VAL{i+1:02}','source_id':r['id'],'text':r['text'],'split':'validation','row':r} for i,r in enumerate(chosen)]
    challenges=[]
    train_texts={r['text'] for r in rows if r['split']=='train'}
    for text in CHALLENGES[system['identity']['language']]:
        try:system['frontend'].encode(text)
        except (ValueError,KeyError):continue
        if text not in train_texts:challenges.append(text)
    cases.extend({'id':f'NEW{i+1:02}','source_id':None,'text':text,'split':'constructed_text'} for i,text in enumerate(challenges))
    results=[];failures=[];sections=[];asr_audio=[];asr_text=[]
    for case in cases:
        try:
            started=time.perf_counter();f,plan,raw=predict(system,case['text']);pred=quiet_declared_pauses(f,raw)
            network_pcm,_=render_curve(raw['curve'],'warped39')
            write_wav(out/'network_raw'/(case['id']+'.wav'),network_pcm)
            pcm,_=render_curve(pred['curve'],'warped39');rel='raw/'+case['id']+'.wav';write_wav(out/rel,pcm)
            np.savez_compressed(out/'raw'/(case['id']+'.npz'),controls=pred['controls'],durations=pred['durations'],
                raw_controls=raw['controls'],raw_durations=raw['durations'],word_plan=plan['word_plan'],unit_plan=plan['unit_plan'],ipa=np.asarray(f['ipa']))
            if not np.array_equal(pred['controls'][f['lexical_word_indices']>=0],raw['controls'][f['lexical_word_indices']>=0]):raise AssertionError('Lexical values changed')
            sr=16000;peak=int(abs(pcm.astype(np.int32)).max());clipped=int((abs(pcm.astype(np.int32))>=32767).sum())
            results.append({'case_id':case['id'],'text':case['text'],'source_id':case['source_id'],
                'wave':rel,'sha256':sha256(out/rel),'seconds':len(pcm)/sr,'wall_seconds':time.perf_counter()-started,
                'peak':peak,'clipped_samples':clipped,'predicted_frames':int(pred['durations'].sum()),'reference_inputs':False,
                'split':case['split'],'independent_test':False})
            section='<section><h2>'+case['id']+'</h2><p>'+html.escape(case['text'])+'</p><p>模型普通文本推理</p><audio controls preload="none" src="'+rel+'"></audio>'
            asr_audio.append({'case_id':case['id'],'profile':'MODEL','wav':str((out/rel).resolve()),'sha256':sha256(out/rel)})
            if case.get('row'):
                # Reference copied only after synthesis; no model input leakage.
                refcase=case['row']
                original=next((refcase[k] for k in ('wav','source_wav_local','source_wav')
                               if refcase.get(k) and Path(refcase[k]).is_file()),None)
                if original:
                    from .romance_data import pcm16
                    ref,_=pcm16(original);refrel='reference/'+case['id']+'.wav';write_wav(out/refrel,ref)
                    save_json(out/'reference'/(case['id']+'.json'),{'source':str(original),'source_sha256':sha256(original),
                        'view':'OmniVoice synthetic teacher resampled to16kHz' if synthetic_teacher else 'real reference recording resampled to16kHz, not DSP synthesis',
                        'output_sha256':sha256(out/refrel),'used_for_model_prediction':False})
                    label='对应 OmniVoice 合成教师（16kHz副本）' if synthetic_teacher else '对应真人录音（16kHz副本）'
                    section+='<p>'+label+'</p><audio controls preload="none" src="'+refrel+'"></audio>'
                    asr_audio.append({'case_id':case['id'],'profile':'SYNTHETIC_TEACHER' if synthetic_teacher else 'REFERENCE','wav':str((out/refrel).resolve()),'sha256':sha256(out/refrel)})
                real=next((refcase[k] for k in ('original_wav','original_source_wav')
                           if refcase.get(k) and Path(refcase[k]).is_file()),None)
                if synthetic_teacher and real:
                    from .romance_data import pcm16
                    real_pcm,_=pcm16(real);realrel='reference/'+case['id']+'_real.wav';write_wav(out/realrel,real_pcm)
                    save_json(out/'reference'/(case['id']+'_real.json'),{'source':str(real),'source_sha256':sha256(real),
                        'view':'original real recording for comparison only','used_for_model_prediction':False})
                    section+='<p>对应原真人录音（仅比较，未参与训练）</p><audio controls preload="none" src="'+realrel+'"></audio>'
            asr_text.append({'case_id':case['id'],'text':case['text']});sections.append(section+'</section>')
        except (ValueError,KeyError) as error:failures.append({'id':case['id'],'text':case['text'],'status':'FAILED','reason':str(error)})
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+system['identity']['language']+'试听</title><style>body{max-width:900px;margin:2em auto;font:17px/1.7 sans-serif}section{padding:1em;border-bottom:1px solid #ddd}audio{width:100%}</style><h1>'+system['identity']['language']+'：新训练模型与参考</h1><p>固定DSP、自然预测时长。原音不参与推理；VAL为反复使用的开发集，不是独立测试。静音参数适配单独记录。尚未真人评分，无MOS结论。</p>'+''.join(sections)+'</html>'
    (out/'index.html').write_text(page)
    save_json(out/'results.json',results);save_json(out/'failures.json',failures)
    save_json(out/'asr_audio.json',asr_audio);save_json(out/'asr_cases.json',asr_text)
    save_json(out/'completed.json',{'status':'COMPLETED' if results else 'FAILED','audio_count':len(results),
        'failures':failures,'identity':system['identity'],'human_listening':'NOT_RUN','MOS':None})
    return results


def run(language,lexicon,training,data,output,deploy_lexicon=None,*,synthetic_teacher=False):
    torch.set_num_threads(1);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    system=load_system(language,lexicon,training,deploy_lexicon=deploy_lexicon);system['training']=str(training)
    # Verify an expanded dictionary on every training/development text, before
    # it is allowed to affect inference. No reference audio is needed here.
    if deploy_lexicon is not None:
        for line in (Path(data)/'manifest.jsonl').read_text().splitlines():
            text=json.loads(line)['text'];a=system['training_frontend'].encode(text);b=system['frontend'].encode(text)
            for k in FEATURE_KEYS:
                if not np.array_equal(a[k],b[k]):raise ValueError('Expanded lexicon changes IR: '+k)
    evaluate(system,data,root/'evaluation')
    export(system,training,lexicon,root/'export',deploy_lexicon)
    results=create_listen(system,data,root/'listen',synthetic_teacher=synthetic_teacher)
    loaded=load_bundle(root/'export');exact=[]
    for item in results:
        f,plan,p=predict(loaded,item['text']);p=quiet_declared_pauses(f,p);pcm,_=render_curve(p['curve'],'warped39')
        sr,ref=wavfile.read(root/'listen'/item['wave'])
        if sr!=16000 or not np.array_equal(pcm,ref):raise AssertionError('Exported PCM differs')
        exact.append(item['case_id'])
    save_json(root/'completed.json',{'status':'COMPLETED','language':language,'export_PCM_exact_cases':exact,
        'parameters':system['identity']['acoustic_parameters']+system['identity']['planner_parameters'],
        'human_listening':'NOT_RUN','test':'NOT_RUN','ARM':'NOT_RUN',
        'host_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--language',required=True);p.add_argument('--lexicon',required=True)
    p.add_argument('--deploy-lexicon');p.add_argument('--training',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--synthetic-teacher',action='store_true');a=p.parse_args()
    run(a.language,a.lexicon,a.training,a.data,a.output,a.deploy_lexicon,synthetic_teacher=a.synthetic_teacher)
