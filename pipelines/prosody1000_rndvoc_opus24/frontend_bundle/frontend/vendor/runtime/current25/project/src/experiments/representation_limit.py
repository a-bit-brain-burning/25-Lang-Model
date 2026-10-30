"""Frozen DSP representation factorial oracle diagnostic; no neural model changes."""
import argparse
import ctypes
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from scipy.fft import dct, idct
from scipy.signal import butter

from ..common import PROJECT, WORKSPACE, save_json, sha256
from ..audio.dsp import Stats, build_native, write_wav
from ..audio.extractor import read_audio, short_time_energy
from ..inference.evaluate import waveform_metrics, aggregate


class GenericDSPAdapter:
    """Same native ABI/math, generalized dim/bands fixed tables only.

    Physical frames: [F0 Hz, log waveform energy, DCT shape(dim-1), AP, VUV].
    All callers must provide real parameters, never neural logit F0/VUV.
    """
    def __init__(self, dim=24, bands=3, taps=128, gain=1., seed=1729, library=None):
        if dim not in (24, 32) or bands not in (3, 5):
            raise ValueError('Factorial diagnostic supports dim24/32 and AP3/5')
        self.dim, self.bands, self.taps, self.gain, self.seed = dim, bands, taps, gain, seed
        self.edges = [0,1000,4000,8000] if bands == 3 else [0,500,1500,3000,5000,8000]
        modes = np.zeros((dim-1, 513))
        modes[np.arange(dim-1),np.arange(1,dim)] = np.sqrt(513)
        cep = np.fft.irfft(.5*idct(modes,type=2,norm='ortho',axis=1),n=1024,axis=1)
        cep[:,1:512] *= 2
        self.basis = np.ascontiguousarray(cep[:,:taps].T,dtype=np.float32)
        self.bank = np.asarray([butter(2,x,fs=16000,output='sos')[0][[0,1,2,4,5]]
                                for x in self.edges[1:-1]],dtype=np.float32)
        self.library = ctypes.CDLL(str(library or build_native()))
        pf = ctypes.POINTER(ctypes.c_float)
        self.library.tetts_render.argtypes = [pf,ctypes.c_int,ctypes.c_int,ctypes.c_int,
            pf,pf,ctypes.c_int,ctypes.c_int,ctypes.c_uint32,ctypes.c_float,pf,
            ctypes.POINTER(ctypes.c_int16),ctypes.POINTER(Stats)]
        self.library.tetts_render.restype = ctypes.c_int
        self.library.tetts_stream_bytes.restype = ctypes.c_size_t

    @staticmethod
    def pointer(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def render(self, physical_frames, sample_count=None):
        frames = np.ascontiguousarray(physical_frames,dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self.dim+self.bands+2 or not np.isfinite(frames).all():
            raise ValueError('Physical frame budget mismatch')
        n = int(sample_count or len(frames)*160)
        floats, pcm, stats = np.empty(n,np.float32),np.empty(n,np.int16),Stats()
        start = time.perf_counter()
        code = self.library.tetts_render(self.pointer(frames),len(frames),self.dim,self.bands,
            self.pointer(self.basis),self.pointer(self.bank),self.taps,n,self.seed,self.gain,
            self.pointer(floats),pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),ctypes.byref(stats))
        if code:
            raise RuntimeError(f'Frozen DSP returned {code}')
        metrics = {k:getattr(stats,k) for k,_ in Stats._fields_}
        metrics.update(wall_seconds=time.perf_counter()-start,seconds=n/16000,
            basis_ROM_bytes=self.basis.nbytes,bank_ROM_bytes=self.bank.nbytes,
            physical_frame_bytes=frames.shape[1]*4,
            native_stream_state_bytes=int(self.library.tetts_stream_bytes()))
        metrics['RTF'] = metrics['wall_seconds']/metrics['seconds']
        return pcm, metrics


def compact_frames(f0, spectrum, ap, energy, dim, bands):
    n = len(energy)
    shape = dct(np.log(np.maximum(spectrum[:n],1e-12)),type=2,norm='ortho',axis=1,workers=1)[:,1:dim]/np.sqrt(513)
    edges = [0,1000,4000,8000] if bands == 3 else [0,500,1500,3000,5000,8000]
    hz = np.linspace(0,8000,513)
    safe = np.clip(ap[:n],.001,1-1e-12)
    compact_ap = []
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask = (hz>=lo)&((hz<=hi) if j==bands-1 else (hz<hi))
        compact_ap.append(10*np.log10(np.mean(safe[:,mask]**2,axis=1))/60)
    return np.ascontiguousarray(np.column_stack([f0[:n],np.log(np.maximum(energy,1e-12)),shape,
        np.column_stack(compact_ap),f0[:n]>0]),dtype=np.float32)


def spectral_metrics(frames, spectrum, ap, energy, dim, bands, source_f0=None):
    n = len(frames)
    logsp = np.log(np.maximum(spectrum[:n],1e-12))
    centered = logsp-logsp.mean(axis=1,keepdims=True)
    modes = np.zeros((n,513))
    modes[:,1:dim] = frames[:,2:dim+1]*np.sqrt(513)
    predicted = idct(modes,type=2,norm='ortho',axis=1)
    db = (predicted-centered)*10/np.log(10)
    active = energy > max(energy.max()/1000,1e-10)
    hz = np.linspace(0,8000,513)
    source_f0=frames[:,0] if source_f0 is None else source_f0[:n]
    result = {'envelope_shape_RMSE_db':float(np.sqrt(np.mean(db[active]**2))),
              'retained_F0_RMSE_Hz':float(np.sqrt(np.mean((frames[:,0]-source_f0)**2))),
              'retained_VUV_error':float(np.mean((frames[:,-1]>0)!=(source_f0>0)))}
    for label,lo,hi in [('low_0_1k',0,1000),('mid_1_4k',1000,4000),('high_4_8k',4000,8001)]:
        mask=(hz>=lo)&(hz<hi)
        result[f'envelope_{label}_RMSE_db'] = float(np.sqrt(np.mean(db[active][:,mask]**2)))
    edges = [0,1000,4000,8000] if bands == 3 else [0,500,1500,3000,5000,8000]
    approx = np.empty_like(ap[:n])
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask = (hz>=lo)&((hz<=hi) if j==bands-1 else (hz<hi))
        approx[:,mask] = (10**(3*frames[:,dim+1+j]))[:,None]
    result['AP_fullbin_RMSE_linear'] = float(np.sqrt(np.mean((approx[active]-ap[:n][active])**2)))
    return result


def compact_world_parameters(frames, original_spectrum, dim, bands):
    """Offline ablation only: compress shape/AP while keeping WORLD decoder fixed.

    Keep original WORLD log-spectrum DC because DSP waveform-energy replaces it
    in production. This isolates DCT/AP truncation, not DSP energy normalization.
    """
    n=len(frames);modes=np.zeros((n,513));modes[:,1:dim]=frames[:,2:dim+1]*np.sqrt(513)
    log_dc=np.log(np.maximum(original_spectrum[:n],1e-12)).mean(axis=1,keepdims=True)
    compact_sp=np.exp(idct(modes,type=2,norm='ortho',axis=1)+log_dc)
    edges=[0,1000,4000,8000] if bands==3 else [0,500,1500,3000,5000,8000]
    hz=np.linspace(0,8000,513);compact_ap=np.empty((n,513))
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(hz>=lo)&((hz<=hi) if j==bands-1 else (hz<hi))
        compact_ap[:,mask]=(10**(3*frames[:,dim+1+j]))[:,None]
    return np.ascontiguousarray(compact_sp),np.ascontiguousarray(compact_ap)


def waveform_spectral_metrics(pcm, original, energy):
    n = len(energy)
    def stft(x):
        padded = np.pad(x,(320,320))
        frames = np.lib.stride_tricks.sliding_window_view(padded,640)[::160][:n]
        return np.abs(np.fft.rfft(frames*np.hanning(640),n=1024,axis=1))**2
    ref, synth = stft(original),stft(pcm.astype(np.float64)/32768)
    # Floor per utterance at 60 dB below its peak to avoid silence dominating.
    floor = max(float(ref.max())*1e-6,1e-12)
    db = 10*np.log10(np.maximum(synth,floor)/np.maximum(ref,floor))
    active = energy > max(energy.max()/1000,1e-10)
    hz = np.linspace(0,8000,513)
    return {'waveform_STFT_active_RMSE_db':float(np.sqrt(np.mean(db[active]**2))),
            'waveform_STFT_high4k_RMSE_db':float(np.sqrt(np.mean(db[active][:,hz>=4000]**2)))}


def listening_pack(output, ids, systems):
    rng = random.Random(1729)
    root = output/'blind_listening'
    root.mkdir(exist_ok=True)
    key, rating_rows, html = [],[],['<!doctype html><meta charset="utf-8"><title>Representation paired blind listening</title>',
        '<h1>Representation paired listening</h1><p>Same sentence, randomized variants. Rate naturalness 1–5 without inspecting key. Include headphones/device and comments.</p>']
    for index,identity in enumerate(ids,1):
        order=list(systems);rng.shuffle(order)
        html.append(f'<h2>Sentence {index:02}</h2>')
        for j,system in enumerate(order):
            label=f'S{index:02}_{chr(65+j)}'
            shutil.copyfile(output/'samples'/identity/(system+'.wav'),root/(label+'.wav'))
            key.append({'label':label,'id':identity,'system':system})
            rating_rows.append([label,'','','','',''])
            html.append(f'<p>{label}<br><audio controls src="{label}.wav"></audio></p>')
    (root/'index.html').write_text('\n'.join(html))
    save_json(output/'blind_key.json',key)
    with (root/'ratings.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['label','listener','naturalness_1_to_5','intelligibility_1_to_5','device','comment']);writer.writerows(rating_rows)


def summarize_paired(output):
    rows=json.loads((output/'per_utterance_metrics.json').read_text())
    by={(r['id'],r['system']):r for r in rows}
    ids=json.loads((output/'config.json').read_text())['ids']
    rng=np.random.default_rng(1729);draws=rng.integers(0,len(ids),(20000,len(ids)))
    comparisons={}
    for a,b in [('d24_ap3','d32_ap3'),('d24_ap3','d24_ap5'),('d24_ap3','d32_ap5'),
                ('d24_ap3','world_d24_ap3'),('world_d24_ap3','world_full'),('world_d32_ap5','world_full')]:
        deltas=np.array([by[i,b]['STOI_original_timeline']-by[i,a]['STOI_original_timeline'] for i in ids])
        comparisons[f'{b}_minus_{a}']={'mean_STOI_delta':float(deltas.mean()),
            'paired_utterance_bootstrap_95CI':np.percentile(deltas[draws].mean(axis=1),[2.5,97.5]).tolist(),
            'utterances_improved':int(np.sum(deltas>0)),'utterance_count':len(ids)}
    save_json(output/'paired_statistics.json',{'seed':1729,'bootstrap_draws':20000,
        'comparisons':comparisons,'MOS':'not measured; STOI difference cannot be converted to MOS'})
    resources={}
    for system in ('d24_ap3','d32_ap3','d24_ap5','d32_ap5'):
        stats=[by[i,system]['DSP'] for i in ids]
        seconds=sum(s['seconds'] for s in stats)
        macs={k:sum(s[k] for s in stats)/seconds for k in ('bank_mac','fir_mac','basis_mac','recursion_mac')}
        resources[system]={'MAC_per_audio_second':macs,'total_counted_MAC_per_audio_second':sum(macs.values()),
            'exp_calls_per_audio_second':sum(s['exp_calls'] for s in stats)/seconds,
            'sqrt_calls_per_audio_second':sum(s['sqrt_calls'] for s in stats)/seconds,
            'host_weighted_DSP_RTF':sum(s['wall_seconds'] for s in stats)/seconds,
            'native_stream_state_bytes':stats[0]['native_stream_state_bytes'],
            'basis_ROM_bytes':stats[0]['basis_ROM_bytes'],'bank_ROM_bytes':stats[0]['bank_ROM_bytes'],
            'physical_frame_bytes':stats[0]['physical_frame_bytes'],
            '200MHz_optimistic_MAC1cycle_fraction':sum(macs.values())/200e6,
            'actual_200MHz_hardware_test':False,'warning':'Counted MACs omit branches/load/store/pow/sqrt/RNG. Host RTF cannot predict MCU runtime.'}
    save_json(output/'resources_preliminary.json',resources)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=PROJECT/'outputs/checkpoints/acoustic_6h_stress/data_view')
    parser.add_argument('--output',type=Path,default=PROJECT/'outputs/bottleneck_20261014/representation')
    parser.add_argument('--limit',type=int,default=20)
    parser.add_argument('--audit-cache',type=int,default=3,help='Fresh WORLD re-extraction for first N cached originals')
    args=parser.parse_args()
    import pyworld as pw
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    rows_by_id={r['id']:r for r in map(json.loads,(args.data/'manifest.jsonl').read_text().splitlines())}
    ids=json.loads((args.data/'split.json').read_text())['validation'][:args.limit]
    systems=['original','world_full','d24_ap3','d32_ap3','d24_ap5','d32_ap5',
             'world_d24_ap3','world_d32_ap5']
    library=build_native(output/'native/libcompact_dsp.so')
    vocoders={f'd{d}_ap{b}':GenericDSPAdapter(d,b,library=library) for d in (24,32) for b in (3,5)}
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'ids':ids,'split':'validation',
        'data':str(args.data.resolve()),'manifest_sha256':sha256(args.data/'manifest.jsonl'),
        'split_sha256':sha256(args.data/'split.json'),'systems':systems,'checkpoint_snapshots':{},
        'diagnostic_true_duration':False,'residual':None,'DSP':{'taps':128,'gain':1.,'seed':1729,
        'rate':16000,'hop':160,'source_sha256':sha256(PROJECT/'src/audio/native.c')},
        'energy':'640 sample centered Hann mean-square; gain1 shared','factorial':'dim24/32 x AP3/5; F0,VUV,energy identical',
        'WORLD_full_is_offline_diagnostic_only':True,'MOS':'not measured; no conversion from WER/STOI'}
    save_json(output/'config.json',config)
    rows=[];sources=[];audits=[];start=time.monotonic()
    for number,identity in enumerate(ids,1):
        row=rows_by_id[identity];directory=output/'samples'/identity;directory.mkdir(parents=True,exist_ok=True)
        original=read_audio(row['source_audio']);n=int(np.ceil(len(original)/160));energy=short_time_energy(original,n)
        cache=Path(row['source_audio']).parent/'world_full.npz'
        if cache.exists():
            with np.load(cache) as z:f0,sp,ap=z['f0'].copy(),z['spectrum'].copy(),z['aperiodicity'].copy()
            source={'id':identity,'wav':row['source_audio'],'wav_sha256':sha256(row['source_audio']),
                'full_world_cache':str(cache),'full_world_cache_sha256':sha256(cache),'fresh_extraction':False}
        else:
            f0,t=pw.harvest(original,16000,f0_floor=50,f0_ceil=400,frame_period=10)
            f0=pw.stonemask(original,f0,t,16000);sp=pw.cheaptrick(original,f0,t,16000,f0_floor=50,fft_size=1024)
            ap=pw.d4c(original,f0,t,16000,fft_size=1024)
            np.savez_compressed(directory/'world_full.npz',f0=f0,spectrum=sp,aperiodicity=ap)
            source={'id':identity,'wav':row['source_audio'],'wav_sha256':sha256(row['source_audio']),'fresh_extraction':True}
        sources.append(source)
        if cache.exists() and number<=args.audit_cache:
            fresh_f0,fresh_times=pw.harvest(original,16000,f0_floor=50,f0_ceil=400,frame_period=10)
            fresh_f0=pw.stonemask(original,fresh_f0,fresh_times,16000)[:n]
            joint=(fresh_f0>0)&(f0[:n]>0)
            audit={'id':identity,'cached_voiced_fraction':float(np.mean(f0[:n]>0)),
                'fresh_voiced_fraction':float(np.mean(fresh_f0>0)),
                'VUV_disagreement':float(np.mean((f0[:n]>0)!=(fresh_f0>0))),
                'joint_voiced_F0_RMSE_Hz':float(np.sqrt(np.mean((fresh_f0[joint]-f0[:n][joint])**2))),
                'fresh_extract_config':{'method':'harvest+stonemask','floor':50,'ceil':400,'frame_period_ms':10}}
            audits.append(audit);save_json(output/'cache_fresh_audit.json',audits)
        reference=json.loads(json.dumps(source));reference.update(text=row['text'],split='validation')
        save_json(directory/'reference.json',reference)
        packed={name:compact_frames(f0,sp,ap,energy,v.dim,v.bands) for name,v in vocoders.items()}
        np.savez_compressed(directory/'oracle_frames.npz',**packed)
        sounds={'original':np.rint(original*32768).clip(-32768,32767).astype(np.int16)}
        full=pw.synthesize(np.ascontiguousarray(f0),np.ascontiguousarray(sp),np.ascontiguousarray(ap),16000,10)
        full=full[:len(original)] if len(full)>=len(original) else np.pad(full,(0,len(original)-len(full)))
        sounds['world_full']=np.rint(full*32768).clip(-32768,32767).astype(np.int16)
        for name,vocoder in vocoders.items():
            pcm,dsp=vocoder.render(packed[name],len(original));sounds[name]=pcm
            metric={'id':identity,'system':name,**spectral_metrics(packed[name],sp,ap,energy,vocoder.dim,vocoder.bands,f0),
                **waveform_metrics(pcm,original),**waveform_spectral_metrics(pcm,original,energy),'DSP':dsp}
            rows.append(metric);save_json(directory/(name+'.metrics.json'),metric)
        for name in ('d24_ap3','d32_ap5'):
            vocoder=vocoders[name]
            csp,cap=compact_world_parameters(packed[name],sp,vocoder.dim,vocoder.bands)
            y=pw.synthesize(np.ascontiguousarray(f0[:n]),csp,cap,16000,10)
            y=y[:len(original)] if len(y)>=len(original) else np.pad(y,(0,len(original)-len(y)))
            sounds['world_'+name]=np.rint(y*32768).clip(-32768,32767).astype(np.int16)
            metric={'id':identity,'system':'world_'+name,
                'offline_only':True,'original_WORLD_log_spectrum_DC_retained':True,
                **waveform_metrics(sounds['world_'+name],original),
                **waveform_spectral_metrics(sounds['world_'+name],original,energy)}
            rows.append(metric);save_json(directory/('world_'+name+'.metrics.json'),metric)
        for name in ('original','world_full'):
            metric={'id':identity,'system':name,**waveform_metrics(sounds[name],original),
                **waveform_spectral_metrics(sounds[name],original,energy)}
            rows.append(metric);save_json(directory/(name+'.metrics.json'),metric)
        for name,pcm in sounds.items():write_wav(directory/(name+'.wav'),pcm)
        save_json(output/'per_utterance_metrics.json',rows)
        save_json(output/'summary.json',{'systems':aggregate(rows),'completed':number,'total':len(ids),'wall_seconds':time.monotonic()-start})
        save_json(output/'sources.json',sources)
        print(json.dumps({'completed':number,'id':identity,'wall_seconds':time.monotonic()-start}),flush=True)
    listening_pack(output,ids,systems)
    summarize_paired(output)


if __name__=='__main__':main()
