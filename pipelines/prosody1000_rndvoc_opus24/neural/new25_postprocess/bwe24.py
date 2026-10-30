"""Independent original Model Opus-BWE backend: PCM16 mono16k -> mono24k.

No GTCRN, codec encode/decode, gain normalization or RNDVoC weight changes.
Original generated C weights, feature extraction and BWE arithmetic are kept.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
from scipy.signal import resample_poly
from .chain import run_pcm16_stages


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantize(pcm):
    value = np.asarray(pcm)
    if value.ndim != 1 or not len(value) or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
        raise ValueError('Expected finite nonempty mono floating PCM')
    # Exactly the PCM16 boundary used by the existing RNDVoC listening writer.
    return np.rint(np.clip(value.astype(np.float32), -1, 32767/32768)*32768).astype(np.int16)


def save_wav(path, pcm, sample_rate):
    if pcm.dtype != np.int16 or pcm.ndim != 1 or sample_rate not in (16000, 24000):
        raise ValueError('Expected mono PCM16 and an explicit16k/24k sample rate')
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sample_rate)
        f.writeframes(pcm.astype('<i2', copy=False).tobytes())


class OpusBWE24:
    input_sample_rate = 16000
    output_sample_rate = 24000
    learned_parameters = 367890
    streaming = False

    def __init__(self, assets=None, cache=None):
        root = Path(__file__).resolve().parent
        self.assets = Path(assets or root/'assets/opus_bwe').resolve()
        manifest = json.loads((root/'assets/BWE_ASSET_MANIFEST.json').read_text())
        for name, expected in manifest['files'].items():
            p = self.assets/name
            if p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
                raise ValueError('Original Model BWE asset changed: '+name)
        fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()+platform.platform().encode()).hexdigest()
        self.cache = Path(cache or Path(tempfile.gettempdir())/('rndvoc-bwe24-'+fingerprint[:20])).resolve()
        if self.cache.is_relative_to(self.assets):
            raise ValueError('Build/cache must be outside frozen assets')
        self.cache.mkdir(parents=True,exist_ok=True)
        self.manifest = manifest
        self.binary = self._build()
        self.identity = {'format':'original_model_opus_bwe24_v1','model_repository_commit':manifest['model_commit'],
                         'bwe_model_checkpoint_sha256':'a6c35c3efc53fc1b125dee1724298e2aa88c72da75bd44bdde77ebac0ce64fd0',
                         'binary_sha256':sha(self.binary),'learned_parameters':self.learned_parameters,
                         'input_sample_rate':16000,'intermediate_sample_rate':48000,'output_sample_rate':24000,
                         'delay_compensation_samples_48k':34,'flush_input_samples':640,
                         'normalization':'none','denoiser_enabled':False,'codec_encode_decode':False,
                         'streaming':False,'quality_pass':False,'whole_process_ram_pass':False}

    def _build(self):
        binary = self.cache/'build/bwe_demo'
        with (self.cache/'compile.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if binary.exists():
                return binary
            start = time.monotonic(); build = binary.parent; build.mkdir(exist_ok=True)
            cmake = shutil.which('cmake')
            log_path = self.cache/'build.log'
            with log_path.open('xb') as log:
                if cmake:
                    commands = [[cmake,'-S',str(self.assets),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                                 '-DPython3_EXECUTABLE='+sys.executable],
                                [cmake,'--build',str(build),'--target','bwe_demo','--parallel','2']]
                    for command in commands:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,180-(time.monotonic()-start)))
                    backend = 'original CMake AUTO'
                else:
                    # Same source list and portable scalar definitions as the
                    # original CMake build; no dependency installation needed.
                    compiler = shutil.which('cc') or shutil.which('gcc')
                    if not compiler:raise RuntimeError('A C compiler is required')
                    generated = build/'generated/bbwenet_data.c'; generated.parent.mkdir(exist_ok=True)
                    subprocess.run([sys.executable,str(self.assets/'cmake/unpack_weights.py'),
                                    str(self.assets/'weights/bbwenet_data.c.gz'),str(generated)],
                                   check=True,stdout=log,stderr=subprocess.STDOUT,timeout=20)
                    definitions = ['OPUS_BUILD','ENABLE_OSCE','ENABLE_OSCE_BWE','DISABLE_LACE','DISABLE_NOLACE',
                                   'DISABLE_DEBUG_FLOAT','VAR_ARRAYS','HAVE_LRINT','HAVE_LRINTF','SKIP_CONFIG_H',
                                   'SUPPRESS_PERF_WARNINGS','OPUS_BWE_SCALAR']
                    common = [compiler,'-O3','-DNDEBUG','-std=gnu99','-ffunction-sections','-fdata-sections']
                    common += ['-D'+x for x in definitions]
                    common += ['-I'+str(self.assets/'upstream'/x) for x in ['', 'include','dnn','celt','silk']]
                    names = ['dnn/bwe_demo.c','dnn/osce.c','dnn/osce_features.c','dnn/nndsp.c','dnn/nnet.c',
                             'dnn/nnet_default.c','dnn/parse_lpcnet_weights.c','dnn/freq.c','dnn/lpcnet_tables.c',
                             'celt/kiss_fft.c','celt/pitch.c']
                    sources = [self.assets/'upstream'/x for x in names]+[generated]
                    def compile_one(item):
                        index,source=item; target=build/(str(index)+'.o')
                        run=subprocess.run(common+['-c',str(source),'-o',str(target)],capture_output=True,
                                           timeout=max(1,170-(time.monotonic()-start)))
                        if run.returncode:raise RuntimeError(run.stderr.decode(errors='replace')[-3000:])
                        return target
                    with ThreadPoolExecutor(max_workers=2) as pool:objects=list(pool.map(compile_one,enumerate(sources)))
                    linker = '-Wl,-dead_strip' if sys.platform=='darwin' else '-Wl,--gc-sections'
                    subprocess.run([compiler,*map(str,objects),linker,'-lm','-o',str(binary)],check=True,
                                   stdout=log,stderr=subprocess.STDOUT,timeout=20)
                    backend = 'equivalent original scalar source list, compiler workers2'
            receipt={'backend':backend,'seconds':time.monotonic()-start,'binary_sha256':sha(binary),
                     'platform':platform.platform(),'parallel_build_workers':2,'fast_math':False,
                     'source_arithmetic_changed':False,'compiled_for_this_host':True}
            (self.cache/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
            return binary

    def _native48(self, padded):
        with tempfile.TemporaryDirectory(prefix='pcm-',dir=self.cache) as temporary:
            src,dst=Path(temporary)/'input.pcm',Path(temporary)/'output.pcm'
            padded.astype('=i2',copy=False).tofile(src)
            r=subprocess.run([str(self.binary),str(src),str(dst)],capture_output=True,timeout=120)
            if r.returncode:raise RuntimeError('Original Opus BWE failed: '+r.stderr.decode(errors='replace')[-2000:])
            output=np.fromfile(dst,dtype='=i2')
        if len(output)!=3*len(padded):raise RuntimeError('Original Opus frame count mismatch')
        return output

    def process_pcm16(self, pcm, *, sample_rate=16000):
        if (sample_rate!=16000 or not isinstance(pcm,np.ndarray) or pcm.dtype!=np.int16
                or pcm.ndim!=1 or not 0<len(pcm)<=3200000):
            raise ValueError('Expected 1..3200000 mono int16 samples at exactly16000Hz')
        start=time.monotonic();n=len(pcm);tail=(-n)%320
        padded=np.pad(pcm,(0,tail+640));native=self._native48(padded)
        aligned=native[34:].astype(np.float64)/32768.0
        target=(3*n+1)//2
        output=resample_poly(aligned,1,2,window=('kaiser',8.6),padtype='constant')[:target]
        if len(output)!=target or not np.isfinite(output).all():raise RuntimeError('Invalid24k result')
        quantized=np.rint(output*32768.0)
        clips=int(np.count_nonzero((quantized>32767)|(quantized<-32768)))
        result=np.clip(quantized,-32768,32767).astype(np.int16)
        report={**self.identity,'input_samples':n,'output_samples':len(result),'input_padding_samples':tail,
                'input_pcm16_sha256':hashlib.sha256(pcm.astype('<i2',copy=False).tobytes()).hexdigest(),
                'output_pcm24_sha256':hashlib.sha256(result.astype('<i2',copy=False).tobytes()).hexdigest(),
                'duration_error_seconds':len(result)/24000-n/16000,'quantizer_clipped_samples':clips,
                'seconds':time.monotonic()-start}
        return result,report

    def process_waveform(self, pcm, *, sample_rate=16000):
        output,report=self.process_pcm16(quantize(pcm),sample_rate=sample_rate)
        return output.astype(np.float32)/32768.0,report

    @staticmethod
    def resample_control(pcm):
        """Only3/2 resampling, with no learned bandwidth extension or gain."""
        if pcm.dtype!=np.int16 or pcm.ndim!=1 or not len(pcm):raise ValueError('Expected PCM16')
        target=(3*len(pcm)+1)//2
        y=resample_poly(pcm.astype(np.float64)/32768,3,2,window=('kaiser',8.6),padtype='constant')[:target]
        return np.clip(np.rint(y*32768),-32768,32767).astype(np.int16)

    def validate_port(self):
        """Small deterministic checks against the original Model wrapper math."""
        x=np.arange(4097,dtype=np.float64)
        probes=[np.zeros(320,dtype=np.int16),np.array([1000],dtype=np.int16),
                np.rint(np.sin(2*np.pi*440*x/16000)*10000).astype(np.int16),
                np.r_[np.array([30000],dtype=np.int16),np.zeros(336,dtype=np.int16)]]
        cases=[]
        for pcm in probes:
            output,report=self.process_pcm16(pcm)
            padded=np.pad(pcm,(0,(-len(pcm))%320+640));y48=self._native48(padded)
            aligned=y48[34:].astype(np.float64)/32768.0
            # Independent transcription of original Model post-GTCRN BWE
            # adapter. GTCRN itself is intentionally not part of this module.
            reference=resample_poly(aligned,1,2,window=('kaiser',8.6),padtype='constant')[:(3*len(pcm)+1)//2]
            reference=np.clip(np.rint(reference*32768),-32768,32767).astype(np.int16)
            assert np.array_equal(output,reference)
            assert report['output_samples']==(3*len(pcm)+1)//2
            cases.append({'input_samples':len(pcm),'output_samples':len(output),'original_wrapper_pcm_equal':True})
        again,_=self.process_pcm16(probes[2]);first,_=self.process_pcm16(probes[2]);assert np.array_equal(again,first)
        try:self.process_pcm16(probes[0],sample_rate=24000)
        except ValueError:pass
        else:raise AssertionError('Wrong input sample rate accepted')
        return {'status':'ORIGINAL_BWE_WRAPPER_EQUAL_LENGTH_AND_RESET_PASSED','cases':cases,
                'repeat_is_deterministic':True,'wrong_sample_rate_rejected':True,
                'cross_platform_bit_exactness_claimed':False,'no_quality_claim':True}


class RNDVoCWithBWE24:
    """Composable optional backend for a vocoder.synthesize() returning16k PCM.

    Returns numpy.float32 [1,N24]. ``last_report`` records actual conversion.
    It requires complete utterances and never changes the base model weights.
    """
    output_sample_rate=24000
    streaming=False

    def __init__(self, vocoder, postprocessor, *, stages=()):
        if getattr(vocoder,'output_sample_rate',16000)!=16000:
            raise ValueError('Base vocoder must provide16k PCM; native22.05k mel output requires its16k adapter first')
        self.vocoder=vocoder;self.postprocessor=postprocessor;self.stages=tuple(stages);self.last_report=None

    def synthesize(self, normalized, language):
        pcm=self.vocoder.synthesize(normalized,language)
        if hasattr(pcm,'detach'):pcm=pcm.detach().cpu().numpy()
        pcm=np.asarray(pcm)
        if pcm.ndim!=2 or pcm.shape[0]!=1:raise ValueError('Base vocoder must return mono [1,N]16k PCM')
        if self.stages:
            original=quantize(pcm[0])
            processed,reports=run_pcm16_stages(original,self.stages)
            pcm24,self.last_report=self.postprocessor.process_pcm16(processed,sample_rate=16000)
            output=pcm24.astype(np.float32)/32768.0
            self.last_report={**self.last_report,'intermediate_stages':reports,
                              'base_vocoder_pcm16_sha256':hashlib.sha256(original.astype('<i2',copy=False).tobytes()).hexdigest()}
        else:
            # Exactly the previously measured empty-chain signal path.
            output,self.last_report=self.postprocessor.process_waveform(pcm[0],sample_rate=16000)
            self.last_report={**self.last_report,'intermediate_stages':[]}
        return output[None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path);p.add_argument('--cache',type=Path);a=p.parse_args()
    with wave.open(str(a.input),'rb') as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,16000):raise ValueError('Input WAV must be mono PCM16 at16k')
        pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.int16)
    backend=OpusBWE24(a.assets,a.cache);output,report=backend.process_pcm16(pcm)
    save_wav(a.output,output,24000);Path(str(a.output)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
