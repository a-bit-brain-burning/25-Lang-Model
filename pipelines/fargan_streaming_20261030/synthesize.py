"""Current acoustic/prosody + repository FARGAN4750 + original Opus BWE24.

No training or source weight changes. Default stream mode emits 20 ms PCM packets;
full mode retains the original whole-unit inference and quiet-boundary join.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FRONTEND = WORK/'new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24'
LANGUAGES = 'bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh'.split()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text',required=True)
    parser.add_argument('--language',choices=LANGUAGES,default='zh')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--output16',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--mode',choices=['stream','full'],default='stream',
                        help='stream (default) emits 20 ms PCM packets; full keeps the original reference')
    parser.add_argument('--frontend-config',type=Path)
    parser.add_argument('--abbreviation-mode',choices=['auto','spell','word'],default='auto')
    parser.add_argument('--identifier-prosody',choices=['natural','original'],default='natural',
                        help='cea342e identifier pitch/connection controls; natural buffers each touched identifier group before using its corrected pitch')
    parser.add_argument('--acoustic-chunk',type=int,default=16)
    parser.add_argument('--pcm-stdout',action='store_true',help='Also emit mono PCM16/24k to stdout as chunks arrive (stream mode)')
    args = parser.parse_args()
    if args.pcm_stdout and args.mode != 'stream':
        parser.error('--pcm-stdout requires stream mode')
    if args.acoustic_chunk < 1:
        parser.error('Chunk sizes must be positive')
    begin = time.perf_counter()
    pcm_stdout = sys.stdout.buffer if args.pcm_stdout else None
    with tempfile.TemporaryDirectory(prefix='current-fargan-') as temporary:
        directory = Path(temporary)
        job = dict(text=args.text,language=args.language,features=str(directory/'features'),
                   plan=str(directory/'plan.json'),abbreviation_mode=args.abbreviation_mode,
                   frontend_config=str(args.frontend_config.resolve()) if args.frontend_config else None)
        write_json(directory/'job.json',job)
        # Keep frontend/runtime module namespaces isolated just as the original
        # released CLI does, and keep stdout clean for raw streamed PCM.
        subprocess.run([sys.executable,str(FRONTEND/'text_worker.py'),str(directory/'job.json')],
                       check=True,stdout=sys.stderr if pcm_stdout else None)
        plan = json.loads((directory/'plan.json').read_text())
        frontend_seconds = time.perf_counter()-begin
        sys.path.insert(0,str(ROOT/'repo'))
        spec = importlib.util.spec_from_file_location('_current_fargan_stream',ROOT/'runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import numpy as np
        import torch
        torch.set_num_threads(1)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        details = []
        first_written = None
        first_model_yield = None
        output_chunks = 0
        packet_samples = []
        model_yield_samples = []
        pcm_fifo = bytearray()
        packet_bytes = 480*2
        buffered = []
        with contextlib.ExitStack() as stack:
            with contextlib.redirect_stdout(sys.stderr):
                engine = stack.enter_context(module.FarganPipeline(seed=args.seed,identifier_prosody=args.identifier_prosody))
            if args.output16:
                args.output16.parent.mkdir(parents=True,exist_ok=True)
                raw = stack.enter_context(wave.open(str(args.output16),'wb'))
                raw.setnchannels(1);raw.setsampwidth(2);raw.setframerate(16000)
                engine.on_pcm16 = lambda pcm: raw.writeframes(pcm.astype('<i2',copy=False).tobytes())
            identity = engine.identity
            with args.output.open('wb') as wav_file, wave.open(wav_file,'wb') as output:
                output.setnchannels(1);output.setsampwidth(2);output.setframerate(24000)
                def write_packet(data):
                    nonlocal first_written,output_chunks
                    assert data and len(data)%2 == 0 and len(data) <= packet_bytes
                    output.writeframes(data)
                    wav_file.flush()
                    if pcm_stdout:
                        pcm_stdout.write(data);pcm_stdout.flush()
                    if first_written is None:
                        first_written = time.perf_counter()-begin
                    output_chunks += 1
                    packet_samples.append(len(data)//2)
                def emit(pcm):
                    if not len(pcm):
                        return
                    assert pcm.dtype == np.int16
                    data = memoryview(pcm.astype('<i2',copy=False).tobytes())
                    offset = 0
                    while offset < len(data):
                        take = min(packet_bytes-len(pcm_fifo),len(data)-offset)
                        pcm_fifo.extend(data[offset:offset+take]);offset += take
                        if len(pcm_fifo) == packet_bytes:
                            write_packet(bytes(pcm_fifo));pcm_fifo.clear()
                for entry in plan['chunks']:
                    feature_path = Path(entry['file']);metadata_path = Path(entry['metadata_file'])
                    assert sha(feature_path)==entry['sha256'] and sha(metadata_path)==entry['metadata_sha256']
                    with np.load(feature_path,allow_pickle=False) as data:
                        features = {key:data[key].copy() for key in data.files}
                    metadata = json.loads(metadata_path.read_text())
                    if args.mode == 'full':
                        pcm,report = engine.full_unit(features,metadata,args.language)
                        buffered.append(pcm.astype(np.float32)/32768)
                    else:
                        for pcm in engine.stream_unit(features,metadata,args.language,
                                acoustic_chunk=args.acoustic_chunk):
                            if len(pcm):
                                if first_model_yield is None:
                                    first_model_yield = time.perf_counter()-begin
                                model_yield_samples.append(len(pcm))
                            emit(pcm)
                        report = engine.last_report
                    details.append(dict(text=entry['text'],feature_SHA256=entry['sha256'],
                                        frontend=metadata,runtime=report))
                if args.mode == 'full':
                    joined,joins = engine.legacy.join_phrases(buffered,[e['text'] for e in plan['chunks']],24000)
                    emit(np.rint(joined.astype(np.float64)*32768).clip(-32768,32767).astype(np.int16))
                else:
                    joins = []
                # Flush only the real EOS remainder; never pad with artificial silence.
                if pcm_fifo:
                    write_packet(bytes(pcm_fifo));pcm_fifo.clear()
        with wave.open(str(args.output),'rb') as output:
            samples = output.getnframes()
        record = dict(status='COMPLETE',text=args.text,language=args.language,mode=args.mode,
                      output=str(args.output.resolve()),output_sha256=sha(args.output),sample_rate=24000,
                      PCM24_samples=samples,audio_seconds=samples/24000,output_chunks=output_chunks,
                      PCM_packet_samples=480,PCM_packet_milliseconds=20,
                      PCM_packet_count=output_chunks,PCM_packet_sample_counts=packet_samples,
                      PCM_final_packet_samples=packet_samples[-1] if packet_samples else 0,
                      PCM_tail_padding_samples=0,
                      model_yield_blocks=len(model_yield_samples),model_yield_block_samples=model_yield_samples,
                      first_model_yield_host_seconds=first_model_yield,
                      frontend_seconds=frontend_seconds,first_written_host_seconds=first_written,
                      first_written_measures_any_PCM_not_effective_speech=True,
                      cold_CLI_total_host_seconds=time.perf_counter()-begin,
                      host_time_is_not_M55_cycles=True,model_identity=identity,segments=details,
                      natural_boundary_joins=joins,
                      stream_multiunit_uses_predicted_pauses_without_global_quiet_join=args.mode=='stream' and len(details)>1,
                      stream_vs_full_file_identity_accepted=False,
                      numeric_note='Original FARGAN valid center [400,160*T-240); no teacher prefill or edge padding. Original FP16 weight bytes and FP32 states; streamed FP batching may differ from offline.',
                      output16=str(args.output16.resolve()) if args.output16 else None,seed=args.seed,vocoder='FARGAN4750',streaming=True,
                      identifier_prosody_mode=args.identifier_prosody,
                      frontend_commit='cea342e5f3ee6d2efd228da78b2f3cf895062118',
                      frontend_source=str(FRONTEND),
                      training_updates=0,weights_modified=False,quality_human_accepted=False)
        write_json(args.output.with_suffix('.json'),record)
        summary = {key:record[key] for key in ['status','output','output_sha256','mode','audio_seconds','output_chunks','training_updates']}
        print(json.dumps(summary,ensure_ascii=False),file=sys.stderr if pcm_stdout else sys.stdout)

if __name__ == '__main__':
    main()
