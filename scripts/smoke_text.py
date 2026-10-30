"""Run all 25 ordinary text examples through the full host INT8 reference.

These original short sentences are dependency/numerics smoke tests, not an
accuracy evaluation or evidence of passing per-language quality thresholds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'int8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'out/smoke')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    from frontend import TextFrontend
    from runtime import load
    from new25_hifigan.inference import write_pcm16
    acoustic,vocoder=load(ROOT/'int8')
    cases=json.loads((ROOT/'examples/smoke_text.json').read_text())
    rows=[]
    with TextFrontend()as frontend:
        for language,text in cases.items():
            start=time.perf_counter()
            features,meta=frontend.encode(text,language)
            result=acoustic.predict(language=language,**features)
            audio=vocoder.synthesize(result.normalized[None],language).detach().cpu().numpy()
            assert np.isfinite(audio).all()and audio.size>0
            assert meta['unknown_phone_count']==0,meta
            wav=args.output/(language+'.wav')
            write_pcm16(wav,audio)
            np.savez_compressed(args.output/(language+'.npz'),**features)
            row=dict(language=language,phones=len(features['phones']),unknown_phone_count=0,
                     frames=int(result.durations.sum()),samples=int(audio.size),sample_rate=16000,
                     wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                     seconds=time.perf_counter()-start)
            rows.append(row)
            print(json.dumps(row),flush=True)
    receipt=dict(passed=len(rows)==25,languages=len(rows),ordinary_text_only=True,
                 reference_audio_or_durations_used=False,quality_evaluation=False,
                 execution='host FP32 compute with current INT8 weights',rows=rows)
    (args.output/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
