"""Ordinary-text whole-pipeline smoke test, including every supported language."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,json,hashlib,time,wave
import numpy as np
from runtime import Engine,write,sha,ROOT
from text_worker import plan

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('output/verification'));p.add_argument('--strict-reference',action='store_true');a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    from new25_postprocess.bwe24 import save_wav
    engine=Engine();cases=json.loads((ROOT/'tests/cases.json').read_text());results=[]
    expected=json.loads((ROOT/'tests/REFERENCE.json').read_text()) if (ROOT/'tests/REFERENCE.json').exists() else {}
    for case in cases:
        start=time.monotonic();row=plan(case['text'],case['language'],a.output/'features'/case['id'])
        raw,final,record=engine.process(row)
        path=a.output/(case['id']+'.wav');save_wav(path,final,24000)
        with wave.open(str(path),'rb') as f:assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(24000,1,2)
        assert np.any(final) and np.isfinite(final).all()
        assert all(s['controls']['frontend']['unknown_phone_count']==0 for s in record['segments'])
        assert ''.join(s['text']for s in record['segments'])==case['text']
        assert not record['identifiers_split']
        if case['language']=='zh':assert record['English_acoustic_calls']==0
        digest=sha(path)
        if a.strict_reference:assert expected[case['id']]['wav24_sha256']==digest,case['id']
        result=dict(id=case['id'],language=case['language'],wav24_sha256=digest,samples=len(final),sample_rate=24000,segments=len(record['segments']),
          max_segment_seconds=max(s['duration_seconds']for s in record['segments']),seconds=time.monotonic()-start,
          rndvoc_clipped_samples=sum(s['rndvoc_clipped_samples']for s in record['segments']),bwe_clipped_samples=sum(s['bwe']['quantizer_clipped_samples']for s in record['segments']))
        results.append(result);write(a.output/'records'/(case['id']+'.json'),record)
        write(a.output/'STATUS.json',dict(completed=len(results),total=len(cases),last_id=case['id']))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    assert len({r['language']for r in results})==25
    write(a.output/'RESULT.json',dict(status='PASSED',cases=len(results),languages=25,strict_reference=a.strict_reference,ordinary_text_only=True,
      all_source_text_preserved=True,teacher_durations_used=False,training_updates=0,quality_acceptance=False,rows=results))
if __name__=='__main__':main()
