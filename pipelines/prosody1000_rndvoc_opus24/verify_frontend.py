"""Exercise flexible readings through the complete unchanged acoustic/BWE chain."""
from pathlib import Path
import os,sys,json,hashlib,wave,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
PACKAGE=Path(__file__).resolve().parent
import argparse
from runtime import Engine,write,sha
from text_worker import plan
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('output/frontend-flex'));args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    engine=Engine()
    from new25_postprocess.bwe24 import save_wav
    cases=json.loads((PACKAGE/'tests/flexible_readings.json').read_text());old=json.loads((PACKAGE/'tests/REFERENCE.json').read_text());rows=[]
    for case in cases:
        cfg=PACKAGE/case['config'] if 'config'in case else None
        row=plan(case['text'],case['language'],out/'features'/case['id'],cfg)
        raw,final,record=engine.process(row);dest=out/'wav24'/(case['id']+'.wav');dest.parent.mkdir(exist_ok=True);save_wav(dest,final,24000)
        assert ''.join(s['text']for s in record['segments'])==case['text']
        assert all(s['controls']['frontend']['unknown_phone_count']==0 for s in record['segments'])
        if case['language']=='zh':assert record['English_acoustic_calls']==0
        if case['id'] in ['flight_cz','flight_mu']:assert sha(dest)==old[case['id']]['wav24_sha256']
        if case['id']=='zh_standalone':
            assert record['segments'][0]['controls']['frontend']['context_anchor_used']
            assert not record['segments'][0]['controls']['frontend']['context_anchor_spoken']
        if case['id']=='en_long_title':
            source=case['text'];assert not any(s['end'] in range(source.index('Dr.'),source.index('Smith')+len('Smith'))for s in record['segments'][:-1])
        record.update(id=case['id'],text=case['text'],language=case['language'],config=case.get('config'),wav24_sha256=sha(dest),duration_seconds=len(final)/24000)
        rows.append(record);write(out/'records'/(case['id']+'.json'),record)
        write(out/'STATUS.json',dict(completed=len(rows),total=len(cases),last_id=case['id']))
        print(case['id'],len(record['segments']),len(final)/24000,flush=True)
    modes={r['id']:r for r in rows}
    assert modes['en_sql_auto']['wav24_sha256']!=modes['en_sql_custom']['wav24_sha256']
    write(out/'RESULT.json',dict(status='PASSED',cases=len(rows),all_unknown_phones_zero=True,flight_reference_wavs_unchanged=True,
      all_Chinese_English_acoustic_calls_zero=True,custom_SQL_reading_changes_generated_audio=True,training_updates=0,rows=rows))
if __name__=='__main__':main()
