"""Ordinary text -> bounded acoustic1000/RNDVoC/Opus -> 24kHz WAV."""
from pathlib import Path
import argparse,sys,subprocess,tempfile,json
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--text',required=True);p.add_argument('--language',choices='bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh'.split(),default='zh');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frontend-config',type=Path,help='JSON pronunciation overrides; defaults remain active')
    p.add_argument('--abbreviation-mode',choices=['auto','spell','word'],default='auto',help='Reading strategy for alphabetic abbreviation candidates')
    p.add_argument('--identifier-prosody',choices=['natural','original'],default='natural',help='Chinese identifier pitch/connection policy; original retains the previous controls')
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='bounded-new25-') as temp:
        d=Path(temp);job=dict(text=args.text,language=args.language,features=str(d/'features'),plan=str(d/'plan.json'),frontend_config=str(args.frontend_config.resolve()) if args.frontend_config else None,abbreviation_mode=args.abbreviation_mode)
        (d/'job.json').write_text(json.dumps(job,ensure_ascii=False))
        subprocess.run([sys.executable,str(ROOT/'text_worker.py'),str(d/'job.json')],check=True)
        row=json.loads((d/'plan.json').read_text())
        from runtime import Engine,write,sha
        engine=Engine(identifier_prosody=args.identifier_prosody);raw,final,record=engine.process(row)
        from new25_postprocess.bwe24 import save_wav
        args.output.parent.mkdir(parents=True,exist_ok=True);save_wav(args.output,final,24000)
        record.update(output=str(args.output),output_sha256=sha(args.output));write(args.output.with_suffix('.json'),record)
        print(json.dumps(dict(output=str(args.output),sample_rate=24000,segments=len(record['segments']),sha256=sha(args.output)),ensure_ascii=False))
if __name__=='__main__':main()
