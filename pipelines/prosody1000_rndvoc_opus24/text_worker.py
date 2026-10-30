"""Ordinary-text entry for the same lossless bounded inference plan."""
from pathlib import Path
import sys,json,hashlib,functools,re
import numpy as np
from frontend_repair import TextFrontend,save,sha
from frontend_live import build_zh,details
from reading_policy import ReadingPolicy
from reference_chinese import FrontendCapacityError
from plan_rules import protected_positions,LIMIT_PHONES
def plan(text,language,directory,frontend_config=None,abbreviation_mode='auto'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    policy=ReadingPolicy(frontend_config,abbreviation_mode)
    with TextFrontend() as front:
        @functools.lru_cache(maxsize=4096)
        def encode(value):
            if language=='zh':return build_zh(front,value,policy)
            spoken,unused,events,identity=details(value,'en',front,policy) if language=='en' else (value,[],[],policy.identity())
            arrays,meta=front.encode(spoken,language);meta.update(input_text=value,normalized_text=spoken,identifier_units=[],reading_decisions=events,frontend_policy=identity);return arrays,meta
        def fits(value):
            try:return len(encode(value)[0]['phones'])<=LIMIT_PHONES
            except FrontendCapacityError:return False
        chunks=[];start=0;forbidden=protected_positions(text)
        if language in ['zh','en']:
            for lo,hi in policy.atomic_spans(text,language):forbidden.update(range(lo+1,hi))
        while start<len(text):
            tail=text[start:]
            if fits(tail):end=len(text);reason='end_of_text'
            else:
                strong=[];weak=[]
                for match in re.finditer(r'[。！？!?；;]|\.(?!\d)|[，,、：:]|\n',tail):
                    stop=start+match.end()
                    if stop in forbidden:continue
                    (strong if match[0] in '。！？!?；;.\n' else weak).append(stop)
                end=None
                for reason,ends in [('sentence_boundary',strong),('clause_boundary_at_context_limit',weak)]:
                    for stop in reversed(ends):
                        if stop-start>200:continue
                        if stop>start and fits(text[start:stop]):end=stop;break
                    if end is not None:break
                if end is None:
                    reason='word_boundary_at_context_limit' if language!='zh' else 'character_boundary_at_context_limit'
                    for stop in range(min(len(text)-1,start+120),start,-1):
                        if stop in forbidden or (language!='zh' and not text[stop-1:stop].isspace()):continue
                        if fits(text[start:stop]):end=stop;break
                if end is None:raise ValueError('A single indivisible identifier exceeds the bounded context budget')
            source=text[start:end];arrays,meta=encode(source);assert meta['unknown_phone_count']==0
            key=hashlib.sha256((language+'\0'+source).encode()).hexdigest()[:24];path=(directory/(key+'.npz')).resolve()
            np.savez_compressed(path,**arrays);save(path.with_suffix('.json'),meta)
            chunks.append(dict(text=source,start=start,end=end,split_reason=reason,phones=len(arrays['phones']),
              file=str(path),metadata_file=str(path.with_suffix('.json')),sha256=sha(path),metadata_sha256=sha(path.with_suffix('.json'))));start=end
        assert ''.join(c['text'] for c in chunks)==text
        assert all(c['end'] not in forbidden for c in chunks[:-1])
        return dict(id='ordinary_cli',text=text,language=language,chunks=chunks,frontend_policy=policy.identity())
def main():
    job=json.loads(Path(sys.argv[1]).read_text());result=plan(job['text'],job['language'],job['features'],job.get('frontend_config'),job.get('abbreviation_mode','auto'))
    save(Path(job['plan']),result)
if __name__=='__main__':main()
