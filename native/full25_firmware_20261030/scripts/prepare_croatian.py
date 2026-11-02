"""Port the frozen Croatian phone/stress/context rules without word fixtures."""
import hashlib
import importlib
import importlib.util
import json
import shutil
import sys
import numpy as np
import build_four
from build_resources import ROOT,WORK,sha,write
from build_espeak import VENDOR

def main():
    source=ROOT/'native_zero_tail_vec';dest=ROOT/'native_croatian';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    build_four.LANGUAGES+='et fi hr'.split();build_four.generate_tables()
    p=ROOT/'generated/four_tables.c';s=p.read_text()
    for language in ('et','fi','hr'):s=s.replace('{"'+language+'","'+language+'",36}', '{"'+language+'","'+language+'",32}')
    p.write_text(s)
    runtime=VENDOR/'models/G5/systems/hr/frontend/hr/runtime';name='_n25_native_croatian'
    spec=importlib.util.spec_from_file_location(name,runtime/'__init__.py',submodule_search_locations=[str(runtime)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    m=importlib.import_module(name+'.frontend.eu_south_slavic')
    rows=['typedef struct{const char*name,*ipa;}CroatianName;','static const CroatianName croatian_names[]={']
    for key,value in sorted(m._NAMES['hr-HR'].items()):rows.append('{'+json.dumps(key)+','+json.dumps(value,ensure_ascii=False)+'},')
    rows.append('};');(dest/'src/croatian_names.h').write_text('\n'.join(rows)+'\n')
    # Independent source implementation generates test hashes only, never linked pronunciations.
    front=m.SouthSlavicFrontend('hr',espeak_library=str(VENDOR/'frontend/libs/libespeak_phonemes.so'),
        espeak_data=str(ROOT/'host_reference/espeak-ng-data'),pause_policy='word')
    vocabulary=json.loads((VENDOR.parent.parent/'int8/assets/vocabulary.json').read_text())
    cases=json.loads((ROOT/'generated/FOUR_FEATURE_IDENTITY.json').read_text())['cases']
    for text in ('S krova, k moru.', 'Prst, krv i vrt.', 'Lijepa riječ, čudna priča?',
                 'Đak čita knjigu.', 'Džep je pun.', 'Ovo je pitanje? Ovo je kraj.',
                 'Dva--tri koraka.', 'Želim čuti glas.','To je test (glasa).'):
        f=front.encode(text);arrays={'phones':np.array([vocabulary.get(p,1)for p in f['ipa']],dtype='<u4'),
            'pause':np.asarray(f['pause_mask'],dtype='<u4'),'phonetic':np.asarray(f['phonetic_features'],dtype='<f4'),
            'context':np.asarray(f['context_features'],dtype='<f4')}
        cases.append(dict(language='hr',text=text,count=len(f['ipa']),dim=32,
            sha256={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in arrays.items()}))
    lines=['#ifndef N25_FOUR_CASES_H','#define N25_FOUR_CASES_H',
        'typedef struct{const char*language,*text;unsigned count,dim,context_dim;unsigned char sha256[4][32];}FourCase;',
        'static const FourCase four_cases[]={']
    for row in cases:
        digests=','.join('{'+','.join(str(b)for b in bytes.fromhex(row['sha256'][k]))+'}'for k in ('phones','pause','phonetic','context'))
        dim=34 if row['language']in ('fi','et','hr')else 22
        lines.append('{'+','.join(json.dumps(row[k],ensure_ascii=False)for k in ('language','text'))+','+str(row['count'])+','+str(row['dim'])+','+str(dim)+',{'+digests+'}},')
    lines+=['};','#endif'];(ROOT/'generated/four_cases.h').write_text('\n'.join(lines)+'\n')
    write(ROOT/'generated/CROATIAN_FEATURE_CASES.json',dict(cases=cases,croatian_source_sha256=sha(runtime/'frontend/eu_south_slavic.py')))
    identity=json.loads((ROOT/'FINNIC_SOURCE_IDENTITY.json').read_text())
    text=json.loads((WORK/'m55_fargan_stream_20261030/language25_20261030/inputs/hr/job.json').read_text())['text']
    identity['native_text_cases'].append(dict(language='hr',language_id=10,text=text))
    identity['native_text_cases'].sort(key=lambda x:x['language_id']);identity['native_text_language_count']=15
    identity['scope']='Ordinary NFC Han subset, eu_four, Finnic and Croatian families; full normalization and 10 remaining families pending.'
    identity['croatian_source_sha256']=sha(runtime/'frontend/eu_south_slavic.py');write(ROOT/'CROATIAN_SOURCE_IDENTITY.json',identity)
    p=dest/'src/multilang_driver.inc';s=p.read_text();a=s.index(' static const struct');b=s.index(' for(unsigned i=',a)
    rows=[' static const struct {unsigned language;const char *text;} cases[]={']
    for row in identity['native_text_cases']:rows.append('{'+str(row['language_id'])+','+json.dumps(row['text'],ensure_ascii=False)+'},')
    rows.append(' };\n');p.write_text(s[:a]+'\n'.join(rows)+s[b:])
    p=dest/'src/multilang_frontend.c';p.write_text(p.read_text().replace('{8,"fi"},','{8,"fi"},\n {10,"hr"},'))

if __name__=='__main__':main()
