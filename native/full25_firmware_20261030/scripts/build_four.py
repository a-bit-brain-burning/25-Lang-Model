"""Generate original phone constants and validate native eu_four features."""
from pathlib import Path
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import unicodedata
import numpy as np
from build_resources import ROOT,WORK,NATIVE,sha,write
from build_espeak import VENDOR
LANGUAGES='cs da de el hu nl no pl sk sv tr'.split()


def generate_tables():
    runtime=VENDOR/'models/G9/systems/hu/frontend/hu/runtime'
    name='_n25_native_feature_constants'
    spec=importlib.util.spec_from_file_location(name,runtime/'__init__.py',submodule_search_locations=[str(runtime)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    m=importlib.import_module(name+'.frontend.eu_four')
    for ipa,manner,place,voiced,lateral,rhotic in[
        ('ʑ','fricative','palatal',True,False,False),('dʑ','affricate','palatal',True,False,False),
        ('pf','affricate','labiodental',False,False,False),('ɧ','fricative','velar',False,False,False),
        ('ɭ','approximant','retroflex',True,True,False),('ɳ','nasal','retroflex',True,False,False),
        ('ʈ','stop','retroflex',False,False,False),('ɖ','stop','retroflex',True,False,False),
        ('ɽ','approximant','retroflex',True,False,True)]:
        m._BASES[ipa]=m.PhoneticSegment(ipa,manner,place,voiced,lateral=lateral,rhotic=rhotic)
    m._BASES['ʉ']=m.PhoneticSegment('ʉ','vowel',voiced=True,vowel_start=(0.,.5,1.))
    m._BASES['ɶ']=m.PhoneticSegment('ɶ','vowel',voiced=True,vowel_start=(1.,0.,1.))
    m._BASES['ɘ']=m.PhoneticSegment('ɘ','vowel',voiced=True,vowel_start=(.33,.5,0.))
    vocab_path=VENDOR.parent.parent/'int8/assets/vocabulary.json'
    vocabulary=json.loads(vocab_path.read_text());rows=[]
    for phone,index in sorted(vocabulary.items()):
        if m._canonical_ipa(phone)!=phone:continue
        try:features=m.encode_phone(phone)
        except ValueError:continue
        core=unicodedata.normalize('NFD',phone).translate(str.maketrans('','','ːˑʰʲˠ̥̬̪̩̯̃'))
        extra=[float(core in('ɾ','ɽ')),float(core=='r'),float(core in('ɕ','tɕ','ʑ','dʑ')),float('ʷ'in phone or core in('w','ɥ'))]
        rows.append(dict(phone=phone,id=index,features=features.tolist(),extra=extra))
    characters=[]
    for cp in range(0x110000):
        c=chr(cp);name=unicodedata.name(c,'')
        script=1 if'LATIN'in name else 2 if'GREEK'in name else 0
        if script and unicodedata.category(c).startswith('L'):
            lower=c.lower();characters.append(dict(cp=cp,lower=ord(lower)if len(lower)==1 else cp,script=script))
    generated=ROOT/'generated';generated.mkdir(exist_ok=True)
    h=['#ifndef N25_FOUR_TABLES_H','#define N25_FOUR_TABLES_H','#include <stdint.h>',
        'typedef struct{const char*name;unsigned id;float features[32],extra[4];}N25FourPhone;',
        'typedef struct{uint32_t cp,lower,script;}N25FourCharacter;',
        'typedef struct{const char*language,*voice;unsigned phonetic_dim;}N25FourLanguage;',
        f'#define N25_FOUR_PHONE_COUNT {len(rows)}',f'#define N25_FOUR_CHARACTER_COUNT {len(characters)}',
        f'#define N25_FOUR_LANGUAGE_COUNT {len(LANGUAGES)}',
        'extern const N25FourPhone n25_four_phones[N25_FOUR_PHONE_COUNT];',
        'extern const N25FourCharacter n25_four_characters[N25_FOUR_CHARACTER_COUNT];',
        'extern const N25FourLanguage n25_four_languages[N25_FOUR_LANGUAGE_COUNT];','#endif']
    (generated/'four_tables.h').write_text('\n'.join(h)+'\n')
    code=['#include "four_tables.h"','const N25FourPhone n25_four_phones[N25_FOUR_PHONE_COUNT]={']
    for row in rows:
        code.append('{'+json.dumps(row['phone'],ensure_ascii=False)+','+str(row['id'])+',{'+
            ','.join(float(x).hex()+'f'for x in row['features'])+'},{'+','.join(str(int(x))+'.f'for x in row['extra'])+'}},')
    code+=['};','const N25FourCharacter n25_four_characters[N25_FOUR_CHARACTER_COUNT]={']
    code.extend('{'+str(row['cp'])+','+str(row['lower'])+','+str(row['script'])+'},'for row in characters)
    code+=['};','const N25FourLanguage n25_four_languages[N25_FOUR_LANGUAGE_COUNT]={']
    for language in LANGUAGES:
        dim=32 if language in('hu','el') else 36
        code.append('{"'+language+'","'+('nb'if language=='no'else language)+'",'+str(dim)+'},')
    code+=['};'];(generated/'four_tables.c').write_text('\n'.join(code)+'\n')
    cases=[]
    existing=WORK/'m55_fargan_stream_20261030/language25_20261030/inputs'
    for language in LANGUAGES:
        case=json.loads((existing/language/'PREPARED.json').read_text());z=np.load(case['source_feature'])
        text=json.loads((existing/language/'job.json').read_text())['text']
        arrays={'phones':np.asarray(z['phones'],dtype='<u4'),'pause':np.asarray(z['pause'],dtype='<u4'),
            'phonetic':np.asarray(z['phonetic'],dtype='<f4'),'context':np.asarray(z['context'],dtype='<f4')}
        cases.append(dict(language=language,text=text,count=len(z['phones']),dim=z['phonetic'].shape[1],
            sha256={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in arrays.items()}))
    lines=['#ifndef N25_FOUR_CASES_H','#define N25_FOUR_CASES_H',
        'typedef struct{const char*language,*text;unsigned count,dim;unsigned char sha256[4][32];}FourCase;',
        'static const FourCase four_cases[]={']
    for row in cases:
        digests=','.join('{'+','.join(str(b)for b in bytes.fromhex(row['sha256'][k]))+'}'for k in('phones','pause','phonetic','context'))
        lines.append('{'+','.join(json.dumps(row[k],ensure_ascii=False)for k in('language','text'))+','+str(row['count'])+','+str(row['dim'])+',{'+digests+'}},')
    lines+=['};','#endif'];(generated/'four_cases.h').write_text('\n'.join(lines)+'\n')
    write(generated/'FOUR_FEATURE_IDENTITY.json',dict(cases=cases,phone_rows=len(rows),
        character_rows=len(characters),source_sha256=sha(runtime/'frontend/eu_four.py'),
        vocabulary_sha256=sha(vocab_path),scope='Ordinary NFC text; numeric, identifier and full normalizer port pending'))


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='four_v1');a=parser.parse_args()
    generate_tables()
    subprocess.run([sys.executable,str(ROOT/'build_espeak.py'),'--tag',a.tag,
        '--probe','four_probe.c','--extra-source',str(NATIVE/'src/four_frontend.c'),
        '--extra-source',str(NATIVE/'src/storage_sha256.c'),
        '--extra-source',str(ROOT/'generated/four_tables.c'),
        '--expect-marker','N25_NATIVE_FOUR_FEATURES_PASS'],check=True)


if __name__=='__main__':main()
