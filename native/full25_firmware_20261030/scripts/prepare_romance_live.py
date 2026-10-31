"""Bind native whole-text Italian/Portuguese G2P to frozen Romance coordinates."""
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import numpy as np
from build_resources import ROOT,WORK,sha,write
from build_espeak import VENDOR

def write_cases(cases):
    lines=['#ifndef N25_FOUR_CASES_H','#define N25_FOUR_CASES_H',
        'typedef struct{const char*language,*text;unsigned count,dim,context_dim;unsigned char sha256[4][32];}FourCase;',
        'static const FourCase four_cases[]={']
    for row in cases:
        digests=','.join('{'+','.join(str(b)for b in bytes.fromhex(row['sha256'][k]))+'}'for k in ('phones','pause','phonetic','context'))
        dim=row.get('context_dim',34 if row['language']in ('fi','et','hr','lv','lt','bg','sl')else 22)
        lines.append('{'+','.join(json.dumps(row[k],ensure_ascii=False)for k in ('language','text'))+','+str(row['count'])+','+str(row['dim'])+','+str(dim)+',{'+digests+'}},')
    lines+=['};','#endif'];(ROOT/'generated/four_cases.h').write_text('\n'.join(lines)+'\n')

def add_cases(cases,languages):
    existing=WORK/'m55_fargan_stream_20261030/language25_20261030/inputs'
    for language in languages:
        row=json.loads((existing/language/'PREPARED.json').read_text());z=np.load(row['source_feature'])
        arrays={k:np.asarray(z[k],dtype='<u4'if k in('phones','pause')else'<f4')for k in('phones','pause','phonetic','context')}
        cases.append(dict(language=language,text=json.loads((existing/language/'job.json').read_text())['text'],
            count=len(z['phones']),dim=z['phonetic'].shape[1],context_dim=z['context'].shape[1],
            sha256={k:hashlib.sha256(a.tobytes()).hexdigest()for k,a in arrays.items()}))

def manifest(source,name,languages):
    d=json.loads((ROOT/source).read_text());all_languages='bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh'.split()
    for language in languages:
        text=json.loads((WORK/'m55_fargan_stream_20261030/language25_20261030/inputs'/language/'job.json').read_text())['text']
        d['native_text_cases'].append(dict(language=language,language_id=all_languages.index(language),text=text))
    d['native_text_cases'].sort(key=lambda x:x['language_id']);n=len(d['native_text_cases']);d['native_text_language_count']=n
    d['scope']=f'Native ordinary text for {n} language families; complete normalization and remaining families are pending.'
    write(ROOT/name,d);return d

def driver(dest,identity):
    p=dest/'src/multilang_driver.inc';s=p.read_text();a=s.index(' static const struct');b=s.index(' for(unsigned i=',a)
    rows=[' static const struct {unsigned language;const char *text;} cases[]={']
    for row in identity['native_text_cases']:rows.append('{'+str(row['language_id'])+','+json.dumps(row['text'],ensure_ascii=False)+'},')
    rows.append(' };\n');p.write_text(s[:a]+'\n'.join(rows)+s[b:])
    p=dest/'src/multilang_frontend.c';s=p.read_text();a=s.index(' static const struct');b=s.index(' const char*language=',a)
    rows=[' static const struct {unsigned id;const char*name;} languages[]={']
    for row in identity['native_text_cases']:
        if row['language']!='zh':rows.append('{'+str(row['language_id'])+','+json.dumps(row['language'])+'},')
    rows.append(' };\n');p.write_text(s[:a]+'\n'.join(rows)+s[b:])

def main():
    source=ROOT/'native_pair_interleaved';dest=ROOT/'native_romance_live';dest.mkdir(exist_ok=False)
    for name in('src','assets','frontend_patched','dictionary_runtime'):shutil.copytree(source/name,dest/name)
    for name in('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):(dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in(dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    sys.path[:0]=[str(VENDOR),str(VENDOR/'runtime/current25')]
    from project.src.frontend import romance
    import fe_codec as fe
    vocabulary=json.loads((VENDOR.parent.parent/'int8/assets/vocabulary.json').read_text())
    names=set(vocabulary)|set(romance.INVENTORY)
    for locale in('es-ES','fr-FR','it-IT','pt-PT','ro-RO'):
        docs,_=fe.raw_tables(ROOT/'resources/packages'/(locale+'.fep'),VENDOR/'frontend/libs/libreader.so')
        m=fe.unpack_metadata(docs[locale]['metadata']);names.update(m['symbols'])
    for name,segment in romance.INVENTORY.items():
        if name=='sil':continue
        names.add(name+'ː')
        if segment.manner=='vowel':names.add(unicodedata.normalize('NFC',name+'̃'))
        else:names.add(name+'ʷ');names.add(name+'ʲ')
    base=[]
    for m in re.finditer(r'\{"([^"\n]*)",(\d+),\{([^}]*)\},\{([^}]*)\}\}',(ROOT/'generated/four_tables.c').read_text()):
        parse=lambda x:float.fromhex(x.rstrip('f')) if '0x'in x else float(x.rstrip('f'))
        features=[parse(v)for v in(m[3]+','+m[4]).split(',')]
        base.append((m[1],features))
    lookup={name:(i,features)for i,(name,features)in enumerate(base)};assert len(base)>100
    rows=[];patches=[];palette=[0.]
    for name in sorted(names):
        if name.startswith(('ˈ','ˌ')):continue
        try:features=romance.encode_ipa(name).tolist()
        except ValueError:continue
        index,before=lookup.get(name,(65535,[0.]*36));start=len(patches)
        for col,(old,value)in enumerate(zip(before,features)):
            if old!=value:
                if value not in palette:palette.append(value)
                patches.append((col,palette.index(value)))
        rows.append((name,vocabulary.get(name,1),index,start,len(patches)-start))
    code=['typedef struct{const char*name;unsigned id;unsigned short base,patch;unsigned char count;}RomanceConstant;',
        'static const float romance_values[]={'+','.join(float(v).hex()+'f'for v in palette)+'};',
        'static const unsigned char romance_patches[][2]={'+','.join('{'+str(a)+','+str(b)+'}'for a,b in patches)+'};',
        'static const RomanceConstant romance_constants[]={']
    for name,index,base,start,count in rows:code.append('{'+json.dumps(name,ensure_ascii=False)+','+','.join(map(str,(index,base,start,count)))+'},')
    code+=['};'];(dest/'src/romance_constants.h').write_text('\n'.join(code)+'\n')
    p=dest/'src/four_frontend.c';s=p.read_text().replace('#include "spanish_frontend.inc"','#include "spanish_frontend.inc"\n#include "romance_live.inc"')
    s=s.replace('if(!strcmp(language,"es"))return n25_spanish_to_view(text,out);','if(!strcmp(language,"es"))return n25_spanish_to_view(text,out);\n if(!strcmp(language,"it")||!strcmp(language,"pt"))return n25_romance_live(language,text,out);');p.write_text(s)
    cases=json.loads((ROOT/'generated/SPANISH_FEATURE_CASES.json').read_text())['cases'];add_cases(cases,['it','pt']);write_cases(cases)
    write(ROOT/'generated/ROMANCE_LIVE_FEATURE_CASES.json',dict(cases=cases))
    d=manifest('SPANISH_SOURCE_IDENTITY.json','ROMANCE_LIVE_SOURCE_IDENTITY.json',['it','pt']);driver(dest,d)
    write(ROOT/'ROMANCE_CONSTANTS_IDENTITY.json',dict(count=len(rows),patches=len(patches),palette=len(palette),
        source_sha256=sha(VENDOR/'runtime/repair_support/it_frontend.py'),
        port_scope='Whole-utterance Italian/Portuguese G2P repair; dictionary fallback policy and extended normalization tracked separately'))
    print(len(rows),'phone constants',len(patches),'coordinate patches')

if __name__=='__main__':main()
