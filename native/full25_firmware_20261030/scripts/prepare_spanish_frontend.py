"""Generate full-resource Spanish constants and native differential cases."""
import hashlib
import importlib
import json
import shutil
import sys
import numpy as np
import build_four
from build_resources import ROOT,WORK,write,sha
from build_espeak import VENDOR

def main():
    source=ROOT/'native_dictionary_views';dest=ROOT/'native_spanish';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched','dictionary_runtime'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    sys.path[:0]=[str(VENDOR),str(VENDOR/'runtime/current25')]
    import fe_codec as fe
    from fe_lexicon import Lexicon,PredictedEntries
    from project.src.frontend import romance
    from project.src.frontend.romance_five_profile import LexicalStressFrontend
    lex=Lexicon(ROOT/'resources/packages/es-ES.fep',None,VENDOR/'frontend/libs')
    document=lex.document();path=ROOT/'host_reference/spanish_lexicon.json';path.write_text(json.dumps(document,ensure_ascii=False))
    front=LexicalStressFrontend('es',path);front.entries=PredictedEntries(front.entries,lex)
    vocabulary=json.loads((VENDOR.parent.parent/'int8/assets/vocabulary.json').read_text())
    words=lex.meta['symbols'];rows=['#ifndef N25_SPANISH_TABLES_H','#define N25_SPANISH_TABLES_H',
        '#define SPANISH_MODEL_BYTES '+str(len(lex.package.read('model'))),
        '#define SPANISH_SYMBOL_COUNT '+str(len(words)),
        'typedef struct{unsigned id;float features[36];}SpanishPhoneConstant;',
        'static const SpanishPhoneConstant spanish_phones[]={']
    for word in words+['sil']:
        features=romance.encode_ipa(word)
        rows.append('{'+str(vocabulary.get(word,1))+',{'+','.join(float(x).hex()+'f'for x in features)+'}},')
    rows += ['};','static const uint32_t spanish_open_alphabet[]={'+','.join(str(ord(x))for x in sorted(lex.policy['alphabet']))+'};',
        'static const unsigned char spanish_default_record[]={'+','.join(str(x)for x in lex.policy['default_record'])+'};','#endif']
    (dest/'dictionary_runtime/spanish_tables.h').write_text('\n'.join(rows)+'\n')
    for name in ('predict_reader.c','predict_reader.h'):
        shutil.copy2(WORK/'m4_frontend_4mb_20261022/support'/name,dest/'dictionary_runtime'/name)
    p=WORK/'m4_frontend_lossy_20261023/open_predict.c'
    (dest/'dictionary_runtime/open_predict.c').write_text(p.read_text().replace(
        '"../m4_frontend_4mb_20261022/support/predict_reader.h"','"predict_reader.h"'))
    build_four.LANGUAGES+='et fi hr lv es'.split();build_four.generate_tables()
    p=ROOT/'generated/four_tables.c';s=p.read_text()
    for language in ('et','fi','hr','lv'):s=s.replace('{"'+language+'","'+language+'",36}', '{"'+language+'","'+language+'",32}')
    p.write_text(s)
    cases=json.loads((ROOT/'generated/FOUR_FEATURE_IDENTITY.json').read_text())['cases']
    old=json.loads((ROOT/'generated/LATVIAN_FEATURE_CASES.json').read_text())['cases'];existing={(x['language'],x['text'])for x in cases}
    cases += [x for x in old if (x['language'],x['text'])not in existing]
    texts=['Hola, ¿cómo estás?', 'La niña tiene una bicicleta.', 'Quiero escuchar una prueba.',
        'Esto es nuevo, pero funciona.', 'El quijotesco motor funciona.', 'Sol; luna: mar.',
        'Dos-tres palabras.', 'La voz es clara.']
    for text in texts:
        f=front.encode(text);arrays={'phones':np.asarray([vocabulary.get(p,1)for p in f['ipa']],dtype='<u4'),
            'pause':np.asarray(f['edge_sil_mask'],dtype='<u4'),'phonetic':np.asarray(f['phonetic_features'],dtype='<f4'),
            'context':np.asarray(f['context_features'],dtype='<f4')}
        cases.append(dict(language='es',text=text,count=len(f['ipa']),dim=36,
            sha256={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in arrays.items()}))
    lines=['#ifndef N25_FOUR_CASES_H','#define N25_FOUR_CASES_H',
        'typedef struct{const char*language,*text;unsigned count,dim,context_dim;unsigned char sha256[4][32];}FourCase;',
        'static const FourCase four_cases[]={']
    for row in cases:
        digests=','.join('{'+','.join(str(b)for b in bytes.fromhex(row['sha256'][k]))+'}'for k in ('phones','pause','phonetic','context'))
        dim=34 if row['language']in ('fi','et','hr','lv')else 22
        lines.append('{'+','.join(json.dumps(row[k],ensure_ascii=False)for k in ('language','text'))+','+str(row['count'])+','+str(row['dim'])+','+str(dim)+',{'+digests+'}},')
    lines+=['};','#endif'];(ROOT/'generated/four_cases.h').write_text('\n'.join(lines)+'\n')
    write(ROOT/'generated/SPANISH_FEATURE_CASES.json',dict(cases=cases,
        canonical_source_sha256=sha(VENDOR/'runtime/current25/project/src/frontend/romance.py'),
        stress_source_sha256=sha(VENDOR/'runtime/current25/project/src/frontend/romance_five_profile.py'),
        full_dictionary_words=len(lex.records),source_package_sha256=sha(ROOT/'resources/packages/es-ES.fep')))
    identity=json.loads((ROOT/'LATVIAN_SOURCE_IDENTITY.json').read_text())
    text=json.loads((WORK/'m55_fargan_stream_20261030/language25_20261030/inputs/es/job.json').read_text())['text']
    identity['native_text_cases'].append(dict(language='es',language_id=6,text=text));identity['native_text_cases'].sort(key=lambda x:x['language_id'])
    identity['native_text_language_count']=17;identity['scope']='Native ordinary text for 17 families; full normalization and 8 remaining families pending.'
    write(ROOT/'SPANISH_SOURCE_IDENTITY.json',identity)
    p=dest/'src/multilang_driver.inc';s=p.read_text();a=s.index(' static const struct');b=s.index(' for(unsigned i=',a)
    rows=[' static const struct {unsigned language;const char *text;} cases[]={']
    for row in identity['native_text_cases']:rows.append('{'+str(row['language_id'])+','+json.dumps(row['text'],ensure_ascii=False)+'},')
    rows.append(' };\n');p.write_text(s[:a]+'\n'.join(rows)+s[b:])
    p=dest/'src/multilang_frontend.c';p.write_text(p.read_text().replace('{4,"el"},','{4,"el"},\n {6,"es"},'))

if __name__=='__main__':main()
