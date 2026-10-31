"""Bind the original English/Lithuanian dictionaries and native phone tables."""
import importlib
import importlib.util
import json
import shutil
import sys
from build_resources import ROOT,write
from build_espeak import VENDOR
from prepare_romance_live import manifest,driver,add_cases,write_cases
def main():
    source=ROOT/'native_south_slavic';dest=ROOT/'native_final25';dest.mkdir(exist_ok=False)
    for name in('src','assets','frontend_patched','dictionary_runtime','resources'):shutil.copytree(source/name,dest/name)
    shutil.copy2(source/'resource_profile.json',dest/'resource_profile.json')
    for name in('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):(dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in(dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    # QSPI space released by replacing obsolete BG/SL containers holds these
    # functional pages. They remain included in ROM accounting and boot SHA.
    p=dest/'dictionary_runtime/south_lexicons.c';s=p.read_text()
    for name in('bg','sl'):s=s.replace(f'{name}_data[]={{',f'{name}_data[] __attribute__((section(".benchmark_assets.south_lexicon"),aligned(16)))={{')
    p.write_text(s)
    sys.path[:0]=[str(VENDOR),str(VENDOR/'runtime/current25')];import fe_codec as fe
    from project.src.frontend.universal_english import ARPABET_TO_IPA
    code=['#include "romance_record.h"'];rows=[]
    for locale,index in [('en-CMU',0),('lt-LT',4)]:
        docs,p=fe.raw_tables(ROOT/'resources/packages'/(locale+'.fep'),VENDOR/'frontend/libs/libreader.so');doc=docs[locale]
        meta=fe.unpack_metadata(doc['metadata'])if locale=='lt-LT'else doc;symbols=meta['symbols']
        code.append(f'static const char*const final_symbols_{index}[]={{'+','.join(json.dumps(x,ensure_ascii=False)for x in symbols)+'};')
        work=320000+max(v[3]for k,v in p.sections.items()if k.startswith(('keys/','pool/','ids/')))
        rows.append('{'+json.dumps(locale[:2])+','+','.join(map(str,(index,p.sections['model'][3],p.sections.get('auxiliary',[0,0,0,0])[3],work,len(symbols))))+f',final_symbols_{index},NULL,0,NULL,0'+'},')
        if locale=='en-CMU':
            h=['static const char*const english_symbols[]={'+','.join(json.dumps(x)for x in symbols)+'};',
                'static const char*const english_ipa[]={'+','.join(json.dumps(ARPABET_TO_IPA[x],ensure_ascii=False)for x in symbols)+'};',
                'typedef struct{const char*ipa;unsigned code;}EnglishPrediction;',
                'static const EnglishPrediction english_prediction_map[]={']
            for ipa,arp in sorted(fe.ARP.items()):h.append('{'+json.dumps(ipa,ensure_ascii=False)+','+str(symbols.index(arp))+'},')
            h+=['};'];(dest/'src/english_tables.h').write_text('\n'.join(h)+'\n')
    code+=['const RomanceDictionary final_dictionaries[]={',*rows,'};']
    (dest/'dictionary_runtime/final_dictionaries.c').write_text('\n'.join(code)+'\n')
    runtime=VENDOR/'models/G6/systems/lt/frontend/lt/runtime';name='_n25_lithuanian_constants'
    spec=importlib.util.spec_from_file_location(name,runtime/'__init__.py',submodule_search_locations=[str(runtime)]);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    m=importlib.import_module(name+'.frontend.eu_baltic')
    h=['static const SouthName lithuanian_names[]={']
    for token,ipa in sorted(m._LT_NAMES.items()):h.append('{'+json.dumps(token)+','+json.dumps(ipa,ensure_ascii=False)+'},')
    h+=['};','static const uint32_t lithuanian_alphabet[]={'+','.join(str(ord(c))for c in sorted(m._ALPHABET['lt-LT']))+'};']
    (dest/'src/lithuanian_tables.h').write_text('\n'.join(h)+'\n')
    d=manifest('SOUTH_SLAVIC_SOURCE_IDENTITY.json','FINAL25_SOURCE_IDENTITY.json',['en','lt']);driver(dest,d)
    cases=json.loads((ROOT/'generated/SOUTH_FEATURE_CASES.json').read_text())['cases'];add_cases(cases,['en','lt']);write_cases(cases)
    write(ROOT/'generated/FINAL25_FEATURE_CASES.json',dict(cases=cases))
if __name__=='__main__':main()
