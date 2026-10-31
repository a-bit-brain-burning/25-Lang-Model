"""Project complete BG/SL text prosody lexicons into bounded native pages."""
import hashlib
import importlib
import importlib.util
import json
import re
import shutil
import sys
import unicodedata
import zlib
from build_resources import ROOT,write,sha
from build_espeak import VENDOR
from prepare_romance_live import manifest,driver
def varint(n):
    out=bytearray()
    while n>=128:out.append((n&127)|128);n>>=7
    out.append(n);return bytes(out)
def record(entry):
    p=entry['prosody'];signature=p['nucleus_signature'];n=p['nucleus_count'];assert n==len(signature)
    families='aeiouər';raw=bytearray([n,p['stress_syllable']+1 if entry.get('stress_known')else 0]);raw+=bytes(families.index(c)for c in signature)
    vowels=p['vowels'];raw.append(len(vowels))
    for v in vowels:
        name=v['ipa'].encode();flags=int(v.get('quantity_known',False))+2*int(v.get('quality_known',False))
        raw+=bytes([v['syllable'],flags,len(name)])+name
    return bytes(raw)
def main():
    source=ROOT/'native_romance_four';dest=ROOT/'native_south_slavic';dest.mkdir(exist_ok=False)
    for name in('src','assets','frontend_patched','dictionary_runtime','resources'):shutil.copytree(source/name,dest/name)
    for name in('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):(dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in(dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    profile=json.loads((source/'resource_profile.json').read_text());profile['exclude']=['bg-BG.fep','sl-SI.fep'];write(dest/'resource_profile.json',profile)
    code=['#include "paged_lexicon.h"'];descriptors=[];identity=[]
    for language in('bg','sl'):
        path=ROOT/'host_reference'/('bg_expanded_lexicon.json'if language=='bg'else'sl_lexicon.json');doc=json.loads(path.read_text());entries=sorted(doc['entries'].items())
        blob=bytearray();pages=[];checked=0
        for start in range(0,len(entries),64):
            group=entries[start:start+64];raw=bytearray(varint(len(group)));previous=b'';want=[]
            for word,value in group:
                key=word.encode();prefix=0
                while prefix<min(len(key),len(previous))and key[prefix]==previous[prefix]:prefix+=1
                data=record(value);raw+=varint(prefix)+varint(len(key)-prefix)+key[prefix:]+varint(len(data))+data;previous=key;want.append((word,data))
            stream=zlib.compressobj(9,zlib.DEFLATED,-15);packed=stream.compress(raw)+stream.flush();assert zlib.decompress(packed,-15)==raw
            # Check every projected functional field, not only the smoke text.
            for (word,value),(same,data)in zip(group,want):assert word==same and data==record(value);checked+=1
            pages.append((group[0][0],len(blob),len(packed),len(raw)));blob+=packed
        code.append(f'static const unsigned char {language}_data[]={{'+','.join(map(str,blob))+'};')
        code.append(f'static const N25LexiconPage {language}_pages[]={{')
        code += ['{'+json.dumps(key,ensure_ascii=False)+','+','.join(map(str,(offset,packed,raw)))+'},'for key,offset,packed,raw in pages];code+=['};']
        digest=','.join(str(x)for x in hashlib.sha256(blob).digest())
        descriptors.append('{'+json.dumps(language)+f',{language}_data,sizeof({language}_data),{language}_pages,'+str(len(pages))+',{'+digest+'}},')
        identity.append(dict(language=language,full_entries=len(entries),functional_records_checked=checked,packed_bytes=len(blob),
            page_count=len(pages),maximum_page_bytes=max(x[3]for x in pages),source_sha256=sha(path)))
    code+=['const N25PagedLexicon n25_paged_lexicons[]={',*descriptors,'};','const unsigned n25_paged_lexicon_count=sizeof(n25_paged_lexicons)/sizeof(n25_paged_lexicons[0]);']
    (dest/'dictionary_runtime/south_lexicons.c').write_text('\n'.join(code)+'\n')
    # Extend only Cyrillic letters required by the actual Bulgarian alphabet.
    sys.path[:0]=[str(VENDOR),str(VENDOR/'runtime/current25')]
    runtime=VENDOR/'models/G5/systems/sl/frontend/sl/runtime';name='_n25_south_constants'
    spec=importlib.util.spec_from_file_location(name,runtime/'__init__.py',submodule_search_locations=[str(runtime)]);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    m=importlib.import_module(name+'.frontend.eu_south_slavic')
    rows=['typedef struct{const char*name,*ipa;}SouthName;','static const SouthName south_names[][64]={']
    for tag in('sl-SI','bg-BG'):
        names=m._NAMES[tag];assert len(names)<=64;rows.append('{'+','.join('{'+json.dumps(k)+','+json.dumps(v,ensure_ascii=False)+'}'for k,v in sorted(names.items()))+'},')
    rows+=['};','static const unsigned south_name_counts[]={'+str(len(m._NAMES['sl-SI']))+','+str(len(m._NAMES['bg-BG']))+'};']
    for language,tag in [('sl','sl-SI'),('bg','bg-BG')]:rows.append(f'static const uint32_t {language}_alphabet[]={{'+','.join(str(ord(c))for c in sorted(m._ALPHABET[tag]))+'};')
    (dest/'src/south_names.h').write_text('\n'.join(rows)+'\n')
    p=ROOT/'generated/four_tables.c';s=p.read_text();a=s.index('const N25FourCharacter');b=s.index('const N25FourLanguage',a)
    chunk=s[a:b];old=re.findall(r'\{(\d+),(\d+),(\d+)\}',chunk);chars={int(cp):(int(lo),int(script))for cp,lo,script in old}
    for c in m._ALPHABET['bg-BG']:
        for ch in(c,c.upper()):chars[ord(ch)]=(ord(c),3)
    s=s[:a]+'const N25FourCharacter n25_four_characters[N25_FOUR_CHARACTER_COUNT]={\n'+''.join('{'+','.join(map(str,(cp,*row)))+'},\n'for cp,row in sorted(chars.items()))+'};\n'+s[b:];p.write_text(s)
    p=ROOT/'generated/four_tables.h';p.write_text(re.sub(r'#define N25_FOUR_CHARACTER_COUNT \d+',f'#define N25_FOUR_CHARACTER_COUNT {len(chars)}',p.read_text()))
    d=manifest('ROMANCE_FOUR_SOURCE_IDENTITY.json','SOUTH_SLAVIC_SOURCE_IDENTITY.json',['bg','sl']);driver(dest,d)
    write(ROOT/'SOUTH_LEXICON_PROJECTION_IDENTITY.json',dict(lexicons=identity,source_records_are_not_sentence_answers=True,
        projection='All retained entries, exact functional stress/signature/quality/quantity fields; provenance/evidence remains in archived source JSON',
        original_resources_preserved=True));print(identity)
if __name__=='__main__':main()
