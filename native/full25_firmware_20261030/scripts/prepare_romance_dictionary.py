"""Port the selected dictionary route, including the explicit PT/IT API override."""
import json
import shutil
import sys
from build_resources import ROOT,sha,write
from build_espeak import VENDOR
from prepare_romance_live import driver
def main():
    source=ROOT/'native_romance_live';dest=ROOT/'native_romance_dictionary';dest.mkdir(exist_ok=False)
    for name in('src','assets','frontend_patched','dictionary_runtime'):shutil.copytree(source/name,dest/name)
    for name in('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):(dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in(dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    sys.path.insert(0,str(VENDOR));import fe_codec as fe
    code=['#include "romance_record.h"']
    descriptors=[]
    for locale,index in [('es-ES',1),('it-IT',3),('pt-PT',5),('fr-FR',2),('ro-RO',6)]:
        docs,p=fe.raw_tables(ROOT/'resources/packages'/(locale+'.fep'),VENDOR/'frontend/libs/libreader.so');m=fe.unpack_metadata(docs[locale]['metadata'])
        policy=fe.Codec([]).decode(p.read('open_policy'))if'open_policy'in p.sections else None
        names=m['symbols'];code.append(f'static const char*const symbols_{index}[]={{'+','.join(json.dumps(n,ensure_ascii=False)for n in names)+'};')
        alphabet=sorted(policy['alphabet'])if policy else'';default=policy['default_record']if policy else b''
        code.append(f'static const uint32_t alphabet_{index}[]={{'+','.join(map(str,(ord(c)for c in alphabet)))+'};')
        code.append(f'static const unsigned char default_{index}[]={{'+','.join(map(str,default))+'};')
        work=320000+max(v[3]for k,v in p.sections.items()if k.startswith(('keys/','pool/','ids/')))
        descriptors.append('{'+json.dumps(locale[:2])+','+','.join(map(str,(index,p.sections['model'][3],p.sections.get('auxiliary',[0,0,0,0])[3],work,len(names))))+
            f',symbols_{index},alphabet_{index},'+str(len(alphabet))+f',default_{index},'+str(len(default))+'},')
    code+=['const RomanceDictionary romance_dictionaries[]={',*descriptors,'};','const unsigned romance_dictionary_count=sizeof(romance_dictionaries)/sizeof(romance_dictionaries[0]);']
    (dest/'dictionary_runtime/romance_dictionaries.c').write_text('\n'.join(code)+'\n')
    import shutil as sh
    for n in('value_decoders.c','value_decoders.h'):sh.copy2(ROOT.parents[1]/'m4_frontend_4mb_20261022/support'/n,dest/'dictionary_runtime'/n)
    p=dest/'src/four_frontend.c';s=p.read_text().replace('#include "romance_live.inc"','#include "romance_live.inc"\n#include "romance_dictionary.inc"')
    s=s.replace('return n25_romance_live(language,text,out);','return n25_romance_dictionary(language,text,out);');p.write_text(s)
    d=json.loads((ROOT/'ROMANCE_LIVE_SOURCE_IDENTITY.json').read_text());d['scope']='19 ordinary native dictionary families, including selected PT/IT original phoneme route; full normalization and six families pending.'
    write(ROOT/'ROMANCE_DICTIONARY_SOURCE_IDENTITY.json',d);driver(dest,d)
    write(ROOT/'ROMANCE_ROUTE_AUDIT.json',dict(selected_api_sha256=sha(VENDOR.parent/'api.py'),
        selected_route='frontend/api.py:_session explicitly calls base_adapter for pt and it; unused BR/IT repair candidate is not selected',
        failed_probe='romance_live_features_v2: count mismatch correctly rejected the unselected repair route',
        original_model_changed=False,training_updates=0))
if __name__=='__main__':main()
