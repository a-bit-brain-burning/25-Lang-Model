"""Export generic native-phoneme name mappings, not sentence pronunciations."""
import build_four
import importlib, importlib.util, sys, json
from pathlib import Path
from build_resources import ROOT,WORK,write,sha
from build_espeak import VENDOR

def generate():
 build_four.LANGUAGES += ['et','fi']
 build_four.generate_tables()
 p=ROOT/'generated/four_tables.c';s=p.read_text().replace('{"et","et",36}', '{"et","et",32}').replace('{"fi","fi",36}', '{"fi","fi",32}');p.write_text(s)
 runtime=VENDOR/'models/G7/systems/fi/frontend/fi/runtime';name='_n25_native_finnic'
 spec=importlib.util.spec_from_file_location(name,runtime/'__init__.py',submodule_search_locations=[str(runtime)])
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 m=importlib.import_module(name+'.frontend.eu_finnic')
 rows=['typedef struct{const char*name,*fi,*et;}FinnicName;','static const FinnicName finnic_names[]={']
 for key in sorted(set(m._FI_NAMES)|set(m._ET_NAMES)):
  rows.append('{'+','.join(json.dumps(x,ensure_ascii=False)if x is not None else'0'for x in (key,m._FI_NAMES.get(key),m._ET_NAMES.get(key)))+'},')
 rows.append('};')
 rows += ['static const char*et_diphthongs[]={'+','.join(json.dumps(x,ensure_ascii=False)for x in sorted(m._ET_DIPHTHONGS))+'};']
 (ROOT/'native_finnic/src/finnic_names.h').write_text('\n'.join(rows)+'\n')
 identity=json.loads((ROOT/'MULTILANG_SOURCE_IDENTITY.json').read_text())
 for language,language_id in [('et',7),('fi',8)]:
  text=json.loads((WORK/'m55_fargan_stream_20261030/language25_20261030/inputs'/language/'job.json').read_text())['text']
  identity['native_text_cases'].append(dict(language=language,language_id=language_id,text=text))
 identity['native_text_cases'].sort(key=lambda x:x['language_id']);identity['native_text_language_count']=14
 identity['scope']='Ordinary NFC Han subset, 11 eu_four and 2 Finnic families; full normalization and remaining 11 families pending.'
 identity['finnic_source_sha256']=sha(runtime/'frontend/eu_finnic.py')
 write(ROOT/'FINNIC_SOURCE_IDENTITY.json',identity)
 p=ROOT/'native_finnic/src/multilang_driver.inc';s=p.read_text();start=s.index(' static const struct');end=s.index(' for(unsigned i=',start)
 rows=[' static const struct {unsigned language;const char *text;} cases[]={']
 for row in identity['native_text_cases']:rows.append('{'+str(row['language_id'])+','+json.dumps(row['text'],ensure_ascii=False)+'},')
 rows.append(' };\n');p.write_text(s[:start]+'\n'.join(rows)+s[end:])
 # Probe SHA sizes depend on actual context dimensionality.
 p=ROOT/'generated/four_cases.h';s=p.read_text().replace('unsigned count,dim;','unsigned count,dim,context_dim;')
 for row in json.loads((ROOT/'generated/FOUR_FEATURE_IDENTITY.json').read_text())['cases']:
  needle=','+str(row['count'])+','+str(row['dim'])+',{'
  s=s.replace(needle,','+str(row['count'])+','+str(row['dim'])+','+str(34 if row['language']in('fi','et')else 22)+',{')
 p.write_text(s)
if __name__=='__main__':generate()
