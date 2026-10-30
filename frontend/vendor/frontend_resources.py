"""Verified FEP resources and host frontend binding; no original dictionary fallback."""
import copy,hashlib,importlib,importlib.util,json,sys,tempfile,types
from pathlib import Path
import numpy as np
import espeakng_loader
import fe_codec as fe
from fe_lexicon import Lexicon,PredictedEntries,Prediction

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Resources:
 def __init__(self,owner):
  self.owner=owner;self.root=owner.root;self.path=Path(tempfile.mkdtemp(prefix='int8-compact-frontend-'));self.libs=self.root/'frontend/libs';self.packages=self.root/'frontend/packages'
  self.manifest=read(self.root/'frontend/MANIFEST.json');self.identity=hashlib.sha256((self.manifest['source_candidate_id']+sha(self.packages/'integration-supplement.fep')).encode()).hexdigest()
  for name,digest in self.manifest['packages'].items():assert sha(self.packages/name)==digest,name
  for name,digest in self.manifest['libraries'].items():assert sha(self.libs/name)==digest,name
  self.sessions={};self.lexicons={};self.adapters={};self.documents={};self.counters={};self._extract()
  self.library=str(self.libs/'libespeak_phonemes.so');self.data=str(self.path/'espeak-ng-data')
  for name in('no_dict','tr_dict','pt_dict','it_dict'):assert(Path(self.data)/name).is_file(),('Required active G2P dependency absent',name)
  espeakng_loader.get_library_path=lambda:self.library;espeakng_loader.get_data_path=lambda:self.data
  self.backend=fe.Backend(Path(self.data));self.backend.library=self.library;self.backend.data=self.data
 def _extract(self):
  p=fe.NativePack(self.packages/'espeak.fep',self.libs/'libreader.so');meta=fe.Codec([]).decode(p.read('files'));chunk=None;number=None
  for row in meta['files']:
   name=Path(row['name']);assert not name.is_absolute()and'..'not in name.parts
   out=self.path/'espeak-ng-data'/name;out.parent.mkdir(parents=True,exist_ok=True);offset=row['offset'];remaining=row['bytes'];blob=bytearray()
   while remaining:
    i,pos=divmod(offset,meta['chunk'])
    if i!=number:chunk=p.read('part/'+str(i));number=i
    part=chunk[pos:pos+remaining];assert part;blob.extend(part);offset+=len(part);remaining-=len(part)
   assert hashlib.sha256(blob).hexdigest()==row['sha256'];out.write_bytes(blob)
  p=fe.NativePack(self.packages/'integration-supplement.fep',self.libs/'libreader.so')
  for row in fe.Codec([]).decode(p.read('manifest')):
   blob=b''.join(p.read(key)for key in row.get('sections',[row.get('section')]));assert hashlib.sha256(blob).hexdigest()==row['sha256'];out=self.path/row['path'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(blob)
 def module(self,l):
  if l not in self.adapters:
   e=next(e for e in self.owner.index['models']if e['short']==l);bundle=self.root/e['bundle'];meta=read(bundle/'frontend_metadata.json')
   self.adapters[l]=self.owner.trainer.load_adapter(bundle/meta['adapter_path'],meta['adapter_sha256'])
  return self.adapters[l]
 def write_document(self,name,document):
  path=self.path/(name+'.json');path.write_text(json.dumps(document,ensure_ascii=False),encoding='utf-8');return path
 def lexicon(self,l):
  if l not in self.lexicons:
   locale={'lt':'lt-LT','es':'es-ES','fr':'fr-FR','it':'it-IT','pt':'pt-PT','ro':'ro-RO'}[l]
   module=self.module('lt').frozen_module('lt','eu_baltic')if l=='lt'else None
   self.lexicons[l]=Lexicon(self.packages/(locale+'.fep'),self.backend,self.libs,module)
  return self.lexicons[l]
 def romance(self,l):
  from project.src.frontend.romance_dispatch import RomanceFrontend
  lex=self.lexicon(l);path=self.write_document(l+'-compressed',lex.document());front=RomanceFrontend(l,path)
  if lex.policy:front.entries=PredictedEntries(front.entries,lex)
  return front,path
 def base_adapter(self,l,spec,meta):
  meta=copy.deepcopy(meta);meta['espeak_library']=self.library;meta['espeak_data']=self.data
  if l in('es','fr','it','pt','ro'):
   front,path=self.romance(l);meta['lexicon_path']=str(path);meta['lexicon_sha256']=sha(path)
   from project.src.models.romance_word_planner import encode_frontend
   def encode(language,text,metadata=None):
    f=front.encode(text)
    if l=='pt':self.contextual_o(f,text,front)
    p=encode_frontend(f);f.update(planner_numeric=p['numeric'],planner_boundary=p['boundary'],planner_owner=p['plan_owner']);return f
   adapter=types.SimpleNamespace(encode=encode,front=front)
  elif l in('lt','bg','sl'):
   if l=='lt':document=self.lexicon(l).document()
   else:document,_,_=fe.restore_bgsl(self.packages/('bg-BG.fep'if l=='bg'else'sl-SI.fep'),self.libs/'libreader.so')
   path=self.write_document(l+'-compressed',document);meta['lexicon_path']=str(path);meta['lexicon_sha256']=sha(path)
   original=self.module(l)
   # This factory uses its frozen relative lexicon path; point its ROOT at a
   # temporary code/resource layout containing only the decoded compressed table.
   entry={'lt':'eu_baltic','bg':'eu_south_slavic','sl':'eu_south_slavic'}[l];m=original.frozen_module(l,entry)
   cls=m.BalticFrontend if l=='lt'else m.SouthSlavicFrontend
   front=cls(l,espeak_library=self.library,espeak_data=self.data,pause_policy='word',lexicon_path=str(path),lexicon_sha256=sha(path))
   original.get_frontend=lambda *a,**kw:front
   adapter=original
  else:
   if l=='en':self.install_english()
   if l=='zh':self.install_chinese()
   adapter=self.module(l)
  return meta,adapter
 def contextual_o(self,f,text,front):
  source=self.module('pt');review=source.decisions(text,front);ipa=list(f['ipa'])
  from project.src.frontend.romance import encode_ipa
  for d in review:
   if d['action']!='CONTEXTUAL_U_UNSTRESSED':continue
   lo,hi=map(int,f['word_phone_ranges'][d['word_index']]);assert hi-lo==1 and ipa[lo]=='o'
   ipa[lo]='u';f['phonetic_features'][lo]=encode_ipa('u');f['stress'][lo]=0;f['stress_known'][lo]=True;f['context_features'][lo,12]=0.;f['context_features'][lo,13]=1.
  f['ipa']=ipa;f['phones']=list(ipa);f['phonetic']=f['phonetic_features'];f['context']=f['context_features'];f['text_features']=f['context_features'][:,:12].copy()
  f['vowel_mask']=f['phonetic_features'][:,1]>.5;f['voiced_phone_mask']=f['phonetic_features'][:,17]>.5;f['metadata']['contextual_o_decisions']=review
 def repair(self,l,base,meta):
  support=self.root/'runtime/repair_support';sys.path.insert(0,str(support));key='_int8_compact_repair_frontends'
  if key not in sys.modules:
   sp=importlib.util.spec_from_file_location(key,support/'repair_frontends.py');m=importlib.util.module_from_spec(sp);sys.modules[key]=m;sp.loader.exec_module(m)
  m=sys.modules[key];m.HERE=support;m.BASE=support/'base';owner=self
  if l=='bg':import bg_frontend;bg_frontend.HERE=self.path
  class Proxy:
   def _system(self,locale):
    if locale=={'it':'it-IT','pt':'pt-PT','bg':'bg-BG'}[l]:return(None,self_spec,meta,base,None,None)
    return owner.owner._system(locale)
  self_spec=self.owner.layout['systems'][l]['spec'];front=m.Frontend(Proxy(),l)
  if l in('it','pt'):front.impl.base=base.front
  return front
 def system(self,entry,spec,metadata):
  l=entry['short']
  if l in self.sessions:return self.sessions[l]
  base_meta=metadata
  bundle=self.root/entry['bundle'];source=bundle/'source_frontend_metadata.json'
  if metadata.get('frontend_override')and source.exists():base_meta=read(source)
  meta,base=self.base_adapter(l,spec,base_meta)
  adapter=self.repair(l,base,meta)if metadata.get('frontend_override')=='native_MFA_repair_v1'else base
  signature=hashlib.sha256((metadata['frontend_signature']+self.identity+l+sha(__file__)).encode()).hexdigest()
  provider=self
  class Bound:
   def encode(self,language,text,selected):
    assert language==l and selected['frontend_signature']==signature
    f=adapter.encode(language,text,meta);f=copy.deepcopy(f)
    f.setdefault('metadata',{}).update(frontend_signature=signature,baseline_frontend_signature=metadata['frontend_signature'],compressed_frontend_candidate_id=provider.identity,
     experimental_frontend=True,ordinary_text_only=True,original_full_dictionary_fallback=False)
    provider.counters[l]=provider.counters.get(l,0)+1;return f
  result=({**metadata,'frontend_signature':signature,'baseline_frontend_signature':metadata['frontend_signature'],'compressed_frontend':True},Bound())
  self.sessions[l]=result;return result
 def install_english(self):
  if 'english'in self.documents:return
  raw,p=fe.raw_tables(self.packages/'en-CMU.fep',self.libs/'libreader.so');doc=raw['en-CMU'];tmp=self.path/'en-model.spr';tmp.write_bytes(p.read('model'))
  native=fe.NativePredictor(tmp,self.libs/'libfrontend4.so');values=fe.Values(self.libs/'libfrontend4.so');prediction=Prediction('en-CMU',doc['symbols'],self.backend);entries={}
  for word,record in doc['entries'].items():
   if record[0]==2:
    restored=native.apply(prediction.alternatives(word)[0],record[1:]);end=restored.index(255);seq=restored[:end];stress=bytearray((len(seq)+3)//4)
    for i,v in enumerate(seq):stress[i//4]|=(v&3)<<(2*(i%4))
    value=bytes([0,len(seq)])+bytes(v//4 for v in seq)+stress
   else:value=values.en(native.predict(word),record)
   entries[word]=value
  a,b=fe.decode_english({**doc,'entries':entries});self.documents['english']=(a,b)
  from project.src.frontend import english
  original=english.TextDictionary
  class Dictionary(original):
   def __init__(self,path=None):self.path=Path('en-CMU.fep');self.entries=a;self.stress_entries=b
  english.TextDictionary=Dictionary;english.default_dictionary.cache_clear()
 def install_chinese(self):
  if 'chinese'in self.documents:return
  chars,_=fe.raw_tables(self.packages/'zh-characters.fep',self.libs/'libreader.so');chars=chars['zh-characters']
  shared,_=fe.raw_tables(self.packages/'zh-shared.fep',self.libs/'libreader.so');phrases=shared['zh-phrases'];values=fe.Values(self.libs/'libfrontend4.so')
  primary={w:values.primary(v,phrases['primary_id_translation'])for w,v in chars['entries'].items()}
  entries={w:values.phrase([primary.get(c,65535)for c in w],v)for w,v in phrases['entries'].items()}
  phrase_dict=fe.restore_chinese('phrases',{**phrases,'entries':entries});char_dict=fe.restore_chinese('characters',chars)
  freq,tags,total=fe.restore_chinese('jieba',shared['zh-jieba'])
  import jieba,jieba.posseg as posseg,pypinyin.constants as constants
  oldchars,oldphrases=constants.PINYIN_DICT,constants.PHRASES_DICT
  for name,module in list(sys.modules.items()):
   if name.startswith('pypinyin')and module:
    for key,value in list(vars(module).items()):
     if value is oldchars:setattr(module,key,char_dict)
     elif value is oldphrases:setattr(module,key,phrase_dict)
  def initialize(tokenizer,*args,**kwargs):tokenizer.FREQ=freq;tokenizer.total=total;tokenizer.initialized=True
  jieba.Tokenizer.initialize=initialize
  for tok in(jieba.dt,posseg.dt.tokenizer):initialize(tok)
  posseg.dt.word_tag_tab=tags
  p=fe.NativePack(self.packages/'frontend-supplement.fep',self.libs/'libreader.so');sources=fe.Codec([]).decode(p.read('sources'))
  resource=self.path/'chinese-assets';resource.mkdir(exist_ok=True)
  for name,row in sources.items():
   blob=p.read(row['section']);assert hashlib.sha256(blob).hexdigest()==row['sha256'];(resource/name).write_bytes(blob)
  docs,_=fe.raw_tables(self.packages/'zh-supplement-phrases.fep',self.libs/'libreader.so');supp=docs['zh-supplement-phrases']
  primary={w:values.primary(v,supp['primary_id_translation'])for w,v in chars['entries'].items()}
  entries={}
  for word,value in supp['entries'].items():
   cells=fe.unpack_cells(values.phrase([primary.get(c,65535)for c in word],value),supp['strings']);assert all(len(c)==1 for c in cells);entries[word]=[c[0]for c in cells]
  (resource/'chinese_polyphone_phrases_v1.json').write_text(json.dumps({**supp['skeleton'],'entries':entries},ensure_ascii=False))
  from project.src.frontend import chinese_polyphone as poly,chinese_erhua as erhua
  for module,attribute,name in((poly,'ASSET','chinese_polyphone_phrases_v1.json'),(poly,'TRAIN_ASSET','chinese_polyphone_train_corrections_v1.json'),
    (poly,'SURNAME_ASSET','chinese_polyphone_surnames_v1.json'),(erhua,'LEXICON_PATH','chinese_erhua_train_v1.json')):
   setattr(module,attribute,resource/name)
  for module in(poly,erhua):
   for function in vars(module).values():
    if callable(function)and hasattr(function,'cache_clear'):function.cache_clear()
  self.documents['chinese']={'characters':len(char_dict),'phrases':len(phrase_dict),'jieba_words':len(freq)}
