"""Host lookup over the user's native-decoded predictor/exception resources."""
import ctypes as C,collections,json,tempfile,unicodedata
from pathlib import Path
from fe_codec import NativePack,Codec,NativePredictor,Values,raw_tables,unpack_metadata,decode_record,english_tokens

class Prediction:
 def __init__(self,name,symbols,backend,lt_module=None):
  self.name=name;self.symbols=symbols;self.ids={s:i for i,s in enumerate(symbols)};self.backend=backend;self.lt=lt_module
  if name=='lt-LT':self.lt_backend=lt_module.EspeakBackend(library=backend.library,data=backend.data,required_version='1.52.0')
 def alternatives(self,word):
  if self.name=='en-CMU':return[english_tokens(self.backend.word(word,'en-us'),self.ids)]
  if self.name=='fr-FR':
   parts=[p for p in self.backend.word(word,'fr').replace(' ','_').split('_')if p]
   return[bytes(self.ids.get(p.translate(str.maketrans('','','ˈˌ'+marks)),254)for p in parts)for marks in('ː','')]
  raw=self.lt_backend.phonemize(word,'lt');names=self.lt._lithuanian_phoneme_names(self.lt_backend,word)
  phones,_=self.lt._parse_backend(raw,'lt-LT',word,native_names=names)
  return[bytes(self.ids.get(('ˈ'if stress and p.stress==1 else'')+p.ipa,self.ids.get(p.ipa,254))for p in phones)for stress in(True,False)]

def admissible(word,alphabet):return 3<=len(word)<=64 and word.isalpha()and word==word.lower()and all(c in alphabet for c in word)

class Lexicon:
 def __init__(self,path,backend,libs,lt_module=None):
  self.name=Path(path).stem;docs,self.package=raw_tables(path,libs/'libreader.so');doc=docs[self.name]
  self.records=doc['entries'];self.meta=unpack_metadata(doc['metadata'])
  self.policy=Codec([]).decode(self.package.read('open_policy'))if'open_policy'in self.package.sections else None
  self._temp=tempfile.TemporaryDirectory(prefix='compact-lexicon-');model=Path(self._temp.name)/'model.spr';model.write_bytes(self.package.read('model'))
  self.native=NativePredictor(model,libs/'libfrontend4.so');self.values=Values(libs/'libfrontend4.so')
  self.aux=self.package.read('auxiliary')if'auxiliary'in self.package.sections else None
  self.predictor=Prediction(self.name,self.meta['symbols'],backend,lt_module)if self.name in('fr-FR','lt-LT')else None
  self.cache=collections.OrderedDict();self.open=C.CDLL(str(libs/'libopen_predict.so'))
  from fe_codec import View
  self.open.open_predict.argtypes=[C.POINTER(View),C.c_void_p,C.c_size_t,C.POINTER(C.c_uint32),C.c_size_t,
   C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
  self.alphabet=self.policy['alphabet']if self.policy else'';self.characters=(C.c_uint32*len(self.alphabet))(*(ord(c)for c in self.alphabet))
 def raw(self,word):
  record=self.records.get(word)
  if record is None:
   if not self.policy or not admissible(word,self.alphabet):raise KeyError(word)
   key=word.encode();n=C.c_size_t();default=self.policy['default_record']
   status=self.open.open_predict(C.byref(self.native.view),key,len(key),self.characters,len(self.characters),default,len(default),
    self.native.pred,len(self.native.pred),self.native.output,len(self.native.output),C.byref(n))
   if status:raise ValueError(('compressed open predictor',status,word))
   raw=self.native.output.raw[:n.value]
  else:
   mode=record[0]>>4
   if mode:
    if self.predictor is None or mode>2:raise ValueError('Invalid predictor selection')
    prediction=self.predictor.alternatives(word)[mode-1];record=bytes([record[0]&15])+record[1:]
   else:prediction=self.native.predict(word)
   raw=self.native.apply(prediction,record)
  if self.aux:raw=self.values.aux(self.aux,raw)
  end=raw.index(255)
  if not end or any(v>=len(self.meta['symbols'])for v in raw[:end]):raise ValueError('Invalid compressed phone sequence')
  return raw
 def lookup(self,word):
  if word in self.cache:self.cache.move_to_end(word);return self.cache[word]
  result=decode_record(self.raw(word),self.meta);self.cache[word]=result
  if len(self.cache)>256:self.cache.popitem(last=False)
  return result
 def document(self):
  entries={};overlays={k:{}for k in self.meta['overlay']}
  words=list(self.records)
  if self.meta.get('entry_order')is not None:words=[words[i]for i in self.meta['entry_order']]
  for word in words:
   entry,extra=self.lookup(word);entries[word]=entry
   for k,v in extra.items():overlays[k][word]=v
  result={**self.meta['skeleton'],'entries':entries,**overlays}
  return{k:result[k]for k in self.meta['top_order']}

class PredictedEntries(dict):
 def __init__(self,entries,lexicon):super().__init__(entries);self.lexicon=lexicon
 def __contains__(self,word):return super().__contains__(word)or(self.lexicon.policy is not None and isinstance(word,str)and admissible(word,self.lexicon.alphabet))
 def __getitem__(self,word):
  if super().__contains__(word):return super().__getitem__(word)
  entry,extra=self.lexicon.lookup(word)
  if extra:raise ValueError('Unexpected generated overlays')
  return tuple(entry)
 def get(self,word,default=None):
  try:return self[word]
  except KeyError:return default
