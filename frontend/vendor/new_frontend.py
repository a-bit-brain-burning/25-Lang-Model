"""Explicit no/tr extension of the pinned numeric36/context22 frontend.

The original frontend and original language registries are never modified on disk.
Unsupported IPA is rejected. Retroflex and central rounded vowels retain their
existing numeric articulatory coordinates; tap/trill and round remain explicit.
"""
import ctypes,hashlib,importlib,re,sys,unicodedata
from functools import lru_cache
import numpy as np
from common import ROOT,HERE,sha

@lru_cache(maxsize=1)
def module():
 sys.path.insert(0,str(ROOT))
 from runtime.loader import ModelRuntime
 r=ModelRuntime(ROOT,cache_dir=HERE/'cache',threads=1)
 r._system('de-DE')[3].encode('de','Guten Tag.',{})
 m=importlib.import_module('_tts25_frozen_de.frontend.eu_four')
 for tag,voice in [('nb-NO','nb'),('tr-TR','tr')]:
  m.LANGUAGES[tag]=voice;m._ALIASES.update({tag.lower():tag,voice:tag})
 m._ALIASES['no']='nb-NO'
 for ipa,manner,place,voiced,lateral,rhotic in [
  ('ɭ','approximant','retroflex',True,True,False),('ɳ','nasal','retroflex',True,False,False),
  ('ʈ','stop','retroflex',False,False,False),('ɖ','stop','retroflex',True,False,False),
  ('ɽ','approximant','retroflex',True,False,True)]:
  m._BASES[ipa]=m.PhoneticSegment(ipa,manner,place,voiced,lateral=lateral,rhotic=rhotic)
 m._BASES['ʉ']=m.PhoneticSegment('ʉ','vowel',voiced=True,vowel_start=(0.,.5,1.))
 m._BASES['ɘ']=m.PhoneticSegment('ɘ','vowel',voiced=True,vowel_start=(.33,.5,0.))
 m.FRONTEND_VERSION='grouped11_no_tr_explicit_v1'
 original_backend=m.EspeakBackend
 class NewBackend(original_backend):
  @lru_cache(maxsize=65536)
  def phonemize(self,word,voice):
   if voice not in ('nb','tr'):return super().phonemize(word,voice)
   assert self.library,'Explicit library backend required'
   with m._LOCK:
    self._select_voice(voice)
    data=ctypes.create_string_buffer(word.encode());pointer=ctypes.c_void_p(ctypes.addressof(data));clauses=[]
    for _ in range(1000):
     if not pointer.value:break
     previous=pointer.value
     value=self.lib.espeak_TextToPhonemes(ctypes.byref(pointer),1,2|(ord('_')<<8))
     clauses.append(value.decode() if value else '')
     if pointer.value==previous:raise ValueError('eSpeak cursor did not advance')
    else:raise ValueError('eSpeak clause limit exceeded')
   raw=' '.join(clauses).strip()
   if not raw or re.search(r'\([^)]*\)',raw):raise ValueError(f'Empty IPA or language switch: {word!r}/{raw!r}')
   return raw
 m.EspeakBackend=NewBackend
 return m

@lru_cache(maxsize=2)
def frontend(language):
 import espeakng_loader
 m=module()
 return m.FourLanguageFrontend('nb-NO' if language=='no' else 'tr-TR',pause_policy='word',
  espeak_library=espeakng_loader.get_library_path(),espeak_data=espeakng_loader.get_data_path())

def encode(language,text,metadata=None):
 assert language in ('no','tr')
 front=frontend(language);f=front.encode(text);records=f['words']
 f['word_records']=records;f['words']=[r['normalized_word'] for r in records]
 f['word_phone_ranges']=np.asarray([(r['phone_unit_indices'][0],r['phone_unit_indices'][-1]+1) for r in records],np.int64)
 f['alignment_words']=[r['lexicon_key'] for r in records]
 x=np.asarray(f['phonetic_features']);extra=np.zeros((len(x),4),np.float32)
 for i,p in enumerate(f['ipa']):
  core=unicodedata.normalize('NFD',p).translate(str.maketrans('','','ːˑʰʲˠ̥̬̪̩̯̃'))
  if core in ('ɾ','ɽ'):extra[i,0]=1.;x[i,6]=0.
  if core=='r':extra[i,1]=1.;x[i,6]=0.
  if core in ('ɕ','tɕ','ʑ','dʑ'):extra[i,2]=1.
  if 'ʷ' in p or core in ('w','ɥ'):extra[i,3]=1.
 f['phonetic_features']=np.concatenate((x,extra),axis=1)
 lexical=np.asarray(f['word_indices'],np.int64);pause=np.asarray(f['ipa'])=='sil';owner=lexical.copy();left=-1
 for i,v in enumerate(lexical):
  if v>=0:left=int(v)
  elif pause[i] and i not in (0,len(lexical)-1):owner[i]=left
 f['lexical_word_indices']=lexical;f['plan_owner']=owner
 f['internal_pause_mask']=pause.copy();f['internal_pause_mask'][[0,-1]]=False
 f['vowel_mask']=f['phonetic_features'][:,31]>.5
 f['phonetic_feature_names']=list(f['phonetic_feature_names'])+['manner_tap','manner_trill','place_alveolopalatal','consonant_round']
 from project.src.models.romance_word_planner import encode_frontend
 plan=encode_frontend(f)
 f.update(planner_numeric=plan['numeric'],planner_boundary=plan['boundary'],planner_owner=plan['plan_owner'])
 signature=hashlib.sha256((front.signature+sha(__file__)).encode()).hexdigest() if hasattr(front,'signature') else hashlib.sha256((f['metadata']['semantic_identity_sha256']+sha(__file__)).encode()).hexdigest()
 f['metadata'].update(frontend_signature=signature,ordinary_text_only=True,adapter_sha256=sha(__file__),language=language)
 return f

