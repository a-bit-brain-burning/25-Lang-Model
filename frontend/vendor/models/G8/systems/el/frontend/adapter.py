"""Opt-in text frontends frozen for tts25 scratch training; no acoustic language rules.

Existing Romance files remain untouched. Legacy EU snapshots retain their own
32-dimensional articulatory and 22/34 context contracts. Seven new languages
explicitly extend the frozen eSpeak text backend with a named numeric36 adapter.
"""
from __future__ import annotations
import hashlib,importlib,importlib.util,json,sys,unicodedata
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ROMANCE=('es','fr','it','pt','ro')
LEGACY={'hu':'eu_four','el':'eu_four','mt':'eu_four','ga':'eu_four','lt':'eu_baltic','lv':'eu_baltic','fi':'eu_finnic','et':'eu_finnic','hr':'eu_south_slavic','sl':'eu_south_slavic','bg':'eu_south_slavic'}
NEW={'cs':'cs-CZ','da':'da-DK','de':'de-DE','nl':'nl-NL','pl':'pl-PL','sk':'sk-SK','sv':'sv-SE'}
ESPEAK_LIBRARY='/home/fomak/ffb/.venv/lib/python3.10/site-packages/espeakng_loader/libespeak-ng.so'
ESPEAK_DATA='/home/fomak/ffb/.venv/lib/python3.10/site-packages/espeakng_loader/espeak-ng-data'
@lru_cache(maxsize=None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@lru_cache(maxsize=25)
def snapshot_sha(language):
 return {p.name:sha(p) for p in sorted((ROOT/'frontend'/language/'runtime/frontend').glob('*.py'))}

def frozen_module(language,entry):
 path=ROOT/'frontend'/language/'runtime';name='_tts25_frozen_'+language
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return importlib.import_module(name+'.frontend.'+entry)

def adapt_words(f):
 """Only structural names: preserve all numeric values, phones and pause units."""
 if f['words'] and isinstance(f['words'][0],dict):
  f['word_records']=f['words'];f['words']=[w['normalized_word'] for w in f['word_records']]
  f['word_phone_ranges']=np.asarray([(w['phone_unit_indices'][0],w['phone_unit_indices'][-1]+1) for w in f['word_records']],np.int64)
  f['alignment_words']=[w['lexicon_key'] for w in f['word_records']]
 if 'word_phone_ranges' not in f:
  ids=np.asarray(f['word_indices']);f['word_phone_ranges']=np.asarray([(int(np.flatnonzero(ids==i)[0]),int(np.flatnonzero(ids==i)[-1])+1) for i in range(len(f['words']))],np.int64)
 f.setdefault('alignment_words',list(f['words']))
 return f

@lru_cache(maxsize=32)
def get_frontend(language,library=None,data=None):
 if language in ROMANCE:
  from project.src.frontend.romance_dispatch import RomanceFrontend
  return RomanceFrontend(language,ROOT/'frontend'/language/'lexicon.json')
 if language not in LEGACY and language not in NEW:raise NotImplementedError('Original '+language+' frontend/planner has a separate explicit adapter')
 entry=LEGACY.get(language,'eu_four');module=frozen_module(language,entry)
 args={'espeak_library':library or ESPEAK_LIBRARY,'espeak_data':data or ESPEAK_DATA,'pause_policy':'word'}
 if language in NEW:
  tag=NEW[language];module.LANGUAGES[tag]=language;module._ALIASES.update({tag.lower():tag,language:tag})
  
  for ipa,manner,place,voiced in [('ʑ','fricative','palatal',True),('dʑ','affricate','palatal',True),('pf','affricate','labiodental',False),('ɧ','fricative','velar',False)]:
   module._BASES[ipa]=module.PhoneticSegment(ipa,manner,place,voiced)
  module._BASES['ʉ']=module.PhoneticSegment('ʉ','vowel',voiced=True,vowel_start=(0.,.5,1.))
  module._BASES['ɶ']=module.PhoneticSegment('ɶ','vowel',voiced=True,vowel_start=(1.,0.,1.))
  module.FRONTEND_VERSION='tts25_explicit_espeak_new7_v1';module.FEATURE_ADAPTER_VERSION='tts25_articulatory36_extra4_v1'
  return module.FourLanguageFrontend(language,**args)
 if language=='ga':args['irish_v2']=True
 lex=list((ROOT/'frontend'/language).glob('*lexicon*.json'))
 if lex:args.update(lexicon_path=str(lex[0]),lexicon_sha256=sha(lex[0]))
 klass={'eu_four':'FourLanguageFrontend','eu_baltic':'BalticFrontend','eu_finnic':'FinnicFrontend','eu_south_slavic':'SouthSlavicFrontend'}[entry]
 return getattr(module,klass)(language,**args)

def encode(language,text,metadata=None):
 metadata=metadata or {}
 if language in ('zh','en'):
  p=ROOT/'training_tools/zh_en_frontend_adapter.py'
  spec=importlib.util.spec_from_file_location('_tts25_zh_en_adapter',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  return m.encode(language,text,metadata)
 front=get_frontend(language,metadata.get('espeak_library'),metadata.get('espeak_data'));f=adapt_words(front.encode(text))
 if language in NEW:
  x=np.asarray(f['phonetic_features']);extra=np.zeros((len(x),4),np.float32)
  for i,p in enumerate(f['ipa']):
   core=unicodedata.normalize('NFD',p).translate(str.maketrans('','','ːˑʰʲˠ̥̬̪̩̯̃'))
   if core=='ɾ':extra[i,0]=1;x[i,6]=0
   if core=='r':extra[i,1]=1;x[i,6]=0
   if core in ('ɕ','tɕ','ʑ','dʑ'):extra[i,2]=1
   if 'ʷ' in p or core in ('w','ɥ'):extra[i,3]=1
  f['phonetic_features']=np.concatenate((x,extra),axis=1)
  lexical=np.asarray(f['word_indices'],np.int64);pause=np.asarray(f['ipa'])=='sil';owner=lexical.copy();left=-1
  for i,v in enumerate(lexical):
   if v>=0:left=int(v)
   elif pause[i] and i not in (0,len(lexical)-1):owner[i]=left
  f['lexical_word_indices']=lexical;f['plan_owner']=owner;f['internal_pause_mask']=pause.copy();f['internal_pause_mask'][[0,-1]]=False;f['vowel_mask']=f['phonetic_features'][:,31]>.5
  f['phonetic_feature_names']=list(f['phonetic_feature_names'])+['manner_tap','manner_trill','place_alveolopalatal','consonant_round']
  f['metadata']['feature_adapter_version']='tts25_articulatory36_extra4_v1'
 f['metadata']['adapter_path']=str(Path(__file__).resolve());f['metadata']['adapter_sha256']=sha(__file__)
 base=f.get('metadata',{}).get('semantic_identity_sha256',getattr(front,'signature',None));payload={'base':base,'adapter_sha256':sha(__file__),'frozen_frontend_source_sha256':snapshot_sha(language),'language':language,'phonetic_dim':f['phonetic_features'].shape[-1],'context_dim':f['context_features'].shape[-1]}
 f['metadata']['frontend_signature']=getattr(front,'signature',None) if language in ROMANCE else hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
 f['metadata']['ordinary_text_only']=True
 if language not in ROMANCE:f['metadata']['frozen_frontend_source_sha256']=snapshot_sha(language)
 return f
