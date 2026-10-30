"""Existing BG phones with an expanded, source-bound lexical-stress resource."""
import hashlib,importlib,sys
import numpy as np
from common import HERE,sha
class Frontend:
 def __init__(self,run):
  self.original=run._system('bg-BG');self.original[3].encode('bg','Добро утро.',self.original[2])
  m=importlib.import_module('_tts25_frozen_bg.frontend.eu_south_slavic')
  from espeakng_loader import get_library_path,get_data_path
  path=HERE/'repairs/bg_dictionary/expanded_lexicon.json'
  self.front=m.SouthSlavicFrontend('bg',espeak_library=get_library_path(),espeak_data=get_data_path(),pause_policy='word',lexicon_path=path,lexicon_sha256=sha(path))
  self.signature=hashlib.sha256((self.original[2]['frontend_signature']+sha(path)+sha(__file__)).encode()).hexdigest()
 def encode(self,language,text,metadata=None):
  assert language=='bg';f=self.front.encode(text);records=f['words'];f['word_records']=records;f['words']=[r['normalized_word']for r in records]
  f['word_phone_ranges']=np.asarray([(r['phone_unit_indices'][0],r['phone_unit_indices'][-1]+1)for r in records],np.int64)
  f['alignment_words']=[r['lexicon_key']for r in records]
  f['metadata'].update(frontend_signature=self.signature,ordinary_text_only=True,expanded_lexical_stress_not_manual_gold=True)
  return f
