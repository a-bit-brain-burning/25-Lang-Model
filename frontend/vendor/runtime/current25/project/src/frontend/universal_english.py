"""Adapter from the frozen English frontend into the numeric shared contract.

All English symbols and lexical policy stay in this adapter. The acoustic
model receives articulatory and generic prosodic coordinates only.
"""
import numpy as np
from .english import EnglishFrontend, PHONE_TO_ID
from .english_policy import prepare_english_features
from .universal import LanguageFrontend, UniversalPhone, encode_phones

ARPABET_TO_IPA = {'AA':'ɑ','AE':'æ','AH':'ʌ','AO':'ɔ','AW':'aʊ','AY':'aɪ','B':'b','CH':'tʃ',
 'D':'d','DH':'ð','EH':'ɛ','ER':'ɝ','EY':'eɪ','F':'f','G':'ɡ','HH':'h','IH':'ɪ','IY':'i',
 'JH':'dʒ','K':'k','L':'l','M':'m','N':'n','NG':'ŋ','OW':'oʊ','OY':'ɔɪ','P':'p','R':'ɹ',
 'S':'s','SH':'ʃ','T':'t','TH':'θ','UH':'ʊ','UW':'u','V':'v','W':'w','Y':'j','Z':'z','ZH':'ʒ',
 '<SIL>':'sil','<BOS>':'sil','<EOS>':'sil','<UNK>':'sil'}
_ID_TO_PHONE = {v:k for k,v in PHONE_TO_ID.items()}

def _stress_level(value):
    # CMU primary=1, secondary=2, unstressed=0; generic strength is ordered.
    if value < 0: return None
    if value not in (0,1,2): raise ValueError('Invalid English dictionary stress')
    return {0:0.,1:1.,2:.5}[int(value)]

def from_legacy_arrays(phone_ids,text_features,stress=None):
    ids = np.asarray(phone_ids,dtype=np.int64)
    positions = np.asarray(text_features,dtype=np.float32)
    if ids.ndim != 1 or positions.shape != (len(ids),12):
        raise ValueError('Legacy arrays need phone_ids[N], text_features[N,12]')
    values = np.full(len(ids),-1,np.int64) if stress is None else np.asarray(stress,np.int64)
    if values.shape != ids.shape: raise ValueError('Stress shape differs from phones')
    names = [_ID_TO_PHONE[int(i)] for i in ids]
    phones = [UniversalPhone(ARPABET_TO_IPA[name],_stress_level(value)) for name,value in zip(names,values)]
    return encode_phones(phones,positions,metadata={'language':'en','legacy_adapter':True},valid_mask=ids!=0)

def from_legacy_features(features,policy=False):
    source = prepare_english_features(features) if policy else features
    result = from_legacy_arrays(source['phone_ids'],source['text_features'],source.get('stress'))
    result['metadata'].update({k:source[k] for k in ('normalized_text','words','oov_words','frontend_version') if k in source})
    result['metadata']['english_policy_applied'] = bool(policy)
    return result

class UniversalEnglishFrontend(LanguageFrontend):
    language='en'
    def __init__(self,policy=False,**kwargs):
        self.legacy = EnglishFrontend(**kwargs)
        self.policy = policy
    def encode(self,text):
        return from_legacy_features(self.legacy.encode(text),self.policy)
