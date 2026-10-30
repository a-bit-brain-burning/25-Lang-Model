"""Explicit contextual PT o=u/unstressed candidate; old/default sources untouched.

Numeric36/context22/plan5 topology is unchanged. Letter-name and uncertain
standalone/quoted contexts retain the parent dictionary entry and a review flag.
"""
import hashlib,json,pathlib,re
import numpy as np
from functools import lru_cache
from project.src.frontend.romance_dispatch import RomanceFrontend as _ParentFrontend
from project.src.frontend.romance import encode_ipa
from project.src.models.romance_word_planner import encode_frontend as _planner_encode
PROFILE='portuguese_contextual_o_u_unstressed_v1'
HERE=pathlib.Path(__file__).resolve().parent
PARENT_SIGNATURE='30a6faaf77db0444480467e894af0ce70f29741b39d902f06a506f7655fd6647'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def signature(lexicon_path):
 return hashlib.sha256(json.dumps({'profile':PROFILE,'parent_frontend_signature':PARENT_SIGNATURE,'adapter_sha256':sha(__file__),'lexicon_sha256':sha(lexicon_path),'rules':'context_o_only_conservative_letter_name_hold_v1','numeric_shape':[36,22,5,39,8]},sort_keys=True).encode()).hexdigest()
def decisions(text,front):
 normalized,tokens=front._tokens(text);words=[w for w,a,b in tokens];result=[]
 for wi,(w,a,b)in enumerate(tokens):
  if w!='o':continue
  previous=words[max(0,wi-3):wi];following=words[wi+1:wi+5];left=normalized[:a].rstrip();right=normalized[b:].lstrip();letter=False;why=None
  if len(tokens)==1:letter=True;why='SINGLE_O_HAS_NO_FUNCTION_WORD_CONTEXT'
  elif any(v in ['letra','letras','vogal','vogais','grafema','grafemas','caractere','caracteres','carácter','caracter','símbolo','simbolo']for v in previous):letter=True;why='EXPLICIT_LETTER_SYMBOL_CONTEXT_RETAIN_PARENT'
  elif(left.endswith(('"','«','“','‘'))and right.startswith(('"','»','”','’'))):letter=True;why='QUOTED_SINGLE_LETTER_CONTEXT_REVIEW'
  elif following[:3]in [['é','uma','vogal'],['é','a','letra'],['é','um','símbolo']]:letter=True;why='METALINGUISTIC_PREDICATE_RETAIN_PARENT'
  elif wi+1<len(words)and len(words[wi+1])==1and words[wi+1]in'aeiu'and wi and len(words[wi-1])==1:letter=True;why='LETTER_ENUMERATION_REVIEW'
  elif wi+1==len(words)and not(left.endswith('-')):letter=True;why='FINAL_STANDALONE_O_READING_UNCERTAIN'
  if not letter:why='SEPARATE_O_IN_FUNCTION_WORD_CONTEXT'if wi+1<len(words)else'HYPHENATED_FINAL_CLITIC_O'
  result.append({'word_index':wi,'word':w,'normalized_span':[a,b],'original_text_span_valid':len(text)==len(normalized),'original_form':text[a:b]if len(text)==len(normalized)else None,'action':'KEEP_PARENT_REVIEW'if letter else'CONTEXTUAL_U_UNSTRESSED','reason':why,'POS_human_gold':False})
 return result

@lru_cache(maxsize=4)
def _cached_parent_frontend(path,expected_sha):
 if sha(path)!=expected_sha:raise ValueError("Frozen contextual PT lexicon SHA changed")
 return _ParentFrontend("pt",path)

def encode(language,text,metadata=None):
 metadata=metadata or{}
 if language!='pt'or metadata.get('frontend_profile')!=PROFILE:raise ValueError('Explicit PT contextual-o profile opt-in required')
 lexicon=pathlib.Path(metadata.get('lexicon_path',HERE/'lexicon.json'))
 if metadata.get('lexicon_sha256')and sha(lexicon)!=metadata['lexicon_sha256']:raise ValueError('PT candidate lexicon SHA differs')
 actual=signature(lexicon)
 if metadata.get('frontend_signature')and metadata['frontend_signature']!=actual:raise ValueError('PT candidate frontend signature differs')
 front=_cached_parent_frontend(str(lexicon.resolve()),sha(lexicon));raw=front.encode(text);f={k:v.copy()if isinstance(v,np.ndarray)else v for k,v in raw.items()};ipa=list(raw['ipa']);review=decisions(text,front);changed=[]
 for d in review:
  if d['action']!='CONTEXTUAL_U_UNSTRESSED':continue
  wi=d['word_index'];lo,hi=map(int,f['word_phone_ranges'][wi])
  if hi-lo!=1or ipa[lo]!='o':raise ValueError('Unexpected parent separate-o topology; new version required')
  ipa[lo]='u';f['phonetic_features'][lo]=encode_ipa('u');f['stress'][lo]=0;f['stress_known'][lo]=True;f['context_features'][lo,12]=0.;f['context_features'][lo,13]=1.;changed.append(lo)
 f['ipa']=ipa;f['phones']=list(ipa);f['phonetic']=f['phonetic_features'];f['context']=f['context_features'];f['text_features']=f['context_features'][:,:12].copy();f['vowel_mask']=f['phonetic_features'][:,1]>.5;f['voiced_phone_mask']=f['phonetic_features'][:,17]>.5
 f['metadata']={**raw['metadata'],'frontend_profile':PROFILE,'frontend_signature':actual,'parent_frontend_signature':PARENT_SIGNATURE,'parent_lexicon_signature_with_candidate_container':front.signature,'adapter_path':str(pathlib.Path(__file__).resolve()),'adapter_sha256':sha(__file__),'lexicon_sha256':sha(lexicon),'contextual_o_decisions':review,'changed_unit_indices':changed,'reference_audio_or_duration_used':False,'pronunciation_gold':False,'same_word_phone_counts_pause_topology':True,'letter_name_contexts_retained_not_claimed_fixed':True};p=_planner_encode(f);return {**f,'planner_numeric':p['numeric'],'planner_boundary':p['boundary'],'planner_owner':f['plan_owner']}
