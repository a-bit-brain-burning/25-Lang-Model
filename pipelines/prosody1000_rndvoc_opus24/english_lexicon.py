"""Dictionary-backed English compounds/inflections over the shipped CMU data.

The accepted release's productive ed/ing/es/s and known-root compound policy is
retained. Dictionary bytes stay unchanged; unresolved words remain explicit OOV.
"""
from functools import lru_cache
import re

class LexiconPolicy:
 def __init__(self,dictionary):
  self.dictionary=dictionary;self.original_lookup=dictionary.lookup_word;self.original_stress=dictionary.stress_for
  self.sources={};dictionary.lookup_word=self.lookup_word;dictionary.stress_for=self.stress_for
 def known(self,word):
  raw=self.dictionary.entries.get(word)
  stressed=self.dictionary.stress_entries.get(word)
  if raw:
   if stressed and [p.rstrip('012')for p in raw.split()]==[p.rstrip('012')for p in stressed.split()]:return tuple(stressed.split())
   return tuple(raw.split())
  if stressed:return tuple(stressed.split())
  return None
 @lru_cache(maxsize=8192)
 def derive(self,word,depth=0):
  direct=self.known(word)
  if direct:return direct,'frozen_CMU',[]
  if depth>=3:return None
  # Letter-name words emitted by the public abbreviation policy reuse the
  # corresponding real CMU letter entry, rather than spelling-rule fallback.
  names={'ef':'f','aitch':'h','zee':'z','cue':'q','en':'n','ess':'s','dee':'d','gee':'g','vee':'v'}
  if word in names:
   pron=self.known(names[word])
   if pron:return pron,'CMU_letter_name',[names[word]]
  for ending in ('ed','ing','es','s'):
   if not word.endswith(ending)or len(word)<=len(ending)+2:continue
   stem=word[:-len(ending)];options=[stem]
   if ending in ('ed','ing'):
    options.append(stem+'e')
    if len(stem)>2 and stem[-1]==stem[-2] and stem[-1]not in 'aeiou':options.append(stem[:-1])
   if ending in ('es','ed')and stem.endswith('i'):options.append(stem[:-1]+'y')
   for root in options:
    found=self.derive(root,depth+1)
    if not found:continue
    phones=found[0];last=phones[-1].rstrip('012');voiceless=last in ('P','K','T','S','F','SH','CH','TH')
    if ending=='ed':suffix=('IH0','D')if last in ('T','D')else ('T'if voiceless else'D',)
    elif ending=='ing':suffix=('IH0','NG')
    elif last in ('S','Z','SH','ZH','CH','JH'):suffix=('IH0','Z')
    else:suffix=('S'if voiceless else'Z',)
    return phones+suffix,'productive_inflection',[root,ending]
  # British spelling only falls back when the proposed American form is an
  # actual known entry; no guessed pronunciation is promoted to a dictionary hit.
  alternatives=[word.replace('our','or')]
  if word.endswith('re'):alternatives.append(word[:-2]+'er')
  for alt in alternatives:
   if alt!=word and self.known(alt):return self.known(alt),'known_spelling_variant',[alt]
  if word.endswith('able'):
   for stem in [word[:-4],word[:-4]+'e']:
    base=self.known(stem)
    if base:return base+('AH0','B','AH0','L'),'productive_able',[stem,'able']
  for at in range(len(word)-3,2,-1):
   left,right=self.known(word[:at]),self.known(word[at:])
   if left and right:
    right=tuple(p[:-1]+'2'if p.endswith('1')else p for p in right)
    return left+right,'known_root_compound',[word[:at],word[at:]]
  return None
 def lookup_word(self,word):
  original=self.original_lookup(word)
  if original:return original
  found=self.derive(word.lower())
  if found:
   pron,rule,parts=found;self.sources[word.lower()]=dict(word=word.lower(),rule=rule,parts=parts,ARPABET=list(pron),approximate_spelling_fallback=False)
   return ' '.join(pron)
  return None
 def stress_for(self,word,phones):
  if self.original_lookup(word):return self.original_stress(word,phones)
  found=self.derive(word.lower())
  if found and [p.rstrip('012')for p in found[0]]==phones:
   return [int(p[-1])if p[-1].isdigit()else-1 for p in found[0]],found[1]
  return self.original_stress(word,phones)
 def camel_case(self,text,overrides,initialisms):
  events=[]
  def change(m):
   word=m[0]
   if word in overrides or self.known(word.lower())or not re.search('[a-z][A-Z]',word):return word
   parts=re.findall(r'[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+',word)
   if ''.join(parts)!=word or len(parts)<2:return word
   if not all(self.known(p.lower())or p.upper()in initialisms or(len(p)==1 and p.isupper())for p in parts):return word
   events.append(dict(source=word,spoken=' '.join(parts),rule='known_parts_CamelCase'));return ' '.join(parts)
  return re.sub(r'[A-Za-z]+',change,text),events

def install(front):
 front._session('en')
 from project.src.frontend.english import default_dictionary
 dictionary=default_dictionary()
 if not hasattr(dictionary,'clarity_policy'):dictionary.clarity_policy=LexiconPolicy(dictionary)
 return dictionary.clarity_policy
