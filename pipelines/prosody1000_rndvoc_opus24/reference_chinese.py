"""Use the exact released Chinese lexicon and rules with NEW25 numeric features.

Kokoro token IDs are never passed to NEW25. The bridge renders the native
frontend's text-derived surface pinyin, word groups and citation/surface tones
into the already trained initial/whole-rime feature contract.
"""
from pathlib import Path
from functools import lru_cache
import ctypes,re,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
CONTOURS=np.asarray([[0,0,0],[1,1,1],[-.5,0,1],[-.5,-1,.5],[1,0,-1],[0,0,0]],np.float32)
class FrontendCapacityError(ValueError):pass
@lru_cache(maxsize=1)
def library():
 path=ROOT/'reference_chinese/libreference_zh.so'
 expected=json.loads((path.parent/'BUILD.json').read_text())['library_sha256']
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Reference Chinese library hash differs from its build receipt')
 lib=ctypes.CDLL(str(path))
 for name in ['qb_t2s','qb_zh_kokoro_t2s','qb_reference_syllables','qb_reference_syllables_separate']:
  f=getattr(lib,name);f.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];f.restype=ctypes.c_int
 return lib
def call(name,text):
 out=ctypes.create_string_buffer(65536)
 result=getattr(library(),name)(text.encode(),out,len(out))
 if result in (-2,-20):raise FrontendCapacityError('Reference Chinese frontend supports at most 128 Han characters per phrase')
 if result<0:raise ValueError('Reference Chinese frontend rejected input (%d): %s'%(result,text))
 return out.value.decode()
def decisions(text,separate=False):
 result=[]
 for line in call('qb_reference_syllables_separate' if separate else 'qb_reference_syllables',text).splitlines():
  start,end,surface,raw,citation,group,level,hint=line.split('\t')
  result.append(dict(start=int(start),end=int(end),surface=surface,raw=raw,citation=int(citation),group=int(group),level=int(level),explicit_hint=bool(int(hint))))
 return result

def encode(text,vocabulary):
 from project.src.frontend.chinese_training import _whole_units
 normalized=call('qb_t2s',text);rows=decisions(text)
 if not rows:raise ValueError('Chinese context requires at least one Han character')
 han=re.findall(r'[\u3400-\u9fff]',re.sub(r'\[[a-zvü]+[0-5]\]','',normalized))
 reference_rows=rows;deferred=[]
 for row in reference_rows:
  units,*_=_whole_units(row['surface']);missing=[unit[1]for unit in units if unit[1]not in vocabulary]
  if missing:
   if row['end']-row['start']!=2 or not row['surface'][:-1].endswith('r') or han[row['end']-1]!='儿':
    raise ValueError('Reference reading has an untrained phone: '+str(missing))
   deferred.append(dict(character=''.join(han[row['start']:row['end']]),reference_surface=row['surface'],reason='fused_rime_absent_from_frozen_acoustic_vocabulary',missing_phones=missing))
 # Re-run the complete text policy with fusion disabled. Splitting a fused
 # surface token after sandhi would incorrectly carry its tone onto a new
 # two-syllable sequence (e.g. 点儿 dianr2 -> dian2 er2). The native re-run
 # recomputes citation and sandhi for the actual separate topology.
 if deferred:rows=decisions(text,separate=True)
 rendered=[(row,_whole_units(row['surface'])[0])for row in rows]
 phones=['sil'];phonetic=[np.asarray([1.]+[0.]*31,np.float32)];context=[np.zeros(34,np.float32)];syllables=[]
 # Question/exclamation cue follows the actual sentence-ending punctuation.
 ends=[];position=0
 for char in re.sub(r'\[[a-zvü]+[0-5]\]','',normalized):
  if '\u3400'<=char<='\u9fff':position+=1
  elif char in '。.!?！？；;':ends.append((position,1 if char in '?？' else 2 if char in '!！' else 0))
 ends.append((len(han),0))
 for i,(row,units) in enumerate(rendered):
  start=len(phones);tone=int(row['surface'][-1]);prior=rendered[i-1][0] if i else None
  cue=next((cue for end,cue in ends if end>=row['end']),0)
  for j,unit in enumerate(units):
   if unit[1] not in vocabulary:raise ValueError('Untrained reference bridge phone: '+unit[1])
   phones.append(unit[1]);phonetic.append(unit[2]);c=np.zeros(34,np.float32)
   c[8]=row['level']/4 if j==len(units)-1 else 0;c[9]=prior['level']/4 if j==0 and prior else 0
   c[14:17]=CONTOURS[tone];c[17]=1;c[20]=j==0;c[21]=j==len(units)-1
   if row['citation']:c[22:25]=CONTOURS[row['citation']];c[25]=1
   if prior and not prior['level']:c[26:29]=CONTOURS[int(prior['surface'][-1])];c[29]=1
   c[30]=tone==5;c[31]=bool(row['citation'] and row['citation']!=tone);c[32]=cue==1;c[33]=cue==2
   context.append(c)
  syllables.append(dict(character=''.join(han[row['start']:row['end']]),pinyin=row['surface'],citation_pinyin=row['raw'][:-1]+str(row['citation']) if row['citation'] else None,phone_start=start,phone_end=len(phones),group_index=row['group'],boundary_after_level=row['level'],reference_character_start=row['start'],reference_character_end=row['end']))
  phones.append('sil');phonetic.append(np.asarray([1.]+[0.]*31,np.float32));c=np.zeros(34,np.float32)
  if i<len(rendered)-1:c[8:10]=row['level']/4
  context.append(c)
 n=len(phones);ctx=np.asarray(context,np.float32);ctx[1:-1,5]=np.arange(n-2,dtype=np.float32)/max(n-3,1);ctx[:,6]=min((n-2)/200.,2.);ctx[1:-1,7]=min((1+max(r['group'] for r in rows))/40.,2.);ctx[0,10]=1;ctx[-1,11]=1
 for group in sorted(set(r['group'] for r in rows)):
  members=[s for s in syllables if s['group_index']==group];first,last=members[0]['phone_start'],members[-1]['phone_end'];size=last-first
  for i in range(first,last):ctx[i,:5]=[float(i==first),float(i==last-1),(i-first)/max(size-1,1),min(size/20.,2.),group/max(max(r['group'] for r in rows),1)]
 arrays=dict(phones=np.asarray([vocabulary[p] for p in phones],np.int64),phonetic=np.asarray(phonetic,np.float32),context=ctx,pause=np.asarray([p=='sil' for p in phones],np.float32))
 meta=dict(ipa=phones,syllables=syllables,reference_normalized_text=normalized,reference_surface_pinyin=[r['surface'] for r in reference_rows],reference_syllables=reference_rows,acoustic_surface_pinyin=[r['surface'] for r,_ in rendered],deferred_fused_erhua=deferred,reference_pronunciation_source='best_bilingual_release_20261029_native_zh_t2s',unknown_phone_count=0,ordinary_text_only=True,reference_audio_or_durations_used=False)
 return arrays,meta
