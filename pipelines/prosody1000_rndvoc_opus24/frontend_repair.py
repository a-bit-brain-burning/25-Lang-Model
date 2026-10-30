"""Offline auditable workbook normalization and whole-utterance feature building."""
from pathlib import Path
import os,sys,re,json,hashlib,unicodedata,time,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frontend_bundle'))
from frontend import TextFrontend
DIGITS='零一二三四五六七八九'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,j):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def cardinal(s):
 if '.' in s:
  a,b=s.split('.',1);return cardinal(a)+'点'+''.join(DIGITS[int(c)] for c in b)
 n=int(s)
 if not n:return '零'
 def small(n):
  out='';zero=False
  for value,unit in [(1000,'千'),(100,'百'),(10,'十'),(1,'')]:
   d,n=divmod(n,value)
   if d:
    if zero:out+='零';zero=False
    out+=('' if value==10 and d==1 and not out else DIGITS[d])+unit
   elif out and n:zero=True
  return out
 out=''
 for value,unit in [(100000000,'亿'),(10000,'万'),(1,'')]:
  d,n=divmod(n,value)
  if d:
   if out and d<1000:out+='零'
   out+=small(d)+unit
 return out
def previous_normalize(text,language):
 text=text.replace('º','°');text=unicodedata.normalize('NFKC',text)
 text=''.join(c for c in text if unicodedata.category(c)!='Cf')
 if language=='en':
  text=re.sub(r'([0-9.]+)\s*°\s*C\b',r'\1 degrees Celsius',text,flags=re.I)
  text=re.sub(r'([0-9.]+)\s*°\s*F\b',r'\1 degrees Fahrenheit',text,flags=re.I)
  for unit,word in [('km','kilometers'),('min','minutes'),('kV','kilovolts')]:text=re.sub(r'(\d[\d,.]*)\s*'+unit+r'\b',r'\1 '+word,text)
  text=re.sub(r'(\d[\d,.]*)\s*V\b',r'\1 volts',text)
  for symbol,word in [('£','pounds'),('$','dollars'),('€','euros')]:text=re.sub(re.escape(symbol)+r'\s*([\d,.]+)',lambda m:m[1]+' '+word,text)
  text=text.replace('>',' then ').replace('=',' equals ')
  return text,[]
 protected=[]
 def protect(kind,value):
  key=chr(0xE000+len(protected));protected.append(dict(marker=key,kind=kind,value=value));return key
 # Identifiers and phone groups are retained digit by digit, with leading zero.
 text=re.sub(r'[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+',lambda m:protect('identifier',m[0]),text)
 text=re.sub(r'(?<!\d)\d{2,4}-\d{5,9}(?!\d)',lambda m:protect('phone',m[0]),text)
 text=re.sub(r'((?:电话(?:是)?|热线|拨打))([0-9]{3,})',lambda m:m[1]+protect('digits',m[2]),text)
 text=re.sub(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)',lambda m:cardinal(m[1])+'点'+('整' if int(m[2])==0 else ('零' if int(m[2])<10 else '')+cardinal(m[2])+'分'),text)
 text=re.sub(r'(?<=[0-9%])\s*-\s*(?=\d)','到',text)
 text=re.sub(r'(\d+(?:\.\d+)?)%',lambda m:'百分之'+cardinal(m[1]),text)
 text=text.replace('%','百分比').replace('/','每')
 text=re.sub(r'(\d{4})(?=年)',lambda m:''.join(DIGITS[int(c)] for c in m[1]),text)
 text=re.sub(r'\d+(?:\.\d+)?',lambda m:cardinal(m[0]),text)
 text=text.replace('+','多').replace('~','。').replace('"','，').replace('(', '，').replace(')', '，')
 text=text.replace('嗯','嗯[en5]')
 return text,protected

from legacy_contract import normalize_chinese
def normalize(text,language):
 if language=='zh':return normalize_chinese(text)
 text=re.sub(r'[A-Za-z]+[0-9]+[A-Za-z0-9]*',lambda m:' '.join((dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ','ay bee see dee ee ef gee aitch eye jay kay el em en oh pee cue ar ess tee you vee double_you ex why zee'.split()))[c.upper()].replace('_',' ') if c.isalpha() else 'zero one two three four five six seven eight nine'.split()[int(c)]) for c in m[0]),text)
 return previous_normalize(text,language)

def build_zh(front,text,normalizer=None):
 normalized,protected=(normalizer or normalize)(text,'zh');protected={r['marker']:r for r in protected}
 # All raw English spans stay in the one Chinese acoustic sequence.
 tokens=[];skeleton=[];han=0;position=0;code_id=0
 pattern=r'[\u3400-\u9fff]\[[a-z]+[0-5]\]|[\ue000-\uf8ff]|[A-Za-z]+(?:[\s\x27,.!?-]+[A-Za-z]+)*'
 for match in re.finditer(pattern,normalized):
  prefix=normalized[position:match.start()];skeleton.append(prefix);han+=len(re.findall(r'[\u3400-\u9fff]',re.sub(r'\[[a-z]+\d\]','',prefix)))
  value=match[0]
  if value.endswith(']') and '[' in value:
   skeleton.append(value);han+=1;position=match.end();continue
  if value in protected:
   event=protected[value]
   if event['kind']=='word':
    tokens.append(dict(kind='english',han=han,value=event['value']));position=match.end();continue
   code=event['value'];symbol_index=0
   for c in code:
    if c=='-':skeleton.append('，');continue
    if c.isdigit():skeleton.append(DIGITS[int(c)]);tokens.append(dict(kind='digit',han=han,code=code,symbol=c,symbol_index=symbol_index,code_id=code_id));han+=1
    else:tokens.append(dict(kind='letter',han=han,value=c.upper(),code=code,symbol=c.upper(),symbol_index=symbol_index,code_id=code_id))
    symbol_index+=1
   code_id+=1
  elif value.strip() in ['AI','ETF'] or (len(value.strip())==1 and value.strip().isupper()):
   code=value.strip()
   for i,c in enumerate(code):tokens.append(dict(kind='letter',han=han,value=c,code=code,symbol=c,symbol_index=i,code_id=code_id))
   code_id+=1
  else:tokens.append(dict(kind='english',han=han,value=value.strip()))
  position=match.end()
 suffix=normalized[position:];skeleton.append(suffix);skeleton=''.join(skeleton)
 # Explicitly select separate 儿 readings; no approximate phone substitution.
 settings,adapter=front._session('zh')
 from project.src.frontend.chinese_delivery_policy import ChineseDeliveryFrontend
 from project.src.frontend.chinese_prosody import ChineseProsodyFrontend
 selected=ChineseDeliveryFrontend(ChineseProsodyFrontend(erhua_policy='separate'))
 from project.src.frontend import phrase_plan
 original=phrase_plan.make_frontend;phrase_plan.make_frontend=lambda:selected
 anchor=not bool(re.search(r'[\u3400-\u9fff]',skeleton))
 try:
  chinese,meta=front.encode(('啊' if anchor else '')+skeleton,'zh');raw=adapter.encode('zh',('啊' if anchor else '')+skeleton,settings)
 finally:phrase_plan.make_frontend=original
 if anchor:
  chinese={k:v[[0,-1]].copy() for k,v in chinese.items()};meta['ipa']=['sil','sil'];raw['metadata']['syllables']=[]
 at={};syllables={};offset=0
 for s in raw['metadata']['syllables']:
  at[offset]=int(s['phone_start']);syllables[offset]=s;offset+=len(re.findall(r'[\u3400-\u9fff]',s['character']))
 at[offset]=len(chinese['phones'])-1
 vocabulary=read(ROOT/'models/acoustic/assets/vocabulary.json');letters=read(ROOT/'models/acoustic/assets/letters.json')
 grouped={};digit_tokens=[]
 for t in tokens:
  if t['kind']=='digit':digit_tokens.append(t);continue
  index=at[t['han']]
  if t['kind']=='letter' and t['value']!='U':
   item=letters[t['value']];ipa=item['ipa'];block=dict(phones=np.array([vocabulary[p] for p in ipa],np.int64),phonetic=np.asarray(item['phonetic'],np.float32),pause=np.zeros(len(ipa),np.float32))
   block['context']=np.repeat(chinese['context'][index:index+1],len(ipa),axis=0)
  else:
   word='衣乌' if t['kind']=='letter' else t['value'];lang='zh' if t['kind']=='letter' else 'en'
   arrays,m=front.encode(word,lang);keep=[i for i,p in enumerate(m['ipa']) if p!='sil'];ipa=[m['ipa'][i] for i in keep];assert m['unknown_phone_count']==0
   block={k:v[keep].copy() for k,v in arrays.items()};ctx=np.repeat(chinese['context'][index:index+1],len(ipa),axis=0)
   if lang=='en':ctx[:,:22]=block['context'];ctx[:,10:12]=0
   else:ctx[:,12:]=block['context'][:,12:]
   block['context']=ctx
  grouped.setdefault(index,[]).append((t,block,ipa))
 pieces={k:[] for k in chinese};phones=[];mapping={};units=[]
 def add(block,ipa):
  for k in pieces:pieces[k].extend(block[k])
  phones.extend(ipa)
 def silence(index):
  block={k:chinese[k][-1:].copy() for k in chinese};block['context'][0]=chinese['context'][index];add(block,['sil'])
 for i in range(len(chinese['phones'])):
  for t,block,ipa in grouped.get(i,[]):
   if phones and phones[-1]!='sil':silence(i)
   start=len(phones);add(block,ipa)
   if t['kind']=='letter':units.append({k:v for k,v in t.items() if k in ['code','symbol','symbol_index','code_id']}|dict(start=start,end=len(phones)))
   silence(i)
  mapping[i]=len(phones);add({k:v[i:i+1] for k,v in chinese.items()},[meta['ipa'][i]])
 for t in digit_tokens:
  s=syllables[t['han']];units.append({k:v for k,v in t.items() if k in ['code','symbol','symbol_index','code_id']}|dict(start=mapping[int(s['phone_start'])],end=mapping[int(s['phone_end'])-1]+1))
 arrays={k:np.asarray(v,dtype=chinese[k].dtype) for k,v in pieces.items()};units.sort(key=lambda u:u['start'])
 n=len(phones);ctx=arrays['context'];ctx[:,10:12]=0;ctx[0,10]=1;ctx[-1,11]=1;ctx[:,5]=0;ctx[1:-1,5]=np.arange(n-2,dtype=np.float32)/max(n-3,1);ctx[:,6]=min((n-2)/200.,2.)
 assert not np.any(arrays['phones']==1),sorted(set(p for p in phones if p not in vocabulary))
 return arrays,dict(normalization_source='best_bilingual_20261029_flow_letters_v2_port',input_text=text,normalized_text=normalized,Chinese_skeleton=skeleton,protected=list(protected.values()),ipa=phones,identifier_units=units,U_reading='yiu1',U_tone=1,one_acoustic_language='zh',one_whole_utterance=True,English_acoustic_calls=0,erhua_policy='separate',unknown_phone_count=0,context_anchor_used=anchor,context_anchor_spoken=False)
