"""Public reading policy over the frozen language frontend and acoustic inputs."""
import re
from functools import lru_cache
import frontend_repair as base
from reading_policy import ReadingPolicy,LexicalLookup,spell_english

@lru_cache(maxsize=1)
def default_policy():return ReadingPolicy()

def details(text,language,front=None,policy=None):
    policy=policy or default_policy()
    text,events=policy.scan(text,language,LexicalLookup(front) if front is not None else None)
    if language=='en':
        pieces=[];cursor=0
        for event in events:
            pieces.append(text[cursor:event['start']]);spoken=spell_english(event['value']) if event['mode']=='spell' else event['value']
            pieces.append(' '+spoken+' ');event['spoken']=spoken;cursor=event['end']
        pieces.append(text[cursor:]);spoken=''.join(pieces)
        for symbol,word in [('/','slash'),('+','plus'),('#','hash'),('@','at'),('_','underscore')]:spoken=spoken.replace(symbol,' '+word+' ')
        spoken=' '.join(base.normalize(spoken,'en')[0].split())
        return spoken,[],events,policy.identity()
    if language!='zh':return text,[],[],policy.identity()
    saved=[]
    def protect(value,kind='digits',event=None):
        if len(saved)>=0x900:raise ValueError('Too many protected tokens in one text input')
        marker=chr(0xf000+len(saved));saved.append(dict(marker=marker,kind=kind,value=value,decision=event));return marker
    pieces=[];cursor=0
    for event in events:
        pieces.append(text[cursor:event['start']])
        if event['mode']=='expand':pieces.append(' '+event['value']+' ')
        else:pieces.append(protect(event['value'],'word' if event['mode']=='word' else 'identifier',event))
        cursor=event['end']
    pieces.append(text[cursor:]);marked=''.join(pieces)
    marked=re.sub(r'(?<![A-Za-z0-9])0\d+(?![A-Za-z0-9]|[ -]\d)',lambda m:protect(m[0]),marked)
    if re.fullmatch(r'[0-9\s，,、]+',marked):marked=re.sub(r'\d+',lambda m:protect(m[0]),marked)
    normalized,items=base.normalize(marked,'zh')
    for item in saved:
        replacement=chr(0xe000+len(items));items.append({**item,'marker':replacement});normalized=normalized.replace(item['marker'],replacement)
    return normalized,items,events,policy.identity()

def normalize(text,language,front=None,policy=None):
    normalized,items,events,identity=details(text,language,front,policy)
    return normalized,(items if language=='zh' else events)

def build_zh(front,text,policy=None):
    normalized,protected,events,identity=details(text,'zh',front,policy)
    arrays,meta=base.build_zh(front,text,normalizer=lambda value,lang:(normalized,protected))
    preview=normalized
    for item in protected:
        value=item['value']
        if item['kind']!='word':value=' '.join(base.DIGITS[int(c)] if c.isdigit() else c for c in value)
        preview=preview.replace(item['marker'],value)
    meta.update(reading_decisions=events,frontend_policy=identity,spoken_text_preview=preview)
    return arrays,meta
