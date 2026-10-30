"""Sentence-aware chunks and letter-name tokens for the Chinese model.

The 26 entries are phonemes, not recordings or calls to the English network.
Chinese text plus initialisms goes through the Chinese acoustic path once.
"""
from pathlib import Path
from functools import lru_cache
import re
import sys

from text_prepare import normalize_zh,normalize_en,split_long,is_han,segments as old_segments
from compact_frontend import compact_phonemes

LETTER_PHONES=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',[
    'ˈA','bˈi','sˈi','dˈi','ˈi','ˈɛf','ʤˈi','ˈAʧ','ˈI','ʤˈA','kˈA','ˈɛl','ˈɛm',
    'ˈɛn','ˈO','pˈi','kjˈu','ˈɑɹ','ˈɛs','tˈi','jˈu','vˈi','dˈʌbəl jˈu','ˈɛks','wˈI','zˈi']))
LEXICAL_UPPER={'YES','NO','OK','HI','OH','HEY','STOP','WATCH','PRO'}
COMMON_INITIALISMS={'ai','api','ar','cpu','dna','etf','gpu','gps','hdmi','pdf','ppt','rna','usb','vr'}
MAX_PHONES=440
PUNCTUATION={**dict.fromkeys('、，,',', '),**dict.fromkeys('。．.','. '),
             **dict.fromkeys('！!','! '),**dict.fromkeys('？?','? '),
             **dict.fromkeys('；;','; '),**dict.fromkeys('：:',': '),
             **dict.fromkeys('（(',' ('),**dict.fromkeys('）)',') '),
             **dict.fromkeys('《「【“','“'),**dict.fromkeys('》」】”','”'),'…':'…','—':'—'}

def chinese_fragment(text):
    if any(is_han(c) for c in text):
        return ''.join(compact_phonemes('zh',text.replace('嗯','嗯[en2]'),'compact_normalized'))
    assert all(c.isspace() or c in PUNCTUATION for c in text),text
    return ''.join(PUNCTUATION.get(c,'') for c in text)

def letter_spans(text):
    result=[]
    for match in re.finditer(r'[A-Za-z]+',text):
        word=match[0]
        next_number=bool(re.match(r'\s*[零一二三四五六七八九十]',text[match.end():]))
        previous_number=match.start()>0 and text[match.start()-1] in '零一二三四五六七八九十'
        single=len(word)==1 and (word.isupper() or next_number or previous_number)
        initialism=word.isupper() and 2<=len(word)<=5 and word not in LEXICAL_UPPER
        code=1<=len(word)<=3 and next_number and word.lower() not in ('pro','max','air')
        # A and I inside a real English phrase remain English words.
        if single and word in ('A','I') and re.match(r'\s+[a-z]',text[match.end():]):single=False
        if single or initialism or code or word.lower() in COMMON_INITIALISMS:result.append((match.start(),match.end(),word))
    return result

def language_runs(text):
    covered=set(i for start,end,_ in letter_spans(text) for i in range(start,end))
    runs=[];start=0;language='zh'
    for i,char in enumerate(text):
        kind='zh' if is_han(char) or i in covered else 'en' if char.isascii() and char.isalpha() else None
        if kind is not None and kind!=language:
            if i>start:runs.append((language,text[start:i]))
            start=i;language=kind
    if start<len(text):runs.append((language,text[start:]))
    assert ''.join(t for _,t in runs)==text
    return [(lang,t) for lang,t in runs if any(is_han(c) or c.isalpha() for c in t)]

@lru_cache(maxsize=4096)
def zh_phonemes(text):
    matches=letter_spans(text)
    assert [(m.start(),m.end()) for m in re.finditer(r'[A-Za-z]+',text)]==[(a,b) for a,b,_ in matches],text
    position=0;parts=[];letters=[]
    for start,end,word in matches:
        fragment=text[position:start]
        if fragment.strip():parts.append(chinese_fragment(fragment))
        phones=' '.join(LETTER_PHONES[c.upper()] for c in word)
        parts.append(' '+phones+' ')
        letters.append(dict(text=word,phonemes=phones,start=start,end=end,acoustic_language='zh'))
        position=end
    fragment=text[position:]
    if fragment.strip():parts.append(chinese_fragment(fragment))
    phones=''.join(parts).strip()
    assert phones
    return phones,letters

def valid_zh(text):
    # The preserved native lexicon has a 128-Han workspace; never overflow it.
    if sum(is_han(c) for c in text)>128:return False
    phones,_=zh_phonemes(text)
    return 0<len(phones)<=MAX_PHONES

def split_chinese(text):
    chunks=[];remaining=text
    while remaining:
        if valid_zh(remaining):
            chunks.append((remaining,'end_of_run'));break
        strong=[m.end() for m in re.finditer(r'[。！？.!?；;]+[”」』）)]*\s*',remaining)]
        weak=[m.end() for m in re.finditer(r'[，,、：:]\s*',remaining)]
        chosen=None
        for reason,candidates in [('sentence_boundary',strong),('clause_boundary_at_context_limit',weak)]:
            for at in reversed(candidates):
                if 0<at<len(remaining) and valid_zh(remaining[:at]):
                    chosen=(at,reason);break
            if chosen:break
        if chosen is None:
            # Last resort for unpunctuated input, keeping complete letter codes.
            forbidden=set(i for a,b,_ in letter_spans(remaining) for i in range(a+1,b))
            for at in range(min(len(remaining)-1,128),0,-1):
                if at not in forbidden and valid_zh(remaining[:at]):
                    chosen=(at,'hard_context_limit_no_punctuation');break
        assert chosen,remaining
        at,reason=chosen;chunks.append((remaining[:at],reason));remaining=remaining[at:]
    assert ''.join(t for t,_ in chunks)==text
    return chunks

def prepare(text,language):
    if language=='en':
        normalized,chunks=old_segments(text,'en')
        for chunk in chunks:
            chunk.update(phonemes=''.join(compact_phonemes(chunk['language'],chunk['text'],'compact_normalized')),
                         letter_spans=[],split_reason='original_English_path')
        return normalized,chunks
    normalized=normalize_zh(text)
    result=[]
    for lang,run in language_runs(normalized):
        if lang=='zh':
            for chunk,reason in split_chinese(run):
                phones,letters=zh_phonemes(chunk)
                result.append(dict(language='zh',text=chunk,phonemes=phones,letter_spans=letters,split_reason=reason))
        else:
            for chunk in split_long(run,'en'):
                if not any(c.isalpha() for c in chunk):continue
                # Real English words/phrases keep the existing English voice;
                # initialisms have already remained inside the Chinese run.
                chunk=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',chunk)
                result.append(dict(language='en',text=chunk,phonemes=''.join(compact_phonemes('en',chunk,'compact_normalized')),
                                   letter_spans=[],split_reason='real_English_words'))
    assert result
    assert all(0<len(c['phonemes'])<=MAX_PHONES for c in result)
    return normalized,result
