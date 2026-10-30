"""Auditable lexical/contextual Mandarin reading resolution.

Never mutates pypinyin's process-global dictionaries. All returned readings are
citation categories; the established neutral-tone/sandhi pass follows once.
Explicit corpus pinyin bypasses this entire policy. Optional inline hints use
``重[chong2]庆`` and remain an input instruction, not a new word exception.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
import json
import re

ASSET=Path(__file__).with_name('assets')/'chinese_polyphone_phrases_v1.json'
TRAIN_ASSET=Path(__file__).with_name('assets')/'chinese_polyphone_train_corrections_v1.json'
SURNAME_ASSET=Path(__file__).with_name('assets')/'chinese_polyphone_surnames_v1.json'
_MODE=ContextVar('chinese_polyphone_mode',default='dictionary')
_CHECKPOINT=ContextVar('chinese_polyphone_checkpoint',default=None)
_THRESHOLD=ContextVar('chinese_polyphone_confidence_threshold',default=.9)
_HANS=re.compile(r'[\u3400-\u9fff]+')
_HINT=re.compile(r'([\u3400-\u9fff])\[([a-zA-ZüÜ:]+[0-5])\]')


@contextmanager
def pronunciation_mode(mode,checkpoint=None,threshold=.9):
    if mode not in ('legacy','dictionary','neural','neural_fallback'):raise ValueError('Unknown pronunciation mode')
    if not 0<=threshold<=1:raise ValueError('Invalid confidence threshold')
    token=_MODE.set(mode);path_token=_CHECKPOINT.set(checkpoint);threshold_token=_THRESHOLD.set(threshold)
    try:yield
    finally:
        _MODE.reset(token);_CHECKPOINT.reset(path_token);_THRESHOLD.reset(threshold_token)


def current_mode():return _MODE.get()


def strip_reading_hints(text):
    """Return clean text and citation hints indexed by original Han position."""
    if not isinstance(text,str):return text,{}
    out=[];hints={};cursor=han=0
    for match in _HINT.finditer(text):
        prefix=text[cursor:match.start()];out.append(prefix);han+=sum('\u3400'<=c<='\u9fff' for c in prefix)
        token=match[2].lower().replace('u:','v').replace('ü','v')
        token=token[:-1]+'5' if token.endswith('0') else token
        hints[han]=token;out.append(match[1]);han+=1;cursor=match.end()
    out.append(text[cursor:])
    return ''.join(out),hints


@lru_cache(maxsize=1)
def phrase_entries():
    if not ASSET.is_file():return {}
    resource=json.loads(ASSET.read_text())
    return resource['entries']


@lru_cache(maxsize=1)
def train_corrections():
    return json.loads(TRAIN_ASSET.read_text())['entries'] if TRAIN_ASSET.is_file() else {}


@lru_cache(maxsize=1)
def surname_entries():
    return json.loads(SURNAME_ASSET.read_text())['entries'] if SURNAME_ASSET.is_file() else {}


def resolve_surnames(run,tokens):
    """Conventional surnames only after explicit 姓 and a complete boundary.

    Never switches the whole sentence into name mode. A bare proper name has
    no automatic guarantee; a user's inline reading still takes precedence.
    """
    import jieba.posseg as posseg
    entries=surname_entries();result=list(tokens);events=[];edges=set();de_positions=set();cursor=0
    for token in posseg.cut(run,HMM=False):
        edges.add(cursor)
        if token.word=='的' and token.flag in ('uj','u','uz'):de_positions.add(cursor)
        cursor+=len(token.word)
    for match in re.finditer('姓',run):
        if match.start() not in edges:continue
        start=match.end();matches=[word for word in entries if run.startswith(word,start)]
        if not matches:continue
        word=max(matches,key=len);end=start+len(word);options=entries[word]
        if (end!=len(run) and end not in de_positions) or len(options)!=1:continue
        readings=options[0]
        if len(readings)!=len(word):raise ValueError('Surname resource changed coverage')
        result[start:end]=readings
        events.append({'start':start,'word':word,'pinyin':readings,'source':'explicit_surname_context_public_lexicon',
            'conventional_not_unique_personal_name_gold':True})
    return result,events


@lru_cache(maxsize=1)
def phrase_index():
    return {'maximum':max(map(len,phrase_entries()),default=0)}


def lexical_segments(run):
    """Protect original multi-character spans before filling lexical gaps.

    A new substring must never steal a character from an existing known word
    (e.g. adjective + 地区). No global pypinyin dictionary is mutated.
    """
    from pypinyin.core import Pinyin
    from .chinese_training import _tokenizer
    added=phrase_entries();pieces=list(Pinyin().seg(run));starts={};protected=set();cursor=0
    lexical_edges={0};word_end=0
    for word in _tokenizer().lcut(run,HMM=False):word_end+=len(word);lexical_edges.add(word_end)
    for word in pieces:
        starts[cursor]=word
        if len(word)>1:protected.update(range(cursor,cursor+len(word)))
        cursor+=len(word)
    if cursor!=len(run):raise ValueError('Pinyin segmentation changed character coverage')
    maximum=phrase_index()['maximum'];cursor=0
    while cursor<len(run):
        word=starts.get(cursor,run[cursor])
        if cursor not in protected:
            for width in range(min(maximum,len(run)-cursor),1,-1):
                candidate=run[cursor:cursor+width]
                if (candidate in added and cursor in lexical_edges and cursor+width in lexical_edges
                        and not any(i in protected for i in range(cursor,cursor+width))):
                    word=candidate;break
        yield cursor,word;cursor+=len(word)


def resolve_citation(text,legacy_tokens,*,normalized_text=None):
    """Resolve ordinary text, preserving a reproducible legacy control arm."""
    from pypinyin import lazy_pinyin,Style
    mode=_MODE.get();plain,hints=strip_reading_hints(text)
    if mode=='legacy' and not hints:return list(legacy_tokens),[]
    if normalized_text is not None and plain!=normalized_text:
        # Explicit digit spelling adds Han positions; hint tone digits have
        # already been removed and corpus #markers never count as text digits.
        remapped={};old=new=0
        for c in re.sub(r'#[1-4]','',plain):
            if '\u3400'<=c<='\u9fff':
                if old in hints:remapped[new]=hints[old]
                old+=1;new+=1
            elif c in '0123456789':new+=1
        hints=remapped;plain=normalized_text
    plain=re.sub(r'#[1-4]','',plain);result=[];events=[];han=0
    for match in _HANS.finditer(plain):
        run=match[0]; protected=set();tokens=[]
        import jieba.posseg as posseg
        from .chinese_training import _tokenizer
        previous_word={};cursor=0;previous=None
        for word in _tokenizer().lcut(run,HMM=False):
            for offset in range(len(word)):previous_word[cursor+offset]=previous
            tagged=list(posseg.cut(word,HMM=False));previous=tagged[0].flag if len(tagged)==1 else None;cursor+=len(word)
        for offset,word in lexical_segments(run):
            added=phrase_entries().get(word)
            if added is not None:
                readings=list(added['pinyin'] if isinstance(added,dict) else added)
                original=lazy_pinyin(word,style=Style.TONE3,neutral_tone_with_five=True,tone_sandhi=False)
                for j,(char,reading) in enumerate(zip(word,readings)):
                    # Neutral 得 is a post-predicate complement marker. A
                    # lexical fragment alone cannot establish that grammar.
                    if char=='得' and reading=='de5' and original[j]!='de5':
                        tag=previous_word.get(offset+j) or ''
                        if not tag.startswith(('v','a')):readings[j]=original[j]
                source='public_large_pinyin_add_only'
            else:
                readings=lazy_pinyin(word,style=Style.TONE3,neutral_tone_with_five=True,tone_sandhi=False)
                source='pypinyin_phrase_dictionary' if len(word)>1 else 'pypinyin_character_default'
            if len(readings)!=len(word):raise ValueError('Pronunciation dictionary changed character coverage')
            tokens.extend(readings)
            if len(word)>1:protected.update(range(offset,offset+len(word)))
            if added is not None:events.append({'start':han+offset,'word':word,'pinyin':list(readings),'source':source})
        offset=0
        for word in _tokenizer().lcut(run,HMM=False):
            correction=train_corrections().get(word)
            if correction:
                readings=correction['pinyin']
                if len(readings)!=len(word):raise ValueError('Train lexical correction changed coverage')
                tokens[offset:offset+len(word)]=readings;protected.update(range(offset,offset+len(word)))
                events.append({'start':han+offset,'word':word,'pinyin':readings,
                    'independent_recordings':correction['independent_recordings'],'source':'train_supported_public_word_correction'})
            offset+=len(word)
        tokens,surname_events=resolve_surnames(run,tokens)
        for event in surname_events:
            protected.update(range(event['start'],event['start']+len(event['word'])))
            events.append({**event,'start':han+event['start']})
        if mode in ('neural','neural_fallback'):
            from .chinese_polyphone_model import predict_readings
            predictions=predict_readings(run,checkpoint=str(_CHECKPOINT.get())) if _CHECKPOINT.get() else predict_readings(run)
            for offset,pred in predictions.items():
                if mode=='neural_fallback' and offset in protected:continue
                if pred['confidence']<_THRESHOLD.get():continue
                old=tokens[offset];tokens[offset]=pred['pinyin']
                if old!=tokens[offset]:events.append({'start':han+offset,'word':run[offset],'previous':old,**pred,'source':'contextual_polyphone_model'})
        for offset in range(len(tokens)):
            if han+offset in hints:
                tokens[offset]=hints[han+offset]
                events.append({'start':han+offset,'word':run[offset],'pinyin':tokens[offset],'source':'explicit_citation_hint'})
        result.extend(tokens);han+=len(run)
    if len(result)!=len(legacy_tokens):raise ValueError('Pronunciation resolution changed Han coverage')
    return result,events
