"""Conservative lexical erhua routing onto the existing rhotic-rime contract.

Only train-derived, unambiguous whole-word entries are applied. Unknown words,
punctuation-separated characters and conflicting readings remain unchanged.
This module neither supplies acoustic targets nor changes the 32D encoding.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import re

LEXICON_PATH=Path(__file__).with_name('assets')/'chinese_erhua_train_v1.json'
_HAN_RUN=re.compile(r'[\u3400-\u9fff]+')


@lru_cache(maxsize=1)
def lexical_entries():
    resource=json.loads(LEXICON_PATH.read_text())
    if resource.get('resource_schema')!=1:
        raise ValueError('Unsupported Chinese erhua lexicon schema')
    return resource['entries']


def fuse_lexical_erhua(text,characters,tokens,text_levels,corpus_levels,tokenizer,entries=None):
    """Fuse supported word-internal 儿 suffixes while retaining both characters.

    The preceding syllable retains its tone and gains the suffix ``r``. The
    separate 儿 syllable and its intervening optional pause cease to exist.
    Entries may be injected by tests; runtime uses a frozen frontend asset.
    All positions in events refer to the original Han-character sequence.
    """
    if not (len(characters)==len(tokens)==len(text_levels)==len(corpus_levels)):
        raise ValueError('Lexical erhua expects one input token per Han character')
    entries=lexical_entries() if entries is None else entries
    fusions={};events=[];han_cursor=0
    for run in _HAN_RUN.finditer(text):
        run_text=run.group();word_cursor=han_cursor
        for word in tokenizer.lcut(run_text,HMM=False):
            entry=entries.get(word)
            if entry:
                proposed=[]
                for span in entry['fusion_positions']:
                    start=int(span['start']);end=int(span['end'])
                    if end-start!=2 or not 0<=start<end<=len(word) or word[start+1]!='儿':
                        raise ValueError('Malformed frozen erhua entry: '+word)
                    index=word_cursor+start;stem=tokens[index][:-1]
                    identity=span['segmental_identity']
                    # No alternative segmental reading is forced by the table.
                    if (stem!=identity or tokens[index+1][:-1]!='er' or
                            characters[index+1]!='儿' or text_levels[index]!=0 or
                            index in fusions or index+1 in fusions):
                        proposed=[];break
                    proposed.append((index,stem+'r'+tokens[index][-1]))
                if proposed:
                    for index,new_token in proposed:
                        fusions[index]=new_token
                        events.append({'word':word,'character_start':index,'character_stop':index+2,
                          'original_pinyin':tokens[index:index+2],'fused_pinyin':new_token,
                          'source':'train_only_unambiguous_whole_word','independent_recordings':entry['positive_recording_count']})
            word_cursor+=len(word)
        if word_cursor!=han_cursor+len(run_text):
            raise ValueError('Tokenizer changed Chinese character coverage')
        han_cursor+=len(run_text)
    if han_cursor!=len(characters):raise ValueError('Erhua text/character coverage differs')
    new_chars=[];new_tokens=[];new_text=[];new_corpus=[];index=0
    while index<len(characters):
        width=2 if index in fusions else 1
        new_chars.append(''.join(characters[index:index+width]))
        new_tokens.append(fusions.get(index,tokens[index]))
        new_text.append(max(text_levels[index:index+width]));new_corpus.append(max(corpus_levels[index:index+width]))
        index+=width
    return new_chars,new_tokens,new_text,new_corpus,events
