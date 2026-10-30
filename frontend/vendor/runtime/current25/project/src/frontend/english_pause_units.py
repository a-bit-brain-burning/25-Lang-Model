"""Versioned English word-boundary pause topology, with no acoustic inference.

Word interpretation remains in the English frontend. Every interword boundary
gets an optional SIL unit; the numeric duration gate decides whether it exists.
The original lexical features/positions are retained for checkpoint compatibility.
"""
from __future__ import annotations
import numpy as np
from .english import EnglishFrontend,PHONE_TO_ID
from .universal_english import from_legacy_arrays
VERSION='english_optional_word_pause_v1'


def insert_word_pauses(legacy):
    ids=np.asarray(legacy['phone_ids'],np.int64);text=np.asarray(legacy['text_features'],np.float32)
    stress=np.asarray(legacy.get('stress',np.full(len(ids),-1)),np.int64)
    if ids.ndim!=1 or text.shape!=(len(ids),12) or stress.shape!=ids.shape:raise ValueError('Invalid legacy features')
    if not np.isfinite(text).all():raise ValueError('Nonfinite legacy text features')
    edge=np.asarray(legacy.get('edge_sil_mask',text[:,10:12].sum(-1)>.5),bool)
    lexical=ids!=PHONE_TO_ID['<SIL>']
    starts=np.flatnonzero(lexical&(text[:,0]>.5));ends=np.flatnonzero(lexical&(text[:,1]>.5))
    if len(starts)!=len(ends) or not len(starts) or np.any(starts>ends):raise ValueError('Word topology mismatch')
    if any(np.any(~lexical[a:b+1]) for a,b in zip(starts,ends)):raise ValueError('Existing internal silence is unsupported')
    oldword=np.full(len(ids),-1,np.int64)
    for w,(a,b) in enumerate(zip(starts,ends)):oldword[a:b+1]=w
    if np.any(lexical&(oldword<0)):raise ValueError('Lexical phone outside word spans')
    after={int(b):w for w,b in enumerate(ends[:-1])};source=[];pause=[];words=[];newids=[];newtext=[];newstress=[];newedge=[]
    for j in range(len(ids)):
        source.append(j);pause.append(-1);words.append(int(oldword[j]));newids.append(ids[j]);newtext.append(text[j]);newstress.append(stress[j]);newedge.append(edge[j])
        if j in after:
            w=after[j];x=np.zeros(12,np.float32);x[4:8]=text[j,4:8];x[5]=.5*(text[j,5]+text[starts[w+1],5]);x[8:10]=(text[j,8],text[starts[w+1],9])
            source.append(-1);pause.append(w);words.append(w);newids.append(PHONE_TO_ID['<SIL>']);newtext.append(x);newstress.append(-1);newedge.append(True)
    result={'phone_ids':np.asarray(newids,np.int64),'text_features':np.asarray(newtext,np.float32),'stress':np.asarray(newstress,np.int64),
            'edge_sil_mask':np.asarray(newedge,bool),'source_indices':np.asarray(source,np.int64),'word_indices':np.asarray(words,np.int64),
            'pause_after_word':np.asarray(pause,np.int64),'internal_pause_mask':np.asarray(pause)>=0,'word_count':len(starts),
            'word_boundary':np.asarray(newtext,np.float32)[:,:2]>.5,'topology_version':VERSION}
    result.update(from_legacy_arrays(result['phone_ids'],result['text_features'],result['stress']))
    # encode_phones derives only utterance-edge masks from context; restore all
    # optional internal slots explicitly, without mislabeling first/last edges.
    result['edge_sil_mask']=np.asarray(newedge,bool)
    for k in ('words','normalized_text','punctuated_text','oov_words'):
        if k in legacy:result[k]=legacy[k]
    return result


class EnglishPauseFrontend:
    def __init__(self,**kwargs):self.legacy=EnglishFrontend(**kwargs)
    def encode(self,text):
        raw=self.legacy.encode(text);result=insert_word_pauses(raw)
        result['metadata'].update(language='en',frontend_version=VERSION,english_policy_applied=False,
            words=raw['words'],normalized_text=raw['normalized_text'],oov_words=raw.get('oov_words',[]))
        return result
