"""Original English word-plan and Mandarin phrase-plan numeric text adapters."""
from functools import lru_cache
import hashlib,json
from pathlib import Path
import numpy as np


@lru_cache(maxsize=2)
def _zh_front():
    from project.src.frontend.phrase_plan import make_frontend
    return make_frontend()


@lru_cache(maxsize=2)
def signature(language):
    import project.src.frontend
    root=Path(project.src.frontend.__file__).parent
    files=['phonetic_features.py','universal.py']+(['english.py','english_word_prosody.py','english_pause_units.py','universal_english.py'] if language=='en' else ['chinese_training.py','chinese_prosody.py','chinese_delivery_policy.py','phrase_plan.py'])
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    hashes['adapter']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps({'language':language,'sources':hashes,'provided_reference_pronunciation':False},sort_keys=True).encode()).hexdigest()


def stamp(f,language):
    f.setdefault('metadata',{}).update(frontend_signature=signature(language),ordinary_text_only=True,reference_pronunciation_used=False)
    return f


def encode(language,text,metadata=None):
    metadata=metadata or {}
    if language=='en':
        from project.src.frontend.english_word_prosody import encode_text
        from project.src.frontend.universal_english import from_legacy_arrays
        f,p=encode_text(text)
        # expanded_frontend already supplies numeric32/context22 and every
        # optional word pause; do not rebuild edge masks from legacy arrays.
        lexical=np.asarray(f['word_indices'],np.int64).copy();lexical[np.asarray(f['internal_pause_mask'],bool)]=-1
        f.update(planner_numeric=p['numeric'],planner_boundary=p['boundary'],planner_owner=p['word_indices'],lexical_word_indices=lexical,word_phone_ranges=p['word_phone_ranges'],alignment_words=list(f['words']))
        return stamp(f,language)
    if language!='zh':raise ValueError('Only original zh/en frontends belong to this adapter')
    from project.src.frontend.phrase_plan import encode_text,syllable_inputs
    f,p=encode_text(text,metadata.get('vocabulary',{'<pad>':0,'<unk>':1}),frontend=_zh_front())
    if f['context_features'].shape[1]==22:f['context_features']=np.column_stack((f['context_features'],f['prosody_features'])).astype(np.float32)
    owner=np.full(len(f['phonetic_features']),-1,np.int64);syllables=f['metadata']['syllables']
    for i,syllable in enumerate(syllables):
        start=int(syllable['phone_start']);end=int(syllables[i+1]['phone_start']) if i+1<len(syllables) else len(owner)
        owner[start:end]=i
    words=[s['character'] for s in syllables];ranges=np.asarray([(int(s['phone_start']),int(s['phone_end'])) for s in syllables],np.int64)
    lexical=np.full(len(owner),-1,np.int64)
    for i,(first,last) in enumerate(ranges):lexical[first:last]=i
    f.update(planner_numeric=p['numeric'],planner_char_ids=p['char_ids'],planner_owner=owner,words=words,word_phone_ranges=ranges,word_indices=lexical,lexical_word_indices=lexical,alignment_words=words)
    return stamp(f,language)
