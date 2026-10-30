"""Preparation-only Romanian OOV adapter for public eSpeak NG pronunciations.

Existing RSS entries always win. New stress is deliberately unknown. Explicit
eSpeak Romanian diphthong onsets and word-final palatal offglides preserve the
RSS nonsyllabic vowel representation; no acoustic reference is consulted.
"""
import unicodedata
import hashlib
import json
from pathlib import Path
from .romanian import encode_ipa
from .romanian import RomanianFrontend
from .romanian_alignment import aligner_phone

POLICY = {
    'source': 'eSpeak NG Romanian language rules; OOV only',
    'existing_RSS_entries': 'immutable and preferred',
    'new_lexical_stress': 'unknown_-1; no default stress promoted to gold',
    'tap': 'Romanian phonemic r; eSpeak allophone ɾ -> RSS r',
    'diphthong_onsets': 'e/o/ɔ immediately before a without intervening stress -> RSS e̯/o̯',
    'final_palatal_offglide': 'terminal eSpeak ʲ run -> single RSS i̯',
    'final_ipa_small_cap_i': 'eSpeak ɪ after vowel at word end -> RSS i̯',
    'reference_audio_or_duration_used': False,
}
PROFILE = 'romanian_rss_espeak_oov_v2'
CV_INVENTORY = frozenset(('a','b','dʒ','d̪','e','f','h','i','k','l̪','m','n̪',
                         'o','p','r̪','s','s̪','tʃ','t̪','t̪s','u','v','z̪','ɜ','ɡ','ɨ','ʃ','ʒ'))


def romanian_oov_pronunciation(ipa):
    """Return validated RSS-compatible phones and conservative unknown stress.

    Unsupported phones, compound output and ambiguous internal palatalization
    fail closed rather than using a nearest-phone fallback.
    """
    if not isinstance(ipa,str) or not ipa or any(c.isspace() for c in ipa):
        raise ValueError('One eSpeak Romanian word without pauses required')
    raw=unicodedata.normalize('NFC',ipa).replace('ɾ','r').replace('g','ɡ')
    tail=False
    while raw.endswith('ʲ'):
        tail=True;raw=raw[:-1]
    if 'ʲ' in raw:raise ValueError('Ambiguous internal palatalization')
    # Keep stress temporarily: an intervening stress indicates hiatus, whereas
    # the contiguous Romanian e/o+a representation is a rising diphthong.
    for onset in ('e','o','ɔ'):
        raw=raw.replace(onset+'a',('e̯' if onset=='e' else 'o̯')+'a')
    raw=raw.replace('ˈ','').replace('ˌ','')
    if raw.endswith('ɪ') and len(raw)>1 and raw[-2] in 'aeiouəɨ':raw=raw[:-1]+'i̯'
    phones=[];index=0
    units=('e̯','o̯','i̯','tʃ','dʒ','ts')
    while index<len(raw):
        unit=next((p for p in units if raw.startswith(p,index)),raw[index])
        index+=len(unit);phones.append(unit)
    if tail:phones.append('i̯')
    if not phones:raise ValueError('Empty pronunciation')
    for phone in phones:
        encode_ipa(phone)
        if aligner_phone(phone) not in CV_INVENTORY:
            raise ValueError('Phone outside verified Romanian CV alignment inventory: '+phone)
    return phones,[-1]*len(phones)


class RomanianExpandedFrontend(RomanianFrontend):
    """Explicit source metadata for a new lexicon; numeric RSS contract intact."""
    def __init__(self, language, lexicon_path, **kwargs):
        data=json.loads(Path(lexicon_path).read_text(encoding='utf-8-sig'))
        if data.get('frontend_profile')!=PROFILE:
            raise ValueError('Explicit Romanian OOV profile required')
        self.word_sources=data.get('word_sources',{})
        if set(self.word_sources)!=set(data.get('entries',{})):
            raise ValueError('Every Romanian word requires a frozen source')
        for word,source in self.word_sources.items():
            if source not in ('RSS_original_training_lexical_annotation','espeak_ng_1_51_romanian_OOV_stress_unknown'):
                raise ValueError('Unknown Romanian word provenance')
            if source.endswith('OOV_stress_unknown') and any(s!=-1 for s in data['stress'][word]):
                raise ValueError('Expanded Romanian OOV stress must remain unknown')
        super().__init__(language,lexicon_path,**kwargs)
        parent=self.signature
        self.contract={**self.contract,'version':PROFILE,
                       'pronunciation_source_policy':'RSS_immutable_plus_explicit_espeak_OOV',
                       'new_OOV_stress':'unknown_-1'}
        self.signature=hashlib.sha256(json.dumps({'parent_signature':parent,
            'contract':self.contract,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def encode(self,text):
        f=super().encode(text)
        sources=[self.word_sources[w] for w in f['words']]
        f['pronunciation_sources']=sources
        f['stress_sources']=['RSS_public_lexical_annotation_not_human_audited'
            if s=='RSS_original_training_lexical_annotation' else 'unknown_new_OOV'
            for s in sources]
        f['contract_version']=PROFILE
        f['metadata']={**f['metadata'],'frontend_signature':self.signature,
            'frontend_version':PROFILE,'pronunciation_source_policy':'per_word_frozen_source',
            'new_OOV_stress_unknown':True,'numeric_RSS_contract_unchanged':True}
        return f
