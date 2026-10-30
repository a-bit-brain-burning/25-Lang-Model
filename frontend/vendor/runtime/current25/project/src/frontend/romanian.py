"""Frozen RSS dictionary frontend; ordinary text only, numeric36/context22.

RSS e@/o@ are nonsyllabic e/o followed by a separately labelled vowel;
ij is the separately timed nonsyllabic i. They are never expanded into ea/oa.
The public dictionary carries lexical stress; no reference timing is read here.
"""
from pathlib import Path
import hashlib
import json
import unicodedata
import numpy as np
from . import romance as base

VERSION = 'romanian_rss_dictionary36_v1'
RSS_IPA = dict(zip(
    ['a','@','a@','b','k','ch','d','e','e@','f','g','dz','h','i','j','ij',
     'zh','l','m','n','o','o@','p','r','s','sh','t','ts','u','w','v','z'],
    ['a','ə','ɨ','b','k','tʃ','d','e','e̯','f','ɡ','dʒ','h','i','j','i̯',
     'ʒ','l','m','n','o','o̯','p','r','s','ʃ','t','ts','u','w','v','z']))


def normalize_text(text):
    return base._normalized(text).translate(str.maketrans({'ş':'ș','ţ':'ț'}))


def encode_ipa(phone):
    if phone in ('e̯','o̯','i̯'):
        value = base.encode_ipa(phone[0]).copy()
        value[31] = 0.  # explicit nonsyllabic vowel, same 36 numeric coordinates
        return value
    return base.encode_ipa(phone)


class RomanianFrontend(base.RomanceFrontend):
    def __init__(self, language, lexicon_path, *, max_phones=256, validate_all=False):
        if language != 'ro':
            raise ValueError('Romanian frontend requires ro')
        self.language = language
        self.path = Path(lexicon_path).resolve()
        data = json.loads(self.path.read_text())
        if data.get('language') != 'ro' or not data.get('entries'):
            raise ValueError('Expected a frozen Romanian dictionary')
        self.entries = {}
        self.stresses = {}
        for word, phones in data['entries'].items():
            canonical = normalize_text(word)
            if canonical != word or not phones or not isinstance(phones, list):
                raise ValueError('Noncanonical Romanian dictionary entry')
            if any(not isinstance(phone,str) or not phone or phone=='sil' for phone in phones):
                raise ValueError('Nonempty lexical phones required; lexical silence forbidden')
            stress = data.get('stress', {}).get(word, [-1]*len(phones))
            if len(stress) != len(phones) or any(s not in (-1,0,1,2) for s in stress):
                raise ValueError('Invalid lexical stress labels')
            self.entries[word], self.stresses[word] = tuple(phones), tuple(stress)
        self.variants = {w:1 for w in self.entries}
        self.provenance = data.get('provenance', {})
        self.max_phones = int(max_phones)
        if not 1 <= self.max_phones <= 256:
            raise ValueError('Phone limit must be within1..256')
        self.lexicon_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.contract = {**base.contract('it'), 'language':'ro', 'version':VERSION,
                         'nonsyllabic_vowels':['e̯','o̯','i̯'],
                         'orthography':'NFC casefold; cedilla s/t -> comma s/t'}
        self.signature = hashlib.sha256(json.dumps({
            'contract':self.contract,'lexicon_sha256':self.lexicon_sha256,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'shared_source_sha256':hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
            'max_phones':self.max_phones},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        self._pronunciations = {}
        if validate_all:
            for word in self.entries:
                self.pronounce(word)

    def _tokens(self, text):
        return super()._tokens(normalize_text(text))

    def pronounce(self, word):
        word = normalize_text(word)
        if word not in self.entries:
            raise ValueError('OOV in frozen ro dictionary: '+word)
        if word not in self._pronunciations:
            for phone in self.entries[word]:
                encode_ipa(phone)
            self._pronunciations[word] = tuple(zip(self.entries[word],self.stresses[word]))
        return self._pronunciations[word]

    def encode(self, text):
        normalized, tokens = self._tokens(text)
        words = [item[0] for item in tokens]
        pronunciations = [self.pronounce(word) for word in words]
        n = 1+len(words)+sum(map(len,pronunciations))
        if n > self.max_phones:
            raise ValueError(f'Text exceeds {self.max_phones} phones including optional pauses')
        ipas, stresses, owners = ['sil'], [-1], [-1]
        plan_owner, pause_after, internal = [-1], [-1], [False]
        ranges, positions = [], [(0,1)]
        for wi, pronunciation in enumerate(pronunciations):
            first = len(ipas)
            for pi,(phone,stress) in enumerate(pronunciation):
                ipas.append(phone);stresses.append(stress);owners.append(wi)
                plan_owner.append(wi);pause_after.append(-1);internal.append(False)
                positions.append((pi,len(pronunciation)))
            ranges.append((first,len(ipas)));is_internal = wi+1 < len(words)
            ipas.append('sil');stresses.append(-1);owners.append(-1)
            plan_owner.append(wi if is_internal else -1)
            pause_after.append(wi if is_internal else -1);internal.append(is_internal)
            positions.append((0,1))
        context = np.zeros((n,len(base.CONTEXT_NAMES)),np.float32)
        before,after = [],[]
        for wi,(_,start,stop) in enumerate(tokens):
            before.append(base._punctuation_strength(normalized[tokens[wi-1][2] if wi else 0:start]))
            after.append(base._punctuation_strength(normalized[stop:tokens[wi+1][1] if wi+1<len(tokens) else len(normalized)]))
        for i,(owner,(pi,size),stress) in enumerate(zip(owners,positions,stresses)):
            context[i,5:8] = i/max(n-1,1),n/256,min(len(words)/64,1.)
            if owner >= 0:
                context[i,:5] = pi==0,pi==size-1,pi/max(size-1,1),min(size/20,1.),owner/max(len(words)-1,1)
                context[i,8:10] = after[owner] if pi==size-1 else 0.,before[owner] if pi==0 else 0.
                if stress >= 0:
                    context[i,12:14] = 1. if stress==1 else .5 if stress==2 else 0.,1.
            elif internal[i]:
                context[i,8:10] = .25,.25
        context[0,10] = context[-1,11] = 1.;context[-1,8] = 1.
        phonetic = np.stack([encode_ipa(p) for p in ipas])
        lexical = np.asarray(owners,np.int64);stress = np.asarray(stresses,np.int64)
        return {'phonetic_features':phonetic,'context_features':context,
                'phonetic':phonetic,'context':context,'text_features':context[:,:12].copy(),
                'valid_mask':np.ones(n,bool),'edge_sil_mask':np.asarray([p=='sil' for p in ipas]),
                'ipa':ipas,'phones':list(ipas),'words':words,'normalized_text':normalized,
                'lexical_word_indices':lexical,'word_indices':lexical.copy(),
                'plan_owner':np.asarray(plan_owner,np.int64),'pause_after_word':np.asarray(pause_after,np.int64),
                'internal_pause_mask':np.asarray(internal,bool),'word_phone_ranges':np.asarray(ranges,np.int64),
                'stress':stress,'stress_known':stress>=0,
                'vowel_mask':(phonetic[:,1]>.5)&(phonetic[:,31]>.5),'voiced_phone_mask':phonetic[:,17]>.5,
                'word_punctuation_before':np.asarray(before,np.float32),'word_punctuation_after':np.asarray(after,np.float32),
                'stress_sources':['RSS_public_lexical_annotation_not_human_audited']*len(words),
                'pronunciation_sources':['frozen_RSS_dictionary']*len(words),
                'phonetic_feature_names':base.FEATURE_NAMES,'context_feature_names':base.CONTEXT_NAMES,
                'contract_version':VERSION,'metadata':{'language':'ro','frontend_signature':self.signature,
                'frontend_version':VERSION,'lexicon_sha256':self.lexicon_sha256,'OOV_policy':'error',
                'reference_audio_or_duration_used':False,'lexical_stress_human_audited':False}}
