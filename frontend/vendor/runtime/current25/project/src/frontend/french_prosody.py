"""Opt-in French text phrasing; no lexical-stress or liaison invention.

French final accent belongs to a phrase, not independently to every word.
This deliberately incomplete, weak text annotation marks one final full vowel
at explicit punctuation boundaries and a small set of clause-start cues. It
does not reconstruct every Accentual Phrase, assign an initial accent, impose
an F0 rise, or call unmarked syllables gold unstressed examples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

from . import romance as base

PROFILE = 'french_phrase_prosody_v1'
POLICY = {
    'profile': PROFILE,
    'annotation': 'weak_text_phrase_final_full_vowel_not_human_gold',
    'boundaries': 'explicit_prose_punctuation_plus_conservative_clause_starts',
    'punctuation': '.,!?;:…—–',
    'clause_starts': ['mais', 'car', 'puis', 'cependant', 'pourtant', 'lorsque', 'quand'],
    'minimum_words_before_unpunctuated_clause_start': 2,
    'accent_location': 'last_syllabic_non_schwa_vowel_in_group_containing_content_candidate',
    'schwa': 'IPA ə only; no guessed deletion or reinterpretation of œ',
    'unmarked_stress': 'unknown_-1_including_other_vowels_not_unstressed_gold',
    'initial_accent': 'unknown',
    'pitch_contour': 'not_prescribed',
    'pronunciation': 'unchanged_frozen_first_dictionary_entry',
    'liaison_and_elision': 'no_insertion_deletion_or_dictionary_rewrite',
    'reference_audio_or_duration_used': False,
    'sources': [
        'https://www.sunahjun.humspace.ucla.edu/pdf/Jun_Fougeron%20in%20Botinis.pdf',
        'https://www.isca-archive.org/speechprosody_2002/welby02_speechprosody.pdf',
    ],
}
# This closed set only prevents an all-clitic fragment being labelled as a
# confident content-bearing group. It is not a POS parser or lexicon rewrite.
FUNCTION_WORDS = frozenset((
    'le la les un une des du de au aux ce cet cette ces mon ma mes ton ta tes '
    'son sa ses notre nos votre vos leur leurs je tu il elle on nous vous ils '
    'elles me te se lui y en ne pas plus que qui quoi dont où à dans par pour '
    'avec sans sous sur chez et ou mais ni car donc or puis si lorsque quand '
    'cependant pourtant est sont suis es sommes êtes était étaient être avoir '
    'a ai as avons avez ont avait avaient c d j l m n qu s t'
).split())


def function_word(word):
    return word.rstrip("'") in FUNCTION_WORDS


def phrase_groups(normalized, tokens):
    """Return auditable word ranges; ambiguous interior phrasing stays unknown."""
    groups = []
    first = 0
    for index, (word, start, stop) in enumerate(tokens):
        if (index-first >= POLICY['minimum_words_before_unpunctuated_clause_start']
                and word in POLICY['clause_starts']):
            groups.append({'first_word':first, 'last_word':index-1,
                           'boundary_source':'weak_clause_start_before_'+word})
            first = index
        gap = normalized[stop:tokens[index+1][1] if index+1<len(tokens) else len(normalized)]
        if any(c in POLICY['punctuation'] for c in gap):
            groups.append({'first_word':first, 'last_word':index,
                           'boundary_source':'explicit_text_punctuation', 'punctuation':gap})
            first = index+1
    if first<len(tokens):
        groups.append({'first_word':first, 'last_word':len(tokens)-1,
                       'boundary_source':'end_of_text_without_explicit_punctuation'})
    return groups


class FrenchProsodyFrontend(base.RomanceFrontend):
    def __init__(self, language, lexicon_path, **kwargs):
        if language != 'fr':
            raise ValueError('French phrase prosody frontend requires fr')
        metadata = json.loads(Path(lexicon_path).read_text(encoding='utf-8-sig'))
        if metadata.get('frontend_profile') != PROFILE:
            raise ValueError('French phrase prosody requires explicit matching lexicon frontend_profile')
        del metadata
        super().__init__(language, lexicon_path, **kwargs)
        parent_signature = self.signature
        self.contract = {**self.contract, 'version':PROFILE,
                         'stress_domain':'weak_text_phrase_not_lexical', 'stress_policy':POLICY}
        self.signature = hashlib.sha256(json.dumps({
            'parent_signature':parent_signature, 'contract':self.contract,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def encode(self, text):
        f = super().encode(text)
        normalized, tokens = self._tokens(text)
        groups = phrase_groups(normalized, tokens)
        n = len(f['ipa'])
        stress = np.full(n, -1, np.int64)
        group_ids = np.full(n, -1, np.int64)
        accent_mask = np.zeros(n, bool)
        context = f['context_features'].copy()
        context[:,12:14] = 0
        sources = ['unknown']*len(f['words'])
        for group_id, group in enumerate(groups):
            first, last = group['first_word'], group['last_word']
            a = int(f['word_phone_ranges'][first,0])
            b = int(f['word_phone_ranges'][last,1])
            # SIL units retain no accent group owner; word/plan ownership stays
            # identical to the existing generic French frontend.
            lexical = f['lexical_word_indices'][a:b]>=0
            indices = np.arange(a,b)[lexical]
            group_ids[indices] = group_id
            content = any(not function_word(w) for w in f['words'][first:last+1])
            candidates = [i for i in indices
                if f['phonetic_features'][i,1]>.5 and f['phonetic_features'][i,31]>.5
                and not unicodedata.normalize('NFD',f['ipa'][i]).startswith('ə')]
            target = candidates[-1] if content and candidates else None
            group['accent_phone'] = int(target) if target is not None else None
            group['annotation'] = POLICY['annotation']
            group['content_candidate_present'] = content
            if target is None:
                continue
            stress[target] = 1
            accent_mask[target] = True
            context[target,12:14] = 1
            owner = int(f['lexical_word_indices'][target])
            group['accent_word'] = owner
            sources[owner] = 'french_phrase_final_weak_text_rule'
        f.update(context_features=context,context=context,
                 # The shared planner's ordinary text features retain their
                 # existing 12 columns; its stress inputs read f['stress'].
                 text_features=context[:,:12].copy(),stress=stress,stress_known=stress>=0,
                 stress_sources=sources,accent_group_ids=group_ids,
                 phrase_final_accent_mask=accent_mask,phrase_groups=groups,
                 contract_version=PROFILE)
        f['metadata'] = {**f['metadata'], 'frontend_signature':self.signature,
            'frontend_version':PROFILE, 'stress_annotation_kind':POLICY['annotation'],
            'stress_domain':'phrase_final_not_lexical', 'all_AP_boundaries_recovered':False,
            'initial_accent_human_audited':False, 'pronunciation_modified':False,
            'reference_audio_or_duration_used':False}
        return f
