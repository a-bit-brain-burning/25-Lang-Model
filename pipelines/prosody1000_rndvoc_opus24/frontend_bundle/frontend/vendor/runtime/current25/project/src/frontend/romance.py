"""Strict text/dictionary frontend for es/fr/it/pt; numeric36 plus context22.

This independent version retains the peer's four extra articulatory fields.
It never reads recordings, aligned phones, durations or reference prosody.
Unknown words/phones raise; dictionary/G2P expansion belongs to a versioned
preparation step, not an acoustic inference fallback.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
import unicodedata

import numpy as np

from . import phonetic_features as v1
from .universal import CONTEXT_NAMES

VERSION = 'romance_dictionary36_plan_owner_v1'
PHONETIC_VERSION = 'experimental_phonetic_features_v2_36_spanish1_labialized'
LANGUAGES = frozenset(('es', 'fr', 'it', 'pt'))
EXTRA_NAMES = ('manner_tap', 'manner_trill', 'place_alveolopalatal', 'consonant_round')
FEATURE_NAMES = v1.FEATURE_NAMES + EXTRA_NAMES
FEATURE_DIM = 36
MODIFIERS = {'ʰ': 'aspirated', 'ː': 'long', '̃': 'nasalized', '̩': 'syllabic', '˞': 'rhotic'}
TIE_BARS = frozenset(('͡', '͜'))
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*'?", flags=re.UNICODE)
_PUNCTUATION = ".,!?;:¿¡—–-()«»\"'…"


@dataclass(frozen=True)
class PhoneticSegment(v1.PhoneticSegment):
    consonant_round: bool = False


def _inventory():
    items = {ipa: PhoneticSegment(**segment.__dict__, consonant_round=ipa in {'w', 'ɥ'})
             for ipa, segment in v1.INVENTORY.items()}
    for ipa in ('ɕ', 'tɕ'):
        items[ipa] = replace(items[ipa], place='alveolopalatal')
    items['r'] = PhoneticSegment('r', 'trill', 'alveolar', voiced=True, rhotic=True)
    items['ɾ'] = PhoneticSegment('ɾ', 'tap', 'alveolar', voiced=True, rhotic=True)
    for ipa, manner, place, voiced in (
        ('t̪', 'stop', 'dental', False), ('d̪', 'stop', 'dental', True),
        ('c', 'stop', 'palatal', False), ('ɟ', 'stop', 'palatal', True),
        ('ɟʝ', 'affricate', 'palatal', True), ('ɱ', 'nasal', 'labiodental', True),
        ('ʝ', 'fricative', 'palatal', True), ('ɣ', 'fricative', 'velar', True),
        ('β', 'fricative', 'bilabial', True),
    ):
        items[ipa] = PhoneticSegment(ipa, manner, place, voiced=voiced)
    return MappingProxyType(items)


INVENTORY = _inventory()
_BASE = {unicodedata.normalize('NFD', ipa): segment for ipa, segment in INVENTORY.items()}
_LONGEST_FIRST = tuple(sorted(_BASE, key=lambda item: (-len(item), item)))


def segment_from_ipa(ipa):
    """One segmented IPA phone; lexical stress is handled by phone_and_stress."""
    if not isinstance(ipa, str) or not ipa or any(c.isspace() for c in ipa):
        raise ValueError('One nonempty segmented IPA phone without whitespace is required')
    normalized = unicodedata.normalize('NFD', ipa).replace('g', 'ɡ')
    rounded = normalized.endswith('ʷ')
    if 'ʷ' in normalized and (not rounded or normalized.count('ʷ') != 1):
        raise ValueError('Unsupported labialization placement: '+ipa)
    if rounded:
        normalized = normalized[:-1]
    ties = [i for i, character in enumerate(normalized) if character in TIE_BARS]
    if ties and (len(ties) != 1 or ties[0] != 1):
        raise ValueError('Tie bar must join the two bases of a supported affricate: '+ipa)
    untied = ''.join(c for c in normalized if c not in TIE_BARS)
    key = next((candidate for candidate in _LONGEST_FIRST if untied.startswith(candidate)), None)
    if key is None:
        raise ValueError('Unsupported segmented IPA phone: '+ipa)
    base = _BASE[key]
    if ties and (base.manner != 'affricate' or len(key) != 2):
        raise ValueError('Tie bar is supported only for known affricates: '+ipa)
    tail = untied[len(key):]
    if base.manner == 'silence' and (tail or rounded):
        raise ValueError('Silence does not accept phonetic modifiers: '+ipa)
    if rounded and base.manner in ('vowel', 'silence'):
        raise ValueError('Only consonants accept labialization: '+ipa)
    seen = set()
    flags = {}
    for mark in tail:
        if mark not in MODIFIERS or mark in seen:
            raise ValueError('Unsupported or repeated IPA modifier/additional segment: '+ipa)
        seen.add(mark)
        flags[MODIFIERS[mark]] = True
    return replace(base, ipa=ipa, consonant_round=rounded or base.consonant_round, **flags)


def encode_segment(segment):
    manner = v1.MANNERS + ('tap', 'trill')
    places = v1.PLACES + ('alveolopalatal',)
    if segment.manner not in manner or (segment.place is not None and segment.place not in places):
        raise ValueError('Unsupported romance articulatory category')
    if segment.manner in ('vowel', 'silence') and segment.consonant_round:
        raise ValueError('Consonant rounding is not a vowel/silence feature')
    x = np.zeros(FEATURE_DIM, np.float32)
    x[v1.MANNERS.index(segment.manner) if segment.manner in v1.MANNERS
      else 32 if segment.manner == 'tap' else 33] = 1
    if segment.place == 'alveolopalatal':
        x[34] = 1
    elif segment.place is not None:
        x[len(v1.MANNERS)+v1.PLACES.index(segment.place)] = 1
    x[17:24] = [segment.voiced, segment.aspirated, segment.lateral, segment.rhotic,
                segment.nasalized, segment.long, segment.geminate]
    end = segment.vowel_end if segment.vowel_end is not None else segment.vowel_start
    if len(segment.vowel_start) != 3 or len(end) != 3:
        raise ValueError('Vowel coordinates require height/backness/roundness')
    x[24:27], x[27:30] = segment.vowel_start, end
    x[30] = segment.vowel_end is not None and tuple(end) != tuple(segment.vowel_start)
    x[31] = segment.syllabic or segment.manner == 'vowel'
    x[35] = segment.consonant_round
    if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('Phonetic features must be finite in [0,1]')
    return x


def encode_ipa(ipa):
    if "ʲ" in ipa:
        if ipa.count("ʲ") != 1 or not ipa.endswith("ʲ"):
            raise ValueError("Unsupported palatalization placement: " + ipa)
        segment = segment_from_ipa(ipa[:-1])
        if segment.manner in ("vowel", "silence"):
            raise ValueError("Only consonants accept palatalization: " + ipa)
        result = encode_segment(segment)
        result[len(v1.MANNERS) + v1.PLACES.index("palatal")] = 1.
        return result
    return encode_segment(segment_from_ipa(ipa))


def phone_and_stress(ipa):
    if not isinstance(ipa, str) or not ipa:
        raise ValueError('Expected nonempty dictionary phone')
    stress = -1
    if ipa.startswith(('ˈ', 'ˌ')):
        stress, ipa = (1 if ipa[0] == 'ˈ' else 2), ipa[1:]
    encode_ipa(ipa)
    if ipa == 'sil':
        raise ValueError('Dictionary pronunciation must contain lexical phones, not SIL')
    return ipa, stress


def _normalized(text):
    return unicodedata.normalize('NFC', text).casefold().replace('’', "'").replace('ʼ', "'")


def contract(language):
    if language not in LANGUAGES:
        raise ValueError('Romance frontend language must be es/fr/it/pt')
    return {'version': VERSION, 'language': language, 'phonetic_version': PHONETIC_VERSION,
            'phonetic_dim': FEATURE_DIM, 'phonetic_names': list(FEATURE_NAMES),
            'context_dim': len(CONTEXT_NAMES), 'context_names': list(CONTEXT_NAMES),
            'pause_topology': 'one_optional_SIL_per_interword_boundary_plus_utterance_edges',
            'plan_mapping': 'lexical owner for phones; left-word owner for internal SIL; -1 at edges',
            'OOV_policy': 'error', 'reference_inputs': False}


def load_lexicon(path, language):
    """JSON entries or MFA word [one/four numeric metadata fields] IPA phones.

    A fixed first dictionary pronunciation is selected without acoustic search.
    Individual phone validity is checked when a word is requested, so a large
    official dictionary's unsupported unused foreign entries do not invalidate
    all otherwise supported text. validate_all=True can audit a frozen subset.
    """
    contract(language)
    path = Path(path)
    content = path.read_text(encoding='utf-8-sig')
    rows = []
    provenance = {}
    if content.lstrip().startswith('{'):
        value = json.loads(content)
        if value.get('language', language) != language:
            raise ValueError('Dictionary language differs from requested frontend')
        entries = value.get('entries')
        if not isinstance(entries, dict) or not entries:
            raise ValueError('Expected a nonempty dictionary entries object')
        rows = list(entries.items())
        provenance = {'format': value.get('format', 'explicit_json_dictionary'),
                      'source': value.get('provenance', value.get('source', {}))}
    else:
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            word, *phones = line.split()
            numeric = []
            while phones:
                try:
                    value = float(phones[0])
                except ValueError:
                    break
                numeric.append(value)
                phones.pop(0)
            if len(numeric) not in (0, 1, 4) or not np.isfinite(numeric).all():
                raise ValueError(f'Unsupported MFA probability columns on dictionary line {number}')
            if numeric and (not 0 <= numeric[0] <= 1 or any(x < 0 for x in numeric)):
                raise ValueError(f'Invalid MFA probability metadata on dictionary line {number}')
            rows.append((word, phones))
        provenance = {'format': 'MFA_text_first_pronunciation'}
    entries = {}
    variants = Counter()
    for word, phones in rows:
        if not isinstance(word, str) or not word or any(c.isspace() for c in word):
            raise ValueError('Invalid dictionary word')
        if not isinstance(phones, list) or not phones or any(not isinstance(p, str) or not p for p in phones):
            raise ValueError('Invalid dictionary pronunciation: '+word)
        normalized = _normalized(word)
        variants[normalized] += 1
        entries.setdefault(normalized, tuple(phones))
    if not entries:
        raise ValueError('Empty dictionary')
    return entries, dict(variants), provenance


def _punctuation_strength(text):
    if any(c in '.!?…' for c in text):
        return 1.
    if any(c in ';:' for c in text):
        return .65
    if ',' in text:
        return .4
    if any(c in '—–' for c in text):
        return .25
    return 0.


class RomanceFrontend:
    def __init__(self, language, lexicon_path, *, max_phones=256, validate_all=False):
        self.language = language
        self.contract = contract(language)
        self.path = Path(lexicon_path).resolve()
        self.entries, self.variants, self.provenance = load_lexicon(self.path, language)
        self.max_phones = int(max_phones)
        if not 1 <= self.max_phones <= 256:
            raise ValueError('Safe romance frontend phone limit must be within 1..256')
        self.lexicon_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.signature = hashlib.sha256(json.dumps(
            {'contract': self.contract, 'lexicon_sha256': self.lexicon_sha256,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'max_phones': self.max_phones}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self._pronunciations = {}
        if validate_all:
            for word in self.entries:
                self.pronounce(word)

    def pronounce(self, word):
        word = _normalized(word)
        if word not in self.entries:
            raise ValueError('OOV in frozen '+self.language+' dictionary: '+word)
        if word not in self._pronunciations:
            try:
                self._pronunciations[word] = tuple(phone_and_stress(p) for p in self.entries[word])
            except ValueError as error:
                raise ValueError('Invalid pronunciation for '+word+': '+str(error)) from error
        return self._pronunciations[word]

    def _tokens(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Nonempty text required')
        text = _normalized(text)
        if any(not (c.isalpha() or c.isspace() or c in _PUNCTUATION) for c in text):
            raise ValueError('Unsupported text character; numbers require language-specific normalization')
        result = []
        for match in _TOKEN.finditer(text):
            word, a, b = match.group(), match.start(), match.end()
            if word in self.entries:
                result.append((word, a, b))
            elif "'" in word:
                prefix, rest = word.split("'", 1)
                prefix += "'"
                if prefix not in self.entries or rest not in self.entries:
                    raise ValueError('OOV contraction in frozen '+self.language+' dictionary: '+word)
                result.extend(((prefix, a, a+len(prefix)), (rest, a+len(prefix), b)))
            else:
                raise ValueError('OOV in frozen '+self.language+' dictionary: '+word)
        if not result:
            raise ValueError('Text contains no dictionary words')
        return text, result

    def words(self, text):
        return [word for word, _, _ in self._tokens(text)[1]]

    def encode(self, text):
        normalized, tokens = self._tokens(text)
        words = [item[0] for item in tokens]
        pronunciations = [self.pronounce(word) for word in words]
        n = 1+len(words)+sum(map(len, pronunciations))
        if n > self.max_phones:
            raise ValueError(f'Text exceeds {self.max_phones} phones including optional pauses')
        ipas, stresses, owners = ['sil'], [-1], [-1]
        plan_owner, pause_after, internal = [-1], [-1], [False]
        ranges, positions = [], [(0, 1)]
        for wi, pronunciation in enumerate(pronunciations):
            first = len(ipas)
            for pi, (phone, stress) in enumerate(pronunciation):
                ipas.append(phone); stresses.append(stress); owners.append(wi)
                plan_owner.append(wi); pause_after.append(-1); internal.append(False)
                positions.append((pi, len(pronunciation)))
            ranges.append((first, len(ipas)))
            is_internal = wi+1 < len(words)
            ipas.append('sil'); stresses.append(-1); owners.append(-1)
            plan_owner.append(wi if is_internal else -1)
            pause_after.append(wi if is_internal else -1); internal.append(is_internal)
            positions.append((0, 1))
        context = np.zeros((n, len(CONTEXT_NAMES)), np.float32)
        word_before, word_after = [], []
        for wi, (_, start, stop) in enumerate(tokens):
            before = normalized[tokens[wi-1][2] if wi else 0:start]
            after = normalized[stop:tokens[wi+1][1] if wi+1 < len(tokens) else len(normalized)]
            word_before.append(_punctuation_strength(before))
            word_after.append(_punctuation_strength(after))
        for i, (owner, (pi, size), stress) in enumerate(zip(owners, positions, stresses)):
            context[i, 5:8] = i/max(n-1, 1), n/256, min(len(words)/64, 1.)
            if owner >= 0:
                context[i, :5] = pi == 0, pi == size-1, pi/max(size-1, 1), min(size/20, 1.), owner/max(len(words)-1, 1)
                context[i, 8:10] = word_after[owner] if pi == size-1 else 0., word_before[owner] if pi == 0 else 0.
                if stress >= 0:
                    context[i, 12:14] = 1. if stress == 1 else .5, 1.
            elif internal[i]:
                # Retain the peer's weak optional-boundary slot. Real punctuation
                # is also explicitly available on lexical ends and word metadata.
                context[i, 8:10] = .25, .25
        context[0, 10] = context[-1, 11] = 1.
        context[-1, 8] = 1.
        phonetic = np.stack([encode_ipa(p) for p in ipas])
        lexical = np.asarray(owners, np.int64)
        stress = np.asarray(stresses, np.int64)
        return {'phonetic_features': phonetic, 'context_features': context,
                'phonetic': phonetic, 'context': context, 'text_features': context[:, :12].copy(),
                'valid_mask': np.ones(n, bool), 'edge_sil_mask': np.asarray([p == 'sil' for p in ipas]),
                'ipa': ipas, 'phones': list(ipas), 'words': words, 'normalized_text': normalized,
                'lexical_word_indices': lexical, 'word_indices': lexical.copy(),
                'plan_owner': np.asarray(plan_owner, np.int64), 'pause_after_word': np.asarray(pause_after, np.int64),
                'internal_pause_mask': np.asarray(internal, bool),
                'word_phone_ranges': np.asarray(ranges, np.int64),
                'stress': stress, 'stress_known': stress >= 0,
                'vowel_mask': phonetic[:, 1] > .5, 'voiced_phone_mask': phonetic[:, 17] > .5,
                'word_punctuation_before': np.asarray(word_before, np.float32),
                'word_punctuation_after': np.asarray(word_after, np.float32),
                'stress_sources': ['dictionary' if any(s >= 0 for _, s in p) else 'unknown' for p in pronunciations],
                'pronunciation_sources': ['frozen_dictionary']*len(words),
                'phonetic_feature_names': FEATURE_NAMES, 'context_feature_names': CONTEXT_NAMES,
                'contract_version': VERSION,
                'metadata': {'language': self.language, 'frontend_signature': self.signature,
                    'frontend_version': VERSION, 'phonetic_version': PHONETIC_VERSION,
                    'lexicon_sha256': self.lexicon_sha256, 'OOV_policy': 'error',
                    'pronunciation_selection': 'first dictionary entry; no acoustic search',
                    'reference_audio_or_duration_used': False, 'lexical_stress_human_audited': False,
                    'unknown_stress_is_not_unstressed_gold': True}}
