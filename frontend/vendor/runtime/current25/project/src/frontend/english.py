"""English text-only frontend with a frozen, alignment-compatible CMU lexicon.

The twelve model features reproduce the existing phase34 contract. Lexical
stress is metadata only: the historical PocketSphinx lexicon removes it. OOV
words use explicitly labelled approximate spelling rules, never acoustic data.
"""
from __future__ import annotations

import abc
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import torch

PHONES = 'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split()
PHONE_TO_ID = {p: i for i, p in enumerate(['<UNK>', '<BOS>', '<EOS>', '<SIL>'] + PHONES)}
VOCABULARY = PHONE_TO_ID
FEATURE_NAMES = ('word_start', 'word_end', 'phone_position_in_word', 'word_phone_length_div12',
                 'word_position_in_sentence', 'phone_position_in_sentence', 'sentence_phone_length_div200',
                 'sentence_word_length_div50', 'punctuation_after_strength', 'punctuation_before_strength',
                 'first_edge', 'last_edge')
FEATURE_DIM = 12
FRONTEND_VERSION = 'english_cmudict_phase34_compatible_v1'
LANGUAGE_IDS = {'en': 1, 'zh': 2, 'ja': 3}
_ASSETS = Path(__file__).resolve().parent / 'assets'
_SMALL = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
_TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
_ABBR = {'mrs': 'missus', 'mr': 'mister', 'ms': 'miss', 'dr': 'doctor', 'prof': 'professor', 'st': 'saint',
         'jr': 'junior', 'sr': 'senior', 'vs': 'versus', 'etc': 'et cetera', 'dept': 'department', 'no': 'number'}
_ORDINAL = {'one': 'first', 'two': 'second', 'three': 'third', 'four': 'fourth', 'five': 'fifth',
            'six': 'sixth', 'seven': 'seventh', 'eight': 'eighth', 'nine': 'ninth', 'ten': 'tenth',
            'eleven': 'eleventh', 'twelve': 'twelfth'}
_VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}


def integer_words(number):
    n = int(number)
    if n < 0:
        return 'minus ' + integer_words(-n)
    if n < 20:
        return _SMALL[n]
    if n < 100:
        return _TENS[n // 10] + (' ' + _SMALL[n % 10] if n % 10 else '')
    if n < 1000:
        return _SMALL[n // 100] + ' hundred' + (' ' + integer_words(n % 100) if n % 100 else '')
    for scale, name in [(10**12, 'trillion'), (10**9, 'billion'), (10**6, 'million'), (1000, 'thousand')]:
        if n >= scale:
            if n >= 10**15:
                raise ValueError('Number outside supported normalization range')
            return integer_words(n // scale) + ' ' + name + (' ' + integer_words(n % scale) if n % scale else '')
    raise ValueError('Invalid number')


def _ordinal(m):
    words = integer_words(m.group(1)).split()
    last = words[-1]
    words[-1] = _ORDINAL.get(last, last[:-1] + 'ieth' if last.endswith('y') else last + 'th')
    return ' '.join(words)


def normalize_with_punctuation(text):
    """Normalize numbers and common abbreviations, preserving phrase punctuation."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty text')
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace('—', ';').replace('–', '-')
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)).lower()
    for abbr, expanded in _ABBR.items():
        text = re.sub(r'\b' + abbr + r'\.(?=\s|$)', expanded, text)
    text = re.sub(r'\b([a-z])\.(?=[a-z]\.)', r'\1 ', text)
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)

    def currency(m):
        whole, decimal = m.group(1), m.group(2)
        value = integer_words(whole) + ' dollar' + ('s' if int(whole) != 1 else '')
        if decimal and int(decimal):
            cents = int(decimal.ljust(2, '0'))
            value += ' and ' + integer_words(cents) + ' cent' + ('s' if cents != 1 else '')
        return value

    text = re.sub(r'\$(\d+)(?:\.(\d{1,2}))?', currency, text)
    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', lambda m: integer_words(m.group(1)) + ' ' +
                  ('oh ' + integer_words(m.group(2)) if 0 < int(m.group(2)) < 10 else integer_words(m.group(2))), text)
    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', _ordinal, text)
    text = re.sub(r'\b(\d+)\.(\d+)\b', lambda m: integer_words(m.group(1)) + ' point ' +
                  ' '.join(_SMALL[int(x)] for x in m.group(2)), text)
    text = re.sub(r'\b\d+\b', lambda m: integer_words(m.group()), text).replace('%', ' percent ').replace('&', ' and ')
    text = text.replace('-', ' ')
    if re.search(r"[^a-z\s'.,!?;:\"()\[\]]", text):
        raise ValueError('Unsupported text symbol in: ' + text)
    text = re.sub(r"(^|\s)'+|'+(?=\s|$)", ' ', text)
    return ' '.join(text.split())


class TextDictionary:
    """Read only the lexicon; no PocketSphinx acoustic models are loaded."""
    def __init__(self, path=None):
        self.path = Path(path) if path else _ASSETS / 'cmudict-pocketsphinx.dict'
        self.entries = self._read(self.path)
        stress_path = _ASSETS / 'cmudict-stress.dict'
        self.stress_entries = self._read(stress_path) if stress_path.exists() else {}

    @staticmethod
    def _read(path):
        entries = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line or line.startswith((';;;', '#')):
                continue
            word, pronunciation = line.split(maxsplit=1)
            if re.search(r'\(\d+\)$', word):
                continue
            pronunciation = pronunciation.split('#', 1)[0].strip()
            entries[word.lower()] = pronunciation
        return entries

    def lookup_word(self, word):
        return self.entries.get(word.lower())

    def stress_for(self, word, phones):
        pronunciation = self.stress_entries.get(word, '').split()
        if [p.rstrip('012') for p in pronunciation] == phones:
            return [int(p[-1]) if p[-1].isdigit() else -1 for p in pronunciation], 'cmudict'
        return [-1] * len(phones), 'unavailable_in_frozen_lexicon'


@lru_cache(maxsize=1)
def default_dictionary():
    return TextDictionary()


default_decoder = default_dictionary


def approximate_g2p(word):
    """Conservative spelling fallback; output quality is explicitly approximate."""
    word = word.replace("'", '')
    digraphs = {'tion': ['SH', 'AH', 'N'], 'sion': ['ZH', 'AH', 'N'], 'tch': ['CH'],
                'igh': ['AY'], 'dge': ['JH'], 'ch': ['CH'], 'sh': ['SH'], 'th': ['TH'],
                'ph': ['F'], 'ng': ['NG'], 'qu': ['K', 'W'], 'ee': ['IY'], 'ea': ['IY'],
                'oo': ['UW'], 'ai': ['EY'], 'ay': ['EY'], 'oa': ['OW'], 'ow': ['AW'],
                'oi': ['OY'], 'oy': ['OY'], 'ou': ['AW'], 'er': ['ER'], 'ir': ['ER'],
                'ur': ['ER'], 'ar': ['AA', 'R'], 'or': ['AO', 'R']}
    single = {'a': ['AE'], 'b': ['B'], 'c': ['K'], 'd': ['D'], 'e': ['EH'], 'f': ['F'],
              'g': ['G'], 'h': ['HH'], 'i': ['IH'], 'j': ['JH'], 'k': ['K'], 'l': ['L'],
              'm': ['M'], 'n': ['N'], 'o': ['AA'], 'p': ['P'], 'q': ['K'], 'r': ['R'],
              's': ['S'], 't': ['T'], 'u': ['AH'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'],
              'y': ['Y'], 'z': ['Z']}
    phones, position = [], 0
    if word.startswith(('kn', 'gn', 'wr')):
        position = 1
    while position < len(word):
        matched = False
        for length in (4, 3, 2):
            letters = word[position:position + length]
            if len(letters) == length and letters in digraphs:
                phones.extend(digraphs[letters])
                position += length
                matched = True
                break
        if matched:
            continue
        letter = word[position]
        # A final silent e is a spelling approximation; no claim of learned G2P.
        if letter == 'e' and position == len(word) - 1 and len(word) > 3:
            position += 1
            continue
        if position and letter == word[position - 1] and letter not in 'aeiou':
            position += 1
            continue
        if letter in 'cg' and position + 1 < len(word) and word[position + 1] in 'eiy':
            phones.append('S' if letter == 'c' else 'JH')
        elif letter == 'y' and position == len(word) - 1 and len(word) > 1:
            phones.append('IY')
        else:
            phones.extend(single[letter])
        position += 1
    if not phones:
        raise ValueError('Cannot phonemize: ' + word)
    stress, stressed = [], False
    for phone in phones:
        stress.append(1 if phone in _VOWELS and not stressed else 0 if phone in _VOWELS else -1)
        stressed = stressed or phone in _VOWELS
    return phones, stress


def _strength(chars):
    return 1. if any(c in '.!?' for c in chars) else .66 if any(c in ';:' for c in chars) else .33 if ',' in chars else 0.


class LanguageFrontend(abc.ABC):
    """Future languages implement the same text-only, aligned phone interface."""
    language: str

    @abc.abstractmethod
    def encode(self, text: str):
        raise NotImplementedError

    def __call__(self, text: str):
        return self.encode(text)


class EnglishFrontend(LanguageFrontend):
    language = 'en'

    def __init__(self, dictionary=None, oov_policy='rules'):
        if oov_policy not in ('rules', 'error'):
            raise ValueError('oov_policy must be rules or error')
        self.dictionary = dictionary or default_dictionary()
        self.oov_policy = oov_policy

    def encode(self, text):
        normalized = normalize_with_punctuation(text)
        tokens = list(re.finditer(r"[a-z]+(?:'[a-z]+)*", normalized))
        if not tokens:
            raise ValueError('Text contains no words')
        words = [m.group() for m in tokens]
        pronunciations, word_stress, sources, stress_sources, unknown = [], [], [], [], []
        for word in words:
            raw = self.dictionary.lookup_word(word)
            if raw:
                phones = [p.rstrip('012') for p in raw.split()]
                stress, stress_source = self.dictionary.stress_for(word, phones)
                source = 'cmudict_frozen_pocketsphinx'
            elif self.oov_policy == 'rules':
                phones, stress = approximate_g2p(word)
                source, stress_source = 'approximate_spelling_rules', 'approximate_first_vowel'
                unknown.append(word)
            else:
                raise ValueError('Dictionary word OOV: ' + word)
            if any(p not in PHONES for p in phones):
                raise ValueError('Pronunciation outside ARPABET39: ' + word)
            pronunciations.append(phones)
            word_stress.append(stress)
            sources.append(source)
            stress_sources.append(stress_source)
        n = sum(map(len, pronunciations))
        phones, features, stress, boundaries, word_indices = ['<SIL>'], [[0.] * 12], [-1], [0], [-1]
        position = 0
        for wi, wordphones in enumerate(pronunciations):
            before = normalized[tokens[wi - 1].end() if wi else 0:tokens[wi].start()]
            after = normalized[tokens[wi].end():tokens[wi + 1].start() if wi + 1 < len(tokens) else len(normalized)]
            for pi, phone in enumerate(wordphones):
                phones.append(phone)
                features.append([float(pi == 0), float(pi == len(wordphones) - 1), pi / max(1, len(wordphones) - 1),
                                 min(len(wordphones) / 12., 2.), wi / max(1, len(words) - 1), position / max(1, n - 1),
                                 min(n / 200., 2.), min(len(words) / 50., 2.),
                                 _strength(after) if pi == len(wordphones) - 1 else 0.,
                                 _strength(before) if pi == 0 else 0., 0., 0.])
                stress.append(word_stress[wi][pi])
                boundaries.append(int(pi == 0))
                word_indices.append(wi)
                position += 1
        phones.append('<SIL>')
        features.append([0.] * 12)
        stress.append(-1)
        boundaries.append(0)
        word_indices.append(-1)
        features[0][6:8] = features[-1][6:8] = [min(n / 200., 2.), min(len(words) / 50., 2.)]
        features[0][10] = features[-1][11] = 1.
        features[-1][4] = features[-1][5] = 1.
        edge = torch.zeros(len(phones), dtype=torch.bool)
        edge[0] = edge[-1] = True
        return {'phone_ids': torch.tensor([PHONE_TO_ID[p] for p in phones], dtype=torch.long),
                'text_features': torch.tensor(features, dtype=torch.float32), 'edge_sil_mask': edge,
                'phones': phones, 'phonemes': phones, 'stress': stress, 'word_boundary': boundaries,
                'word_indices': word_indices, 'language': self.language, 'language_id': LANGUAGE_IDS[self.language],
                'normalized_text': ' '.join(words), 'punctuated_text': normalized, 'words': words,
                'pronunciation_sources': sources, 'stress_sources': stress_sources, 'oov_words': unknown,
                'oov_backend': 'approximate_spelling_rules' if unknown else 'none',
                'feature_names': FEATURE_NAMES, 'frontend_version': FRONTEND_VERSION}


def get_frontend(language='en', **kwargs):
    if language.lower() in ('en', 'en-us', 'en-gb', 'english'):
        return EnglishFrontend(**kwargs)
    raise NotImplementedError('Language frontend not implemented: ' + language)


def from_text(text, decoder=None, language='en', oov_policy='rules'):
    return get_frontend(language, dictionary=decoder, oov_policy=oov_policy).encode(text)


def from_manifest_row(row, decoder=None):
    def value(field):
        return field.get('value') if isinstance(field, dict) else field
    metadata = value(row.get('metadata', {})) or {}
    text = value(row['text'])
    plain = from_text(text, decoder)
    candidate = row.get('punctuated_text') or metadata.get('publisher_normalized_text')
    result = from_text(candidate, decoder) if candidate else plain
    if result['normalized_text'] != plain['normalized_text'] or result['phones'] != plain['phones']:
        raise ValueError('Publisher text differs from frozen words/phones for ' + str(value(row.get('id'))))
    expected = value(row.get('phonemes'))
    if expected is not None and result['phones'] != list(expected):
        raise ValueError('Text-derived phone order differs from frozen manifest for ' + str(value(row.get('id'))))
    result['text_source'] = 'publisher_normalized_text' if candidate else 'normalized_manifest_text'
    return result


def to_serializable(result):
    return {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in result.items()}


def export_dictionary(texts, path, decoder=None):
    decoder = decoder or default_dictionary()
    words = sorted({word for text in texts for word in from_text(text, decoder)['words']})
    entries = {word: decoder.lookup_word(word) for word in words}
    payload = {'frontend_version': FRONTEND_VERSION, 'phone_to_id': PHONE_TO_ID, 'feature_names': FEATURE_NAMES,
               'pronunciations': entries, 'word_count': len(entries), 'stress_used_as_feature': False,
               'oov_policy': 'explicit_approximate_spelling_rules'}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'path': str(path), 'file_bytes': path.stat().st_size, 'word_count': len(entries), 'general_G2P': True,
            'oov_G2P_quality': 'approximate_spelling_rules'}


build_text_features = from_text
text_features = from_text
features_for_row = from_manifest_row
