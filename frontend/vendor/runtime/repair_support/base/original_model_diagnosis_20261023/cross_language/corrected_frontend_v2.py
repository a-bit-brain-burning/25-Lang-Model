"""Text-only experimental eSpeak BR frontend; frozen original word planner route.

Unlike the earlier identity-only patch, this permits phone-count changes.
This is an automatic pronunciation candidate, not manually verified gold.
"""
import copy
import ctypes
import hashlib
import json
import sys
import unicodedata

import numpy as np


class CandidateFrontend:
    def __init__(self, run):
        from project.src.frontend.romance import RomanceFrontend, encode_ipa
        from espeakng_loader import get_library_path, get_data_path
        self.original = run._system('pt-PT')
        de = run._system('de-DE')
        de[3].encode('de', 'Guten Tag.', de[2])
        self.module = sys.modules['_tts25_frozen_de.frontend.eu_four']
        self.backend = self.module.EspeakBackend(library=get_library_path(), data=get_data_path(), required_version='1.52.0')
        self.base = RomanceFrontend('pt', self.original[2]['lexicon_path'])
        self.encode_ipa = encode_ipa
        self.signature = hashlib.sha256((self.original[2]['frontend_signature'] +
            hashlib.sha256(open(__file__, 'rb').read()).hexdigest() + self.backend.version).encode()).hexdigest()
        self.cache = {}

    def phonemize(self, text):
        with self.module._LOCK:
            self.backend._select_voice('pt-br')
            data = ctypes.create_string_buffer(text.encode('utf-8'))
            pointer = ctypes.c_void_p(ctypes.addressof(data))
            chunks = []
            for _ in range(1000):
                if not pointer.value:
                    break
                previous = pointer.value
                value = self.backend.lib.espeak_TextToPhonemes(ctypes.byref(pointer), 1, 2 | (ord('_') << 8))
                chunks.append(value.decode('utf-8') if value else '')
                if pointer.value == previous:
                    raise ValueError('eSpeak cursor did not advance')
        raw = ' '.join(chunks).strip()
        if '(' in raw or ')' in raw:
            raise ValueError('eSpeak foreign language switch')
        return raw

    def split_supported(self, token):
        token = unicodedata.normalize('NFC', token.replace('g', 'ɡ'))
        try:
            self.encode_ipa(token)
            return [token]
        except ValueError:
            pass
        # Preserve the entire IPA string; split only at supported segment boundaries.
        for cut in range(len(token) - 1, 0, -1):
            first = token[:cut]
            try:
                self.encode_ipa(first)
                return [first] + self.split_supported(token[cut:])
            except ValueError:
                continue
        raise ValueError('Unsupported G2P IPA token: ' + token)

    def parse_word(self, chunk, word):
        phones, stresses = [], []
        for token in chunk.split('_'):
            if not token:
                continue
            strength = 1 if 'ˈ' in token else 2 if 'ˌ' in token else None
            token = token.replace('ˈ', '').replace('ˌ', '')
            segments = self.split_supported(token)
            for phone in segments:
                nucleus = self.encode_ipa(phone)[31] > .5
                phones.append(phone)
                stresses.append((strength or 0) if nucleus else -1)
                if nucleus:
                    strength = None
        # eSpeak BR sometimes emits palatal lh/nh as l+j or n+j.
        for first, replacement, spelling in (('l', 'ʎ', 'lh'), ('n', 'ɲ', 'nh')):
            if spelling in word:
                for i in range(len(phones) - 2, -1, -1):
                    if phones[i:i + 2] == [first, 'j']:
                        phones[i:i + 2] = [replacement]
                        stresses[i:i + 2] = [-1]
        # eSpeak represents some reduced final-o vowels as a glide, but these
        # need a syllabic acoustic segment. The mapping is explicit and text-only.
        ending = -1 if word.endswith('o') else -2 if word.endswith('os') and len(phones) >= 2 else None
        if ending is not None and phones[ending] == 'w':
            phones[ending], stresses[ending] = 'u', 0
        if not phones:
            raise ValueError('Empty G2P word')
        return tuple(zip(phones, stresses))

    def encode(self, language, text, metadata=None):
        if language != 'pt':
            raise ValueError('Portuguese candidate only')
        if text in self.cache:
            return copy.deepcopy(self.cache[text])
        parent = self.original[3].encode('pt', text, self.original[2])
        raw = self.phonemize(parent['normalized_text'])
        chunks = raw.split()
        words = parent['words']
        if len(chunks) != len(words):
            raise ValueError('G2P word topology differs: ' + str((len(chunks), len(words))))
        instance = copy.copy(self.base)
        pronunciations = [self.parse_word(chunk, word) for chunk, word in zip(chunks, words)]
        iterator = iter(zip(words, pronunciations))
        def pronounce_at_position(word):
            expected, pronunciation = next(iterator)
            if word != expected:
                raise ValueError('Positional pronunciation word mismatch')
            return pronunciation
        instance.pronounce = pronounce_at_position
        front = instance.encode(text)
        front['context_features'][front['stress'] == 0, 12] = 0.
        assert front['words'] == parent['words']
        # Keep old per-word planner inputs, adapting only its unit ownership to
        # the new phone count. This prevents changing the frozen planner factor.
        front.update(planner_numeric=parent['planner_numeric'].copy(),
                     planner_boundary=parent['planner_boundary'].copy(),
                     planner_owner=front['plan_owner'].copy())
        front['metadata'] = {**front['metadata'], 'frontend_signature': self.signature,
            'parent_frontend_signature': self.original[2]['frontend_signature'],
            'profile': 'experimental_full_sequence_espeak_br_1_52_positional_v2',
            'espeak_raw': raw, 'automatic_not_gold': True,
            'reference_audio_or_duration_used': False,
            'planner_route': 'unchanged original word-plan inputs, new acoustic unit owners'}
        self.cache[text] = copy.deepcopy(front)
        return front


def ctc_tokens(phones, vocabulary):
    """Explicit alignment-only mappings; no unreported inventory fallback."""
    vocabulary = {unicodedata.normalize('NFC', k): v for k, v in vocabulary.items()}
    result, owner, mappings = [], [], []
    for i, value in enumerate(phones):
        if value == 'sil':
            continue
        raw = unicodedata.normalize('NFC', str(value).replace('g', 'ɡ'))
        mapped = [raw]
        if raw not in vocabulary:
            if raw.endswith('ʷ') and raw[:-1] in vocabulary and 'w' in vocabulary:
                mapped = [raw[:-1], 'w']
            elif raw.endswith('ː') and raw[:-1] in vocabulary:
                mapped = [raw[:-1]]
            elif raw in ('w̃', 'j̃', 'ʊ̃') and raw[:-1] in vocabulary:
                # Explicit CTC-inventory approximation only: the recognizer
                # lacks these nasal glides. Acoustic inputs retain nasality.
                mapped = [raw[:-1]]
            else:
                raise ValueError('Unsupported CTC unit: ' + raw)
            mappings.append({'unit': i, 'phone': raw, 'ctc_phones': mapped,
                             'scope': 'alignment emission inventory only; acoustic features unchanged'})
        for phone in mapped:
            result.append(vocabulary[phone]); owner.append(i)
    return result, owner, mappings
