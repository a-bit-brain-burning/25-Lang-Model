"""Mandarin training/inference frontend for the frozen numeric shared contract.

Corpus pinyin is authoritative: supplied tones are not changed a second time.
Without corpus labels, pypinyin supplies phrase-disambiguated citation forms and
conservative local tone sandhi is applied inside punctuation/prosody phrases.
This is a text policy, not an acoustic or DSP rule.
"""
from __future__ import annotations

import re
import logging
from functools import lru_cache
from collections.abc import Sequence
import numpy as np

from .universal import LanguageFrontend, UniversalPhone, encode_phones
from .universal_chinese import INITIALS, FINALS, TONE_CONTOURS, split_pinyin
from .phonetic_features import PhoneticSegment, encode_segment, encode_ipa, segment_from_ipa

VERSION = 'chinese_training_frontend_v1'
_HAN = re.compile(r'[\u3400-\u9fff]')
_PUNCTUATION = '，。！？；：、,.!?;:…—“”‘’（）()《》〈〉【】[]「」『』·'
_STRONG = '。！？.!?；;'


@lru_cache(maxsize=1)
def _tokenizer():
    import jieba
    jieba.setLogLevel(logging.ERROR)
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    return tokenizer


def normalize_text(text: str, digit_policy: str = 'reject') -> str:
    """Preserve BZNSYP #1..#4 boundaries; never silently drop unknown input.

    Optional digit spelling is explicit, not full Chinese number normalization.
    Training text with supplied pinyin should use the corpus-normalized wording.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty Chinese text')
    from .chinese_polyphone import strip_reading_hints
    text,_ = strip_reading_hints(text)
    text = text.replace('\u3000', ' ').strip()
    if digit_policy not in ('reject', 'digits'):
        raise ValueError('digit_policy must be reject or digits')
    # Protect prosody marker digits from number spelling.
    if digit_policy == 'digits':
        text = re.sub(r'(?<!#)[0-9]+', lambda m: ''.join('零一二三四五六七八九'[int(c)] for c in m[0]), text)
    clean = re.sub(r'#[1-4]', '', text)
    unknown = [c for c in clean if not _HAN.fullmatch(c) and not c.isspace() and c not in _PUNCTUATION]
    if unknown:
        raise ValueError('Normalize unsupported Chinese input explicitly: ' + ''.join(dict.fromkeys(unknown)))
    if not _HAN.search(clean):
        raise ValueError('No Han characters')
    return text


def parse_prosody_text(text: str):
    """Return one syllable character and boundary levels per Han character."""
    characters, levels = [], []
    i = 0
    while i < len(text):
        c = text[i]
        if _HAN.fullmatch(c):
            characters.append(c)
            levels.append(0)
        elif c == '#':
            if i + 1 >= len(text) or text[i + 1] not in '1234' or not characters:
                raise ValueError('Invalid Chinese prosody marker near ' + text[max(0, i - 4):i + 4])
            levels[-1] = max(levels[-1], int(text[i + 1]))
            i += 1
        elif c in _PUNCTUATION and characters:
            level = 4 if c in '。！？.!?' else 3 if c in '；：;:' else 2 if c in '，、,' else 0
            levels[-1] = max(levels[-1], level)
        elif not c.isspace() and c not in _PUNCTUATION:
            raise ValueError('Unsupported Chinese input character: ' + c)
        i += 1
    return characters, levels


def normalize_pinyin_token(token: str) -> str:
    """Accept corpus ASCII tone suffixes (0/5 neutral), ü/v/u: and erhua."""
    token = str(token).lower().strip().replace('u:', 'v').replace('ü', 'v')
    token = re.sub(r'([0-5])r$', r'r\1', token)
    if not re.fullmatch(r'[a-zv]+[0-5]', token):
        raise ValueError('Expected pinyin syllable with trailing tone 0..5: ' + token)
    if token[-1] == '0':
        token = token[:-1] + '5'
    return token


def _split(token: str):
    """Broad IPA initial/final segmentation, with explicit syllabic nasals."""
    token = normalize_pinyin_token(token)
    base, tone = token[:-1], int(token[-1])
    if base in ('m', 'n', 'ng', 'hm', 'hng'):
        phones = ['m̩'] if base in ('m', 'hm') else ['n̩'] if base == 'n' else ['ŋ̩']
        return (['h'] + phones if base.startswith('h') else phones), tone, '', base
    # Suffix -r is an annotated erhua syllable, not an additional syllable.
    erhua = base.endswith('r') and base != 'er'
    root = base[:-1] if erhua else base
    from pypinyin.contrib.tone_convert import to_initials, to_finals
    initial = to_initials(root.replace('v', 'ü'), strict=True)
    final = to_finals(root.replace('v', 'ü'), strict=True).replace('v', 'ü')
    # Common corpus spelling nue/lue denotes nüe/lüe.
    if root in ('nue', 'lue'):
        initial, final = root[0], 'üe'
    if initial not in INITIALS or final not in FINALS:
        raise ValueError(f'Unsupported Chinese pinyin {token}: initial={initial!r}, final={final!r}')
    canonical = root + str(tone)
    if root in ('nue', 'lue'):
        canonical = root[0] + 've' + str(tone)
    phones, _ = split_pinyin(canonical)
    if erhua:
        # Keep the coda explicitly and encode rhoticity on the final vowel.
        vowel_indices = [i for i, p in enumerate(phones) if p not in ('j', 'w', 'ɥ', 'n', 'ŋ') and i >= bool(initial)]
        if not vowel_indices:
            raise ValueError('Erhua has no vowel nucleus: ' + token)
        phones[vowel_indices[-1]] += '˞'
    return phones, tone, initial, final


def _rime_features(ipas):
    """One native whole-rime unit; no invented component durations.

    The frozen 32-coordinate contract has two vowel trajectory endpoints. For a
    triphthong the start descriptor is a 35/65 blend of glide and nucleus and
    the end descriptor is the offglide. Nasal-coda place/nasality are explicit.
    This broad compound descriptor is not an articulatory time alignment.
    """
    if len(ipas) == 1:
        return encode_ipa(ipas[0])
    coda = ipas[-1] if ipas[-1] in ('n', 'ŋ') else None
    vowel_ipas = ipas[:-1] if coda else ipas
    glide_vowels = {'j': 'i', 'w': 'u', 'ɥ': 'y'}
    segments = [segment_from_ipa(glide_vowels.get(x, x)) for x in vowel_ipas]
    if not all(s.manner == 'vowel' for s in segments):
        raise ValueError('Unsupported whole-rime composition: ' + repr(ipas))
    coords = [np.asarray(s.vowel_start, np.float32) for s in segments]
    start = coords[0]
    if len(coords) >= 3:
        start = .35 * coords[0] + .65 * coords[1]
    return encode_segment(PhoneticSegment(''.join(ipas), 'vowel',
        place='alveolar' if coda == 'n' else 'velar' if coda == 'ŋ' else None,
        voiced=True, nasalized=coda is not None, rhotic=any(s.rhotic for s in segments),
        vowel_start=tuple(float(x) for x in start),
        vowel_end=tuple(float(x) for x in coords[-1]) if len(coords) > 1 else None,
        syllabic=True))


def _whole_units(token):
    ipas, tone, initial, final = _split(token)
    initial_count = int(bool(initial))
    tail = ipas[initial_count:]
    final_label = final.replace('ü', 'v')
    if final == 'i' and initial in ('z', 'c', 's'):
        final_label = 'ii'
    elif final == 'i' and initial in ('zh', 'ch', 'sh', 'r'):
        final_label = 'iii'
    if token[:-1].endswith('r') and token[:-1] != 'er':
        final_label += 'r'
    units = []
    if initial:
        units.append((initial, ipas[0], encode_ipa(ipas[0]), ipas[0]))
    # encode_phones needs a supported representative phone to build context;
    # the actual compound features and IPA replace that representative below.
    units.append((final_label + str(tone), ''.join(tail), _rime_features(tail), tail[0]))
    return units, tone, initial, final, ipas


def _sandhi(characters, tokens, levels):
    """Conservative third-tone/一/不 text rules, not measured surface tones.

    Punctuation and annotated #2..#4 boundaries stop sandhi. Long tone-three
    runs are a simplified left-to-right rule; lexical prosody can differ.
    """
    original = [int(t[-1]) for t in tokens]
    result = list(tokens)
    for i, (char, token) in enumerate(zip(characters, tokens)):
        has_next = i + 1 < len(tokens) and levels[i] < 2
        next_tone = original[i + 1] if has_next else None
        tone = original[i]
        if tone == 3 and next_tone == 3:
            tone = 2
        if char == '不' and token[:-1] == 'bu' and next_tone == 4:
            tone = 2
        if char == '一' and token[:-1] == 'yi':
            # Ordinals and number sequences retain citation tone.
            numerical_neighbor = (i > 0 and characters[i - 1] in '第零一二三四五六七八九十百千万亿') or (has_next and characters[i + 1] in '零一二三四五六七八九十百千万亿')
            if numerical_neighbor:
                tone = 1
            elif has_next:
                tone = 2 if next_tone == 4 else 4 if next_tone in (1, 2, 3) else 1
        result[i] = token[:-1] + str(tone)
    return result


def _fuse_annotated_erhua(characters, tokens, text_levels, corpus_levels):
    """An annotated huar1 may cover 花儿; retain both source characters."""
    if len(characters) == len(tokens):
        return characters, text_levels, corpus_levels
    fused, new_text, new_corpus, position = [], [], [], 0
    for token in tokens:
        if position >= len(characters):
            break
        size = 2 if token[:-1].endswith('r') and token[:-1] != 'er' and position + 1 < len(characters) and characters[position + 1] == '儿' else 1
        fused.append(''.join(characters[position:position + size]))
        new_text.append(max(text_levels[position:position + size]))
        new_corpus.append(max(corpus_levels[position:position + size]))
        position += size
    if position != len(characters) or len(fused) != len(tokens):
        raise ValueError(f'Chinese text/pinyin syllable count mismatch: {len(characters)} Han characters vs {len(tokens)} pinyin tokens (not explained by explicitly annotated erhua)')
    return fused, new_text, new_corpus


class ChineseTrainingFrontend(LanguageFrontend):
    language = 'zh'

    def __init__(self, *, tone_sandhi: bool = True, digit_policy: str = 'reject', use_corpus_prosody: bool = False, erhua_policy: str = 'lexicon'):
        self.tone_sandhi = bool(tone_sandhi)
        self.digit_policy = digit_policy
        self.use_corpus_prosody = bool(use_corpus_prosody)
        if erhua_policy not in ('lexicon','separate'):
            raise ValueError('erhua_policy must be lexicon or separate')
        self.erhua_policy = erhua_policy

    def encode(self, text: str, pinyin: str | Sequence[str] | None = None):
        normalized = normalize_text(text, self.digit_policy)
        characters, corpus_levels = parse_prosody_text(normalized)
        _, text_levels = parse_prosody_text(re.sub(r'#[1-4]', '', normalized))
        levels = corpus_levels if self.use_corpus_prosody else text_levels
        from pypinyin import lazy_pinyin, Style, __version__
        supplied = pinyin is not None
        pronunciation_events=[]
        if supplied:
            tokens = pinyin.split() if isinstance(pinyin, str) else list(pinyin)
            tokens = [normalize_pinyin_token(t) for t in tokens]
        else:
            # Convert complete punctuation-separated phrases, preserving pypinyin's
            # phrase dictionary disambiguation; no hidden double-sandhi.
            tokens = []
            start = 0
            for i, level in enumerate(levels):
                if level >= 2 or i == len(levels) - 1:
                    phrase = ''.join(characters[start:i + 1])
                    def reject(chars):
                        raise ValueError('No pinyin pronunciation for: ' + chars)
                    tokens.extend(lazy_pinyin(phrase, style=Style.TONE3, neutral_tone_with_five=True, tone_sandhi=False, errors=reject))
                    start = i + 1
            tokens = [normalize_pinyin_token(t) for t in tokens]
            from .chinese_polyphone import resolve_citation
            tokens,pronunciation_events=resolve_citation(text,tokens,normalized_text=normalized)
            tokens=[normalize_pinyin_token(t) for t in tokens]
        if len(tokens) != len(characters):
            characters, text_levels, corpus_levels = _fuse_annotated_erhua(characters, tokens, text_levels, corpus_levels)
            levels = corpus_levels if self.use_corpus_prosody else text_levels
        erhua_events=[]
        if not supplied and self.erhua_policy=='lexicon' and all(len(c)==1 for c in characters):
            from .chinese_erhua import fuse_lexical_erhua
            characters,tokens,text_levels,corpus_levels,erhua_events=fuse_lexical_erhua(
                normalized,characters,tokens,text_levels,corpus_levels,_tokenizer())
            levels=corpus_levels if self.use_corpus_prosody else text_levels
        citation = list(tokens)
        if not supplied and self.tone_sandhi:
            tokens = _sandhi(characters, tokens, levels)

        phones = [UniversalPhone('sil')]
        unit_features = [encode_ipa('sil')]
        unit_ipas = ['sil']
        unit_labels = ['sil']
        positions = [[0.] * 12]
        syllables = []
        optional_pause_indices = []
        has_word_labels = self.use_corpus_prosody and bool(re.search(r'#[1-4]', normalized))
        groups, start = [], 0
        if has_word_labels:
            for i, level in enumerate(levels):
                if level >= 1 or i == len(levels) - 1:
                    groups.append((start, i + 1))
                    start = i + 1
        else:
            tokenizer = _tokenizer()
            # Same text-derived word segmentation for training and inference;
            # never feed unavailable gold word/prosody boundaries by default.
            for i, level in enumerate(text_levels):
                if level >= 2 or i == len(levels) - 1:
                    clause = ''.join(characters[start:i + 1])
                    widths = [0]
                    for c in characters[start:i + 1]:
                        widths.append(widths[-1] + len(c))
                    offset, char_end = start, 0
                    for word in tokenizer.lcut(clause, HMM=False):
                        char_end += len(word)
                        # A tokenizer boundary cannot split one annotated
                        # erhua syllable. Merge that lexical boundary explicitly.
                        if char_end in widths:
                            end = start + widths.index(char_end)
                            groups.append((offset, end))
                            offset = end
                    if offset != i + 1:
                        raise ValueError('Word segmentation changed character coverage')
                    start = i + 1
        syllable_to_group = {i: (wi, a, b) for wi, (a, b) in enumerate(groups) for i in range(a, b)}
        for si, (char, token) in enumerate(zip(characters, tokens)):
            units, tone, initial, final, ipas = _whole_units(token)
            first = len(phones)
            previous_level = levels[si - 1] if si else 0
            for pi, (label, ipa, vector, representative) in enumerate(units):
                phones.append(UniversalPhone(representative, tone_contour=TONE_CONTOURS[tone], syllable_start=pi == 0, syllable_end=pi == len(units) - 1))
                unit_features.append(vector)
                unit_ipas.append(ipa)
                unit_labels.append(label)
                row = [0.] * 12
                row[8] = levels[si] / 4. if pi == len(units) - 1 else 0.
                row[9] = previous_level / 4. if pi == 0 else 0.
                positions.append(row)
            syllables.append({'character': char, 'pinyin': token, 'citation_pinyin': citation[si], 'tone': tone,
                              'initial': initial, 'final': final, 'ipa': ipas,
                              'phone_start': first, 'phone_end': len(phones), 'boundary_after_level': levels[si],
                              'corpus_boundary_after_level': corpus_levels[si],
                              'group_index': syllable_to_group[si][0]})
            if si < len(characters) - 1:
                optional_pause_indices.append(len(phones))
                phones.append(UniversalPhone('sil'))
                unit_features.append(encode_ipa('sil'))
                unit_ipas.append('sil')
                unit_labels.append('sp')
                row = [0.] * 12
                row[8] = row[9] = levels[si] / 4.
                positions.append(row)
        phones.append(UniversalPhone('sil'))
        unit_features.append(encode_ipa('sil'))
        unit_ipas.append('sil')
        unit_labels.append('sil')
        positions.append([0.] * 12)
        n = len(phones) - 2
        for i in range(1, len(phones) - 1):
            positions[i][5] = (i - 1) / max(n - 1, 1)
            positions[i][6] = min(n / 200., 2.)
            positions[i][7] = min(len(groups) / 40., 2.)
        for wi, (a, b) in enumerate(groups):
            first, last = syllables[a]['phone_start'], syllables[b - 1]['phone_end']
            size = last - first
            for offset, i in enumerate(range(first, last)):
                positions[i][0:5] = [float(offset == 0), float(offset == size - 1), offset / max(size - 1, 1), min(size / 20., 2.), wi / max(len(groups) - 1, 1)]
        positions[0][10] = positions[-1][11] = 1.
        positions[0][6] = positions[-1][6] = min(n / 200., 2.)
        ranges = [[s['phone_start'], s['phone_end']] for s in syllables]
        result = encode_phones(phones, np.asarray(positions, np.float32), metadata={
            'language': 'zh', 'frontend_version': VERSION if supplied or self.erhua_policy=='separate' else 'chinese_training_frontend_v2_erhua', 'text': text, 'normalized_text': normalized,
            'pypinyin_version': __version__, 'syllables': syllables,
            'pinyin_source': 'corpus_annotation' if supplied else 'pypinyin_phrase_dictionary',
            'tone_sandhi_applied': not supplied and self.tone_sandhi,
            'erhua_policy': 'explicit_annotation' if supplied else self.erhua_policy,
            'erhua_fusions': erhua_events,
            'pronunciation_events': pronunciation_events,
            'erhua_feature_contract': 'existing_whole_rime_rhotic_bit_no_extra_er_syllable',
            'tone_contour_kind': 'nominal_template_not_measured_F0',
            'word_segmentation': 'corpus_prosody_groups' if has_word_labels else 'jieba_0.42.1_HMM_false',
            'use_corpus_prosody': self.use_corpus_prosody,
            'gold_prosody_is_input': has_word_labels,
            'tone_policy': 'supplied_tones_authoritative_no_second_sandhi' if supplied else 'conservative_phrase_local_sandhi',
            'digit_policy': self.digit_policy, 'surface_tone_accuracy_guaranteed': False})
        result['phonetic_features'] = np.stack(unit_features)
        result['ipa'] = unit_ipas
        result['unit_labels'] = unit_labels
        # Reuse the model's existing optional-duration gate. The name is a
        # historical API name; first_edge/last_edge context stays edge-only.
        result['edge_sil_mask'][optional_pause_indices] = True
        result['optional_pause_indices'] = np.asarray(optional_pause_indices, np.int32)
        result['metadata']['segmentation_unit'] = 'native_initial_plus_whole_rime'
        result['metadata']['compound_rime_encoding'] = 'two_endpoint_trajectory_nasal_coda_place; triphthong_start=.35_glide+.65_nucleus'
        result['metadata']['optional_pause_policy'] = 'one_zero_or_positive_duration_sil_between_all_syllables'
        result['metadata']['edge_sil_mask_semantics'] = 'optional_duration_mask_includes_syllable_boundary_sil; first_last_context_remains_edge_only'
        result['syllable_phone_ranges'] = np.asarray(ranges, np.int32)
        result['pinyin'] = tokens
        return result

    def from_baker_annotation(self, annotation, pinyin, interval_labels):
        """Map native labelled initial/rime intervals, preserving all pauses.

        Returns native_interval_groups[N], one list of original indices per
        frontend unit. Empty lists are genuine zero-duration pause candidates.
        Whole-rime durations are never divided into artificial subunits.
        """
        result = self.encode(annotation, pinyin)
        expected_pairs = [(i, label) for i, label in enumerate(result['unit_labels']) if label not in ('sil', 'sp')]
        expected = [label for _, label in expected_pairs]
        native = [str(x).strip().lower() for x in interval_labels]
        if not native:
            raise ValueError('Empty native interval sequence')
        silence = lambda x: x in ('', 'sil', 'sp', 'sp1', 'sp2', 'sp3', 'sp4')
        actual = [x for x in native if not silence(x)]
        if actual != expected:
            mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
            raise ValueError(f'Native initial/rime vs pinyin mismatch at unit {mismatch}: native={actual[mismatch:mismatch+3]}, expected={expected[mismatch:mismatch+3]} (length {len(actual)}/{len(expected)})')
        groups = [[] for _ in result['unit_labels']]
        nonpause_native = [i for i, label in enumerate(native) if not silence(label)]
        native_to_frontend = {ni: fi for ni, (fi, _) in zip(nonpause_native, expected_pairs)}
        for i, label in enumerate(native):
            if not silence(label):
                groups[native_to_frontend[i]].append(i)
            elif i < nonpause_native[0]:
                groups[0].append(i)
            elif i > nonpause_native[-1]:
                groups[-1].append(i)
            else:
                before = max(x for x in nonpause_native if x < i)
                after = min(x for x in nonpause_native if x > i)
                a, b = native_to_frontend[before], native_to_frontend[after]
                candidates = [j for j in range(a + 1, b) if result['unit_labels'][j] == 'sp']
                if len(candidates) != 1:
                    raise ValueError(f'Native pause {label!r} at interval {i} is inside an initial/rime or has no unique syllable-boundary candidate')
                groups[candidates[0]].append(i)
        if sorted(x for group in groups for x in group) != list(range(len(native))):
            raise ValueError('Native interval mapping coverage failed')
        result['native_interval_groups'] = groups
        result['native_interval_indices'] = np.asarray([g[0] if g else -1 for g in groups], np.int32)
        result['metadata']['alignment_source'] = 'native_baker_initial_whole_rime_intervals'
        result['metadata']['subphoneme_duration_fabricated'] = False
        result['metadata']['internal_pause_units'] = sum(silence(x) for x in native[1:-1])
        result['metadata']['native_labels'] = native
        result['metadata']['zero_duration_pause_candidates'] = sum(not g for g in groups)
        return result

