"""Text-only Latvian/Lithuanian frontends for independent tiny TTS students.

Language rules stay here; the network and DSP consume numbers.  A pinned eSpeak
NG supplies a pronunciation candidate, not a certificate of lexical stress.
Unwritten lexical pitch accents remain unknown unless an explicitly supplied,
SHA-bound text lexicon provides them. See the experiment docs/FRONTEND.md.
"""
from __future__ import annotations

from collections import Counter
import ctypes
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from .eu_four import (EspeakBackend, FrontendError, _Phone, _canonical_ipa,
                      _LOCK, _MARKS, _strength, _syllables, _tokenize, encode_phone,
                      mfa_dictionary_entries, mfa_transcript, to_serializable)
from .phonetic_features import FEATURE_NAMES
from .universal import CONTEXT_NAMES as BASE_CONTEXT_NAMES


FRONTEND_VERSION = "eu_baltic_frontend_v1"
FEATURE_ADAPTER_VERSION = "baltic_articulatory32_v1"
CONTRACT_VERSION = "baltic_frontend_32_34_v1"
LANGUAGES = {"lv-LV": "lv", "lt-LT": "lt"}
_ALIASES = {**{k.lower(): k for k in LANGUAGES}, **{v: k for k, v in LANGUAGES.items()}}
EXTRA_CONTEXT_NAMES = (
    "syllable_position_in_word", "syllable_count_scale", "phone_position_in_syllable",
    "heavy_syllable_candidate", "syllable_nucleus_long", "syllable_nucleus_diphthong",
    "syllables_before_primary_stress", "syllables_after_primary_stress",
    "stress_source_reliability", "word_final_syllable", "phrase_final_syllable",
    "question_clause",
)
CONTEXT_NAMES = tuple(BASE_CONTEXT_NAMES) + EXTRA_CONTEXT_NAMES
CONTEXT_DIM = len(CONTEXT_NAMES)
_ALPHABET = {
    "lv-LV": set("aābcčdeēfgģhiījkķlļmnņoprsštuūvzžqwxy"),
    "lt-LT": set("aąbcčdeęėfghiįyjklmnoprsštuųūvzžqwx"),
}
_FRONT_VOWELS = set("iɪeɛæ")
_SONORANTS = {"m", "n", "ŋ", "ɲ", "l", "ɫ", "ʎ", "r", "ɾ"}
_PITCH_CODES = {"lv-LV": {"level": .25, "falling": .5, "broken": .75},
                "lt-LT": {"acute": .25, "circumflex": .75}}
_LT_NAMES = {
    "@": "ə", "a": "ʌ", "A": "ʌ", "a:": "ɐː", "e": "ɛ", "E": "ɛ",
    "e:": "æː", "ee": "eː", "eA": "æː", "i": "ɪ", "I": "ɪ", "i:": "iː",
    "o": "ɔ", "O": "ɔ", "o:": "oː", "u": "ʊ", "U": "ʊ", "u:": "uː",
    "ai": "ʌɪ", "au": "ɒʊ", "ei": "ɛɪ", "ie": "iɛ", "uo": "uɔ",
    "oi": "ɔɪ", "ui": "ʊɪ", "aU": "ɒʊ", "eU": "ɛʊ", "oU": "ɔʊ",
    "aI": "ʌɪ", "eI": "ɛɪ", "p": "p", "b": "b", "t": "t", "d": "d",
    "k": "k", "g": "ɡ", "f": "f", "v": "v", "s": "s", "z": "z",
    "S": "ʃ", "Z": "ʒ", "h": "h", "x": "x", "j": "j", "m": "m",
    "n": "n", "N": "ŋ", "r": "r", "l": "ɫ", "l;": "lʲ",
    "ts": "ts", "dz": "dz", "tS": "tʃ", "dZ": "dʒ",
}


@lru_cache(maxsize=65536)
def _lithuanian_phoneme_names(backend, word):
    """Use the named-phoneme API before its lossy IPA rendering.

    Pinned eSpeak maps both z; and Z; to IPA ʑ, and maps soft s; to ʂ.
    The public API's native names preserve those distinctions. This deliberately
    shares the backend's process lock and exact voice/fallback checks.
    """
    if backend.library:
        with _LOCK:
            backend._select_voice("lt")
            data = ctypes.create_string_buffer(word.encode("utf-8"))
            pointer = ctypes.c_void_p(ctypes.addressof(data))
            clauses = []
            for _ in range(1000):
                if not pointer.value:
                    break
                previous = pointer.value
                raw = backend.lib.espeak_TextToPhonemes(ctypes.byref(pointer), 1, ord("_") << 8)
                clauses.append(raw.decode("utf-8") if raw else "")
                if pointer.value == previous:
                    raise FrontendError("eSpeak native-phoneme cursor did not advance")
            else:
                raise FrontendError("eSpeak native-phoneme clause limit exceeded")
            result = " ".join(clauses).strip()
    else:
        # phonemize() is called first and verifies the CLI's exact lt voice.
        result = backend._run(["-q", "-v", "lt", "-x", "--sep=_", "--stdin"], word + "\n").strip()
    if not result or re.search(r"\([^)]*\)", result):
        raise FrontendError(f"Empty or switched-language Lithuanian phoneme names: {result!r}")
    return result


def _parse_lithuanian_names(raw, word):
    pieces = [t for t in re.split(r"[_\s]+", raw) if t]
    phones, audit, pending = [], [], 0.
    for index, value in enumerate(pieces):
        token = value
        if "'" in token or "," in token:
            pending = 1. if "'" in token else .5
            token = token.replace("'", "").replace(",", "")
        if not token:
            continue
        if token == "@-" and index + 1 < len(pieces) and pieces[index + 1].lstrip("',") in ("r", "*"):
            audit.append({"rule": "lt_native_linking_schwa_null_before_r", "from": value, "to": None})
            continue
        if token == ";":
            if not phones or encode_phone(phones[-1].ipa)[31]:
                raise FrontendError(f"Unattached lt softness token in {word!r}/{raw!r}")
            before = phones[-1].ipa
            target = before.replace("ɫ", "l")
            target = target if "ʲ" in target else target + "ʲ"
            phones[-1] = replace(phones[-1], ipa=target)
            audit.append({"rule": "lt_native_softness_marker", "from": before, "to": target})
            continue
        if token in _LT_NAMES:
            ipa = _LT_NAMES[token]
        elif token.endswith(";") and token[:-1] in _LT_NAMES:
            base = _LT_NAMES[token[:-1]]
            if encode_phone(base)[31]:
                raise FrontendError(f"Softness modifier on vowel in {raw!r}")
            ipa = base.replace("ɫ", "l") + "ʲ"
        else:
            raise FrontendError(f"Unmapped lt native phoneme {token!r} in {word!r}; raw={raw!r}")
        x = encode_phone(ipa)
        phones.append(_Phone(ipa, pending if x[31] else 0., False))
        if x[31]:
            pending = 0.
    if pending or not phones:
        raise FrontendError(f"No phones or unattached native stress in {word!r}/{raw!r}")
    return phones, audit


def _language(language):
    try:
        return _ALIASES[language.lower()]
    except (KeyError, AttributeError) as exc:
        raise FrontendError(f"Unsupported Baltic language {language!r}") from exc


def _core(phone):
    return phone.translate(str.maketrans("", "", _MARKS))


def _stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _orthographic_nuclei(word, language):
    """Only transparent grapheme nuclei; no spelling-derived lexical accent."""
    vowels = set("aāeēiīouū") if language == "lv-LV" else set("aąeęėiįyouųū")
    diphthongs = {"ai", "au", "ei", "ie", "ui", "iu", "oi", "eu", "ou"}
    if language == "lt-LT":
        diphthongs = {"ai", "au", "ei", "ie", "ui", "uo", "eu", "oi", "ou"}
    output, i = [], 0
    while i < len(word):
        if word[i] not in vowels:
            i += 1
            continue
        # Lithuanian i before a back vowel marks consonant softness, except at
        # a vowel/word boundary. It does not create a second nucleus.
        softness = False
        if (language == "lt-LT" and word[i] == "i" and i and word[i - 1] not in vowels
                and i + 1 < len(word) and word[i + 1] in "aąouųū"):
            i += 1
            softness = True
        pair = word[i:i + 2]
        width = 2 if pair in diphthongs else 1
        group = word[i:i + width]
        if softness and group[0] in "aą":
            group = {"a": "e", "ą": "ę"}[group[0]] + group[1:]
        output.append(group)
        i += width
    return output


def _merge_split_diphthongs(phones, word, language, audit):
    """Join only a monotonic, unambiguous ordinary-spelling nucleus match.

    eSpeak lt sometimes emits ei as e_I, creating a spurious extra syllable.
    Matching uses the entire word's nucleus sequence, not character indexes in
    an IPA string. Unmatched words retain the backend sequence and a warning.
    """
    if language != "lt-LT":
        return phones, None
    expected = _orthographic_nuclei(word, language)
    indices = [i for i, p in enumerate(phones) if encode_phone(p.ipa)[31]]
    families = {"a": "aɑɐʌɒ", "ą": "aɑɐʌɒ", "e": "eɛæ", "ę": "eɛæ",
                "ė": "eɛæ", "i": "iɪ", "į": "iɪ", "y": "iɪ", "o": "oɔ",
                "u": "uʊ", "ų": "uʊ", "ū": "uʊ"}
    paths = []

    def visit(gi, vi, groups):
        if len(paths) > 1:
            return
        if gi == len(expected):
            if vi == len(indices):
                paths.append(groups)
            return
        if vi >= len(indices):
            return
        group = expected[gi]
        token = _core(phones[indices[vi]].ipa)
        if len(group) == 1:
            if len(token) == 1 and token in families[group]:
                visit(gi + 1, vi + 1, groups + [(indices[vi],)])
        else:
            left, right = families[group[0]], families[group[1]]
            if len(token) == 2 and token[0] in left and token[1] in right:
                visit(gi + 1, vi + 1, groups + [(indices[vi],)])
            if (vi + 1 < len(indices) and indices[vi + 1] == indices[vi] + 1
                    and len(token) == 1 and token in left
                    and _core(phones[indices[vi + 1]].ipa) in right):
                visit(gi + 1, vi + 2, groups + [(indices[vi], indices[vi + 1])])

    visit(0, 0, [])
    if len(paths) != 1:
        return phones, "orthographic_nucleus_alignment_unresolved"
    joins = {pair[0]: pair for pair in paths[0] if len(pair) == 2}
    merged, i = [], 0
    while i < len(phones):
        if i in joins:
            a, b = phones[i:i + 2]
            # A single diphthong remains one acoustic unit with moving vowel
            # coordinates; this does not impose a fixed temporal F0 contour.
            ipa = _core(a.ipa) + _core(b.ipa)
            encode_phone(ipa)
            merged.append(_Phone(ipa, max(a.stress or 0., b.stress or 0.), False))
            audit.append({"rule": "lt_spelling_aligned_split_diphthong", "from": [a.ipa, b.ipa], "to": ipa})
            i += 2
        else:
            merged.append(phones[i]); i += 1
    return merged, None


def _parse_backend(raw, language, word, native_names=None):
    if language == "lt-LT" and native_names is not None:
        phones, audit = _parse_lithuanian_names(native_names, word)
        raw_to_parse = ""
    else:
        phones, audit = [], []
        raw_to_parse = raw
    pending = 0.
    for raw_token in re.split(r"[_\s]+", raw_to_parse):
        if not raw_token:
            continue
        token = raw_token
        if "ˈ" in token or "ˌ" in token:
            pending = 1. if "ˈ" in token else .5
            token = token.replace("ˈ", "").replace("ˌ", "")
        if not token:
            continue
        token = _canonical_ipa(token)
        # Duplicate modifiers are an eSpeak serialization artifact. Preserve one
        # copy, never collapse different modifiers or unsupported base phones.
        for mark in "ːʲ":
            token = re.sub(re.escape(mark) + r"{2,}", mark, token)
        if language == "lv-LV":
            # These are eSpeak engine-duration allophones explicitly documented
            # in ph_latvian. They are not additional Latvian vowel phonemes.
            token = {"a`": "a", "o`": "o", "z-": "z"}.get(token, token)
        if language == "lv-LV" and token == "ʐ" and "dž" in word and phones and phones[-1].ipa == "d":
            # ph_latvian Z` is the second half of the orthographic dž affricate.
            phones[-1] = replace(phones[-1], ipa="dʒ")
            audit.append({"rule": "lv_native_dzh_affricate", "from": ["d", "ʐ"], "to": "dʒ"})
            continue
        if language == "lt-LT":
            # These names are explicit in pinned ph_lithuanian / lt_rules:
            # l is erroneously declared a syllabic vowel; l; is declared retroflex
            # but denotes the soft lateral. ee / eA are long ė / ę vowels.
            repairs = {"l̩": "ɫ", "ɭ": "lʲ", "ɭʲ": "lʲ", "ee": "eː", "eA": "æː"}
            token = repairs.get(token, token)
            if token == "ʲ":
                if not phones or encode_phone(phones[-1].ipa)[31]:
                    raise FrontendError(f"Unattached Lithuanian palatalization in {raw!r}")
                before = phones[-1].ipa
                phones[-1] = replace(phones[-1], ipa=before if "ʲ" in before else before + "ʲ")
                audit.append({"rule": "lt_linking_palatal_marker", "from": raw_token, "to": phones[-1].ipa})
                continue
        if token != _canonical_ipa(raw_token.replace("ˈ", "").replace("ˌ", "")):
            audit.append({"rule": "pinned_backend_symbol_repair", "from": raw_token, "to": token})
        try:
            x = encode_phone(token)
        except FrontendError as exc:
            raise FrontendError(f"{language}/{word!r}: {exc}; raw={raw!r}") from exc
        phones.append(_Phone(token, pending if x[31] else 0., False))
        if x[31]:
            pending = 0.
    if pending or not phones:
        raise FrontendError(f"Empty phones or unattached stress in {language}/{word!r}: {raw!r}")
    if language == "lv-LV":
        merged = []
        for p in phones:
            if merged and p.ipa == merged[-1].ipa and not encode_phone(p.ipa)[31]:
                merged[-1] = replace(p, ipa=p.ipa + "ː")
                audit.append({"rule": "lv_adjacent_identical_backend_consonants", "from": [p.ipa, p.ipa], "to": p.ipa + "ː"})
            else:
                merged.append(p)
        phones = merged
    else:
        # Productive secondary palatalization before front vowels and palatalized
        # consonants. Velars in clusters are excluded (Bakšienė et al. 2023).
        for i in range(len(phones) - 2, -1, -1):
            p, following = phones[i], phones[i + 1]
            x, nxt = encode_phone(p.ipa), encode_phone(following.ipa)
            if x[31] or _core(p.ipa) == "j":
                continue
            direct = bool(nxt[31] and _core(following.ipa)[0] in _FRONT_VOWELS)
            cluster = bool(not nxt[31] and ("ʲ" in following.ipa or _core(following.ipa) == "j")
                           and _core(p.ipa) not in ("k", "ɡ"))
            if (direct or cluster) and "ʲ" not in p.ipa:
                target = p.ipa.replace("ɫ", "l") + "ʲ"
                encode_phone(target)
                phones[i] = replace(p, ipa=target)
                audit.append({"rule": "lt_productive_secondary_palatalization", "from": p.ipa, "to": target})
        phones, alignment_warning = _merge_split_diphthongs(phones, word, language, audit)
        if alignment_warning:
            audit.append({"warning": alignment_warning})
    if not any(encode_phone(p.ipa)[31] for p in phones):
        raise FrontendError(f"No vowel nucleus in {language}/{word!r}; unsupported abbreviation or backend failure")
    return phones, audit


def _parse_override(entry, language):
    entry = {"phones": entry} if isinstance(entry, list) else dict(entry)
    if not isinstance(entry.get("phones"), list) or not entry["phones"]:
        raise FrontendError("A pronunciation override requires a nonempty phones list")
    phones, pending = [], 0.
    for value in entry["phones"]:
        if not isinstance(value, str):
            raise FrontendError("Override phones must be strings")
        if "ˈ" in value or "ˌ" in value:
            pending = 1. if "ˈ" in value else .5
        ipa = _canonical_ipa(value.replace("ˈ", "").replace("ˌ", ""))
        x = encode_phone(ipa)
        phones.append(_Phone(ipa, pending if x[31] else 0., bool(entry.get("stress_known", True))))
        if x[31]:
            pending = 0.
    if pending or not any(encode_phone(p.ipa)[31] for p in phones):
        raise FrontendError("Override has no nucleus or unattached stress")
    if not any(p.stress for p in phones):
        phones = [replace(p, stress_known=False) for p in phones]
    accent = entry.get("pitch_accent")
    if accent is not None:
        if (not isinstance(accent, dict) or accent.get("kind") not in _PITCH_CODES[language]
                or not isinstance(accent.get("syllable"), int)):
            raise FrontendError(f"Pitch accent must use language-specific {_PITCH_CODES[language]} and zero-based syllable")
        count = sum(bool(encode_phone(p.ipa)[31]) for p in phones)
        if not 0 <= accent["syllable"] < count:
            raise FrontendError("Pitch-accent syllable outside pronunciation")
    return phones, accent


class BalticFrontend:
    def __init__(self, language, *, espeak_library=None, espeak_data=None, espeak_binary=None,
                 required_espeak_version="1.52.0", pause_policy="punctuation",
                 pronunciation_overrides=None, lexicon_path=None, lexicon_sha256=None):
        self.language = _language(language)
        if pause_policy not in ("punctuation", "word", "none"):
            raise FrontendError("pause_policy must be punctuation, word, or none")
        self.pause_policy = pause_policy
        self.overrides = json.loads(json.dumps(pronunciation_overrides or {}, ensure_ascii=False))
        self.lexicon, self.lexicon_info = {}, None
        if bool(lexicon_path) != bool(lexicon_sha256):
            raise FrontendError("External text lexicons require both path and expected SHA256")
        if lexicon_path:
            raw = Path(lexicon_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != lexicon_sha256:
                raise FrontendError("External text lexicon SHA256 mismatch")
            document = json.loads(raw)
            if (document.get("schema") != "baltic_pronunciation_lexicon_v1"
                    or document.get("language") != self.language or not document.get("provenance")
                    or not isinstance(document.get("entries"), dict)):
                raise FrontendError("Invalid Baltic lexicon schema, language or provenance")
            self.lexicon = document["entries"]
            self.lexicon_info = {"sha256": lexicon_sha256, "provenance": document["provenance"],
                                 "entries": len(self.lexicon)}
        for collection in (self.lexicon, self.overrides):
            for word, entry in collection.items():
                if not isinstance(word, str) or word != unicodedata.normalize("NFC", word).lower():
                    raise FrontendError("Lexicon/override keys must be NFC lowercase words")
                _parse_override(entry, self.language)
        self.backend = EspeakBackend(library=espeak_library, data=espeak_data,
                                     binary=espeak_binary, required_version=required_espeak_version)
        self.backend_info = self.backend.provenance(LANGUAGES[self.language])
        self._config = {"language": self.language, "required_espeak_version": required_espeak_version,
                        "pause_policy": pause_policy, "pronunciation_overrides": self.overrides,
                        "lexicon_path": str(Path(lexicon_path).resolve()) if lexicon_path else None,
                        "lexicon_sha256": lexicon_sha256}

    def get_config(self):
        return {**self._config, "espeak_library": self.backend.library,
                "espeak_binary": self.backend.binary, "espeak_data": self.backend.data}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def identity(self):
        config = {k: v for k, v in self._config.items() if k != "lexicon_path"}
        config.update(frontend_version=FRONTEND_VERSION, feature_adapter_version=FEATURE_ADAPTER_VERSION,
                      contract_version=CONTRACT_VERSION, context_feature_names=CONTEXT_NAMES,
                      backend_version=self.backend_info["version"],
                      backend_data_sha=self.backend_info.get("data_identity_sha256"))
        return {"semantic_identity_sha256": _stable_hash(config), "semantic_config": config,
                "backend": self.backend_info, "lexicon": self.lexicon_info,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def encode(self, text, *, utterance_id=None):
        normalized, tokens = _tokenize(text, self.language)
        words = []
        for wi, token in enumerate(tokens):
            word = token.group()
            invalid = set(word) - _ALPHABET[self.language] - set("'-")
            if invalid:
                raise FrontendError(f"Unsupported {self.language} spelling {word!r}: {sorted(invalid)!r}; normalize foreign names or use an explicit text lexicon")
            before = normalized[tokens[wi - 1].end() if wi else 0:token.start()]
            after = normalized[token.end():tokens[wi + 1].start() if wi + 1 < len(tokens) else len(normalized)]
            accent = None
            stress_source = "backend_candidate_unknown"
            if word in self.overrides or word in self.lexicon:
                explicit = word in self.overrides
                entry = self.overrides[word] if explicit else self.lexicon[word]
                phones, accent = _parse_override(entry, self.language)
                raw, native_names, repairs = None, None, []
                source = "explicit_text_pronunciation_override" if explicit else "sha_bound_text_lexicon"
                reliability = 1. if any(p.stress_known for p in phones) else 0.
                stress_source = source if reliability else "lexicon_unmarked_unknown"
                if isinstance(entry, dict) and entry.get("stress_fallback") == "backend_same_nucleus_count":
                    raw = self.backend.phonemize(word, LANGUAGES[self.language])
                    native_names = _lithuanian_phoneme_names(self.backend, word) if self.language == "lt-LT" else None
                    candidate, _ = _parse_backend(raw, self.language, word, native_names=native_names)
                    left = [i for i, p in enumerate(phones) if encode_phone(p.ipa)[31]]
                    right = [p for p in candidate if encode_phone(p.ipa)[31]]
                    if len(left) == len(right):
                        for index, p in zip(left, right):
                            phones[index] = replace(phones[index], stress=p.stress, stress_known=False)
                        reliability = .25
                        stress_source = "backend_same_nucleus_count_unknown"
                        repairs.append({"rule": "lexicon_unmarked_stress_backend_candidate", "nuclei": len(left)})
                    else:
                        repairs.append({"warning": "lexicon_backend_nucleus_count_mismatch_no_stress_guessed"})
            else:
                raw = self.backend.phonemize(word, LANGUAGES[self.language])
                native_names = _lithuanian_phoneme_names(self.backend, word) if self.language == "lt-LT" else None
                phones, repairs = _parse_backend(raw, self.language, word, native_names=native_names)
                source = "espeak_ng_1.52_candidate_with_baltic_rules"
                # Retain backend non-initial stress (greetings, loanwords and
                # lexical exceptions); do not impose Latvian stress on Lithuanian.
                nuclei = [p for p in phones if encode_phone(p.ipa)[31]]
                reliability = .5 if self.language == "lv-LV" and nuclei[0].stress == 1. else .25
            words.append({"word_index": wi, "normalized_word": word,
                          "text_start": token.start(), "text_end": token.end(),
                          "punctuation_before": before, "punctuation_after": after,
                          "boundary_before": _strength(before, self.language),
                          "boundary_after": _strength(after, self.language),
                          "phones": [p.ipa for p in phones], "ipa": " ".join(p.ipa for p in phones),
                          "raw_backend_ipa": raw, "raw_backend_phoneme_names": native_names,
                          "_phones": phones, "_accent": accent,
                          "pronunciation": {"source": source, "repairs": repairs,
                                            "stress_reliability": reliability,
                                            "stress_source": stress_source,
                                            "lexicon_covered": word in self.lexicon,
                                            "stress_known": all(p.stress_known for p in phones),
                                            "pitch_accent_known": accent is not None}})
        identity = hashlib.sha256((str(utterance_id or "") + "\0" + self.language + "\0" + normalized).encode()).hexdigest()[:20]
        ipa, word_indices, contexts, pauses, boundaries = ["sil"], [-1], [np.zeros(CONTEXT_DIM, np.float32)], [], []
        total = sum(len(w["phones"]) for w in words)
        position = 0
        # Question type is available from text within the current punctuation
        # clause, not from an utterance-wide question flag leaking across sentences.
        question = [False] * len(words)
        end = len(words)
        for wi in range(len(words) - 1, -1, -1):
            if words[wi]["boundary_after"] >= .7 or wi == len(words) - 1:
                end = wi
            question[wi] = "?" in words[end]["punctuation_after"]
        for wi, word in enumerate(words):
            phones, accent = word.pop("_phones"), word.pop("_accent")
            syllables = _syllables(phones)
            primary = [i for i, (_, _, stress, _) in enumerate(syllables) if stress == 1.]
            rows = np.zeros((len(phones), CONTEXT_DIM), np.float32)
            word["syllables"] = []
            for si, (start, stop, stress, known) in enumerate(syllables):
                nuclei = [i for i in range(start, stop) if encode_phone(phones[i].ipa)[31]]
                if len(nuclei) != 1:
                    raise FrontendError("Baltic syllable segmentation must contain exactly one nucleus")
                nucleus = nuclei[0]
                nx = encode_phone(phones[nucleus].ipa)
                long, diphthong = float(nx[22]), float(nx[30])
                heavy = bool(long or diphthong or any(_core(p.ipa) in _SONORANTS for p in phones[nucleus + 1:stop]))
                rows[start:stop, 12:14] = (stress, float(known))
                rows[start, 20] = rows[stop - 1, 21] = 1.
                if accent is not None and accent["syllable"] == si:
                    rows[start:stop, 18:20] = (_PITCH_CODES[self.language][accent["kind"]], 1.)
                for pi in range(start, stop):
                    before_stress = max(primary[0] - si, 0) / 4. if len(primary) == 1 else 0.
                    after_stress = max(si - primary[0], 0) / 4. if len(primary) == 1 else 0.
                    rows[pi, 22:] = (si / max(1, len(syllables) - 1), min(len(syllables) / 8., 2.),
                                     (pi - start) / max(1, stop - start - 1), float(heavy), long, diphthong,
                                     min(before_stress, 2.), min(after_stress, 2.),
                                     word["pronunciation"]["stress_reliability"],
                                     float(si == len(syllables) - 1),
                                     float(si == len(syllables) - 1 and (word["boundary_after"] > 0. or wi == len(words) - 1)),
                                     float(question[wi]))
                word["syllables"].append({"phone_start": start, "phone_end": stop,
                                           "nucleus_phone_index": nucleus, "stress": stress,
                                           "stress_known": known, "heavy_candidate": heavy,
                                           "pitch_accent": accent["kind"] if accent and accent["syllable"] == si else None})
            word["phone_stress"] = rows[:, 12].tolist()
            word["phone_stress_known"] = rows[:, 13].astype(bool).tolist()
            word["lexicon_key"] = f"u{identity}w{wi:04d}"
            word["phone_unit_indices"] = list(range(len(ipa), len(ipa) + len(phones)))
            for pi, phone in enumerate(phones):
                row = rows[pi]
                row[:12] = (float(pi == 0), float(pi == len(phones) - 1), pi / max(1, len(phones) - 1),
                            min(len(phones) / 12., 2.), wi / max(1, len(words) - 1), position / max(1, total - 1),
                            min(total / 200., 2.), min(len(words) / 50., 2.),
                            word["boundary_after"] if pi == len(phones) - 1 else 0.,
                            word["boundary_before"] if pi == 0 else 0., 0., 0.)
                ipa.append(phone.ipa); contexts.append(row); word_indices.append(wi)
                position += 1
            if wi + 1 < len(words):
                boundary = {"left_word_index": wi, "right_word_index": wi + 1,
                            "left_phone_unit_index": len(ipa) - 1, "strength": word["boundary_after"],
                            "punctuation": word["punctuation_after"], "pause_unit_index": None}
                if self.pause_policy == "word" or (self.pause_policy == "punctuation" and word["boundary_after"] > 0):
                    row = np.zeros(CONTEXT_DIM, np.float32)
                    row[4:8] = (wi / max(1, len(words) - 1), position / max(1, total - 1), min(total / 200., 2.), min(len(words) / 50., 2.))
                    row[8:10] = word["boundary_after"]
                    boundary["pause_unit_index"] = len(ipa)
                    pauses.append({"unit_index": len(ipa), "kind": "optional_word_pause", "after_word_index": wi,
                                   "strength": word["boundary_after"]})
                    ipa.append("sil"); contexts.append(row); word_indices.append(-1)
                boundaries.append(boundary)
        ipa.append("sil"); contexts.append(np.zeros(CONTEXT_DIM, np.float32)); word_indices.append(-1)
        x, c = np.stack([encode_phone(p) for p in ipa]), np.stack(contexts)
        c[0, 10] = c[-1, 11] = 1.
        c[-1, 4:6] = 1.
        c[0, 6:8] = c[-1, 6:8] = (min(total / 200., 2.), min(len(words) / 50., 2.))
        pause = x[:, 0].astype(bool)
        edge = np.zeros(len(ipa), bool); edge[0] = edge[-1] = True
        ident = self.identity()
        return {"phonetic_features": x, "context_features": c, "valid_mask": np.ones(len(ipa), bool),
                "edge_sil_mask": pause.copy(), "outer_edge_mask": edge, "pause_mask": pause,
                "word_indices": np.asarray(word_indices, np.int64), "ipa": ipa, "phoneme_tokens": ipa.copy(),
                "phonetic_feature_names": list(FEATURE_NAMES), "context_feature_names": CONTEXT_NAMES,
                "contract_version": CONTRACT_VERSION, "language": self.language, "normalized_text": normalized,
                "words": words, "word_boundaries": boundaries,
                "metadata": {"frontend_version": FRONTEND_VERSION, "feature_adapter_version": FEATURE_ADAPTER_VERSION,
                             "context_dim": CONTEXT_DIM, "config": self.get_config(), "backend": self.backend_info,
                             "lexicon": self.lexicon_info, "semantic_identity_sha256": ident["semantic_identity_sha256"],
                             "utterance_id": utterance_id, "text_only": True, "no_reference_audio_or_alignment": True,
                             "syllabification": "one_consonant_onset_heuristic_with_lt_spelling_aligned_diphthongs",
                             "optional_mask_semantics": "all_silence_units", "pause_units": pauses,
                             "limitations": ["backend_lexical_stress_is_a_candidate", "unwritten_pitch_accent_unknown",
                                             "word_internal_sandhi_only", "syllable_boundaries_are_heuristic"]}}

    prepare = encode


def encode(text, language, *, utterance_id=None, **config):
    return BalticFrontend(language, **config).encode(text, utterance_id=utterance_id)


def coverage_probe(texts, language, **config):
    """Audit corpus coverage without dropping failures or reading any audio."""
    frontend = BalticFrontend(language, **config)
    phones, sources, warnings, failures = Counter(), Counter(), Counter(), []
    accepted = total_words = known_stress_words = known_pitch_words = lexicon_words = 0
    for i, text in enumerate(texts):
        try:
            item = frontend.encode(text, utterance_id=f"coverage_{i}")
        except FrontendError as exc:
            failures.append({"row": i, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "reason": str(exc)})
            continue
        accepted += 1
        phones.update(item["ipa"])
        for word in item["words"]:
            total_words += 1
            info = word["pronunciation"]
            known_stress_words += bool(info["stress_known"])
            known_pitch_words += bool(info["pitch_accent_known"])
            lexicon_words += bool(info["lexicon_covered"])
            sources[info["source"]] += 1
            for repair in info["repairs"]:
                warnings.update([repair.get("warning", repair.get("rule"))])
    return {"language": frontend.language, "accepted_utterances": accepted, "rejected_utterances": len(failures),
            "words": total_words, "stress_known_words": known_stress_words,
            "pitch_accent_known_words": known_pitch_words, "lexicon_covered_words": lexicon_words,
            "lexicon_oov_words": total_words - lexicon_words,
            "phone_counts": dict(sorted(phones.items())), "source_counts": dict(sources),
            "rule_and_warning_counts": dict(warnings), "failures": failures,
            "frontend_identity": frontend.identity(), "reads_reference_audio": False}
