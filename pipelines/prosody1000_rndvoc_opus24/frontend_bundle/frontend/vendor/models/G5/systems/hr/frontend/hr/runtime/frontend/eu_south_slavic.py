"""Text-only HR/SL/BG frontend, with candidates separated from known prosody.

Each language has an independent model. The shared numerical network receives
32 articulation + 34 context coordinates, never text or language-specific rules.
Unmarked spelling does not certify HR pitch/length, SL tone/quality/stress, or
BG mobile stress. Optional text resources must be explicitly SHA-bound.
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
                      _LOCK, _MARKS, _strength, encode_phone as _encode_phone,
                      mfa_dictionary_entries, mfa_transcript, to_serializable)
from .phonetic_features import FEATURE_NAMES
from .universal import CONTEXT_NAMES as BASE_CONTEXT_NAMES

FRONTEND_VERSION = "eu_south_slavic_frontend_v2"
FEATURE_ADAPTER_VERSION = "south_slavic_articulatory32_v1"
CONTRACT_VERSION = "south_slavic_frontend_32_34_v1"
LANGUAGES = {"hr-HR": "hr", "sl-SI": "sl", "bg-BG": "bg"}
_ALIASES = {**{k.lower(): k for k in LANGUAGES}, **{v: k for k, v in LANGUAGES.items()}}
EXTRA_CONTEXT_NAMES = (
    "syllable_position_in_word", "syllable_count_scale", "phone_position_in_syllable",
    "phrase_final_syllable", "question_clause", "vowel_quantity_known", "vowel_quality_known",
    "hr_accent_rising", "hr_accent_known", "sl_toneme_acute", "sl_toneme_known",
    "syllable_distance_to_stress_candidate",
)
CONTEXT_NAMES = tuple(BASE_CONTEXT_NAMES) + EXTRA_CONTEXT_NAMES
CONTEXT_DIM = len(CONTEXT_NAMES)
_ALPHABET = {"hr-HR": set("abcdefghijklmnopqrstuvwxyzčćđšž"),
             "sl-SI": set("abcdefghijklmnopqrstuvwxyzčšžćđ"),
             "bg-BG": set("абвгдежзийклмнопрстуфхцчшщъьюяѝ")}
_TOKEN = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)
_GAP = set(" \t\r\n.,;:!?…—–-\"'()[]«»‹›")
# Ordinary prepositions must not become eSpeak's isolated letter names /Cə/.
# They contain no nucleus. Their stress is not certified by this table.
_CONSONANT_CLITICS = {"hr-HR": {"s": ["s"], "k": ["k"]},
                      "sl-SI": {"v": ["ʋ"], "s": ["s"], "z": ["z"], "k": ["k"], "h": ["x"]},
                      "bg-BG": {"в": ["v"], "с": ["s"]}}
_COMMON = {"p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
           "f": "f", "v": "v", "s": "s", "z": "z", "S": "ʃ", "Z": "ʒ",
           "h": "h", "x": "x", "C": "ç", "j": "j", "w": "w", "m": "m", "n": "n",
           "N": "ŋ", "r": "r", "R": "r", "R2": "r", "l": "l", "L": "ɫ",
           "ts": "ts", "dz": "dz", "tS": "tʃ", "dZ": "dʒ", "tS;": "tɕ", "dZ;": "dʑ",
           "l^": "ʎ", "n^": "ɲ", "@": "ə", "@-": "ə", "@2": "ə"}
_NAMES = {
    "hr-HR": {**_COMMON, "a": "a", "A": "a", "&": "a", "e": "e", "E": "e",
              "i": "i", "I": "i", "o": "o", "u": "u", "U": "u", "v": "ʋ",
              "aI": "ai", "oU": "ou", "R2": "r̩", "l-": "l̩", "*": "ɾ"},
    "sl-SI": {**_COMMON, "a": "a", "e": "e", "E": "ɛ", "o": "o", "O": "ɔ",
              "i": "i", "u": "u", "y": "y", "v": "ʋ", "v#": "ʋ"},
    "bg-BG": {**_COMMON, "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
              "n^": "nʲ", "l^": "lʲ", "g#": "ɡ", "g;": "ɡʲ", "t;": "tʲ", "d;": "dʲ"},
}


def _language(value):
    try:
        return _ALIASES[value.lower()]
    except (KeyError, AttributeError) as exc:
        raise FrontendError(f"Unsupported South Slavic language {value!r}") from exc


def _core(phone):
    return phone.translate(str.maketrans("", "", _MARKS))


def _stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _implementation_hashes():
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("eu_south_slavic.py", "eu_four.py", "phonetic_features.py", "universal.py")}


def encode_phone(ipa):
    """Local voiced palatal affricate extension; no shared inventory mutation."""
    canonical = _canonical_ipa(ipa)
    if _core(canonical) == "dʑ":
        result = _encode_phone(canonical.replace("dʑ", "tɕ")).copy()
        result[17] = 0. if "̥" in canonical else 1.
        return result
    return _encode_phone(canonical)


@lru_cache(maxsize=65536)
def _backend_phones(backend, word, voice, ipa=False):
    """Use the pinned backend without changing another frontend's whitelist."""
    if voice not in LANGUAGES.values():
        raise FrontendError(f"Unsupported South Slavic backend voice {voice!r}")
    if backend.library:
        with _LOCK:
            backend._select_voice(voice)
            data = ctypes.create_string_buffer(word.encode("utf-8"))
            pointer = ctypes.c_void_p(ctypes.addressof(data))
            clauses = []
            for _ in range(1000):
                if not pointer.value:
                    break
                previous = pointer.value
                raw = backend.lib.espeak_TextToPhonemes(ctypes.byref(pointer), 1,
                                                       (2 if ipa else 0) | (ord("_") << 8))
                clauses.append(raw.decode("utf-8") if raw else "")
                if pointer.value == previous:
                    raise FrontendError("eSpeak phoneme cursor did not advance")
            else:
                raise FrontendError("eSpeak phoneme clause limit exceeded")
            result = " ".join(clauses).strip()
    else:
        if voice not in backend._voices_checked:
            listing = backend._run(["--voices=" + voice])
            if not any(len(parts := line.split()) >= 2 and parts[1] == voice for line in listing.splitlines()):
                raise FrontendError(f"eSpeak voice {voice!r} is unavailable; fallback forbidden")
            backend._voices_checked.add(voice)
        result = backend._run(["-q", "-v", voice, "--ipa" if ipa else "-x", "--sep=_", "--stdin"], word + "\n").strip()
    if not result or re.search(r"\([^)]*\)", result):
        raise FrontendError(f"Empty or switched-language {voice} phones for {word!r}: {result!r}")
    return result


def _tokenize(text, language, resources):
    if not isinstance(text, str) or not text.strip():
        raise FrontendError("Nonempty ordinary text is required")
    typography = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"', "„": '"', "‟": '"', "‐": "-", "‑": "-"})
    text = re.sub(r"-{2,}", "—", unicodedata.normalize("NFC", text).translate(typography))
    if len(text) > 20000:
        raise FrontendError("Text exceeds 20,000 characters; split into sentences")
    # Preserve ordinary title case. Multi-letter ALL-CAPS tokens are ambiguous
    # initialisms/words; require an expansion or a bound explicit pronunciation.
    for match in _TOKEN.finditer(text):
        token = match.group()
        if len(token) > 1 and token.isupper() and token.lower() not in resources:
            raise FrontendError(f"Unexpanded uppercase acronym/word {token!r}; use ordinary case or a text lexicon")
    normalized = text.lower()
    tokens = list(_TOKEN.finditer(normalized))
    cursor = 0
    for token in tokens:
        unsupported = set(normalized[cursor:token.start()]) - _GAP
        if unsupported:
            raise FrontendError(f"Unsupported text symbols {sorted(unsupported)!r}; spell out numbers and abbreviations")
        invalid = set(token.group()) - _ALPHABET[language] - set("'-")
        if invalid:
            raise FrontendError(f"Mixed/unsupported {language} spelling {token.group()!r}: {sorted(invalid)!r}")
        cursor = token.end()
    unsupported = set(normalized[cursor:]) - _GAP
    if unsupported or not tokens:
        raise FrontendError(f"No lexical text or unsupported trailing symbols {sorted(unsupported)!r}")
    return normalized, tokens


def _parse_native(raw, word, language):
    tokens = [token for token in re.split(r"[_\s]+", raw) if token]
    phones, audit, pending = [], [], 0.
    names = _NAMES[language]
    for index, original in enumerate(tokens):
        token = original
        if "'" in token or "," in token:
            pending = 1. if "'" in token else .5
            token = token.replace("'", "").replace(",", "")
        if not token:
            continue
        if token == "r*" and language == "hr-HR":
            if index + 1 == len(tokens) or tokens[index + 1] != "*":
                raise FrontendError(f"HR zero-IPA rhotic bridge without flap in {word!r}")
            audit.append({"rule": "hr_zero_ipa_rhotic_bridge", "from": token, "to": None})
            continue
        if token == "|" and language == "sl-SI":
            # Native _| is a 1 ms stop-release separator in sl_rules (prst).
            audit.append({"rule": "sl_native_stop_separator_not_lexical_pause", "from": "_|", "to": None})
            continue
        if token == ";":
            if not phones:
                raise FrontendError(f"Unattached palatal/hiatus marker in {word!r}/{raw!r}")
            if encode_phone(phones[-1].ipa)[31] and _core(phones[-1].ipa) == "i":
                audit.append({"rule": "native_zero_ipa_i_hiatus_bridge", "from": token, "to": None})
            elif language == "bg-BG" and not encode_phone(phones[-1].ipa)[31]:
                phones[-1] = replace(phones[-1], ipa=phones[-1].ipa + "ʲ")
                audit.append({"rule": "bg_native_palatal_marker_on_consonant", "to": phones[-1].ipa})
            else:
                raise FrontendError(f"Unsupported linking marker in {word!r}/{raw!r}")
            continue
        if token == ":":
            if not phones or not encode_phone(phones[-1].ipa)[31]:
                raise FrontendError(f"Unattached vowel length in {word!r}/{raw!r}")
            phones[-1] = replace(phones[-1], ipa=phones[-1].ipa.rstrip("ː") + "ː")
            audit.append({"rule": "native_virtual_vowel_length_candidate", "to": phones[-1].ipa})
            continue
        long = token.endswith(":")
        base = token[:-1] if long else token
        if base in names:
            value = names[base]
        elif language == "bg-BG" and base.endswith(";") and base[:-1] in names:
            value = names[base[:-1]] + "ʲ"
        else:
            raise FrontendError(f"Unmapped {language} native phone {original!r} in {word!r}; raw={raw!r}")
        # Standard Slovene lj/nj are l/n outside prevocalic position; unlike HR.
        if language == "sl-SI" and base in ("l^", "n^"):
            value = "l" if base == "l^" else "n"
            following = tokens[index + 1].lstrip("',").rstrip(":") if index + 1 < len(tokens) else ""
            audit.append({"rule": "sl_lj_nj_standard_alveolar", "from": base, "to": value})
            if following in names and encode_phone(names[following])[31]:
                phones.append(_Phone(value, 0., False))
                value = "j"
                audit.append({"rule": "sl_lj_nj_prevocalic_j", "from": base, "to": [phones[-1].ipa, "j"]})
        # Native HR dz is not the separate letter đ; keep its alveolar value.
        # The broad symbol dz also represents coarticulated d+z in this backend.
        if base == "dz":
            audit.append({"rule": "native_dz_alveolar_not_dj", "from": base, "to": value})
        if language == "hr-HR" and base in ("A", "&", "E", "I", "U"):
            audit.append({"rule": "hr_five_vowel_phonemic_quality", "from": base, "to": value})
        if language == "hr-HR" and base == "R2":
            audit.append({"rule": "hr_consonantal_r_nucleus", "from": base, "to": value})
        value += "ː" if long else ""
        x = encode_phone(value)
        phones.append(_Phone(value, pending if x[31] else 0., False))
        if x[31]:
            pending = 0.
    if pending or not phones:
        raise FrontendError(f"No phones or unattached stress in {word!r}/{raw!r}")
    if language == "bg-BG":
        # In the native table light l also stands for Cyrillic л before я/ю/ь.
        # Match only the exact number/order of l phones; never guess alignment.
        spelling_l = [i for i, ch in enumerate(word) if ch == "л"]
        phone_l = [i for i, p in enumerate(phones) if _core(p.ipa) in ("l", "ɫ")]
        if len(spelling_l) == len(phone_l):
            for gi, pi in zip(spelling_l, phone_l):
                if word[gi + 1:gi + 2] in ("я", "ю", "ь") and "ʲ" not in phones[pi].ipa:
                    previous = phones[pi].ipa
                    phones[pi] = replace(phones[pi], ipa=previous.replace("ɫ", "l") + "ʲ")
                    audit.append({"rule": "bg_orthographic_l_before_iotated_vowel", "from": previous, "to": phones[pi].ipa})
        elif any(word[i + 1:i + 2] in ("я", "ю", "ь") for i in spelling_l):
            raise FrontendError(f"Cannot align Bulgarian palatal л in {word!r}")
    return phones, audit


def _syllables(phones):
    nuclei = [i for i, p in enumerate(phones) if encode_phone(p.ipa)[31]]
    if not nuclei:
        return []
    starts = [0] + [right if right == left + 1 else right - 1 for left, right in zip(nuclei, nuclei[1:])]
    return [(start, stop, nucleus) for start, stop, nucleus in zip(starts, starts[1:] + [len(phones)], nuclei)]


def _parse_override(entry, language):
    if not isinstance(entry, dict) or not isinstance(entry.get("source"), str) or not entry["source"].strip():
        raise FrontendError("Text pronunciation entries need a nonempty source")
    if not isinstance(entry.get("phones"), list) or not entry["phones"]:
        raise FrontendError("Text pronunciation entries need a nonempty phones list")
    if type(entry.get("stress_known", False)) is not bool:
        raise FrontendError("stress_known must be boolean")
    phones, pending = [], 0.
    for value in entry["phones"]:
        if not isinstance(value, str):
            raise FrontendError("Override phones must be strings")
        if "ˈ" in value or "ˌ" in value:
            pending = 1. if "ˈ" in value else .5
        ipa = _canonical_ipa(value.replace("ˈ", "").replace("ˌ", ""))
        if ipa == "sil":
            raise FrontendError("Lexical overrides cannot contain silence")
        x = encode_phone(ipa)
        phones.append(_Phone(ipa, pending if x[31] else 0., bool(entry.get("stress_known", False))))
        if x[31]:
            pending = 0.
    if pending or not _syllables(phones):
        raise FrontendError("Override has no nucleus or unattached stress")
    if entry.get("stress_known") and sum(p.stress == 1. for p in phones) != 1:
        raise FrontendError("Certified word stress requires exactly one primary nucleus")
    masks = {}
    for key in ("vowel_quantity_known", "vowel_quality_known"):
        indices = entry.get(key, [])
        if (not isinstance(indices, list)
                or any(type(i) is not int or not 0 <= i < len(phones) or not encode_phone(phones[i].ipa)[31] for i in indices)
                or len(set(indices)) != len(indices)):
            raise FrontendError(f"{key} must list unique nucleus phone indices")
        masks[key] = set(indices)
    accent = entry.get("accent")
    if accent is not None:
        categories = {"hr-HR": ("rising", "falling"), "sl-SI": ("acute", "circumflex")}
        syllables = _syllables(phones)
        if (not isinstance(accent, dict) or language not in categories
                or accent.get("category") not in categories[language]
                or type(accent.get("syllable")) is not int or not 0 <= accent["syllable"] < len(syllables)
                or not isinstance(accent.get("source"), str) or not accent["source"].strip()
                or not entry.get("stress_known")
                or phones[syllables[accent["syllable"]][2]].stress != 1.):
            raise FrontendError("Accent needs language-specific category, source, and the certified stressed syllable")
    if any(key in entry for key in ("tone_contour", "pitch_accent", "foot_quantities")):
        raise FrontendError("Use language-specific accent labels; F0 contours/foot quantities are not supported")
    return phones, masks, accent


def _validate_prosody(entry, language):
    """Validate externally evidenced labels independently of a G2P phone list.

    Dynamic Slovene accents are NOT tonemes. BG lexical stress is NOT an
    observed F0 or reduction label. No IPA/allophone guesses are certified here.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("source"), str) or not entry["source"].strip():
        raise FrontendError("Prosody entries need a nonempty source")
    if any(k in entry for k in ("phones", "accent", "tone_contour", "pitch_accent", "foot_quantities",
                                "vowel_quantity_known", "vowel_quality_known")):
        raise FrontendError("Prosody-only entries cannot mix full phone/accent overrides")
    p = entry.get("prosody")
    if not isinstance(p, dict):
        raise FrontendError("prosody must be an object")
    if set(p) - {"nucleus_count", "nucleus_signature", "stress_syllable", "vowels"}:
        raise FrontendError("Unknown prosody labels are not supported")
    n = p.get("nucleus_count")
    signature = p.get("nucleus_signature")
    if (type(n) is not int or not 1 <= n <= 32 or not isinstance(signature, list)
            or len(signature) != n or any(v not in ("a", "e", "i", "o", "u", "ə", "r") for v in signature)):
        raise FrontendError("Prosody needs a bounded nucleus count and explicit signature")
    stress = p.get("stress_syllable")
    if stress is not None and (type(stress) is not int or not 0 <= stress < n):
        raise FrontendError("Invalid prosody stress syllable")
    if type(entry.get("stress_known", False)) is not bool or bool(entry.get("stress_known")) != (stress is not None):
        raise FrontendError("Prosody stress_known must agree with its evidenced stress index")
    vowels = p.get("vowels", [])
    if not isinstance(vowels, list):
        raise FrontendError("Prosody vowels must be a list")
    seen = set()
    for vowel in vowels:
        if not isinstance(vowel, dict):
            raise FrontendError("Invalid prosody vowel")
        i, ipa = vowel.get("syllable"), vowel.get("ipa")
        if type(i) is not int or not 0 <= i < n or i in seen or not isinstance(ipa, str):
            raise FrontendError("Prosody vowel indices must be unique and in range")
        seen.add(i)
        if ipa not in {base + length for base in ("a", "e", "ɛ", "i", "o", "ɔ", "u", "ə") for length in ("", "ː")}:
            raise FrontendError("Prosody vowel override must be one supported vowel")
        if set(vowel) - {"syllable", "ipa", "quality_known", "quantity_known"}:
            raise FrontendError("Unknown prosody vowel labels are not supported")
        for key in ("quality_known", "quantity_known"):
            if type(vowel.get(key, False)) is not bool:
                raise FrontendError("Prosody vowel evidence masks must be boolean")
        if language == "bg-BG" or not (vowel.get("quality_known") or vowel.get("quantity_known")):
            raise FrontendError("BG reduction/quantity and unsupported vowel guesses are not lexicon labels")
    return p


def _apply_prosody(phones, entry, language):
    p = _validate_prosody(entry, language)
    nuclei = [i for i, phone in enumerate(phones) if encode_phone(phone.ipa)[31]]
    families = {"a": {"a", "ɐ"}, "e": {"e", "ɛ", "ə"}, "i": {"i", "ɪ"},
                "o": {"o", "ɔ"}, "u": {"u", "ʊ"}, "ə": {"ə", "ɤ"}, "r": {"ə", "r̩"}}
    masks = {"vowel_quantity_known": set(), "vowel_quality_known": set()}
    aligned = (len(nuclei) == p["nucleus_count"]
               and all(_core(phones[ni].ipa) in families[letter] for ni, letter in zip(nuclei, p["nucleus_signature"])))
    if not aligned:
        return phones, masks, [{"rule": "lexical_prosody_alignment_mismatch_unknown",
                                "expected_nuclei": p["nucleus_count"], "actual_nuclei": len(nuclei)}], False
    result = list(phones)
    if entry.get("stress_known"):
        for si, ni in enumerate(nuclei):
            result[ni] = replace(result[ni], stress=float(si == p["stress_syllable"]), stress_known=True)
    for value in p.get("vowels", []):
        ni = nuclei[value["syllable"]]
        target = value["ipa"]
        # Evidence masks authorize each attribute separately. A quantity-only
        # entry must never smuggle in an uncertified vowel-quality substitution.
        base = _core(target) if value.get("quality_known") else _core(result[ni].ipa)
        length = ("ː" if "ː" in target else "") if value.get("quantity_known") else ("ː" if "ː" in result[ni].ipa else "")
        result[ni] = replace(result[ni], ipa=base + length)
        if value.get("quality_known"):
            masks["vowel_quality_known"].add(ni)
        if value.get("quantity_known"):
            masks["vowel_quantity_known"].add(ni)
    return result, masks, [{"rule": "text_lexicon_consensus_prosody_applied", "source": entry["source"]}], True


class SouthSlavicFrontend:
    def __init__(self, language, *, espeak_library=None, espeak_data=None, espeak_binary=None,
                 required_espeak_version="1.52.0", pause_policy="word", pronunciation_overrides=None,
                 lexicon_path=None, lexicon_sha256=None):
        self.language = _language(language)
        if pause_policy not in ("punctuation", "word", "none"):
            raise FrontendError("pause_policy must be punctuation, word, or none")
        self.pause_policy = pause_policy
        self.overrides = json.loads(json.dumps(pronunciation_overrides or {}, ensure_ascii=False))
        self.lexicon, self.lexicon_info, self.unresolved_lexicon = {}, None, {}
        if bool(lexicon_path) != bool(lexicon_sha256):
            raise FrontendError("External text lexicons require path and expected SHA256")
        if lexicon_path:
            raw = Path(lexicon_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != lexicon_sha256:
                raise FrontendError("External text lexicon SHA256 mismatch")
            document = json.loads(raw)
            if (document.get("schema") != "south_slavic_pronunciation_lexicon_v1"
                    or document.get("language") != self.language or not document.get("provenance")
                    or not isinstance(document.get("entries"), dict)):
                raise FrontendError("Invalid South Slavic lexicon schema, language or provenance")
            self.lexicon = document["entries"]
            self.unresolved_lexicon = document.get("unresolved_entries", {})
            if not isinstance(self.unresolved_lexicon, dict):
                raise FrontendError("Unresolved lexicon entries must be a word mapping")
            self.lexicon_info = {"sha256": lexicon_sha256, "provenance": document["provenance"], "entries": len(self.lexicon),
                                 "unresolved_entries": len(self.unresolved_lexicon)}
        for collection in (self.lexicon, self.overrides, self.unresolved_lexicon):
            if not isinstance(collection, dict):
                raise FrontendError("Pronunciation overrides must be a word mapping")
            for word, entry in collection.items():
                if (not isinstance(word, str) or word != unicodedata.normalize("NFC", word).lower()
                        or not _TOKEN.fullmatch(word) or set(word) - _ALPHABET[self.language] - set("'-")):
                    raise FrontendError("Lexicon/override keys must be one NFC lowercase word in the language's script")
                if not isinstance(entry, dict):
                    raise FrontendError("Lexicon/override entries must be objects")
                if collection is self.unresolved_lexicon:
                    if (not isinstance(entry, dict) or not isinstance(entry.get("reason"), str)
                            or not entry["reason"] or word in self.lexicon):
                        raise FrontendError("Unresolved lexicon entries require a reason and cannot override a resolved entry")
                elif "prosody" in entry:
                    _validate_prosody(entry, self.language)
                else:
                    _parse_override(entry, self.language)
        self.backend = EspeakBackend(library=espeak_library, data=espeak_data, binary=espeak_binary,
                                     required_version=required_espeak_version)
        self.backend_info = self.backend.provenance(LANGUAGES[self.language])
        required_data = {f"{LANGUAGES[self.language]}_dict", "phondata", "phonindex", "phontab", "intonations"}
        if (not required_data.issubset(self.backend_info["data_sha256"])
                or not any(k.startswith("lang/") and k.endswith("/" + LANGUAGES[self.language])
                           for k in self.backend_info["data_sha256"])):
            raise FrontendError("Pinned backend data identity is incomplete")
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
        implementation = _implementation_hashes()
        config = {k: v for k, v in self._config.items() if k != "lexicon_path"}
        config.update(frontend_version=FRONTEND_VERSION, feature_adapter_version=FEATURE_ADAPTER_VERSION,
                      contract_version=CONTRACT_VERSION, context_feature_names=CONTEXT_NAMES,
                      backend_version=self.backend_info["version"], backend_data_sha=self.backend_info["data_identity_sha256"],
                      implementation_sha256=implementation)
        return {"semantic_identity_sha256": _stable_hash(config), "semantic_config": config,
                "backend": self.backend_info, "lexicon": self.lexicon_info,
                "source_sha256": implementation["eu_south_slavic.py"], "implementation_sha256": implementation}

    def encode(self, text, *, utterance_id=None):
        normalized, tokens = _tokenize(text, self.language, set(self.overrides) | set(self.lexicon))
        words = []
        for wi, token in enumerate(tokens):
            word = token.group()
            before = normalized[tokens[wi - 1].end() if wi else 0:token.start()]
            after = normalized[token.end():tokens[wi + 1].start() if wi + 1 < len(tokens) else len(normalized)]
            masks, accent, raw, native, repairs = {"vowel_quantity_known": set(), "vowel_quality_known": set()}, None, None, None, []
            entry = self.overrides.get(word, self.lexicon.get(word))
            applied = False
            if entry is not None and "prosody" not in entry:
                explicit = word in self.overrides
                phones, masks, accent = _parse_override(entry, self.language)
                source = "explicit_text_pronunciation_override" if explicit else "sha_bound_text_lexicon"
                applied = True
            elif word in _CONSONANT_CLITICS[self.language]:
                phones = [_Phone(p, 0., False) for p in _CONSONANT_CLITICS[self.language][word]]
                source = "ordinary_consonant_preposition"
                repairs = [{"rule": "consonant_preposition_not_letter_name"}]
            else:
                raw = _backend_phones(self.backend, word, LANGUAGES[self.language], True)
                native = _backend_phones(self.backend, word, LANGUAGES[self.language])
                phones, repairs = _parse_native(native, word, self.language)
                source = "espeak_ng_1.52_candidate_with_language_adapter"
                if not _syllables(phones):
                    raise FrontendError(f"No nucleus in non-clitic {word!r}")
                if self.language == "hr-HR":
                    masks["vowel_quality_known"] = {i for i, p in enumerate(phones) if _core(p.ipa) in ("a", "e", "i", "o", "u")}
            if entry is not None and "prosody" in entry:
                phones, masks, additions, applied = _apply_prosody(phones, entry, self.language)
                repairs.extend(additions)
                if applied:
                    source = "explicit_text_prosody_override" if word in self.overrides else "sha_bound_text_lexicon_prosody"
            nuclei = [p for p in phones if encode_phone(p.ipa)[31]]
            words.append({"word_index": wi, "normalized_word": word, "text_start": token.start(), "text_end": token.end(),
                          "punctuation_before": before, "punctuation_after": after,
                          "boundary_before": _strength(before, self.language), "boundary_after": _strength(after, self.language),
                          "phones": [p.ipa for p in phones], "ipa": " ".join(p.ipa for p in phones),
                          "raw_backend_ipa": raw, "raw_backend_phoneme_names": native,
                          "_phones": phones, "_masks": masks, "_accent": accent,
                          "pronunciation": {"source": source, "repairs": repairs, "lexicon_covered": word in self.lexicon,
                                            "lexicon_applied": applied and word in self.lexicon,
                                            "lexicon_unresolved": self.unresolved_lexicon.get(word),
                                            "stress_known": bool(nuclei) and all(p.stress_known for p in nuclei),
                                            "primary_stress_candidate_present": any(p.stress == 1. for p in nuclei),
                                            "pitch_accent_known": accent is not None, "accent": accent,
                                            "quantity_known_nuclei": sorted(masks["vowel_quantity_known"]),
                                            "quality_known_nuclei": sorted(masks["vowel_quality_known"]),
                                            "quantity_applicable": self.language != "bg-BG",
                                            "quality_known_means": "phonemic_inventory_not_measured_acoustic_formants",
                                            "stress_source": source if any(p.stress_known for p in nuclei) else "uncertified_backend_candidate"}})
        identity = hashlib.sha256((str(utterance_id or "") + "\0" + self.language + "\0" + normalized).encode()).hexdigest()[:20]
        ipa, word_indices, contexts = ["sil"], [-1], [np.zeros(CONTEXT_DIM, np.float32)]
        syllable_indices, pauses, boundaries = [-1], [], []
        total = sum(len(w["phones"]) for w in words)
        position = global_syllable = 0
        question, end = [False] * len(words), len(words)
        for wi in range(len(words) - 1, -1, -1):
            if words[wi]["boundary_after"] >= .7 or wi == len(words) - 1:
                end = wi
            question[wi] = "?" in words[end]["punctuation_after"]
        for wi, word in enumerate(words):
            phones, masks, accent = word.pop("_phones"), word.pop("_masks"), word.pop("_accent")
            syllables = _syllables(phones)
            rows = np.zeros((len(phones), CONTEXT_DIM), np.float32)
            local_syllable = np.full(len(phones), -1, np.int64)
            stressed = [si for si, (_, _, ni) in enumerate(syllables) if phones[ni].stress == 1.]
            word["syllables"] = []
            for si, (start, stop, nucleus) in enumerate(syllables):
                stress, known = phones[nucleus].stress or 0., phones[nucleus].stress_known
                rows[start:stop, 12:14] = (stress, float(known))
                rows[start, 20] = rows[stop - 1, 21] = 1.
                for pi in range(start, stop):
                    rows[pi, 22:27] = (si / max(1, len(syllables) - 1), min(len(syllables) / 8., 2.),
                                       (pi - start) / max(1, stop - start - 1),
                                       float(si == len(syllables) - 1 and (word["boundary_after"] > 0. or wi == len(words) - 1)), float(question[wi]))
                    rows[pi, 33] = (si - stressed[0]) / max(1, len(syllables) - 1) if len(stressed) == 1 else 0.
                rows[nucleus, 27:29] = (float(nucleus in masks["vowel_quantity_known"]), float(nucleus in masks["vowel_quality_known"]))
                if accent is not None and accent["syllable"] == si:
                    column, positive = (29, "rising") if self.language == "hr-HR" else (31, "acute")
                    rows[start:stop, column:column + 2] = (float(accent["category"] == positive), 1.)
                local_syllable[start:stop] = global_syllable + si
                word["syllables"].append({"syllable_index": global_syllable + si, "phone_start": start, "phone_end": stop,
                                           "nucleus_phone_index": nucleus, "stress_candidate": stress, "stress_known": known,
                                           "quantity_known": nucleus in masks["vowel_quantity_known"],
                                           "quality_known": nucleus in masks["vowel_quality_known"]})
            global_syllable += len(syllables)
            word["phone_stress"], word["phone_stress_known"] = rows[:, 12].tolist(), rows[:, 13].astype(bool).tolist()
            word["lexicon_key"] = f"u{identity}w{wi:04d}"
            word["phone_unit_indices"] = list(range(len(ipa), len(ipa) + len(phones)))
            for pi, phone in enumerate(phones):
                row = rows[pi]
                row[:12] = (float(pi == 0), float(pi == len(phones) - 1), pi / max(1, len(phones) - 1),
                            min(len(phones) / 12., 2.), wi / max(1, len(words) - 1), position / max(1, total - 1),
                            min(total / 200., 2.), min(len(words) / 50., 2.),
                            word["boundary_after"] if pi == len(phones) - 1 else 0., word["boundary_before"] if pi == 0 else 0., 0., 0.)
                ipa.append(phone.ipa); contexts.append(row); word_indices.append(wi)
                syllable_indices.append(int(local_syllable[pi])); position += 1
            if wi + 1 < len(words):
                boundary = {"left_word_index": wi, "right_word_index": wi + 1, "left_phone_unit_index": len(ipa) - 1,
                            "strength": word["boundary_after"], "punctuation": word["punctuation_after"], "pause_unit_index": None}
                if self.pause_policy == "word" or (self.pause_policy == "punctuation" and word["boundary_after"] > 0):
                    row = np.zeros(CONTEXT_DIM, np.float32)
                    row[4:8] = (wi / max(1, len(words) - 1), position / max(1, total - 1), min(total / 200., 2.), min(len(words) / 50., 2.))
                    row[8:10] = word["boundary_after"]
                    boundary["pause_unit_index"] = len(ipa)
                    pauses.append({"unit_index": len(ipa), "kind": "optional_word_pause", "after_word_index": wi, "strength": word["boundary_after"]})
                    ipa.append("sil"); contexts.append(row); word_indices.append(-1); syllable_indices.append(-1)
                boundaries.append(boundary)
        ipa.append("sil"); contexts.append(np.zeros(CONTEXT_DIM, np.float32)); word_indices.append(-1); syllable_indices.append(-1)
        x, c = np.stack([encode_phone(p) for p in ipa]), np.stack(contexts)
        c[0, 10] = c[-1, 11] = 1.; c[-1, 4:6] = 1.
        c[0, 6:8] = c[-1, 6:8] = (min(total / 200., 2.), min(len(words) / 50., 2.))
        pause = x[:, 0].astype(bool)
        edge = np.zeros(len(ipa), bool); edge[0] = edge[-1] = True
        return {"phonetic_features": x, "context_features": c, "valid_mask": np.ones(len(ipa), bool),
                "edge_sil_mask": pause.copy(), "outer_edge_mask": edge, "pause_mask": pause,
                "word_indices": np.asarray(word_indices, np.int64), "foot_indices": np.full(len(ipa), -1, np.int64),
                "syllable_indices": np.asarray(syllable_indices, np.int64), "ipa": ipa, "phoneme_tokens": ipa.copy(),
                "phonetic_feature_names": list(FEATURE_NAMES), "context_feature_names": CONTEXT_NAMES,
                "contract_version": CONTRACT_VERSION, "language": self.language, "normalized_text": normalized,
                "words": words, "word_boundaries": boundaries,
                "metadata": {"frontend_version": FRONTEND_VERSION, "feature_adapter_version": FEATURE_ADAPTER_VERSION,
                             "context_dim": CONTEXT_DIM, "config": self.get_config(), "backend": self.backend_info,
                             "lexicon": self.lexicon_info, "semantic_identity_sha256": self.identity()["semantic_identity_sha256"],
                             "utterance_id": utterance_id, "text_only": True, "no_reference_audio_or_alignment": True,
                             "syllabification": "one_consonant_onset_heuristic_nucleusless_clitics_minus_one",
                             "foot_grouping": "not_provided", "foot_grouping_is_gold": False,
                             "optional_mask_semantics": "all_silence_units", "pause_units": pauses,
                             "limitations": ["unmarked_word_stress_is_backend_candidate_not_gold",
                                             "hr_unwritten_pitch_and_quantity_unknown", "hr_ije_syllabification_not_lexically_disambiguated",
                                             "sl_unwritten_vowel_quality_quantity_and_toneme_unknown",
                                             "bg_reduction_learned_from_audio_not_imposed_by_unverified_stress",
                                             "word_internal_sandhi_only_no_clitic_stress_transfer",
                                             "text_syllable_boundaries_are_heuristic", "no_text_derived_F0_contour"]}}

    prepare = encode


def encode(text, language, *, utterance_id=None, **config):
    return SouthSlavicFrontend(language, **config).encode(text, utterance_id=utterance_id)


def coverage_probe(texts, language, **config):
    """Audit only text; include all rejections and counts of unknown categories."""
    frontend = SouthSlavicFrontend(language, **config)
    phones, sources, warnings, failures = Counter(), Counter(), Counter(), []
    counts = Counter()
    max_units = 0
    for i, text in enumerate(texts):
        try:
            item = frontend.encode(text, utterance_id=f"coverage_{i}")
        except FrontendError as exc:
            failures.append({"row": i, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "reason": str(exc)})
            continue
        counts["accepted_utterances"] += 1
        max_units = max(max_units, len(item["ipa"])); phones.update(item["ipa"])
        for word in item["words"]:
            info = word["pronunciation"]
            counts["words"] += 1
            counts["nuclei"] += len(word["syllables"])
            for key in ("stress_known", "primary_stress_candidate_present", "pitch_accent_known", "lexicon_covered"):
                counts[key + "_words"] += int(info[key])
            counts["quantity_known_nuclei"] += len(info["quantity_known_nuclei"])
            counts["quality_known_nuclei"] += len(info["quality_known_nuclei"])
            counts["syllabic_r_units"] += sum("r̩" in p for p in word["phones"])
            counts["palatalized_consonant_units"] += sum("ʲ" in p for p in word["phones"])
            counts["nucleusless_clitic_words"] += not word["syllables"]
            sources[info["source"]] += 1
            warnings.update(r["rule"] for r in info["repairs"])
    counts["stress_unknown_words"] = counts["words"] - counts["stress_known_words"]
    counts["pitch_accent_unknown_words"] = counts["words"] - counts["pitch_accent_known_words"] if frontend.language != "bg-BG" else 0
    counts["quantity_unknown_nuclei"] = counts["nuclei"] - counts["quantity_known_nuclei"] if frontend.language != "bg-BG" else 0
    return {"language": frontend.language, **dict(counts), "rejected_utterances": len(failures),
            "max_units": max_units, "phone_counts": dict(sorted(phones.items())), "source_counts": dict(sources),
            "rule_and_warning_counts": dict(warnings), "failures": failures,
            "frontend_identity": frontend.identity(), "reads_reference_audio": False}
