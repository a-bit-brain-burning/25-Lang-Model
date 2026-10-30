"""Text-only Finnish/Estonian frontend with explicit, masked Estonian quantity.

The backend supplies candidates. Q2/Q3, lexical palatalization and secondary
stress are not recovered from unmarked spelling by pretending they are known.
Only a SHA-bound text lexicon or explicit text override may certify ET quantity.
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

FRONTEND_VERSION = "eu_finnic_frontend_v1"
FEATURE_ADAPTER_VERSION = "finnic_articulatory32_v1"
CONTRACT_VERSION = "finnic_frontend_32_34_v1"
LANGUAGES = {"fi-FI": "fi", "et-EE": "et"}
_ALIASES = {**{k.lower(): k for k in LANGUAGES}, **{v: k for k, v in LANGUAGES.items()}}
EXTRA_CONTEXT_NAMES = (
    "syllable_position_in_word", "syllable_count_scale", "phone_position_in_syllable",
    "foot_head_syllable", "syllable_position_in_foot", "foot_syllable_count_scale",
    "et_foot_quantity_q1", "et_foot_quantity_q2", "et_foot_quantity_q3",
    "et_foot_quantity_known", "phrase_final_syllable", "question_clause",
)
CONTEXT_NAMES = tuple(BASE_CONTEXT_NAMES) + EXTRA_CONTEXT_NAMES
CONTEXT_DIM = len(CONTEXT_NAMES)
_ALPHABET = {
    "fi-FI": set("abcdefghijklmnopqrstuvwxyzåäöšž"),
    "et-EE": set("abcdefghijklmnopqrstuvwxyzõäöüšž"),
}
_COMMON = {"p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
           "f": "f", "v": "v", "s": "s", "z": "z", "S": "ʃ", "Z": "ʒ",
           "h": "h", "x": "x", "j": "j", "w": "w", "m": "m", "n": "n",
           "N": "ŋ", "r": "r", "R": "r", "R2": "r", "l": "l",
           "s2": "s", "ts": "ts", "tS": "tʃ", "dZ": "dʒ"}
_FI_NAMES = {**_COMMON, "a": "ɑ", "a2": "ɑ", "e": "e", "&": "æ", "o": "o",
             "Y": "ø", "i": "i", "I": "i", "u": "u", "u1": "u", "U": "u", "y": "y",
             "v": "ʋ", "ai": "ɑi", "&i": "æi", "oi": "oi", "ei": "ei", "Yi": "øi",
             "ui": "ui", "yi": "yi", "au": "ɑu", "ou": "ou", "eu": "eu", "iu": "iu",
             "&y": "æy", "ey": "ey", "Yy": "øy", "iy": "iy", "uo": "uo", "ie": "ie", "yY": "yø"}
_ET_NAMES = {**_COMMON, "a": "ɑ", "A": "ɑ", "A1": "ɑ", "e": "e", "E": "e", "E1": "e",
             "i": "i", "I": "i", "I1": "i", "o": "o", "O": "o", "O1": "o",
             "u": "u", "U": "u", "U1": "u", "8": "ɤ", "&": "æ", "Y": "ø", "y": "y",
             "&i": "æi", "Yi": "øi", "yi": "yi", "b": "p", "d": "t", "g": "k",
             "z": "s", "Z": "ʃ", "l^": "lʲ", "n^": "nʲ", "s^": "sʲ", "t^": "tʲ", "d^": "tʲ"}
# All 36 combinations in Asu & Teras (2009), table 1, including loans. The
# spelling-matched merger is still a G2P candidate, not a lexical quantity label.
_ET_DIPHTHONGS = set("ɑe ɑi ɑo ɑu eɑ ei eo eu iɑ ie io iu oɑ oe oi ou uɑ ue ui uo ɤɑ ɤe ɤi ɤo ɤu æe æi æo æu øɑ øe øi yɑ ye yi yo".split())
_ET_UNSTRESSED_DIPHTHONGS = {"ɑi", "ei", "ui"}


def _language(value):
    try:
        return _ALIASES[value.lower()]
    except (KeyError, AttributeError) as exc:
        raise FrontendError(f"Unsupported Finnic language {value!r}") from exc


def _core(phone):
    return phone.translate(str.maketrans("", "", _MARKS))


def _stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _implementation_hashes():
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("eu_finnic.py", "eu_four.py", "phonetic_features.py", "universal.py")}


@lru_cache(maxsize=65536)
def _native_names(backend, word, voice):
    """Preserve named phones lost in the backend's IPA rendering (ET s^, õ)."""
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
                raw = backend.lib.espeak_TextToPhonemes(ctypes.byref(pointer), 1, ord("_") << 8)
                clauses.append(raw.decode("utf-8") if raw else "")
                if pointer.value == previous:
                    raise FrontendError("eSpeak native-phoneme cursor did not advance")
            else:
                raise FrontendError("eSpeak native-phoneme clause limit exceeded")
            result = " ".join(clauses).strip()
    else:
        result = backend._run(["-q", "-v", voice, "-x", "--sep=_", "--stdin"], word + "\n").strip()
    if not result or re.search(r"\([^)]*\)", result):
        raise FrontendError(f"Empty or switched-language {voice} native phones: {result!r}")
    return result


def _parse_native(raw, word, language):
    names = _FI_NAMES if language == "fi-FI" else _ET_NAMES
    tokens = [t for t in re.split(r"[_\s]+", raw) if t]
    phones, original_names, audit, pending = [], [], [], 0.
    for index, value in enumerate(tokens):
        token = value
        if "'" in token or "," in token:
            pending = 1. if "'" in token else .5
            token = token.replace("'", "").replace(",", "")
        if not token:
            continue
        if token == ";" and language == "fi-FI":
            # ph_finnish i/I IfNextVowelAppend(;): zero-IPA hiatus bridge.
            if not phones or not encode_phone(phones[-1].ipa)[31]:
                raise FrontendError(f"Unattached Finnish hiatus bridge in {word!r}/{raw!r}")
            audit.append({"rule": "fi_zero_ipa_hiatus_bridge", "from": token, "to": None})
            continue
        if token == "i1" and language == "et-EE":
            following = tokens[index + 1].lstrip("',").rstrip(":") if index + 1 < len(tokens) else ""
            if not following.endswith("^"):
                raise FrontendError(f"ET i1 transition without palatal consonant in {word!r}/{raw!r}")
            audit.append({"rule": "et_i1_pre_palatal_transition_encoded_on_consonant", "from": token, "to": None})
            continue
        if token == ":":
            if not phones or not encode_phone(phones[-1].ipa)[31]:
                raise FrontendError(f"Unattached native vowel length in {word!r}/{raw!r}")
            phones[-1] = replace(phones[-1], ipa=phones[-1].ipa.rstrip("ː") + "ː")
            audit.append({"rule": "native_virtual_vowel_length", "from": token, "to": phones[-1].ipa})
            continue
        long = token.endswith(":")
        base = token[:-1] if long else token
        if base not in names:
            raise FrontendError(f"Unmapped {language} native phone {token!r} in {word!r}; raw={raw!r}")
        ipa = names[base] + ("ː" if long else "")
        x = encode_phone(ipa)
        phones.append(_Phone(ipa, pending if x[31] else 0., False))
        original_names.append(base)
        if x[31]:
            pending = 0.
    if pending or not phones or not any(encode_phone(p.ipa)[31] for p in phones):
        raise FrontendError(f"Missing nucleus or unattached stress in {word!r}/{raw!r}")
    if language == "et-EE":
        for i, (p, name) in enumerate(zip(phones, original_names)):
            # p/t/k contrast with b/d/g by quantity, not voicedness. Encode
            # intervocalic / sonorant+stop+vowel fortis as a long segment. The
            # Q2/Q3 contrast is *separate*, never inferred by this rule. The
            # control symbol ˑ gives length coordinate .5, versus explicit
            # double-stop ː = 1. This is a relative model encoding, not a
            # claimed absolute IPA duration or fixed millisecond target.
            previous = encode_phone(phones[i - 1].ipa) if i else None
            following = encode_phone(phones[i + 1].ipa) if i + 1 < len(phones) else None
            if (name in ("p", "t", "k", "t^", "f", "S") and "ː" not in p.ipa
                    and previous is not None and following is not None
                    and (previous[31] or _core(phones[i - 1].ipa) in ("m", "n", "ŋ", "l", "r", "v"))
                    and following[31]):
                phones[i] = replace(p, ipa=p.ipa + "ˑ")
                audit.append({"rule": "et_medial_fortis_relative_length", "from": p.ipa, "to": phones[i].ipa})
    merged, i = [], 0
    while i < len(phones):
        p = phones[i]
        if i + 1 < len(phones):
            q = phones[i + 1]
            px, qx = encode_phone(p.ipa), encode_phone(q.ipa)
            same = _core(p.ipa) == _core(q.ipa)
            if same and not px[31] and not qx[31]:
                target = _core(p.ipa) + ("ʲ" if "ʲ" in p.ipa or "ʲ" in q.ipa else "") + "ː"
                merged.append(replace(p, ipa=target))
                audit.append({"rule": "adjacent_identical_consonants_to_geminate", "from": [p.ipa, q.ipa], "to": target})
                i += 2
                continue
            if language == "et-EE" and px[31] and qx[31]:
                pair = _core(p.ipa) + _core(q.ipa)
                if same and len(_core(p.ipa)) == 1 and (px[22] or qx[22]):
                    target = _core(p.ipa) + "ː"
                    merged.append(_Phone(target, max(p.stress or 0., q.stress or 0.), False))
                    audit.append({"rule": "et_split_double_vowel_to_one_long_nucleus", "from": [p.ipa, q.ipa], "to": target})
                    i += 2
                    continue
                allowed = _ET_DIPHTHONGS if p.stress == 1. else _ET_UNSTRESSED_DIPHTHONGS
                spelling = pair.translate(str.maketrans("ɑɤæøy", "aõäöü"))
                if pair in allowed and spelling in word and not q.stress:
                    merged.append(_Phone(pair, p.stress, False))
                    audit.append({"rule": "et_adjacent_vowel_diphthong_candidate", "from": [p.ipa, q.ipa], "to": pair})
                    i += 2
                    continue
        merged.append(p)
        i += 1
    phones = merged
    if language == "fi-FI":
        first = next(i for i, p in enumerate(phones) if encode_phone(p.ipa)[31])
        # Finnish primary word stress is initial. Secondary stress remains a
        # backend candidate, including compound constituents and exceptions.
        if phones[first].stress != 1.:
            audit.append({"rule": "fi_initial_primary_stress", "from": phones[first].stress, "to": 1.})
        phones[first] = replace(phones[first], stress=1., stress_known=True)
        for i, p in enumerate(phones):
            if i != first and p.stress == 1.:
                phones[i] = replace(p, stress=.5, stress_known=False)
                audit.append({"rule": "fi_noninitial_backend_primary_to_secondary_candidate", "phone": i})
    return phones, audit


def _parse_override(entry, language):
    entry = {"phones": entry} if isinstance(entry, list) else dict(entry)
    if not isinstance(entry.get("phones"), list) or not entry["phones"]:
        raise FrontendError("A pronunciation override needs a nonempty phones list")
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
        phones.append(_Phone(ipa, pending if x[31] else 0., bool(entry.get("stress_known", True))))
        if x[31]:
            pending = 0.
    if pending or not any(encode_phone(p.ipa)[31] for p in phones):
        raise FrontendError("Override has no nucleus or unattached stress")
    if not any(p.stress for p in phones):
        phones = [replace(p, stress_known=False) for p in phones]
    quantities = entry.get("foot_quantities", [])
    if not isinstance(quantities, list) or (quantities and language != "et-EE"):
        raise FrontendError("foot_quantities is an ET-only list")
    feet = _foot_groups(_syllables(phones))
    seen = set()
    for quantity in quantities:
        if (not isinstance(quantity, dict) or type(quantity.get("foot")) is not int
                or type(quantity.get("grade")) is not int or quantity["grade"] not in (1, 2, 3)
                or not 0 <= quantity["foot"] < len(feet) or quantity["foot"] in seen
                or not isinstance(quantity.get("source"), str) or not quantity["source"].strip()):
            raise FrontendError("ET quantity needs unique zero-based foot, grade 1/2/3 and source")
        seen.add(quantity["foot"])
    if entry.get("pitch_accent") is not None or entry.get("tone_contour") is not None:
        raise FrontendError("Finnic frontend does not certify lexical F0 contours from text")
    return phones, quantities


def _foot_groups(syllables):
    """Text grouping heuristic; not a claim of gold phonological foot parsing."""
    starts = [0] + [i for i, (_, _, stress, _) in enumerate(syllables) if i and stress > 0.]
    return list(zip(starts, starts[1:] + [len(syllables)]))


class FinnicFrontend:
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
            raise FrontendError("External text lexicons require path and expected SHA256")
        if lexicon_path:
            raw = Path(lexicon_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != lexicon_sha256:
                raise FrontendError("External text lexicon SHA256 mismatch")
            document = json.loads(raw)
            if (document.get("schema") != "finnic_pronunciation_lexicon_v1"
                    or document.get("language") != self.language or not document.get("provenance")
                    or not isinstance(document.get("entries"), dict)):
                raise FrontendError("Invalid Finnic lexicon schema, language or provenance")
            self.lexicon = document["entries"]
            self.lexicon_info = {"sha256": lexicon_sha256, "provenance": document["provenance"], "entries": len(self.lexicon)}
        for collection in (self.lexicon, self.overrides):
            for word, entry in collection.items():
                if not isinstance(word, str) or word != unicodedata.normalize("NFC", word).lower():
                    raise FrontendError("Lexicon/override keys must be NFC lowercase words")
                _parse_override(entry, self.language)
        self.backend = EspeakBackend(library=espeak_library, data=espeak_data, binary=espeak_binary,
                                     required_version=required_espeak_version)
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
        implementation = _implementation_hashes()
        config = {k: v for k, v in self._config.items() if k != "lexicon_path"}
        config.update(frontend_version=FRONTEND_VERSION, feature_adapter_version=FEATURE_ADAPTER_VERSION,
                      contract_version=CONTRACT_VERSION, context_feature_names=CONTEXT_NAMES,
                      backend_version=self.backend_info["version"],
                      backend_data_sha=self.backend_info.get("data_identity_sha256"),
                      implementation_sha256=implementation)
        return {"semantic_identity_sha256": _stable_hash(config), "semantic_config": config,
                "backend": self.backend_info, "lexicon": self.lexicon_info,
                "source_sha256": implementation["eu_finnic.py"], "implementation_sha256": implementation}

    def encode(self, text, *, utterance_id=None):
        text = re.sub(r"-{2,}", "—", text) if isinstance(text, str) else text
        normalized, tokens = _tokenize(text, self.language)
        words = []
        for wi, token in enumerate(tokens):
            word = token.group()
            invalid = set(word) - _ALPHABET[self.language] - set("'-")
            if invalid and word not in self.overrides and word not in self.lexicon:
                raise FrontendError(f"Unsupported {self.language} spelling {word!r}: {sorted(invalid)!r}")
            before = normalized[tokens[wi - 1].end() if wi else 0:token.start()]
            after = normalized[token.end():tokens[wi + 1].start() if wi + 1 < len(tokens) else len(normalized)]
            quantities = []
            if word in self.overrides or word in self.lexicon:
                explicit = word in self.overrides
                phones, quantities = _parse_override(self.overrides[word] if explicit else self.lexicon[word], self.language)
                raw, native, repairs = None, None, []
                source = "explicit_text_pronunciation_override" if explicit else "sha_bound_text_lexicon"
            else:
                raw = self.backend.phonemize(word, LANGUAGES[self.language])
                native = _native_names(self.backend, word, LANGUAGES[self.language])
                phones, repairs = _parse_native(native, word, self.language)
                source = "espeak_ng_1.52_candidate_with_finnic_rules"
            nuclei = [p for p in phones if encode_phone(p.ipa)[31]]
            words.append({"word_index": wi, "normalized_word": word, "text_start": token.start(), "text_end": token.end(),
                          "punctuation_before": before, "punctuation_after": after,
                          "boundary_before": _strength(before, self.language), "boundary_after": _strength(after, self.language),
                          "phones": [p.ipa for p in phones], "ipa": " ".join(p.ipa for p in phones),
                          "raw_backend_ipa": raw, "raw_backend_phoneme_names": native,
                          "_phones": phones, "_quantities": quantities,
                          "pronunciation": {"source": source, "repairs": repairs, "lexicon_covered": word in self.lexicon,
                                            "stress_known": all(p.stress_known for p in nuclei),
                                            "primary_stress_known": any(p.stress == 1. and p.stress_known for p in nuclei),
                                            "pitch_accent_known": False, "quantity_known": bool(quantities),
                                            "quantity_source": "explicit_text_resource" if quantities else "unknown",
                                            "stress_source": "fi_initial_rule_secondary_candidate" if source.startswith("espeak") and self.language == "fi-FI" else source}})
        identity = hashlib.sha256((str(utterance_id or "") + "\0" + self.language + "\0" + normalized).encode()).hexdigest()[:20]
        ipa, word_indices, contexts = ["sil"], [-1], [np.zeros(CONTEXT_DIM, np.float32)]
        foot_indices, syllable_indices, pauses, boundaries = [-1], [-1], [], []
        total = sum(len(w["phones"]) for w in words)
        position = global_foot = global_syllable = 0
        question, end = [False] * len(words), len(words)
        for wi in range(len(words) - 1, -1, -1):
            if words[wi]["boundary_after"] >= .7 or wi == len(words) - 1:
                end = wi
            question[wi] = "?" in words[end]["punctuation_after"]
        for wi, word in enumerate(words):
            phones, quantities = word.pop("_phones"), word.pop("_quantities")
            syllables = _syllables(phones)
            feet = _foot_groups(syllables)
            qmap = {q["foot"]: q for q in quantities}
            rows = np.zeros((len(phones), CONTEXT_DIM), np.float32)
            word["syllables"], word["feet"] = [], []
            local_foot = np.zeros(len(phones), np.int64)
            local_syllable = np.zeros(len(phones), np.int64)
            for fi, (sstart, sstop) in enumerate(feet):
                word["feet"].append({"foot_index": global_foot + fi, "syllable_start": sstart, "syllable_end": sstop,
                                      "head_stress_candidate": syllables[sstart][2], "grouping_known": False,
                                      "source": "text_stress_to_next_stress_heuristic",
                                      "quantity": qmap[fi]["grade"] if fi in qmap else None,
                                      "quantity_known": fi in qmap, "quantity_source": qmap[fi]["source"] if fi in qmap else None})
                for si in range(sstart, sstop):
                    start, stop, stress, known = syllables[si]
                    nuclei = [i for i in range(start, stop) if encode_phone(phones[i].ipa)[31]]
                    if len(nuclei) != 1:
                        raise FrontendError("Finnic syllable must contain exactly one nucleus")
                    rows[start:stop, 12:14] = (stress, float(known))
                    rows[start, 20] = rows[stop - 1, 21] = 1.
                    for pi in range(start, stop):
                        rows[pi, 22:28] = (si / max(1, len(syllables) - 1), min(len(syllables) / 8., 2.),
                                           (pi - start) / max(1, stop - start - 1), float(si == sstart),
                                           (si - sstart) / max(1, sstop - sstart - 1), min((sstop - sstart) / 4., 2.))
                        if fi in qmap:
                            rows[pi, 27 + qmap[fi]["grade"]] = 1.
                            rows[pi, 31] = 1.
                        rows[pi, 32:] = (float(si == len(syllables) - 1 and (word["boundary_after"] > 0. or wi == len(words) - 1)), float(question[wi]))
                    local_foot[start:stop], local_syllable[start:stop] = global_foot + fi, global_syllable + si
                    word["syllables"].append({"syllable_index": global_syllable + si, "phone_start": start, "phone_end": stop,
                                               "nucleus_phone_index": nuclei[0], "stress": stress, "stress_known": known,
                                               "foot_index": global_foot + fi, "foot_head": si == sstart})
            global_foot += len(feet)
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
                foot_indices.append(int(local_foot[pi])); syllable_indices.append(int(local_syllable[pi]))
                position += 1
            if wi + 1 < len(words):
                boundary = {"left_word_index": wi, "right_word_index": wi + 1, "left_phone_unit_index": len(ipa) - 1,
                            "strength": word["boundary_after"], "punctuation": word["punctuation_after"], "pause_unit_index": None}
                if self.pause_policy == "word" or (self.pause_policy == "punctuation" and word["boundary_after"] > 0):
                    row = np.zeros(CONTEXT_DIM, np.float32)
                    row[4:8] = (wi / max(1, len(words) - 1), position / max(1, total - 1), min(total / 200., 2.), min(len(words) / 50., 2.))
                    row[8:10] = word["boundary_after"]
                    boundary["pause_unit_index"] = len(ipa)
                    pauses.append({"unit_index": len(ipa), "kind": "optional_word_pause", "after_word_index": wi, "strength": word["boundary_after"]})
                    ipa.append("sil"); contexts.append(row); word_indices.append(-1)
                    foot_indices.append(-1); syllable_indices.append(-1)
                boundaries.append(boundary)
        ipa.append("sil"); contexts.append(np.zeros(CONTEXT_DIM, np.float32)); word_indices.append(-1)
        foot_indices.append(-1); syllable_indices.append(-1)
        x, c = np.stack([encode_phone(p) for p in ipa]), np.stack(contexts)
        c[0, 10] = c[-1, 11] = 1.
        c[-1, 4:6] = 1.
        c[0, 6:8] = c[-1, 6:8] = (min(total / 200., 2.), min(len(words) / 50., 2.))
        pause = x[:, 0].astype(bool)
        edge = np.zeros(len(ipa), bool); edge[0] = edge[-1] = True
        ident = self.identity()
        return {"phonetic_features": x, "context_features": c, "valid_mask": np.ones(len(ipa), bool),
                "edge_sil_mask": pause.copy(), "outer_edge_mask": edge, "pause_mask": pause,
                "word_indices": np.asarray(word_indices, np.int64), "foot_indices": np.asarray(foot_indices, np.int64),
                "syllable_indices": np.asarray(syllable_indices, np.int64), "ipa": ipa, "phoneme_tokens": ipa.copy(),
                "phonetic_feature_names": list(FEATURE_NAMES), "context_feature_names": CONTEXT_NAMES,
                "contract_version": CONTRACT_VERSION, "language": self.language, "normalized_text": normalized,
                "words": words, "word_boundaries": boundaries,
                "metadata": {"frontend_version": FRONTEND_VERSION, "feature_adapter_version": FEATURE_ADAPTER_VERSION,
                             "context_dim": CONTEXT_DIM, "config": self.get_config(), "backend": self.backend_info,
                             "lexicon": self.lexicon_info, "semantic_identity_sha256": ident["semantic_identity_sha256"],
                             "utterance_id": utterance_id, "text_only": True, "no_reference_audio_or_alignment": True,
                             "syllabification": "one_consonant_onset_heuristic_single_unit_geminates",
                             "foot_grouping": "within_word_initial_then_candidate_stress_to_next_stress",
                             "foot_grouping_is_gold": False, "optional_mask_semantics": "all_silence_units", "pause_units": pauses,
                             "limitations": ["et_unwritten_quantity_unknown_without_text_resource", "et_backend_stress_and_palatalization_are_candidates",
                                             "fi_secondary_stress_is_a_candidate", "text_syllable_and_foot_boundaries_are_heuristic",
                                             "word_internal_sandhi_only", "no_text_derived_lexical_F0_contour"]}}

    prepare = encode


def encode(text, language, *, utterance_id=None, **config):
    return FinnicFrontend(language, **config).encode(text, utterance_id=utterance_id)


def coverage_probe(texts, language, **config):
    """Text-only coverage; every rejection and every uncertified ET foot counted."""
    frontend = FinnicFrontend(language, **config)
    phones, sources, warnings, failures = Counter(), Counter(), Counter(), []
    accepted = words = known_stress = known_primary_stress = known_quantity = feet = lexicon_words = 0
    max_units = 0
    for i, text in enumerate(texts):
        try:
            item = frontend.encode(text, utterance_id=f"coverage_{i}")
        except FrontendError as exc:
            failures.append({"row": i, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "reason": str(exc)})
            continue
        accepted += 1
        max_units = max(max_units, len(item["ipa"]))
        phones.update(item["ipa"])
        for word in item["words"]:
            words += 1
            info = word["pronunciation"]
            known_stress += info["stress_known"]
            known_primary_stress += info["primary_stress_known"]
            lexicon_words += info["lexicon_covered"]
            feet += len(word["feet"])
            known_quantity += sum(f["quantity_known"] for f in word["feet"])
            sources[info["source"]] += 1
            for repair in info["repairs"]:
                warnings.update([repair.get("warning", repair.get("rule"))])
    return {"language": frontend.language, "accepted_utterances": accepted, "rejected_utterances": len(failures),
            "words": words, "stress_known_words": known_stress, "primary_stress_known_words": known_primary_stress,
            "quantity_applicable_feet": feet if frontend.language == "et-EE" else 0,
            "quantity_known_feet": known_quantity,
            "quantity_unknown_feet": feet - known_quantity if frontend.language == "et-EE" else 0, "lexicon_covered_words": lexicon_words,
            "lexicon_oov_words": words - lexicon_words, "max_units": max_units,
            "phone_counts": dict(sorted(phones.items())), "source_counts": dict(sources),
            "rule_and_warning_counts": dict(warnings), "failures": failures,
            "frontend_identity": frontend.identity(), "reads_reference_audio": False}
