"""Auditable hu/mt/el/ga text frontends for the unchanged 32 + 22 contract.

No neural parameters or reference audio are used here.  Hungarian, Modern Greek
and Irish use a pinned external eSpeak NG; Maltese deliberately does not use its
incomplete 1.52 rules.  See the experiment's docs/FRONTEND.md for limitations.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import unicodedata

import numpy as np

from .phonetic_features import FEATURE_NAMES, INVENTORY, PLACES, PhoneticSegment, encode_segment
from .universal import CONTEXT_NAMES, CONTRACT_VERSION


FRONTEND_VERSION = "eu_four_frontend_v1"
FEATURE_ADAPTER_VERSION = "eu_four_articulatory32_v1"
MALTESE_RULE_VERSION = "maltese_transparent_rules_v1"
LANGUAGES = {"hu-HU": "hu", "mt-MT": "mt", "el-GR": "el", "ga-IE": "ga"}
_ALIASES = {**{x.lower(): x for x in LANGUAGES}, **{v: k for k, v in LANGUAGES.items()}}
_LOCK = threading.RLock()
_LIBRARIES: dict[str, tuple[object, str | None]] = {}


class FrontendError(ValueError):
    """A rejected input, backend failure or unmapped phone; never a silence OOV."""


def _language(value: str) -> str:
    try:
        return _ALIASES[value.lower()]
    except (KeyError, AttributeError) as exc:
        raise FrontendError(f"Unsupported language {value!r}; expected {tuple(LANGUAGES)}") from exc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Voice(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("languages", ctypes.c_char_p),
                ("identifier", ctypes.c_char_p), ("gender", ctypes.c_ubyte),
                ("age", ctypes.c_ubyte), ("variant", ctypes.c_ubyte),
                ("reserved", ctypes.c_ubyte), ("score", ctypes.c_int),
                ("spare", ctypes.c_void_p)]


class EspeakBackend:
    """Library or CLI G2P, with an explicit voice and a process-wide library lock."""

    def __init__(self, *, library=None, data=None, binary=None, required_version="1.52.0"):
        if library is not None and binary is not None:
            raise FrontendError("Choose one explicit eSpeak library or binary, not both")
        # An explicit backend selection takes precedence over all backend env vars.
        # In particular, binary=... must not silently use an ambient library.
        if library is None and binary is None:
            library = os.environ.get("EU_FOUR_ESPEAK_LIBRARY")
            binary = None if library else os.environ.get("EU_FOUR_ESPEAK_BINARY")
        data = data or os.environ.get("EU_FOUR_ESPEAK_DATA")
        self.library = None
        self.binary = None
        if not library and not binary:
            try:
                import espeakng_loader
                library, data = espeakng_loader.get_library_path(), data or espeakng_loader.get_data_path()
            except ImportError:
                library = ctypes.util.find_library("espeak-ng")
                if not library:
                    binary = shutil.which("espeak-ng")
        if library:
            key = str(Path(library).resolve()) if Path(library).exists() else str(library)
            requested_data = str(Path(data).resolve()) if data else None
            with _LOCK:
                if key in _LIBRARIES:
                    self.lib, initialized_data = _LIBRARIES[key]
                    if requested_data is not None and initialized_data != requested_data:
                        raise FrontendError("eSpeak library already initialized with a different data directory")
                else:
                    try:
                        self.lib = ctypes.CDLL(str(library))
                    except OSError as exc:
                        raise FrontendError(f"Cannot load eSpeak library: {library}") from exc
                    self.lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
                    self.lib.espeak_Initialize.restype = ctypes.c_int
                    self.lib.espeak_Info.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
                    self.lib.espeak_Info.restype = ctypes.c_char_p
                    self.lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
                    self.lib.espeak_SetVoiceByName.restype = ctypes.c_int
                    self.lib.espeak_GetCurrentVoice.restype = ctypes.POINTER(_Voice)
                    self.lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
                    self.lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
                    if self.lib.espeak_Initialize(2, 0, requested_data.encode() if requested_data else None, 0) <= 0:
                        raise FrontendError("eSpeak initialization failed")
                    actual = ctypes.c_char_p()
                    self.lib.espeak_Info(ctypes.byref(actual))
                    initialized_data = str(Path(actual.value.decode()).resolve()) if actual.value else None
                    _LIBRARIES[key] = (self.lib, initialized_data)
                actual = ctypes.c_char_p()
                self.version = self.lib.espeak_Info(ctypes.byref(actual)).decode()
                self.data = actual.value.decode() if actual.value else requested_data
            self.library = key
            self.kind = "espeak_ng_library"
        elif binary:
            self.binary = str(Path(binary).resolve()) if Path(binary).exists() else shutil.which(str(binary))
            if not self.binary:
                raise FrontendError(f"eSpeak binary does not exist: {binary}")
            self.data = str(Path(data).resolve()) if data else None
            self.kind = "espeak_ng_cli"
            info = self._run(["--version"])
            match = re.search(r"eSpeak NG text-to-speech:\s*([0-9.]+)", info)
            if not match:
                raise FrontendError(f"Cannot identify eSpeak NG version: {info!r}")
            self.version = match.group(1)
            if not self.data:
                match = re.search(r"Data at:\s*(.+)", info)
                self.data = match.group(1).strip() if match else None
        else:
            raise FrontendError("eSpeak NG not found; set EU_FOUR_ESPEAK_LIBRARY/DATA or EU_FOUR_ESPEAK_BINARY")
        if required_version and self.version != required_version:
            raise FrontendError(f"eSpeak version {self.version!r} differs from pinned {required_version!r}")
        self._voices_checked: set[str] = set()

    def _run(self, args, text=None):
        command = [self.binary]
        if self.data:
            command += ["--path=" + self.data]
        try:
            result = subprocess.run(command + args, input=text, text=True, capture_output=True, timeout=30, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise FrontendError(f"eSpeak command failed: {exc}") from exc
        if result.stderr.strip():
            raise FrontendError(f"eSpeak diagnostic: {result.stderr.strip()}")
        return result.stdout

    def _select_voice(self, voice):
        if self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise FrontendError(f"eSpeak voice {voice!r} is unavailable")
        current = self.lib.espeak_GetCurrentVoice()
        raw = current.contents.languages if current else b""
        actual = raw[1:].decode() if raw else ""
        if actual != voice:
            raise FrontendError(f"Requested {voice!r}, eSpeak selected {actual!r}; fallback forbidden")

    @lru_cache(maxsize=65536)
    def phonemize(self, word: str, voice: str) -> str:
        if voice not in ("hu", "el", "ga", "lv", "lt", "fi", "et", "sv"):
            raise FrontendError("This backend is enabled only for hu/el/ga/lv/lt/fi/et rules; adapters validate their phones")
        if self.library:
            with _LOCK:
                self._select_voice(voice)
                data = ctypes.create_string_buffer(word.encode("utf-8"))
                pointer = ctypes.c_void_p(ctypes.addressof(data))
                clauses = []
                for _ in range(1000):
                    if not pointer.value:
                        break
                    previous = pointer.value
                    value = self.lib.espeak_TextToPhonemes(ctypes.byref(pointer), 1, 2 | (ord("_") << 8))
                    clauses.append(value.decode("utf-8") if value else "")
                    if pointer.value == previous:
                        raise FrontendError("eSpeak text cursor did not advance")
                else:
                    raise FrontendError("eSpeak clause limit exceeded")
                raw = " ".join(clauses).strip()
        else:
            if voice not in self._voices_checked:
                listing = self._run(["--voices=" + voice])
                if not any(len(parts := line.split()) >= 2 and parts[1] == voice for line in listing.splitlines()):
                    raise FrontendError(f"Requested eSpeak voice {voice!r} missing from --voices")
                self._voices_checked.add(voice)
            raw = self._run(["-q", "-v", voice, "--ipa", "--sep=_", "--stdin"], word + "\n").strip()
        if not raw or re.search(r"\([^)]*\)", raw):
            raise FrontendError(f"Empty pronunciation or language switch for {voice}/{word!r}: {raw!r}")
        return raw

    def provenance(self, voice):
        root = Path(self.data) if self.data else None
        hashes = {}
        if root and root.is_dir():
            for rel in (f"{voice}_dict", "phondata", "phonindex", "phontab", "intonations"):
                path = root / rel
                if path.is_file():
                    hashes[rel] = _sha(path)
            for path in sorted((root / "lang").rglob(voice)):
                if path.is_file():
                    hashes[str(path.relative_to(root))] = _sha(path)
        executable = self.library or self.binary
        return {"kind": self.kind, "version": self.version, "voice": voice,
                "library": self.library, "binary": self.binary, "data": self.data,
                "executable_sha256": _sha(Path(executable)) if Path(executable).is_file() else None,
                "data_sha256": hashes,
                "data_identity_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


_EXTRA = {
    "r": ("approximant", "alveolar", True), "ɾ": ("approximant", "alveolar", True),
    "c": ("stop", "palatal", False), "ɟ": ("stop", "palatal", True),
    "ʔ": ("stop", "glottal", False), "ʝ": ("fricative", "palatal", True),
    "ɣ": ("fricative", "velar", True), "β": ("fricative", "bilabial", True),
    "ħ": ("fricative", "uvular", False), "χ": ("fricative", "uvular", False),
    "ɦ": ("fricative", "glottal", True), "ʋ": ("approximant", "labiodental", True),
    "dz": ("affricate", "alveolar", True), "ɱ": ("nasal", "labiodental", True),
    "ɫ": ("approximant", "alveolar", True),
}
_BASES = dict(INVENTORY)
for _ipa, (_manner, _place, _voiced) in _EXTRA.items():
    _BASES[_ipa] = PhoneticSegment(_ipa, _manner, _place, _voiced,
                                   lateral=_ipa == "ɫ", rhotic=_ipa in ("r", "ɾ"))
_MARKS = "ːˑʲˠʰ̩̯̪̥̬̃̆˞̝̞̟̠"


def _canonical_ipa(ipa):
    # IPA ç is a base segment, although Unicode NFD treats it as c + cedilla.
    return unicodedata.normalize("NFD", ipa).replace("ç", "ç").replace("g", "ɡ").replace("͡", "").replace("͜", "")


@lru_cache(maxsize=4096)
def _feature_tuple(ipa: str):
    canonical = _canonical_ipa(ipa)
    if "ˈ" in canonical or "ˌ" in canonical:
        raise FrontendError("Stress must be in context, not a canonical phone")
    base_name = canonical.translate(str.maketrans("", "", _MARKS))
    if base_name in _BASES:
        segment = _BASES[base_name]
    elif 2 <= len(base_name) <= 3 and all(x in _BASES and _BASES[x].manner == "vowel" for x in base_name):
        segment = PhoneticSegment(base_name, "vowel", voiced=True,
                                  vowel_start=_BASES[base_name[0]].vowel_start,
                                  vowel_end=_BASES[base_name[-1]].vowel_start)
    else:
        raise FrontendError(f"Unmapped canonical IPA phone {ipa!r}")
    consonant = segment.manner not in ("vowel", "silence")
    segment = replace(segment, ipa=ipa, long="ː" in canonical or segment.long,
                      geminate=consonant and "ː" in canonical,
                      aspirated="ʰ" in canonical or segment.aspirated,
                      nasalized="̃" in canonical or segment.nasalized,
                      syllabic="̩" in canonical or segment.syllabic,
                      rhotic="˞" in canonical or segment.rhotic)
    x = encode_segment(segment)
    if "̯" in canonical:
        x[31] = 0.  # Explicit non-syllabic offglides must not create a nucleus.
    if "̥" in canonical:
        x[17] = 0.
    if "̬" in canonical:
        x[17] = 1.
    if "̪" in canonical:
        x[7:17] = 0.
        x[7 + PLACES.index("dental")] = 1.
    for mark, place in (("ʲ", "palatal"), ("ˠ", "velar")):
        if mark in canonical:
            x[7 + PLACES.index(place)] = 1.
    if base_name == "ɫ":
        x[7 + PLACES.index("velar")] = 1.
    if base_name == "ħ":
        x[7 + PLACES.index("glottal")] = 1.  # Documented pharyngeal surrogate; no 33rd feature.
    if "ˑ" in canonical:
        x[22] = .5
    # Continuous vowel coordinates admit these small, explicitly phonetic shifts.
    for mark, offset, delta in (("̝", 0, -.1), ("̞", 0, .1), ("̟", 1, -.1), ("̠", 1, .1)):
        if mark in canonical and segment.manner == "vowel":
            x[24 + offset] = np.clip(x[24 + offset] + delta, 0, 1)
            x[27 + offset] = np.clip(x[27 + offset] + delta, 0, 1)
    return tuple(float(v) for v in x)


def encode_phone(ipa: str) -> np.ndarray:
    """Encode a segmented, stress-free phone; unsupported symbols raise."""
    return np.asarray(_feature_tuple(ipa), dtype=np.float32)


@dataclass(frozen=True)
class _Phone:
    ipa: str
    stress: float | None = 0.
    stress_known: bool = True


def _parse_espeak(raw: str, language: str):
    phones, audit = [], []
    pending = 0.
    for token in re.split(r"[_\s]+", raw):
        if not token:
            continue  # eSpeak's explicit zero-IPA linking marker /;/ and clause separators.
        if "ˈ" in token or "ˌ" in token:
            pending = 1. if "ˈ" in token else .5
            token = token.replace("ˈ", "").replace("ˌ", "")
        if not token:
            continue
        token = _canonical_ipa(token)
        if language == "ga-IE" and token == "A":
            # ph_irish phoneme 0 incorrectly declares ASCII "ipa A" in eSpeak 1.52.
            token = "a"
            audit.append("espeak_irish_ascii_A_to_short_a")
        if language == "ga-IE" and token == "ʲ" and not phones:
            # ph_irish /;/ is a linking j, including after a silent initial fh.
            token = "j"
            audit.append("espeak_irish_initial_standalone_palatal_marker_to_linking_j")
        try:
            features = encode_phone(token)
        except FrontendError as exc:
            raise FrontendError(f"{language}: {exc}; raw={raw!r}") from exc
        stress = pending if features[31] else 0.
        if features[31]:
            pending = 0.
        phones.append(_Phone(token, stress))
    if not phones:
        raise FrontendError(f"No lexical phones in {raw!r}")
    if pending:
        raise FrontendError(f"Unattached stress marker in {raw!r}")
    if language in ("hu-HU", "el-GR"):
        # eSpeak sometimes represents an orthographic geminate as C_C, sometimes Cː.
        merged = []
        for phone in phones:
            if merged and phone.ipa == merged[-1].ipa and not encode_phone(phone.ipa)[31]:
                if language == "hu-HU":
                    merged[-1] = replace(merged[-1], ipa=phone.ipa + "ː")
                    audit.append("hungarian_adjacent_identical_consonants_to_geminate")
                else:
                    audit.append("standard_greek_double_spelling_to_single_consonant")
            else:
                merged.append(phone)
        phones = merged
    return phones, audit


_IRISH_C = {
    "b": {"b"}, "bh": {"v", "w"}, "c": {"k", "c"}, "ch": {"x", "ç"},
    "d": {"d", "dʒ"}, "dh": {"ɣ", "j", "ɡ", "x"}, "f": {"f"}, "fh": {"h"},
    "g": {"ɡ"}, "gh": {"ɣ", "j", "ɡ", "x"}, "h": {"h"}, "l": {"l"},
    "m": {"m"}, "mh": {"v", "w"}, "n": {"n", "ŋ", "ɲ"}, "p": {"p"},
    "ph": {"f"}, "r": {"r", "ɾ"}, "s": {"s", "ʃ"}, "sh": {"h"},
    "t": {"t", "c", "tʃ"}, "th": {"h"},
    "mb": {"b"}, "gc": {"ɡ"}, "nd": {"d", "dʒ"}, "ng": {"ŋ", "ɡ"},
    "bp": {"b"}, "dt": {"d", "dʒ"}, "bhf": {"v", "w"}, "ts": {"t", "tʃ"},
}
_IRISH_VOWELS = set("aeiouáéíóú")


def _irish_secondary(word, phones, *, v2=False):
    """Fill omitted broad/slender features using monotonic grapheme/phone matches.

    Explicit eSpeak palatalization and affricates win.  Unmatched graphemes/phones
    are recorded, not assigned a guessed spelling index or converted to English.
    """
    # v2 follows the pinned backend's actual phones. This preserves backend
    # contrasts; it does not certify a dialect or repair the backend's phonology.
    consonants = _IRISH_C
    optional_zero = {"fh"}
    if v2:
        consonants = {**_IRISH_C, "r": {"r", "ɾ", "ɹ"},
                      "ll": {"l"}, "nn": {"n", "ŋ", "ɲ"}, "rr": {"r", "ɾ", "ɹ"},
                      "mb": {"b", "m"}, "nd": {"d", "dʒ", "n"}}
        optional_zero = {"fh", "bh", "dh", "gh", "mh", "th"}
    graphemes = []
    i = 0
    while i < len(word):
        if word[i] in _IRISH_VOWELS or word[i] in "'-":
            i += 1
            continue
        mutations = {"mb", "gc", "nd", "ng", "bp", "dt", "bhf", "ts"}
        choices = [g for g in consonants if word.startswith(g, i) and (g not in mutations or i == 0)]
        if not choices:
            i += 1
            continue
        g = max(choices, key=len)
        before = word[i - 1] if i else ""
        after = word[i + len(g)] if i + len(g) < len(word) else ""
        adjacent = after if after in _IRISH_VOWELS else before if before in _IRISH_VOWELS else ""
        graphemes.append((g, adjacent in "eiéí" if adjacent else None))
        i += len(g)
    result = list(phones)
    cursor = 0
    changed, unmatched, matched_indices = 0, [], set()
    for grapheme, slender in graphemes:
        candidates = []
        for j in range(cursor, len(result)):
            core = result[j].ipa.translate(str.maketrans("", "", _MARKS))
            if core in consonants[grapheme]:
                candidates.append(j)
                break
        if not candidates:
            if grapheme not in optional_zero:
                unmatched.append(grapheme)
            continue
        j = candidates[0]
        matched_indices.add(j)
        cursor = j + 1
        core = result[j].ipa.translate(str.maketrans("", "", _MARKS))
        if slender is None or core in ("c", "ɟ", "j", "ç", "ʃ", "tʃ", "dʒ", "h", "w", "ŋ", "ɲ"):
            continue
        if "ʲ" not in result[j].ipa and "ˠ" not in result[j].ipa:
            result[j] = replace(result[j], ipa=result[j].ipa + ("ʲ" if slender else "ˠ"))
            changed += 1
    orphan_phones = [p.ipa for i, p in enumerate(phones)
                     if i not in matched_indices and not encode_phone(p.ipa)[1] and p.ipa != "j"]
    if unmatched or orphan_phones:
        # Ambiguous word-to-phone alignment is not enough evidence to change phones.
        return phones, {"irish_secondary_articulation_added": 0,
                        "irish_secondary_status": "incomplete_mapping_retained_backend",
                        "irish_unmatched_consonant_graphemes": unmatched,
                        "irish_unmatched_consonant_phones": orphan_phones}
    return result, {"irish_secondary_articulation_added": changed,
                    "irish_secondary_status": "complete_monotonic_mapping",
                    "irish_unmatched_consonant_graphemes": unmatched,
                    "irish_unmatched_consonant_phones": orphan_phones}


_MT_V = {"a": "ɐ", "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ", "ie": "ɪː"}
_MT_C = {"b": "b", "ċ": "tʃ", "d": "d", "f": "f", "ġ": "dʒ", "g": "ɡ", "ħ": "ħ",
         "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "q": "ʔ",
         "r": "r", "s": "s", "t": "t", "v": "v", "w": "w", "x": "ʃ", "z": "ts", "ż": "z"}
_MT_DEVOICE = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s", "dʒ": "tʃ", "dz": "ts", "ʒ": "ʃ"}
_MT_VOICE = {v: k for k, v in _MT_DEVOICE.items()}
_MT_ACCENTS = str.maketrans("àèìòùáéíóú", "aeiouaeiou")


def _maltese(word: str):
    """Conservative spelling rules; lexical stress/quantity exceptions stay visible."""
    graphemes, accents = [], []
    i = 0
    while i < len(word):
        if word[i] in "'-":
            graphemes.append(word[i]); accents.append(False); i += 1
            continue
        g = next((x for x in ("għ", "ie") if word.startswith(x, i)), word[i])
        accent = g.translate(_MT_ACCENTS) != g
        g = g.translate(_MT_ACCENTS)
        if g not in _MT_V and g not in _MT_C and g not in ("għ", "h"):
            raise FrontendError(f"Maltese grapheme {word[i:i + len(g)]!r} unsupported in {word!r}; no foreign-word fallback")
        graphemes.append(g); accents.append(accent); i += len(g)
    # Punctuation inside orthographic clitics has no acoustic pause.
    gs = [g for g in graphemes if g not in "'-"]
    ac = [a for g, a in zip(graphemes, accents) if g not in "'-"]
    values = []
    explicit_stress = []
    ambiguous = []
    for j, g in enumerate(gs):
        previous = gs[j - 1] if j else ""
        following = gs[j + 1] if j + 1 < len(gs) else ""
        if g in _MT_V:
            value = _MT_V[g]
            if previous in ("għ", "h") or following in ("għ", "h"):
                value = value.rstrip("ː") + "ː"
            if g == "i" and following in ("ħ", "q"):
                value = "iː"
            values.append(value)
            if ac[j]:
                explicit_stress.append(len(values) - 1)
        elif g in ("għ", "h"):
            if j == len(gs) - 1 or (g == "għ" and following == "h"):
                values.append("ħ")
            elif previous in ("i", "ie") and following in _MT_V:
                values.append("j")
            elif previous == "u" and following in _MT_V:
                values.append("w")
            # Other positions are a documented zero consonant with adjacent length.
        else:
            values.append(_MT_C[g])
            if g == "z" or g == "x":
                ambiguous.append(g)  # Loanword /dz, ʒ/ exceptions require a lexicon.
    # Same vowel across a silent għ/h is one long nucleus (e.g. lagħab).
    merged, merged_accents = [], []
    for j, value in enumerate(values):
        if merged and value == merged[-1] and "ː" in value and encode_phone(value)[31]:
            if j in explicit_stress:
                merged_accents.append(len(merged) - 1)
            continue
        if j in explicit_stress:
            merged_accents.append(len(merged))
        merged.append(value)
    values = merged
    # Diphthongs are single units; vowel + j/w before a consonant or word end.
    joined, accented = [], []
    i = 0
    while i < len(values):
        value = values[i]
        if i + 1 < len(values) and encode_phone(value)[31] and values[i + 1] in ("j", "w") and (i + 2 == len(values) or not encode_phone(values[i + 2])[31]):
            value = value.rstrip("ː") + ("ɪ" if values[i + 1] == "j" else "ʊ")
            if i in merged_accents:
                accented.append(len(joined))
            joined.append(value); i += 2
        else:
            if i in merged_accents:
                accented.append(len(joined))
            joined.append(value); i += 1
    # Regressive obstruent voicing and final devoicing, performed before gemination.
    for i in range(len(joined) - 1, -1, -1):
        nxt = joined[i + 1] if i + 1 < len(joined) else None
        if nxt is None or nxt in _MT_VOICE or nxt in ("ħ", "ʔ"):
            joined[i] = _MT_DEVOICE.get(joined[i], joined[i])
        elif nxt in _MT_DEVOICE:
            joined[i] = _MT_VOICE.get(joined[i], joined[i])
    output, output_accents = [], []
    for j, value in enumerate(joined):
        if output and value == output[-1] and not encode_phone(value)[31]:
            output[-1] += "ː"
        else:
            if j in accented:
                output_accents.append(len(output))
            output.append(value)
    nuclei = [i for i, p in enumerate(output) if encode_phone(p)[31]]
    if output and not nuclei and (word.endswith("'") or word.startswith("'")) and len(word) <= 3:
        return [_Phone(p, None, False) for p in output], {"rule_version": MALTESE_RULE_VERSION,
                "stress_source": "vowelless_clitic", "stress_known": False,
                "limitations": ["clitic_junction_not_audio_verified"]}
    if not output or not nuclei:
        raise FrontendError(f"Maltese word {word!r} has no predictable vowel nucleus")
    last = nuclei[-1]
    heavy = "ː" in output[last] or encode_phone(output[last])[30] or len(output) - last - 1 >= 2 or any("ː" in p for p in output[last + 1:])
    stress = output_accents[-1] if output_accents else last if heavy or len(nuclei) == 1 else nuclei[-2]
    # Stressed open vowels tend to be long; retain an explicit uncertainty label.
    if stress + 1 == len(output) or (stress + 2 < len(output) and not encode_phone(output[stress + 1])[31] and "ː" not in output[stress + 1] and encode_phone(output[stress + 2])[31]):
        if "ː" not in output[stress] and not encode_phone(output[stress])[30]:
            output[stress] += "ː"
    known = bool(output_accents) or len(nuclei) == 1
    phones = [_Phone(p, 1. if i == stress else 0., known) for i, p in enumerate(output)]
    return phones, {"rule_version": MALTESE_RULE_VERSION,
                    "stress_source": "written_accent" if output_accents else "monosyllable" if len(nuclei) == 1 else "syllable_weight_heuristic",
                    "stress_known": known,
                    "limitations": ["quantity_and_lexical_stress_not_lexicon_verified"] + (["x_z_loanword_ambiguity"] if ambiguous else [])}


_TOKEN = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*'?", re.UNICODE)
_GAP = set(" \t\r\n.,;:!?…—–-()[]{}\"'«»‹›··")


def _tokenize(text, language):
    if not isinstance(text, str) or not text.strip():
        raise FrontendError("Nonempty ordinary text is required")
    typography = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"', "„": '"', "‟": '"', "‐": "-", "‑": "-"})
    normalized = unicodedata.normalize("NFC", text).translate(typography).lower()
    if len(normalized) > 20000:
        raise FrontendError("Text exceeds 20,000 characters; split into sentences")
    matches = list(_TOKEN.finditer(normalized))
    cursor = 0
    for match in matches:
        unsupported = set(normalized[cursor:match.start()]) - _GAP
        if unsupported:
            raise FrontendError(f"Unsupported text symbols {sorted(unsupported)!r}; spell out numbers in the requested language")
        letters = [c for c in match.group() if c not in "'-"]
        expected = "GREEK" if language == "el-GR" else "LATIN"
        if any(expected not in unicodedata.name(c, "") for c in letters):
            raise FrontendError(f"Mixed or unsupported script in {match.group()!r} for {language}")
        cursor = match.end()
    unsupported = set(normalized[cursor:]) - _GAP
    if unsupported or not matches:
        raise FrontendError(f"No lexical text or unsupported trailing symbols {sorted(unsupported)!r}")
    return normalized, matches


def _strength(punctuation, language):
    if any(c in punctuation for c in ".!?…") or (language == "el-GR" and ";" in punctuation):
        return 1.
    if any(c in punctuation for c in ";:—–··"):
        return .7
    if "," in punctuation:
        return .4
    return 0.


def _syllables(phones):
    nuclei = [i for i, p in enumerate(phones) if encode_phone(p.ipa)[31]]
    if not nuclei:
        return [(0, len(phones), 0., False)]
    starts = [0]
    for left, right in zip(nuclei, nuclei[1:]):
        starts.append(right if right == left + 1 else right - 1)
    ends = starts[1:] + [len(phones)]
    return [(start, end, phones[nucleus].stress or 0., phones[nucleus].stress_known)
            for start, end, nucleus in zip(starts, ends, nuclei)]


# Multiword dictionary entries in the pinned eSpeak NG 1.52.0 ga_list.
# Group only these exact whitespace-separated spans; never guess word/phone
# alignment from a whole-sentence IPA string. MFA sees one lexical key per span.
_GA_V2_PHRASES = (
    "go raibh maith", "go raibh míle", "tiocfaidh ár lá", "ar ais", "is ea",
    "maigh eo", "mhaigh eo", "s é", "s í", "sinn féin",
)


@dataclass(frozen=True)
class _IrishSpan:
    text: str
    begin: int
    stop: int

    def group(self): return self.text
    def start(self): return self.begin
    def end(self): return self.stop


def _irish_phrase_spans(normalized, tokens):
    output, i = [], 0
    phrases = sorted((p.split() for p in _GA_V2_PHRASES), key=len, reverse=True)
    while i < len(tokens):
        chosen = None
        for phrase in phrases:
            end = i + len(phrase)
            if end > len(tokens) or [x.group() for x in tokens[i:end]] != phrase:
                continue
            if any(not normalized[tokens[j].end():tokens[j + 1].start()].isspace()
                   for j in range(i, end - 1)):
                continue
            chosen = end
            break
        if chosen is None:
            output.append(tokens[i]); i += 1
        else:
            output.append(_IrishSpan(" ".join(x.group() for x in tokens[i:chosen]),
                                     tokens[i].start(), tokens[chosen - 1].end()))
            i = chosen
    return output


class FourLanguageFrontend:
    def __init__(self, language, *, espeak_library=None, espeak_data=None, espeak_binary=None,
                 required_espeak_version="1.52.0", pause_policy="punctuation", pronunciation_overrides=None, irish_v2=False):
        self.language = _language(language)
        self.irish_v2 = bool(irish_v2)
        if self.irish_v2 and self.language != "ga-IE":
            raise FrontendError("irish_v2 is valid only for ga-IE")
        if pause_policy not in ("punctuation", "word", "none"):
            raise FrontendError("pause_policy must be punctuation, word, or none")
        self.pause_policy = pause_policy
        self.overrides = dict(pronunciation_overrides or {})
        for word, phones in self.overrides.items():
            if not isinstance(word, str) or word != unicodedata.normalize("NFC", word).lower() or not isinstance(phones, list) or not phones:
                raise FrontendError("Overrides require normalized word keys and nonempty IPA token lists")
            for phone in phones:
                if not isinstance(phone, str):
                    raise FrontendError("Override phones must be strings")
                encode_phone(phone.replace("ˈ", "").replace("ˌ", ""))
        self.backend = None if self.language == "mt-MT" else EspeakBackend(
            library=espeak_library, data=espeak_data, binary=espeak_binary, required_version=required_espeak_version)
        self._config = {"language": self.language, "espeak_library": espeak_library, "espeak_data": espeak_data,
                        "espeak_binary": espeak_binary, "required_espeak_version": required_espeak_version,
                        "pause_policy": pause_policy, "pronunciation_overrides": self.overrides}
        if self.irish_v2:
            self._config["irish_v2"] = True
        self.backend_info = (self.backend.provenance(LANGUAGES[self.language]) if self.backend else
                             {"kind": "local_maltese_rules", "version": MALTESE_RULE_VERSION, "voice": "mt"})

    def get_config(self):
        """JSON-compatible constructor arguments, with resolved external paths."""
        result = dict(self._config)
        if self.backend:
            result.update(espeak_library=self.backend.library, espeak_binary=self.backend.binary,
                          espeak_data=self.backend.data)
        return json.loads(json.dumps(result))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def encode(self, text, *, utterance_id=None):
        normalized, tokens = _tokenize(text, self.language)
        if self.irish_v2:
            tokens = _irish_phrase_spans(normalized, tokens)
        words = []
        for wi, token in enumerate(tokens):
            word = token.group()
            before = normalized[tokens[wi - 1].end() if wi else 0:token.start()]
            after = normalized[token.end():tokens[wi + 1].start() if wi + 1 < len(tokens) else len(normalized)]
            if word in self.overrides:
                raw = "_".join(self.overrides[word])
                phones, repairs = _parse_espeak(raw, self.language)
                info = {"source": "explicit_text_pronunciation_override", "repairs": repairs}
            elif self.language == "mt-MT":
                clitic = "'" + word if len(word) == 1 and word in "blmfsdtn" and before.endswith("'") else word
                phones, info = _maltese(clitic)
                raw = None
                info["source"] = MALTESE_RULE_VERSION
            else:
                # eSpeak reads isolated one-letter articles as letter names.  These
                # common function words have explicit lexical pronunciations here.
                single_words = {"hu-HU": {"a": "ɑ", "s": "ʃ"},
                                "el-GR": {"ο": "o", "η": "i", "ή": "i"}}
                raw = single_words.get(self.language, {}).get(word)
                if raw is None:
                    raw = self.backend.phonemize(word, LANGUAGES[self.language])
                phones, repairs = _parse_espeak(raw, self.language)
                info = {"source": "espeak_ng_1.52_lexical", "repairs": repairs}
                if word in single_words.get(self.language, {}):
                    info["source"] = "single_letter_function_word_rule"
                if self.language == "ga-IE":
                    if self.irish_v2 and " " in word:
                        # Do not invent internal word boundaries for a dictionary
                        # contraction. Preserve the backend's whole-span IPA.
                        info.update(source="espeak_ng_1.52_multiword_dictionary",
                                    irish_secondary_status="phrase_backend_retained",
                                    lexical_span_words=word.split())
                    else:
                        phones, irish = _irish_secondary(word, phones, v2=self.irish_v2)
                        info.update(irish)
                    if self.irish_v2:
                        info["irish_secondary_version"] = "ga_monotonic_v2"
            raw_phone_tokens = [p.ipa for p in phones]
            if self.language == "ga-IE":
                collapsed = []
                for pi, phone in enumerate(phones):
                    canonical = phone.ipa
                    core = canonical.translate(str.maketrans("", "", _MARKS))
                    if core in ("k", "x", "ɡ", "ɣ"):
                        canonical = canonical.replace("ˠ", "")
                    if canonical == "ŭ":
                        canonical = "u"
                    if canonical != phone.ipa:
                        collapsed.append({"from": phone.ipa, "to": canonical})
                        phones[pi] = replace(phone, ipa=canonical)
                info["canonical_allophone_merges"] = collapsed
            for p in phones:
                encode_phone(p.ipa)  # Also validate all rule-generated modifiers.
            words.append({"word_index": wi, "normalized_word": word, "text_start": token.start(), "text_end": token.end(),
                          "punctuation_before": before, "punctuation_after": after,
                          "boundary_before": _strength(before, self.language), "boundary_after": _strength(after, self.language),
                          "phones": [p.ipa for p in phones], "ipa": " ".join(p.ipa for p in phones),
                          "raw_phone_tokens_before_canonical_merges": raw_phone_tokens,
                          "raw_backend_ipa": raw, "pronunciation": info, "_phone_objects": phones})
        identity = hashlib.sha256((str(utterance_id or "") + "\0" + self.language + "\0" + normalized).encode()).hexdigest()[:20]
        all_phones, word_indices, context, pause_metadata = ["sil"], [-1], [np.zeros(22, np.float32)], []
        lexical_count = sum(len(w["phones"]) for w in words)
        lexical_position = 0
        boundaries = []
        for wi, word in enumerate(words):
            phones = word.pop("_phone_objects")
            syllables = _syllables(phones)
            local_context = np.zeros((len(phones), 22), np.float32)
            for start, end, stress, known in syllables:
                local_context[start:end, 12] = stress
                local_context[start:end, 13] = float(known)
                local_context[start, 20] = 1.
                local_context[end - 1, 21] = 1.
            word["syllables"] = [{"phone_start": a, "phone_end": b, "stress": s, "stress_known": k} for a, b, s, k in syllables]
            word["phone_stress"] = local_context[:, 12].tolist()
            word["phone_stress_known"] = local_context[:, 13].astype(bool).tolist()
            word["lexicon_key"] = f"u{identity}w{wi:04d}"
            word["phone_unit_indices"] = list(range(len(all_phones), len(all_phones) + len(phones)))
            for pi, phone in enumerate(phones):
                row = local_context[pi]
                row[:12] = [float(pi == 0), float(pi == len(phones) - 1), pi / max(1, len(phones) - 1),
                            min(len(phones) / 12., 2.), wi / max(1, len(words) - 1), lexical_position / max(1, lexical_count - 1),
                            min(lexical_count / 200., 2.), min(len(words) / 50., 2.),
                            word["boundary_after"] if pi == len(phones) - 1 else 0., word["boundary_before"] if pi == 0 else 0., 0., 0.]
                all_phones.append(phone.ipa); word_indices.append(wi); context.append(row)
                lexical_position += 1
            if wi + 1 < len(words):
                boundary = {"left_word_index": wi, "right_word_index": wi + 1, "left_phone_unit_index": len(all_phones) - 1,
                            "strength": word["boundary_after"], "punctuation": word["punctuation_after"], "pause_unit_index": None}
                if self.pause_policy == "word" or (self.pause_policy == "punctuation" and word["boundary_after"] > 0):
                    row = np.zeros(22, np.float32)
                    row[4:8] = [wi / max(1, len(words) - 1), lexical_position / max(1, lexical_count - 1),
                                min(lexical_count / 200., 2.), min(len(words) / 50., 2.)]
                    row[8:10] = word["boundary_after"]
                    boundary["pause_unit_index"] = len(all_phones)
                    pause_metadata.append({"unit_index": len(all_phones), "kind": "optional_word_pause", "after_word_index": wi,
                                           "strength": word["boundary_after"]})
                    all_phones.append("sil"); word_indices.append(-1); context.append(row)
                boundaries.append(boundary)
        all_phones.append("sil"); word_indices.append(-1); context.append(np.zeros(22, np.float32))
        x = np.stack([encode_phone(p) for p in all_phones])
        feature_names = list(FEATURE_NAMES)
        if self.irish_v2:
            # A fixed numeric type code preserves r/ɾ/ɹ while retaining 32 inputs.
            # This is not a claim of a three-way phonemic contrast in Irish.
            feature_names[20] = "rhotic_backend_type"
            for pi, phone in enumerate(all_phones):
                core = phone.translate(str.maketrans("", "", _MARKS))
                if core in ("r", "ɾ", "ɹ"):
                    x[pi, 20] = {"r": 1., "ɾ": .75, "ɹ": .5}[core]
        c = np.stack(context)
        c[0, 10] = c[-1, 11] = 1.
        c[-1, 4:6] = 1.
        c[0, 6:8] = c[-1, 6:8] = [min(lexical_count / 200., 2.), min(len(words) / 50., 2.)]
        edge = np.zeros(len(all_phones), bool); edge[0] = edge[-1] = True
        config = self.get_config()
        semantic_config = {k: v for k, v in config.items() if k not in ("espeak_library", "espeak_binary", "espeak_data")}
        frontend_version = "eu_four_frontend_ga_v2" if self.irish_v2 else FRONTEND_VERSION
        feature_adapter_version = "ga_articulatory32_v2" if self.irish_v2 else FEATURE_ADAPTER_VERSION
        semantic_config.update(frontend_version=frontend_version, feature_adapter_version=feature_adapter_version,
                               backend_version=self.backend_info["version"], backend_data_sha=self.backend_info.get("data_identity_sha256"))
        pause_mask = x[:, 0].astype(bool)
        return {"phonetic_features": x, "context_features": c, "valid_mask": np.ones(len(all_phones), bool),
                "edge_sil_mask": pause_mask.copy(), "outer_edge_mask": edge, "pause_mask": pause_mask,
                "word_indices": np.asarray(word_indices, np.int64),
                "ipa": all_phones, "phoneme_tokens": all_phones.copy(), "phonetic_feature_names": feature_names,
                "context_feature_names": CONTEXT_NAMES, "contract_version": CONTRACT_VERSION,
                "language": self.language, "normalized_text": normalized, "words": words, "word_boundaries": boundaries,
                "metadata": {"frontend_version": frontend_version, "feature_adapter_version": feature_adapter_version,
                             "config": config, "backend": self.backend_info, "utterance_id": utterance_id,
                             "text_only": True, "syllabification": "one_consonant_onset_heuristic",
                             "optional_mask_semantics": "all_silence_units",
                             "pause_units": pause_metadata, "no_reference_audio_or_alignment": True,
                             "semantic_identity_sha256": hashlib.sha256(json.dumps(semantic_config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}}


def encode(text: str, language: str, *, utterance_id=None, **config):
    """Convenience entry point; reuse FourLanguageFrontend for large corpora."""
    return FourLanguageFrontend(language, **config).encode(text, utterance_id=utterance_id)


def mfa_dictionary_entries(encoded):
    """Per-utterance word keys prevent spelling collisions; stress is context only."""
    return {w["lexicon_key"]: tuple(w["phones"]) for w in encoded["words"]}


def mfa_transcript(encoded):
    return " ".join(w["lexicon_key"] for w in encoded["words"])


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value
