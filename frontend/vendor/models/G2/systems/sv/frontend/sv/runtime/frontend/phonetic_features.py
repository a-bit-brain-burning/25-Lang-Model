"""Numeric articulatory features; no language-specific phoneme IDs or rules.

IPA lookup is deliberately a finite prototype inventory. Unsupported segments
raise instead of silently becoming an English sound. Features describe phones,
not language, spelling, stress or acoustic voicing labels.
"""
from dataclasses import dataclass
import numpy as np

VERSION = 'universal_phonetic_features_v1'
MANNERS = ('silence', 'vowel', 'stop', 'fricative', 'affricate', 'nasal', 'approximant')
PLACES = ('bilabial', 'labiodental', 'dental', 'alveolar', 'postalveolar',
          'retroflex', 'palatal', 'velar', 'uvular', 'glottal')
FEATURE_NAMES = (tuple('manner_' + x for x in MANNERS) + tuple('place_' + x for x in PLACES)
                 + ('voiced', 'aspirated', 'lateral', 'rhotic', 'nasalized', 'long', 'geminate',
                    'vowel_height_start', 'vowel_backness_start', 'vowel_round_start',
                    'vowel_height_end', 'vowel_backness_end', 'vowel_round_end', 'diphthong', 'syllabic'))
FEATURE_DIM = len(FEATURE_NAMES)

@dataclass(frozen=True)
class PhoneticSegment:
    ipa: str
    manner: str
    place: str | None = None
    voiced: bool = False
    aspirated: bool = False
    lateral: bool = False
    rhotic: bool = False
    nasalized: bool = False
    long: bool = False
    geminate: bool = False
    vowel_start: tuple[float, float, float] = (0., 0., 0.)
    vowel_end: tuple[float, float, float] | None = None
    syllabic: bool = False

def encode_segment(segment: PhoneticSegment):
    if segment.manner not in MANNERS or (segment.place is not None and segment.place not in PLACES):
        raise ValueError('Unsupported articulatory category')
    x = np.zeros(FEATURE_DIM, np.float32)
    x[MANNERS.index(segment.manner)] = 1.
    if segment.place is not None:
        x[len(MANNERS) + PLACES.index(segment.place)] = 1.
    x[17:24] = [segment.voiced, segment.aspirated, segment.lateral, segment.rhotic,
                segment.nasalized, segment.long, segment.geminate]
    end = segment.vowel_end or segment.vowel_start
    if len(segment.vowel_start) != 3 or len(end) != 3:
        raise ValueError('Vowel coordinates need height/backness/roundness')
    x[24:27], x[27:30] = segment.vowel_start, end
    x[30] = segment.vowel_end is not None and end != segment.vowel_start
    x[31] = segment.syllabic or segment.manner == 'vowel'
    if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError('Phonetic features must be finite within [0,1]')
    return x

# Height: close=0, open=1; backness: front=0, back=1; rounding: 0/1.
_VOWELS = {'i': (0.,0.,0.), 'ɪ': (.17,.17,0.), 'e': (.33,0.,0.), 'ɛ': (.67,0.,0.),
           'æ': (.83,0.,0.), 'a': (1.,0.,0.), 'ɑ': (1.,1.,0.), 'ɒ': (1.,1.,1.),
           'ʌ': (.67,.67,0.), 'ə': (.5,.5,0.), 'ɜ': (.67,.5,0.), 'ɐ': (.83,.5,0.),
           'ɔ': (.67,1.,1.), 'o': (.33,1.,1.), 'ʊ': (.17,.83,1.), 'u': (0.,1.,1.),
           'y': (0.,0.,1.), 'ʏ': (.17,.17,1.), 'ø': (.33,0.,1.), 'œ': (.67,0.,1.),
           'ɯ': (0.,1.,0.), 'ɤ': (.33,1.,0.), 'ɨ': (0.,.5,0.)}
INVENTORY = {'sil': PhoneticSegment('sil', 'silence')}
for ipa, coords in _VOWELS.items():
    INVENTORY[ipa] = PhoneticSegment(ipa, 'vowel', voiced=True, vowel_start=coords)
for ipa, start, end in [('aɪ','a','ɪ'),('aʊ','a','ʊ'),('eɪ','e','ɪ'),('oʊ','o','ʊ'),('ɔɪ','ɔ','ɪ')]:
    INVENTORY[ipa] = PhoneticSegment(ipa, 'vowel', voiced=True, vowel_start=_VOWELS[start], vowel_end=_VOWELS[end])
INVENTORY['ɝ'] = PhoneticSegment('ɝ','vowel', voiced=True, rhotic=True,vowel_start=_VOWELS['ɜ'])
_CONSONANTS = {
 'p':('stop','bilabial',False), 'b':('stop','bilabial',True),
 't':('stop','alveolar',False), 'd':('stop','alveolar',True),
 'k':('stop','velar',False), 'ɡ':('stop','velar',True),
 'f':('fricative','labiodental',False), 'v':('fricative','labiodental',True),
 'θ':('fricative','dental',False), 'ð':('fricative','dental',True),
 's':('fricative','alveolar',False), 'z':('fricative','alveolar',True),
 'ʃ':('fricative','postalveolar',False), 'ʒ':('fricative','postalveolar',True),
 'h':('fricative','glottal',False), 'x':('fricative','velar',False), 'ç':('fricative','palatal',False),
 'ʁ':('fricative','uvular',True), 'ɕ':('fricative','palatal',False), 'ʂ':('fricative','retroflex',False),
 'tʃ':('affricate','postalveolar',False), 'dʒ':('affricate','postalveolar',True),
 'ts':('affricate','alveolar',False), 'tɕ':('affricate','palatal',False), 'tʂ':('affricate','retroflex',False),
 'm':('nasal','bilabial',True), 'n':('nasal','alveolar',True), 'ŋ':('nasal','velar',True),
 'ɲ':('nasal','palatal',True),
 'ɹ':('approximant','alveolar',True), 'ɻ':('approximant','retroflex',True),
 'l':('approximant','alveolar',True), 'ʎ':('approximant','palatal',True),
 'j':('approximant','palatal',True), 'w':('approximant','bilabial',True),
 'ɥ':('approximant','palatal',True)}
for ipa, (manner, place, voiced) in _CONSONANTS.items():
    INVENTORY[ipa] = PhoneticSegment(ipa,manner,place,voiced,lateral=ipa in ('l','ʎ'),rhotic=ipa in ('ɹ','ɻ','r','ɾ'))

def segment_from_ipa(ipa):
    """Accept one already segmented IPA unit, with explicit diacritics."""
    if not isinstance(ipa,str) or not ipa:
        raise ValueError('An IPA segment string is required')
    normalized = ipa.replace('g','ɡ').replace('͡','').replace('͜','')
    modifiers = {'aspirated': 'ʰ', 'long': 'ː', 'nasalized': '̃',
                 'syllabic':'̩', 'rhotic':'˞'}
    flags = {key: mark in normalized for key,mark in modifiers.items()}
    for mark in modifiers.values():
        normalized = normalized.replace(mark,'')
    if normalized not in INVENTORY:
        raise ValueError('Unsupported segmented IPA phone: ' + ipa)
    base = INVENTORY[normalized]
    flags = {key: value or getattr(base,key) for key,value in flags.items()}
    return PhoneticSegment(**{**base.__dict__, **flags, 'ipa': ipa})

def encode_ipa(ipa):
    return encode_segment(segment_from_ipa(ipa))
