"""Text-derived English supervision groups; never acoustic-model inputs.

Lexical stress, vowels and word-final obstruents are interpreted here so the
shared objective sees only numeric masks. No assumed duration or VOT targets.
"""
import numpy as np

from .english import PHONE_TO_ID

VOWELS = 'AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split()
OBSTRUENTS = 'B CH D DH F G HH JH K P S SH T TH V Z ZH'.split()


def supervision_groups(phone_ids, text_features, stress):
    ids = np.asarray(phone_ids)
    positions = np.asarray(text_features)
    stress = np.asarray(stress)
    if ids.ndim != 1 or positions.shape != (len(ids), 12) or stress.shape != ids.shape:
        raise ValueError('Expected ids[N], positions[N,12], stress[N]')
    vowels = np.isin(ids, [PHONE_TO_ID[p] for p in VOWELS])
    consonants = np.isin(ids, [PHONE_TO_ID[p] for p in OBSTRUENTS])
    lexical = np.isin(ids, [v for k, v in PHONE_TO_ID.items() if not k.startswith('<')])
    stressed = vowels & np.isin(stress, [1, 2])
    final = consonants & (positions[:, 1] > .5)
    # Adjacent distinct vowel categories, within this utterance only. The
    # target remains the observed spectrum difference, not an invented margin.
    indices = np.flatnonzero(vowels)
    pairs = np.column_stack((indices[:-1], indices[1:]))
    pairs = pairs[ids[pairs[:, 0]] != ids[pairs[:, 1]]]
    return {'vowel': vowels, 'obstruent': consonants, 'lexical': lexical,
            'stressed_vowel': stressed, 'final_obstruent': final,
            'duration_weight': lexical.astype(np.float32) + .5 * stressed + .5 * final,
            'vowel_pairs': pairs.astype(np.int64)}
