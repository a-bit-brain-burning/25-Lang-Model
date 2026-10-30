"""Application policy for the user's preferred Mandarin D listening profile."""
import numpy as np


def numeric_policy(features, durations, protected_fraction=(.4, .6)):
    """Encode the frozen r2 eligibility and spans as language-neutral arrays."""
    p, c = features['phonetic_features'], features['context_features']
    d = np.asarray(durations, dtype=np.int64)
    if p.shape != (len(d), 32) or c.shape != (len(d), 22) or (d < 0).any():
        raise ValueError('Expected trained phonetic32/context22 contract')
    allowed = np.zeros(len(d), bool)
    active = np.flatnonzero(d > 0)
    for left, right in zip(active[:-1], active[1:]):
        if p[left, 0] > .5 or p[right, 0] > .5:
            continue
        if c[left:right + 1, 8:10].max() > 0:
            continue
        if not (c[left, 21] > .5 and c[right, 20] > .5):
            continue
        if not (p[left, 17] >= .5 and p[right, 17] >= .5):
            continue
        if not (p[left, [1, 5, 6]].sum() > .5 and p[right, [1, 5, 6]].sum() > .5):
            continue
        if (p[left + 1:right, 0] < .5).any():
            continue
        allowed[right] = True
    offsets = np.r_[0, d.cumsum()]
    protected = np.zeros(int(d.sum()), bool)
    spans = [[int(a), int(b)] for a, b in zip(offsets[:-1], offsets[1:]) if b > a]
    begin, end = protected_fraction
    if not 0 <= begin < end <= 1:
        raise ValueError('Invalid protected syllable fraction')
    for start_unit, end_unit in features['syllable_phone_ranges']:
        a, b = int(offsets[start_unit]), int(offsets[end_unit])
        spans.append([a, b])
        protected[a + int(np.floor(begin * (b - a))):a + int(np.ceil(end * (b - a)))] = True
    return allowed, protected, np.asarray(spans, dtype=np.int64).reshape(-1, 2)
