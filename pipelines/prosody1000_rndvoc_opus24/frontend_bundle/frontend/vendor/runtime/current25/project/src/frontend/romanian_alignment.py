"""Explicit RSS-to-official-Romanian-CV aligner phone inventory adapter.

This is a coarse automatic alignment vocabulary, not a change to the deployed
RSS pronunciations. Glides share vowel HMMs; schwa uses the model's central
vowel HMM. Restore each original RSS symbol after sequence validation. These
approximations must remain in the supervision provenance.
"""
import unicodedata

CV_ALIASES = {'ə':'ɜ','d':'d̪','l':'l̪','n':'n̪','r':'r̪','t':'t̪',
              'z':'z̪','ts':'t̪s','e̯':'e','o̯':'o','i̯':'i','j':'i','w':'u'}


def aligner_phone(phone):
    return CV_ALIASES.get(phone,phone)


def restore_intervals(original_phones, intervals):
    """Keep actual aligner timestamps; reject omissions or sequence changes."""
    original = [p for p in original_phones if p != 'sil']
    position = 0
    restored = []
    for interval in intervals:
        label = interval['ipa']
        if label in ('','sp','sil'):
            restored.append({**interval,'ipa':'sil'})
            continue
        if position >= len(original):
            raise ValueError('Extra aligned Romanian CV phone')
        expected = aligner_phone(original[position])
        if unicodedata.normalize('NFD',label) != unicodedata.normalize('NFD',expected):
            raise ValueError(f'Romanian CV phone sequence mismatch: {label} != {expected}')
        restored.append({**interval,'ipa':original[position]})
        position += 1
    if position != len(original):
        raise ValueError('Missing aligned Romanian CV phones')
    return restored
