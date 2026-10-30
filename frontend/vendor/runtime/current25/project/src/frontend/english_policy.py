"""English lexical/prosodic policy, owned by the English frontend."""
import copy

FUNCTION_WORDS = frozenset(('an','of'))
VOWELS = frozenset('AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split())
UNVOICED = frozenset('P T K F TH S SH HH CH'.split())

def prepare_english_features(features, unstress_function_words=True):
    if features.get('language') != 'en':
        raise ValueError('English policy requires the English frontend')
    result = copy.deepcopy(features)
    if unstress_function_words:
        for i,wi in enumerate(result['word_indices']):
            if wi >= 0 and result['words'][wi] in FUNCTION_WORDS and result['stress'][i]>=0:
                result['stress'][i] = 0
    result['english_policy'] = {'function_word_unstress':unstress_function_words,
                              'pitch_spread_factor':1.4,'source':'retained_user_preference'}
    return result

def prepare_legacy_profile_features(features, config):
    """Exact legacy lexical policy; caller handles baseline and language check."""
    result = copy.deepcopy(features)
    words = set(config['function_words_unstressed'])
    for i,wi in enumerate(result['word_indices']):
        if wi >= 0 and result['words'][wi] in words and result['phones'][i] in VOWELS:
            result['stress'][i] = 0
    return result

def unvoiced_spans(phones,durations):
    """Legacy English source override mask as sample-free frame intervals."""
    if len(phones)!=len(durations): raise ValueError('Phones/durations mismatch')
    offset=0
    for phone,n in zip(phones,durations):
        n=int(n)
        if n<0: raise ValueError('Negative duration')
        if phone in UNVOICED: yield offset,offset+n
        offset+=n

def legacy_vowel_mask(phones):
    return [phone in VOWELS for phone in phones]
