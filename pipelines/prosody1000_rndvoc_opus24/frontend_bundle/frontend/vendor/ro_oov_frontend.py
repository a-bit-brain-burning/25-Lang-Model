"""Versioned Romanian text-only G2P for words absent from retained RSS entries."""
from functools import lru_cache
import hashlib, json, re, unicodedata

def install(front, backend, dictionary_sha256):
    from project.src.frontend import romanian

    @lru_cache(maxsize=128)
    def predict(word):
        if not 1 <= len(word) <= 64 or any(not c.isalpha() or 'LATIN' not in unicodedata.name(c,'') for c in word):
            raise ValueError('Unsupported Romanian OOV word: '+word)
        raw=backend.word(word,'ro')
        if not raw or '(' in raw or ')' in raw:
            raise ValueError('Empty Romanian pronunciation or language switch: '+repr((word,raw)))
        result=[];pending=0
        for token in re.split(r'[_\s]+',raw):
            if not token:continue
            if 'ˈ' in token or 'ˌ' in token:
                pending=1 if 'ˈ' in token else 2
                token=token.replace('ˈ','').replace('ˌ','')
            if not token:continue
            token=unicodedata.normalize('NFD',token).replace('g','ɡ').replace('͡','').replace('͜','')
            # eSpeak can duplicate the same secondary-palatal marker. It does
            # not denote a second independently timed segment; retain one ʲ.
            token=re.sub('ʲ{2,}$','ʲ',token)
            features=romanian.encode_ipa(token)
            stress=pending if features[31] else 0
            if features[31]:pending=0
            result.append((token,stress))
        if not result or pending:raise ValueError('Invalid Romanian stress/phones: '+repr((word,raw)))
        return tuple(result)

    front._oov_resolver=predict
    front.oov_identity=dict(policy='retained_RSS_first_then_espeak_1_52_native_segments',dictionary_sha256=dictionary_sha256,source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),cache_words=128,reference_audio_used=False,human_pronunciation_audited=False)
    front.signature=hashlib.sha256((front.signature+json.dumps(front.oov_identity,sort_keys=True)).encode()).hexdigest()
