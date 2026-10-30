"""Opt-in dictionary stress, preserving the frozen pronunciation and SIL topology.

Stress is supplied by a preparation-time public dictionary/G2P audit.  Runtime
never invokes G2P or reads reference audio. Unmapped nuclei remain unknown.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from . import romance as base

PROFILE = 'romance_lexical_stress_v1'


class LexicalStressFrontend(base.RomanceFrontend):
    def __init__(self, language, lexicon_path, **kwargs):
        data = json.loads(Path(lexicon_path).read_text(encoding='utf-8-sig'))
        if language not in ('es', 'it', 'pt') or data.get('frontend_profile') != PROFILE:
            raise ValueError('Explicit es/it/pt lexical stress profile required')
        super().__init__(language, lexicon_path, **kwargs)
        self.stresses = data.get('stress', {})
        self.stress_provenance = data.get('stress_sources', {})
        for word, values in self.stresses.items():
            if word not in self.entries or len(values) != len(self.entries[word]):
                raise ValueError('Stress/phone length differs: '+word)
            if any(v not in (-1, 0, 1, 2) for v in values):
                raise ValueError('Invalid stress label: '+word)
            for phone, value in zip(self.entries[word], values):
                if value >= 0 and base.encode_ipa(base.phone_and_stress(phone)[0])[31] < .5:
                    raise ValueError('Only syllabic nuclei accept known stress: '+word)
        parent = self.signature
        self.contract = {**self.contract, 'version': PROFILE,
                         'stress_policy': 'frozen_explicit_nucleus_labels_unknown_on_uncertainty'}
        self.signature = hashlib.sha256(json.dumps({
            'parent_signature': parent, 'contract': self.contract,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def pronounce(self, word):
        word = base._normalized(word)
        phones = super().pronounce(word)
        values = getattr(self, 'stresses', {}).get(word)
        return tuple(zip((p for p, _ in phones), values)) if values is not None else phones

    def encode(self, text):
        features = super().encode(text)
        # The legacy class has no explicit unstressed dictionary labels and
        # maps every known nonprimary label to .5. This opt-in contract adds
        # known zero, consistently with the existing Romanian frontend.
        features['context_features'][features['stress']==0,12] = 0.
        features['stress_sources'] = [self.stress_provenance.get(w, 'unknown')
                                      for w in features['words']]
        features['metadata'].update(frontend_version=PROFILE,
            frontend_signature=self.signature, lexical_stress_human_audited=False,
            stress_policy='explicit_frozen_dictionary_nuclei_unknown_on_uncertainty')
        features['contract_version'] = PROFILE
        return features
