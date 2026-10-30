"""Language-neutral frontend contract and numeric encoder."""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from .phonetic_features import FEATURE_NAMES, FEATURE_DIM, encode_ipa

CONTEXT_NAMES = ('word_start','word_end','phone_position_in_word','word_length_scale',
                 'word_position','phone_position','sentence_phone_length_scale','sentence_word_length_scale',
                 'boundary_after','boundary_before','first_edge','last_edge',
                 'stress_level','stress_known','tone_start','tone_mid','tone_end','tone_known',
                 'pitch_accent','pitch_accent_known','syllable_start','syllable_end')
CONTEXT_DIM = len(CONTEXT_NAMES)
CONTRACT_VERSION = 'universal_frontend_v1'

@dataclass(frozen=True)
class UniversalPhone:
    ipa: str
    stress: float | None = None
    tone_contour: tuple[float,float,float] | None = None
    pitch_accent: float | None = None
    syllable_start: bool = False
    syllable_end: bool = False

class LanguageFrontend(ABC):
    @abstractmethod
    def encode(self,text: str):
        raise NotImplementedError

def encode_phones(phones, positions=None, metadata=None, valid_mask=None):
    phones = list(phones)
    n = len(phones)
    if not n:
        raise ValueError('Empty phone sequence')
    base = np.zeros((n,12),np.float32) if positions is None else np.asarray(positions,np.float32)
    if base.shape != (n,12) or not np.isfinite(base).all():
        raise ValueError('Position features must be finite N x 12')
    context = np.zeros((n,CONTEXT_DIM),np.float32)
    context[:,:12] = base
    for i,p in enumerate(phones):
        if p.stress is not None:
            if not 0 <= p.stress <= 1: raise ValueError('Stress level outside [0,1]')
            context[i,12:14] = (p.stress,1.)
        if p.tone_contour is not None:
            contour = np.asarray(p.tone_contour,np.float32)
            if contour.shape != (3,) or not np.isfinite(contour).all() or (np.abs(contour)>1).any():
                raise ValueError('Tone contour must contain three finite [-1,1] coordinates')
            context[i,14:18] = (*contour,1.)
        if p.pitch_accent is not None:
            if not 0 <= p.pitch_accent <= 1: raise ValueError('Pitch accent outside [0,1]')
            context[i,18:20] = (p.pitch_accent,1.)
        context[i,20:22] = (p.syllable_start,p.syllable_end)
    phonetic = np.stack([encode_ipa(p.ipa) for p in phones])
    valid = np.ones(n,bool) if valid_mask is None else np.asarray(valid_mask,bool)
    if valid.shape != (n,): raise ValueError('valid_mask shape differs from phones')
    result = {'phonetic_features':phonetic,'context_features':context,'valid_mask':valid,
              'edge_sil_mask':base[:,10:12].sum(-1)>.5,
              'phonetic_feature_names':FEATURE_NAMES,'context_feature_names':CONTEXT_NAMES,
              'contract_version':CONTRACT_VERSION,'ipa':[p.ipa for p in phones], 'metadata':metadata or {}}
    return result

def to_serializable(result):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
