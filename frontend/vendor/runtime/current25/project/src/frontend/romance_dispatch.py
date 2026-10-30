"""Language dispatch preserving the frozen es/fr/it/pt frontend identity."""
import json
from pathlib import Path
from .romance import RomanceFrontend as OriginalRomanceFrontend
from .romanian import RomanianFrontend


def RomanceFrontend(language, lexicon_path, **kwargs):
    if language == 'ro':
        path = Path(lexicon_path)
        profile = json.loads(path.read_text(encoding='utf-8-sig')).get('frontend_profile')
        if profile:
            from .romanian_expansion import RomanianExpandedFrontend, PROFILE as RO_PROFILE
            if profile != RO_PROFILE:
                raise ValueError('Unknown Romanian frontend_profile: '+str(profile))
            return RomanianExpandedFrontend(language, lexicon_path, **kwargs)
    if language in ('es', 'it', 'pt'):
        path = Path(lexicon_path)
        with path.open(encoding='utf-8-sig') as source:
            first = source.read(1)
            while first and first.isspace():
                first = source.read(1)
        if first == '{':
            profile = json.loads(path.read_text(encoding='utf-8-sig')).get('frontend_profile')
            if profile:
                from .romance_five_profile import LexicalStressFrontend, PROFILE
                if profile != PROFILE:
                    raise ValueError('Unknown Romance frontend_profile: '+str(profile))
                return LexicalStressFrontend(language, lexicon_path, **kwargs)
    if language == 'fr':
        path = Path(lexicon_path)
        # Existing MFA text dictionaries and untagged JSON dictionaries keep
        # the original class and signature. Only an explicit new JSON profile
        # opts in; unknown profiles fail rather than silently using old inputs.
        with path.open(encoding='utf-8-sig') as source:
            first = source.read(1)
            while first and first.isspace():
                first = source.read(1)
        if first == '{':
            profile = json.loads(path.read_text(encoding='utf-8-sig')).get('frontend_profile')
            if profile:
                from .french_prosody import FrenchProsodyFrontend, PROFILE
                if profile != PROFILE:
                    raise ValueError('Unknown French frontend_profile: '+str(profile))
                return FrenchProsodyFrontend(language, lexicon_path, **kwargs)
    frontend = RomanianFrontend if language == 'ro' else OriginalRomanceFrontend
    return frontend(language,lexicon_path,**kwargs)
