"""Text-only adapter for unchanged numeric36/context22/plan5 Romance frontends."""
from pathlib import Path
import numpy as np
from project.src.frontend.romance_dispatch import RomanceFrontend
from project.src.models.romance_word_planner import encode_frontend


def encode(language,text,metadata):
    frontend=RomanceFrontend(language,Path(metadata['lexicon_path']))
    expected=metadata.get('frontend_signature')
    if expected and frontend.signature!=expected:raise ValueError('Frontend identity changed')
    f=frontend.encode(text);p=encode_frontend(f)
    return {**f,'planner_numeric':p['numeric'],'planner_boundary':p['boundary'],'planner_owner':f['plan_owner']}
