"""Text-only syllable phrase-plan contract; language policy stays in frontend.

No reference audio, duration, or publisher pinyin is accepted by encode_text.
Acoustic consumers see eight numeric columns, never characters or rules.
"""
from __future__ import annotations
import re
import numpy as np

PLAN_DIM = 8
PLAN_VERSION = 'phrase_plan8_v1'
PLAN_NAMES = ('boundary_none', 'boundary_1', 'boundary_2', 'boundary_3',
              'prominence_z', 'log_duration_ratio', 'pitch_level_z', 'pitch_range_z')
INPUT_DIM = 66


def clipped_plan(plan):
    values = np.asarray(plan, np.float32).copy()
    if values.ndim != 2 or values.shape[1] != PLAN_DIM or not np.isfinite(values).all():
        raise ValueError('Expected finite K x 8 phrase plan')
    values[:, :4] = np.clip(values[:, :4], 0, 1)
    values[:, [4, 6, 7]] = np.clip(values[:, [4, 6, 7]], -2, 2)
    values[:, 5] = np.clip(values[:, 5], -.5, .5)
    return values


def plan_for_units(frontend_result, syllable_plan):
    """Broadcast K syllable plans onto N native units, incl. following pause.

    Leading silence receives all zeros. Gaps after a syllable (optional pause)
    inherit its plan. This routing uses topology, never acoustic measurements.
    All-zero plans are legal as a matched zero-conditioning control.
    """
    syllables = frontend_result['metadata']['syllables']
    values = clipped_plan(syllable_plan)
    if len(values) != len(syllables):
        raise ValueError('Phrase-plan syllable count differs from frontend')
    n = len(frontend_result.get('unit_labels', frontend_result.get('phonetic_features', [])))
    result = np.zeros((n, PLAN_DIM), np.float32)
    previous_end = 0
    for i, syllable in enumerate(syllables):
        start, end = int(syllable['phone_start']), int(syllable['phone_end'])
        following = int(syllables[i + 1]['phone_start']) if i + 1 < len(syllables) else n
        if not 0 <= previous_end <= start < end <= following <= n:
            raise ValueError('Invalid frontend syllable/native-unit topology')
        result[start:following] = values[i]
        previous_end = end
    return result


def project_by_characters(source_frontend, source_plan, target_frontend):
    """Route text predictions to training-native topology by characters only.

    Handles a fused erhua syllable without reading target pinyin or acoustics.
    The final constituent supplies boundary probabilities; continuous values
    are averaged when multiple ordinary syllables map onto one target unit.
    """
    source = source_frontend['metadata']['syllables']
    target = target_frontend['metadata']['syllables']
    values = clipped_plan(source_plan)
    if len(source) != len(values) or ''.join(s['character'] for s in source) != ''.join(s['character'] for s in target):
        raise ValueError('Source/target phrase-plan character sequences differ')
    source_ends = np.cumsum([len(s['character']) for s in source])
    source_starts = np.r_[0, source_ends[:-1]]
    result, cursor = [], 0
    for syllable in target:
        end = cursor + len(syllable['character'])
        overlapping = np.flatnonzero((source_starts < end) & (source_ends > cursor))
        row = values[overlapping].mean(0)
        row[:4] = values[overlapping[-1], :4]
        result.append(row)
        cursor = end
    return np.asarray(result, np.float32).reshape(-1, PLAN_DIM)


def syllable_inputs(frontend_result, vocabulary):
    phonetic = np.asarray(frontend_result['phonetic_features'], np.float32)
    context = np.asarray(frontend_result['context_features'], np.float32)
    if context.shape[1] == 22:
        context = np.concatenate((context, np.asarray(frontend_result['prosody_features'], np.float32)), axis=1)
    if phonetic.shape[1] != 32 or context.shape != (len(phonetic), 34):
        raise ValueError('Planner requires ordinary phonetic32 + context34')
    numeric, identifiers = [], []
    for syllable in frontend_result['metadata']['syllables']:
        start, end = int(syllable['phone_start']), int(syllable['phone_end'])
        numeric.append(np.concatenate((phonetic[start:end].mean(0), context[start:end].mean(0))))
        identifiers.append(vocabulary.get(syllable['character'][0], 1))
    return {'numeric': np.asarray(numeric, np.float32).reshape(-1, INPUT_DIM),
            'char_ids': np.asarray(identifiers, np.int64)}


def make_frontend():
    from .chinese_delivery_policy import ChineseDeliveryFrontend
    from .chinese_prosody import ChineseProsodyFrontend
    return ChineseDeliveryFrontend(ChineseProsodyFrontend())


def encode_text(text, vocabulary, frontend=None):
    """Ordinary text only. Corpus prosody #markers are not planner inputs."""
    clean = re.sub(r'#[1-4]', '', text)
    result = (frontend or make_frontend()).encode(clean)
    if result['metadata'].get('provided_pinyin_authoritative'):
        raise ValueError('Planner text frontend unexpectedly used reference pinyin')
    return result, syllable_inputs(result, vocabulary)
