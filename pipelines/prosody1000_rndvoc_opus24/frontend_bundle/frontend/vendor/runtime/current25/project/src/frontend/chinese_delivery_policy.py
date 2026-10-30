"""Opt-in Mandarin delivery policy; acoustic adapters receive numerical arrays.

This module does not infer focus or prescribe a fifth-tone F0 template.  The
closed lexical list is deliberately independent of general AA reduplication.
No frozen frontend, checkpoint, or default synthesis profile is changed.
"""
from __future__ import annotations

from collections import defaultdict
import re
import numpy as np

VERSION = 'chinese_delivery_policy_v1'
KINSHIP_NEUTRAL = frozenset(('妈妈', '爸爸', '哥哥', '姐姐', '弟弟', '妹妹',
    '爷爷', '奶奶', '叔叔', '舅舅', '姑姑', '伯伯', '婶婶'))


def _word_groups(features):
    groups = defaultdict(list)
    for i, syllable in enumerate(features['metadata']['syllables']):
        groups[syllable['group_index']].append(i)
    return list(groups.values())


def _punctuation_after(features):
    """Map actual text punctuation to syllables, including fused erhua safely."""
    text = re.sub(r'#[1-4]', '', features['metadata']['normalized_text'])
    after, count = defaultdict(str), 0
    for char in text:
        if '\u3400' <= char <= '\u9fff':
            count += 1
        elif char in '，,、。.!?！？；;：:':
            after[count] += char
    result, position = [], 0
    for syllable in features['metadata']['syllables']:
        position += len(syllable['character'])
        result.append(after[position])
    return result


class ChineseDeliveryFrontend:
    """Wrap v1/v2; only reliable ordinary-text kinship words may be relabelled.

    Explicit pinyin remains authoritative.  protected_syllables prevents this
    adapter's changes (it does not reverse the wrapped frontend's own rules).
    """
    def __init__(self, base_frontend):
        self.base_frontend = base_frontend

    def encode(self, text, pinyin=None, *, protected_syllables=()):
        features = self.base_frontend.encode(text, pinyin=pinyin)
        protected = set(int(i) for i in protected_syllables)
        syllables = features['metadata']['syllables']
        hint_positions={e['start'] for e in features['metadata'].get('pronunciation_events',[]) if e['source']=='explicit_citation_hint'}
        cursor=0
        for index,syllable in enumerate(syllables):
            if cursor in hint_positions:protected.add(index)
            cursor+=len(syllable['character'])
        if any(i < 0 or i >= len(syllables) for i in protected):
            raise ValueError('protected_syllables index outside sentence')
        original_tokens = list(features['pinyin'])
        tokens = list(original_tokens)
        lexical = []
        for indices in _word_groups(features):
            word = ''.join(syllables[i]['character'] for i in indices)
            if len(indices) != 2 or word not in KINSHIP_NEUTRAL:
                continue
            a, b = indices
            # Full-token and equal segmental pronunciation guards; punctuation
            # must not bisect a lexical word and explicit emphasis is unknown.
            if (syllables[a]['boundary_after_level'] or a in protected or b in protected
                    or tokens[a][:-1] != tokens[b][:-1]):
                continue
            if pinyin is None:
                tokens[b] = tokens[b][:-1] + '5'
                # v1 may already have applied 3+3 -> 2+3. Once the second
                # lexical syllable is neutral, that particular trigger no
                # longer exists. Restore only its recorded citation-3 first
                # syllable, never guess from an arbitrary lexical tone-2.
                citation = syllables[a].get('citation_pinyin')
                if (tokens[a].endswith('2') and citation
                        and citation == tokens[a][:-1] + '3'):
                    tokens[a] = citation
            if tokens[b].endswith('5'):
                lexical.append((a, b))
        changed = [i for i, (old, new) in enumerate(zip(original_tokens, tokens)) if old != new]
        if changed:
            # Reuse the wrapped encoder so tone vectors/prosody12 stay coherent.
            # These generated tokens are explicitly NOT publisher annotation.
            features = self.base_frontend.encode(text, pinyin=tokens)
            features['metadata'].update(pinyin_source='ordinary_text_delivery_policy_generated',
                provided_pinyin_authoritative=False,
                tone_sandhi_applied=False,
                delivery_previous_frontend_tokens=original_tokens)
            for i, s in enumerate(features['metadata']['syllables']):
                if i in changed:
                    s['delivery_pronunciation_source'] = 'closed_kinship_whole_word'
                    if 'sandhi_source' in s:
                        s['sandhi_source'] = ['ordinary_text_closed_kinship_policy']
        n = len(features['unit_labels'])
        comma_mask = np.zeros(n, bool)
        marks = _punctuation_after(features)
        syllables = features['metadata']['syllables']
        optional = set(int(i) for i in features['optional_pause_indices'])
        for i, s in enumerate(syllables[:-1]):
            index = int(s['phone_end'])
            if (('，' in marks[i] or ',' in marks[i]) and index in optional
                    and features['edge_sil_mask'][index]
                    and features['unit_labels'][index] == 'sp'):
                comma_mask[index] = True
        neutral_mask = np.zeros(n, bool)
        pairs = []
        for a, b in lexical:
            left, right = syllables[a], syllables[b]
            neutral_mask[right['phone_start']:right['phone_end']] = True
            pairs.append((left['phone_start'], left['phone_end'],
                          right['phone_start'], right['phone_end']))
        features['delivery_comma_mask'] = comma_mask
        features['delivery_kinship_neutral_mask'] = neutral_mask
        features['delivery_kinship_pairs'] = np.asarray(pairs, np.int32).reshape(-1, 4)
        features['delivery_kinship_pair_final'] = np.asarray([
            b == len(syllables) - 1 or syllables[b]['boundary_after_level'] >= 2
            for a, b in lexical], bool)
        features['metadata']['delivery_policy'] = {
            'version': VERSION, 'ordinary_text_only': pinyin is None,
            'supplied_pinyin_rewritten': False, 'changed_pinyin_indices': changed,
            'closed_kinship_pairs': [list(x) for x in lexical],
            'protected_syllables': sorted(protected),
            'punctuation_after': marks, 'comma_pause_unit_indices': np.flatnonzero(comma_mask).tolist(),
            'focus_inferred': False, 'fixed_neutral_F0_target': False,
            'duration_policy_applied': False,
        }
        return features


def numeric_duration_policy(features, durations, *, comma_min_frames=22,
                            kinship_max_ratio=None, minimum_rime_frames=4,
                            minimum_neutral_scale=.75, include_final_kinship=False):
    """Return numerical bounds/targets for a generic duration adapter.

    A 22-frame (220 ms) comma floor is an empirical opt-in candidate, not a
    universal prosodic rule. A neutral ratio is disabled by default. When enabled
    it shortens only a closed-list second rime, preserving its initial and at
    least 75% of that syllable's original duration. No other neutral is changed.
    """
    durations = np.asarray(durations)
    n = len(features['unit_labels'])
    if (durations.shape != (n,) or not np.isfinite(durations).all()
            or np.any(durations < 0) or np.any(durations != np.floor(durations))):
        raise ValueError('durations must be one nonnegative integer per unit')
    if (not isinstance(comma_min_frames, (int, np.integer)) or not 0 <= comma_min_frames <= 150):
        raise ValueError('comma_min_frames must be an integer in [0,150]')
    if kinship_max_ratio is not None and not 0 < kinship_max_ratio <= 2:
        raise ValueError('kinship_max_ratio must be in (0,2] or None')
    if not 0 < minimum_neutral_scale <= 1 or minimum_rime_frames < 1:
        raise ValueError('invalid neutral safety limits')
    original = durations.astype(np.int64)
    targets = original.copy()
    floor = np.zeros(n, np.int64)
    ceiling = np.full(n, -1, np.int64)  # -1 means no additional upper bound.
    comma = np.asarray(features['delivery_comma_mask'], bool)
    neutral = np.asarray(features['delivery_kinship_neutral_mask'], bool)
    floor[comma] = comma_min_frames
    targets = np.maximum(targets, floor)
    if kinship_max_ratio is not None:
        for pair_index, (a, b, c, d) in enumerate(features['delivery_kinship_pairs']):
            if features['delivery_kinship_pair_final'][pair_index] and not include_final_kinship:
                continue
            total = int(original[c:d].sum())
            preceding = int(original[a:b].sum())
            rime = int(d) - 1
            # Preserve the nasal initial; change only an existing positive rime.
            if total == 0 or preceding == 0 or original[rime] <= minimum_rime_frames:
                continue
            desired = max(int(round(preceding * kinship_max_ratio)),
                          int(np.ceil(total * minimum_neutral_scale)),
                          total - int(original[rime]) + int(minimum_rime_frames))
            reduced = max(0, total - desired)
            if reduced:
                ceiling[rime] = original[rime] - reduced
                targets[rime] = min(targets[rime], ceiling[rime])
    changed = targets != original
    return {'duration_floor_frames': floor, 'duration_ceiling_frames': ceiling,
            'target_durations': targets, 'comma_mask': comma.copy(),
            'kinship_neutral_mask': neutral.copy(), 'changed_mask': changed,
            'metadata': {'version': VERSION, 'frame_ms': 10,
                'comma_min_frames': int(comma_min_frames), 'kinship_max_ratio': kinship_max_ratio,
                'minimum_neutral_scale': minimum_neutral_scale,
                'include_final_kinship': bool(include_final_kinship),
                'changed_indices': np.flatnonzero(changed).tolist(),
                'F0_AP_VUV_controls_changed': False,
                'policy_parameters_are_empirical_candidates': True}}
