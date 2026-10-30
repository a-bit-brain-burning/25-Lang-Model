import hashlib
import json
from pathlib import Path
import numpy as np

LANGUAGES = ('bg','cs','da','de','el','en','es','et','fi','fr','hr','hu','it',
             'lt','lv','nl','no','pl','pt','ro','sk','sl','sv','tr','zh')
SR, HOP, RAM_LIMIT = 16000, 160, 10_000_000
SCHEMA = 'newtts_physical_31_39_strict_v1'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def json_write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)

def safe_path(root, relative):
    root = Path(root).resolve()
    p = (root/relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(root):
        raise ValueError(f'Nonportable/outside path: {relative}')
    return p

def load_arrays(directory, row, verify=False):
    """Never execute dataset Python. Strict arrays override record arrays."""
    import zipfile
    arrays = {}
    for field, digest in [('record_path','sha256'), ('strict_overlay','strict_overlay_sha256')]:
        p = safe_path(directory, row[field])
        if verify and sha256(p) != row[digest]: raise ValueError(f'SHA mismatch {p}')
        with zipfile.ZipFile(p) as z:
            if sum(i.file_size for i in z.infolist()) > 128*1024**2:
                raise ValueError('NPZ uncompressed size limit')
        with np.load(p, allow_pickle=False) as z:
            arrays.update({k:z[k] for k in z.files})
    return arrays

def validate_arrays(a, row):
    c = np.asarray(a['target_curve'], np.float32).copy()
    duration = np.asarray(a['durations'])
    if c.ndim != 2 or c.shape[1] not in (31,39): raise ValueError('Expected 31/39D')
    t, d = c.shape
    n = len(duration)
    if duration.shape != (n,) or np.any(duration < 0) or np.any(duration != duration.astype('int64')):
        raise ValueError('Noninteger/negative durations')
    if duration.sum() != t or t != row['frames'] or n != row['phones']:
        raise ValueError('Duration/frame/phone mismatch')
    if a['ipa'].shape != (n,): raise ValueError('IPA length mismatch')
    for k, dims in [('phonetic_features',(32,36)),('context_features',(22,34))]:
        if a[k].shape[0] != n or a[k].shape[1] not in dims or not np.isfinite(a[k]).all():
            raise ValueError(f'Unsupported frontend contract: {k} {a[k].shape}')
    for k in ('frame_voice','frame_voice_weight','frame_f0_valid','confirmed_silence'):
        if k not in a or a[k].shape != (t,): raise ValueError(f'Missing strict frame label {k}')
    # Stored curves have *physical* probability, NOT logits. Prefer strict voice.
    c[:,-1] = a['frame_voice']
    if not np.isfinite(c).all() or not np.isfinite(a['frame_voice_weight']).all():
        raise ValueError('Nonfinite labels')
    if np.any((c[:,-1] < 0)|(c[:,-1] > 1)) or np.any(a['frame_voice_weight'] < 0):
        raise ValueError('Invalid voice labels')
    if np.any((c[:,d-6:d-1] < -1.001)|(c[:,d-6:d-1] > .001)):
        raise ValueError('AP5 semantic contract')
    fv = np.asarray(a['frame_f0_valid'],bool)
    if np.any((c[fv,0] < np.log(40))|(c[fv,0] > np.log(600))):
        raise ValueError('Voiced log-F0 outside contract')
    return c, duration.astype('int64')

def frontend(a):
    """Features from the original frontend only; no waveform-derived input."""
    p = len(a['ipa'])
    pause = np.isin(a['ipa'].astype(str), ['sil','sp','spn','<sil>'])
    # Do not feed duration-derived pause masks or acoustic confidence to encoder.
    return (a['phonetic_features'].astype('float32'),
            a['context_features'].astype('float32'), pause.astype('float32'))
