#!/usr/bin/env python3
"""Verify the acoustic-only release manifest without loading executable pickle."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for name, record in manifest['files'].items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Invalid manifest path: ' + name)
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Release file changed or missing: ' + name)
    model_dir = root / 'models'
    export = json.loads((model_dir / 'EXPORT.json').read_text())
    for name, record in export['files'].items():
        path = model_dir / name
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Model export identity mismatch: ' + name)
    raw = (model_dir / 'acoustic_fp16.safetensors').read_bytes()
    if len(raw) < 8:
        raise ValueError('Missing safetensors header')
    size = struct.unpack('<Q', raw[:8])[0]
    if size < 2 or size > min(len(raw) - 8, 1024 * 1024):
        raise ValueError('Invalid safetensors header bound')
    header = json.loads(raw[8:8 + size])
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if len(tensors) != 97:
        raise ValueError('Expected exactly 97 acoustic tensors')
    ranges = []
    total = 0
    for name, item in tensors.items():
        if item['dtype'] != 'F16' or any(x in name.lower() for x in ['waveform', 'vocoder', 'discriminator']):
            raise ValueError('Non-acoustic / wrong dtype tensor: ' + name)
        count = math.prod(item['shape'])
        start, end = item['data_offsets']
        if start < 0 or end - start != count * 2:
            raise ValueError('Tensor byte count mismatch: ' + name)
        ranges.append((start, end)); total += count
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError('Overlapping or gapped weight ranges')
        cursor = end
    if total != 1186071 or cursor != len(raw) - 8 - size:
        raise ValueError('Unexpected acoustic payload / trailing data')
    provenance = json.loads((root / 'receipts/acoustic_provenance.json').read_text())
    if not provenance['original_80000_vs_released_fp32_all_equal'] or provenance['vocoder_tensors_released'] != 0:
        raise ValueError('Acoustic provenance / exclusion receipt failed')
    return {'status': 'RELEASE_SHA_AND_ACOUSTIC_LAYOUT_VERIFIED',
            'files': len(manifest['files']), 'acoustic_tensors': len(tensors),
            'parameters': total, 'fp16_tensor_bytes': cursor,
            'quality_or_ram_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
