#!/usr/bin/env python3
"""Verify the acoustic/HiFi-GAN release without loading executable pickle."""
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
    hifigan = None
    hifigan_dir = model_dir / 'hifigan'
    if hifigan_dir.exists():
        he = json.loads((hifigan_dir / 'EXPORT.json').read_text())
        for name, record in he['files'].items():
            path = hifigan_dir / name
            if not path.resolve().is_relative_to(hifigan_dir.resolve()) or path.is_symlink():
                raise ValueError('Invalid HiFi-GAN export path')
            if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise ValueError('HiFi-GAN model export identity mismatch: ' + name)
        blob = (hifigan_dir / 'vocoder_adapter_fp16.safetensors').read_bytes()
        hs = struct.unpack('<Q', blob[:8])[0]
        if not 2 <= hs <= min(len(blob) - 8, 1024 * 1024):
            raise ValueError('Invalid HiFi-GAN safetensors header bound')
        ht = {k: v for k, v in json.loads(blob[8:8 + hs]).items() if k != '__metadata__'}
        if len(ht) != 219:
            raise ValueError('Expected 219 HiFi-GAN/adapter tensors')
        hranges = []
        components = {'generator': [0, 0], 'physical_to_mel': [0, 0]}
        for name, item in ht.items():
            component = name.split('.', 1)[0]
            if component not in components or item['dtype'] != 'F16':
                raise ValueError('Unexpected HiFi-GAN component/dtype: ' + name)
            count = math.prod(item['shape'])
            begin, end = item['data_offsets']
            if begin < 0 or end - begin != count * 2:
                raise ValueError('HiFi-GAN tensor byte count mismatch: ' + name)
            components[component][0] += 1; components[component][1] += count
            hranges.append((begin, end))
        if components != {'generator': [156, 925985], 'physical_to_mel': [63, 1902992]}:
            raise ValueError('HiFi-GAN/adapter tensor topology differs')
        hc = 0
        for begin, end in sorted(hranges):
            if begin != hc:
                raise ValueError('HiFi-GAN overlapping/gapped tensor ranges')
            hc = end
        if hc != 5657954 or hc != len(blob) - 8 - hs:
            raise ValueError('HiFi-GAN payload / trailing bytes mismatch')
        if hashlib.sha256(blob[8 + hs:]).hexdigest() != he['source_raw_fp16_sha256']:
            raise ValueError('HiFi-GAN FP16 data differs from original experiment export')
        hp = json.loads((root / 'receipts/hifigan_provenance.json').read_text())
        if he['source_checkpoint_sha256'] != hp['source_checkpoint_sha256'] or not hp['all_fp16_bytes_equal_original_export']:
            raise ValueError('HiFi-GAN provenance mismatch')
        upstream = json.loads((root / 'third_party/hifi_gan/PROVENANCE.json').read_text())
        if sha256(root / 'third_party/hifi_gan/LICENSE') != upstream['upstream_license_sha256']:
            raise ValueError('Original upstream MIT license changed')
        hifigan = {'tensors': len(ht), 'parameters': 2828977, 'fp16_tensor_bytes': hc}
    return {'status': 'RELEASE_SHA_AND_MODEL_LAYOUTS_VERIFIED',
            'files': len(manifest['files']), 'acoustic_tensors': len(tensors),
            'acoustic_parameters': total, 'acoustic_fp16_tensor_bytes': cursor,
            'hifigan': hifigan,
            'parameters': total + (hifigan['parameters'] if hifigan else 0),
            'fp16_tensor_bytes': cursor + (hifigan['fp16_tensor_bytes'] if hifigan else 0),
            'quality_or_ram_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
