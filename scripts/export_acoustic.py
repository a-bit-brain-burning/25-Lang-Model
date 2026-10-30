#!/usr/bin/env python3
"""Export this 285-phone, 80000-step acoustic release from a SHA-bound checkpoint.

The default loader uses weights_only=True. Full training checkpoints also contain
NumPy RNG state: loading those requires both an expected SHA and the explicit
--trusted-full-checkpoint option. A bare state_dict lacks training provenance and
is insufficient input. Combined containers require the original tensor proof.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch


LANGUAGES = 'bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh'.split()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_safetensors(path, state):
    header = {'__metadata__': {'format': 'pt', 'component': 'acoustic', 'training_step': '80000'}}
    blocks = []
    offset = 0
    for name, tensor in state.items():
        array = tensor.detach().cpu().float().numpy()
        if not np.isfinite(array).all() or np.max(np.abs(array), initial=0) > 65504:
            raise ValueError('Cannot represent finite FP16 tensor: ' + name)
        data = array.astype('<f2').tobytes(order='C')
        header[name] = {'dtype': 'F16', 'shape': list(array.shape),
                        'data_offsets': [offset, offset + len(data)]}
        blocks.append(data)
        offset += len(data)
    raw_header = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    raw_header += b' ' * (-len(raw_header) % 8)
    with Path(path).open('xb') as f:
        f.write(struct.pack('<Q', len(raw_header)))
        f.write(raw_header)
        for block in blocks:
            f.write(block)
    return header, offset


def validate_training_identity(cp, checkpoint_sha, state, completed=None, proof=None):
    """This release exporter must never label a partial/smoke model as 80k."""
    if cp.get('smoke_test') is not False:
        raise ValueError('Only an explicitly non-smoke training checkpoint can be exported')
    phase, step = cp.get('phase'), cp.get('step')
    if phase == 'acoustic':
        if type(step) is not int or step != 80000:
            raise ValueError('This release requires the actual 80000-step acoustic checkpoint')
        if completed is not None and completed.get('checkpoint_sha256') != checkpoint_sha:
            raise ValueError('Acoustic completion belongs to another checkpoint')
    else:
        if phase not in ('vocoder_pretrain', 'vocoder_gan', 'vocoder_adapt') or completed is None or proof is None:
            raise ValueError('A combined container requires the original acoustic completion and tensor proof')
        original = proof.get('original_acoustic_checkpoint', {})
        if original.get('sha256') != completed.get('checkpoint_sha256') or original.get('step') != 80000:
            raise ValueError('Original acoustic proof/completion identity differs')
        expected = proof.get('tensors', {})
        actual = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
                  for key, value in state.items()}
        if expected != actual or proof.get('original_80000_vs_released_fp32_all_equal') is not True:
            raise ValueError('Combined container acoustic tensors differ from the verified 80000-step state')
    if completed is not None and (completed.get('step') != 80000 or completed.get('smoke_test') is not False
                                  or completed.get('status') != 'MAX_STEPS_REACHED'):
        raise ValueError('Invalid acoustic training completion receipt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-full-checkpoint', action='store_true')
    parser.add_argument('--acoustic-completion', type=Path)
    parser.add_argument('--acoustic-provenance', type=Path,
                        help='Required for combined containers: original 80000-step per-tensor identity proof')
    args = parser.parse_args()
    sha = digest(args.checkpoint)
    if sha != args.expected_checkpoint_sha256:
        raise ValueError('Checkpoint SHA mismatch; refusing to load')
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=not args.trusted_full_checkpoint)
    model = cp.get('model', cp)
    state = {key[len('acoustic.'):]: tensor.detach().cpu().float().contiguous().clone()
             for key, tensor in model.items() if key.startswith('acoustic.')}
    if not state:
        # Standalone acoustic checkpoints use unprefixed names.
        state = {key: tensor.detach().cpu().float().contiguous().clone()
                 for key, tensor in model.items() if isinstance(tensor, torch.Tensor)}
    if len(state) != 97 or sum(t.numel() for t in state.values()) != 1186071:
        raise ValueError('Unexpected acoustic tensor topology / vocabulary')
    if any(any(word in key.lower() for word in ('waveform', 'vocoder', 'discriminator')) for key in state):
        raise ValueError('Non-acoustic tensor in export')
    completed = json.loads(args.acoustic_completion.read_text()) if args.acoustic_completion else None
    proof = json.loads(args.acoustic_provenance.read_text()) if args.acoustic_provenance else None
    validate_training_identity(cp, sha, state, completed, proof)
    vocab, norm = cp['vocab'], cp['normalization']
    if len(vocab) != 285 or set(norm) != set(LANGUAGES):
        raise ValueError('Vocabulary/language identity mismatch')
    for lang in LANGUAGES:
        expected = 31 if lang == 'zh' else 39
        if len(norm[lang]['mean']) != expected or len(norm[lang]['std']) != expected:
            raise ValueError('Acoustic dimension mismatch: ' + lang)
    args.output.mkdir(parents=True, exist_ok=True)
    targets = ['acoustic_fp16.safetensors', 'acoustic_fp32.pt', 'vocabulary.json',
               'normalization.json', 'model_config.json', 'EXPORT.json']
    if any((args.output / name).exists() for name in targets):
        raise FileExistsError('Export refuses to overwrite any existing artifact')
    header, weight_bytes = write_safetensors(args.output / targets[0], state)
    torch.save(state, args.output / targets[1])
    write_json(args.output / 'vocabulary.json', vocab)
    write_json(args.output / 'normalization.json', norm)
    write_json(args.output / 'model_config.json', {
        'component': 'acoustic', 'format_version': 1, 'vocab_size': 285,
        'languages': LANGUAGES, 'language_id': {lang: i for i, lang in enumerate(LANGUAGES)},
        'acoustic_dimension_by_language': {lang: 31 if lang == 'zh' else 39 for lang in LANGUAGES},
        'phone_width': 128, 'frame_width': 192, 'phonetic_dimensions': [32, 36],
        'context_dimensions': [22, 34], 'sample_rate': 16000, 'hop_samples': 160,
        'training_step': 80000, 'stored_parameters': 1186071,
        'fp16_tensor_bytes': weight_bytes, 'vocoder_included': False,
        'new_text_frontend_included': False, 'full_linux_process_ram_verified': False,
        'precision': 'FP16 stored weights; reference CPU inference expands to FP32'})
    files = {name: {'sha256': digest(args.output / name), 'bytes': (args.output / name).stat().st_size}
             for name in targets if name != 'EXPORT.json'}
    write_json(args.output / 'EXPORT.json', {
        'status': 'ACOUSTIC_ONLY_EXTRACTED_AND_FP16_ROUNDED',
        'source_combined_checkpoint_sha256': sha,
        'source_acoustic_checkpoint_sha256': completed.get('checkpoint_sha256') if completed else None,
        'acoustic_training_steps': 80000, 'source_container_phase': cp.get('phase'),
        'source_container_step': cp.get('step'), 'source_identity': cp.get('identity'),
        'tensors': len(state), 'parameters': sum(t.numel() for t in state.values()),
        'fp16_tensor_bytes': weight_bytes, 'files': files,
        'contains_vocoder': False, 'contains_optimizer_or_rng': False,
        'contains_training_audio_or_features': False,
        'quality_pass': False, 'full_linux_process_ram_pass': False,
        'original_acoustic_optimizer_included': False,
        'export_torch_version': torch.__version__, 'export_numpy_version': np.__version__})
    print(json.dumps({'status': 'EXPORTED', 'tensors': len(state),
                      'parameters': sum(t.numel() for t in state.values()),
                      'fp16_tensor_bytes': weight_bytes, 'files': files}, indent=2))


if __name__ == '__main__':
    main()
