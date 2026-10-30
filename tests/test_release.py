import hashlib
import json
from pathlib import Path
import unittest

import torch

from new25_acoustic.vocoder import VocoderStub, validate_condition, validate_output
from scripts.export_acoustic import validate_training_identity


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_release_export_rejects_partial_or_smoke_training(self):
        for checkpoint in [{'phase': 'acoustic', 'step': 1, 'smoke_test': False},
                           {'phase': 'acoustic', 'step': 80000, 'smoke_test': True},
                           {'phase': 'acoustic', 'step': 80000}]:
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                validate_training_identity(checkpoint, 'sha', {})
        validate_training_identity({'phase': 'acoustic', 'step': 80000, 'smoke_test': False}, 'sha', {})
        with self.assertRaises(ValueError):
            validate_training_identity({'phase': 'vocoder_pretrain', 'step': 60000, 'smoke_test': False}, 'sha', {})

    def test_fp32_payload_is_exactly_the_verified_acoustic_state(self):
        state = torch.load(ROOT / 'models/acoustic_fp32.pt', map_location='cpu', weights_only=True)
        proof = json.loads((ROOT / 'receipts/acoustic_provenance.json').read_text())
        self.assertEqual(set(state), set(proof['tensors']))
        self.assertEqual(len(state), 97)
        for name, tensor in state.items():
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(list(tensor.shape), proof['tensors'][name]['shape'])
            self.assertEqual(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
                             proof['tensors'][name]['sha256'])
        self.assertEqual(sum(t.numel() for t in state.values()), 1186071)

    def test_reserved_vocoder_rejects_head_logits_and_never_fabricates_pcm(self):
        normalized = torch.zeros(1, 2, 31)
        physical = torch.zeros_like(normalized)
        lang = torch.tensor([24], dtype=torch.long)
        validate_condition(normalized, physical, lang)
        with self.assertRaises(NotImplementedError):
            VocoderStub()(normalized, physical, lang)
        physical[..., -1] = -0.1
        with self.assertRaises(ValueError):
            validate_condition(normalized, physical, lang)
        physical[..., -1] = 0.5
        with self.assertRaises(ValueError):
            validate_condition(normalized, physical, torch.tensor([0]))

    def test_vocoder_output_must_follow_the_frame_timeline(self):
        condition = torch.zeros(1, 3, 39)
        validate_output(torch.zeros(1, 480), condition)
        with self.assertRaises(ValueError):
            validate_output(torch.zeros(1, 479), condition)
        with self.assertRaises(ValueError):
            validate_output(torch.full((1, 480), float('nan')), condition)


if __name__ == '__main__':
    unittest.main()
