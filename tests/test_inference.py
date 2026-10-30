"""Run with: python -m unittest discover -s tests -v"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from new25_acoustic.inference import (
    LANGUAGES, AcousticPredictor, denormalize, frame_layout,
    load_safetensors, predict_durations, waveform_condition,
)
from new25_acoustic.model import Acoustic, FP32GRUCell


def normalizations():
    return {
        language: {
            "mean": [0.0] * (31 if language == "zh" else 39),
            "std": [1.0] * (31 if language == "zh" else 39),
            "phonetic_mean": [0.25] * 32, "phonetic_std": [0.5] * 32,
            "context_mean": [0.1] * 22, "context_std": [0.2] * 22,
        }
        for language in LANGUAGES
    }


def write_safetensors(path, state):
    header, payload = {}, bytearray()
    for name, tensor in state.items():
        value = tensor.detach().cpu().numpy().astype("<f2")
        begin = len(payload)
        payload.extend(value.tobytes())
        header[name] = {"dtype": "F16", "shape": list(value.shape), "data_offsets": [begin, len(payload)]}
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * ((-len(data)) % 8)
    path.write_bytes(struct.pack("<Q", len(data)) + data + payload)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.predictor = AcousticPredictor(Acoustic(4), normalizations(), {"<pad>": 0, "<unk>": 1, "sil": 2, "a": 3})
        cls.features = dict(phones=np.array([2, 3, 3]), phonetic=np.zeros((3, 32), np.float32),
            context=np.zeros((3, 22), np.float32), language="en", durations=np.array([0, 1, 3]))

    def test_original_frame_layout_zero_and_single_frame(self):
        ids, pos = frame_layout([0, 1, 3])
        np.testing.assert_array_equal(ids, [1, 2, 2, 2])
        np.testing.assert_allclose(pos, [[0, 1, 1, 1], [0, 1 / 3, 1, 0], [0.5, 1 / 3, 0, 0], [1, 1 / 3, 0, 1]])

    def test_duration_rule_pause_suppression_and_bounds(self):
        prosody = torch.tensor([[[0., 0., 0., 0., 0.], [2., 3., -0.1, 0., 0.], [-20., 20., 0., 0., 0.]]])
        duration = predict_durations(prosody, torch.tensor([[0., 1., 1.]]))
        torch.testing.assert_close(duration, torch.tensor([[1, 0, 400]]))
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            predict_durations(prosody, torch.tensor([[0., 0.5, 1.]]))

    def test_gru_gate_convention_matches_reference(self):
        cell = FP32GRUCell(3, 2)
        x, h = torch.randn(1, 3), torch.randn(1, 2)
        ar, az, an = cell.ih(x).float().chunk(3, -1)
        br, bz, bn = cell.hh(h).float().chunk(3, -1)
        r, z = torch.sigmoid(ar + br), torch.sigmoid(az + bz)
        torch.testing.assert_close(cell(x, h), (1 - z) * torch.tanh(an + r * bn) + z * h)

    def test_physical_vuv_and_condition_differ_from_head_logit(self):
        prediction = torch.zeros(1, 2, 31)
        prediction[0, 0, -1] = 2.
        norm = {"mean": [3.] * 31, "std": [2.] * 31}
        physical = denormalize(prediction, norm)
        torch.testing.assert_close(physical[..., 0], torch.full((1, 2), 3.))
        torch.testing.assert_close(physical[..., -1], torch.sigmoid(prediction[..., -1]))
        condition = waveform_condition(physical, norm)
        torch.testing.assert_close(condition[..., -1], (physical[..., -1] - 3.) / 2.)

    def test_single_sentence_shapes_state_and_chunk_equivalence(self):
        result = self.predictor.predict(**self.features)
        chunked = self.predictor.predict(**self.features, chunk_frames=1)
        self.assertEqual(result.physical.shape, (4, 39))
        self.assertEqual(result.acoustic_state["gru"].shape, (1, 192))
        self.assertEqual([x.shape for x in result.acoustic_state["conv"]], [(1, 4, 192), (1, 8, 192), (1, 16, 192)])
        np.testing.assert_allclose(result.prediction, chunked.prediction, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result.acoustic_state["gru"], chunked.acoustic_state["gru"], rtol=1e-5, atol=1e-6)
        self.assertTrue(np.isfinite(result.normalized).all())
        self.assertTrue(np.all((result.physical[:, -1] >= 0) & (result.physical[:, -1] <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.npz"
            result.save(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["physical"].shape, (4, 39))
                self.assertEqual(data["acoustic_state_conv_2"].shape, (1, 16, 192))

    def test_chinese_uses_31_dimensional_head(self):
        inputs = dict(self.features, language="zh")
        self.assertEqual(self.predictor.predict(**inputs).physical.shape, (4, 31))

    def test_raw_inputs_are_normalized_before_encoding(self):
        result = self.predictor.predict(**self.features)
        norm = self.predictor.normalization["en"]
        phones = torch.tensor([[2, 3, 3]])
        phonetic = torch.from_numpy((self.features["phonetic"] - np.float32(0.25)) / np.float32(0.5))[None]
        context = torch.from_numpy((self.features["context"] - np.float32(0.1)) / np.float32(0.2))[None]
        ids, pos = frame_layout(self.features["durations"])
        with torch.inference_mode():
            e, p = self.predictor.model.encode(phones, phonetic, context, torch.tensor([5]), torch.ones(1, 3))
            expected, _ = self.predictor.model.decode(e, p, torch.from_numpy(ids)[None], torch.from_numpy(pos)[None], torch.tensor([5]), 39)
        np.testing.assert_array_equal(result.prediction, expected[0].numpy())

    def test_reject_invalid_features_timelines_and_language(self):
        cases = [
            {"language": "xx"}, {"phones": [2, -1, 3]}, {"phones": [2, 0, 3]},
            {"phones": [2, 1.1, 3]}, {"phones": [2, 4, 3]},
            {"phonetic": np.zeros((3, 36))}, {"context": np.full((3, 22), np.nan)},
            {"durations": [0, 0, 0]}, {"durations": [1, 1]}, {"durations": [1, 1, -1]},
            {"durations": [1, 1, 20000]}, {"durations": [1, 1, 0.5]},
            {"chunk_frames": 0}, {"pause": [0, 0.5, 1]},
        ]
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.predictor.predict(**dict(self.features, **override))
        inputs = dict(self.features)
        inputs.pop("durations")
        with self.assertRaisesRegex(ValueError, "pause"):
            self.predictor.predict(**inputs)

    def test_safetensors_f16_roundtrip_and_strict_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.predictor.model.state_dict()
            write_safetensors(root / "acoustic_fp16.safetensors", state)
            loaded = load_safetensors(root / "acoustic_fp16.safetensors")
            self.assertEqual(len(loaded), 97)
            for name in state:
                torch.testing.assert_close(loaded[name], state[name].half().float(), rtol=0, atol=0)
            (root / "normalization.json").write_text(json.dumps(normalizations()))
            (root / "vocabulary.json").write_text(json.dumps(self.predictor.vocabulary))
            predictor = AcousticPredictor.from_pretrained(root)
            self.assertEqual(predictor.predict(**self.features).physical.shape, (4, 39))
            state.pop("phone.weight")
            write_safetensors(root / "acoustic_fp16.safetensors", state)
            with self.assertRaisesRegex(ValueError, "97 acoustic"):
                AcousticPredictor.from_pretrained(root)

    def test_safetensors_rejects_overlapping_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            header = json.dumps({"a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "b": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                load_safetensors(path)

    def test_shipped_weights_contract_when_present(self):
        root = Path(__file__).resolve().parents[1] / "models"
        if not (root / "acoustic_fp16.safetensors").exists():
            self.skipTest("Published weight bundle is not present")
        predictor = AcousticPredictor.from_pretrained(root)
        self.assertEqual(len(predictor.model.state_dict()), 97)
        self.assertEqual(sum(p.numel() for p in predictor.model.parameters()), 1186071)
        self.assertEqual(len(predictor.vocabulary), 285)
        for language in LANGUAGES:
            norm = predictor.normalization[language]
            result = predictor.predict([2, 3], np.zeros((2, len(norm["phonetic_mean"])), np.float32),
                np.zeros((2, len(norm["context_mean"])), np.float32), language, durations=[1, 2])
            self.assertEqual(result.physical.shape, (3, 31 if language == "zh" else 39))
            self.assertTrue(np.isfinite(result.normalized).all())


if __name__ == "__main__":
    unittest.main()
