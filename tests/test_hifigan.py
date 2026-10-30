"""Offline HiFi-GAN contract and numerical checks; no quality/MOS claims."""

import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

import numpy as np
import torch

try:
    from scipy.signal import resample_poly
except ModuleNotFoundError as error:
    if error.name != "scipy":
        raise
    HAS_SCIPY = False
else:
    HAS_SCIPY = True
    from new25_hifigan.frontend import Mel, mel_matrix, sample_layout, to_16k, to_native
    from new25_hifigan.inference import (
        HiFiGANVocoder, LANGUAGES, METADATA, load_acoustic_npz, main,
        validate_metadata, write_pcm16,
    )
    from new25_hifigan.model import GENERATOR_CONFIG, Generator


def write_safetensors(path, state):
    header, payload = {}, bytearray()
    for name, tensor in state.items():
        array = tensor.detach().cpu().numpy().astype("<f2")
        begin = len(payload)
        payload.extend(array.tobytes())
        header[name] = {"dtype": "F16", "shape": list(array.shape), "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def write_bundle(root, state):
    (root / "config.json").write_text(json.dumps(GENERATOR_CONFIG))
    (root / "model_config.json").write_text(json.dumps(METADATA))
    write_safetensors(root / "vocoder_adapter_fp16.safetensors", state)


@unittest.skipUnless(HAS_SCIPY, "Install the optional hifigan extra for SciPy waveform checks")
class HiFiGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cls.vocoder = HiFiGANVocoder()

    def test_architecture_counts_and_folded_names(self):
        model = self.vocoder
        self.assertEqual(sum(p.numel() for p in model.generator.parameters()), 925985)
        self.assertEqual(sum(p.numel() for p in model.physical_to_mel.parameters()), 1902992)
        self.assertEqual(len(model.generator.state_dict()), 156)
        self.assertEqual(len(model.physical_to_mel.state_dict()), 63)
        self.assertEqual(len(model.state_dict()), 219)
        self.assertFalse(any("weight_g" in key or "weight_v" in key for key in model.state_dict()))

    def test_generator_preserves_upstream_final_default_leaky_relu_slope(self):
        generator = Generator().eval()
        with torch.no_grad():
            for parameter in generator.parameters():
                parameter.zero_()
            for up in generator.ups:
                up.bias.fill_(-1)
        captured = []
        hook = generator.conv_post.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        try:
            with torch.inference_mode():
                result = generator(torch.zeros(1, 80, 2))
        finally:
            hook.remove()
        torch.testing.assert_close(captured[0], torch.full_like(captured[0], -.01), rtol=0, atol=0)
        self.assertEqual(result.shape, (1, 1, 512))

    def test_exact_lengths_and_pinned_polyphase_math(self):
        for frames in (1, 2, 3, 7, 16, 29, 160, 1001):
            with self.subTest(frames=frames):
                length, native_length, mel_frames = sample_layout(frames)
                self.assertEqual(length, frames * 160)
                self.assertEqual(native_length, math.ceil(length * 441 / 320))
                self.assertEqual(mel_frames, math.ceil(native_length / 256))
                signal = np.sin(np.arange(length, dtype=np.float32) * np.float32(.071))
                native, unpadded = to_native(signal)
                reference = resample_poly(signal, 441, 320).astype(np.float32)
                self.assertEqual(unpadded, native_length)
                self.assertEqual(len(native), mel_frames * 256)
                np.testing.assert_array_equal(native[:native_length], reference)
                np.testing.assert_array_equal(native[native_length:], 0)
                np.testing.assert_array_equal(to_16k(native, length),
                                              resample_poly(native, 320, 441).astype(np.float32)[:length])
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                sample_layout(value)
        with self.assertRaisesRegex(ValueError, "length loss"):
            to_16k(np.zeros(1, np.float32), 160)

    def test_adapter_interpolation_frame_centers_and_original_boundary(self):
        adapter = self.vocoder.physical_to_mel
        for dimension, language in ((31, 24), (39, 5)):
            x = torch.randn(1, 4, dimension)
            lang = torch.tensor([language])
            with torch.inference_mode():
                z = adapter.norm(adapter.inputs[str(dimension)](x.transpose(1, 2)).transpose(1, 2)
                                 + adapter.language(lang)[:, None])
                for block in adapter.blocks:
                    z = block(z, torch.ones(1, 4, 1))
                z = adapter.head(z).float().transpose(1, 2)
                result = adapter(x, lang, 6, context_frames=0)
                expected = []
                for j in range(6):
                    position = ((np.float32(j) * np.float32(256) + np.float32(128))
                                * np.float32(100) / np.float32(22050) - np.float32(.5))
                    low = min(max(int(np.floor(position)), 0), 3)
                    high = min(low + 1, 3)
                    fraction = float(position - np.floor(position))
                    expected.append(z[:, :, low] * (1 - fraction) + z[:, :, high] * fraction)
                torch.testing.assert_close(result, torch.stack(expected, -1), rtol=0, atol=0)

    def test_complete_utterance_shape_and_refusal_of_chunk_state(self):
        for language, dimension in (("en", 39), ("zh", 31)):
            for frames in (1, 2, 3, 7):
                with self.subTest(language=language, frames=frames):
                    audio = self.vocoder.synthesize(np.zeros((1, frames, dimension), np.float32),
                                                   torch.tensor([LANGUAGES.index(language)]))
                    self.assertEqual(audio.shape, (1, frames * 160))
                    self.assertEqual(audio.dtype, torch.float32)
                    self.assertEqual(audio.device.type, "cpu")
                    self.assertTrue(torch.isfinite(audio).all())
        for kwargs in ({"state": {}}, {"chunk_frames": 1}, {"chunk_frames": 0}):
            with self.assertRaisesRegex(ValueError, "complete utterance"):
                self.vocoder.synthesize(np.zeros((1, 2, 31)), "zh", **kwargs)
        with self.assertRaisesRegex(ValueError, "streaming/chunk"):
            self.vocoder.synthesize_chunk(None)

    def test_reject_incompatible_and_nonfinite_features(self):
        cases = [
            (np.zeros((2, 3, 31)), "zh"), (np.zeros((3, 31)), "zh"),
            (np.zeros((1, 0, 31)), "zh"), (np.zeros((1, 20001, 31)), "zh"),
            (np.zeros((1, 2, 39)), "zh"), (np.zeros((1, 2, 31)), "en"),
            (np.full((1, 2, 31), np.nan), "zh"), (np.full((1, 2, 31), 1e100), "zh"),
            (np.zeros((1, 2, 31)), [25]), (np.zeros((1, 2, 31)), [-1]),
            (np.zeros((1, 2, 31)), [24.0]), (np.zeros((1, 2, 31)), [True]),
            (np.zeros((1, 2, 31)), "xx"), (np.zeros((1, 2, 31)), 24),
        ]
        for features, language in cases:
            with self.subTest(shape=features.shape, language=language), self.assertRaises(ValueError):
                self.vocoder.synthesize(features, language)

    def test_strict_bundle_roundtrip_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.vocoder.state_dict()
            write_bundle(root, state)
            loaded = HiFiGANVocoder.from_pretrained(root)
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, state[name].half().float(), rtol=0, atol=0)
            self.assertTrue(all(p.dtype == torch.float32 for p in loaded.parameters()))
            torch.save(state, root / "model.pt")
            full = HiFiGANVocoder.from_pretrained(root, weights="model.pt")
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            wrong = dict(state)
            wrong.pop("generator.conv_pre.bias")
            torch.save(wrong, root / "model.pt")
            with self.assertRaisesRegex(ValueError, "219"):
                HiFiGANVocoder.from_pretrained(root, weights="model.pt")
            wrong = dict(state, **{"generator.conv_pre.bias": torch.zeros(1)})
            torch.save(wrong, root / "model.pt")
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                HiFiGANVocoder.from_pretrained(root, weights="model.pt")
            wrong = dict(state, **{"generator.conv_pre.bias": torch.full((128,), float("nan"))})
            torch.save(wrong, root / "model.pt")
            with self.assertRaisesRegex(ValueError, "dtype/value"):
                HiFiGANVocoder.from_pretrained(root, weights="model.pt")
            metadata = dict(METADATA, adapter_context_frames=1)
            (root / "model_config.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "adapter_context_frames"):
                HiFiGANVocoder.from_pretrained(root)
        for key, value in (("languages", list(reversed(LANGUAGES))), ("streaming", True),
                           ("output_sample_rate", 22050)):
            with self.assertRaisesRegex(ValueError, key):
                validate_metadata(dict(METADATA, **{key: value}))
        config = copy.deepcopy(GENERATOR_CONFIG)
        config["upsample_initial_channel"] = 256
        with self.assertRaisesRegex(ValueError, "upsample_initial_channel"):
            Generator(config)

    def test_acoustic_npz_contract_and_cli_pcm16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "acoustic.npz"
            features = dict(normalized=np.zeros((3, 31), np.float32), physical=np.zeros((3, 31), np.float32),
                            language=np.asarray("zh"), language_id=np.asarray(24), durations=np.array([1, 2]),
                            sample_rate=np.asarray(16000), hop_size=np.asarray(160))
            np.savez(path, **features)
            self.assertEqual(load_acoustic_npz(path, "zh").shape, (1, 3, 31))
            write_bundle(root, self.vocoder.state_dict())
            wav = root / "out.wav"
            main(["--acoustic", str(path), "--language", "zh", "--model-dir", str(root), "--output", str(wav)])
            with wave.open(str(wav), "rb") as handle:
                self.assertEqual((handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()),
                                 (1, 2, 16000, 480))
            for change in ({"durations": np.array([1, 3])}, {"sample_rate": np.asarray(22050)},
                           {"language": np.asarray("en")}, {"physical": np.zeros((2, 31))},
                           {"lang": np.array([5])}):
                np.savez(path, **dict(features, **change))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_acoustic_npz(path, "zh")
            np.savez(path, physical=np.zeros((3, 31)))
            with self.assertRaisesRegex(ValueError, "normalized"):
                load_acoustic_npz(path, "zh")
            write_pcm16(wav, np.array([-2., -1., 0., .5, 1., 2.], np.float32))
            with wave.open(str(wav), "rb") as handle:
                np.testing.assert_array_equal(np.frombuffer(handle.readframes(6), "<i2"),
                                              [-32768, -32768, 0, 16384, 32767, 32767])

    def test_true_mel_frontend_is_finite_and_not_a_synthesis_requirement(self):
        basis = mel_matrix()
        self.assertEqual(basis.shape, (80, 513))
        self.assertTrue(np.isfinite(basis).all())
        native, _ = to_native(np.zeros(1600, np.float32))
        mel = Mel()(torch.from_numpy(native)[None])
        self.assertEqual(mel.shape, (1, 80, len(native) // 256))
        torch.testing.assert_close(mel, torch.full_like(mel, math.log(1e-5)))

    def test_shipped_weights_contract_when_present(self):
        root = Path(__file__).resolve().parents[1] / "models" / "hifigan"
        if not (root / "vocoder_adapter_fp16.safetensors").exists():
            self.skipTest("Published HiFi-GAN weight bundle is not present")
        vocoder = HiFiGANVocoder.from_pretrained(root)
        for language in LANGUAGES:
            dimension = 31 if language == "zh" else 39
            audio = vocoder.synthesize(np.zeros((1, 3, dimension), np.float32), language)
            self.assertEqual(audio.shape, (1, 480))
            self.assertTrue(torch.isfinite(audio).all())
        fp32 = root / "vocoder_adapter_fp32.pt"
        if fp32.exists():
            full = HiFiGANVocoder.from_pretrained(root, weights=fp32.name)
            for name, value in vocoder.state_dict().items():
                torch.testing.assert_close(value, full.state_dict()[name].half().float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
