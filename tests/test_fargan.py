"""Selected-weight offline interface tests; no training or quality evaluation."""

import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np
import torch

try:
    import scipy
except ModuleNotFoundError:
    HAS_SCIPY = False
else:
    HAS_SCIPY = True
    from new25_acoustic.inference import AcousticPredictor, LANGUAGES
    from new25_fargan import FARGANVocoder
    from new25_fargan.inference import main, sample_layout

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_SCIPY, "Install the optional fargan extra for waveform checks")
class FARGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = FARGANVocoder.from_pretrained(ROOT / "models/fargan")

    def test_released_weights_all_languages_and_valid_center(self):
        self.assertEqual(len(self.model.state_dict()), 91)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 2708937)
        for language in LANGUAGES:
            dimension = 31 if language == "zh" else 39
            pcm = self.model.synthesize(np.zeros((1, 8, dimension), np.float32), language, seed=71)
            self.assertEqual(pcm.shape, (1, 640))
            self.assertEqual(pcm.dtype, torch.float32)
            self.assertTrue(torch.isfinite(pcm).all())
        self.assertEqual(sample_layout(8)["source_sample_start"], 400)
        self.assertEqual(sample_layout(8)["source_sample_end"], 1040)

    def test_repeatability_rng_and_stream_contract(self):
        x = np.zeros((1, 5, 31), np.float32)
        self.assertTrue(self.model.streaming)
        rng = torch.random.get_rng_state().clone()
        a = self.model.synthesize(x, "zh", seed=91)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(a, self.model.synthesize(x, "zh", seed=91), rtol=0, atol=0)
        self.assertFalse(torch.equal(a, self.model.synthesize(x, "zh", seed=92)))
        for kwargs in ({"state": {}}, {"seed": True}, {"seed": -1}):
            with self.assertRaises(ValueError):
                self.model.synthesize(x, "zh", **kwargs)
        for chunk_frames in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.model.synthesize(x, "zh", chunk_frames=chunk_frames)
        self.assertEqual(self.model.synthesize(x, "zh", chunk_frames=1).shape, (1, 160))
        # The persistent stream keeps every recurrent/noise/de-emphasis state.
        state = self.model.begin_stream("zh", seed=91)
        pieces = []
        for begin in (0, 2, 4):
            pcm, state = self.model.synthesize_chunk(
                x[:, begin:begin + 2], "zh", state=state, final=begin + 2 >= x.shape[1])
            pieces.append(pcm)
        streamed = torch.cat(pieces, dim=1)
        self.assertEqual(streamed.shape, (1, 160))
        self.assertTrue(torch.isfinite(streamed).all())
        report = state.report()
        self.assertEqual((report["received_frames"], report["output_samples"], report["finished"]),
                         (5, 160, True))
        with self.assertRaises(ValueError):
            self.model.synthesize_chunk(x, "zh", state=state)
        with self.assertRaises(ValueError):
            self.model.synthesize_chunk(x, "en")
        # Original offline reference stays available and separate.
        offline = self.model.synthesize_offline(x, "zh", seed=91)
        self.assertEqual(offline.shape, (1, 160))
        self.assertTrue(torch.isfinite(offline).all())

    def test_invalid_inputs_and_short_timeline(self):
        for x, language in [(np.zeros((1, 4, 31)), "zh"),
                            (np.zeros((1, 5, 39)), "zh"),
                            (np.zeros((2, 5, 31)), "zh"),
                            (np.full((1, 5, 31), np.nan), "zh"),
                            (np.zeros((1, 5, 31)), [24.0])]:
            with self.assertRaises(ValueError):
                self.model.synthesize(x, language)

    def test_existing_acoustic_api_to_cli_wav_and_alignment_sidecar(self):
        acoustic = AcousticPredictor.from_pretrained(ROOT / "models")
        for language in ("en", "zh"):
            norm = acoustic.normalization[language]
            result = acoustic.predict(np.array([2, 3]),
                                      np.tile(norm["phonetic_mean"], (2, 1)),
                                      np.tile(norm["context_mean"], (2, 1)),
                                      language, durations=np.array([4, 4]))
            expected = self.model.synthesize(result.normalized[None], language, seed=101)
            self.assertEqual(expected.shape, (1, 640))
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "acoustic.npz"
                dest = Path(directory) / "fargan.wav"
                result.save(source)
                main(["--acoustic", str(source), "--language", language, "--model-dir",
                      str(ROOT / "models/fargan"), "--seed", "101", "--output", str(dest)])
                with wave.open(str(dest), "rb") as handle:
                    self.assertEqual((handle.getnchannels(), handle.getsampwidth(),
                                      handle.getframerate(), handle.getnframes()), (1, 2, 16000, 640))
                    expected_pcm = np.rint(np.clip(expected.numpy(), -1, 1) * 32768).clip(-32768, 32767).astype("<i2")
                    np.testing.assert_array_equal(np.frombuffer(handle.readframes(640), dtype="<i2"), expected_pcm[0])
                metadata = json.loads(Path(str(dest) + ".json").read_text())
                self.assertEqual(metadata["source_sample_start"], 400)
                self.assertEqual(metadata["source_sample_end"], 1040)
                self.assertEqual(metadata["selected_checkpoint_step"], 4750)
                self.assertTrue(metadata["streaming"])
                self.assertFalse(metadata["artifact_streaming"])


if __name__ == "__main__":
    unittest.main()
