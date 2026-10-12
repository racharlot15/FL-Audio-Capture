import math
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from capture import analyze_wav, compare_reports, measure_stereo


class StereoMeasurementTests(unittest.TestCase):
    def test_channel_levels_difference_and_flags(self):
        audio = np.column_stack((np.full(1000, 0.5), np.full(1000, 0.25))).astype(np.float32)
        result = measure_stereo(audio)
        self.assertAlmostEqual(result["channels"][0]["rms_dbfs"], 20 * math.log10(0.5))
        self.assertAlmostEqual(result["channels"][1]["peak_dbfs"], 20 * math.log10(0.25))
        self.assertAlmostEqual(result["left_right_rms_delta_db"], 20 * math.log10(2))
        self.assertAlmostEqual(result["waveform_difference_rms"], 0.25)
        self.assertFalse(result["any_silence"])
        self.assertFalse(result["any_clipping"])

    def test_silence_and_clipping_are_reported_per_channel(self):
        audio = np.column_stack((np.zeros(10), np.array([1.0] + [0.0] * 9))).astype(np.float32)
        result = measure_stereo(audio)
        self.assertTrue(result["channels"][0]["silent"])
        self.assertTrue(result["channels"][1]["clipping"])
        self.assertEqual(result["channels"][1]["clipped_samples"], 1)
        self.assertIsNone(result["channels"][0]["rms_dbfs"])
        json.dumps(result, allow_nan=False)

    def test_requires_stereo_nonempty_audio(self):
        with self.assertRaises(ValueError):
            measure_stereo(np.zeros((10, 1)))
        with self.assertRaises(ValueError):
            measure_stereo(np.zeros((0, 2)))


class WavAnalysisTests(unittest.TestCase):
    def test_timeline_silence_levels_and_clipping(self):
        quiet = np.zeros((8000, 2), dtype="<i2")
        loud = np.column_stack((np.full(7999, 16384), np.full(7999, 8192)))
        loud = np.vstack((loud, np.array([[32767, 8192]], dtype="<i2"))).astype("<i2")
        pcm = np.vstack((quiet, loud)).astype("<i2")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "sample.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(pcm.tobytes())
            report = analyze_wav(path, window_seconds=0.5)

        self.assertEqual(len(report["timeline"]), 4)
        self.assertTrue(report["timeline"][0]["left_silent"])
        self.assertTrue(report["timeline"][1]["left_silent"])
        self.assertFalse(report["timeline"][2]["left_silent"])
        self.assertAlmostEqual(report["channels"][0]["silence_duration_seconds"], 1.0)
        self.assertEqual(report["channels"][0]["clipped_samples"], 1)
        self.assertTrue(report["any_clipping"])
        json.dumps(report, allow_nan=False)

    def test_mix_details_detect_tone_stereo_width_and_crest_factor(self):
        rate = 48000
        time = np.arange(rate, dtype=np.float64) / rate
        left = 0.4 * np.sin(2 * math.pi * 750 * time)
        right = 0.2 * np.sin(2 * math.pi * 750 * time)
        pcm = np.column_stack((left, right))
        pcm16 = np.round(pcm * 32767).astype("<i2")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "tone.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm16.tobytes())
            report = analyze_wav(path, window_seconds=0.25)

        mid_band = next(band for band in report["frequency_bands"]
                        if band["band"] == "mid_500_2000_hz")
        self.assertGreater(mid_band["left_energy_percent"], 99.0)
        self.assertAlmostEqual(mid_band["left_rms_dbfs"], 20 * math.log10(0.4 / math.sqrt(2)), delta=0.05)
        self.assertGreater(report["stereo_correlation"], 0.999)
        self.assertAlmostEqual(report["stereo_image"]["side_to_mid_db"], -9.54, delta=0.1)
        self.assertAlmostEqual(report["channels"][0]["crest_factor_db"], 3.01, delta=0.1)
        self.assertGreater(report["timeline"][0]["frequency_bands"][3]["left_energy_percent"], 99.0)

    def test_comparison_uses_measured_change_and_listening_preference(self):
        before = {
            "wav_file": "before.wav",
            "channels": [
                {"rms_dbfs": -16.0, "peak_dbfs": -2.0},
                {"rms_dbfs": -16.2, "peak_dbfs": -1.8},
            ],
            "frequency_bands": [
                {"band": "sub_bass_20_60_hz", "left_rms_dbfs": -18.0, "right_rms_dbfs": -18.2},
            ],
            "any_clipping": False,
        }
        trial = {
            "wav_file": "trial.wav",
            "channels": [
                {"rms_dbfs": -18.0, "peak_dbfs": -4.0},
                {"rms_dbfs": -18.2, "peak_dbfs": -3.8},
            ],
            "frequency_bands": [
                {"band": "sub_bass_20_60_hz", "left_rms_dbfs": -20.0, "right_rms_dbfs": -20.2},
            ],
            "any_clipping": False,
        }
        result = compare_reports(before, trial, "tighter")
        self.assertAlmostEqual(result["frequency_band_changes"][0]["change_db"], -2.0)
        self.assertIn("Keep the trial setting", result["recommendation"])
        self.assertIn("tighter feel", result["recommendation"])

    def test_requires_stereo_pcm_with_audio(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "mono.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(np.zeros(8, dtype="<i2").tobytes())
            with self.assertRaisesRegex(ValueError, "Expected stereo"):
                analyze_wav(path)


if __name__ == "__main__":
    unittest.main()
