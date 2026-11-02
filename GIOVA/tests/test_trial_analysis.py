"""Run: .venv/Scripts/python -m unittest discover -s tests -v"""
import csv
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
import uuid
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TEST_TEMP = Path(__file__).resolve().parent / ".tmp"
TEST_TEMP.mkdir(exist_ok=True)


@contextmanager
def scratch():
    # Python 3.13's Windows 0700 temporary-directory ACL excludes the sandbox
    # identity. An ordinary workspace directory inherits usable permissions.
    directory = TEST_TEMP / uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        # sub-folders too, and never an error of its own that would hide the test's
        import shutil
        shutil.rmtree(directory, ignore_errors=True)

import numpy as np
from scipy.signal import welch

from eeg.core.sources import Recording
from eeg.core.trials import (Settings, Trial, analyze, measure, periodic_trials,
                             prepare_channel, trial_status)
from eeg.formats.recordings import load_recording
from eeg.formats.trial_analysis import (export_csv, load_session, power_comparison,
                                       save_session)


class TrialMathTests(unittest.TestCase):
    def setUp(self):
        self.fs = 128
        self.settings = Settings(duration=4, window=2)
        self.time = np.arange(512) / self.fs
        self.wave = 3 * np.sin(2 * np.pi * 10 * self.time)

    def test_fft_amplitude_power_and_welch(self):
        result = measure(self.wave + 900, self.fs, self.settings)
        self.assertEqual(result.fft_freqs[result.fft.argmax()], 10)
        self.assertAlmostEqual(result.fft.max(), 3)
        self.assertAlmostEqual(result.power, 4.5)
        f, p = welch(self.wave, fs=self.fs, window="hann", nperseg=256,
                     noverlap=128, detrend="constant", scaling="density")
        np.testing.assert_array_equal(f, result.psd_freqs)
        np.testing.assert_allclose(p, result.psd, atol=1e-12)
        self.assertAlmostEqual(np.trapezoid(result.psd, result.psd_freqs), 4.5)
        self.assertAlmostEqual(result.bands["alpha"], 4.5)
        doubled = measure(self.wave * 2, self.fs, self.settings)
        self.assertAlmostEqual(doubled.power, result.power * 4)

    def test_fft_even_nyquist_and_odd_last_bin(self):
        even = measure(2 * (-1.0) ** np.arange(128), 128, Settings(duration=1))
        self.assertAlmostEqual(even.fft[-1], 2)
        n = 127
        odd = measure(2 * np.cos(2 * np.pi * 63 * np.arange(n) / n), 127, Settings(duration=1))
        self.assertAlmostEqual(odd.fft[-1], 2)

    def test_opposite_phase_separates_two_averages(self):
        result = analyze(np.r_[self.wave, -self.wave][None, :], self.fs, 0, "O1",
                         [Trial("1", "open", 0, 4), Trial("2", "open", 4, 8)], self.settings)
        condition = result.conditions[0]
        np.testing.assert_allclose(condition.signal, 0, atol=1e-14)
        self.assertLess(condition.mean_signal.power, 1e-25)
        self.assertAlmostEqual(condition.mean_trials.power, 4.5)
        self.assertAlmostEqual(condition.mean_trials.fft.max(), 3)
        self.assertAlmostEqual(condition.mean_trials.bands["alpha"], 4.5)

    def test_selection_short_trial_and_condition_isolation(self):
        data = np.r_[self.wave, 2 * self.wave, 100 * self.wave, self.wave[:128]][None, :]
        trials = [Trial("1", "open", 0, 4), Trial("2", "closed", 4, 8),
                  Trial("3", "open", 8, 12, False), Trial("4", "closed", 12, 13)]
        result = analyze(data, self.fs, 0, "O1", trials, self.settings)
        self.assertEqual([t.trial.id for t in result.trials], ["1", "2"])
        self.assertEqual(result.rejected["3"], "Excluded by selection")
        self.assertEqual(result.rejected["4"], "Too short for common duration")
        self.assertAlmostEqual(result.conditions[0].mean_trials.power, 4.5)
        self.assertAlmostEqual(result.conditions[1].mean_trials.power, 18)
        rows = power_comparison(result.conditions, "mean_trials", "open")
        closed = next(r for r in rows if r[0] == "closed" and r[2] == "Mean square")
        self.assertAlmostEqual(closed[4], 13.5)
        self.assertAlmostEqual(closed[5], 300)

    def test_sample_boundaries_and_equal_length(self):
        settings = Settings(duration=1)
        data = np.arange(512)[None, :]
        trials = [Trial("1", "a", 0, 2), Trial("2", "a", 2, 4)]
        result = analyze(data, 128, 0, "O1", trials, settings)
        continuous = prepare_channel(data[0], 128, settings)
        np.testing.assert_array_equal(result.trials[0].signal, continuous[:128])
        np.testing.assert_array_equal(result.trials[1].signal, continuous[256:384])
        self.assertEqual(len(result.times), 128)
        self.assertEqual(trial_status(Trial("x", "a", -1, 1), 128, 512, 1)[2], "Require 0 <= start < end")
        self.assertEqual(trial_status(Trial("x", "a", 3, 5), 128, 512, 1)[2], "Outside recording")
        self.assertEqual(trial_status(Trial("x", "a", 3, 4), 128, 512, 1)[2], "Ready")

    def test_filter_is_applied_before_slicing(self):
        settings = Settings(duration=4, filters=dict(low=1, high=40, order=4))
        data = np.r_[self.wave, self.wave * 2][None, :]
        result = analyze(data, self.fs, 0, "O1", [Trial("1", "a", 4, 8)], settings)
        expected = prepare_channel(data[0], self.fs, settings)[512:1024]
        np.testing.assert_array_equal(result.trials[0].signal, expected)

    def test_generation_and_partial_tail(self):
        trials = periodic_trials(2, [("open", 30), ("closed", 30)], 4, 97)
        self.assertEqual([(t.start, t.end) for t in trials], [(2, 32), (32, 62), (62, 92), (92, 97)])
        self.assertEqual([t.condition for t in trials], ["open", "closed", "open", "closed"])
        self.assertEqual(trial_status(trials[-1], 128, 97 * 128, 30)[2], "Too short for common duration")

    def test_empty_and_invalid_settings_rejected(self):
        for trials in ([], [Trial("1", "a", 0, 4, False)], [Trial("1", "a", 0, 1)]):
            with self.assertRaises(ValueError):
                analyze(self.wave[None, :], self.fs, 0, "O1", trials, self.settings)
        with self.assertRaises(ValueError):
            analyze(self.wave[None, :], self.fs, 0, "O1", [Trial("1", "a", 0, 4)], Settings(overlap=1))
        with self.assertRaises(ValueError):
            prepare_channel(self.wave, self.fs, Settings(filters=dict(low=40, high=10)))

    def test_zero_reference_is_undefined(self):
        result = analyze(np.zeros((1, 512)), 128, 0, "O1", [Trial("1", "a", 0, 4)], self.settings)
        self.assertTrue(all(row[-1] is None for row in power_comparison(result.conditions, "mean_trials", "a")))

    def test_session_roundtrip_and_incompatible_recording(self):
        recording = Recording(self.wave[None, :], self.fs, ["O1"])
        trials = [Trial("1", "open", 0, 4)]
        with scratch() as directory:
            path = Path(directory) / "selection.json"
            source = Path(directory) / "recording.csv"
            save_session(path, recording, source, "O1", trials, self.settings, {"mode": 1})
            channel, restored, settings, display = load_session(path, recording, source)
            self.assertEqual((channel, restored, settings, display), (["O1"], trials, self.settings, {"mode": 1}))
            for other in (Recording(self.wave[None, :], 256, ["O1"]),
                          Recording(self.wave[None, :-1], 128, ["O1"]),
                          Recording(self.wave[None, :], 128, ["O2"])):
                with self.assertRaises(ValueError):
                    load_session(path, other, source)
            with self.assertRaises(ValueError):
                load_session(path, recording, "another.csv")
            payload = json.loads(path.read_text())
            payload["version"] = 99
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_session(path, recording, source)

    def test_csv_has_individuals_both_averages_units_and_settings(self):
        result = analyze(self.wave[None, :], self.fs, 0, "O1", [Trial("1", "open", 0, 4)], self.settings)
        with scratch() as directory:
            path = Path(directory) / "out.csv"
            export_csv(path, result, "example.csv", "open")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            calculations = {row["calculation"] for row in rows}
            self.assertTrue({"Individual trial", "Analysis of mean signal", "Mean of trial analyses"} <= calculations)
            self.assertTrue(all(row["channel"] == "O1" and row["duration_s"] == "4.0" for row in rows))
            self.assertEqual(next(r for r in rows if r["quantity"] == "PSD")["unit"], "uV^2/Hz")
            self.assertTrue(any(r["reference_condition"] == "open" for r in rows))


#: a real eyes open / closed recording with its events, not part of the repository
ALPHA = Path(os.environ.get("GIOVA_ALPHA_RECORDING", Path(__file__).resolve().parents[1] / "Data" / "alpha.csv"))


@unittest.skipUnless(ALPHA.is_file(), "no eyes open / closed recording (set GIOVA_ALPHA_RECORDING)")
class WorkbenchRealRecordingTests(unittest.TestCase):
    """A real eyes open / closed recording with the built-in Eyes open / closed profile, as a quick analysis on the
    Workbench: the same trials and the same numbers as core.trials.analyze (the Analysis page's computation)."""

    def test_real_recording_o1_o2_events_and_both_calculations(self):
        from eeg.core import analysis_profiles as ap
        from eeg.pipeline import Runner, compute
        from eeg.pipeline.convert import from_profile
        path = ALPHA
        eyes = next(p for p in ap.builtin_profiles() if p.name == "Eyes open / closed")
        graph, notes = from_profile(eyes, path)
        graph.set("psd1", signal_mean=True)
        outputs = Runner(graph).run()
        epochs = outputs["epochs1"]["epochs"]
        psd = outputs["psd1"]["spectrum"]
        self.assertEqual(set(epochs.labels), {"open", "close"})
        self.assertEqual(psd.names, ["O1", "O2"])                # the profile's region
        self.assertTrue(all(m["count"] == 30 * epochs.fs for m in epochs.meta))
        raw = compute.read_recording(path)
        trials = compute.defined_trials(raw, "events", True)
        settings = Settings(duration=30.0, window=eyes.window, overlap=eyes.overlap,
                            filters=dict(eyes.filters), profile=dict(eyes.feature_profile))
        o1, o2 = (raw.channel_names.index(n) for n in ("O1", "O2"))
        page = analyze(raw.data, raw.fs, [o1, o2], "ROI O1+O2", trials, settings, names=raw.channel_names)
        self.assertEqual(page.channel, "ROI O1+O2")
        self.assertEqual(epochs.ids, [t.trial.id for t in page.trials])
        for k, item in enumerate(page.trials):
            np.testing.assert_allclose(psd.power[k].mean(axis=0), item.metrics.psd, rtol=1e-8, atol=1e-12)
        labels = np.asarray(psd.labels)
        for condition in page.conditions:
            np.testing.assert_allclose(psd.power[labels == condition.name].mean(axis=(0, 1)),
                                       condition.mean_trials.psd, rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(psd.signal_means[condition.name].mean(axis=0), condition.mean_signal.psd,
                                       rtol=1e-8, atol=1e-9)
            self.assertTrue(np.isfinite(condition.mean_signal.psd).all())


if __name__ == "__main__":
    unittest.main()
