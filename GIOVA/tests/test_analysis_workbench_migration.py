"""The Analysis page's trial analysis carried onto the Workbench without losing what it meant.

Each test compares the Workbench path (pipeline.compute, as the blocks and an exported script run it)
with the Analysis page's own core functions (core.trials, core.epochs) on the same samples and the same
trials: ids, conditions, start / end, the common duration, reasons, explicit notches, detrend and the
time-frequency settings. T-numbers refer to the migration plan's acceptance matrix.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_pipeline import save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core import analysis_profiles as ap  # noqa: E402
from eeg.core import epochs as ep  # noqa: E402
from eeg.core.processing import filter_finite  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.core.trials import Settings, Trial, analyze, periodic_trials, trial_status  # noqa: E402
from eeg.pipeline import Graph, Runner, compute  # noqa: E402
from eeg.pipeline.convert import from_profile, from_session  # noqa: E402
from eeg.pipeline.export import to_python  # noqa: E402

RTOL, ATOL = 1e-8, 1e-12


def recording(fs=500.0, seconds=80.0, seed=1, events=None):
    """Three channels: 10 Hz, 50 Hz mains, a slow drift and noise, different on each channel."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    rows = [(5 + k) * np.sin(2 * np.pi * 10 * t + k) + 30 * np.sin(2 * np.pi * 50 * t) + 4 * t
            + rng.normal(0, 3, t.size) for k in range(3)]
    return Recording(np.vstack(rows), fs, ["C3", "Cz", "C4"], events=events)


def table(trials):
    """Workbench trial-table rows from core Trials."""
    return [[t.id, t.condition, t.start, t.end, t.included] for t in trials]


class ManualTrialTests(unittest.TestCase):
    """T02, T03: trials typed by hand keep their ids, conditions and selection, judged as the page judges them."""

    def setUp(self):
        self.raw = recording()
        self.trials = [Trial("rest-1", "Rest", 1.0, 11.0), Trial("T 2", "Task", 12.25, 25.0),
                       Trial("3", "Rest", 30.0, 38.0),                       # shorter than the common 10 s
                       Trial("x", "Task", 40.0, 50.0, included=False),       # left out of the selection
                       Trial("late", "Task", 75.0, 85.0),                    # beyond the recording
                       Trial("ok", "Task", 60.004, 70.5)]                    # a start between samples

    def test_ids_conditions_and_reasons_are_those_of_trial_status(self):
        epochs = compute.cut_epochs(self.raw, tmin=0.0, tmax=10.0, source="manual", trials=table(self.trials))
        expected = {t.id: trial_status(t, self.raw.fs, self.raw.n_samples, 10.0) for t in self.trials}
        self.assertEqual(epochs.ids, ["rest-1", "T 2", "ok"])
        self.assertEqual(epochs.labels, ["Rest", "Task", "Task"])
        self.assertEqual({i: r for i, _, r in epochs.rejected},
                         {i: s[2] for i, s in expected.items() if s[2] != "Ready"})
        for row, meta in zip(epochs.data, epochs.meta):
            start, count, _ = expected[meta["id"]]
            self.assertEqual((meta["sample"], meta["count"]), (start, count))
            np.testing.assert_array_equal(row, self.raw.data[:, start:start + count])
        # the trial as defined is kept beside the samples used: no padding, no quiet change of length
        self.assertEqual((epochs.meta[1]["start"], epochs.meta[1]["end"]), (12.25, 25.0))
        self.assertEqual(epochs.data.shape[-1], 5000)

    def test_an_empty_table_is_refused_not_replaced_by_the_events(self):
        raw = recording(events=ep.EventTable([1.0, 20.0], [5.0, 5.0], ["A", "B"]))
        with self.assertRaisesRegex(ValueError, "trial table is empty"):
            compute.cut_epochs(raw, tmin=0.0, tmax=4.0, source="manual", trials=[])

    def test_select_trials_takes_ids_that_are_not_numbers(self):
        epochs = compute.cut_epochs(self.raw, tmin=0.0, tmax=10.0, source="manual", trials=table(self.trials))
        kept = compute.select_trials(epochs, keep=["T 2", "ok"])
        self.assertEqual(kept.ids, ["T 2", "ok"])
        self.assertEqual([m["id"] for m in kept.meta], kept.ids)
        np.testing.assert_array_equal(kept.data, epochs.data[1:])
        self.assertEqual(compute.trial_text(["1", "2", "3", "rest-1"]), ["1-3", "rest-1"])

    def test_the_spectra_are_the_analysis_pages_per_trial_spectra(self):
        """Same filter (explicit notch), same samples, same Welch: each trial's PSD and FFT as analyze gives."""
        settings = Settings(duration=10.0, window=2.0, overlap=0.5,
                            filters=dict(low=1.0, high=45.0, order=4, notches=[50.0], notch_q=30.0))
        result = analyze(self.raw.data, self.raw.fs, 0, "C3", self.trials, settings, names=self.raw.channel_names)
        filtered = compute.filter_recording(self.raw, low=1.0, high=45.0, notch=0.0, notches=[50.0])
        epochs = compute.cut_epochs(filtered, tmin=0.0, tmax=10.0, source="manual", trials=table(self.trials))
        psd = compute.welch_spectrum(epochs, window=2.0, overlap=0.5)
        fft = compute.fft_spectrum(epochs)
        self.assertEqual(epochs.ids, [r.trial.id for r in result.trials])
        self.assertEqual({i: r for i, _, r in epochs.rejected}, result.rejected)
        for k, item in enumerate(result.trials):
            np.testing.assert_allclose(psd.power[k, 0], item.metrics.psd, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(fft.power[k, 0], item.metrics.fft, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(psd.freqs, result.trials[0].metrics.psd_freqs)


class SequenceTests(unittest.TestCase):
    """T04: a labelled sequence keeps its phases, its start offset and its incomplete tail as a refused trial."""

    def test_phases_offset_repeats_and_tail(self):
        raw = recording(seconds=110.0)
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=20.0, source="sequence", sequence="A:30; B:20",
                                    sequence_start=2.5, repeats=3)
        expected = periodic_trials(2.5, ap.parse_sequence("A:30; B:20"), 3, raw.duration)
        self.assertEqual([(t.id, t.condition) for t in expected],
                         [("1", "A"), ("2", "B"), ("3", "A"), ("4", "B"), ("5", "A")])
        self.assertEqual(epochs.ids, ["1", "2", "3", "4"])
        self.assertEqual(epochs.labels, ["A", "B", "A", "B"])
        # the fifth phase starts at 102.5 s and the recording ends at 110 s: kept as a trial, refused
        self.assertEqual(epochs.rejected, [("5", "A", "Too short for common duration")])
        self.assertEqual(epochs.meta[0]["sample"], 1250)
        self.assertEqual(set(epochs.counts()), {"A", "B"})


class EventTrialTests(unittest.TestCase):
    """T05: each event's own length; an event of zero length makes no trial, as on the page."""

    def test_event_lengths_and_zero_length_markers(self):
        events = ep.EventTable([1.5, 15.0, 30.0006, 45.0], [10.0, 0.0, 12.0, 8.0], ["open", "marker", "closed", "open"])
        raw = recording(events=events)
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=10.0, event_length=True)
        page = [Trial(str(i + 1), label, onset, onset + length)
                for i, (onset, length, label) in enumerate(zip(events.onsets, events.durations, events.labels))
                if length > 0]
        self.assertEqual(epochs.ids, ["1", "3"])
        self.assertEqual(epochs.rejected, [("4", "open", "Too short for common duration")])
        for meta in epochs.meta:
            trial = next(t for t in page if t.id == meta["id"])
            start, count, _ = trial_status(trial, raw.fs, raw.n_samples, 10.0)
            self.assertEqual((meta["sample"], meta["count"]), (start, count))
            self.assertAlmostEqual(meta["zero"], start / raw.fs)

    def test_a_window_around_events_keeps_its_old_cut(self):
        """events without event_length: the cut every saved pipeline made before, unchanged."""
        events = ep.EventTable([5.0, 20.0, 40.0], [1.0, 1.0, 1.0], ["A", "B", "A"])
        raw = recording(events=events)
        epochs = compute.cut_epochs(raw, tmin=-1.0, tmax=3.0)
        self.assertEqual(epochs.ids, ["1", "2", "3"])
        np.testing.assert_array_equal(epochs.data[0], raw.data[:, 2000:4000])
        self.assertEqual(epochs.meta[0]["zero"], 5.0)


class ConditionFieldTests(unittest.TestCase):
    """T06: the page's Condition field is not a filter on the recording's events."""

    def test_eyes_open_closed_keeps_both_conditions(self):
        eyes = next(p for p in ap.builtin_profiles() if p.name == "Eyes open / closed")
        self.assertEqual(eyes.condition, "Eyes open")
        onsets = [0.0, 30.0, 60.0, 90.0]
        labels = ["Eyes open", "Eyes closed"] * 2
        with scratch() as directory:
            raw = Recording(np.random.default_rng(0).normal(0, 5, (2, 125 * 250)), 250.0, ["O1", "O2"],
                            events=ep.EventTable(onsets, [30.0] * 4, labels))
            path = save_fif(raw, directory)
            graph, notes = from_profile(eyes, path)
            self.assertEqual(graph.params("epochs1")["conditions"], [])
            self.assertTrue(graph.params("epochs1")["event_length"])
            # the FIF annotations' 30 s lengths are read with them, so the profile runs as it is
            outputs = Runner(graph).run(["epochs1"])
            self.assertEqual(outputs["epochs1"]["epochs"].counts(), {"Eyes open": 2, "Eyes closed": 2})
            self.assertEqual(list(compute.read_recording(path).events.durations), [30.0] * 4)
            # cutting a fixed 30 s after each event gives the same trials
            windowed = ap.AnalysisProfile.from_dict({**eyes.to_dict(), "event_length": False})
            graph, notes = from_profile(windowed, path)
            self.assertEqual(graph.params("epochs1")["conditions"], [])
            outputs = Runner(graph).run(["epochs1"])
            self.assertEqual(outputs["epochs1"]["epochs"].counts(), {"Eyes open": 2, "Eyes closed": 2})


class FilterTests(unittest.TestCase):
    """T07: the page's notch list exactly, no harmonics added; a single mains value still adds them."""

    def test_explicit_notches_match_the_page_and_add_no_harmonics(self):
        raw = recording()
        for notches in ([50.0], [50.0, 75.0]):
            page = filter_finite(raw.data[1], raw.fs, low=1.0, high=45.0, order=4, notches=notches, notch_q=30.0)
            ours = compute.filter_recording(raw, low=1.0, high=45.0, notch=0.0, notches=notches).data[1]
            np.testing.assert_allclose(ours, page, rtol=RTOL, atol=ATOL)
        legacy = compute.filter_recording(raw, low=1.0, high=0.0, notch=50.0).data[1]
        listed = compute.filter_recording(raw, low=1.0, high=0.0, notch=0.0, notches=[50.0]).data[1]
        # at 500 Hz the harmonics 100 and 150 Hz are below Nyquist: only the legacy form removes them
        from scipy.signal import welch
        f, a = welch(legacy[5000:], raw.fs, nperseg=2000)
        _, b = welch(listed[5000:], raw.fs, nperseg=2000)
        self.assertTrue(np.allclose(a[f == 10], b[f == 10], rtol=1e-2))
        from eeg.pipeline.filters import steps
        self.assertEqual(steps(raw.fs, {"low": 1.0, "high": 0.0, "notch": 0.0, "notches": [50.0]})[0][1].shape[0],
                         1 + 2)       # one notch section + a 4th-order high-pass in two sections
        self.assertEqual(steps(raw.fs, {"low": 1.0, "high": 0.0, "notch": 50.0})[0][1].shape[0], 3 + 2)

    def test_a_high_pass_alone_is_the_pages_butterworth(self):
        """High cut 0 (a profile with only a low cut): designed, and the same as core.processing's high-pass.
        Before, scipy was handed a list of one edge and the block failed."""
        raw = recording()
        page = filter_finite(raw.data[0], raw.fs, low=1.0, order=4)
        ours = compute.filter_recording(raw, low=1.0, high=0.0, notch=0.0).data[0]
        np.testing.assert_allclose(ours, page, rtol=RTOL, atol=ATOL)
        lowpass = compute.filter_recording(raw, low=0.0, high=30.0, notch=0.0).data[0]
        np.testing.assert_allclose(lowpass, filter_finite(raw.data[0], raw.fs, high=30.0, order=4), rtol=RTOL,
                                   atol=ATOL)

    def test_linear_detrend_and_average_reference_as_the_page(self):
        raw = recording()
        trials = [Trial("1", "A", 2.0, 12.0), Trial("2", "B", 20.0, 30.0)]
        settings = Settings(duration=10.0, window=2.0, overlap=0.5,
                            filters=dict(low=1.0, high=45.0, order=4, notches=[50.0, 75.0], notch_q=30.0,
                                         detrend="linear", reference="average"))
        result = analyze(raw.data, raw.fs, [0, 2], "ROI", trials, settings, names=raw.channel_names)
        referenced = compute.rereference(raw, "average")
        filtered = compute.filter_recording(referenced, low=1.0, high=45.0, notch=0.0, notches=[50.0, 75.0])
        epochs = compute.cut_epochs(filtered, tmin=0.0, tmax=10.0, source="manual", trials=table(trials))
        psd = compute.welch_spectrum(epochs, 2.0, 0.5, detrend="linear")
        fft = compute.fft_spectrum(epochs, detrend="linear")
        for k, item in enumerate(result.trials):
            for slot, channel in enumerate((0, 2)):
                np.testing.assert_allclose(psd.power[k, channel], item.channel_metrics[slot].psd, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(fft.power[k, channel], item.channel_metrics[slot].fft, rtol=RTOL, atol=ATOL)
        constant = compute.welch_spectrum(epochs, 2.0, 0.5)
        self.assertFalse(np.allclose(constant.power[0, 0, :3], psd.power[0, 0, :3]))


class TimeFrequencySettingsTests(unittest.TestCase):
    """T11: the ERD average, Morlet cycles, the CWT wavelet and rejection reach the computation and the script."""

    def setUp(self):
        self.raw = lateralised(trials=12)
        self.common = dict(channels=["C3", "C4"], tmin=0.0, tmax=7.0, baseline=[0.5, 1.5], active=[2.5, 4.5])

    def test_the_defaults_are_the_settings_used_before(self):
        result = compute.time_frequency(self.raw, **self.common)
        self.assertEqual((result.settings.erd_mode, result.settings.n_cycles, result.settings.wavelet,
                          result.settings.reject_uv), (ep.POOLED, 0.5, "cmor1.5-1.0", 0.0))

    def test_pooled_and_mean_of_trials_differ_when_baselines_differ(self):
        rng = np.random.default_rng(3)
        raw = lateralised(trials=12)
        for k in range(12):           # trials with very different baseline power
            scale = 1.0 + 4.0 * rng.random()
            raw.data[:, k * 7 * 250:(k + 1) * 7 * 250] *= scale
        pooled = compute.time_frequency(raw, **self.common)
        mean = compute.time_frequency(raw, **self.common, erd_mode=ep.MEAN_OF_TRIALS)
        self.assertEqual(mean.settings.erd_mode, ep.MEAN_OF_TRIALS)
        channel_p, channel_m = pooled.tf[0], mean.tf[0]
        np.testing.assert_allclose(channel_p.power["left"], channel_m.power["left"])
        a = ep.condition_map(pooled, channel_p, "left")
        b = ep.condition_map(mean, channel_m, "left")
        np.testing.assert_allclose(a, ep.erd_map(channel_p, "left", ep.POOLED))
        np.testing.assert_allclose(b, ep.erd_map(channel_m, "left", ep.MEAN_OF_TRIALS))
        self.assertGreater(np.nanmax(np.abs(a - b)), 0.1)

    def test_morlet_cycles_and_rejection_change_the_result(self):
        three = compute.time_frequency(self.raw, **self.common, method=ep.MORLET, n_cycles=0.3)
        default = compute.time_frequency(self.raw, **self.common, method=ep.MORLET)
        self.assertFalse(np.allclose(three.tf[0].power["left"], default.tf[0].power["left"]))
        raw = lateralised(trials=12)
        raw.data[0, 3 * 7 * 250 + 100] += 500.0
        rejected = compute.time_frequency(raw, **self.common, reject_uv=200.0)
        self.assertIn("Amplitude > 200 µV (C3)", [status for _, _, _, status in rejected.epochs])

    def test_the_settings_reach_the_script_and_the_methods(self):
        from eeg.pipeline import block
        graph = Graph("tf")
        load, erd = graph.chain(("load", {"path": "x.fif"}),
                                ("erd", {"erd_mode": ep.MEAN_OF_TRIALS, "method": ep.MORLET, "n_cycles": 0.4,
                                         "reject_uv": 150.0}))
        script = to_python(graph)
        for text in ("erd_mode='mean_of_trials'", "n_cycles=0.4", "reject_uv=150.0"):
            self.assertIn(text, script)
        sentence = block("erd").describe(graph.params(erd))
        self.assertIn("0.4 cycles per Hz", sentence)
        self.assertIn("averaged over the trials", sentence)
        self.assertIn("150", sentence)


def opposite_phase(fs=128.0, seconds=4.0, roi=False):
    """Two 'A' trials of a 10 Hz sine in opposite phase (and with `roi`, two channels in opposite phase)."""
    t = np.arange(int(2 * seconds * fs)) / fs
    first = np.where(t < seconds, 1.0, -1.0) * 3 * np.sin(2 * np.pi * 10 * t)
    rows = [first, -first] if roi else [first]
    raw = Recording(np.vstack(rows), fs, ["C3", "C4"][:len(rows)])
    return raw, [Trial("1", "A", 0.0, seconds), Trial("2", "A", seconds, 2 * seconds)]


class TwoAveragesTests(unittest.TestCase):
    """T08, T09: the mean of trial spectra and the spectrum of the mean signal stay two things, and a region is
    measured channel by channel."""

    def test_opposite_phase_trials_cancel_only_in_the_mean_signal(self):
        raw, trials = opposite_phase()
        settings = Settings(duration=4.0, window=2.0)
        page = analyze(raw.data, raw.fs, 0, "C3", trials, settings).conditions[0]
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=4.0, source="manual", trials=table(trials))
        for kind, metric in ((compute.welch_spectrum(epochs, 2.0, 0.5, signal_mean=True), "psd"),
                             (compute.fft_spectrum(epochs, signal_mean=True), "fft")):
            np.testing.assert_allclose(kind.power[:, 0].mean(axis=0), getattr(page.mean_trials, metric),
                                       rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(kind.signal_means["A"][0], getattr(page.mean_signal, metric),
                                       rtol=RTOL, atol=1e-9)
            self.assertLess(kind.signal_means["A"][0].max(), 1e-6 * kind.power[:, 0].mean(axis=0).max())
        plain = compute.welch_spectrum(epochs, 2.0, 0.5)
        self.assertEqual(plain.signal_means, {})
        self.assertEqual(plain.power.shape[0], 2)          # still one spectrum per trial

    def test_a_region_of_opposite_channels_keeps_its_power(self):
        raw, trials = opposite_phase(roi=True)
        settings = Settings(duration=4.0, window=2.0)
        page = analyze(raw.data, raw.fs, [0, 1], "ROI", trials, settings, names=raw.channel_names)
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=4.0, source="manual", trials=table(trials))
        psd = compute.welch_spectrum(epochs, 2.0, 0.5, signal_mean=True)
        for k, item in enumerate(page.trials):
            np.testing.assert_allclose(psd.power[k].mean(axis=0), item.metrics.psd, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(psd.signal_means["A"].mean(axis=0), page.conditions[0].mean_signal.psd,
                                   rtol=RTOL, atol=1e-9)
        self.assertGreater(psd.power.mean(axis=(0, 1)).max(), 1.0)

    def test_the_preview_draws_either_average_and_says_which(self):
        from test_epochs import app
        from eeg.ui.workbench.previews import MEAN_SIGNAL, SpectrumPreview
        app()
        raw, trials = opposite_phase()
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=4.0, source="manual", trials=table(trials))
        preview = SpectrumPreview(compute.welch_spectrum(epochs, 2.0, 0.5, signal_mean=True))
        self.assertEqual(preview.average.count(), 2)
        preview.average.setCurrentText(MEAN_SIGNAL)
        self.assertIn("mean waveform", preview.caption.text())
        again = SpectrumPreview(preview.value).choose(preview.choices())
        self.assertEqual(again.average.currentText(), MEAN_SIGNAL)
        self.assertIn(("Average", MEAN_SIGNAL), [(n, b.currentText()) for n, b in again.said_choices()])


class SpectralMeasureTests(unittest.TestCase):
    """T10: SNR, the condition contrast and strongest frequencies through core.spectral; FFT bins and dB."""

    def test_contrast_is_core_cohens_d_of_the_regions_trial_spectra(self):
        from test_epochs import app
        from eeg.core.spectral import condition_contrast
        from eeg.ui.workbench.previews import CONTRAST, SpectrumPreview
        app()
        raw = recording(seconds=60.0)
        trials = [Trial(str(k + 1), "A" if k % 2 else "B", 5.0 * k, 5.0 * k + 5.0) for k in range(12)]
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=5.0, source="manual", trials=table(trials))
        spectrum = compute.welch_spectrum(epochs, 2.0, 0.5)
        preview = SpectrumPreview(spectrum)
        preview.show_box.setCurrentText(CONTRAST)
        preview.a_box.setCurrentText("A")
        preview.b_box.setCurrentText("B")
        freqs, d, na, nb = preview.contrast()
        labels = np.asarray(spectrum.labels)
        expected = condition_contrast(spectrum.power[labels == "A", 0], spectrum.power[labels == "B", 0],
                                      spectrum.freqs)
        np.testing.assert_allclose(d, expected, rtol=RTOL, atol=ATOL, equal_nan=True)
        self.assertEqual((na, nb), (6, 6))
        self.assertIn("A > B", preview.caption.text())

    def test_snr_scale_is_the_neighbour_ratio(self):
        from test_epochs import app
        from eeg.ui.workbench.previews import SNR, SpectrumPreview
        app()
        raw = recording(seconds=30.0)
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=10.0, source="manual",
                                    trials=[["1", "A", 0.0, 10.0], ["2", "A", 10.0, 20.0]])
        preview = SpectrumPreview(compute.welch_spectrum(epochs, 2.0, 0.5))
        preview.scale_box.setCurrentText(SNR)
        self.assertEqual(preview.plot.getAxis("left").labelText, "SNR (× neighbours)")
        self.assertEqual(SpectrumPreview(compute.fft_spectrum(epochs)).scale_box.findText(SNR), -1)

    def test_fft_even_and_odd_lengths_read_a_sine_and_its_nyquist(self):
        for n in (128, 129):
            fs = 128.0
            t = np.arange(n) / fs
            raw = Recording(np.vstack([2 * np.sin(2 * np.pi * 8 * t) + 0.5 * np.cos(np.pi * fs * t)]), fs, ["Cz"])
            epochs = compute.cut_epochs(raw, tmin=0.0, tmax=n / fs, source="manual", trials=[["1", "A", 0.0, n / fs]])
            spectrum = compute.fft_spectrum(epochs)
            page = analyze(raw.data, fs, 0, "Cz", [Trial("1", "A", 0.0, n / fs)], Settings(duration=n / fs, window=0.5))
            np.testing.assert_allclose(spectrum.power[0, 0], page.trials[0].metrics.fft, rtol=RTOL, atol=ATOL)


class FeatureProfileTests(unittest.TestCase):
    """T12: the whole feature profile - relative, ratio, peak, its own total range - as the page computes it."""

    def test_mu_beta_profile_with_a_custom_total_range(self):
        from eeg.core.feature_profiles import FeatureProfile
        raw = recording(seconds=60.0)
        profile = FeatureProfile(name="Mu / beta", bands=[["mu", 8.0, 13.0], ["beta", 13.0, 30.0]],
                                 ratios=[["mu/beta", "mu", "beta"]],
                                 selected=["band:mu", "relative:mu", "ratio:mu/beta", "peak:beta", "rms", "total"],
                                 total_range=[2.0, 40.0])
        trials = [Trial("a", "A", 1.0, 11.0), Trial("b", "B", 20.0, 30.0), Trial("c", "A", 40.0, 50.0)]
        for detrend in ("constant", "linear"):
            settings = Settings(duration=10.0, window=2.0, overlap=0.5, profile=profile.to_dict(),
                                filters=dict(detrend=detrend))
            page = analyze(raw.data, raw.fs, [0, 2], "ROI", trials, settings, names=raw.channel_names)
            epochs = compute.cut_epochs(compute.pick_channels(raw, ["C3", "C4"]), tmin=0.0, tmax=10.0,
                                        source="manual", trials=table(trials))
            features = compute.signal_features(epochs, profile.selected, 2.0, 0.5, detrend, profile.bands,
                                               profile.ratios, profile.total_range, roi=True)
            self.assertEqual(features.names[0], "ROI C3+C4 mu power")
            for k, trial in enumerate(trials):
                values = page.features.trials[trial.id].values
                np.testing.assert_allclose(features.values[k], [values[s] for s in profile.selected], rtol=RTOL,
                                           atol=ATOL)
            self.assertEqual(features.units[1], "%")

    def test_a_missing_value_keeps_its_reason(self):
        raw = recording(fs=100.0, seconds=30.0)
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=10.0, source="manual", trials=[["1", "A", 0.0, 10.0]])
        features = compute.signal_features(epochs, ["band:gamma2"], bands=[["gamma2", 45.0, 60.0]], ratios=[],
                                           total_range=[1.0, 45.0])
        self.assertTrue(np.isnan(features.values[0, 0]))
        self.assertIn("Nyquist", features.reasons[0][features.names[0]])

    def test_the_old_default_features_are_unchanged(self):
        from eeg.core.feature_profiles import FeatureProfile, extract
        raw = recording(seconds=30.0)
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=10.0, source="manual", trials=[["1", "A", 0.0, 10.0]])
        features = compute.signal_features(epochs)
        profile = FeatureProfile(name="Workbench", selected=["rms", "mobility", "complexity", "entropy"])
        expected = [extract(row, raw.fs, profile, window=1.0).values[s] for row in epochs.data[0]
                    for s in profile.selected]
        np.testing.assert_array_equal(features.values[0], expected)

    def test_the_converter_carries_the_whole_profile(self):
        cho = next(p for p in ap.builtin_profiles() if "Cho2017" in p.name)
        graph, _ = from_profile(cho, "s01.mat")
        params = graph.params("features1")
        self.assertEqual(params["selected"], ["band:mu", "band:beta", "relative:mu", "relative:beta"])
        self.assertEqual(params["ratios"], [["mu/beta", "mu", "beta"]])
        self.assertEqual(params["total_range"], [1.0, 40.0])
        self.assertTrue(params["roi"])


class HeadMapChannelTests(unittest.TestCase):
    """T13: picking the analysed channels does not take the other electrodes away from the head map."""

    def test_the_time_frequency_block_reads_the_recording_before_channels_are_picked(self):
        cho = next(p for p in ap.builtin_profiles() if "Cho2017" in p.name)
        cho = ap.AnalysisProfile.from_dict({**cho.to_dict(), "tab": "brain map"})
        graph, _ = from_profile(cho, "s01.mat")
        wire = graph.feeds("erd1")["raw"]
        self.assertEqual(graph.nodes[wire.source].type, "filter")
        self.assertTrue(graph.params("erd1")["topography"])
        self.assertEqual(graph.params("erd1")["channels"], ["C3", "Cz", "C4"])

    def test_a_profile_of_another_trial_source_feeds_the_epochs_trials(self):
        eyes = next(p for p in ap.builtin_profiles() if p.name == "Eyes open / closed")
        eyes = ap.AnalysisProfile.from_dict({**eyes.to_dict(), "tab": "time-frequency"})
        graph, _ = from_profile(eyes, "eyes.fif")
        # the trials are cut once from every electrode; the page's channels are picked from the trials
        self.assertEqual(graph.feeds("erd1")["raw"].source, "epochs1")
        self.assertNotIn("channels1", graph.nodes)
        self.assertEqual(graph.feeds("psd1")["epochs"].source, "trial_channels1")
        self.assertEqual(graph.feeds("features1")["epochs"].source, "trial_channels1")
        self.assertEqual(graph.params("trial_channels1")["channels"], ["O1", "O2"])
        self.assertEqual(graph.params("erd1")["baseline_mode"], "condition")
        self.assertEqual(graph.problems(), [])

    def test_channels_of_trials_keeps_the_trials(self):
        raw = recording()
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=4.0, source="manual",
                                    trials=[["a", "A", 1.0, 5.0], ["b", "B", 6.0, 10.0]])
        picked = compute.pick_trial_channels(epochs, ["C4", "C3"])
        self.assertEqual(picked.names, ["C4", "C3"])
        np.testing.assert_array_equal(picked.data, epochs.data[:, [2, 0]])
        self.assertEqual((picked.ids, picked.meta), (epochs.ids, epochs.meta))
        with self.assertRaisesRegex(ValueError, "Not in these trials: Pz"):
            compute.pick_trial_channels(epochs, ["Pz"])


class SessionImportTests(unittest.TestCase):
    """T01, T18 (numbers): a saved Analysis session becomes a pipeline with its own trials, settings and choice of
    average, without the page; the numbers are the page's; a wrong recording is refused; nothing is written."""

    def make(self, directory, version=3, profile=True):
        from eeg.core.feature_profiles import FeatureProfile
        from eeg.formats.trial_analysis import save_session
        raw = recording(fs=250.0, seconds=70.0, events=ep.EventTable([1.0, 15.0], [10.0, 10.0], ["open", "closed"]))
        path = save_fif(raw, directory, "rec_raw.fif")
        read = compute.read_recording(path)
        self.trials = [Trial("open-1", "Eyes open", 1.0, 13.0), Trial("closed-1", "Eyes closed", 15.0, 26.0),
                       Trial("open-2", "Eyes open", 30.0, 41.0, included=False),
                       Trial("closed-2", "Eyes closed", 45.0, 52.0),           # too short for 10 s
                       Trial("open-3", "Eyes open", 55.0, 66.0)]
        mu = FeatureProfile(name="Mu", bands=[["mu", 8.0, 13.0], ["beta", 13.0, 30.0]],
                            ratios=[["mu/beta", "mu", "beta"]],
                            selected=["band:mu", "relative:mu", "ratio:mu/beta", "rms"], total_range=[2.0, 40.0])
        self.settings = Settings(duration=10.0, window=2.0, overlap=0.5, profile=mu.to_dict(),
                                 filters=dict(low=1.0, high=45.0, order=4, notches=[50.0], notch_q=30.0,
                                              detrend="linear", reference="average"))
        display = {"visibility": {"open-3": False}, "calculation": "mean_signal", "mode": 0}
        if profile:
            page = ap.AnalysisProfile(name="Mine", channels=["C3", "C4"], filters=self.settings.filters,
                                      trial_source=ap.MANUAL, common_duration=10.0, window=2.0,
                                      feature_profile=mu.to_dict(), tab="time-frequency",
                                      epoch_settings=dict(method="stft", fmin=4.0, fmax=40.0, window=0.5, overlap=0.9,
                                                          baseline_mode="condition", reference="",
                                                          active=[1.0, 9.0], erd_mode="mean_of_trials",
                                                          bands=[["mu", 8.0, 13.0]]))
            display["profile"] = page.to_dict()
        session = Path(directory) / "rec.analysis.json"
        save_session(session, read, path, ["C3", "C4"], self.trials, self.settings, display)
        if version != 3:
            import json
            payload = json.loads(session.read_text(encoding="utf-8"))
            payload["version"] = version
            if version == 1:
                payload["channel"] = payload.pop("channels")[0]
            session.write_text(json.dumps(payload), encoding="utf-8")
        return raw, read, path, session

    def test_the_session_runs_to_the_pages_numbers(self):
        with scratch() as directory:
            raw, read, path, session = self.make(directory)
            before = session.read_bytes()
            graph, notes = from_session(session)
            self.assertEqual(session.read_bytes(), before)
            self.assertEqual(graph.params("epochs1")["source"], "manual")
            self.assertEqual(graph.params("epochs1")["trials"][2], ["open-2", "Eyes open", 30.0, 41.0, False])
            self.assertEqual(graph.params("filter1")["notches"], [50.0])
            self.assertEqual(graph.params("trial_channels1")["channels"], ["C3", "C4"])
            self.assertEqual(graph.params("erd1")["erd_mode"], "mean_of_trials")
            self.assertEqual(graph.params("erd1")["reference"], "Eyes open")
            self.assertTrue(any("hidden" in n for n in notes))
            self.assertTrue(any("open-2" in n for n in notes))
            outputs = Runner(graph).run()
            page = analyze(read.data, read.fs, [0, 2], "ROI", self.trials, self.settings, names=read.channel_names)
            epochs = outputs["epochs1"]["epochs"]
            self.assertEqual(epochs.ids, [r.trial.id for r in page.trials])
            self.assertEqual({i: r for i, _, r in epochs.rejected}, page.rejected)
            psd = outputs["psd1"]["spectrum"]
            fft = outputs["fft1"]["spectrum"]
            for k, item in enumerate(page.trials):
                np.testing.assert_allclose(psd.power[k].mean(axis=0), item.metrics.psd, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(fft.power[k].mean(axis=0), item.metrics.fft, rtol=RTOL, atol=ATOL)
            for condition in page.conditions:
                labels = np.asarray(psd.labels)
                np.testing.assert_allclose(psd.power[labels == condition.name].mean(axis=(0, 1)),
                                           condition.mean_trials.psd, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(psd.signal_means[condition.name].mean(axis=0), condition.mean_signal.psd,
                                           rtol=RTOL, atol=1e-9)
            features = outputs["features1"]["features"]
            selected = self.settings.profile["selected"]
            for k, item in enumerate(page.trials):
                values = page.features.trials[item.trial.id].values
                np.testing.assert_allclose(features.values[k], [values[s] for s in selected], rtol=RTOL, atol=ATOL)
            self.assertIn("tf", outputs["erd1"])

    def test_saved_and_opened_again_it_is_the_same_pipeline_and_view(self):
        from eeg.pipeline.graph import Graph as G
        with scratch() as directory:
            _, _, _, session = self.make(directory)
            graph, _ = from_session(session)
            self.assertEqual(graph.views["psd1"]["spectrum"]["choosers"], [["Average", "Mean signal"]])
            target = Path(directory) / "rec.pipeline.json"
            graph.save(target)
            again = G.load(target)
            self.assertEqual(again.to_dict(), graph.to_dict())
            self.assertEqual(again.views, graph.views)
            self.assertEqual(again.method_fingerprint(), graph.method_fingerprint())
            again.views["psd1"]["spectrum"]["choosers"] = [["Average", "Mean of trials"]]
            self.assertEqual(again.method_fingerprint(), graph.method_fingerprint())     # a view is not the method

    def test_older_versions_and_a_session_without_profile(self):
        for version in (1, 2):
            with scratch() as directory:
                _, _, _, session = self.make(directory, version=version, profile=False)
                graph, notes = from_session(session)
                self.assertTrue(any("no profile" in n for n in notes))
                self.assertEqual(graph.params("epochs1")["tmin"], 0.0)
                self.assertNotIn("erd1", graph.nodes)
                expected = ["C3"] if version == 1 else ["C3", "C4"]
                self.assertEqual(graph.params("trial_channels1")["channels"], expected)
                self.assertFalse(graph.params("features1")["roi"] if version == 1 else False)

    def test_another_recording_or_a_missing_one_is_refused(self):
        import json
        with scratch() as directory:
            raw, read, path, session = self.make(directory)
            payload = json.loads(session.read_text(encoding="utf-8"))
            payload["recording"]["n_samples"] += 1
            session.write_text(json.dumps(payload), encoding="utf-8")
            before = session.read_bytes()
            with self.assertRaisesRegex(ValueError, "different recording"):
                from_session(session)
            self.assertEqual(session.read_bytes(), before)
            payload["recording"]["path"] = str(Path(directory) / "gone.fif")
            session.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not at"):
                from_session(session)

    def test_the_workbench_opens_it_in_a_new_unsaved_tab(self):
        from test_epochs import app
        from test_workbench import fresh_sheet
        from eeg.ui.workbench.page import WorkbenchPage
        app()
        fresh_sheet(self)
        with scratch() as directory:
            _, _, _, session = self.make(directory)
            page = WorkbenchPage()
            first = page.sheet
            sheet = page.import_session(str(session))
            self.assertIsNotNone(sheet)
            self.assertIsNot(sheet, first)
            self.assertIn(first, page.sheets)
            self.assertTrue(sheet.modified)
            self.assertIsNone(sheet.path)
            self.assertIsNone(page.import_session(str(Path(directory) / "missing.analysis.json")))
            self.assertEqual(len(page.sheets), 2)
            page.close()


class ReviewTests(unittest.TestCase):
    """T15: flat and unusually large trials are marked for review, as the page marks them, and stay in."""

    def test_review_marks_never_leave_a_trial_out(self):
        raw = recording(fs=250.0, seconds=80.0)
        raw.data[1, int(20 * 250):int(26 * 250)] = 0.0                    # Cz flat in trial 3
        raw.data[0, int(40 * 250):int(46 * 250)] *= 40.0                  # C3 huge in trial 5
        trials = [Trial(str(k + 1), "A", 10.0 * k, 10.0 * k + 6.0) for k in range(7)]
        epochs = compute.cut_epochs(raw, tmin=0.0, tmax=6.0, source="manual", trials=table(trials))
        self.assertEqual(epochs.ids, [t.id for t in trials])               # every trial used
        flags = {m["id"]: m["flags"] for m in epochs.meta}
        self.assertEqual(flags["3"], ["Flat signal (Cz)"])
        self.assertEqual(flags["5"], ["High amplitude (median + 6 MAD)"])
        self.assertEqual(flags["1"], [])
        page = analyze(raw.data, raw.fs, [0, 1, 2], "ROI", trials, Settings(duration=6.0, window=2.0),
                       names=raw.channel_names)
        self.assertIn("High amplitude (median + 6 MAD)", page.quality["5"])
        self.assertTrue(any("Flat" in f for f in page.quality["3"]))
        from eeg.ui.workbench.trial_editor import statuses
        self.assertEqual(statuses(epochs)["5"], "Used · review: High amplitude (median + 6 MAD)")


class TrialTableTests(unittest.TestCase):
    """T16, T17: the trial table beside the Epochs block is its `trials` parameter: a span dragged and added, a cell
    typed, a box ticked change the pipeline, make what follows run again, are saved, and stay in their own tab."""

    @classmethod
    def setUpClass(cls):
        from test_epochs import app
        app()

    def sheet_with_table(self, directory, page):
        raw = recording(fs=250.0, seconds=40.0, events=ep.EventTable([1.0], [1.0], ["x"]))
        path = save_fif(raw, directory, "table_raw.fif")
        graph = Graph("table")
        load, epochs, psd = graph.chain(("load", {"path": str(path)}),
                                        ("epochs", {"source": "manual", "tmin": 0.0, "tmax": 5.0,
                                                    "trials": [["1", "rest", 1.0, 7.0, True]]}), ("psd", {}))
        sheet = page.new_tab(graph=graph)
        page.run()
        from test_workbench import wait_for_run
        wait_for_run(page)
        page.canvas.select(epochs)
        return sheet, graph, epochs, psd

    def test_drag_add_type_tick_save_and_open(self):
        from PySide6.QtTest import QTest
        from pyqtgraph.Qt import QtCore
        from test_workbench import fresh_sheet
        from eeg.ui.workbench.page import WorkbenchPage
        fresh_sheet(self)
        with scratch() as directory:
            page = WorkbenchPage()
            sheet, graph, epochs, psd = self.sheet_with_table(directory, page)
            editor = page.inspector.trial_editor
            self.assertIsNotNone(editor)
            self.assertEqual(editor.table.item(0, 5).text(), "Used")
            before = Runner(graph, page.cache).current_keys()[psd]
            editor.region.setRegion((10.0, 16.5))
            QTest.keyClicks(editor.condition, "task")
            QTest.mouseClick(editor.add_button, QtCore.Qt.MouseButton.LeftButton)
            self.assertEqual(graph.params(epochs)["trials"][-1], ["2", "task", 10.0, 16.5, True])
            self.assertNotEqual(Runner(graph, page.cache).current_keys()[psd], before)     # stale: runs again
            self.assertTrue(sheet.modified)
            editor = page.inspector.trial_editor or editor
            editor.table.item(1, 2).setText("11.25")
            editor.table.item(0, 4).setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.assertEqual(graph.params(epochs)["trials"], [["1", "rest", 1.0, 7.0, False],
                                                              ["2", "task", 11.25, 16.5, True]])
            editor.table.item(1, 3).setText("5")                  # end before start: refused, table unchanged
            self.assertEqual(graph.params(epochs)["trials"][1][3], 16.5)
            self.assertIn("start < end", editor.message.text())
            target = Path(directory) / "table.pipeline.json"
            page.save_pipeline(str(target))
            self.assertEqual(Graph.load(target).params(epochs)["trials"], graph.params(epochs)["trials"])
            page.run()
            from test_workbench import wait_for_run
            wait_for_run(page)
            result = page.outputs()[epochs]["epochs"]
            self.assertEqual(result.ids, ["2"])
            self.assertEqual(result.rejected, [("1", "rest", "Excluded by selection")])
            page.close()

    def test_a_table_made_from_the_events_and_kept_to_its_tab(self):
        from test_workbench import fresh_sheet
        from eeg.ui.workbench.page import WorkbenchPage
        fresh_sheet(self)
        with scratch() as directory:
            page = WorkbenchPage()
            raw = recording(fs=250.0, seconds=40.0, events=ep.EventTable([2.0, 12.0, 22.0], [1.0] * 3, ["a", "b", "a"]))
            path = save_fif(raw, directory, "events_raw.fif")
            graph = Graph("events")
            _, epochs = graph.chain(("load", {"path": str(path)}), ("epochs", {"tmin": -1.0, "tmax": 4.0}))
            first = page.new_tab(graph=graph)
            other = page.new_tab(graph=graph.copy())
            page.activate(first)
            page.run()
            from test_workbench import wait_for_run
            wait_for_run(page)
            page.canvas.select(epochs)
            raw_in = page.input_for(epochs, "raw")
            self.assertTrue(page.inspector.make_table(raw_in))
            rows = graph.params(epochs)["trials"]
            self.assertEqual([r[:2] for r in rows], [["1", "a"], ["2", "b"], ["3", "a"]])
            self.assertEqual(rows[0][2:4], [1.0, 6.0])
            self.assertEqual(graph.params(epochs)["source"], "manual")
            self.assertEqual(other.graph.params(epochs)["source"], "events")           # the other tab is its own
            self.assertEqual(other.graph.params(epochs)["trials"], [])
            page.close()


class RoutingTests(unittest.TestCase):
    """T19-T21: Quick analysis, a saved session from Projects, and Continue go to the Workbench, in a
    new tab, without building the Analysis page; the tab in view is never replaced."""

    @classmethod
    def setUpClass(cls):
        from test_epochs import app
        app()

    def window(self):
        from test_workbench import fresh_sheet
        from eeg.ui.app import MainWindow
        fresh_sheet(self)
        window = MainWindow()
        self.addCleanup(window.close)
        return window

    def test_quick_analysis_and_continue_open_a_new_tab(self):
        with scratch() as directory:
            raw = recording(fs=250.0, seconds=40.0, events=ep.EventTable([1.0, 15.0], [10.0, 10.0], ["A", "B"]))
            path = str(save_fif(raw, directory, "quick_raw.fif"))
            window = self.window()
            eyes = next(p for p in ap.builtin_profiles() if p.name == "Eyes open / closed")
            sheet = window.quick_analysis(path, eyes)
            self.assertEqual(window.current_mode(), "Workbench")
            self.assertEqual(sheet.graph.params("load1")["path"], path)
            self.assertTrue(sheet.modified)
            first = window.workbench.sheets[0]
            self.assertIsNot(sheet, first)
            from unittest import mock
            with mock.patch("eeg.ui.workbench.quick.QuickDialog.exec", return_value=False):
                self.assertIsNone(window.run_task("analyse"))         # cancelled: nothing opened
                self.assertFalse(window.reopen_recent("recordings", path))
            self.assertNotIn("Analysis", window._pages)
            self.assertEqual(len(window.workbench.sheets), 2)

    def test_a_saved_session_from_projects_opens_on_the_workbench(self):
        with scratch() as directory:
            maker = SessionImportTests()
            maker.make = SessionImportTests.make.__get__(maker)
            _, _, path, session = maker.make(directory)
            window = self.window()
            window.projects.analysis_activated.emit(str(path), str(session))
            self.assertEqual(window.current_mode(), "Workbench")
            self.assertEqual(window.workbench.graph.params("epochs1")["source"], "manual")
            self.assertNotIn("Analysis", window._pages)

    def test_the_window_has_no_assistant_and_finds_the_tab_recording(self):
        """The Assistant was removed: no button or panel. The Workbench still
        says which recording the tab in view reads, for a profile opened from the Library."""
        from pyqtgraph.Qt import QtWidgets
        with scratch() as directory:
            raw = recording(fs=250.0, seconds=40.0, events=ep.EventTable([1.0], [1.0], ["x"]))
            path = str(save_fif(raw, directory, "one_raw.fif"))
            window = self.window()
            self.assertFalse(hasattr(window, "assistant"))
            self.assertNotIn("Assistant", [b.text() for b in window.findChildren(QtWidgets.QAbstractButton)])
            self.assertEqual(window.findChildren(QtWidgets.QDockWidget), [])
            workbench = window.open_mode("Workbench")
            workbench.open_recording(path)
            self.assertEqual(workbench.recording_context()[2], path)
            workbench.graph.add("load", {"path": path})
            self.assertIsNone(workbench.recording_context())        # two recordings: ambiguous, none taken
            self.assertNotIn("Analysis", window._pages)


class ProfilesAndRecordingToolsTests(unittest.TestCase):
    """T20, T22: analysis profiles imported, exported, copied and deleted from the Library (built-in ones
    read-only); a recording renamed with its sidecars, its channels and events named, from the Projects pane."""

    @classmethod
    def setUpClass(cls):
        from test_epochs import app
        app()

    def test_profiles_in_the_library(self):
        from eeg.ui.workbench.library import Library, PROFILES
        with scratch() as directory:
            library = Library(project_folders=lambda: [])
            library.profile_root = str(directory)
            library.refresh()
            names = [i.text(0) for i in library.entries() if i.data(0, 1 + 0x0100) == PROFILES]
            self.assertTrue(set(ap.BUILTIN_NAMES) <= set(names))
            mine = library.copy_profile("Eyes open / closed", "My eyes")
            self.assertEqual(mine.name, "My eyes")
            self.assertIsNone(library.copy_profile("Eyes open / closed", "Blank"))      # a built-in name: refused
            exported = library.export_profile("My eyes", str(directory / "shared.profile.json"))
            self.assertTrue(Path(exported).exists())
            self.assertTrue(library.delete_profile("My eyes"))
            self.assertFalse(library.delete_profile("Blank"))                          # built in: read-only
            again = library.import_profile(str(exported))
            self.assertEqual(again.name, "My eyes")
            self.assertIn("My eyes", [p.name for p in ap.available(str(directory))])

    def test_recording_tools_in_the_projects_pane(self):
        from eeg.formats.channels import load_names
        from eeg.ui.project_panel import ProjectPanel
        with scratch() as directory:
            raw = recording(fs=250.0, seconds=10.0, events=ep.EventTable([1.0], [1.0], ["1"]))
            path = save_fif(raw, directory, "tools_raw.fif")
            panel = ProjectPanel()
            self.assertEqual(panel.channel_names(str(path), ["Fz", "Cz", "Pz"]), ["Fz", "Cz", "Pz"])
            self.assertEqual(load_names(path, 3), ["Fz", "Cz", "Pz"])
            self.assertEqual(panel.event_names(str(path), {"1": "cue"}), {"1": "cue"})
            moved = panel.rename_recording(str(path), "renamed_raw")
            self.assertEqual(moved.name, "renamed_raw.fif")
            self.assertEqual(load_names(moved, 3), ["Fz", "Cz", "Pz"])                 # the sidecar came along
            self.assertEqual(compute.read_recording(moved).events.labels, ["cue"])
            panel.close()


def mi_result(erd_mode=ep.POOLED, topography=True):
    """A lateralised motor-imagery result on the 10-20 positions of C3, Cz, C4, with three more electrodes."""
    raw = lateralised(trials=16)
    rng = np.random.default_rng(5)
    extra = rng.normal(0, 1, (3, raw.n_samples))
    raw = Recording(np.vstack([raw.data, extra]), raw.fs, raw.channel_names + ["Fz", "Pz", "Oz"], events=raw.events)
    return compute.time_frequency(raw, channels=["C3", "C4"], tmin=0.0, tmax=7.0, baseline=[0.5, 1.5],
                                  active=[2.5, 4.5], topography=topography, erd_mode=erd_mode)


class BandErdViewTests(unittest.TestCase):
    """T14: band course ± SEM / n, the statistics table and A, B, B − A head maps from the same core numbers."""

    @classmethod
    def setUpClass(cls):
        from test_epochs import app
        app()

    def preview(self, result):
        from eeg.ui.workbench.previews import TFPreview
        return TFPreview(result)

    def test_statistics_are_core_statistics_of_the_conditions_chosen(self):
        from eeg.ui.workbench.previews import STATS_VIEW
        result = mi_result()
        preview = self.preview(result)
        preview.condition.setCurrentText("left")
        preview.compare.setCurrentText("right")
        preview.view_box.setCurrentText(STATS_VIEW)
        rows = preview.statistics()
        expected = [r for r in ep.statistics(result, "left", "right") if r["band"] == "mu"]
        self.assertEqual([r["channel"] for r in rows], ["C3", "C4"])
        for got, want in zip(rows, expected):
            for key in ("mean_a", "mean_b", "difference", "cliffs_delta", "p_welch", "n_a", "n_b"):
                np.testing.assert_allclose(got[key], want[key], equal_nan=True)
        self.assertEqual(preview.table.rowCount(), 2)
        self.assertIn("not corrected", preview.caption.text())
        # imagery of the left hand lowers mu on C4, of the right hand on C3: B − A has opposite signs
        diff = {r["channel"]: r["difference"] for r in rows}
        self.assertLess(diff["C3"], 0)
        self.assertGreater(diff["C4"], 0)

    def test_head_maps_a_b_and_b_minus_a(self):
        from eeg.ui.workbench.previews import HEAD_VIEW
        result = mi_result()
        preview = self.preview(result)
        preview.condition.setCurrentText("left")
        preview.compare.setCurrentText("right")
        preview.view_box.setCurrentText(HEAD_VIEW)
        panels = preview.head_panels(ep.ERD_MAP)
        self.assertEqual([title for title, _ in panels][2], "right − left (percentage points)")
        index = result.topo_channels
        np.testing.assert_allclose(panels[0][1][index], result.topo_pooled["mu"]["left"])
        np.testing.assert_allclose(panels[2][1], panels[1][1] - panels[0][1])
        self.assertEqual(len(result.topo_channels), 6)          # every electrode, not just C3 and C4
        power = preview.head_panels(ep.POWER_MAP)
        self.assertIn("(dB)", power[2][0])

    def test_band_course_follows_the_erd_average(self):
        from eeg.ui.workbench.previews import COURSE_VIEW
        for mode, said in ((ep.POOLED, "pooled"), (ep.MEAN_OF_TRIALS, "mean of each trial")):
            result = mi_result(mode, topography=False)
            preview = self.preview(result)
            preview.view_box.setCurrentText(COURSE_VIEW)
            curves = [item for item in preview.course.getPlotItem().items
                      if isinstance(item, __import__("pyqtgraph").PlotDataItem)]
            mean, sem, n = ep.band_course(result.tf[0], result.conditions[0], (8.0, 13.0), mode)
            np.testing.assert_allclose(curves[0].yData, mean, equal_nan=True)
            self.assertIn(said, preview.caption.text())
            self.assertIn("n={}".format(n), preview.caption.text())

    def test_choices_carry_the_view_and_the_comparison(self):
        from eeg.ui.workbench.previews import HEAD_VIEW
        result = mi_result()
        preview = self.preview(result)
        preview.compare.setCurrentText("right")
        preview.view_box.setCurrentText(HEAD_VIEW)
        again = self.preview(result).choose(preview.choices())
        self.assertEqual((again.view_box.currentText(), again.compare.currentText()), (HEAD_VIEW, "right"))
        said = dict((name, box.currentText()) for name, box in again.said_choices())
        self.assertEqual(said["View"], HEAD_VIEW)
        self.assertNotIn("Trial", said)


if __name__ == "__main__":
    unittest.main()
