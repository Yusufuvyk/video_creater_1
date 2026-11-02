"""Units, ROI, filters, profiles, events, loaders and the Epoch / ERD page.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from test_trial_analysis import ALPHA, scratch  # noqa: E402
from eeg.core import analysis_profiles as ap  # noqa: E402
from eeg.core import epochs as ep  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.core.trials import Settings, Trial, analyze, measure, prepare_channel, reference_signal  # noqa: E402


def app():
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def lateralised(fs=250, trials=20, seconds=7, seed=0):
    """10 Hz on C3, Cz, C4; its amplitude halves 2-5 s into 'left' trials on C4
    and into 'right' trials on C3 (power -75%)."""
    rng = np.random.default_rng(seed)
    total = trials * seconds * fs
    t = np.arange(total) / fs
    data = rng.normal(0, 1, (3, total))
    envelope = np.ones((3, total))
    onsets, labels = [], []
    for k in range(trials):
        label = "left" if k % 2 == 0 else "right"
        onsets.append(k * seconds)
        labels.append(label)
        envelope[2 if label == "left" else 0, int((k * seconds + 2) * fs):int((k * seconds + 5) * fs)] = 0.5
    data += 10 * np.sin(2 * np.pi * 10 * t) * envelope
    return Recording(data, fs, ["C3", "Cz", "C4"], events=ep.EventTable(onsets, [seconds] * trials, labels))


MI = ep.EpochSettings(tmin=0, tmax=7, baseline=[0.5, 1.5], active=[2.5, 4.5], fmin=4, fmax=40,
                      window=0.5, overlap=0.9, filters=dict(low=1.0, high=40.0, order=4))


class UnitLabelTests(unittest.TestCase):
    def test_psd_axis_never_gets_an_si_prefix(self):
        from eeg.ui.plots import value_axis
        app()
        plot = pg.PlotWidget()
        value_axis(plot, "PSD", "µV²/Hz")
        axis = plot.getAxis("left")
        for top in (5e4, 5e6):
            axis.setRange(0, top)
            axis.updateAutoSIPrefix()
            self.assertIn("PSD (µV²/Hz)", axis.labelString())
            self.assertNotIn("kµ", axis.labelString())
            self.assertNotIn("Mµ", axis.labelString())

    def test_parseval_mean_square_equals_psd_integral(self):
        fs = 250
        t = np.arange(fs * 8) / fs
        x = 20 * np.sin(2 * np.pi * 11 * t) + np.random.default_rng(3).normal(0, 5, t.size) + 3000
        m = measure(x, fs, Settings(duration=8, window=2))
        self.assertAlmostEqual(np.trapezoid(m.psd, m.psd_freqs), m.power, delta=0.05 * m.power)


class RoiAndFilterTests(unittest.TestCase):
    fs = 128

    def setUp(self):
        t = np.arange(self.fs * 8) / self.fs
        self.wave = 3 * np.sin(2 * np.pi * 10 * t) + np.random.default_rng(1).normal(0, 1, t.size)
        self.trials = [Trial("1", "a", 0, 4), Trial("2", "b", 4, 8)]

    def test_roi_of_identical_channels_equals_single_channel(self):
        data = np.vstack([self.wave, self.wave])
        single = analyze(data, self.fs, 0, "C3", self.trials, Settings(duration=4))
        roi = analyze(data, self.fs, [0, 1], "ROI", self.trials, Settings(duration=4))
        for s, r in zip(single.conditions, roi.conditions):
            np.testing.assert_allclose(s.mean_trials.psd, r.mean_trials.psd)
            self.assertAlmostEqual(s.mean_trials.bands["alpha"], r.mean_trials.bands["alpha"])
        a = single.features.trials["1"].values
        b = roi.features.trials["1"].values
        for key in ("band:alpha", "rms", "entropy"):
            self.assertAlmostEqual(a[key], b[key])

    def test_roi_averages_power_not_waveforms(self):
        data = np.vstack([self.wave, -self.wave])
        roi = analyze(data, self.fs, [0, 1], "ROI", self.trials, Settings(duration=4))
        single = analyze(data, self.fs, 0, "C3", self.trials, Settings(duration=4))
        np.testing.assert_allclose(roi.trials[0].signal, 0, atol=1e-12)   # waveforms cancel...
        self.assertAlmostEqual(roi.conditions[0].mean_trials.bands["alpha"],
                               single.conditions[0].mean_trials.bands["alpha"])  # ...power does not

    def _gain(self, frequency, filters):
        t = np.arange(self.fs * 20) / self.fs
        x = np.sin(2 * np.pi * frequency * t)
        y = prepare_channel(x, self.fs, Settings(duration=4, filters=filters))
        middle = slice(self.fs * 5, self.fs * 15)
        return 20 * np.log10(np.std(y[middle]) / np.std(x[middle]))

    def test_highpass_lowpass_and_notch_switch_independently(self):
        highpass = dict(low=2.0, high=None, order=4)
        self.assertLess(self._gain(0.3, highpass), -20)
        self.assertGreater(self._gain(10, highpass), -0.5)
        self.assertGreater(self._gain(50, highpass), -0.5)
        lowpass = dict(low=None, high=20.0, order=4)
        self.assertLess(self._gain(40, lowpass), -20)
        self.assertGreater(self._gain(0.5, lowpass), -0.5)
        notch = dict(notches=[25.0], notch_q=30)
        self.assertLess(self._gain(25, notch), -20)
        self.assertGreater(self._gain(50, notch), -0.5)

    def test_common_average_and_channel_reference(self):
        data = np.random.default_rng(2).normal(0, 1, (4, 500))
        settings = Settings(filters=dict(reference="average"))
        common = reference_signal(data, settings)
        np.testing.assert_allclose((data - common).sum(axis=0), 0, atol=1e-9)
        linked = reference_signal(data, Settings(filters=dict(reference="channels", reference_channels=["M1", "M2"])),
                                  ["C3", "C4", "M1", "M2"])
        np.testing.assert_allclose(linked, data[2:].mean(axis=0))
        with self.assertRaises(ValueError):
            reference_signal(data, Settings(filters=dict(reference="channels", reference_channels=["X"])), ["C3"])

    def test_linear_detrend_removes_ramp_power(self):
        ramp = np.linspace(-100, 100, 512)
        constant = measure(ramp, 128, Settings(duration=4, filters=dict(detrend="constant")))
        linear = measure(ramp, 128, Settings(duration=4, filters=dict(detrend="linear")))
        self.assertGreater(constant.power, 1000)
        self.assertLess(linear.power, 1e-6)

    def test_filter_group_roundtrip_and_monitor_layout_unchanged(self):
        from eeg.ui.filters import FilterGroup
        app()
        monitor = FilterGroup(125, low=3.0)
        self.assertEqual(set(monitor.kwargs), {"low", "high", "order", "notches", "notch_q"})
        offline = FilterGroup(512, offline=True)
        offline.set_sampling_rate(512)
        offline.set_channel_names(["C3", "C4", "M1", "M2"])
        wanted = dict(low=None, high=30.0, order=3, notches=[60.0], notch_q=25.0, detrend="linear",
                      reference="channels", reference_channels=["M1", "M2"])
        offline.set_kwargs(wanted)
        self.assertEqual(offline.kwargs, wanted)
        self.assertEqual(offline.problem(), "")
        offline.set_kwargs(dict(low=40.0, high=30.0))
        self.assertIn("below", offline.problem())
        self.assertEqual(FilterGroup(512, offline=True).kwargs["notches"], [50.0, 100.0, 150.0])


class ProfileTests(unittest.TestCase):
    def test_builtin_eyes_profile_and_missing_channels(self):
        eyes = next(p for p in ap.builtin_profiles() if p.name == "Eyes open / closed")
        found, missing = eyes.resolve_channels(["c3", " o1 ", "Oz"])
        self.assertEqual((found, missing), ([1], ["O2"]))
        self.assertEqual(ap.parse_sequence(eyes.sequence), [("Eyes open", 30.0), ("Eyes closed", 30.0)])
        for profile in ap.builtin_profiles():
            profile.validate()
            settings = ep.EpochSettings.from_dict({**profile.epoch_settings, "filters": profile.filters,
                                                   "reference": "x"})
            settings.tmin, settings.tmax = -profile.event_window[0], max(profile.common_duration, 7.0)
            settings.validate(512)

    def test_two_page_profiles_still_load(self):
        old = dict(name="Old MI", kind="epochs", channels=["C3"], version=1,
                   epoch_settings=dict(tmin=-1.0, tmax=4.0, baseline=[-1.0, 0.0], active=[0.5, 2.5]))
        profile = ap.AnalysisProfile.from_dict(old)
        self.assertEqual((profile.trial_source, profile.event_window, profile.tab), ("events", [1.0, 4.0], "band"))

    def test_user_profiles_save_load_delete_and_reserved_names(self):
        with scratch() as root:
            profile = ap.AnalysisProfile(name="SSVEP Oz", channels=["Oz"], sequence="flicker:10; rest:10",
                                         filters=dict(low=5.0, high=45.0), common_duration=10)
            path = ap.save_user_profile(profile, root)
            self.assertEqual(ap.load_profile(path), profile)
            self.assertIn("SSVEP Oz", [p.name for p in ap.available(root)])
            with self.assertRaises(ValueError):
                ap.save_user_profile(ap.AnalysisProfile(name="Eyes open / closed"), root)
            broken = ap.profiles_dir(root) / ("broken" + ap.SUFFIX)
            broken.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                ap.load_profile(broken)
            self.assertEqual(len(ap.user_profiles(root)), 1)   # a broken file is skipped
            self.assertTrue(ap.delete_user_profile("SSVEP Oz", root))
            broken.unlink()
            ap.profiles_dir(root).rmdir()
        with self.assertRaises(ValueError):
            ap.AnalysisProfile(sequence="A:zero").validate()


class EpochMathTests(unittest.TestCase):
    def test_lateralised_erd_is_found_where_it_was_put(self):
        rec = lateralised()
        result = ep.compute_epochs(rec.data, rec.fs, rec.channel_names, [0, 1, 2], rec.events, MI)
        self.assertEqual(result.conditions, ["left", "right"])
        self.assertEqual({c: len(v) for c, v in result.epoch_ids.items()}, {"left": 10, "right": 10})
        pooled = result.topo_pooled["mu"]
        self.assertAlmostEqual(pooled["left"][2], -75, delta=5)    # C4, left
        self.assertAlmostEqual(pooled["right"][0], -75, delta=5)   # C3, right
        for value in (pooled["left"][0], pooled["left"][1], pooled["right"][1], pooled["right"][2]):
            self.assertLess(abs(value), 8)
        self.assertEqual(int(np.argmin(pooled["left"])), 2)
        course, sem, n = ep.band_course(result.tf[2], "left", (8, 13))
        inside = (result.tf[2].times > 2.8) & (result.tf[2].times < 4.2)
        self.assertAlmostEqual(float(np.mean(course[inside])), -75, delta=6)
        self.assertEqual(n, 10)
        rows = {(r["channel"], r["band"]): r for r in ep.statistics(result, "left", "right")}
        self.assertLess(rows[("C4", "mu")]["p_welch"], 1e-6)
        self.assertGreater(rows[("C4", "mu")]["difference"], 60)
        erd = ep.erd_map(result.tf[2], "left")
        self.assertEqual(erd.shape, (result.tf[2].freqs.size, result.tf[2].times.size))

    def test_morlet_agrees_with_stft_on_band_erd(self):
        rec = lateralised()
        settings = ep.EpochSettings(**{**MI.to_dict(), "method": ep.MORLET, "topography": False})
        result = ep.compute_epochs(rec.data, rec.fs, rec.channel_names, [2], rec.events, settings)
        course, _, _ = ep.band_course(result.tf[0], "left", (8, 13))
        inside = (result.tf[0].times > 2.8) & (result.tf[0].times < 4.2)
        self.assertAlmostEqual(float(np.mean(course[inside])), -75, delta=6)
        self.assertEqual(result.topo_channels, [2])

    def test_morlet_power_is_mean_square_whatever_the_sampling_rate(self):
        # A 10 µV sine: mean square 50 µV². Unscaled MNE output read 1763 at
        # 125 Hz and 7222 at 512 Hz under the same "µV²" label.
        for fs in (125.0, 512.0):
            t = np.arange(int(7 * fs)) / fs
            settings = ep.EpochSettings(tmin=0, tmax=7, method=ep.MORLET, fmin=4, fmax=40, smooth=0.0)
            for f0 in (6.0, 10.0, 31.0):        # 31 Hz: above 24 Hz the cycles are clipped at 12
                freqs, times, power, units = ep.morlet_power(10 * np.sin(2 * np.pi * f0 * t)[None], fs, settings)
                row = int(np.argmin(np.abs(freqs - f0)))
                self.assertAlmostEqual(float(power[0, row, power.shape[2] // 2]), 50.0, delta=0.5)
                self.assertEqual(units, "µV²")

    def test_amplitude_map_reads_the_sines_amplitude(self):
        # Random phase per trial: the mean waveform cancels, the mean amplitude does not.
        fs, seconds, trials = 250.0, 7, 12
        rng = np.random.default_rng(3)
        t = np.arange(int(seconds * fs)) / fs
        data = np.concatenate([10 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
                               for _ in range(trials)])[None]
        events = ep.EventTable(np.arange(trials) * seconds, None, ["a"] * trials)
        for method, window in ((ep.STFT, 1.0), (ep.MORLET, 0.5)):
            settings = ep.EpochSettings(tmin=0, tmax=seconds, baseline=[0.5, 1.5], active=[2, 5], method=method,
                                        window=window, overlap=0.75, fmin=4, fmax=40, topography=False)
            channel = ep.compute_epochs(data, fs, ["x"], [0], events, settings).tf[0]
            values = ep.amplitude_map(channel, "a")
            self.assertEqual(values.shape, (channel.freqs.size, channel.times.size))
            row = int(np.argmin(np.abs(channel.freqs - 10)))
            middle = (channel.times > 1.5) & (channel.times < 5.5)
            self.assertAlmostEqual(float(values[row, middle].mean()), 10.0, delta=0.2)
            self.assertLess(float(values[int(np.argmin(np.abs(channel.freqs - 30))), middle].mean()), 0.2)

    def test_pooled_erd_is_not_dragged_up_by_small_baselines(self):
        # Two trials: baseline 1 -> 0.5 (-50%), baseline 0.01 -> 0.02 (+100%).
        channel = ep.ChannelTF("C3", 0, np.array([10.0]), np.array([0.0, 1.0]),
                               {"a": np.array([[[1.0, 0.5]], [[0.01, 0.02]]])},
                               {"a": np.array([[1.0], [0.01]])}, "µV²/Hz")
        pooled = ep.erd_map(channel, "a", ep.POOLED)[0, 1]
        mean_of_ratios = ep.erd_map(channel, "a", ep.MEAN_OF_TRIALS)[0, 1]
        self.assertAlmostEqual(pooled, 100 * (0.52 - 1.01) / 1.01)
        self.assertAlmostEqual(mean_of_ratios, 25.0)
        self.assertLess(pooled, mean_of_ratios)

    def test_rejection_is_reported_not_silent(self):
        rec = lateralised(trials=6)
        rec.data[0, int(7.5 * rec.fs)] = 5000               # artefact in epoch 2
        rec.data[1, int(15 * rec.fs):int(15.2 * rec.fs)] = np.nan    # gap in epoch 3
        events = ep.EventTable(list(rec.events.onsets) + [40.0], None, rec.events.labels + ["left"])
        settings = ep.EpochSettings(**{**MI.to_dict(), "reject_uv": 500})
        result = ep.compute_epochs(rec.data, rec.fs, rec.channel_names, [0, 2], events, settings)
        status = {eid: s for eid, _, _, s in result.epochs}
        self.assertTrue(status["2"].startswith("Amplitude > 500"))
        self.assertTrue(status["3"].startswith("Missing samples (Cz)"))
        self.assertEqual(status["7"], "Outside recording")
        self.assertEqual(len(result.rejected), 3)

    def test_invalid_windows_are_explained(self):
        rec = lateralised(trials=2)
        for change in (dict(baseline=[-1, 0]), dict(active=[6, 8]), dict(fmax=500), dict(bands=[])):
            with self.assertRaises(ValueError):
                ep.compute_epochs(rec.data, rec.fs, rec.channel_names, [0], rec.events,
                                  ep.EpochSettings(**{**MI.to_dict(), **change}))

    def test_event_table_sidecars_and_label_map(self):
        table = ep.EventTable([3.0, 1.0, 2.0], None, ["769", "770", "1023"])
        self.assertEqual(table.labels, ["770", "1023", "769"])
        renamed = table.relabel({"769": "left", "770": "right", "1023": ""})
        self.assertEqual((renamed.labels, list(renamed.onsets)), (["right", "left"], [1.0, 3.0]))
        with scratch() as directory:
            recording_path = directory / "run.gdf"
            table.save(ep.EventTable.path_for(recording_path))
            ep.save_label_map(recording_path, {"769": "left", "770": "right", "1023": ""})
            found, source = ep.events_for(Recording(np.zeros((1, 10)), 1.0), recording_path)
            self.assertEqual(found.labels, ["right", "left"])
            self.assertEqual(ep.EventTable.load(ep.EventTable.path_for(recording_path)), table)

    def test_export_csv_has_every_section(self):
        import csv
        rec = lateralised(trials=4)
        result = ep.compute_epochs(rec.data, rec.fs, rec.channel_names, [0, 2], rec.events, MI)
        with scratch() as directory:
            path = directory / "epochs.csv"
            ep.export_csv(path, result, "synthetic", "left", "right")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        sections = {r["section"] for r in rows}
        self.assertEqual(sections, {"epoch", "evoked", "psd_active_window", "band_erd_course",
                                    "topography_pooled_erd", "statistics"})
        self.assertEqual({r["unit"] for r in rows if r["section"] == "psd_active_window"}, {"uV^2/Hz"})


class LoaderTests(unittest.TestCase):
    def test_cho2017_joined_conditions_become_labelled_events(self):
        from eeg.formats import matfile as mat, presets as ps
        fs, per_trial, n = 64.0, 7 * 64, 3
        rng = np.random.default_rng(0)
        variables = {"eeg.srate": np.array([[fs]]), "eeg.n_imagery_trials": np.array([[n]]),
                     "eeg.imagery_left": rng.normal(0, 320, (68, per_trial * n)),
                     "eeg.imagery_right": rng.normal(0, 320, (68, per_trial * n))}
        preset = ps.detect(variables)
        self.assertEqual(preset.labels[0], "imagery_left + imagery_right")
        mapping = preset.mapping_for("imagery_left + imagery_right")
        recording = mat.apply_mapping(variables, mat.Mapping.from_dict(json.loads(json.dumps(mapping.to_dict()))))
        gap = int(mat.PART_GAP_SECONDS * fs)
        self.assertEqual(recording.data.shape, (64, 2 * per_trial * n + gap))
        self.assertTrue(np.isnan(recording.data[:, per_trial * n:per_trial * n + gap]).all())
        self.assertAlmostEqual(recording.data[5, 10], variables["eeg.imagery_left"][5, 10] / 32)
        self.assertEqual(recording.events.labels, ["left"] * 3 + ["right"] * 3)
        np.testing.assert_allclose(recording.events.onsets, [0, 7, 14, 22, 29, 36])
        self.assertIsNone(ps.paradigm_for(variables, mapping))
        self.assertIn("2 parts joined, 6 trials", mat.preview(variables, mapping))
        single = mat.apply_mapping(variables, preset.mapping_for("imagery_left"))
        self.assertIsNone(single.events)

    def test_mne_raw_and_epochs_files(self):
        import mne
        from eeg.formats.recordings import load_recording
        mne.set_log_level("ERROR")
        fs = 160.0
        t = np.arange(int(fs * 20)) / fs
        info = mne.create_info(["EEG-C3", "cz", "EEG-FP1", "EOG1", "STI 014"], fs,
                               ["eeg", "eeg", "eeg", "eog", "stim"])
        signal = np.sin(2 * np.pi * 10 * t) * 20e-6
        raw = mne.io.RawArray(np.vstack([signal, signal, signal, signal, np.zeros_like(t)]), info)
        raw.set_annotations(mne.Annotations([2.0, 6.0, 10.0, 14.0], [4, 4, 4, 1], ["769", "770", "769", "BAD_blink"]))
        with scratch() as directory:
            path = directory / "run_raw.fif"
            raw.save(path, overwrite=True)
            recording = load_recording(path)
            self.assertEqual(recording.channel_names, ["C3", "Cz", "Fp1"])
            self.assertAlmostEqual(float(np.abs(recording.data).max()), 20.0, places=3)
            self.assertEqual(recording.events.labels, ["769", "770", "769"])
            np.testing.assert_allclose(recording.events.onsets, [2, 6, 10])
            # a bad mark written in lower case is still a bad mark, not an event (as MNE reads it)
            raw.set_annotations(mne.Annotations([2.0, 14.0], [4, 1], ["769", "bad_motion"]))
            lower = directory / "lower_raw.fif"
            raw.save(lower, overwrite=True)
            marked = load_recording(lower)
            self.assertEqual(marked.events.labels, ["769"])
            self.assertEqual([label for _, _, label in marked.bad_spans], ["bad_motion"])
            raw.set_annotations(mne.Annotations([2.0, 6.0, 10.0, 14.0], [4, 4, 4, 1],
                                                ["769", "770", "769", "BAD_blink"]))
            ep.save_label_map(path, {"769": "left", "770": "right"})
            self.assertEqual(ep.events_for(recording, path)[0].labels, ["left", "right", "left"])
            epochs = mne.Epochs(raw, np.array([[320, 0, 1], [960, 0, 2]]), dict(left=1, right=2),
                                tmin=-1, tmax=2, baseline=None, preload=True)
            epo_path = directory / "run-epo.fif"
            epochs.save(epo_path, overwrite=True)
            stacked = load_recording(epo_path)
            per_epoch = len(epochs.times)
            self.assertEqual(stacked.events.labels, ["left", "right"])
            np.testing.assert_allclose(stacked.events.onsets, [1.0, (per_epoch + fs + fs) / fs])
            for child in list(directory.iterdir()):
                child.unlink()


class ProfileOnWorkbenchTests(unittest.TestCase):
    """The built-in profiles as quick analyses on the Workbench (the Analysis page's before)."""

    def test_choosing_cho2017_profile_builds_event_trials_and_band_erd(self):
        from test_pipeline import save_fif
        from eeg.pipeline import Runner
        from eeg.pipeline.convert import from_profile
        from eeg.ui.workbench.previews import MAP_VIEW, STATS_VIEW, TF_KINDS, TFPreview
        app()
        cho = next(p for p in ap.builtin_profiles() if "Cho2017" in p.name)
        with scratch() as directory:
            path = save_fif(lateralised(trials=8), directory)
            graph, _ = from_profile(cho, path)
            self.assertEqual(graph.params("channels1")["channels"], ["C3", "Cz", "C4"])
            graph.set("erd1", active=[2.5, 4.5])
            outputs = Runner(graph).run()
        epochs = outputs["epochs1"]["epochs"]
        self.assertEqual(len(epochs.ids), 8)
        self.assertEqual((epochs.labels[1], epochs.meta[1]["start"], epochs.meta[1]["end"]), ("right", 7.0, 14.0))
        result = outputs["erd1"]["tf"]
        self.assertFalse(result.settings.topography)
        self.assertAlmostEqual(result.topo_pooled["mu"]["left"][2], -75, delta=6)       # C4, left-hand imagery
        preview = TFPreview(result)
        preview.compare.setCurrentText("right")
        preview.view_box.setCurrentText(STATS_VIEW)
        self.assertEqual({row["channel"] for row in preview.statistics()}, {"C3", "Cz", "C4"})
        preview.view_box.setCurrentText(MAP_VIEW)
        for k, _ in enumerate(TF_KINDS):                # every map kind draws, with its own colour bar
            preview.kind_box.setCurrentIndex(k)
            self.assertTrue(preview.bar.axis.labelText)
        preview.kind_box.setCurrentIndex(0)             # the map kind is remembered for the session: put it back
        # ERD maps: a decrease is red, an increase blue (Graimann et al. 2002)
        from eeg.ui.plots import erd_colormap
        low, high = erd_colormap().map([0.0, 1.0], mode="byte")
        self.assertGreater(int(low[0]), int(low[2]))
        self.assertGreater(int(high[2]), int(high[0]))

    @unittest.skipUnless(ALPHA.is_file(), "no eyes open / closed recording (set GIOVA_ALPHA_RECORDING)")
    def test_eyes_profile_uses_markers_and_condition_reference(self):
        from eeg.pipeline import Runner
        from eeg.pipeline.convert import from_profile
        eyes = next(p for p in ap.builtin_profiles() if p.name == "Eyes open / closed")
        eyes = ap.AnalysisProfile.from_dict({**eyes.to_dict(), "tab": "band"})
        graph, _ = from_profile(eyes, ALPHA)
        self.assertEqual(graph.params("erd1")["baseline_mode"], ep.CONDITION)
        outputs = Runner(graph).run()
        epochs = outputs["epochs1"]["epochs"]
        self.assertEqual(len(epochs.ids), 8)
        self.assertEqual(set(epochs.labels), {"open", "close"})
        result = outputs["erd1"]["tf"]
        self.assertEqual(result.settings.reference, "open")      # none named: the first condition, as before
        self.assertAlmostEqual(float(np.nanmax(np.abs(result.topo_pooled["alpha"]["open"]))), 0, delta=1e-6)
        self.assertGreater(float(np.min(result.topo_pooled["alpha"]["close"])), 20)   # alpha rises with eyes closed


if __name__ == "__main__":
    unittest.main()
