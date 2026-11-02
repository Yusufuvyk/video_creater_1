"""Numerical contracts, lifecycle regressions, and linked feature interactions."""
from test_trial_analysis import scratch
import os
from pathlib import Path
import csv
import json
import time
import unittest
from unittest.mock import patch
import numpy as np
os.environ["_MNE_FAKE_HOME_DIR"] = str(Path(__file__).parent / ".tmp")
from pyqtgraph.Qt import QtCore, QtWidgets

from eeg.core.spectral import band_integral, welch_psd, spectral_entropy, hjorth
from eeg.core.feature_profiles import (FeatureProfile, extract, summary, cliffs_delta,
                                       time_features, save_profile, load_profile, condition_values)
from eeg.core.features import FeatureSet
from eeg.core.trials import Trial, Settings, analyze
from eeg.core.sources import Recording, FileSource, load_brainflow_csv
from eeg.core.processing import filter_finite, StreamFilter, design_filter, AutoScaler
from eeg.core import epochs as ep
from eeg.formats.trial_analysis import save_session, load_session, export_csv
from eeg.formats.matfile import parse_rows, MatError
from eeg.ui.playback import Player


class FeatureMathTests(unittest.TestCase):
    def setUp(self):
        self.fs = 125
        self.x = 3*np.sin(2*np.pi*8*np.arange(1250)/self.fs)
        self.profile = FeatureProfile()

    def test_band_edges_partition_and_interpolated_edges(self):
        f, p = welch_psd(self.x, self.fs)
        powers = [band_integral(f, p, a, b) for a, b in [(0, 4), (4, 8), (8, 13), (13, 30), (30, 62.5)]]
        self.assertAlmostEqual(sum(powers), 4.5)
        self.assertAlmostEqual(band_integral([0, 1, 2], [0, 2, 4], .25, 1.75), 3)

    def test_channel_scales_are_independent_and_do_not_change_power(self):
        from eeg.core.processing import ChannelAutoScaler
        data = np.stack([self.x, 100*self.x])
        original = data.copy()
        scales = ChannelAutoScaler().update(data, [0, 1], now=0)
        self.assertGreater(scales[1], scales[0]*20)
        small = ChannelAutoScaler().update(data, [0], now=0)
        self.assertEqual(small[0], scales[0])
        np.testing.assert_array_equal(data, original)
        scaler = ChannelAutoScaler()
        scaler.update(data, [0, 1], now=0)
        data[1] *= 100
        self.assertEqual(scaler.update(data, [0, 1], now=1)[0], small[0])

    def test_live_offline_same_processed_samples(self):
        result = analyze(self.x[None, :], self.fs, 0, "O1", [Trial("1", "open", 0, 10)], Settings(10))
        live = FeatureSet(self.fs, 2, profile=self.profile).compute(result.trials[0].signal[None, :])[0]
        offline = [result.features.trials["1"].values[k] for k in self.profile.selected]
        np.testing.assert_allclose(live, offline, rtol=1e-12)

    def test_relative_uses_direct_total_not_custom_band_sum(self):
        p = FeatureProfile(bands=[["alpha", 8, 13], ["beta", 8, 13]], ratios=[["alpha/beta", "alpha", "beta"]])
        f = np.arange(0, 63, .5)
        power = np.ones(len(f))
        values = extract(self.x, self.fs, p, psd=(f, power)).values
        self.assertAlmostEqual(values["relative:alpha"], 100*5/44)
        self.assertAlmostEqual(values["ratio:alpha/beta"], 1)

    def test_entropy_fixed_grid_and_zero_denominator_missing(self):
        self.assertAlmostEqual(float(spectral_entropy([1, 1, 0, 0])), .5)
        self.assertAlmostEqual(float(spectral_entropy([1, 1, 1, 1])), 1)
        self.assertTrue(np.isnan(spectral_entropy([0, 0, 0, 0])))
        row = extract(np.zeros(500), self.fs, self.profile)
        for key in ("entropy", "ratio:alpha/beta", "relative:alpha", "mobility", "complexity"):
            self.assertTrue(np.isnan(row.values[key]), key)
            self.assertIn(key, row.reasons)

    def test_nyquist_missing_and_profile_validation(self):
        row = extract(np.sin(np.arange(100)), 20, self.profile)
        self.assertTrue(np.isnan(row.values["band:alpha"]))
        self.assertIn("Nyquist", row.reasons["band:alpha"])
        with self.assertRaises(ValueError):
            FeatureProfile(total_range=[45, 1]).validate()
        with self.assertRaises(ValueError):
            FeatureProfile(ratios=[["bad", "alpha", "missing"]]).validate()

    def test_hjorth_uses_sampling_rate_and_dc_removal(self):
        for fs in (500, 1000):
            x = 8+3*np.sin(2*np.pi*5*np.arange(fs*10)/fs)
            activity, mobility, complexity = hjorth(x, fs)
            self.assertAlmostEqual(activity, 4.5)
            self.assertAlmostEqual(mobility, 2*np.pi*5, delta=.04)
            self.assertAlmostEqual(complexity, 1, delta=.003)
            row = extract(x, fs, self.profile)
            self.assertAlmostEqual(row.values["rms"], 3/np.sqrt(2))

    def test_cliffs_delta_ties_direction_and_minimum(self):
        self.assertEqual(cliffs_delta([1, 2, 3], [4, 5, 6]), 1)
        self.assertEqual(cliffs_delta([4, 5, 6], [1, 2, 3]), -1)
        self.assertEqual(cliffs_delta([1, 2, 3], [1, 2, 3]), 0)
        self.assertTrue(np.isnan(cliffs_delta([1, 2, np.nan], [4, 5, 6])))
        self.assertEqual(summary([1, np.nan, 3])["n"], 2)

    def test_time_windows_do_not_mix_trials_or_inflate_n(self):
        data = np.r_[self.x[:625], 2*self.x[:625]][None, :]
        result = analyze(data, self.fs, 0, "O1", [Trial("a", "open", 0, 5), Trial("b", "closed", 5, 10)], Settings(5))
        rows = time_features(result, self.profile)
        np.testing.assert_array_equal(rows["a"][0], [1, 2, 3, 4])
        self.assertEqual(len(condition_values(result, "open", "rms")), 1)
        for a, b in zip(rows["a"][1], rows["b"][1]):
            self.assertAlmostEqual(b.values["rms"], 2*a.values["rms"])
        self.profile.time_window = 2.3
        self.assertEqual(len(time_features(result, self.profile)["a"][1]), 3)

    def test_gaps_preserved_and_filtered_runs_independent(self):
        x = self.x.copy()
        x[625] = np.nan
        filtered = filter_finite(x, self.fs, low=1, high=45)
        self.assertTrue(np.isnan(filtered[625]))
        np.testing.assert_allclose(filtered[:625], filter_finite(x[:625], self.fs, low=1, high=45))
        result = analyze(x[None, :], self.fs, 0, "O1", [Trial("1", "open", 0, 4), Trial("2", "closed", 4, 8)],
                         Settings(4))
        self.assertEqual(len(result.trials), 1)
        self.assertIn("2", result.rejected)
        stream = StreamFilter(design_filter(self.fs, low=1, high=45), 1)
        output = stream.process(x[None, :])
        self.assertTrue(np.isfinite(output[0, 626:]).all())
        self.assertTrue(np.isfinite(AutoScaler().update(output)))

    def test_csv_invalid_and_blank_rows_preserve_sample_clock(self):
        with scratch() as directory:
            path = directory/"damaged.csv"
            rows = np.zeros((4, 32))
            rows[:, 1] = [1, np.nan, np.nan, 4]
            rows[:, 30] = 100 + np.arange(4) / 125
            lines = ["\t".join(map(str, row)) for row in rows]
            lines[1] = lines[1].replace("nan", "bad")
            lines[2] = ""
            path.write_text("\n".join(lines) + "\n")
            record = load_brainflow_csv(path, board_id=2)
            self.assertEqual(record.n_samples, 4)
            self.assertTrue(np.isnan(record.data[0, 1]))
            self.assertTrue(np.isnan(record.data[:, 2]).all())
        record = Recording(np.zeros((1, 4)), 125, ["O1"], [np.nan, 100.008, 100.016, 100.024])
        self.assertAlmostEqual(record.start_time, 100)

    def test_invalid_mat_rows_never_mean_all(self):
        for value in ("100", "1-100", "bad"):
            with self.assertRaises(MatError):
                parse_rows(value, 16)

    def test_profile_session_v1_migration_and_feature_export(self):
        result = analyze(self.x[None, :], self.fs, 0, "O1", [Trial("1", "open", 0, 10)], Settings(10))
        record = Recording(self.x[None, :], self.fs, ["O1"])
        with scratch() as directory:
            profile = directory/"profile.json"
            save_profile(profile, self.profile)
            self.assertEqual(load_profile(profile), self.profile)
            path = directory/"analysis.json"
            save_session(path, record, "synthetic.csv", "O1", [Trial("1", "open", 0, 10)], Settings(10))
            payload = json.loads(path.read_text())
            payload["version"] = 1
            payload["settings"].pop("profile")
            payload["settings"].pop("quality_mad")
            path.write_text(json.dumps(payload))
            settings = load_session(path, record, "synthetic.csv")[2]
            self.assertEqual(FeatureProfile.from_dict(settings.profile), self.profile)
            out = directory/"features.csv"
            export_csv(out, result, "synthetic.csv", "open", time_features(result, self.profile))
            with out.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            row = next(r for r in rows if r["calculation"] == "Trial feature")
            self.assertEqual(row["trial_ids"], "1")
            self.assertEqual(json.loads(row["profile"])["name"], self.profile.name)
            self.assertTrue(row["definition"])
            self.assertTrue(any(r["calculation"] == "Within-trial feature" for r in rows))

    def test_quality_flags_do_not_exclude_calculable_trials(self):
        chunks = [self.x[:500]*a for a in (1, 1, 1, 1, 100, 0)]
        result = analyze(np.concatenate(chunks)[None, :], self.fs, 0, "O1",
                         [Trial(str(i), "open", i*4, (i+1)*4) for i in range(6)], Settings(4))
        self.assertEqual(len(result.trials), 6)
        self.assertTrue(result.quality["4"])
        self.assertTrue(result.quality["5"])

    def test_stft_density_matches_welch_and_trial_boundaries(self):
        whole = ep.EpochSettings(tmin=0, tmax=10, baseline=[0, 1], active=[1, 3], fmin=1, fmax=45, window=2,
                                 overlap=.5, smooth=0, topography=False)
        freqs, _, power, units = ep.stft_power(self.x[None, :], self.fs, whole)
        f, p = welch_psd(self.x, self.fs)
        keep = (f >= 1) & (f <= 45)
        np.testing.assert_allclose(freqs, f[keep])
        np.testing.assert_allclose(power[0].mean(axis=1), p[keep], atol=1e-12)    # Welch is the mean of frames
        self.assertEqual(units, "µV²/Hz")
        events = ep.EventTable([0.0], None, ["open"])
        settings = ep.EpochSettings(tmin=0, tmax=4, baseline=[0, 1], active=[1, 3], fmin=1, fmax=45, window=1,
                                    overlap=.5, smooth=0, topography=False, bands=[["alpha", 8, 13]])
        first = ep.compute_epochs(self.x[None, :], self.fs, ["O1"], [0], events, settings, ids=["1"])
        changed = self.x.copy()
        changed[500:] *= 20
        other = ep.compute_epochs(changed[None, :], self.fs, ["O1"], [0], events, settings, ids=["1"])
        self.assertEqual(first.epoch_ids, {"open": ["1"]})
        self.assertEqual(first.tf[0].units, "µV²/Hz")
        np.testing.assert_allclose(first.tf[0].power["open"], other.tf[0].power["open"], atol=1e-9)

    def test_erd_baseline_bounds_and_morlet_units(self):
        events = ep.EventTable([0.0], None, ["open"])
        with self.assertRaisesRegex(ValueError, "inside"):
            ep.compute_epochs(self.x[None, :], self.fs, ["O1"], [0], events,
                              ep.EpochSettings(tmin=0, tmax=4, baseline=[3, 5], active=[1, 2]))
        result = ep.compute_epochs(self.x[None, :], self.fs, ["O1"], [0], events,
                                   ep.EpochSettings(tmin=0, tmax=4, baseline=[0, 1], active=[1, 3], method="morlet",
                                                    topography=False))
        self.assertEqual(result.tf[0].units, "µV²")

    def test_erd_band_trace_uses_band_power_ratio(self):
        x = np.sin(2*np.pi*10*np.arange(1000)/125)
        x[500:] *= 2
        settings = ep.EpochSettings(tmin=0, tmax=8, baseline=[0.5, 3.5], active=[4.5, 7.5], fmin=2, fmax=40, window=1,
                                    overlap=.5, smooth=0, topography=False, bands=[["alpha", 8, 13]])
        result = ep.compute_epochs(x[None, :], 125, ["O1"], [0], ep.EventTable([0.0], None, ["open"]), settings)
        course, _, _ = ep.band_course(result.tf[0], "open", (8, 13))
        times = result.tf[0].times
        self.assertAlmostEqual(float(np.mean(course[(times > 1) & (times < 3)])), 0, delta=2)
        self.assertAlmostEqual(float(np.mean(course[(times > 5) & (times < 7)])), 300, delta=6)


class PlaybackTests(unittest.TestCase):
    def test_fractional_clock_pause_speed_and_seek(self):
        source = FileSource(Recording(np.zeros((1, 10000)), 125, ["O1"]))
        player = Player(source, 125)
        for _ in range(200):
            player.advance(.05)
        self.assertEqual(player.position, 1250)
        player.pause()
        player.advance(10)
        self.assertEqual(player.position, 1250)
        player.seek_samples(0)
        player.speed = .25
        player.play()
        for _ in range(200):
            player.advance(.05)
        self.assertEqual(player.position, 312)
        player.seek_samples(0)
        player.speed = 1
        player.advance(.004)
        self.assertEqual(player.position, 0)
        player.advance(.004)
        self.assertEqual(player.position, 1)


class FeatureUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_profile_dialog_and_linked_trial_mean_mode(self):
        from eeg.ui.feature_view import FeatureView, ProfileDialog
        fs = 125
        x = np.sin(2*np.pi*10*np.arange(1000)/fs)
        result = analyze(x[None, :], fs, 0, "O1", [Trial("a", "open", 0, 4), Trial("b", "closed", 4, 8)], Settings(4))
        view = FeatureView()
        view.set_result(result)
        view.select_row(0)
        import pyqtgraph as pg
        points = [p for p in view.plot.items() if isinstance(p, pg.ScatterPlotItem) and len(p.points())]
        received = []
        view.trial_requested.connect(received.append)
        points[0].sigClicked.emit(points[0], [points[0].points()[0]], None)
        self.assertIn(received[0], ["a", "b"])
        view.source.setCurrentIndex(1)
        self.assertEqual(view.table.item(0, 6).text(), "")      # Cliff's delta of one mean each: does not apply
        self.assertFalse(view.time_toggle.isEnabled())
        dialog = ProfileDialog(None, FeatureProfile())
        dialog.high.setValue(40)
        self.assertEqual(dialog.read().total_range, [1, 40])
        dialog.close()
        view.close()

    def test_stale_worker_does_not_replace_current_result(self):
        from eeg.ui.jobs import Jobs
        jobs = Jobs()
        received = []
        jobs.finished.connect(lambda value, error: received.append(value))
        jobs.start(lambda: (time.sleep(.03), "old")[1])
        jobs.invalidate()
        jobs.start(lambda: "new")
        QtCore.QThreadPool.globalInstance().waitForDone(5000)
        self.app.processEvents()
        self.assertEqual(received, ["new"])

    def test_monitor_trace_throttling_and_paused_redraw(self):
        from eeg.ui.monitor import MonitorPage
        page = MonitorPage()
        page.show()
        self.app.processEvents()
        page._ingest(np.tile(np.sin(np.arange(1250)), (16, 1)))
        clock = [0.]
        with patch("eeg.ui.monitor.time.perf_counter", side_effect=lambda: clock[0]), \
                patch.object(page.time_view, "draw", wraps=page.time_view.draw) as draw:
            for i in range(20):
                clock[0] = i*.05
                page.sample_position += 6
                page._new_samples = True
                page._render()
            self.assertEqual(draw.call_count, 20)
            clock[0] = 2
            page._render()
            for _ in range(10):
                page._render()
            self.assertEqual(draw.call_count, 20)
        page.close()

    def test_trace_labels_show_individual_scales(self):
        from eeg.ui.plots import TimeSeriesView
        plot = TimeSeriesView()
        data = np.array([[0., 5., -5.], [0., 500., -500.]])
        original = data.copy()
        plot.draw(data, [0, 1], np.array([5., 500.]), 125, ["O1", "O2"])
        labels = plot.getAxis("left")._tickLevels[0]
        self.assertIn("5 uV", labels[0][1])
        self.assertIn("500 uV", labels[1][1])
        _, first = plot._channel_curves[0].getData()
        _, second = plot._channel_curves[1].getData()
        self.assertAlmostEqual(np.ptp(first), np.ptp(second))
        np.testing.assert_array_equal(data, original)
        plot.close()

    def test_live_log_settings_revision_splits_files(self):
        from eeg.ui.features_window import FeaturesWindow
        with scratch() as directory:
            with patch("eeg.ui.features_window.data_dir", lambda: directory):
                view = FeaturesWindow()
                view.on_source(125, ["O1"])
                view.record.setChecked(True)
                x = np.sin(2*np.pi*10*np.arange(375)/125)[None, :]
                view.on_frame(x, 125, ["O1"], 3)
                first = view.log.path
                view.reset()
                second = view.log.path
                view.on_frame(x, 125, ["O1"], 4)
                view.record.setChecked(False)
                self.assertNotEqual(first, second)
                with first.open() as handle:
                    a = list(csv.DictReader(handle))
                with second.open() as handle:
                    b = list(csv.DictReader(handle))
                self.assertEqual(len(a), 5)
                self.assertEqual(len(b), 5)
                self.assertNotEqual(json.loads(a[0]["settings"])["revision"], json.loads(b[0]["settings"])["revision"])
                view.close()

    def test_png_embeds_definitions_and_mat_invalid_preview(self):
        from eeg.ui.exporting import save_png, result_metadata
        from eeg.ui.matdialog import MatMappingDialog
        from pyqtgraph.Qt import QtGui
        x = np.sin(np.arange(500))
        result = analyze(x[None, :], 125, 0, "O1", [Trial("1", "open", 0, 4)], Settings(4))
        widget = QtWidgets.QLabel("Features")
        widget.resize(300, 100)
        with scratch() as directory:
            path = directory/"result.png"
            self.assertTrue(save_png(widget, path, result_metadata(result, "synthetic.csv")))
            data = json.loads(QtGui.QImage(str(path)).text("Analysis"))
            self.assertEqual(data["trials"][0]["id"], "1")
            self.assertEqual(len(data["feature_definitions"]), 5)
        dialog = MatMappingDialog(None, "sample.mat", "", {"eeg": np.zeros((2, 500)), "fs": np.array([[125.]])})
        dialog.rows.setText("100")
        self.assertFalse(dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).isEnabled())
        dialog.close()
        widget.close()

    def test_monitor_independent_buffer_pause_seek_filters_and_source(self):
        from eeg.ui.monitor import MonitorPage
        page = MonitorPage()
        page.show()
        page.open_features()
        self.app.processEvents()
        fs = 125
        x = np.tile(np.sin(2*np.pi*10*np.arange(5000)/fs), (16, 1))
        page.source = FileSource(Recording(x, fs, page.panel.channel_names))
        page.player = Player(page.source, fs)
        page.fs = fs
        page.panel.window_seconds.setValue(1)
        page._resize_buffer()
        page.features_window.window_seconds.setValue(4)
        page._tick(6)
        self.assertEqual(page.buffer.shape[1], 125)
        self.assertEqual(page.feature_buffer.shape[1], 625)
        self.assertEqual(page.features_window.history.filled, 1)
        page.player.pause()
        for _ in range(5):
            page._tick(.05)
        self.assertEqual(page.features_window.history.filled, 1)
        page._seek(0)
        self.assertEqual(page.features_window.history.filled, 0)
        page.player.play()
        page._tick(6)
        page._rebuild_stream()
        self.assertEqual(page.features_window.history.filled, 0)
        self.assertEqual(page.feature_buffer.shape[1], 0)
        page.features_window.on_source(fs, page.panel.channel_names)
        self.assertEqual(page.features_window.history.filled, 0)
        page.player.seek_samples(page.source.n_total)
        page.player.pause()
        page._toggle_play()
        self.assertEqual(page.sample_position, 0)
        page.stop()
        page.close()


if __name__ == "__main__":
    unittest.main()
