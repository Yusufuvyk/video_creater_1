"""Downsampling on opening, the unit of the PSD graphs, and BrainVision files.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace  # noqa: F401
from unittest.mock import patch
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

from test_epochs import app  # noqa: E402
from test_settings import release  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core import sources  # noqa: E402
from eeg.core.epochs import EventTable  # noqa: E402
from eeg.core.spectral import (AMPLITUDE, AMPLITUDE_DENSITY, POWER_DENSITY, PSD_UNITS,  # noqa: E402
                               psd_in_unit, welch_psd)
from eeg.ui import help_content as hc  # noqa: E402
from eeg.ui import preferences as prefs  # noqa: E402


def sine(fs, seconds, *tones):
    t = np.arange(int(fs * seconds)) / fs
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in tones)


def amplitude_at(signal, fs, hz):
    freqs, density = welch_psd(signal, fs)
    return float(psd_in_unit(freqs, density, AMPLITUDE)[int(round(hz / freqs[1]))])


class PsdUnitTests(unittest.TestCase):
    def test_a_sine_reads_its_amplitude_in_microvolts(self):
        freqs, density = welch_psd(sine(250.0, 10, (10, 20)), 250.0)
        self.assertEqual(freqs[1], 0.5)
        self.assertAlmostEqual(psd_in_unit(freqs, density, AMPLITUDE)[20], 20.0, delta=0.05)
        # The other two are the density and its root, nothing else.
        np.testing.assert_array_equal(psd_in_unit(freqs, density, POWER_DENSITY), density)
        np.testing.assert_allclose(psd_in_unit(freqs, density, AMPLITUDE_DENSITY) ** 2, density)
        np.testing.assert_array_equal(psd_in_unit(freqs, density, "nonsense"), density)
        self.assertTrue(np.isnan(psd_in_unit(freqs[:1], density[:1], AMPLITUDE)).all())

    def test_units_keep_the_shape_of_many_channels(self):
        freqs, density = welch_psd(np.vstack([sine(250.0, 4, (10, 5)), sine(250.0, 4, (20, 7))]), 250.0)
        for unit in PSD_UNITS:
            self.assertEqual(psd_in_unit(freqs, density, unit).shape, density.shape)


class DownsampleTests(unittest.TestCase):
    def test_the_factor_is_whole_and_never_goes_below_the_target(self):
        for fs, target, factor in ((500, 250, 2), (500, 125, 4), (500, 200, 2), (512, 250, 2),
                                   (250, 250, 1), (125, 250, 1), (500, 0, 1), (500, None, 1),
                                   (1000, 250, 4), (500, 499, 1)):
            self.assertEqual(sources.downsample_factor(fs, target), factor, (fs, target))

    def recording(self):
        data = np.vstack([sine(500.0, 20, (10, 5), (180, 5)), sine(500.0, 20, (10, 3))])
        events = EventTable([1.0, 7.5], None, ["a", "b"])
        recording = sources.Recording(data, 500.0, ["Cz", "Pz"], 1000.0 + np.arange(data.shape[1]) / 500.0,
                                      events=events)
        recording.motion = np.ones((3, data.shape[1]))
        recording.motion_names, recording.motion_units = ["Accel X", "Accel Y", "Accel Z"], ["g"] * 3
        return recording

    def test_the_band_is_kept_and_what_would_alias_is_removed(self):
        original = self.recording()
        before = original.data.copy()
        reduced = sources.downsample(original, 250)
        self.assertEqual((reduced.fs, reduced.n_samples, reduced.original_fs), (250.0, 5000, 500.0))
        self.assertAlmostEqual(reduced.duration, original.duration)
        self.assertIn("downsampled from 500 Hz", reduced.describe())
        self.assertAlmostEqual(amplitude_at(reduced.data[0], 250.0, 10), 5.0, delta=0.05)
        self.assertAlmostEqual(amplitude_at(reduced.data[1], 250.0, 10), 3.0, delta=0.05)
        # 180 Hz at 250 Hz would fold back to 70 Hz with its full 5 µV.
        self.assertLess(amplitude_at(reduced.data[0], 250.0, 70), 0.05)
        self.assertAlmostEqual(amplitude_at(original.data[0][::2], 250.0, 70), 5.0, delta=0.1)
        # Events are in seconds; the clock keeps every second stamp; motion follows.
        self.assertIs(reduced.events, original.events)
        np.testing.assert_array_equal(reduced.timestamps, original.timestamps[::2])
        self.assertAlmostEqual(reduced.start_time, original.start_time)
        self.assertEqual(reduced.motion.shape, (3, 5000))
        np.testing.assert_allclose(reduced.motion, 1.0, atol=1e-9)
        self.assertEqual(reduced.channel_names, ["Cz", "Pz"])
        np.testing.assert_array_equal(original.data, before)           # the source is untouched
        self.assertIsNone(original.original_fs)

    def test_nothing_to_do_returns_the_same_recording(self):
        original = self.recording()
        for target in (0, 500, 1000, 300):
            self.assertIs(sources.downsample(original, target), original)

    def test_missing_samples_keep_their_place(self):
        original = self.recording()
        original.data[0, 2000:2400] = np.nan
        original.gap_samples = 400
        reduced = sources.downsample(original, 125)                    # every 4th sample
        self.assertEqual(reduced.fs, 125.0)
        missing = ~np.isfinite(reduced.data[0])
        self.assertEqual((int(missing.argmax()), int(missing.sum())), (500, 100))
        self.assertTrue(np.isfinite(reduced.data[1]).all())           # other channels are not touched
        self.assertEqual(reduced.gap_samples, 100)
        # Right up to the gap the signal is still the 10 Hz sine, not an edge transient.
        t = np.arange(reduced.n_samples) / 125.0
        np.testing.assert_allclose(reduced.data[0, 300:495], 5 * np.sin(2 * np.pi * 10 * t[300:495]), atol=0.25)


class BrainVisionTests(unittest.TestCase):
    """What Help says a .vhdr gives: EEG in µV, markers as events, no accelerometer."""

    def write(self, directory):
        fs, n = 500, 2000
        eeg = np.vstack([sine(fs, n / fs, (10, 20)), sine(fs, n / fs, (6, 8))])
        stored = np.vstack([eeg / 0.5, np.full((1, n), 980.0)]).astype("<f4")      # resolution 0.5 µV
        stored.T.tofile(directory / "rec.eeg")
        (directory / "rec.vhdr").write_text(
            "BrainVision Data Exchange Header File Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
            "DataFile=rec.eeg\nMarkerFile=rec.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\n"
            "NumberOfChannels=3\nSamplingInterval=2000\n\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n\n"
            "[Channel Infos]\nCh1=Fz,,0.5,µV\nCh2=Cz,,0.5,µV\nCh3=x_dir,,1,mg\n", encoding="utf-8")
        (directory / "rec.vmrk").write_text(
            "BrainVision Data Exchange Marker File Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
            "DataFile=rec.eeg\n\n[Marker Infos]\nMk1=New Segment,,1,1,0,20261007154144599073\n"
            "Mk2=Primary,P  1,501,1,0\nMk3=Primary,P  1,1501,1,0\n", encoding="utf-8")
        return eeg

    def test_a_vhdr_opens_with_its_markers(self):
        from eeg.core.config import RECORDING_FILTER, RECORDING_SUFFIXES
        from eeg.formats.recordings import load_recording
        self.assertIn(".vhdr", RECORDING_SUFFIXES)
        self.assertIn("*.vhdr", RECORDING_FILTER)
        with scratch() as directory:
            eeg = self.write(directory)
            recording = load_recording(directory / "rec.vhdr")
            self.assertEqual((recording.fs, recording.channel_names), (500.0, ["Fz", "Cz"]))
            np.testing.assert_allclose(recording.data, eeg, atol=1e-3)             # microvolts
            # x_dir is not EEG: kept apart as the recording's motion, in the file's own unit
            self.assertEqual(recording.motion_names, ["x_dir"])
            np.testing.assert_allclose(recording.motion[0], 980.0)
            marks = [(round(float(t), 3), label) for t, label in
                     zip(recording.events.onsets, recording.events.labels) if label.startswith("Primary")]
            self.assertEqual(marks, [(1.0, "Primary/P  1"), (3.0, "Primary/P  1")])
            reduced = sources.downsample(recording, 250)
            self.assertEqual((reduced.fs, reduced.n_samples), (250.0, 1000))
            self.assertIs(reduced.events, recording.events)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        app()
        self.store = prefs.user_settings()
        self.before = {key: self.store.value(key) for key in prefs.DEFAULTS}

    def tearDown(self):
        for key, value in self.before.items():
            if value is None:
                self.store.remove(key)
            else:
                self.store.setValue(key, value)

    def test_defaults_change_nothing(self):
        self.assertEqual(prefs.DEFAULTS["graphs/psd_unit"][0], POWER_DENSITY)
        self.assertEqual(prefs.DEFAULTS["files/downsample_hz"][0], 0.0)
        self.store.setValue("graphs/psd_unit", "furlongs")             # a hand-edited ini
        self.assertEqual(prefs.psd_unit(), POWER_DENSITY)

    def test_files_are_reduced_when_opened(self):
        from eeg.ui import matdialog
        from eeg.ui.settings_dialog import FileSettings
        page = FileSettings()
        try:
            recording = sources.Recording(np.zeros((2, 5000)), 500.0)
            self.store.remove("files/downsample_hz")
            self.assertIs(matdialog._reduced(recording, "note")[0], recording)       # off by default
            page.downsample.setValue(250)
            self.assertEqual(prefs.downsample_hz(), 250.0)
            reduced, note = matdialog._reduced(recording, "note")
            self.assertEqual((reduced.fs, reduced.n_samples, note), (250.0, 2500, "note"))
            self.assertEqual(matdialog._reduced(None, "cancelled"), (None, "cancelled"))
            self.store.remove("files/downsample_hz")                   # what Restore defaults does
            page.reload()
            self.assertEqual(page.downsample.value(), 0)
        finally:
            release(page)

    def test_the_unit_reaches_the_monitor_and_analysis_graphs(self):
        from eeg.ui.app import MainWindow
        self.store.remove("graphs/psd_unit")
        window = MainWindow()
        try:
            dialog = window.open_settings("Graphs")
            self.assertEqual(dialog.categories.currentItem().text(), "Graphs")
            panel = window.monitor.multi_spectrum
            panel.mode.setCurrentText("PSD")
            freqs, density = welch_psd(np.vstack([sine(250.0, 4, (10, 20))] * 2), 250.0)
            panel._freqs, panel._values = freqs, density
            self.assertEqual(panel.unit_label(), "µV²/Hz")
            np.testing.assert_array_equal(panel._display(density), density)
            combo = window.graph_settings.unit
            combo.setCurrentIndex(combo.findData(AMPLITUDE))
            self.assertEqual(prefs.psd_unit(), AMPLITUDE)
            self.assertEqual((panel.psd_unit, panel.unit_label()), (AMPLITUDE, "µV"))
            self.assertAlmostEqual(panel._display(density)[0, 20], 20.0, delta=0.1)
            panel.mode.setCurrentText("FFT")                           # the FFT is not converted twice
            panel._freqs = freqs
            np.testing.assert_array_equal(panel._display(density), density)
            self.assertEqual(panel.unit_label(), "µV")
            combo.setCurrentIndex(combo.findData(AMPLITUDE_DENSITY))
            panel.mode.setCurrentText("PSD")
            self.assertEqual(panel.unit_label(), "µV/√Hz")
            self.store.remove("graphs/psd_unit")
            window._settings_reset()
            self.assertEqual((combo.currentData(), panel.psd_unit), (POWER_DENSITY, POWER_DENSITY))
        finally:
            window.close()
            QtCore.QThreadPool.globalInstance().waitForDone(3000)
            release(window)

    def test_the_workbench_spectrum_draws_the_chosen_unit(self):
        from eeg.pipeline.data import Spectrum, AMPLITUDE as FFT_AMPLITUDE
        from eeg.ui.workbench.previews import SNR, SpectrumPreview
        freqs, density = welch_psd(sine(250.0, 4, (10, 20)), 250.0)
        spectrum = Spectrum(freqs, density[None, None, :], ["Cz"], ["A"], ["1"])
        drawn = {}
        try:
            for scale in ("Linear", "dB", SNR):
                for unit in PSD_UNITS:
                    with patch.object(SpectrumPreview, "density_unit", staticmethod(lambda u=unit: u)):
                        preview = SpectrumPreview(spectrum)
                        preview.scale_box.setCurrentText(scale)
                        x, y = preview.plot.listDataItems()[0].getData()
                        drawn[scale, unit] = dict(zip(np.round(x, 6), y))
                        release(preview)
            at = lambda key, f: drawn[key][round(float(f), 6)]
            peak = float(freqs[20])                                  # the sine's frequency (bin 20)
            self.assertAlmostEqual(at(("Linear", AMPLITUDE), peak), 20.0, delta=0.1)
            self.assertAlmostEqual(at(("Linear", AMPLITUDE_DENSITY), peak), float(np.sqrt(density[20])))
            self.assertAlmostEqual(at(("Linear", POWER_DENSITY), peak), float(density[20]))
            # dB of a power and of its root are the same number; the SNR has no unit at all
            self.assertAlmostEqual(at(("dB", AMPLITUDE_DENSITY), peak), at(("dB", POWER_DENSITY), peak))
            self.assertAlmostEqual(at(("dB", AMPLITUDE), peak), 20 * np.log10(20), delta=0.05)
            self.assertAlmostEqual(at((SNR, AMPLITUDE), peak), at((SNR, POWER_DENSITY), peak))
            with patch.object(SpectrumPreview, "density_unit", staticmethod(lambda: AMPLITUDE)):
                fft = SpectrumPreview(Spectrum(freqs, density[None, None, :], ["Cz"], ["A"], ["1"],
                                               kind=FFT_AMPLITUDE))
                fft.scale_box.setCurrentText("Linear")
                x, y = fft.plot.listDataItems()[0].getData()
                from eeg.ui.workbench.previews import SPECTRUM_HZ
                keep = (freqs >= SPECTRUM_HZ[0]) & (freqs <= SPECTRUM_HZ[1])
                np.testing.assert_allclose(y, np.maximum(density[keep], 1e-12))     # FFT untouched
                release(fft)
        finally:
            pass


class LargeWaveTests(unittest.TestCase):
    """A blink larger than the row scale is drawn in full, not cut flat."""

    def setUp(self):
        app()

    def drawn(self, view, data):
        view.draw(data, [0, 1, 2], 150.0, 500.0, ["Fp1", "Fz", "Cz"])
        return view._channel_curves[0].getData()[1], view._channel_curves[0].boundingRect()

    def test_a_wave_crosses_rows_and_stops_at_the_plot_edge(self):
        from eeg.ui.plots import TimeSeriesView
        view = TimeSeriesView()
        try:
            data = np.zeros((3, 200))
            data[0, 50:60] = -450.0                 # three times the ±150 µV row scale
            data[0, 100:110] = 150000.0             # a railed stretch
            self.assertTrue(view.overflow)
            y, rect = self.drawn(view, data)        # rows sit at 4, 2 and 0; the plot spans -2 to 6
            self.assertAlmostEqual(y.min(), 1.0)    # 4 - 3: its true size, below the Fz row at 2
            self.assertAlmostEqual(y.max(), 6.0)    # held at the plot's top edge
            self.assertLessEqual(rect.top(), -2.0)
            self.assertGreaterEqual(rect.bottom(), 6.0)
            view.overflow = False
            y, rect = self.drawn(view, data)
            self.assertEqual((y.min(), y.max()), (3.0, 5.0))      # cut flat at its own row
            self.assertAlmostEqual(rect.top(), 2.9)
            self.assertAlmostEqual(rect.bottom(), 5.1)
            np.testing.assert_array_equal(data[0, 50:60], -450.0)  # samples are never changed
        finally:
            release(view)

    def test_the_monitor_always_draws_them_in_full_and_help_says_so(self):
        """Not a setting: nobody wants a blink with a flat top."""
        from eeg.ui.monitor import MonitorPage
        self.assertNotIn("monitor/trace_overflow", prefs.DEFAULTS)
        page = MonitorPage()
        try:
            page.settings_page()
            self.assertTrue(page.time_view.overflow)
            self.assertFalse(hasattr(page, "trace_overflow"))
            self.assertIn("large waves always in full", HelpTests().text("Parameters explained"))
            self.assertIn("flat top", HelpTests().text("Troubleshooting"))
        finally:
            page.close()
            QtCore.QThreadPool.globalInstance().waitForDone(3000)
            release(page)


class SweepTests(unittest.TestCase):
    """Sweep: a sample stays at one place on the plot until the cursor returns."""

    FS, WINDOW, SCALE = 500.0, 1000, 1e4

    def setUp(self):
        app()
        self.store = prefs.user_settings()
        self.before = self.store.value("monitor/trace_sweep")

    def tearDown(self):
        if self.before is None:
            self.store.remove("monitor/trace_sweep")
        else:
            self.store.setValue("monitor/trace_sweep", self.before)

    def draw(self, view, position, n):
        """Each sample's value is its own number in the stream, so it can be recognised when drawn."""
        data = np.arange(position - n, position, dtype=float)[None, :]
        view.draw(data, [0], self.SCALE, self.FS, ["Cz"], position=position, window=self.WINDOW)
        x, y = view._channel_curves[0].getData()
        return np.rint(x * self.FS).astype(int), np.rint(y * self.SCALE).astype(int)

    def test_every_sample_is_drawn_at_its_own_place(self):
        from eeg.ui.plots import TimeSeriesView
        view = TimeSeriesView()
        try:
            view.sweep = True
            slots, samples = self.draw(view, 2300, 1000)
            np.testing.assert_array_equal(slots, samples % self.WINDOW)      # sample k sits at k mod window
            self.assertTrue((np.diff(slots) > 0).all())                      # one line, left to right
            self.assertEqual(samples.max(), 2299)                            # the newest sample is there
            gap = int(self.WINDOW * view.SWEEP_GAP)
            self.assertFalse(((slots >= 300) & (slots < 300 + gap)).any())   # blank ahead of the cursor
            self.assertIn(300 + gap, slots)
            self.assertTrue(view._cursor.isVisible())
            self.assertAlmostEqual(view._cursor.value(), 300 / self.FS)
            full = view.viewRange()[0]                                       # the whole window, padded
            self.assertAlmostEqual(sum(full) / 2, self.WINDOW / self.FS / 2)
            self.assertLess(full[1] - full[0], 1.1 * self.WINDOW / self.FS)
            # The line lifts between the newest sample and the oldest one beside it.
            lifted = np.flatnonzero(view._channel_curves[0].opts["connect"] == 0)
            self.assertEqual((len(lifted), samples[lifted[0]], samples[lifted[0] + 1]), (1, 2299, 1300 + gap))
            # A little later the same sample is still at the same place.
            later_slots, later_samples = self.draw(view, 2350, 1000)
            self.assertEqual(later_slots[later_samples == 2000][0], slots[samples == 2000][0])
            # A window that is still filling starts at the left, as a sweep does.
            slots, samples = self.draw(view, 400, 400)
            np.testing.assert_array_equal(slots, samples)
            self.assertAlmostEqual(view._cursor.value(), 400 / self.FS)
            self.assertEqual(view.viewRange()[0], full)                      # not a growing axis
        finally:
            release(view)

    def test_scrolling_is_unchanged_and_movement_bands_follow_a_sweep(self):
        from eeg.ui.plots import TimeSeriesView
        view = TimeSeriesView()
        try:
            self.assertFalse(view.sweep)
            slots, samples = self.draw(view, 2300, 1000)
            np.testing.assert_array_equal(slots, samples - 1300)             # oldest left, newest right
            self.assertFalse(view._cursor.isVisible())
            view.shade_spans([(1.0, 1.6)])
            self.assertEqual([tuple(r.getRegion()) for r in view._shades if r.isVisible()], [(1.0, 1.6)])
            view.sweep = True
            self.draw(view, 2300, 1000)             # the oldest sample shown now sits at 0.6 s
            view.shade_spans([(1.0, 1.6)])
            shown = [tuple(round(v, 6) for v in r.getRegion()) for r in view._shades if r.isVisible()]
            self.assertEqual(shown, [(1.6, 2.0), (0.0, 0.2)])
        finally:
            release(view)

    def test_the_setting_reaches_the_monitor_and_help_names_it(self):
        from eeg.ui.monitor import MonitorPage
        self.assertIs(prefs.DEFAULTS["monitor/trace_sweep"][0], False)
        self.store.remove("monitor/trace_sweep")
        page = MonitorPage()
        try:
            settings = page.settings_page()
            self.assertFalse(page.time_view.sweep)
            page.trace_sweep.setCurrentIndex(page.trace_sweep.findData(True))
            self.assertTrue(page.time_view.sweep)
            self.assertTrue(prefs.setting("monitor/trace_sweep"))
            self.store.remove("monitor/trace_sweep")                # what Restore defaults does
            page.apply_settings()
            self.assertFalse(page.time_view.sweep)
            self.assertFalse(page.trace_sweep.currentData())
            label = settings.findChild(QtWidgets.QFormLayout).labelForField(page.trace_sweep).text()
            words = HelpTests().text("Parameters explained")
            self.assertIn("settings → monitor → " + label.lower(), words)
            self.assertIn("sweep", words)
        finally:
            page.close()
            QtCore.QThreadPool.globalInstance().waitForDone(3000)
            release(page)


class TimeLineTests(unittest.TestCase):
    """A click on the playback time line goes to that moment."""

    def test_a_click_seeks_to_where_it_landed(self):
        from pyqtgraph.Qt import QtTest
        from eeg.ui.playback import TransportBar
        app()
        bar = TransportBar()
        try:
            bar.resize(900, 40)
            bar.show()
            QtWidgets.QApplication.processEvents()
            bar.configure(245 * 500, 500.0)                 # 245 s at 500 Hz
            sought = []
            bar.seek_requested.connect(sought.append)
            slider = bar.slider
            for share in (0.75, 0.2, 0.5):
                point = QtCore.QPoint(int(slider.width() * share), slider.height() // 2)
                QtTest.QTest.mouseClick(slider, QtCore.Qt.MouseButton.LeftButton, pos=point)
                self.assertEqual(len(sought), 1, share)     # one seek per click, not page steps
                self.assertAlmostEqual(sought.pop() / 500.0, 245 * share, delta=4.0)
                self.assertFalse(bar._scrubbing)
            # A click on the handle itself stays where it is.
            before = slider.value()
            self.assertEqual(slider.value_at(slider.width() * 0.5), before)
        finally:
            release(bar)


class HelpTests(unittest.TestCase):
    def text(self, title):
        from eeg.ui.help_dialog import plain_text
        return " ".join(plain_text(dict(hc.TOPICS)[title]).split())

    def test_help_says_what_can_be_changed_and_where(self):
        app()
        from eeg.ui.settings_dialog import FileSettings, GraphSettings
        words = self.text("Parameters explained")
        graphs, analysis = GraphSettings(), FileSettings()
        try:
            for unit in PSD_UNITS.values():
                self.assertIn(unit.lower(), words)
            self.assertIn("graphs → psd graphs in", words)
            self.assertIn("files → downsample files to", words)
            self.assertIn("500 hz with 250 hz → 250 hz", words)
            self.assertIn("512 hz → 256 hz", words)
            self.assertEqual((sources.downsample_factor(500, 250), sources.downsample_factor(512, 250)), (2, 2))
            # The labels Help quotes are the ones on the Settings pages.
            self.assertEqual(graphs.layout().labelForField(graphs.unit).text(), "PSD graphs in")
            self.assertEqual(analysis.layout().labelForField(analysis.downsample).text(), "Downsample files to")
            self.assertEqual([graphs.unit.itemData(i) for i in range(graphs.unit.count())], list(PSD_UNITS))
        finally:
            release(graphs, analysis)

    def test_help_gives_the_unit_formulas_the_code_uses(self):
        from eeg.core.spectral import HANN_ENBW_BINS
        words = self.text("Monitor: how it computes")
        self.assertIn("settings → graphs → psd graphs in", words)
        self.assertIn("√(2 · psd · {:g} · δf)".format(HANN_ENBW_BINS), words)
        self.assertIn("√psd", words)

    def test_help_says_brainvision_files_open(self):
        from eeg.formats.mne_io import MNE_SUFFIXES
        words = self.text("Recording files")
        for suffix in MNE_SUFFIXES + (".mat", ".eeg", ".vmrk"):
            self.assertIn(suffix, words)
        for quoted in ("brainvision", "primary/p 1", "keep all three together", "x_dir",
                       "each event's length", "downsample files to"):
            self.assertIn(quoted, words)
        self.assertIn(".vhdr", self.text("Troubleshooting"))


if __name__ == "__main__":
    unittest.main()
