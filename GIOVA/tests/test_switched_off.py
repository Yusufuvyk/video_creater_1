"""Channels powered down on the board carry no EEG: kept out of results.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

from test_epochs import app  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core import boards  # noqa: E402
from eeg.core.monitoring import flat_channels  # noqa: E402
from eeg.core.sources import hardware_log_path, load_brainflow_csv, load_channel_log  # noqa: E402

T0 = 1700000000.0
NAMES = list(boards.CYTON_DAISY.names)


def game_recording(directory, on=("O1", "O2"), seconds=60, later_off=None):
    """A Cyton+Daisy CSV as the game leaves it: only `on` powered, 10 Hz alpha
    in 'close' blocks, markers 1/2 every 6 s, and the hardware log beside it."""
    board, fs = boards.CYTON_DAISY, boards.CYTON_DAISY.fs
    n = seconds * fs
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    rows = np.zeros((n, board.n_rows))
    rows[:, 0] = (np.arange(n) * 2) % 256
    rows[:, board.timestamp_row] = T0 + t
    for k in range(0, seconds - 6, 6):
        rows[k * fs, board.marker_row] = 1 if (k // 6) % 2 == 0 else 2
    alpha = np.where(((t // 6) % 2 == 1), 12.0, 3.0) * np.sin(2 * np.pi * 10 * t)
    for name in on:
        rows[:, board.eeg_rows[NAMES.index(name)]] = rng.normal(0, 4, n) + alpha
    path = directory / "live_game.csv"
    np.savetxt(path, rows, delimiter="\t")
    lines = ["unix_time\tlocal_time\tnote\ton_bias_per_channel\tsummary"]
    settings = ["10" if name in on else "01" for name in NAMES]
    lines.append("{:.3f}\t12:00:00\tstart\t{}\t-".format(T0, ";".join(settings)))
    if later_off is not None:
        when, name = later_off
        settings[NAMES.index(name)] = "01"
        rows[int(when * fs):, board.eeg_rows[NAMES.index(name)]] = 0
        np.savetxt(path, rows, delimiter="\t")
        lines.append("{:.3f}\t12:00:{:02d}\tapplied (1 command)\t{}\t-".format(T0 + when, int(when), ";".join(settings)))
    hardware_log_path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class CoreTests(unittest.TestCase):
    def test_log_says_which_channels_were_off_and_when(self):
        with scratch() as directory:
            path = game_recording(directory, on=("O1", "O2", "P3", "P4"), later_off=(30, "P4"))
            log = load_channel_log(path, 16)
            self.assertEqual(len(log), 2)
            rec = load_brainflow_csv(path)
            names = lambda indices: sorted(rec.channel_names[i] for i in indices)
            off_everywhere = sorted(set(NAMES) - {"O1", "O2", "P3", "P4"})
            self.assertEqual(names(rec.switched_off(0, 20)), off_everywhere)
            self.assertEqual(names(rec.switched_off(20, 40)), sorted(off_everywhere + ["P4"]))
            self.assertEqual(names(rec.switched_off(40, 50)), sorted(off_everywhere + ["P4"]))
            hardware_log_path(path).write_text("unix_time\tlocal_time\tnote\ton_bias_per_channel\tsummary\n"
                                               "1\t12:00\tstart unicorn\t-\tUnicorn: no channel settings\n")
            self.assertIsNone(load_channel_log(path, 16))          # boards without settings log '-'

    def test_flat_means_the_raw_signal_does_not_move(self):
        fs = 125
        t = np.arange(fs * 30) / fs
        data = np.stack([np.zeros_like(t),                           # powered down: exactly 0
                         np.full_like(t, 3.0),                       # constant offset
                         np.sin(2 * t),                              # 0.3 Hz, nothing in 1-45 Hz: still measured
                         np.random.default_rng(0).normal(0, 5, t.size)])
        self.assertEqual(flat_channels(data), {0, 1})


class AnalysisTests(unittest.TestCase):
    """The Workbench's trial analysis (the Analysis page's before it): channels switched off or flat during the
    trials leave a region and the head map, and are named."""

    def setUp(self):
        app()
        self._scratch = scratch()
        self.directory = self._scratch.__enter__()

    def tearDown(self):
        self._scratch.__exit__(None, None, None)

    def open(self, path):
        from eeg.pipeline import compute
        raw = compute.read_recording(str(path))
        return compute.cut_epochs(raw, tmin=0.0, tmax=6.0)

    def head_map(self, epochs, channels):
        from eeg.pipeline import compute
        from eeg.ui.workbench.previews import HEAD_VIEW, TFPreview
        result = compute.time_frequency(epochs, channels=channels, baseline=[0.0, 1.0], active=[1.0, 5.0],
                                        topography=True)
        preview = TFPreview(result)
        preview.view_box.setCurrentText(HEAD_VIEW)
        return preview

    def test_switched_off_channels_leave_the_roi_and_are_named(self):
        from eeg.pipeline import compute
        epochs = self.open(game_recording(self.directory))
        self.assertIn("Fp1", epochs.unmeasured)
        self.assertNotIn("O1", epochs.unmeasured)
        self.assertIn("not measured (switched off or flat): ", epochs.describe())
        picked = compute.pick_trial_channels(epochs, ["O1", "O2", "Fp1"])
        self.assertEqual(picked.names, ["O1", "O2"])
        features = compute.signal_features(epochs, ["rms"], roi=True)
        self.assertEqual(features.names, ["ROI O1+O2 RMS"])
        with self.assertRaisesRegex(ValueError, "switched off or flat"):
            compute.pick_trial_channels(epochs, ["Fp1"])

    def test_two_measured_electrodes_are_not_painted_over_the_head(self):
        preview = self.head_map(self.open(game_recording(self.directory)), ["O1", "O2"])
        labels = [item.text for item in preview.head.items() if hasattr(item, "text") and isinstance(item.text, str)]
        self.assertTrue(any("at least three measured electrodes" in text for text in labels), labels)
        self.assertIn("left out: ", preview.caption.text())

    def test_brain_map_uses_only_the_measured_electrodes(self):
        import pyqtgraph as pg
        preview = self.head_map(self.open(game_recording(self.directory, on=("O1", "O2", "P3", "P4"))),
                                ["O1", "O2"])
        scatter = [i for i in preview.head.items() if isinstance(i, pg.ScatterPlotItem)][0]
        tips = [p.data() for p in scatter.points()]
        self.assertIn("Fp1 (not measured)", tips)
        self.assertIn("P3", tips)
        image = [i for i in preview.head.items() if isinstance(i, pg.ImageItem)][0].image.T   # (y, x)
        rows = np.linspace(-1, 1, image.shape[0])
        front, back = image[rows > .5], image[rows < -.5]     # nose up: +y is the front of the head
        self.assertFalse(np.isfinite(front).any())            # no electrode measured there
        self.assertTrue(np.isfinite(back).any())

    def test_a_full_montage_still_fills_the_head(self):
        from eeg.ui.topomap import interpolate
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)]) * .8
        values = np.cos(angles)
        full = interpolate(points, values, 1.0)
        inside = np.hypot(*np.meshgrid(np.linspace(-1, 1, full.shape[0]), np.linspace(-1, 1, full.shape[0]))) <= 1
        self.assertTrue(np.isfinite(full[inside]).all())
        part = values.copy(); part[6:] = np.nan                # half the electrodes unmeasured
        limited = interpolate(points, part, 1.0, reach=.5)
        self.assertLess(np.isfinite(limited[inside]).mean(), .6)


class MonitorTests(unittest.TestCase):
    def test_quality_table_says_switched_off(self):
        app()
        from eeg.core.sources import FileSource
        from eeg.ui.monitor import MonitorPage
        from eeg.ui.playback import Player
        with scratch() as directory:
            rec = load_brainflow_csv(game_recording(directory, seconds=20))
            page = MonitorPage()
            try:
                page.show()
                page.source = FileSource(rec)
                page.player = Player(page.source, rec.fs)
                page.quality_table.set_channels(rec.channel_names)
                page._tick(6)
                page._last_quality = -float("inf")
                page._update_live_views()
                status = {page.quality_table.item(r, 0).text(): page.quality_table.item(r, 1).text()
                          for r in range(page.quality_table.rowCount())}
                self.assertEqual(status["Fp1"], "Switched off")
                self.assertNotEqual(status["O1"], "Switched off")
            finally:
                page.stop(); page.close()
                QtCore.QThreadPool.globalInstance().waitForDone(3000)

    def test_switched_off_channels_are_not_drawn(self):
        app()
        from eeg.core.sources import FileSource
        from eeg.ui.monitor import MonitorPage
        from eeg.ui.playback import Player
        with scratch() as directory:
            rec = load_brainflow_csv(game_recording(directory, seconds=20))
            page = MonitorPage()
            try:
                page.show()
                page.source = FileSource(rec)
                page.fs, page.n_channels = rec.fs, rec.n_channels
                page.panel.set_channels(rec.channel_names)
                page.panel.channel_button.set_selected(range(rec.n_channels))
                page.player = Player(page.source, rec.fs)
                page.panel.window_seconds.setValue(4)    # spectra within the 6 s played
                page._begin_streaming()
                page.timer.stop()
                for _ in range(120):
                    page._tick(.05)
                drawn = [rec.channel_names[c] for c in page._drawn_channels()]
                self.assertEqual(drawn, ["O1", "O2"])
                self.assertIn("14 switched off (hidden)", page.time_view.plotItem.titleLabel.text)
                spectrum = page.multi_spectrum
                self.assertEqual([spectrum._names[c] for c in spectrum._channels], ["O1", "O2"])
                self.assertIn("Fp1", spectrum.caption.text())
                # The detail window shows the state, not an empty spectrum.
                page.panel.psd_channel.setCurrentIndex(rec.channel_names.index("Fp1"))
                page.detail.show()
                for _ in range(40):                       # let a spectrum snapshot arrive
                    if getattr(page, "_latest_snapshot", None) is not None:
                        break
                    page._tick(.05)
                    QtCore.QThreadPool.globalInstance().waitForDone(500)
                    QtWidgets.QApplication.processEvents()
                self.assertIsNotNone(getattr(page, "_latest_snapshot", None))
                page._draw_detail()
                self.assertIn("switched off", page.spectrum_view.plotItem.titleLabel.text)
                self.assertEqual(len(page.spectrum_view._curve.getData()[0] or []), 0)
            finally:
                page.stop(); page.close()
                QtCore.QThreadPool.globalInstance().waitForDone(3000)


if __name__ == "__main__":
    unittest.main()
