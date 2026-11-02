"""Motion sensor data and the Unicorn's software channel choice.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

from test_epochs import app  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core import boards, hardware as hw  # noqa: E402
from eeg.core.sources import (FileSource, LiveSource, Recording, load_brainflow_csv,  # noqa: E402
                              load_channel_log, hardware_log_path)
from eeg.core.monitoring import MOVEMENT_MG, movement_mg, movement_spans  # noqa: E402
from eeg.ui.live_views import motion_summary  # noqa: E402

T0 = 1700000000.0


def unicorn_csv(directory, seconds=20, gap=None, move_from=None):
    """Unicorn rows: EEG, accel (g), gyro (°/s); the head still, or shaken from `move_from` s."""
    board, fs = boards.UNICORN, boards.UNICORN.fs
    n = seconds * fs
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    rows = np.zeros((n, board.n_rows))
    rows[:, list(board.eeg_rows)] = rng.normal(0, 5, (n, len(board.eeg_rows)))
    rows[:, board.accel_rows[1]] = 1.0                            # gravity on Y
    if move_from is not None:
        moving = t >= move_from
        rows[moving, board.accel_rows[0]] += .6 * np.sin(2 * np.pi * 2 * t[moving])   # a real shake, >30 mg
        rows[moving, board.gyro_rows[2]] = 40 * np.sin(2 * np.pi * 2 * t[moving])
    rows[:, board.timestamp_row] = T0 + t
    if gap is not None:                                           # a real pause in the stream
        rows[int(gap * fs):, board.timestamp_row] += 2
    path = directory / "unicorn.csv"
    np.savetxt(path, rows, delimiter="\t")
    return path


class MotionDataTests(unittest.TestCase):
    def test_unicorn_and_cyton_motion_rows_are_read_with_names_and_units(self):
        self.assertEqual(boards.motion_layout(boards.UNICORN)[2], ["g"] * 3 + ["°/s"] * 3)
        self.assertEqual(boards.motion_layout(boards.CYTON_DAISY)[1], ["Accel X", "Accel Y", "Accel Z"])
        with scratch() as directory:
            rec = load_brainflow_csv(unicorn_csv(directory, gap=10))
            self.assertEqual(rec.motion.shape, (6, rec.n_samples))
            self.assertEqual(rec.motion_names[3:], ["Gyro X", "Gyro Y", "Gyro Z"])
            # Missing samples are missing for the motion sensor too, at the same places.
            np.testing.assert_array_equal(np.isnan(rec.motion[0]), np.isnan(rec.data[0]))
            self.assertGreater(rec.gap_samples, 0)
            source = FileSource(rec)
            source.seek(100)
            chunk = source.read(50)
            np.testing.assert_array_equal(source.last_motion, rec.motion[:, 100:150])
            self.assertEqual(chunk.shape[1], source.last_motion.shape[1])
        plain = FileSource(Recording(np.zeros((2, 100)), 125))
        plain.read(10)
        self.assertIsNone(plain.last_motion)
        self.assertEqual(plain.motion_names, [])

    def test_movement_is_zero_at_rest_and_rises_when_the_head_moves(self):
        fs = 250
        still = np.zeros((3, fs)); still[1] = 1.0            # gravity on one axis, head still
        self.assertAlmostEqual(float(np.max(np.abs(movement_mg(still)))), 0.0, places=6)
        moving = still.copy()
        moving[0] = .5 * np.sin(2 * np.pi * 2 * np.arange(fs) / fs)
        mg = movement_mg(moving)
        self.assertGreater(float(np.max(np.abs(mg))), MOVEMENT_MG)
        spans = movement_spans(mg, fs)
        self.assertTrue(spans)
        self.assertTrue(all(0 <= a < b <= 1 for a, b in spans))
        self.assertEqual(movement_spans(np.zeros(fs), fs), [])          # nothing to mark
        self.assertEqual(list(movement_mg(np.zeros((2, 10)))), [0.0] * 10)   # no accelerometer

    def test_summary_tells_still_from_moving(self):
        fs = 250
        still = np.zeros((6, fs)); still[1] = 1.0
        text = motion_summary(still, fs, ["g"] * 3 + ["°/s"] * 3)
        self.assertIn("|a| 1.00 g", text); self.assertIn("still ±0 mg", text)
        self.assertIn("rotation 0.0", text)
        moving = still.copy()
        moving[0] = .5 * np.sin(2 * np.pi * 2 * np.arange(fs) / fs)
        moving[5] = 40
        text = motion_summary(moving, fs, ["g"] * 3 + ["°/s"] * 3)
        self.assertIn("moving ±", text)
        self.assertIn("rotation 40.0", text)
        self.assertEqual(motion_summary(still[:3], fs, ["g"] * 3).count("rotation"), 0)   # Cyton: no gyro


class MonitorMotionTests(unittest.TestCase):
    def setUp(self):
        app()
        from eeg.ui.monitor import MonitorPage
        self.page = MonitorPage()
        self.page.resize(1366, 768)
        self.page.show()

    def tearDown(self):
        self.page.stop(); self.page.close()
        QtCore.QThreadPool.globalInstance().waitForDone(3000)

    def play(self, recording, seconds):
        from eeg.ui.playback import Player
        page = self.page
        page.source = FileSource(recording)
        page.fs, page.n_channels = recording.fs, recording.n_channels
        page.panel.set_channels(recording.channel_names)
        page.player = Player(page.source, recording.fs)
        page._begin_streaming()
        page.timer.stop()
        for _ in range(int(seconds / .05)):
            page._tick(.05)

    def test_playback_draws_the_motion_sensor_under_the_traces(self):
        with scratch() as directory:
            rec = load_brainflow_csv(unicorn_csv(directory, move_from=5))
        self.play(rec, 8)
        view = self.page.motion_view
        self.assertTrue(view.isVisibleTo(self.page))
        self.assertIs(view.parentWidget(), self.page.signals_page)
        self.assertEqual(self.page.motion_buffer.shape[0], 6)
        x, y = view._curves[0][1].getData()
        self.assertGreater(len(x), 10)
        self.assertIn("rotation", view.readout.text())
        self.assertIsNotNone(view.movement)                           # movement, above the axes
        mx, my = view.movement.getData()
        self.assertGreater(float(np.max(np.abs(my))), MOVEMENT_MG)
        self.assertTrue(self.page.time_view._shades)                  # and marked on the traces
        self.assertTrue(any(s.isVisible() for s in self.page.time_view._shades))
        self.assertIn("moving ±", view.readout.text())               # the head moves after 5 s
        # A moment sits at the same x in the traces and the motion plots.
        def edges(box, widget):
            return [widget.mapToGlobal(widget.mapFromScene(point)).x() for point in (box.topLeft(), box.topRight())]
        traces = self.page.time_view.plotItem.vb
        for _ in range(2):
            QtWidgets.QApplication.processEvents()
        for plot in view._plots:
            for a, b in zip(edges(plot.vb.sceneBoundingRect(), view.graphics),
                            edges(traces.sceneBoundingRect(), self.page.time_view)):
                self.assertLessEqual(abs(a - b), 1)
            np.testing.assert_allclose(plot.vb.viewRange()[0], traces.viewRange()[0], atol=1e-6)
        view.toggle.setChecked(False)                                 # hidden on request
        self.assertFalse(view.graphics.isVisibleTo(self.page))

    def test_no_motion_sensor_no_view_and_the_stream_goes_on(self):
        self.play(Recording(np.random.default_rng(0).normal(0, 5, (8, 2500)), 250), 2)
        self.assertFalse(self.page.motion_view.isVisibleTo(self.page))
        self.assertGreater(self.page.filled, 0)
        # A source reporting nonsense for its motion sensor must not stop the stream.
        self.page.source.last_motion = Mock()
        self.page._ingest_motion(self.page.source.last_motion)
        self.page._tick(.05)
        self.assertGreater(self.page.filled, 0)


class UnicornSoftwareChannelTests(unittest.TestCase):
    def test_log_records_the_choice_so_analysis_can_leave_channels_out(self):
        with scratch() as directory:
            source = LiveSource.__new__(LiveSource)                      # no headset needed
            source.info = boards.UNICORN
            source.record_path = directory / "live_unicorn.csv"
            settings = [hw.ChannelSetting(name not in ("Cz", "C4"), False) for name in boards.UNICORN.names]
            source._log_hardware(settings, "start unicorn")
            text = hardware_log_path(source.record_path).read_text(encoding="utf-8")
            self.assertIn("software only", text); self.assertIn("off: Cz, C4", text)
            log = load_channel_log(source.record_path, 8)
            self.assertEqual([boards.UNICORN.names[i] for i, on in enumerate(log[0][1]) if not on], ["Cz", "C4"])

    def test_apply_marks_channels_off_without_touching_the_headset(self):
        app()
        from eeg.ui.monitor import MonitorPage
        page = MonitorPage()
        try:
            page.panel.mode.setCurrentIndex(2)                         # Live (Unicorn)
            source = Mock(fs=250, motion_names=[], motion_units=[])
            source.info = boards.UNICORN
            source.read.return_value = np.random.default_rng(0).normal(0, 5, (8, 25))
            source.last_motion = None
            page.source, page.player = source, None
            page.fs, page.n_channels = 250, 8
            page.panel.set_channels(list(boards.UNICORN.names))
            page._begin_streaming(); page.timer.stop()
            self.assertTrue(page.is_live)
            page.panel.unicorn_channels.only(["Fz", "C3", "Pz"])
            page._apply_hardware()
            self.assertEqual([s.on for s in page._applied_hardware].count(True), 3)
            source.configure.assert_not_called()                       # no command reaches the headset
            self.assertEqual(source.log_note.call_args[0][0], "channels in use (software)")
            self.assertEqual(page.panel.selected_channels, [0, 1, 4])
            for _ in range(90):                                        # four seconds of samples
                page._tick(.05)
            page._last_quality = -float("inf")
            page._update_live_views()
            status = {page.quality_table.item(r, 0).text(): page.quality_table.item(r, 1).text()
                      for r in range(page.quality_table.rowCount())}
            self.assertEqual(status["Cz"], "Switched off")
            self.assertNotEqual(status["Fz"], "Switched off")
            # A game's DISABLED_CHANNELS arrive the same way.
            reply = page.game_configure(["Cz", "C4", "Pz", "PO7", "Oz", "PO8"], False, "flicker_game_v6_gtec")
            self.assertIn("applied in software", reply)
            self.assertEqual([s.on for s in page._applied_hardware], [True, True] + [False] * 6)
            # A game reads the raw stream, and a channel switched off in
            # software must be off there too.
            page._ingest(np.full((8, 10), 7.0))
            data, _ = page.tap.last(1)
            newest = data[:, -10:]                             # what the push just added
            np.testing.assert_array_equal(newest[2:], 0.0)
            np.testing.assert_array_equal(newest[:2], 7.0)
        finally:
            page.stop(); page.close()
            QtCore.QThreadPool.globalInstance().waitForDone(3000)


if __name__ == "__main__":
    unittest.main()
