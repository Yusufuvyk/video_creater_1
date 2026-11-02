"""Smooth display rate: 30 fps when the computer keeps up, 20 fps when it does not."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from eeg.ui import monitor as mon
from eeg.ui.preferences import set_setting, user_settings

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class DisplayRateTests(unittest.TestCase):
    def setUp(self):
        user_settings().remove('monitor/display_fps')
        self.page = mon.MonitorPage()
        self.page._frame_ms = mon.SMOOTH_FRAME_MS
        self.page.timer.start(mon.SMOOTH_FRAME_MS)

    def tearDown(self):
        self.page.timer.stop()
        user_settings().remove('monitor/display_fps')

    def test_precise_timer(self):
        self.assertEqual(self.page.timer.timerType(), QtCore.Qt.TimerType.PreciseTimer)

    def test_keeps_30_fps_when_frames_are_on_time(self):
        self.page._frame_costs, self.page._frame_gaps = [15.0] * 20, [33.0] * 20
        self.page._keep_up()
        self.assertEqual(self.page._frame_ms, mon.SMOOTH_FRAME_MS)

    def test_drops_to_20_fps_when_frames_are_late(self):
        said = []
        self.page.status_changed.connect(said.append)
        self.page._frame_costs, self.page._frame_gaps = [15.0] * 20, [60.0] * 20
        self.page._keep_up()
        self.assertEqual(self.page._frame_ms, mon.FRAME_MS)
        self.assertEqual(self.page.timer.interval(), mon.FRAME_MS)
        self.assertTrue(any("20 fps" in text for text in said))

    def test_drops_to_20_fps_when_frames_cost_too_much(self):
        self.page._frame_costs, self.page._frame_gaps = [30.0] * 20, [33.0] * 20
        self.page._keep_up()
        self.assertEqual(self.page._frame_ms, mon.FRAME_MS)

    def test_rate_setting_chooses_30_or_60_fps(self):
        set_setting('monitor/display_fps', 60)
        self.assertEqual(self.page._target_frame_ms(), mon.FAST_FRAME_MS)
        set_setting('monitor/display_fps', 30)
        self.assertEqual(self.page._target_frame_ms(), mon.SMOOTH_FRAME_MS)
        set_setting('monitor/display_fps', 20)                  # an older stored choice
        self.assertEqual(self.page._target_frame_ms(), mon.SMOOTH_FRAME_MS)
        self.settings = self.page.settings_page()
        self.assertEqual([self.page.display_rate.itemData(i) for i in range(self.page.display_rate.count())],
                         [30, 60])

    def test_60_fps_steps_down_to_30_then_20(self):
        said = []
        self.page.status_changed.connect(said.append)
        self.page._frame_ms = mon.FAST_FRAME_MS
        self.page.timer.start(mon.FAST_FRAME_MS)
        self.page._frame_costs, self.page._frame_gaps = [5.0] * 20, [17.5] * 20     # 57 fps: fine
        self.page._keep_up()
        self.assertEqual(self.page._frame_ms, mon.FAST_FRAME_MS)
        self.page._frame_costs, self.page._frame_gaps = [5.0] * 20, [25.0] * 20     # 40 fps: late
        self.page._keep_up()
        self.assertEqual(self.page._frame_ms, mon.SMOOTH_FRAME_MS)
        self.assertEqual(self.page.timer.interval(), mon.SMOOTH_FRAME_MS)
        self.assertIn("display at 30 fps", said[-1])
        self.assertIn("keep up with 60 fps", said[-1])
        self.page._frame_costs, self.page._frame_gaps = [5.0] * 20, [48.0] * 20
        self.page._keep_up()
        self.assertEqual(self.page._frame_ms, mon.FRAME_MS)
        self.assertIn("display at 20 fps", said[-1])

    def test_one_slow_frame_does_not_step_down(self):
        self.page._frame_ms = mon.FAST_FRAME_MS
        self.page._frame_costs = [4.0] * 19 + [40.0]
        self.page._frame_gaps = [17.0] * 19 + [60.0]
        self.page._keep_up()
        self.assertEqual(self.page._frame_ms, mon.FAST_FRAME_MS)

    def test_trace_width_setting(self):
        """Thin 1 px by default (fast); Bold 2 px restyles the drawn curves at once."""
        from eeg.ui import plots
        user_settings().remove('monitor/trace_width')
        try:
            page = mon.MonitorPage()
            self.assertEqual(plots.TRACE_WIDTH, 1)
            data = np.random.default_rng(0).normal(0, 10, (4, 500))
            page.time_view.draw(data, [0, 1, 2, 3], 20.0, 125.0, ["a", "b", "c", "d"])
            widths = lambda: {c.opts['pen'].widthF() for c in page.time_view._channel_curves.values()}
            self.assertEqual(widths(), {1.0})
            self.settings = page.settings_page()
            page.trace_width.setCurrentIndex(page.trace_width.findData(2))
            self.assertEqual(widths(), {2.0})
            self.assertEqual(plots.channel_pen(0).widthF(), 2.0)
            from eeg.ui.preferences import setting
            self.assertEqual(setting('monitor/trace_width'), 2)
        finally:
            user_settings().remove('monitor/trace_width')
            plots.set_trace_width(1)

    def test_samples_do_not_depend_on_the_display_rate(self):
        """The same 2 s of file give the same buffer at 20, 30 and 60 fps."""
        from eeg.core.sources import FileSource, Recording
        from eeg.ui.playback import Player
        t = np.arange(125 * 10) / 125
        data = np.array([np.sin(2 * np.pi * (8 + i) * t) for i in range(4)])
        buffers = []
        for frame_ms in (mon.FRAME_MS, mon.SMOOTH_FRAME_MS, mon.FAST_FRAME_MS):
            page = mon.MonitorPage()
            page.source = FileSource(Recording(data, 125, ["a", "b", "c", "d"]))
            page.fs, page.n_channels = 125.0, 4
            page.panel.set_channels(["a", "b", "c", "d"])
            page.player = Player(page.source, 125)
            page._begin_streaming(); page.timer.stop()
            page.transport.set_playing(True)
            steps = int(round(2000 / frame_ms))
            for _ in range(steps):
                page._tick(2.0 / steps)
            buffers.append(page.raw_buffer.copy())
            page.close()
        self.assertGreater(buffers[0].shape[1], 200)
        for other in buffers[1:]:
            np.testing.assert_array_equal(buffers[0], other)


if __name__ == "__main__":
    unittest.main()
