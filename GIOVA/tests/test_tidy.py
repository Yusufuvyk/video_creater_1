"""Only what applies is shown; the traces are ruled by the second and carry the recording's markers.

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

from test_downsample_units import BrainVisionTests  # noqa: E402
from test_epochs import app  # noqa: E402
from test_projects import scratch_tree  # noqa: E402
from test_settings import release  # noqa: E402
from eeg.core.boards import CYTON_DAISY, UNICORN, BOARDS  # noqa: E402
from eeg.ui import help_content as hc  # noqa: E402
from eeg.ui import preferences as prefs  # noqa: E402

KEYS = ("ui/show_games", "spectrum_options_shown", "tab", "monitor/views/layout", "monitor/trace_sweep")


def help_words(title):
    from eeg.ui.help_dialog import plain_text
    return " ".join(plain_text(dict(hc.TOPICS)[title]).split())


def xs(item):
    """Where a 'pairs' curve has its vertical lines."""
    x = item.getData()[0]
    return [] if x is None else [round(float(v), 4) for v in x[::2]]


class Stored(unittest.TestCase):
    def setUp(self):
        app()
        self.store = prefs.user_settings()
        self.before = {key: self.store.value(key) for key in KEYS}
        for key in KEYS:
            self.store.remove(key)

    def tearDown(self):
        for key, value in self.before.items():
            if value is None:
                self.store.remove(key)
            else:
                self.store.setValue(key, value)


class OnlyWhatAppliesTests(Stored):
    def test_the_source_bar_shows_the_boxes_of_the_chosen_source(self):
        from eeg.ui.panel import ControlPanel
        panel = ControlPanel(125.0)
        try:
            panel.source_bar.show()
            panel.show()
            shown = lambda: [w.isVisibleTo(panel.source_bar) for w in
                             (panel.file_combo, panel.browse_button, panel.port, panel.refresh_port_button)]
            panel.mode.setCurrentIndex(0)                                   # File playback
            self.assertEqual(shown(), [True, True, False, False])
            self.assertFalse(panel.hardware.isVisibleTo(panel))            # a file has no board channels
            self.assertFalse(panel.unicorn_channels.isVisibleTo(panel))
            panel.mode.setCurrentIndex(1 + BOARDS.index(CYTON_DAISY))
            self.assertEqual(shown(), [False, False, True, True])
            self.assertTrue(panel.hardware.isVisibleTo(panel))
            self.assertFalse(panel.unicorn_channels.isVisibleTo(panel))
            panel.mode.setCurrentIndex(1 + BOARDS.index(UNICORN))          # a serial number: no ports to refresh
            self.assertEqual(shown(), [False, False, True, False])
            self.assertFalse(panel.hardware.isVisibleTo(panel))
            self.assertTrue(panel.unicorn_channels.isVisibleTo(panel))
            panel.mode.setCurrentIndex(0)                                   # and back: nothing is lost
            self.assertEqual(shown(), [True, True, False, False])
        finally:
            release(panel.source_bar, panel)

    def test_spectrum_options_sit_behind_a_button_that_names_them(self):
        from eeg.ui.live_views import SpectrumPanel
        panel = SpectrumPanel()
        try:
            panel.show()
            self.assertFalse(panel.options_button.isChecked())
            self.assertTrue(panel.mode.isVisible())                         # always in view
            self.assertTrue(panel.channel_menu_button.isVisible())
            for widget in (panel.arrangement, panel.input, panel.log, panel.minimum, panel.maximum, panel.scale):
                self.assertFalse(widget.isVisible())
            self.assertIn("Grid · Filtered · 1–60 Hz · log", panel.options_button.text())
            panel.input.setCurrentText("Raw")                               # set from elsewhere: the button says so
            panel.maximum.setValue(45)
            panel.log.setChecked(False)
            self.assertIn("Grid · Raw · 1–45 Hz ", panel.options_button.text())
            panel.options_button.click()
            for widget in (panel.arrangement, panel.input, panel.log, panel.minimum, panel.maximum, panel.scale):
                self.assertTrue(widget.isVisible())
            self.assertTrue(prefs.user_settings().value("spectrum_options_shown", False, type=bool))
            again = SpectrumPanel()                                          # left open, it opens again
            self.assertTrue(again.options_button.isChecked())
            release(again)
        finally:
            release(panel)

    def test_board_settings_have_their_own_page_and_the_games_tab_can_be_hidden(self):
        from eeg.ui.app import MainWindow
        window = MainWindow()
        try:
            dialog = window.open_settings("OpenBCI boards")
            self.assertEqual(dialog.categories.currentItem().text(), "OpenBCI boards")
            monitor = window.monitor
            self.assertIs(monitor.panel.hardware.limits.window(), dialog)
            self.assertIs(monitor.panel.hardware.diagnostics.window(), dialog)
            # ... and are no longer among the Monitor's general settings.
            page = dialog.pages.widget([dialog.categories.item(i).text()
                                        for i in range(dialog.categories.count())].index("Monitor"))
            self.assertFalse(page.isAncestorOf(monitor.panel.hardware.limits))
            self.assertTrue(page.isAncestorOf(monitor.quality_settings))
            # Games: shown unless switched off, and never left as the page in view.
            games = window.tabs.indexOf(window.games)
            self.assertTrue(window.tabs.isTabVisible(games))
            window.tabs.setCurrentWidget(window.games)
            window.interface_settings.games.setChecked(False)
            self.assertFalse(window.tabs.isTabVisible(games))
            self.assertIs(window.tabs.currentWidget(), window.home)
            self.assertFalse(window.mode_actions["Games"].isVisible())
            self.assertFalse(window.home.is_shown("Games"))
            self.assertFalse(prefs.setting("ui/show_games"))
            self.store.remove("ui/show_games")                               # what Restore defaults does
            window._settings_reset()
            self.assertTrue(window.tabs.isTabVisible(games))
            self.assertTrue(window.interface_settings.games.isChecked())
        finally:
            window.close()
            QtCore.QThreadPool.globalInstance().waitForDone(3000)
            release(window)

    def test_help_says_where_each_hidden_thing_is(self):
        from eeg.ui.live_views import SpectrumPanel
        from eeg.ui.settings_dialog import InterfaceSettings
        words = help_words("Projects and side panes")
        panel, interface = SpectrumPanel(), InterfaceSettings()
        try:
            self.assertIn(interface.games.text().lower(), words)
            for quoted in ("settings → openbci boards", "file playback", "refresh ports", "board channels",
                           "press options", "nothing is removed"):
                self.assertIn(quoted, words)
            self.assertTrue(panel.options_button.text().startswith("Options"))
            self.assertIn("settings → openbci boards → impedance colours", help_words("Impedance check"))
        finally:
            release(panel, interface)


class RulingTests(Stored):
    FS = 250.0

    def view(self):
        from eeg.ui.plots import TimeSeriesView
        return TimeSeriesView()

    def test_a_line_every_second_stays_on_its_second(self):
        view = self.view()
        try:
            data = np.zeros((2, 1000))                                       # 4 s
            view.draw(data, [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=1000, window=1000)
            self.assertEqual(xs(view._grid["major"]), [0.0, 1.0, 2.0, 3.0, 4.0])
            minor = xs(view._grid["minor"])                                  # 4 s is short enough for 0.2 s lines
            self.assertEqual((len(minor), minor[1], minor[-1]), (21, 0.2, 4.0))
            x, y = view._grid["zero"].getData()                              # one zero line per row, full width
            self.assertEqual((list(x), list(y)), ([0.0, 4.0, 0.0, 4.0], [2.0, 2.0, 0.0, 0.0]))
            _, y = view._grid["major"].getData()
            self.assertEqual((y.min(), y.max()), (-2.0, 4.0))               # across every row
            # 0.3 s later the signal has moved left by 0.3 s, and so have the lines.
            view.draw(data, [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=1075, window=1000)
            self.assertEqual(xs(view._grid["major"]), [0.7, 1.7, 2.7, 3.7])
            # A sweep is a fixed ruler: the lines do not move at all.
            view.sweep = True
            for position in (1075, 1300):
                view.draw(data, [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=position, window=1000)
                self.assertEqual(xs(view._grid["major"]), [0.0, 1.0, 2.0, 3.0, 4.0])
            # A long window keeps the seconds and drops the 0.2 s lines.
            view.sweep = False
            view.draw(np.zeros((2, 5000)), [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=5000, window=5000)
            self.assertEqual((len(xs(view._grid["major"])), xs(view._grid["minor"])), (21, []))
            # Nothing to draw: no ruling left behind.
            view.draw(data, [], 50.0, self.FS, ["Fz", "Cz"])
            self.assertEqual(xs(view._grid["major"]), [])
        finally:
            release(view)

    def test_markers_are_drawn_where_their_samples_are(self):
        from eeg.ui.plots import marker_text
        self.assertEqual([marker_text(t) for t in ("Primary/P  1", "Stimulus/S 12", "769", "eyes  open", "New Segment/")],
                         ["P 1", "S 12", "769", "eyes open", "New Segment/"])
        view = self.view()
        try:
            view.set_markers([9.0, 1.0, float("nan"), 5.5], ["Primary/P  2", "Primary/P  1", "x", "Stimulus/S  7"])
            self.assertEqual((list(view._marker_times), view._marker_names), ([1.0, 5.5, 9.0], ["P 1", "S 7", "P 2"]))
            data = np.zeros((2, 1000))                                       # the window shows 3.0 to 7.0 s
            view.draw(data, [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=1750, window=1000)
            self.assertEqual(xs(view._marker_lines), [2.5])                 # 5.5 s, 2.5 s after the left edge
            shown = [(label.toPlainText(), round(label.pos().x(), 3)) for label in view._marker_labels
                     if label.isVisible()]
            self.assertEqual(shown, [("S 7", 2.5)])
            # Sweeping, sample k sits at k mod window: 5.5 s -> 1.5 s.
            view.sweep = True
            view.draw(data, [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=1750, window=1000)
            self.assertEqual(xs(view._marker_lines), [1.5])
            # Further on, the marker has left the window and its label goes with it.
            view.draw(data, [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=2600, window=1000)
            self.assertEqual(xs(view._marker_lines), [1.0])                 # 9.0 s -> 1.0 s
            self.assertEqual([l.toPlainText() for l in view._marker_labels if l.isVisible()], ["P 2"])
            view.set_markers()
            view.draw(data, [0, 1], 50.0, self.FS, ["Fz", "Cz"], position=2600, window=1000)
            self.assertEqual(xs(view._marker_lines), [])
            self.assertFalse(any(l.isVisible() for l in view._marker_labels))
        finally:
            release(view)

    def test_playback_marks_the_recordings_events_and_help_describes_both(self):
        from eeg.ui.monitor import MonitorPage
        with scratch_tree() as directory:
            BrainVisionTests().write(directory)                              # markers at 1.0 and 3.0 s
            page = MonitorPage()
            try:
                self.assertTrue(page.open_file(str(directory / "rec.vhdr")))
                view = page.time_view
                marks = [(round(float(t), 3), name) for t, name in zip(view._marker_times, view._marker_names)]
                self.assertIn((1.0, "P 1"), marks)
                self.assertIn((3.0, "P 1"), marks)
                page.stop()
                words = help_words("Parameters explained")
                for quoted in ("time rulings always on", "0.2 s", "{:g} s or shorter".format(view.MINOR_GRID_SECONDS),
                               "event markers file playback", "p 1 for primary/p 1"):
                    self.assertIn(quoted, words)
            finally:
                page.close()
                QtCore.QThreadPool.globalInstance().waitForDone(3000)
                release(page)


if __name__ == "__main__":
    unittest.main()
