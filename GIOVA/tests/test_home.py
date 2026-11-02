"""Home, the mode menu, starting in the last mode, and pages built only when first opened.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
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

from pyqtgraph.Qt import QtWidgets  # noqa: E402

from eeg.ui import preferences as prefs  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.store = prefs.user_settings()
        # the Workbench's remembered sheet too: an unsaved one left by another test would be asked about
        self.saved = {k: self.store.value(k) for k in ("ui/last_mode", "ui/start_last", "ui/show_games",
                                                       "workbench/sheet", "workbench/file", "workbench/modified",
                                                       "workbench/sheets", "workbench/current")}
        for key in self.saved:
            self.store.remove(key)

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                self.store.remove(key)
            else:
                self.store.setValue(key, value)

    def window(self):
        from eeg.ui.app import MainWindow
        window = MainWindow()
        self.addCleanup(window.close)
        return window

    def test_first_start_is_home_and_builds_no_mode(self):
        window = self.window()
        self.assertIs(window.tabs.currentWidget(), window.home)
        self.assertEqual(window._pages, {})
        self.assertTrue(window.rail.modes["Home"].isChecked())
        self.assertTrue(window.side.isHidden())          # Home is the choice alone: no side panel
        self.assertFalse(any(b.isVisibleTo(window.rail) for b in window.rail.panels.values()))

    def test_a_card_opens_its_mode_and_the_next_start_goes_straight_there(self):
        window = self.window()
        window.home.choose("Workbench")
        self.assertIs(window.tabs.currentWidget(), window.workbench)
        self.assertEqual(sorted(window._pages), ["Workbench"])     # nothing else was loaded
        self.assertTrue(window.rail.modes["Workbench"].isChecked())
        again = self.window()
        self.assertIs(again.tabs.currentWidget(), again.workbench)
        self.assertNotIn("Monitor", again._pages)

    def test_a_click_on_a_mode_card_opens_it(self):
        from pyqtgraph.Qt import QtCore
        from PySide6.QtTest import QTest
        window = self.window()
        window.resize(1400, 900)
        window.show()
        QtWidgets.QApplication.processEvents()
        card = window.home.cards["Workbench"]
        QTest.mouseClick(card, QtCore.Qt.MouseButton.LeftButton, pos=card.rect().center())
        self.assertEqual(window.current_mode(), "Workbench")
        # Quick analysis is a card too: the task on the Workbench, asking for a recording and a profile
        from unittest import mock
        window.open_mode("Home")
        card = window.home.cards["Quick"]
        with mock.patch("eeg.ui.workbench.quick.QuickDialog.exec", return_value=False) as asked:
            QTest.mouseClick(card, QtCore.Qt.MouseButton.LeftButton, pos=card.rect().center())
        self.assertTrue(asked.called)
        self.assertEqual(window.current_mode(), "Workbench")

    def test_continue_lists_what_was_used_last(self):
        from eeg.ui import recent
        from test_trial_analysis import scratch
        saved = {k: self.store.value("recent/" + k) for k in (recent.RECORDINGS, recent.PIPELINES, recent.REPORTS)}
        self.addCleanup(lambda: [self.store.setValue("recent/" + k, v) if v is not None else
                                 self.store.remove("recent/" + k) for k, v in saved.items()])
        recent.forget()
        with scratch() as folder:
            pipeline = folder / "mine.pipeline.json"
            pipeline.write_text("{}", encoding="utf-8")
            recent.remember(recent.PIPELINES, pipeline, name="Mine")
            recent.remember(recent.PIPELINES, pipeline, accuracy=0.8125)        # a run adds its accuracy
            recent.remember(recent.RECORDINGS, folder / "gone.csv")            # never there: not listed
            window = self.window()
            window.home.refresh()
            rows = [w for w in window.home.findChildren(QtWidgets.QPushButton) if w.objectName() == "recentRow"]
            self.assertEqual(len(rows), 1)
            self.assertIn("Mine", rows[0].text())
            self.assertIn("81.2%", rows[0].text())
            self.assertEqual(recent.entries(recent.RECORDINGS), [])
            for k in range(8):
                other = folder / "p{}.pipeline.json".format(k)
                other.write_text("{}", encoding="utf-8")
                recent.remember(recent.PIPELINES, other)
            self.assertEqual(len(recent.entries(recent.PIPELINES)), recent.KEEP)
            self.assertEqual(Path(recent.entries(recent.PIPELINES)[0]["path"]).name, "p7.pipeline.json")

    def test_tasks_open_their_mode_set_up(self):
        window = self.window()
        window.run_task("classify")
        self.assertEqual(window.current_mode(), "Workbench")
        self.assertEqual(window.workbench.graph.name, "CSP + LDA")
        compare = window.run_task("literature")
        names = [g.name for g, _ in compare.graphs]
        self.assertEqual(len(names), 4)
        self.assertTrue(any("FBCSP" in n for n in names) and any("MDRM" in n for n in names))
        compare.close()
        window.run_task("build")
        self.assertEqual(len(window.workbench.graph.nodes), 0)
        window.run_task("record")
        self.assertEqual(window.current_mode(), "Monitor")

    def test_switched_off_the_next_start_shows_home(self):
        window = self.window()
        window.open_mode("Workbench")
        window.home.remember.setChecked(False)
        again = self.window()
        self.assertIs(again.tabs.currentWidget(), again.home)

    def test_every_mode_is_one_shortcut_away(self):
        window = self.window()
        from unittest import mock
        for name, keys, mode in (("Monitor", "Ctrl+1", "Monitor"), ("Workbench", "Ctrl+2", "Workbench"),
                                 ("Quick", "Ctrl+3", "Workbench"), ("Games", "Ctrl+4", "Games")):
            action = window.mode_actions[name]
            self.assertEqual(action.shortcut().toString(), keys)
            with mock.patch("eeg.ui.workbench.quick.QuickDialog.exec", return_value=False):
                action.trigger()
            self.assertEqual(window.current_mode(), mode)
        self.assertIs(window.games.monitor, window.monitor)       # Games built on the one Monitor

    def test_the_analysis_mode_of_an_earlier_version_opens_the_workbench(self):
        self.store.setValue("ui/last_mode", "Analysis")
        window = self.window()
        self.assertEqual(window.current_mode(), "Workbench")
        self.assertNotIn("Analysis", [window.tabs.tabText(i) for i in range(window.tabs.count())])
        import sys
        self.assertNotIn("eeg.ui.analysis_page", sys.modules)

    def test_help_follows_the_mode_in_view(self):
        from eeg.ui.help_content import TAB_TOPICS
        window = self.window()
        self.assertEqual(window.open_help().key(), TAB_TOPICS["Home"])
        window.open_mode("Workbench")
        self.assertEqual(window.open_help().key(), "Workbench")
        window.help_dialog.close()


if __name__ == "__main__":
    unittest.main()
