"""The program's Settings, the recording folder/name, and renaming recordings.

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
from eeg.core import sources  # noqa: E402
from eeg.ui import preferences as prefs  # noqa: E402


def release(*widgets):
    """Delete top-level widgets now, not at interpreter exit.

    Left for Python's shutdown, a parentless page was destroyed after Qt's own
    teardown in about half the runs: the process then aborted (0xC0000409)
    after printing OK, and build_exe.bat, which checks the exit code, stopped.
    """
    for widget in widgets:
        widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()


class RecordingNameTests(unittest.TestCase):
    def test_the_prefix_is_used_and_never_overwrites(self):
        with scratch() as directory:
            first = sources.live_record_path(directory, "subject ssvep")
            self.assertTrue(first.name.startswith("subject ssvep_"))
            self.assertTrue(first.name.endswith(".csv"))
            first.write_text("", encoding="utf-8")
            second = sources.live_record_path(directory, "subject ssvep")
            self.assertNotEqual(first.name, second.name)         # same second, new file
            # Anything a file name cannot hold is dropped, and an empty name
            # falls back rather than writing ".csv".
            self.assertTrue(sources.live_record_path(directory, "a/b:c").name.startswith("abc_"))
            self.assertTrue(sources.live_record_path(directory, "///").name.startswith("live_"))

    def test_renaming_takes_the_sidecars_with_it(self):
        with scratch() as directory:
            path = directory / "live_one.csv"
            path.write_text("1\n", encoding="utf-8")
            for suffix in (".hardware.tsv", ".events.json", ".channels.json", "_events.csv"):
                path.with_name(path.stem + suffix).write_text("x", encoding="utf-8")
            other = directory / "unrelated.csv"
            other.write_text("2\n", encoding="utf-8")
            moved = sources.rename_recording(path, "ssvep 15 Hz")
            self.assertEqual(moved.name, "ssvep 15 Hz.csv")
            self.assertFalse(path.exists())
            self.assertEqual(sorted(p.name for p in sources.sidecar_paths(moved)),
                             ["ssvep 15 Hz.channels.json", "ssvep 15 Hz.events.json",
                              "ssvep 15 Hz.hardware.tsv", "ssvep 15 Hz_events.csv"])
            self.assertTrue(other.exists())                      # nothing else was touched

    def test_a_name_already_there_moves_nothing(self):
        with scratch() as directory:
            path = directory / "a.csv"
            path.write_text("1\n", encoding="utf-8")
            path.with_name("a.events.json").write_text("x", encoding="utf-8")
            (directory / "b.events.json").write_text("y", encoding="utf-8")
            with self.assertRaises(ValueError):
                sources.rename_recording(path, "b")
            self.assertTrue(path.exists())                       # the recording stayed put
            self.assertTrue(path.with_name("a.events.json").exists())
            self.assertEqual((directory / "b.events.json").read_text(encoding="utf-8"), "y")


class SettingsDialogTests(unittest.TestCase):
    def setUp(self):
        app()
        self.store = prefs.user_settings()
        # Restored afterwards: these tests must not leave the operator's (or the
        # next test's) thresholds changed.
        self.before = {key: self.store.value(key) for key in list(prefs.DEFAULTS) + ["quality_0"]}

    def tearDown(self):
        for key, value in self.before.items():
            if value is None:
                self.store.remove(key)
            else:
                self.store.setValue(key, value)

    def test_settings_reach_the_pages_that_use_them(self):
        from eeg.ui.app import MainWindow
        window = MainWindow()
        try:
            dialog = window.open_settings("Monitor")
            self.assertEqual([dialog.categories.item(i).text() for i in range(dialog.categories.count())],
                             ["Recording", "Interface", "Graphs", "Monitor", "Files", "OpenBCI boards",
                              "Advanced"])
            self.assertEqual(dialog.categories.currentItem().text(), "Monitor")
            # The thresholds are the Monitor's own widgets, not copies.
            monitor = window.monitor
            self.assertIs(monitor.quality_settings.window(), dialog)
            self.assertIs(monitor.panel.hardware.limits.window(), dialog)
            monitor._last_quality = 0.0
            monitor.quality_controls[0].setValue(0.4)
            self.assertAlmostEqual(monitor.quality_controls[0].value(), 0.4, places=5)
            self.assertEqual(monitor._last_quality, -float("inf"))   # the change is applied at once
            # The arrangement is stored and applied at once.
            monitor.resize(1200, 700)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(monitor._wide)
            monitor.layout_choice.setCurrentIndex(monitor.layout_choice.findData("split"))
            self.assertTrue(monitor._wide)
            self.assertEqual(prefs.setting("monitor/views/layout"), "split")
            monitor.layout_choice.setCurrentIndex(monitor.layout_choice.findData("auto"))
            self.assertFalse(monitor._wide)
        finally:
            window.close()
            QtCore.QThreadPool.globalInstance().waitForDone(3000)
            release(window)

    def test_settings_show_the_pages_of_the_mode_they_are_opened_from(self):
        from eeg.ui.settings_dialog import SettingsDialog
        every = ["Recording", "Interface", "Graphs", "Monitor", "Files", "OpenBCI boards", "Advanced"]
        dialog = SettingsDialog(pages=[(name, QtWidgets.QLabel(name)) for name in every])
        try:
            dialog.open_at(mode="Workbench")
            self.assertEqual(dialog.names(shown_only=True), ["Interface", "Advanced"])
            self.assertEqual(dialog.windowTitle(), "Settings · Workbench")
            self.assertEqual(dialog.categories.currentItem().text(), "Interface")   # a page that is shown
            self.assertTrue(dialog.show_all.isVisibleTo(dialog))
            dialog.show_all.click()
            self.assertEqual(dialog.names(shown_only=True), every)
            self.assertFalse(dialog.show_all.isVisibleTo(dialog))
            dialog.open_at(mode="Games")
            self.assertEqual(dialog.names(shown_only=True),
                             ["Recording", "Interface", "Monitor", "OpenBCI boards", "Advanced"])
            dialog.open_at(mode="Monitor")
            self.assertEqual(dialog.names(shown_only=True), every)
            dialog.open_at(mode="Home")                                  # the main menu: every page
            self.assertEqual(dialog.names(shown_only=True), every)
            self.assertEqual(dialog.windowTitle(), "Settings")
            # a page asked for by name is never out of reach, whatever the mode
            dialog.open_at("OpenBCI boards", mode="Workbench")
            self.assertEqual(dialog.categories.currentItem().text(), "OpenBCI boards")
            self.assertEqual(dialog.names(shown_only=True), every)
        finally:
            dialog.close()

    def test_hover_tips_are_switched_in_settings(self):
        from eeg.ui.app import MainWindow
        window = MainWindow()
        try:
            switch = QtWidgets.QApplication.instance()._tooltip_switch
            window.open_settings("Interface")
            box = window.interface_settings.tooltips
            box.setChecked(False)
            self.assertFalse(switch.enabled)
            self.assertFalse(prefs.setting("ui/tooltips"))
            prefs.user_settings().remove("ui/tooltips")    # what Restore defaults does ...
            window._settings_reset()                        # ... then the pages reload: tips back on
            self.assertTrue(box.isChecked())
            self.assertTrue(switch.enabled)
        finally:
            window.close()
            QtCore.QThreadPool.globalInstance().waitForDone(3000)
            release(window)

    def test_the_recording_folder_and_name_come_from_the_settings(self):
        from eeg.ui.settings_dialog import RecordingSettings
        with scratch() as directory:
            self.store.setValue("recording/folder", str(directory))
            page = RecordingSettings()
            page.prefix.setText("mi train")
            self.assertEqual(prefs.setting("recording/prefix"), "mi train")
            self.assertEqual(prefs.data_dir(), directory)
            self.assertIn("mi train_", page.preview.text())
            page.prefix.setText("")                        # never an empty file name
            self.assertEqual(prefs.recording_prefix(), "live")
            self.store.setValue("recording/folder", str(directory / "gone"))
            self.assertNotEqual(prefs.data_dir(), directory / "gone")   # a folder that is not there
            release(page)


if __name__ == "__main__":
    unittest.main()
