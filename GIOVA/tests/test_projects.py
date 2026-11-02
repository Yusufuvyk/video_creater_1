"""Projects (folders shown as trees of recordings) and the side panes that fold away.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import contextlib
import os
from pathlib import Path
import shutil
import sys
import unittest
import unittest.mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

from test_downsample_units import BrainVisionTests  # noqa: E402
from test_epochs import app  # noqa: E402
from test_settings import release  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core import projects as pr  # noqa: E402
from eeg.ui import help_content as hc  # noqa: E402
from eeg.ui import preferences as prefs  # noqa: E402
from eeg.ui import project_panel as pp  # noqa: E402
from eeg.ui.collapsible import STRIP, Collapsible  # noqa: E402

KEYS = ("recording/folder", pp.FOLDERS_KEY, "window/panel/Monitor", "window/side/panel",
        "monitor/quality_pane/collapsed", "monitor/spectrum_collapsed", "analysis/controls_pane/collapsed",
        "monitor/views/layout", "tab")


class ProjectNameTests(unittest.TestCase):
    def test_two_projects_of_one_name_are_told_apart_by_the_folder_above(self):
        def at(*parts):             # an absolute path on this system: E:\work\... on Windows, /work/... elsewhere
            return str(Path(Path.cwd().anchor, *parts))
        exe, central = at("work", "eegstudio", "dist", "GIOVA", "Data"), at("work", "eegstudio", "Data")
        twins = (at("a", "study", "Data"), at("b", "study", "Data"))
        coffee = at("x", "Coffee")
        names = pp.project_names([exe, central, coffee, *twins])
        self.assertEqual(names[exe], ("Data", "GIOVA"))
        self.assertEqual(names[central], ("Data", "eegstudio"))
        self.assertEqual(names[coffee], ("Coffee", ""))                         # a name used once: nothing added
        self.assertEqual(names[twins[0]], ("Data", os.path.join("a", "study")))  # same folder above: one more up
        self.assertEqual(names[twins[1]], ("Data", os.path.join("b", "study")))

    def test_from_source_recordings_are_never_kept_inside_the_package(self):
        # Whatever is inside eeg/ is packed into the exe with the code (build_exe.bat): a recording there
        # would travel in the download.
        from eeg.core import config
        package = Path(config.__file__).resolve().parents[1]
        self.assertEqual(config.PROJECT_DIR, ROOT)
        self.assertFalse(config.find_data_dir().resolve().is_relative_to(package))
        self.assertFalse(config.GAMES_DIR.resolve().is_relative_to(package))

    def test_an_exe_in_the_projects_dist_folder_records_into_the_projects_data(self):
        from eeg.core import config
        with scratch_tree() as directory:
            exe = directory / "dist" / "GIOVA" / "GIOVA.exe"
            exe.parent.mkdir(parents=True)
            frozen = dict(frozen=True, executable=str(exe))
            with unittest.mock.patch.multiple(sys, create=True, **frozen):
                self.assertEqual(config.find_data_dir(), exe.parent / "Data")     # no project Data: beside the exe
                (directory / "Data").mkdir()
                self.assertEqual(config.find_data_dir(), directory / "Data")


class Stored(unittest.TestCase):
    """Leaves the settings these tests touch as it found them."""

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


@contextlib.contextmanager
def scratch_tree():
    """A scratch folder that may hold sub-folders (the shared one only clears files)."""
    with scratch() as directory:
        try:
            yield directory
        finally:
            for child in directory.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)


def brainflow_csv(path, rows=40):
    """A tab-separated numeric file, which is what a BrainFlow recording looks like."""
    np.savetxt(path, np.arange(rows * 4, dtype=float).reshape(rows, 4), delimiter="\t")


def project_folder(directory):
    """Coffee study/: two participants, sidecars, a saved analysis, and things that are not recordings."""
    root = directory / "Coffee study"
    for sub in ("P01", "P02", "notes", ".git", "dist"):
        (root / sub).mkdir(parents=True)
    BrainVisionTests().write(root / "P01")                                     # rec.vhdr + .eeg + .vmrk
    (root / "P01" / "rec.analysis.json").write_text("{}", encoding="utf-8")
    brainflow_csv(root / "P02" / "live_one.csv")
    (root / "P02" / "live_one_events.csv").write_text("1.0,open\n", encoding="utf-8")
    (root / "P02" / "live_one.channels.json").write_text("[]", encoding="utf-8")
    (root / "P02" / "features_live.csv").write_text("a\tb\n", encoding="utf-8")
    (root / "P02" / "exported.csv").write_text("trial,value\n1,2\n", encoding="utf-8")      # a table, not a recording
    (root / "notes" / "protocol.txt").write_text("x", encoding="utf-8")
    brainflow_csv(root / ".git" / "hidden.csv")
    brainflow_csv(root / "dist" / "built.csv")
    brainflow_csv(root / "top.csv")
    return root


class ProjectTreeTests(unittest.TestCase):
    def test_a_folder_is_read_as_a_tree_of_recordings(self):
        with scratch_tree() as directory:
            root = project_folder(directory)
            nodes = pr.tree(root)
            # Folders first, then recordings; no notes/, no hidden or build folders, no sidecars.
            self.assertEqual([(n.name, n.kind) for n in nodes],
                             [("P01", pr.FOLDER), ("P02", pr.FOLDER), ("top.csv", pr.RECORDING)])
            first = nodes[0].children
            self.assertEqual([n.name for n in first], ["rec.vhdr"])             # .eeg and .vmrk belong to it
            self.assertEqual([(n.name, n.kind) for n in first[0].children], [("rec.analysis.json", pr.ANALYSIS)])
            self.assertEqual([n.name for n in nodes[1].children], ["live_one.csv"])
            self.assertEqual(pr.count_recordings(nodes), 3)
            self.assertEqual(pr.tree(root / "missing"), [])
            self.assertEqual(pr.tree(root, max_depth=1), [nodes[2]])            # one level: the top file only

    def test_info_is_optional_and_leaves_no_file_when_empty(self):
        with scratch_tree() as directory:
            root = project_folder(directory)
            self.assertEqual(pr.load_info(root), {})
            self.assertEqual(pr.display_name(root), "Coffee study")
            kept = pr.save_info(root, dict(name=" Olfactory pilot ", participant="P01", notes="", other="x"))
            self.assertEqual(kept, dict(name="Olfactory pilot", participant="P01"))
            self.assertEqual(pr.load_info(root), kept)
            self.assertEqual(pr.display_name(root), "Olfactory pilot")
            (root / pr.INFO_FILE).write_text("not json", encoding="utf-8")      # a damaged file is no project-breaker
            self.assertEqual(pr.load_info(root), {})
            pr.save_info(root, dict(name="", participant=" ", notes=""))
            self.assertFalse((root / pr.INFO_FILE).exists())
            self.assertEqual(sorted(p.name for p in root.iterdir() if p.is_file()), ["top.csv"])


class ProjectPanelTests(Stored):
    def test_projects_are_listed_activated_and_removed_without_touching_files(self):
        with scratch_tree() as directory:
            root = project_folder(directory)
            other = directory / "Other"
            other.mkdir()
            self.store.setValue("recording/folder", str(other))
            panel = pp.ProjectPanel()
            try:
                tops = lambda: [("● " if panel.tree.topLevelItem(i).data(0, pp.ACTIVE_ROLE) else "")
                                + panel.tree.topLevelItem(i).text(0) for i in range(panel.tree.topLevelItemCount())]
                self.assertEqual(tops(), ["● Other"])                    # where recordings go now is always listed
                messages, changed = [], []
                panel.status.connect(messages.append)
                panel.active_changed.connect(changed.append)
                panel.add_project(str(root))
                self.assertEqual(tops(), ["● Other", "Coffee study"])
                self.assertEqual(panel.selected_project(), str(root))
                self.assertTrue(panel.use_button.isEnabled())
                self.assertIsNotNone(panel.find(root / "P01" / "rec.vhdr"))
                self.assertIsNone(panel.find(root / "P01" / "rec.eeg"))
                # Making it active is the same setting as Settings → Recording → Folder.
                self.assertTrue(panel.activate(str(root)))
                self.assertEqual(prefs.data_dir(), root)
                self.assertEqual(changed, [str(root)])
                self.assertEqual(tops(), ["Other", "● Coffee study"])
                self.assertIn("Coffee study", panel.active_label.text())
                self.assertFalse(panel.use_button.isEnabled())
                # The active project cannot be taken off the list; another one can, and its files stay.
                panel.remove_project(str(root))
                self.assertEqual(tops(), ["Other", "● Coffee study"])
                panel.remove_project(str(other))
                self.assertEqual(tops(), ["● Coffee study"])
                self.assertTrue(other.is_dir())
                self.assertTrue((root / "P02" / "live_one.csv").is_file())
                self.assertIn("files untouched", messages[-1])
                # A folder that is gone is not recorded into.
                self.assertFalse(panel.activate(None))
            finally:
                release(panel)

    def test_double_clicks_open_recordings_and_saved_analyses(self):
        with scratch_tree() as directory:
            root = project_folder(directory)
            self.store.setValue("recording/folder", str(root))
            panel = pp.ProjectPanel()
            try:
                opened, analyses = [], []
                panel.recording_activated.connect(opened.append)
                panel.analysis_activated.connect(lambda recording, saved: analyses.append((recording, saved)))
                recording = panel.find(root / "P01" / "rec.vhdr")
                panel._double_clicked(recording)
                self.assertEqual(opened, [str(root / "P01" / "rec.vhdr")])
                panel._double_clicked(recording.child(0))
                self.assertEqual(analyses, [(str(root / "P01" / "rec.vhdr"), str(root / "P01" / "rec.analysis.json"))])
                folder = panel.find(root / "P02")
                was = folder.isExpanded()
                panel._double_clicked(folder)                           # a folder only opens or closes
                self.assertEqual((folder.isExpanded(), len(opened)), (not was, 1))
                # Reading the folders again keeps what was open and selected.
                panel.tree.setCurrentItem(panel.find(root / "top.csv"))
                brainflow_csv(root / "P02" / "live_two.csv")
                panel.refresh()
                self.assertIsNotNone(panel.find(root / "P02" / "live_two.csv"))
                self.assertEqual(panel.find(root / "P02").isExpanded(), not was)
                self.assertEqual(panel.tree.currentItem().text(0), "top.csv")
            finally:
                release(panel)

    def test_project_info_is_shown_and_saved_from_the_dialog(self):
        with scratch_tree() as directory:
            root = project_folder(directory)
            self.store.setValue("recording/folder", str(root))
            dialog = pp.ProjectInfoDialog(None, root)
            try:
                self.assertEqual(dialog.name.placeholderText(), "Coffee study")
                dialog.participant.setText("P07")
                dialog.notes.setPlainText("vanilla, 2 s")
                pr.save_info(root, dialog.values())
                panel = pp.ProjectPanel()
                panel.select_project(root)
                self.assertIn("participant: P07", panel.detail.text())
                self.assertIn("Notes: vanilla, 2 s", panel.tree.currentItem().toolTip(0))
                release(panel)
            finally:
                release(dialog)


class CollapsibleTests(Stored):
    def test_a_pane_folds_to_a_strip_and_keeps_its_width_for_later(self):
        from eeg.ui.detach import ResizableSide
        split = ResizableSide("test/side", self.store)
        try:
            pane = Collapsible("Things", QtWidgets.QLabel("content"), "test/pane", self.store)
            split.add_side(pane, "left", 300, minimum=260, maximum=640)
            split.add_main(QtWidgets.QLabel("main"))
            pane.restore()
            split.resize(1000, 400)
            split.show()
            QtWidgets.QApplication.processEvents()
            self.assertEqual((pane.collapsed, pane.width()), (False, 300))
            folded = []
            pane.toggled.connect(folded.append)
            pane.fold.click()
            QtWidgets.QApplication.processEvents()
            self.assertEqual((pane.collapsed, pane.width(), folded), (True, STRIP, [True]))
            self.assertFalse(pane.body.isVisible())
            self.assertTrue(pane.strip.isVisible())
            self.assertGreater(split.widget(1).width(), 900)                 # the main area took the room
            split.save()
            self.assertEqual(int(self.store.value("test/side/left")), 300)   # not the strip's width
            self.assertTrue(self.store.value("test/pane/collapsed", False, type=bool))
            pane.strip.click()
            QtWidgets.QApplication.processEvents()
            self.assertEqual((pane.collapsed, pane.width(), folded), (False, 300, [True, False]))
            self.assertEqual((pane.minimumWidth(), pane.maximumWidth()), (260, 640))
            self.assertFalse(self.store.value("test/pane/collapsed", True, type=bool))
            # Left folded, it comes back folded.
            pane.set_collapsed(True)
            again = Collapsible("Things", QtWidgets.QLabel("content"), "test/pane", self.store)
            again.restore()
            self.assertTrue(again.collapsed)
            release(again)
        finally:
            for key in ("test/side/left", "test/pane/collapsed"):
                self.store.remove(key)
            release(split)


class WindowTests(Stored):
    def window(self):
        from eeg.ui.app import MainWindow
        window = MainWindow()
        window.open_mode("Monitor")         # these look at the Monitor's panes, so it must be in view
        window.resize(1800, 950)
        window.show()
        QtWidgets.QApplication.processEvents()
        return window

    def close(self, window):
        window.close()
        QtCore.QThreadPool.globalInstance().waitForDone(3000)
        release(window)

    def test_everything_is_shown_first_and_each_side_folds(self):
        self.store.setValue("monitor/views/layout", "split")
        window = self.window()
        try:
            monitor = window.monitor
            self.assertTrue(window.centralWidget().isAncestorOf(window.shell))   # beside the rail
            self.assertTrue(monitor._wide)
            self.assertEqual(window.side.current, "Projects")
            self.assertFalse(monitor.quality_pane.collapsed)
            self.assertTrue(window.projects.isVisible())
            self.assertTrue(monitor.quality_table.isVisible())
            self.assertTrue(monitor.multi_spectrum.isVisible())
            signals = monitor.signals_page.width()
            # Fold all three: the signals get the room (the side panel by its icon on the rail).
            window.rail.panels["Projects"].click()
            monitor.quality_pane.fold.click()
            monitor.set_spectrum_collapsed(True)
            QtWidgets.QApplication.processEvents()
            self.assertTrue(window.side.isHidden())
            self.assertEqual(monitor.quality_pane.width(), STRIP)
            self.assertEqual(monitor.wide_slots["Spectrum"].width(), STRIP)
            self.assertFalse(monitor.multi_spectrum.isVisible())
            self.assertTrue(monitor.spectrum_strip.isVisible())
            self.assertGreater(monitor.signals_page.width(), signals + 600)
            self.assertTrue(self.store.value("monitor/spectrum_collapsed", False, type=bool))
            # A click on each strip (and on the rail's icon) brings it back.
            window.rail.panels["Projects"].click()
            monitor.quality_pane.strip.click()
            monitor.spectrum_strip.click()
            QtWidgets.QApplication.processEvents()
            self.assertTrue(monitor.multi_spectrum.isVisible())
            self.assertFalse(monitor.spectrum_strip.isVisible())
            self.assertGreater(monitor.wide_slots["Spectrum"].width(), 200)
            self.assertAlmostEqual(monitor.signals_page.width(), signals, delta=40)
            # As tabs there is no right-hand pane: a folded spectrum is still a whole tab.
            monitor.set_spectrum_collapsed(True)
            self.store.setValue("monitor/views/layout", "tabs")
            monitor._arrange()
            monitor.view_tabs.setCurrentWidget(monitor.multi_spectrum)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(monitor._wide)
            self.assertTrue(monitor.multi_spectrum.isVisible())
        finally:
            self.close(window)

    def test_a_recording_opens_on_the_page_in_view_and_the_active_project_is_named(self):
        with scratch_tree() as directory:
            root = project_folder(directory)
            window = self.window()
            try:
                window.projects.add_project(str(root))
                self.assertTrue(window.projects.activate(str(root)))
                self.assertIn("Coffee study", window.project_label.text())
                self.assertEqual(prefs.data_dir(), root)
                # The Monitor's own file list follows the active project.
                combo = window.monitor.panel.file_combo
                self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["top.csv"])
                recording = str(root / "P01" / "rec.vhdr")
                # From the Workbench: the pipeline in view runs on it.
                window.open_mode("Workbench")
                window.projects.recording_activated.emit(recording)
                from test_workbench import wait_for_run
                wait_for_run(window.workbench)
                load = window.workbench.graph.sources("load")[0]
                self.assertEqual(window.workbench.graph.params(load)["path"], recording)
                self.assertEqual(window.workbench.outputs()[load]["raw"].channel_names, ["Fz", "Cz"])
                for sheet in window.workbench.sheets:
                    sheet.modified = False
                # From Monitor: played.
                window.tabs.setCurrentWidget(window.monitor)
                window.projects.recording_activated.emit(recording)
                self.assertIs(window.tabs.currentWidget(), window.monitor)
                self.assertEqual(window.monitor.panel.selected_file, recording)
                self.assertEqual((window.monitor.fs, window.monitor.n_channels), (500.0, 2))
                self.assertIsNotNone(window.monitor.player)
                # The right-click choice, Quick analysis, asks for a profile and opens a new Workbench tab on the
                # recording, whatever is in view (the dialog is answered with its first profile here).
                window.monitor.stop()
                from unittest import mock
                with mock.patch("eeg.ui.workbench.quick.QuickDialog.exec", return_value=True):
                    window.projects.open_in_analysis.emit(recording)
                self.assertIs(window.tabs.currentWidget(), window.workbench)
                self.assertEqual(window.workbench.graph.params("load1")["path"], recording)
                self.assertTrue(window.workbench.graph.name.startswith("Quick analysis"))
                for sheet in window.workbench.sheets:
                    sheet.modified = False
                # Settings → Recording shows the same folder, and choosing one there moves the mark.
                dialog = window.open_settings("Recording")
                self.assertEqual(window.recording_settings.folder.text(), str(root))
                other = directory / "Elsewhere"
                other.mkdir()
                prefs.set_setting("recording/folder", str(other))
                window.recording_settings.reload()
                self.assertIn("Elsewhere", window.project_label.text())
                top = window.projects.tree.topLevelItem(0)
                self.assertEqual((top.text(0), top.data(0, pp.ACTIVE_ROLE)), ("Elsewhere", True))
                dialog.close()
            finally:
                self.close(window)

    def test_a_live_session_is_not_ended_by_opening_a_file(self):
        window = self.window()
        try:
            monitor = window.monitor
            monitor.source = object()                       # something is streaming
            monitor.panel.mode.setCurrentIndex(1)           # ... live
            said = []
            monitor.status_changed.connect(said.append)
            self.assertFalse(monitor.open_file("anything.csv"))
            self.assertIn("Stop the live stream", said[-1])
            self.assertEqual(monitor.panel.mode.currentIndex(), 1)
            monitor.source = None
        finally:
            self.close(window)


class HelpTests(unittest.TestCase):
    def test_help_names_the_panes_and_buttons_as_the_program_does(self):
        app()
        from eeg.ui.help_dialog import plain_text
        words = " ".join(plain_text(dict(hc.TOPICS)["Projects and side panes"]).split())
        panel = pp.ProjectPanel()
        try:
            for button in (panel.add_button, panel.use_button):
                self.assertIn(button.accessibleName().lower(), words)      # icons: their name on hover
            for quoted in (pr.INFO_FILE, "remove from this list", "project info…", "settings → recording → folder",
                           "all channels · raw signal quality", "rename recording…", "channel names…", "spectrum",
                           "nothing is copied", ".vhdr", "<recording>.analysis.json"):
                self.assertIn(quoted, words)
        finally:
            release(panel)


if __name__ == "__main__":
    unittest.main()
