"""Workbench tabs: several sheets, at most two computing at once, Run all, Stop, progress, own windows.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import json
import os
from pathlib import Path
import sys
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_pipeline import imagery, save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from test_workbench import fresh_sheet  # noqa: E402
from eeg.pipeline import Graph, Runner, templates  # noqa: E402
from eeg.pipeline.graph import Stopped  # noqa: E402
from eeg.ui.preferences import user_settings  # noqa: E402
from eeg.ui.theme import color  # noqa: E402
from eeg.ui.workbench import WorkbenchPage  # noqa: E402
from eeg.ui.workbench import sheets as sh  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(condition, seconds=120):
    """Let the window's thread work until `condition()` holds; False if it never did."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def slow(path, permutations=60):
    """CSP + LDA with a permutation test: long enough to be seen running, stopped and waiting."""
    graph, _, cv = imagery(path)
    graph.set(cv, permutations=permutations)
    return graph


class TabTests(unittest.TestCase):
    def setUp(self):
        fresh_sheet(self)
        self.page = WorkbenchPage()
        self.addCleanup(self._close)

    def _close(self):
        for sheet in self.page.sheets:
            sheet.modified = False
        self.page.shutdown()
        wait_until(lambda: not self.page.jobs, 60)
        self.page.close()

    def test_tabs_each_hold_a_sheet_and_come_back_after_a_restart(self):
        page = self.page
        first = page.graph
        page.new_tab("ERD / ERS")
        self.assertEqual(page.tabs.count(), 2)
        self.assertEqual(page.graph.name, "ERD / ERS")            # the page speaks for the tab in front
        self.assertTrue(page.run_all_button.isVisibleTo(page))
        page.tabs.setCurrentIndex(0)
        self.assertIs(page.graph, first)
        self.assertEqual(page.name.text(), first.name)
        page.new_tab(None)
        page.canvas.add_block("load")
        self.assertTrue(page.modified)
        self.assertTrue(page.tabs.tabText(2).startswith("*"))      # unsaved, said on the tab
        page.tabs.setCurrentIndex(1)
        stored = json.loads(user_settings().value("workbench/sheets"))
        self.assertEqual([s["sheet"]["name"] for s in stored], [first.name, "ERD / ERS", "New pipeline"])
        again = WorkbenchPage()
        try:
            self.assertEqual([again._sheet_at(i).graph.name for i in range(again.tabs.count())],
                             [first.name, "ERD / ERS", "New pipeline"])
            self.assertEqual(again.graph.name, "ERD / ERS")          # the tab that was in front
            self.assertTrue(again.sheets[2].modified)
        finally:
            for sheet in again.sheets:
                sheet.modified = False
            again.close()

    def test_closing_a_tab_asks_about_its_changes_and_the_last_one_stays(self):
        from unittest import mock
        page = self.page
        page.new_tab(None)
        page.canvas.add_block("load")
        buttons = QtWidgets.QMessageBox.StandardButton
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=buttons.Cancel) as ask:
            self.assertFalse(page.close_sheet(page.sheet))
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(page.tabs.count(), 2)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=buttons.Discard):
            self.assertTrue(page.close_sheet(page.sheet))
        self.assertEqual(page.tabs.count(), 1)
        self.assertTrue(page.close_sheet(page.sheet))             # the last: an empty sheet instead
        self.assertEqual(page.tabs.count(), 1)
        self.assertEqual(page.graph.name, "New pipeline")

    def test_a_tab_in_its_own_window_and_back(self):
        page = self.page
        first = page.sheet
        self.assertIsNone(page.detach(first))                     # the only tab stays
        second = page.new_tab("ERD / ERS")
        window = page.detach(second)
        self.assertIsNotNone(window)
        self.assertEqual(page.tabs.count(), 1)
        self.assertIs(second.canvas.window(), window)
        page.activate(second)                                     # clicked in its window: the settings follow
        self.assertEqual(page.graph.name, "ERD / ERS")
        stored = json.loads(user_settings().value("workbench/sheets"))
        self.assertEqual(len(stored), 2)                          # still remembered, as a tab
        window.close()
        self.assertEqual(page.tabs.count(), 2)
        self.assertIsNone(second.window)
        self.assertIs(page.tabs.currentWidget(), second.canvas)

    def test_a_pipeline_already_open_in_a_tab_is_shown_not_opened_again(self):
        with scratch() as directory:
            path = directory / "erd.pipeline.json"
            templates.erd().save(path)
            page = self.page
            self.assertTrue(page.open_pipeline(str(path)))
            page.new_tab(None)
            self.assertTrue(page.open_pipeline(str(path)))        # from Home's Continue, say
            self.assertEqual(page.tabs.count(), 2)
            self.assertEqual(page.tabs.currentIndex(), 0)


class RunTests(unittest.TestCase):
    def setUp(self):
        fresh_sheet(self)
        self.page = WorkbenchPage()
        self.addCleanup(self._close)

    def _close(self):
        for sheet in self.page.sheets:
            sheet.modified = False
        self.page.shutdown()
        wait_until(lambda: not self.page.jobs, 60)
        self.page.close()

    def test_run_all_computes_two_at_a_time_and_the_rest_wait(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            page = self.page
            page.set_graph(slow(path, 30))
            for _ in range(2):
                page.new_tab(graph=slow(path, 30))
            self.assertEqual(page.run_all(), 3)
            statuses = [s.status for s in page.sheets]
            self.assertEqual(statuses.count(sh.RUNNING), 2)
            self.assertEqual(statuses.count(sh.QUEUED), 1)
            self.assertLessEqual(len(page.jobs), sh.MAX_RUNNING)
            self.assertIn("queued", page.tabs.tabText(2))
            most = [0]

            def done():
                most[0] = max(most[0], len(page.jobs))
                return all(s.status == sh.DONE for s in page.sheets)
            self.assertTrue(wait_until(done, 300), [s.status for s in page.sheets])
            self.assertLessEqual(most[0], sh.MAX_RUNNING)
            for k in range(3):
                self.assertIn("accuracy", page.tabs.tabText(k))
            # nothing running: the strip stays in its place, still, saying how the run ended
            self.assertTrue(page.strip.isVisibleTo(page))
            self.assertFalse(page.strip.trace.timer.isActive())
            self.assertEqual(page.run_button.accessibleName(), "Run")   # Stop again only while it runs
            self.assertIn("Done in", page.strip.text.text())
            # every run is in the Log, folded under the sheet until the strip is clicked
            log = page.log.toPlainText()
            self.assertEqual(log.count("Run started"), 3)
            self.assertEqual(log.count("Done in"), 3)
            page.show_bottom(False)
            self.assertFalse(page.bottom.isVisibleTo(page))
            page.strip.clicked.emit()
            self.assertTrue(page.bottom.isVisibleTo(page))
            compare = page.open_compare()                       # Compare lives in Results now
            self.assertIs(page.bottom.currentWidget(), page.results)
            self.assertTrue(page.results.isAncestorOf(compare))
            self.assertEqual(len(compare.graphs), 1)             # this sheet, ready to compare
            page.show_bottom(False)

    def test_stop_keeps_what_was_computed_and_run_goes_on_from_there(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            page = self.page
            page.set_graph(slow(path, 400))
            self.assertTrue(page.run())
            sheet = page.sheet
            self.assertTrue(wait_until(lambda: sheet.message.startswith("Permutation"), 120), sheet.message)
            self.assertTrue(page.strip.isVisibleTo(page))
            self.assertIn("Permutation test", page.strip.text.text())
            self.assertGreater(sheet.fraction, 0.5)                  # four of five blocks done
            states = {n: item.state for n, item in page.canvas.cards.items()}
            self.assertEqual(states["cross_validate1"], "running")
            self.assertEqual(states["filter1"], "done")
            self.assertEqual(page.run_button.accessibleName(), "Stop")  # Run turned into Stop, in its place
            self.assertEqual(page.run_button.text(), "")                # an icon alone
            page.run_button.click()
            self.assertEqual(sheet.status, sh.STOPPING)
            self.assertTrue(wait_until(lambda: sheet.status == sh.STOPPED, 60))
            self.assertIn("stopped", page.tabs.tabText(0))
            self.assertIn("Stopped", page.state.text())
            outputs = page.outputs()
            self.assertIn("epochs1", outputs)                       # computed before the stop: kept
            self.assertNotIn("cross_validate1", outputs)
            page.graph.set("cross_validate1", permutations=0)
            page._changed()
            self.assertTrue(page.run())
            self.assertEqual(sheet.nodes, ["cross_validate1"])       # only what is missing is computed
            self.assertTrue(wait_until(lambda: sheet.status == sh.DONE, 120))


class ProgressTests(unittest.TestCase):
    def test_progress_within_a_block_and_the_time_left(self):
        sheet = sh.Sheet.__new__(sh.Sheet)
        sh.Sheet.__init__(sheet, canvas=None)
        sheet.begin(["load1", "filter1", "epochs1", "cross_validate1"])
        sheet.advance("load1", "Recording", "Recording")
        self.assertEqual(sheet.fraction, 0.0)
        sheet.advance("cross_validate1", "Cross-validate", "Cross-validate")
        # the filter and epochs came from the cache another tab filled: done all the same
        self.assertEqual(sheet.finished, {"load1", "filter1", "epochs1"})
        sheet.advance("cross_validate1", "Fold 3/5", "Cross-validate")
        self.assertAlmostEqual(sheet.fraction, (3 + 2 / 5) / 4)       # fold 3 begins: two of five done
        self.assertEqual(sheet.message, "Fold 3/5")
        sheet.phase_started = time.monotonic() - 1.0
        self.assertIsNone(sheet.remaining())                         # too early to judge
        # two folds in 20 s: three more take 30 s, whatever share of the blocks is done
        sheet.phase_started = time.monotonic() - 20.0
        self.assertAlmostEqual(sheet.remaining(), 30.0, delta=1.0)
        self.assertEqual(sheet.later_blocks(), [])
        # permutations after the folds: their own pace from when they begin
        sheet.advance("cross_validate1", "Permutation 11/101", "Cross-validate")
        self.assertIsNone(sheet.remaining())

    def test_quick_blocks_do_not_promise_seconds_when_minutes_are_left(self):
        # Recording, Filter, Epochs, Covariances in a second; then the folds of a Riemann model take minutes
        sheet = sh.Sheet.__new__(sh.Sheet)
        sh.Sheet.__init__(sheet, canvas=None)
        sheet.begin(["load1", "filter1", "epochs1", "cov1", "tangent1", "lr1", "cross_validate1"])
        sheet.started = time.monotonic() - 3.0
        sheet.advance("cross_validate1", "Cross-validate", "Cross-validate")
        self.assertGreater(sheet.fraction, 0.8)                      # six of seven blocks done…
        self.assertIsNone(sheet.remaining())                         # …but nothing is known of the folds yet
        sheet.advance("cross_validate1", "Fold 2/5", "Cross-validate")
        sheet.phase_started = time.monotonic() - 40.0                 # the first fold took 40 s
        self.assertAlmostEqual(sheet.remaining(), 160.0, delta=2.0)   # four more: 2:40, not 0:02
        self.assertIn(sh.clock(sheet.remaining()), ("2:39", "2:40"))
        self.assertEqual(sh.clock(75), "1:15")
        self.assertEqual(sh.clock(3725), "1:02:05")

    def test_the_trace_is_symmetric_and_fades_at_both_ends(self):
        trace = sh.SignalTrace()
        trace.resize(300, trace.height())
        trace.phases = [1.3, 0.4]
        top, bottom = trace.points(1), trace.points(-1)
        middle = trace.height() / 2
        for a, b in zip(top, bottom):
            self.assertAlmostEqual(a.y() - middle, middle - b.y(), places=9)
        self.assertAlmostEqual(top[0].y(), middle, places=9)              # the ends sit on the line
        self.assertAlmostEqual(top[-1].y(), middle, places=6)
        self.assertLess(min(p.y() for p in top), middle - 3)               # and the middle swells

    def test_the_trace_moves_only_while_running_at_the_pace_agreed(self):
        trace = sh.SignalTrace()
        self.assertEqual(trace.STEPS, (0.108, 0.18))                      # 90 % of the draft's pace
        trace.show_progress(0.0, None)
        self.assertFalse(trace.timer.isActive())                          # idle: the faint line, still
        trace.show_progress(0.3, "run")
        self.assertTrue(trace.timer.isActive())
        before = list(trace.phases)
        trace._step()
        self.assertEqual([round(b - a, 6) for a, b in zip(before, trace.phases)], [0.108, 0.18])
        for state in ("done", "stopped", "error", None):
            trace.show_progress(0.4, state)
            self.assertFalse(trace.timer.isActive())

    def test_a_runner_stops_between_blocks_and_keeps_what_it_did(self):
        with scratch() as directory:
            graph, _, cv = imagery(save_fif(lateralised(trials=20), directory))
            cache, seen = {}, []
            runner = Runner(graph, cache)
            with self.assertRaises(Stopped):
                runner.run(progress=lambda node, text: seen.append(node), stop=lambda: "filter1" in seen)
            self.assertEqual(seen[-1], "filter1")                   # stopped before the block after the filter
            self.assertEqual(len(cache), len(set(seen)))             # every block begun was finished and kept
            self.assertLess(len(cache), len(graph.nodes))
            with self.assertRaises(Stopped):
                Runner(graph, {}).run(stop=lambda: True)


class CompareTabsTests(unittest.TestCase):
    def test_compare_takes_the_other_tabs(self):
        from eeg.ui.workbench.compare import CompareWindow
        tabs = [templates.csp_lda("a.mat"), Graph("empty"), templates.riemann("a.mat")]
        window = CompareWindow(None, None, tabs=tabs)
        try:
            self.assertEqual(window.add_open_tabs(), ["CSP + LDA", tabs[2].name])
            self.assertEqual(len(window.graphs), 2)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
