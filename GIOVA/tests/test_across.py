"""Leaving one recording out and trials without markers.

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

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_pipeline import imagery, save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.pipeline import Graph, Runner, batch, block, learn  # noqa: E402
from eeg.pipeline.graph import Context  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class AcrossTests(unittest.TestCase):
    def test_each_recording_is_tested_with_a_model_trained_on_the_others(self):
        with scratch() as directory:
            files = [save_fif(lateralised(trials=30, seed=s), directory, "p{}_raw.fif".format(s)) for s in (1, 2, 3)]
            graph, _, _ = imagery(files[0])
            result = batch.across([graph], files)
        self.assertEqual(result.evaluation, batch.ACROSS)
        self.assertEqual(result.accuracy.shape, (3, 1))
        self.assertTrue((result.accuracy > 0.85).all(), result.accuracy)   # the same effect in everyone
        scores = result.scores[("p1_raw", result.columns[0])]
        self.assertEqual(scores.scheme, learn.ACROSS)
        self.assertEqual(scores.tested, 30)                                  # all of p1, none of it trained on
        self.assertNotIn("±", scores.describe())
        self.assertTrue(scores.better_than_random)

    def test_the_report_says_what_was_run_not_the_cross_validate_settings(self):
        from eeg.ui.workbench.report import comparison_report
        with scratch() as directory:
            files = [save_fif(lateralised(trials=20, seed=s), directory, "p{}_raw.fif".format(s)) for s in (1, 2)]
            graph, _, cv = imagery(files[0])
            graph.set(cv, folds=5, permutations=19)
            report = comparison_report(batch.across([graph], files))
            within = comparison_report(batch.compare([graph], files[:1]))
        methods = " ".join(c for kind, c in report.parts if kind == "p")
        self.assertIn("Leave one recording out", methods)
        self.assertIn("folds and permutation test were not run", methods)
        self.assertNotIn("5-fold", methods)
        self.assertNotIn("19 label shuffles", methods)
        within = " ".join(c for kind, c in within.parts if kind == "p")
        self.assertIn("5-fold", within)
        self.assertIn("19 label shuffles", within)

    def test_leave_one_out_never_trains_on_the_tested_recording(self):
        rng = np.random.default_rng(0)
        # three recordings share a rule, the fourth has it backwards
        groups = []
        for k, sign in enumerate((1, 1, 1, -1)):
            y = np.repeat(["a", "b"], 30)
            x = rng.normal(size=(60, 4))
            x[:, 0] += sign * np.where(y == "a", 2.0, -2.0)
            groups.append(("r{}".format(k), x, y, [str(i) for i in range(60)]))
        scores = learn.leave_one_out(groups, learn.Model().then("lda"))
        self.assertLess(scores[3].accuracy, 0.2)      # it learned the others' rule, which is backwards here
        self.assertGreater(scores[0].accuracy, 0.6)   # two of its three teachers agree with it

    def test_too_many_trials_for_memory_are_refused_with_a_way_out(self):
        with scratch() as directory:
            files = [save_fif(lateralised(trials=10, seed=s), directory, "m{}_raw.fif".format(s)) for s in (1, 2)]
            graph, _, _ = imagery(files[0])
            with self.assertRaisesRegex(ValueError, "Downsample"):
                batch.across([graph], files, limit=1000)

    def test_recordings_of_different_shape_are_refused(self):
        groups = [("a", np.zeros((4, 3, 10)), ["x", "y"] * 2, list("1234")),
                  ("b", np.zeros((4, 2, 10)), ["x", "y"] * 2, list("1234"))]
        with self.assertRaisesRegex(ValueError, "different shapes"):
            learn.leave_one_out(groups, learn.Model().then("csp", n_components=2).then("lda"))

    def test_compare_window_runs_across_recordings(self):
        from eeg.ui.workbench.compare import CompareWindow
        with scratch() as directory:
            files = [save_fif(lateralised(trials=20, seed=s), directory, "w{}_raw.fif".format(s)) for s in (1, 2)]
            graph, _, _ = imagery(files[0])
            window = CompareWindow(graph)
            window.recordings.clear()
            window.add_files([str(f) for f in files])
            window.evaluation.setCurrentIndex(window.evaluation.findData(batch.ACROSS))
            loop = QtCore.QEventLoop()
            window.finished_run.connect(loop.quit)
            window.run()
            loop.exec()
            self.assertEqual(window.result.evaluation, batch.ACROSS)
            window.save_report_as(str(directory / "across.html"))
            text = (directory / "across.html").read_text(encoding="utf-8")
        self.assertIn("trained on all the other recordings", text)


class WithoutMarkersTests(unittest.TestCase):
    def test_trials_repeat_every_period_without_markers(self):
        recording = Recording(np.random.default_rng(1).normal(size=(2, 250 * 30)), 250.0, ["C3", "C4"])
        kind = block("epochs")
        out = kind.run({"raw": recording}, {**kind.defaults(), "tmin": 0.0, "tmax": 7.0, "every": 7.0,
                                            "first": 1.0}, Context())["epochs"]
        self.assertEqual(len(out.labels), 4)          # starts at 1, 8, 15, 22 s; 29 s would run past 30
        self.assertEqual(out.data.shape[-1], 7 * 250)
        self.assertIn("without markers", kind.describe({**kind.defaults(), "every": 7.0, "tmax": 7.0}))
        with self.assertRaisesRegex(ValueError, "Without markers"):
            kind.run({"raw": recording}, kind.defaults(), Context())


if __name__ == "__main__":
    unittest.main()
