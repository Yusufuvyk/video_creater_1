"""Python export, reports, the Compare window and opening the Analysis page's set-up on the Workbench.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import subprocess
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
from test_workbench import fresh_sheet, wait_for_run  # noqa: E402
from eeg.core import analysis_profiles as ap  # noqa: E402
from eeg.pipeline import Graph, PipelineError, Runner, batch, templates  # noqa: E402
from eeg.pipeline import api as ee  # noqa: E402
from eeg.pipeline.convert import from_profile  # noqa: E402
from eeg.pipeline.export import to_python  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class ApiTests(unittest.TestCase):
    def test_blocks_are_functions_with_their_settings_documented(self):
        self.assertIn("low = 1.0", ee.filter.__doc__)
        with self.assertRaisesRegex(TypeError, "no setting"):
            ee.filter(None, cutoff=3)
        with self.assertRaisesRegex(TypeError, "needs its input"):
            ee.filter()
        with self.assertRaises(AttributeError):
            ee.no_such_block

    def test_a_script_of_calls_gives_the_sheets_numbers(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            raw = ee.filter(ee.load(path=str(path)), low=8.0, high=30.0)
            trials = ee.epochs(raw, tmin=2.0, tmax=5.0)
            scores = ee.cross_validate(trials, ee.lda(ee.csp(n_components=2)), folds=4)
            graph, _, cv = imagery(path)
            sheet = Runner(graph).run()[cv]["scores"]
        self.assertEqual(scores.predictions, sheet.predictions)
        with self.assertRaises(PipelineError):
            ee.epochs(raw, conditions=["up"])


class ExportTests(unittest.TestCase):
    def test_the_exported_script_runs_and_prints_the_same_accuracy(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            graph, _, cv = imagery(path)
            graph.nodes["filter1"].label = "Mu and beta"
            script = to_python(graph)
            self.assertIn('filter_recording(load1, low=8.0, high=30.0)', script)
            self.assertIn("(Mu and beta)", script)
            self.assertLess(script.index("filter1 ="), script.index("lda1 ="))      # signal first, then model
            self.assertLess(script.index("lda1 ="), script.index("cross_validate1 ="))
            target = directory / "pipeline.py"
            target.write_text(script, encoding="utf-8")
            done = subprocess.run([sys.executable, str(target)], capture_output=True, text=True, timeout=300,
                                  encoding="utf-8", errors="replace", cwd=str(directory),
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
            expected = Runner(graph).run()[cv]["scores"].describe()
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn(expected, done.stdout)


class ReportTests(unittest.TestCase):
    def test_pipeline_report_as_html_and_pdf(self):
        from eeg.ui.workbench import WorkbenchPage
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            page = WorkbenchPage()
            page.set_graph(templates.riemann(str(path)))
            self.assertIsNone(page.save_report(str(directory / "early.html")))     # nothing run yet
            page.run()
            wait_for_run(page)
            html_path = page.save_report(str(directory / "report.html"))
            pdf_path = page.save_report(str(directory / "report.pdf"))
            text = Path(html_path).read_text(encoding="utf-8")
            pdf = Path(pdf_path).read_bytes()
        self.assertIn("<h2>Methods</h2>", text)
        self.assertIn("tangent space", text)
        self.assertIn("Barachant", text)                       # references follow the blocks used
        self.assertNotIn("Ramoser", text)
        self.assertIn("Tuning chose", text)
        self.assertGreaterEqual(text.count("data:image/png;base64,"), 5)
        self.assertNotIn("Save predictions", text)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 50000)                    # the figures went in

    def test_report_figures_drop_their_controls(self):
        from eeg.pipeline.data import Epochs
        from eeg.ui.workbench.previews import preview
        epochs = Epochs(np.zeros((4, 3, 50)), 100.0, ["Fz", "C3", "Pz"], ["a", "b", "a", "b"], 0.0,
                        ["1", "2", "3", "4"])
        widget = preview(epochs).for_report()
        self.assertTrue(all(w.isHidden() for w in widget.controls))
        captions = [l.text() for l in widget.findChildren(QtWidgets.QLabel)]
        self.assertIn("Channel C3", captions)                 # opened on C3, said in words


class RocTests(unittest.TestCase):
    def test_the_roc_curve_steps_through_ties_and_its_area_is_the_auc(self):
        from eeg.pipeline.learn import roc_auc
        from eeg.ui.workbench.previews import roc_points
        positive = [True, False, True, True, False, False, True, False]
        scores = [0.9, 0.8, 0.8, 0.6, 0.5, 0.3, 0.3, 0.1]           # two ties, across the conditions
        fpr, tpr = roc_points(positive, scores)
        self.assertEqual((fpr[0], tpr[0], fpr[-1], tpr[-1]), (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(len(fpr), 7)                                # one point per distinct score, and the origin
        self.assertAlmostEqual(float(np.trapezoid(tpr, fpr)), roc_auc(positive, scores), places=12)

    def test_cross_validation_results_draw_their_roc_curve(self):
        import pyqtgraph as pg
        from eeg.ui.workbench.previews import preview
        with scratch() as directory:
            graph, _, cv = imagery(save_fif(lateralised(trials=30), directory))
            scores = Runner(graph).run()[cv]["scores"]
        widget = preview(scores)
        roc = [p for p in widget.findChildren(pg.PlotWidget) if "True positive" in p.getAxis("left").labelText]
        self.assertEqual(len(roc), 1)
        notes = " ".join(l.text() for l in widget.findChildren(QtWidgets.QLabel))
        self.assertIn("ROC: every tested trial", notes)


class CompareTests(unittest.TestCase):
    def test_compare_window_runs_pipelines_over_recordings(self):
        from eeg.ui.workbench.compare import CompareWindow
        with scratch() as directory:
            files = [save_fif(lateralised(trials=30, seed=s), directory, "s{}_raw.fif".format(s)) for s in (1, 2)]
            graph, _, _ = imagery(files[0])
            graph.name = "CSP"
            window = CompareWindow(graph)
            window.recordings.clear()                  # the list it remembers from earlier runs
            other = templates.riemann()
            other.set("filter1", low=8.0, high=30.0)
            self.assertTrue(window.add_pipeline(other, "template"))
            self.assertFalse(window.add_pipeline(Graph("no cv"), "x"))
            window.add_files([str(f) for f in files])
            loop = QtCore.QEventLoop()
            window.finished_run.connect(loop.quit)
            window.run()
            loop.exec()
            result = window.result
            self.assertEqual(result.accuracy.shape, (2, 2))
            self.assertTrue(np.isfinite(result.accuracy).all(), result.errors)
            # one recording's full result: its name double-clicked, every pipeline; a cell, that pipeline
            from eeg.ui.workbench.previews import ScoresPreview
            both = window.open_recording(1, 0)
            self.assertEqual(sorted(both.previews), sorted(result.column_name(c) for c in result.columns))
            self.assertTrue(all(isinstance(w, ScoresPreview) for w in both.previews.values()))
            one = window.open_recording(1, 2)
            only = result.column_name(result.columns[1])
            self.assertEqual(list(one.previews), [only])
            self.assertIs(one.previews[only].value, result.scores[(result.subjects[1], result.columns[1])])
            both.close()
            one.close()
            window.save_table(str(directory / "c.csv"))
            window.save_report_as(str(directory / "c.html"))
            text = (directory / "c.html").read_text(encoding="utf-8")
        self.assertIn("Per recording", text)
        self.assertIn("Riemann + logistic regression", text)

    def test_stop_keeps_what_is_done(self):
        with scratch() as directory:
            files = [save_fif(lateralised(trials=20, seed=s), directory, "t{}_raw.fif".format(s)) for s in (1, 2, 3)]
            graph, _, _ = imagery(files[0])
            asked = []
            result = batch.compare([graph], files, stop=lambda: asked.append(1) or len(asked) > 1)
        self.assertEqual(result.subjects, ["t1_raw"])
        self.assertEqual(result.accuracy.shape, (1, 1))


class FromAnalysisTests(unittest.TestCase):
    def test_cho2017_profile_becomes_the_same_steps(self):
        cho = next(p for p in ap.builtin_profiles() if "Cho2017" in p.name)
        graph, notes = from_profile(cho, "s01.mat")
        self.assertEqual(graph.params("filter1")["low"], 1.0)
        # the page's notch list exactly, no harmonics added: nothing to warn about any more
        self.assertEqual(graph.params("filter1")["notches"], [60.0])
        self.assertEqual(graph.params("filter1")["notch"], 0.0)
        self.assertEqual(graph.params("channels1")["channels"], ["C3", "Cz", "C4"])
        self.assertEqual((graph.params("epochs1")["tmin"], graph.params("epochs1")["tmax"]), (0.0, 7.0))
        self.assertEqual(graph.params("band_power1")["bands"], [["mu", 8.0, 13.0], ["beta", 13.0, 30.0]])
        self.assertEqual(graph.params("erd1")["baseline"], [0.5, 1.5])
        self.assertEqual(graph.params("erd1")["erd_mode"], "pooled")
        self.assertEqual(graph.problems(), [])
        self.assertFalse(any("harmonics" in n for n in notes))

    def test_a_typed_sequence_is_reported_not_hidden(self):
        blank = next(p for p in ap.builtin_profiles() if p.name == "Blank")
        graph, notes = from_profile(blank, None)
        self.assertTrue(any("by hand" in n for n in notes))
        self.assertNotIn("erd1", graph.nodes)


if __name__ == "__main__":
    unittest.main()
