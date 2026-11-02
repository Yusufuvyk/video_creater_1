"""The second outside review (24 October 2026), R1-R9: one test each.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtWidgets  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_pipeline import save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.formats.recordings import dependencies, load_recording  # noqa: E402
from eeg.pipeline import REGISTRY, Graph, Runner, api as ee, batch, templates  # noqa: E402
from eeg.pipeline.data import Epochs  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
BUTTONS = QtWidgets.QMessageBox.StandardButton


def answering(button):
    """The unsaved-changes question answered with `button`, and how often it was asked."""
    return mock.patch.object(QtWidgets.QMessageBox, "question", return_value=button)


def band_power_lda(path, name):
    graph = Graph(name)
    *_, features = graph.chain(("load", {"path": str(path)}), ("epochs", {"tmin": 0.0, "tmax": 2.0}),
                               ("psd", {"window": 1.0}), ("band_power", {"log": True}))
    lda, cv = graph.add("lda"), graph.add("cross_validate", {"folds": 4})
    graph.connect(features, "features", cv, "data")
    graph.connect(lda, "model", cv, "model")
    return graph


class PageTest(unittest.TestCase):
    """Pages made here leave no unsaved sheet behind for the next test's page to restore."""

    def setUp(self):
        self.pages = []

    def page(self):
        from eeg.ui.workbench import WorkbenchPage
        page = WorkbenchPage()
        self.pages.append(page)
        return page

    def tearDown(self):
        for page in self.pages:
            page.modified = False
            page._remember()
            page.close()


class R1ComparisonReport(unittest.TestCase):
    def test_the_report_describes_the_pipelines_that_were_run(self):
        from eeg.ui.workbench.compare import CompareWindow
        from eeg.ui.workbench.report import comparison_report
        with scratch() as directory:
            path = save_fif(lateralised(trials=20), directory)
            result = batch.compare([band_power_lda(path, "A: band power")], [path])
            self.assertEqual([g.name for g in result.graphs], ["A: band power"])
            window = CompareWindow(None)
            window.add_pipeline(band_power_lda(path, "A: band power"), "sheet")
            window.result = result
            window.show_result(result)
            window.pipelines.setCurrentRow(0)
            window._remove_pipeline()
            window.add_pipeline(templates.riemann(), "template")
            self.assertIn("still those of the last run (A: band power)", window.state.text())
            report = directory / "comparison.html"
            window.save_report_as(report)
            text = report.read_text(encoding="utf-8")
            self.assertIn("<title>Comparison: A: band power</title>", text)
            self.assertNotIn("tangent space", text)
            window.close()
            # results that do not carry the pipelines behind them are not reported with any others
            result.graphs = []
            with self.assertRaisesRegex(ValueError, "run the comparison again"):
                comparison_report(result)


class R2EeglabDataFile(unittest.TestCase):
    def test_the_data_file_a_set_names_is_watched(self):
        from scipy.io import savemat
        with scratch() as directory:
            header, binary = directory / "renamed.set", directory / "original.fdt"

            def store(value):
                np.full((500, 1), value, dtype="<f4").tofile(binary)
            store(10.0)
            savemat(header, {"EEG": dict(data="original.fdt", nbchan=1, pnts=500, trials=1, srate=100.0, xmin=0.0,
                                         xmax=4.99, chanlocs=np.array([dict(labels="C3")], dtype=object))})
            self.assertIn(binary, dependencies(header))
            self.assertIn(header.with_suffix(".fdt"), dependencies(header))     # MNE's fallback counts too
            graph = Graph("eeglab")
            node = graph.add("load", {"path": str(header)})
            runner = Runner(graph)
            self.assertAlmostEqual(float(runner.run()[node]["raw"].data.mean()), 10.0)
            store(20.0)
            self.assertAlmostEqual(float(runner.run()[node]["raw"].data.mean()), 20.0)


class R3SettingsPanels(unittest.TestCase):
    def test_every_block_opens_its_settings_and_ranges_read_back(self):
        from eeg.ui.workbench.inspector import Inspector, field_kind, from_text, to_text
        for type_name in REGISTRY:
            graph = Graph("panel")
            node = graph.add(type_name)
            panel = Inspector()
            panel.show_node(graph, node)                 # raised for the filter bank before
            panel.close()
        bands = next(p for p in REGISTRY["filter_bank"].params if p.name == "bands")
        self.assertEqual(field_kind(bands), "ranges")
        self.assertEqual(to_text("ranges", [[4.0, 8.0], [8.0, 12.5]]), "4-8, 8-12.5")
        self.assertEqual(from_text("ranges", "4-8, 8 – 12.5"), [[4.0, 8.0], [8.0, 12.5]])
        for bad in ("mu 8-13", "8-4", ""):
            with self.assertRaises(ValueError, msg=bad):
                from_text("ranges", bad)
        graph = Graph("panel")
        node = graph.add("filter_bank")
        panel = Inspector()
        panel.show_node(graph, node)
        panel.fields["bands"].widget.setText("6-10, 10-14")
        panel.fields["bands"]._text_done()
        self.assertEqual(graph.params(node)["bands"], [[6.0, 10.0], [10.0, 14.0]])
        panel.close()


class R4UnsavedOnEveryPath(PageTest):
    def test_continue_asks_before_replacing_an_edited_saved_pipeline(self):
        with scratch() as directory:
            page = self.page()
            graph = Graph("mine")
            node = graph.add("filter", {"low": 1.0})
            path, other = directory / "mine.pipeline.json", directory / "other.pipeline.json"
            graph.save(path)
            Graph("other").save(other)
            page.set_graph(graph, path)
            graph.set(node, low=9.0)
            page._changed()
            with answering(BUTTONS.Cancel) as ask:
                self.assertFalse(page.open_pipeline(other))      # as Home's Continue calls it
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(page.graph.name, "mine")
            self.assertEqual(page.graph.params(node)["low"], 9.0)
            with answering(BUTTONS.Save):
                self.assertTrue(page.open_pipeline(other))
            self.assertEqual(Graph.load(path).params(node)["low"], 9.0)     # saved before it was replaced
            self.assertEqual(page.graph.name, "other")

    def test_a_sheet_never_saved_is_asked_about_too(self):
        with scratch() as directory:
            page = self.page()
            page.set_graph(Graph("draft"))
            page.canvas.add_block("load")
            self.assertTrue(page.modified)
            with answering(BUTTONS.Cancel) as ask:
                page.new_pipeline("ERD / ERS")
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(page.graph.name, "draft")
            target = directory / "draft.pipeline.json"
            with answering(BUTTONS.Save), mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName",
                                                            return_value=(str(target), "")):
                page.new_pipeline("ERD / ERS")
            self.assertTrue(target.is_file())
            self.assertEqual(page.graph.name, "ERD / ERS")
            page.canvas.add_block("load")
            with answering(BUTTONS.Discard):
                page.new_pipeline(None)
            self.assertEqual(page.graph.name, "New pipeline")


class R5ModifiedAfterRestart(PageTest):
    def test_the_unsaved_mark_survives_a_restart(self):
        with scratch() as directory:
            page = self.page()
            graph = Graph("saved")
            node = graph.add("filter", {"low": 1.0})
            path = directory / "saved.pipeline.json"
            graph.save(path)
            page.set_graph(graph, path)
            graph.set(node, low=7.0)
            page._changed()
            again = self.page()                           # the start-up path
            self.assertEqual(again.graph.params(node)["low"], 7.0)
            self.assertTrue(again.modified)
            with answering(BUTTONS.Cancel) as ask:
                self.assertFalse(again._may_discard())
            self.assertEqual(ask.call_count, 1)
            # a sheet that differs from its file is unsaved even if the flag was lost
            from eeg.ui.preferences import user_settings
            user_settings().setValue("workbench/modified", False)
            self.assertTrue(self.page().modified)


class R6BrainVisionSubfolder(unittest.TestCase):
    def test_a_header_naming_files_in_a_subfolder_opens_and_is_watched(self):
        with scratch() as directory:
            data = directory / "data"
            data.mkdir()
            header = directory / "valid.vhdr"
            header.write_text("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nCodepage=UTF-8\n"
                              "DataFile=data/signal.eeg\nMarkerFile=data/signal.vmrk\nDataFormat=BINARY\n"
                              "DataOrientation=MULTIPLEXED\nNumberOfChannels=1\nSamplingInterval=10000\n"
                              "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=C3,,1,µV\n",
                              encoding="utf-8")
            (data / "signal.vmrk").write_text("Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\n"
                                              "Codepage=UTF-8\nDataFile=signal.eeg\n[Marker Infos]\n", encoding="utf-8")
            np.full(500, 5.0, dtype="<f4").tofile(data / "signal.eeg")
            self.assertEqual(load_recording(header).data.shape, (1, 500))
            self.assertIn(data / "signal.eeg", dependencies(header))
            graph = Graph("brainvision")
            node = graph.add("load", {"path": str(header)})
            runner = Runner(graph)
            self.assertAlmostEqual(float(runner.run()[node]["raw"].data.mean()), 5.0)
            np.full(500, 6.0, dtype="<f4").tofile(data / "signal.eeg")
            self.assertAlmostEqual(float(runner.run()[node]["raw"].data.mean()), 6.0)


class R7TrialsWithoutMarkers(unittest.TestCase):
    def test_every_window_that_fits_is_cut(self):
        recording = Recording(np.arange(1000, dtype=float)[None, :], 100, ["C3"])      # 10 s

        def starts(every, length):
            trials = ee.epochs(recording, every=every, tmin=0.0, tmax=length)
            return (trials.data[:, 0, 0] / 100).tolist()
        self.assertEqual(starts(4.0, 2.0), [0, 4, 8])           # shorter than the period; the last ends at 10 s
        self.assertEqual(starts(2.0, 2.0), [0, 2, 4, 6, 8])     # back to back
        self.assertEqual(starts(2.0, 3.0), [0, 2, 4, 6])        # overlapping; one more would end at 11 s
        self.assertEqual(starts(5.0, 5.0), [0, 5])              # ending exactly at the last sample


class R8BaselineInsideTrials(unittest.TestCase):
    def test_a_window_reaching_outside_the_trials_is_refused(self):
        trials = Epochs(np.arange(400, dtype=float).reshape(1, 1, -1), 100, ["C3"], ["a"], 0.0, ["1"])
        for start, end in ((-1.0, 0.5), (3.5, 4.5), (1.0, 1.0)):
            with self.assertRaisesRegex(Exception, "inside the trials", msg=(start, end)):
                ee.baseline(trials, start=start, end=end)
        whole = ee.baseline(trials, start=0.0, end=4.0)
        self.assertAlmostEqual(float(whole.data.mean()), 0.0)
        self.assertAlmostEqual(float(trials.data[0, 0, 0] - ee.baseline(trials, start=0.0, end=0.5).data[0, 0, 0]),
                               24.5)


class R9ZeroThreshold(unittest.TestCase):
    def test_flat_channels_come_back_unchanged_and_finite(self):
        rng = np.random.default_rng(0)
        data = np.zeros((2, 3, 1024))
        data[:, 1] = 7.0                                       # constant
        data[:, 2] = rng.normal(0, 5, (2, 1024))               # a signal beside them
        trials = Epochs(data, 256, ["Z", "K", "S"], ["a", "b"], 0.0, ["1", "2"])
        for mode in ("soft", "hard"):
            out = ee.wavelet_denoise(trials, mode=mode)
            self.assertTrue(np.isfinite(out.data).all(), mode)
            np.testing.assert_allclose(out.data[:, :2], data[:, :2], atol=1e-9, err_msg=mode)
            self.assertFalse(np.allclose(out.data[:, 2], data[:, 2]), mode)     # the signal is still denoised


if __name__ == "__main__":
    unittest.main()
