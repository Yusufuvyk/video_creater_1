"""MATLAB files of any layout: runs in a cell array, events in variables, layouts kept for files alike; a block's
problem shown where it is seen with what to do; nothing wider than a laptop screen.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
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
from pyqtgraph.Qt import QtWidgets  # noqa: E402
from scipy.io import savemat  # noqa: E402

from test_trial_analysis import scratch  # noqa: E402
from eeg.formats import layouts as lo  # noqa: E402
from eeg.formats import matfile as mat  # noqa: E402
from eeg.pipeline import Graph, PipelineError, Runner, compute  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

FS = 250.0
CLASSES = np.array(["left hand", "right hand", "feet", "tongue"], dtype=object)


def run(seconds, trials=(), codes=(), bad=(), channels=25, seed=0):
    """One BNCI-like run: X samples x channels in µV, trial starts counted from 1, codes 1-4."""
    rng = np.random.default_rng(seed)
    return {"X": rng.normal(0, 10, (int(seconds * FS), channels)), "trial": np.asarray(trials, dtype=float),
            "y": np.asarray(codes, dtype=float), "fs": FS, "classes": CLASSES,
            "artifacts": np.asarray(bad, dtype=float), "gender": "f", "age": 22.0}


def bnci_like(path, channels=25):
    """Three runs in a 1x3 cell array: one without trials (eye calibration), then two of four trials each."""
    cell = np.empty((1, 3), dtype=object)
    cell[0, 0] = run(20, channels=channels)
    cell[0, 1] = run(40, [251, 2251, 4251, 6251], [1, 2, 3, 4], [0, 1, 0, 0], channels, seed=1)
    cell[0, 2] = run(40, [251, 2251, 4251, 6251], [4, 3, 2, 1], [0, 0, 0, 0], channels, seed=2)
    savemat(str(path), {"data": cell})
    return path


class CellArrayTests(unittest.TestCase):
    def test_a_cell_array_of_runs_is_listed_run_by_run(self):
        with scratch() as directory:
            variables = mat.read_variables(bnci_like(directory / "A01T.mat"))
        self.assertIn("data{2}.X", variables)
        self.assertEqual(np.asarray(variables["data{2}.X"]).shape, (10000, 25))
        self.assertIn("data{*}.X", mat.patterns(variables))
        self.assertEqual([n for n, _ in mat.runs(variables, "data{*}.X")], [1, 2, 3])
        self.assertEqual(mat.run_key("data{*}.y", 3), "data{3}.y")

    def test_files_read_before_read_as_before(self):
        """A struct array of channel labels stays one entry; a plain array is still the one guessed."""
        with scratch() as directory:
            path = directory / "plain.mat"
            chanlocs = np.empty((1, 3), dtype=object)
            for i, name in enumerate(("C3", "Cz", "C4")):
                chanlocs[0, i] = {"labels": name, "X": float(i)}
            savemat(str(path), {"eeg": np.random.default_rng(0).normal(size=(3, 2000)), "fs": 200.0,
                                "chanlocs": chanlocs})
            variables = mat.read_variables(path)
            self.assertIn("chanlocs", variables)
            self.assertFalse(any("{" in key for key in variables))
            mapping = mat.autodetect(variables)
            self.assertEqual((mapping.data_key, mapping.fs), ("eeg", 200.0))
            recording = compute.read_recording(path)
        self.assertEqual(recording.channel_names, ["C3", "Cz", "C4"])
        self.assertIsNone(recording.events)
        self.assertEqual(recording.bad_spans, [])

    def test_the_built_in_bnci_layout_joins_runs_with_their_events_and_bad_trials(self):
        with scratch() as directory:
            recording = compute.read_recording(bnci_like(directory / "A01T.mat"))
        self.assertEqual(recording.channel_names[:3], ["Fz", "FC3", "FC1"])
        self.assertEqual(recording.n_channels, 22)                        # the 3 EOG columns left out
        self.assertEqual(recording.n_samples, int(80 * FS) + int(FS))     # two runs and a 1 s gap; run 1 left out
        self.assertEqual(recording.events.labels[:4], ["left hand", "right hand", "feet", "tongue"])
        np.testing.assert_allclose(recording.events.onsets[:2], [1.0, 9.0])  # sample 251 counted from 1: 1 s
        self.assertAlmostEqual(recording.events.onsets[4], 41.0 + 1.0)       # the second run after the gap
        self.assertEqual(recording.bad_spans, [(11.0, 15.0, "BAD_artifacts")])   # 2-6 s after trial 2's start
        trials = compute.cut_epochs(recording, tmin=2.0, tmax=6.0)
        self.assertEqual(len(trials.labels), 7)
        self.assertEqual(trials.rejected, [("2", "right hand", "Marked BAD_artifacts in the file")])

    def test_events_in_variables_say_what_is_wrong(self):
        with scratch() as directory:
            variables = mat.read_variables(bnci_like(directory / "A01T.mat"))
        mapping = mat.Mapping("data{*}.X", fs=FS, transpose=True, onsets_key="data{*}.trial",
                              onset_unit=mat.SECONDS, codes_key="data{*}.y")
        with self.assertRaisesRegex(mat.MatError, "run past the signal.*samples"):
            mat.apply_mapping(variables, mapping)
        mapping.onset_unit, mapping.codes_key = mat.SAMPLES_1, "data{*}.fs"     # one value for four events
        with self.assertRaisesRegex(mat.MatError, "each event needs one time and one code"):
            mat.apply_mapping(variables, mapping)

    def test_the_checklist_says_what_is_missing_and_suggests_the_fields_beside_the_signal(self):
        with scratch() as directory:
            variables = mat.read_variables(bnci_like(directory / "A01T.mat"))
        self.assertEqual(mat.suggest(variables, "data{*}.X"),
                         {"onsets_key": "data{*}.trial", "codes_key": "data{*}.y", "labels_key": "data{*}.classes",
                          "bad_key": "data{*}.artifacts", "fs_key": "data{*}.fs"})
        lines = mat.checklist(variables, mat.Mapping("data{*}.X", fs=FS, transpose=True))
        self.assertEqual([s for s, _ in lines], [mat.READY, mat.READY, mat.OPTIONAL])
        self.assertIn("'data{*}.trial' looks like event times", lines[-1][1])
        wrong_way = mat.checklist(variables, mat.Mapping("data{*}.X", fs=FS, transpose=False))
        self.assertEqual(wrong_way[0][0], mat.MISSING)
        self.assertIn("Rows are samples", wrong_way[0][1])
        self.assertEqual(mat.checklist(variables, mat.Mapping("nothing", fs=FS))[0][0], mat.MISSING)


class KeptLayoutTests(unittest.TestCase):
    def test_a_kept_layout_reads_every_file_laid_out_alike_and_no_other(self):
        with scratch() as directory:
            for name, channels in (("one.mat", 8), ("two.mat", 8), ("other.mat", 4)):
                cell = np.empty((1, 2), dtype=object)
                cell[0, 0] = {"sig": np.random.default_rng(0).normal(size=(channels, 3000)), "pos": [100.0, 900.0],
                              "kind": [1.0, 2.0], "rate": 100.0}
                cell[0, 1] = {"sig": np.random.default_rng(1).normal(size=(channels, 3000)), "pos": [500.0],
                              "kind": [2.0], "rate": 100.0}
                savemat(str(directory / name), {"rec": cell})
            with self.assertRaises(mat.MatLayoutUnknown):
                mat.mapping_for(directory / "one.mat", mat.read_variables(directory / "one.mat"), directory)
            mapping = mat.Mapping("rec{*}.sig", fs=100.0, onsets_key="rec{*}.pos", onset_unit=mat.SAMPLES_0,
                                  codes_key="rec{*}.kind")
            lo.save(lo.Layout("Lab export", "test", mapping, channels=8), directory)
            self.assertEqual([layout.name for layout in lo.saved(directory)], ["Lab export"])
            two = mat.read_variables(directory / "two.mat")
            self.assertEqual(mat.mapping_for(directory / "two.mat", two, directory).data_key, "rec{*}.sig")
            recording = mat.apply_mapping(two, mat.mapping_for(directory / "two.mat", two, directory))
            self.assertEqual(recording.events.labels, ["1", "2", "2"])
            np.testing.assert_allclose(recording.events.onsets, [1.0, 9.0, 30.0 + 1.0 + 5.0])
            other = mat.read_variables(directory / "other.mat")
            self.assertIsNone(lo.matching(other, directory))              # 4 channels, not 8
            self.assertIs(lo.matching(mat.read_variables(bnci_like(directory / "A.mat")), directory),
                          lo.BNCI2014_001)


class DialogTests(unittest.TestCase):
    def test_the_dialog_offers_what_lies_beside_the_signal_and_keeps_the_layout(self):
        from eeg.ui.matdialog import MatMappingDialog
        with scratch() as directory:
            variables = mat.read_variables(bnci_like(directory / "A01T.mat"))
            dialog = MatMappingDialog(None, "A01T.mat", "", variables, layouts_root=directory)
            ok = dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
            mapping = dialog.mapping()
            self.assertEqual((mapping.data_key, mapping.onsets_key, mapping.codes_key, mapping.labels_key,
                              mapping.bad_key, mapping.fs, mapping.transpose),
                             ("data{*}.X", "data{*}.trial", "data{*}.y", "data{*}.classes", "data{*}.artifacts",
                              FS, True))
            self.assertTrue(ok.isEnabled())
            self.assertIn("Events: 8 (left hand x2", dialog.checks.text())
            self.assertIn("Channel names: CH1", dialog.checks.text())
            dialog.transpose.setChecked(False)                          # the wrong way round: not readable
            self.assertFalse(ok.isEnabled())
            self.assertIn("Missing", dialog.checks.text())
            dialog.transpose.setChecked(True)
            dialog.keep_name.setText("BNCI-like")
            kept = dialog.keep_layout(dialog.mapping())
            self.assertEqual(kept.channels, 25)
            self.assertEqual([layout.name for layout in lo.saved(directory)], ["BNCI-like"])
            dialog.close()


class ProblemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_epochs import app
        app()

    def test_a_file_that_cannot_be_read_stops_on_its_own_block_with_a_way_out(self):
        from eeg.ui.workbench.page import WorkbenchPage
        from eeg.ui.workbench import problems
        with scratch() as directory:
            path = directory / "odd.mat"
            savemat(str(path), {"note": "nothing numeric here"})
            graph = Graph("t")
            load, epochs = graph.chain(("load", {"path": str(path)}), ("epochs", {}))
            with self.assertRaises(PipelineError) as caught:
                Runner(graph).run()
            self.assertEqual(caught.exception.node, load)              # on the Recording, not only in the status
            self.assertIn(compute.MAT_LAYOUT, str(caught.exception))
            page = WorkbenchPage()
            page.set_graph(graph)
            page.sheet.errors[load] = str(caught.exception)
            labels = [label for label, _, _ in problems.fixes(page, load, str(caught.exception))]
            self.assertEqual(labels[0], "Choose how to read this file…")
            page.canvas.select(load)
            app.processEvents()
            self.assertTrue(page.inspector.error.isVisibleTo(page.inspector))
            self.assertTrue(page.inspector.error.headline.text().startswith("GIOVA cannot tell"))
            page.close()

    def test_no_events_offers_the_remedies_at_epochs_and_points_to_the_recording(self):
        from eeg.ui.workbench.page import WorkbenchPage
        from eeg.ui.workbench import problems
        graph = Graph("t")
        load, epochs = graph.chain(("load", {"path": "rec.csv"}), ("epochs", {}))
        page = WorkbenchPage()
        page.set_graph(graph)
        message = ("Epochs: This recording has no events to cut trials around. For trials repeated without "
                   "markers, set 'Without markers: every' in Options")
        page.sheet.errors[epochs] = message
        self.assertEqual(problems.split(message, "Epochs"),
                         ("This recording has no events to cut trials around.",
                          "For trials repeated without markers, set 'Without markers: every' in Options"))
        fixes = problems.fixes(page, epochs, message)
        self.assertEqual([f[0] for f in fixes], ["Cut a trial every few seconds…", "Type the trials in a table",
                                                 "Show the Recording", "Help on this block"])
        page.canvas.select(epochs)
        app.processEvents()
        fixes[0][2]()                                                  # Options open, 'every' in focus
        self.assertTrue(page.inspector.fields["every"].isVisibleTo(page.inspector))
        fixes[2][2]()
        self.assertEqual(page.canvas.selected(), load)
        page.close()


class WidthTests(unittest.TestCase):
    """Nothing asks for more room than a laptop at high scaling has: text wraps or ends in …"""

    def test_every_block_s_settings_fit_a_narrow_panel(self):
        from eeg.pipeline import REGISTRY
        from eeg.ui.workbench.inspector import Inspector
        inspector = Inspector()
        inspector.resize(320, 800)
        inspector.show()
        graph = Graph("x")
        for name in sorted(REGISTRY):
            node = graph.add(name, {})
            inspector.show_node(graph, node)
            inspector.show_error("A long message that explains in some detail what went wrong. " * 3)
            app.processEvents()
            inspector.layout().activate()
            self.assertLessEqual(inspector.minimumSizeHint().width(), 2 * 320, name)
        inspector.close()

    def test_a_long_message_does_not_widen_the_window(self):
        from eeg.ui.app import MainWindow
        from test_workbench import fresh_sheet
        fresh_sheet(self)
        window = MainWindow()
        workbench = window.open_mode("Workbench")
        app.processEvents()
        before = window.minimumSizeHint().width()
        workbench._say("Stopped at Epochs: " + "a very long explanation of what went wrong " * 10)
        workbench.strip.text.setText("Computing " + "a long step name " * 20)
        app.processEvents()
        self.assertLessEqual(window.minimumSizeHint().width(), before)
        self.assertIn("a very long explanation", workbench.state.text())    # whole, in the tooltip and text()
        for sheet in workbench.sheets:
            sheet.modified = False
        window.close()


DATA = ROOT / "Data" / "BNCI2014_001" / "A01T.mat"


@unittest.skipUnless(DATA.is_file(), "BNCI 2014-001 A01T.mat not in Data")
class RealFileTests(unittest.TestCase):
    def test_a01t_reads_with_its_288_trials(self):
        recording = compute.read_recording(DATA)
        self.assertEqual(recording.n_channels, 22)
        self.assertEqual(len(recording.events), 288)
        self.assertEqual(set(recording.events.counts().values()), {72})
        self.assertEqual(recording.fs, 250.0)


if __name__ == "__main__":
    unittest.main()
