"""MATLAB layouts beyond plain signals, and the program saying what it did on its own.

Trials already cut (3-D arrays), events kept in a struct array (EEG.event), EEGLAB .set files saved as v7.3,
MATLAB objects that are not read; and every automatic step - layout, runs and channels left out, bad marks,
trials left out - in the Recording's notes, the pipeline report and the comparison report. The MATLAB files
in tests/data/matlab were written by MATLAB R2025b (make_fixture.m, make_eeglab.m).

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import importlib.util
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

from test_mat_layouts import bnci_like  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.formats import matfile as mat  # noqa: E402
from eeg.pipeline import Graph, Runner, batch, compute, provenance  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

DATA = ROOT / "tests" / "data" / "matlab"


def trials_mapping(variables, **kwargs):
    return mat.Mapping("epo", fs=100.0, axes=mat.guess_axes(np.asarray(variables["epo"]).shape), trial_start=-0.1,
                       codes_key="epo_y", labels_key="epo_classes", bad_key="epo_bad", **kwargs)


class TrialsAlreadyCutTests(unittest.TestCase):
    """epo is channels x samples x trials (4 x 50 x 6) at 100 Hz, labels 1 2 1 2 1 2, trial 3 marked bad."""

    def test_each_stored_trial_comes_back_exactly(self):
        for name in ("fixture_v7.mat", "fixture_v73.mat"):
            variables = mat.read_variables(DATA / name)
            stored = np.asarray(variables["epo"], dtype=float)
            self.assertEqual(mat.guess_axes(stored.shape), "cst")
            recording = mat.apply_mapping(variables, trials_mapping(variables))
            self.assertEqual(recording.events.labels, ["left", "right"] * 3)
            self.assertEqual(recording.bad_spans, [(1.4, 1.9, "BAD_epo_bad")])
            # From the trial's start to its end, as the checklist says: the stored trials, one per event
            trials = compute.cut_epochs(recording, tmin=-0.1, tmax=0.4)
            self.assertEqual(trials.ids, ["1", "2", "4", "5", "6"])
            self.assertEqual(trials.rejected, [("3", "left", "Marked BAD_epo_bad in the file")])
            np.testing.assert_allclose(trials.data[0], stored[:, :, 0])
            np.testing.assert_allclose(trials.data[2], stored[:, :, 3])
            lines = mat.checklist(variables, trials_mapping(variables))
            self.assertIn("From -0.1 s To 0.4 s", " ".join(t for _, t in lines))
            self.assertTrue(any("6 trials of 4 channels x 50 samples" in t for _, t in lines))

    def test_a_label_list_of_the_wrong_length_is_named(self):
        variables = mat.read_variables(DATA / "fixture_v7.mat")
        wrong = trials_mapping(variables)
        wrong.codes_key = "vec_row"                                          # 5 values for 6 trials
        with self.assertRaisesRegex(mat.MatError, "5 values for 6 trials"):
            mat.apply_mapping(variables, wrong)
        self.assertEqual(mat.checklist(variables, wrong)[2][0], mat.MISSING)


class StructArrayEventTests(unittest.TestCase):
    def test_eeg_event_fields_are_lists_the_dialog_offers(self):
        for name in ("fixture_v7.mat", "fixture_v73.mat"):
            variables = mat.read_variables(DATA / name)
            self.assertEqual(variables["EEG.event.latency"].tolist(), [10.0, 120.0])
            self.assertEqual(variables["EEG.event.type"].tolist(), ["S 1", "S 2"])
            self.assertEqual(variables["grid_struct.v"].tolist(), [11.0, 21.0, 12.0, 22.0])   # MATLAB's [s.v]
            self.assertEqual(mat.suggest(variables, "EEG.data"),
                             {"onsets_key": "EEG.event.latency", "codes_key": "EEG.event.type", "fs_key": "EEG.srate"})
            recording = mat.apply_mapping(variables, mat.Mapping("EEG.data", fs_key="EEG.srate",
                                                                 onsets_key="EEG.event.latency",
                                                                 codes_key="EEG.event.type"))
            self.assertEqual(recording.events.labels, ["S 1", "S 2"])
            np.testing.assert_allclose(recording.events.onsets, [0.09, 1.19])   # latency 10 counted from 1


@unittest.skipUnless(importlib.util.find_spec("pymatreader"), "pymatreader not installed")
class EeglabV73Tests(unittest.TestCase):
    def test_a_v73_set_reads_as_the_v7_one(self):
        old = compute.read_recording(DATA / "eeglab_v7.set")
        new = compute.read_recording(DATA / "eeglab_v73.set")
        np.testing.assert_array_equal(old.data, new.data)
        self.assertEqual(old.channel_names, ["C3", "Cz", "C4", "Pz"])
        self.assertEqual(old.channel_names, new.channel_names)
        self.assertEqual(new.events.labels, ["S 1", "S 2", "S 1"])
        np.testing.assert_allclose(new.events.onsets, [0.4, 2.0, 3.2])
        np.testing.assert_array_equal(old.events.onsets, new.events.onsets)
        self.assertIn("Read with MNE-Python (EEGLAB).", new.reading)

    def test_the_data_file_a_v73_set_names_is_found(self):
        import h5py
        from eeg.formats.recordings import dependencies
        with scratch() as directory:
            path = directory / "split.set"
            with h5py.File(str(path), "w", userblock_size=512) as handle:
                name = np.frombuffer("split_data.fdt".encode("utf-16-le"), dtype="<u2").reshape(-1, 1)
                handle.create_dataset("data", data=name).attrs["MATLAB_class"] = np.bytes_("char")
            with open(path, "r+b") as raw:
                raw.write(b"MATLAB 7.3 MAT-file".ljust(128, b" "))
            self.assertEqual(mat.v73_text(path, ("data", "EEG/data")), "split_data.fdt")
            self.assertIn(directory / "split_data.fdt", dependencies(path))


class UnreadTests(unittest.TestCase):
    def test_matlab_objects_are_named_with_what_to_do(self):
        for name in ("fixture_v7.mat", "fixture_v73.mat"):
            variables = mat.read_variables(DATA / name)
            self.assertEqual({k: v.kind for k, v in mat.unread(variables).items()}, {"words": "string", "tbl": "table"})
            last = mat.checklist(variables, mat.Mapping("eeg", fs=250.0))[-1]
            self.assertEqual(last[0], mat.OPTIONAL)
            self.assertIn("words (MATLAB string)", last[1])
            self.assertIn("cellstr(x)", last[1])
            recording = mat.apply_mapping(variables, mat.Mapping("eeg", fs=250.0))
            self.assertIn("Not read: tbl (table), words (string) (MATLAB objects GIOVA cannot open).",
                          recording.reading)


class DialogTrialsTests(unittest.TestCase):
    def test_a_3d_signal_asks_for_its_axes_and_start(self):
        from eeg.ui.matdialog import MatMappingDialog
        variables = mat.read_variables(DATA / "fixture_v73.mat")
        dialog = MatMappingDialog(None, "fixture_v73.mat", "", variables, layouts_root=str(ROOT / "tests" / ".tmp"))
        dialog.keep.setChecked(False)
        dialog.data_key.setCurrentText("epo")
        self.assertTrue(dialog.form.isRowVisible(dialog.axes))
        self.assertFalse(dialog.form.isRowVisible(dialog.onsets_key))
        self.assertEqual(dialog.axes.currentData(), "cst")
        self.assertIn("channels (4) x samples (50) x trials (6)", dialog.axes.currentText())
        dialog.codes_key.setCurrentText("epo_y")
        dialog.trial_start.setValue(-0.1)
        mapping = dialog.mapping()
        self.assertEqual((mapping.axes, mapping.trial_start, mapping.onsets_key, mapping.codes_key),
                         ("cst", -0.1, None, "epo_y"))
        self.assertIn("6 trials of 4 channels x 50 samples", dialog.checks.text())
        dialog.data_key.setCurrentText("eeg")
        self.assertFalse(dialog.form.isRowVisible(dialog.axes))
        self.assertTrue(dialog.form.isRowVisible(dialog.onsets_key))
        dialog.close()


class ReportTests(unittest.TestCase):
    def test_the_report_says_what_was_done_and_every_trial_left_out(self):
        from eeg.ui.workbench.report import pipeline_report
        with scratch() as directory:
            path = bnci_like(directory / "A01T.mat")
            graph = Graph("t")
            graph.chain(("load", {"path": str(path)}), ("epochs", {"tmin": 2.0, "tmax": 6.0}))
            outputs = Runner(graph).run()
            handled = provenance.summary(graph, outputs)
            parts = pipeline_report(graph, outputs).parts
        self.assertEqual([h for h, _, _ in handled], ["Recording · A01T.mat", "Epochs"])
        self.assertTrue(any("Left out: run(s) 1 of data{*}.X" in line for line in handled[0][1]))
        self.assertIn("1 left out: Marked BAD_artifacts in the file x1.", handled[1][1])
        self.assertEqual(handled[1][2], [("2", "right hand", "Marked BAD_artifacts in the file")])
        kinds = [(k, c) for k, c in parts]
        self.assertIn(("h2", "Data handling"), kinds)
        tables = [c for k, c in parts if k == "table" and c[0] == ("Trial", "Condition", "Reason")]
        self.assertEqual(tables[0][1], [("2", "right hand", "Marked BAD_artifacts in the file")])

    def test_a_comparison_reports_each_recording_s_trials_and_reading(self):
        from test_pipeline import imagery
        from eeg.ui.workbench.report import comparison_report
        with scratch() as directory:
            files = [bnci_like(directory / "A0{}T.mat".format(k)) for k in (1, 2)]
            graph, _, _ = imagery(files[0])
            graph.set(next(n for n in graph.nodes if graph.nodes[n].type == "epochs"), tmin=2.0, tmax=6.0,
                      conditions=[])
            result = batch.compare([graph], files)
            parts = comparison_report(result).parts
        # two trials per condition are too few to cross-validate: the run stops, and what it did is still said
        self.assertIn(("A01T", "test"), result.errors)
        self.assertEqual(result.trials[("A01T", "test")][:2], (7, 1))
        self.assertIn("Signal: runs 2, 3 of data{*}.X read one after another, 1 s of missing samples between them.",
                      result.reading["A01T"])
        tables = {tuple(c[0]): c[1] for k, c in parts if k == "table"}
        self.assertIn(("Recording", "Pipeline", "Trials used", "Left out", "Why"), tables)
        texts = " ".join(c for k, c in parts if isinstance(c, str))
        self.assertIn("Every recording: Read with the layout 'BNCI 2014-001", texts)


if __name__ == "__main__":
    unittest.main()
