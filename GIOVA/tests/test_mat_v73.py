"""MATLAB v7.3 files (HDF5) read as scipy reads v7 ones.

tests/data/matlab holds the same variables saved by MATLAB R2025b as -v7 and -v7.3
(make_fixture.m writes them): numbers of every kind, text with Turkish letters,
char matrices, logicals, empties, complex numbers, 3-D arrays (trials already cut), cells
(nested, 1 x 1, 2 x 3), structs (nested, arrays 1 x 3 and 2 x 2), EEGLAB's EEG.event, a
BNCI-like cell array of runs, and a MATLAB string and table (named, not read). make_eeglab.m
writes an EEGLAB dataset as v7 and v7.3 .set files. Read with scipy (v7) and with GIOVA's h5py reader
(v7.3), every value must match.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import builtins
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from test_trial_analysis import scratch  # noqa: E402
from eeg.formats import matfile as mat  # noqa: E402

DATA = ROOT / "tests" / "data" / "matlab"
V7, V73 = DATA / "fixture_v7.mat", DATA / "fixture_v73.mat"


def differences(a, b, where=""):
    """Where two read values differ: structure, names, text, shapes and numbers (the number type aside: MATLAB
    v7 may keep whole numbers as small integers, v7.3 keeps them double)."""
    marks = [mat._unread(x) or (x if isinstance(x, mat.Unread) else None) for x in (a, b)]
    if any(m is not None for m in marks):
        # a MATLAB object neither reader opens: both must name the same kind
        same = all(m is not None for m in marks) and marks[0].kind == marks[1].kind
        return [] if same else ["{}: {!r} against {!r}".format(where, a, b)]
    if hasattr(a, "_fieldnames") or hasattr(b, "_fieldnames"):
        if not (hasattr(a, "_fieldnames") and hasattr(b, "_fieldnames")):
            return ["{}: a struct against {}".format(where, type(b).__name__)]
        if list(a._fieldnames) != list(b._fieldnames):
            return ["{}: fields {} against {}".format(where, a._fieldnames, b._fieldnames)]
        return [d for f in a._fieldnames for d in differences(getattr(a, f), getattr(b, f), where + "." + f)]
    if isinstance(a, str) or isinstance(b, str):
        return [] if a == b else ["{}: {!r} against {!r}".format(where, a, b)]
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape:
        return ["{}: shape {} against {}".format(where, aa.shape, bb.shape)]
    if aa.dtype == object or bb.dtype == object:
        return [d for i, (x, y) in enumerate(zip(aa.flat, bb.flat)) for d in differences(x, y, "{}[{}]".format(where, i))]
    if aa.dtype.kind in "US" or bb.dtype.kind in "US":
        return [] if aa.tolist() == bb.tolist() else ["{}: {!r} against {!r}".format(where, aa.tolist(), bb.tolist())]
    return [] if np.array_equal(aa.astype(complex), bb.astype(complex)) else ["{}: values differ".format(where)]


class MatlabV73Tests(unittest.TestCase):
    def test_the_header_says_which_reader(self):
        self.assertFalse(mat.is_v73(V7))
        self.assertTrue(mat.is_v73(V73))

    def test_every_variable_reads_as_scipy_reads_the_v7_file(self):
        v7 = {k: v for k, v in loadmat(str(V7), squeeze_me=True, struct_as_record=False).items()
              if not k.startswith("__")}
        v73 = mat.read_v73(V73)
        self.assertEqual(sorted(v7), sorted(v73))
        self.assertEqual(len(v7), 34)
        found = [d for key in sorted(v7) for d in differences(v7[key], v73[key], key)]
        self.assertEqual(found, [])

    def test_the_flattened_list_the_dialog_shows_is_the_same(self):
        flat7, flat73 = mat.read_variables(V7), mat.read_variables(V73)
        self.assertEqual(sorted(flat7), sorted(flat73))
        self.assertIn("data{2}.X", flat73)
        self.assertEqual([d for key in sorted(flat7) for d in differences(flat7[key], flat73[key], key)], [])
        self.assertEqual([r[1:] for r in mat.describe_variables(flat7)], [r[1:] for r in mat.describe_variables(flat73)])

    def test_matlab_order_survives_hdf5_s_reversed_storage(self):
        v = mat.read_v73(V73)
        self.assertEqual(np.asarray(v["eeg"]).shape, (3, 1000))                 # channels x samples, as saved
        self.assertEqual(np.asarray(v["wide"])[0].tolist(), [1.5, 4.5, 7.5, 10.5])
        self.assertEqual(np.asarray(v["nd"]).shape, (2, 3, 4))
        grid = v["grid_struct"]
        self.assertEqual([[grid[i, j].v for j in range(2)] for i in range(2)], [[11, 12], [21, 22]])
        self.assertEqual(grid[1, 0].name, "a21")
        self.assertEqual(np.asarray(v["grid_cell"], dtype=float).tolist(), [[11, 12, 13], [21, 22, 23]])
        self.assertEqual(v["name"], "Subject ölçüm ÇĞİÖŞÜ")
        self.assertEqual(np.asarray(v["charmat"]).tolist(), ["ab", "cd"])
        self.assertEqual(v["s"]._fieldnames, ["X", "fs", "classes", "trial", "y", "sub"])  # MATLAB's order
        self.assertEqual([e.type for e in v["EEG"].event], ["S 1", "S 2"])

    def test_runs_in_a_cell_array_give_the_same_recording(self):
        mapping = mat.Mapping("data{*}.X", fs_key="data{*}.fs", transpose=True, onsets_key="data{*}.trial",
                              codes_key="data{*}.y", labels_key="data{*}.classes", bad_key="data{*}.artifacts")
        a = mat.apply_mapping(mat.read_variables(V7), mapping)
        b = mat.apply_mapping(mat.read_variables(V73), mapping)
        np.testing.assert_array_equal(a.data, b.data)
        self.assertEqual((a.n_channels, a.fs), (4, 100.0))
        self.assertEqual(a.events.labels, b.events.labels)
        np.testing.assert_array_equal(a.events.onsets, b.events.onsets)
        self.assertEqual(a.bad_spans, b.bad_spans)
        self.assertEqual(len(a.events), 4)                                       # run 1 has no trials: left out

    def test_without_h5py_the_message_says_what_to_do(self):
        real = builtins.__import__

        def no_h5py(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("no h5py")
            return real(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=no_h5py):
            with self.assertRaisesRegex(mat.MatVersionUnsupported, r"save\(name, '-v7'\)"):
                mat.read_variables(V73)

    def test_matlab_objects_are_passed_over_not_a_crash(self):
        import h5py
        with scratch() as directory:
            path = directory / "objects.mat"
            with h5py.File(str(path), "w", userblock_size=512) as handle:
                handle.create_dataset("eeg", data=np.arange(20.0).reshape(10, 2)).attrs["MATLAB_class"] = \
                    np.bytes_("double")
                handle.create_dataset("words", data=np.zeros((1, 6), dtype=np.uint32)).attrs["MATLAB_class"] = \
                    np.bytes_("string")
            with open(path, "r+b") as raw:
                raw.write(b"MATLAB 7.3 MAT-file".ljust(128, b" "))
            variables = mat.read_variables(path)
        self.assertEqual(variables["words"].kind, "string")
        self.assertEqual(np.asarray(variables["eeg"]).shape, (2, 10))
        self.assertIn(("words", (), "MATLAB string: not read"), mat.describe_variables(variables))

    def test_a_file_that_is_not_hdf5_says_so(self):
        with scratch() as directory:
            path = directory / "broken.mat"
            path.write_bytes(b"MATLAB 7.3 MAT-file".ljust(600, b"x"))
            with self.assertRaisesRegex(mat.MatError, "Could not read broken.mat as a MATLAB v7.3 file"):
                mat.read_variables(path)


if __name__ == "__main__":
    unittest.main()
