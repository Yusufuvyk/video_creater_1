"""Re-reference by a nearest-neighbour Laplacian: each channel minus the mean of its four nearest electrodes."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from eeg.core.sources import Recording
from eeg.pipeline import api as ee
from eeg.pipeline import compute

NAMES = ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4"]


def recording(data):
    return Recording(data=np.asarray(data, dtype=float), fs=250.0, channel_names=list(NAMES))


class Laplacian(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.data = rng.standard_normal((len(NAMES), 1000))

    def test_c3_is_itself_minus_its_four_neighbours(self):
        out = ee.reference(recording(self.data), mode="laplacian").data
        rows = [NAMES.index(n) for n in ("FC3", "C5", "C1", "CP3")]
        np.testing.assert_allclose(out[NAMES.index("C3")], self.data[NAMES.index("C3")] - self.data[rows].mean(0))
        neighbours = compute.laplacian_neighbours(NAMES)
        self.assertEqual(sorted(NAMES[i] for i in neighbours[NAMES.index("C3")]), ["C1", "C5", "CP3", "FC3"])
        self.assertEqual(sorted(NAMES[i] for i in neighbours[NAMES.index("Cz")]), ["C1", "C2", "CPz", "FCz"])

    def test_a_signal_common_to_every_channel_is_removed(self):
        common = np.sin(np.linspace(0, 40, 1000))
        out = ee.reference(recording(self.data + common), mode="laplacian").data
        np.testing.assert_allclose(out, ee.reference(recording(self.data), mode="laplacian").data, atol=1e-12)

    def test_channels_keep_their_names_order_and_rate(self):
        raw = recording(self.data)
        out = ee.reference(raw, mode="laplacian")
        self.assertEqual(out.channel_names, raw.channel_names)
        self.assertEqual(out.fs, raw.fs)
        self.assertEqual(out.data.shape, raw.data.shape)
        np.testing.assert_array_equal(raw.data, self.data)      # the input is left as it was

    def test_a_missing_sample_of_one_neighbour_leaves_the_others(self):
        data = self.data.copy()
        data[NAMES.index("C5"), 10] = np.nan
        out = ee.reference(recording(data), mode="laplacian").data
        rows = [NAMES.index(n) for n in ("FC3", "C1", "CP3")]
        c3 = NAMES.index("C3")
        self.assertAlmostEqual(out[c3, 10], data[c3, 10] - data[rows, 10].mean())

    def test_unknown_names_and_too_few_channels_are_refused(self):
        with self.assertRaisesRegex(ValueError, "10-20/10-05"):
            compute.laplacian(np.zeros((5, 10)), ["C3", "C4", "Cz", "Fz", "EXG1"])
        with self.assertRaisesRegex(ValueError, "at least 5"):
            compute.laplacian(np.zeros((4, 10)), ["C3", "C4", "Cz", "Fz"])

    def test_the_other_modes_are_unchanged(self):
        raw = recording(self.data)
        np.testing.assert_allclose(ee.reference(raw).data, self.data - self.data.mean(0))
        np.testing.assert_allclose(ee.reference(raw, mode="channels", channels=["Cz"]).data,
                                   self.data - self.data[NAMES.index("Cz")])

    def test_the_methods_sentence_names_no_paper(self):
        from eeg.pipeline import block
        text = block("reference").describe({"mode": "laplacian", "channels": []})
        self.assertIn("four nearest electrodes", text)
        self.assertNotIn("19", text)


class LaplacianExport(unittest.TestCase):
    def test_the_exported_script_computes_the_same_laplacian_without_giova(self):
        from test_standalone_export import Matching, cho_file, on_the_sheet, run_package_alone, scratch
        from eeg.pipeline import Graph, templates
        matcher = type("M", (Matching, unittest.TestCase), {"runTest": lambda self: None})()
        with scratch() as directory:
            path = cho_file(directory, trials=20)
            graph = Graph("Laplacian")
            graph.chain(("load", {"path": str(path), "condition": templates.CHO2017_IMAGERY}),
                        ("reference", {"mode": "laplacian"}), ("filter", {"low": 8.0, "high": 30.0, "notch": 0.0}),
                        ("epochs", {"tmin": 2.0, "tmax": 5.0}), "psd")
            alone, folder = run_package_alone(graph, directory)
            matcher.assert_same(on_the_sheet(graph), alone, "laplacian")
            self.assertIn("def laplacian(", (folder / "eeg_studio_code.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
