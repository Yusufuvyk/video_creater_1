"""Regressions for the 23 October 2026 software review (B1-B6).

Each test failed on the code the review looked at.
Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from test_trial_analysis import scratch  # noqa: E402
from eeg.core import boards  # noqa: E402
from eeg.core.sources import LiveSource, STRANDED_LOG, write_board_metadata  # noqa: E402
from eeg.formats.channels import save_names  # noqa: E402
from eeg.pipeline import Graph, Runner, batch, learn  # noqa: E402
from eeg.pipeline.data import Epochs, Features  # noqa: E402
from eeg.pipeline.templates import csp_lda  # noqa: E402


class UnequalConditionsTests(unittest.TestCase):
    """B1: 90 trials of a, 10 of b, and nothing to learn from."""

    def test_always_guessing_the_larger_condition_is_not_better_than_random(self):
        y = np.array(["a"] * 90 + ["b"] * 10)
        scores = learn.cross_validate(np.zeros((100, 3)), y, learn.Model().then("lda"))
        self.assertAlmostEqual(scores.accuracy, 0.9)
        self.assertAlmostEqual(scores.balanced_accuracy, 0.5)
        self.assertGreater(scores.accuracy, scores.random_limit)      # the binomial limit alone would pass it
        self.assertTrue(scores.unequal)
        self.assertFalse(scores.better_than_random)
        self.assertIn("unequal trials", scores.random_text())
        self.assertIn("always guessing a", scores.random_text())

    def test_a_real_effect_with_unequal_conditions_still_counts(self):
        rng = np.random.default_rng(3)
        y = np.array(["a"] * 60 + ["b"] * 30)
        x = rng.normal(size=(90, 4)) + 2.0 * (y == "b")[:, None]
        scores = learn.cross_validate(x, y, learn.Model().then("lda"))
        self.assertTrue(scores.unequal)
        self.assertGreater(scores.balanced_accuracy, scores.balanced_limit)
        self.assertTrue(scores.better_than_random)

    def test_balanced_limit_is_a_random_classifiers(self):
        self.assertTrue(np.isnan(learn.balanced_limit(np.repeat(["a", "b"], 50))))    # equal: not used
        limit = learn.balanced_limit(np.array(["a"] * 90 + ["b"] * 10))
        self.assertGreater(limit, 0.5)
        self.assertLess(limit, 0.75)

    def test_a_permutation_test_that_disagrees_decides(self):
        y = np.array(["a"] * 90 + ["b"] * 10)
        scores = learn.cross_validate(np.zeros((100, 3)), y, learn.Model().then("lda"), permutations=9)
        self.assertEqual(scores.permutation_p, 1.0)
        self.assertFalse(scores.better_than_random)
        x, y = np.random.default_rng(1).normal(size=(80, 3)), np.repeat(["a", "b"], 40)
        x[y == "b"] += 1.5
        scores = learn.cross_validate(x, y, learn.Model().then("lda"), permutations=9)   # p at least 0.1
        self.assertGreater(scores.accuracy, scores.random_limit)
        self.assertFalse(scores.better_than_random)
        self.assertIn("permutation test does not confirm", scores.random_text())


class SidecarCacheTests(unittest.TestCase):
    """B2: what is read with a recording belongs to the Recording block's key."""

    def test_changing_creating_and_deleting_a_sidecar_reads_the_file_again(self):
        with scratch() as folder:
            path = folder / "raw.csv"
            b = boards.CYTON
            rows = np.zeros((500, b.n_rows))
            rows[:, list(b.eeg_rows)] = np.random.default_rng(2).normal(size=(500, b.n_channels))
            rows[:, b.timestamp_row] = 1700000000 + np.arange(500) / b.fs
            np.savetxt(path, rows, delimiter="\t")
            write_board_metadata(path, b)
            graph = Graph()
            load = graph.add("load", {"path": str(path)})
            runner = Runner(graph)
            default = runner.run()[load]["raw"].channel_names
            names = ["A{}".format(i) for i in range(b.n_channels)]
            save_names(path, names)                                     # created
            self.assertEqual(runner.run()[load]["raw"].channel_names, names)
            renamed = ["X{}".format(i) for i in range(b.n_channels)]
            save_names(path, renamed)                                   # changed
            self.assertEqual(runner.run()[load]["raw"].channel_names, renamed)
            (folder / "raw.channels.json").unlink()                     # deleted
            self.assertEqual(runner.run()[load]["raw"].channel_names, default)
            before = runner.run()[load]["raw"]
            self.assertIs(runner.run()[load]["raw"], before)            # nothing changed: still kept

    def test_multi_file_formats_name_their_data_files(self):
        from eeg.formats.recordings import dependencies
        with scratch() as folder:
            header = folder / "s1.vhdr"
            header.write_text("[Common Infos]\nDataFile=data_s1.eeg\nMarkerFile=marks_s1.vmrk\n", encoding="latin-1")
            files = [p.name for p in dependencies(header)]
            self.assertEqual(files[0], "s1.vhdr")
            self.assertIn("data_s1.eeg", files)
            self.assertIn("marks_s1.vmrk", files)
            self.assertIn("s1.fdt", [p.name for p in dependencies(folder / "s1.set")])
            self.assertIn("alpha_events.csv", [p.name for p in dependencies(folder / "alpha.csv")])


class AcrossSchemaTests(unittest.TestCase):
    """B3: equal shapes are not enough to train on one recording and test on another."""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.y = np.array(["a", "b"] * 30)
        self.a = Features(np.column_stack([self.y == "a", self.y == "b"]).astype(float)
                          + rng.normal(0, 0.03, (len(self.y), 2)), ["C3 alpha", "C4 alpha"], list(self.y),
                          [str(i) for i in range(len(self.y))])

    def across(self, first, second):
        model = learn.Model().then("lda")
        with patch.object(batch.Runner, "run", side_effect=[
                {"features": {"features": first}, "lda": {"model": model}},
                {"features": {"features": second}, "lda": {"model": model}}]), \
                patch.object(batch, "_cv_feeds", return_value=("features", "features", "lda", "model")):
            return batch.across([csp_lda()], ["a.csv", "b.csv"])

    def test_features_in_another_order_are_put_in_the_first_ones_order(self):
        swapped = Features(self.a.values[:, ::-1].copy(), list(reversed(self.a.names)), list(self.y), self.a.ids)
        result = self.across(self.a, swapped)
        self.assertFalse(result.errors)
        self.assertTrue(np.all(result.accuracy == 1), result.accuracy)

    def test_other_features_are_refused_with_a_reason(self):
        other = Features(self.a.values.copy(), ["O1 alpha", "O2 alpha"], list(self.y), self.a.ids)
        result = self.across(self.a, other)
        message = result.errors[("b", csp_lda().name)]
        self.assertIn("missing C3 alpha", message)

    def test_fif_files_with_swapped_channels_are_read_alike(self):
        import mne
        with scratch() as folder:
            fs, count = 128, 40
            labels = ["left", "right"] * (count // 2)
            data = np.random.default_rng(10).normal(0, 1e-6, (2, fs * 2 * count))
            for k, label in enumerate(labels):
                data[0 if label == "left" else 1, k * fs * 2:(k + 1) * fs * 2] *= 5
            paths = []
            for name, names, values in [("a_raw.fif", ["C3", "C4"], data), ("b_raw.fif", ["C4", "C3"], data[::-1])]:
                raw = mne.io.RawArray(values, mne.create_info(names, fs, "eeg"), verbose="error")
                raw.set_annotations(mne.Annotations(np.arange(count) * 2.0, [2.0] * count, labels))
                raw.save(folder / name, verbose="error")
                paths.append(folder / name)
            graph = Graph("order")
            *_, features = graph.chain(("load", {"path": str(paths[0])}), ("epochs", {"tmin": 0.0, "tmax": 1.0}),
                                       ("psd", {"window": 0.5}), ("band_power", {"bands": [["alpha", 8, 13]]}))
            lda, cv = graph.add("lda"), graph.add("cross_validate")
            graph.connect(features, "features", cv, "data")
            graph.connect(lda, "model", cv, "model")
            across = batch.across([graph], paths)
        self.assertFalse(across.errors)
        self.assertTrue(np.all(across.accuracy == 1), across.accuracy)     # was 0% before the review

    def test_trials_at_another_rate_are_refused(self):
        x = np.zeros((4, 2, 100))
        first = Epochs(x, 250.0, ["C3", "C4"], ["a", "b"] * 2, 0.0, ["1", "2", "3", "4"])
        second = Epochs(x, 125.0, ["C3", "C4"], ["a", "b"] * 2, 0.0, ["1", "2", "3", "4"])
        reference = batch._schema(first)
        with self.assertRaisesRegex(ValueError, "125 Hz"):
            batch._aligned(second.data, batch._schema(second), reference)
        swapped = Epochs(x[:, ::-1], 250.0, ["C4", "C3"], ["a", "b"] * 2, 0.0, ["1", "2", "3", "4"])
        swapped.data[:, 0] = 1.0                                         # C4
        aligned = batch._aligned(swapped.data, batch._schema(swapped), reference)
        self.assertTrue(np.all(aligned[:, 1] == 1.0))                     # C4 is second, as in the first


class StrandedRecordingTests(unittest.TestCase):
    """B4: a recording that cannot be moved to its folder is reported, and its place kept."""

    def test_a_failed_move_keeps_the_path_and_leaves_a_note(self):
        with scratch() as folder:
            stand_in = folder / "stand-in.csv"
            stand_in.write_text("sample data", encoding="utf-8")
            source = LiveSource.__new__(LiveSource)
            source.board = SimpleNamespace(stop_stream=lambda: None, release_session=lambda: None)
            source._stream_path = stand_in
            source.record_path = folder / "expected.csv"
            with patch("shutil.move", side_effect=OSError("disk full (test)")):
                source.close()
            self.assertEqual(source.stranded_path, stand_in)
            self.assertIn("disk full", source.move_error)
            self.assertIn("expected.csv", (folder / STRANDED_LOG).read_text(encoding="utf-8"))
            source.close()                                              # a second try moves it
            self.assertTrue(source.record_path.exists())
            self.assertIsNone(source.stranded_path)


class TuningSearchTests(unittest.TestCase):
    """B5: the inner search refits the whole chain on each inner training part."""

    def test_selection_and_augmentation_are_inside_the_inner_folds(self):
        from sklearn.model_selection import GridSearchCV
        fitted = []
        original = learn.SelectBest.fit

        def spy(self, x, y):
            fitted.append(len(y))
            return original(self, x, y)
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(40, 30)), np.array(["a", "b"] * 20)
        with patch.object(learn.SelectBest, "fit", spy), patch.object(GridSearchCV, "fit") as grid:
            steps, *_ = learn.fit_and_test(learn.Model().then("select", k=3).then("svm", tune=True), 42, x, y, x[:4])
        grid.assert_not_called()
        # once per inner fold of about 27 trials (the 5 settings share it), then once on all 40
        self.assertEqual(len(fitted), 3 + 1)
        self.assertTrue(all(n < 40 for n in fitted[:-1]))
        self.assertEqual(fitted[-1], 40)
        self.assertIn("C", steps[-1].chosen)

        seen = []
        original_fit = learn.Library.fit

        def watch(self, a, b):
            seen.append(np.asarray(a).copy())
            return original_fit(self, a, b)
        with patch.object(learn.Library, "fit", watch):
            learn._search(learn.Model().then("augment", copies=1, noise=0, scale=0, shift=0).then("svm", tune=True),
                          42, [1], x, y)
        # every inner fit holds each of its original trials twice (copy), and only those
        for trained in seen:
            self.assertEqual(len(trained) % 2, 0)
            half = len(trained) // 2
            np.testing.assert_array_equal(trained[:half], trained[half:])


class SameNameTests(unittest.TestCase):
    """B6: two recordings called session.csv in different folders keep their own results."""

    def test_names_tell_twins_apart(self):
        names = batch.recording_names(["p_a/session.csv", "p_b/session.csv", "p_a/rest.csv"])
        self.assertEqual(names, ["p_a/session", "p_b/session", "rest"])
        self.assertEqual(batch.recording_names(["x/s.csv", "x/s.csv"]), ["s (1)", "s (2)"])

    def test_comparison_keeps_both_results(self):
        graph = csp_lda()
        node = batch.score_nodes(graph)[0]
        first = SimpleNamespace(accuracy=0.1, auc=0.2, better_than_random=False)
        second = SimpleNamespace(accuracy=0.9, auc=0.95, better_than_random=True)
        with patch.object(batch.Runner, "run", side_effect=[{node: {"scores": first}}, {node: {"scores": second}}]):
            result = batch.compare([graph], ["participant_a/session.csv", "participant_b/session.csv"])
        self.assertEqual(result.subjects, ["participant_a/session", "participant_b/session"])
        self.assertEqual(len(result.scores), 2)
        self.assertIs(result.better(0, 0), False)
        self.assertIs(result.better(1, 0), True)


if __name__ == "__main__":
    unittest.main()
