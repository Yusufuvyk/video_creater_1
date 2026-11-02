"""The Workbench engine: graphs, wires, caching, files, blocks and classification.

CSP, LDA and Ledoit-Wolf are checked against MNE and scikit-learn when those are
installed (development only; the program does not need them). The Cho2017 test
runs when the dataset is on this computer.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.pipeline import Graph, PipelineError, Runner, block  # noqa: E402
from eeg.pipeline import batch, learn, templates  # noqa: E402
from eeg.pipeline.data import Epochs  # noqa: E402

try:
    import sklearn  # noqa: F401
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.covariance import ledoit_wolf_shrinkage
    from mne.decoding import CSP as MNECSP
    HAVE_REFERENCE = True
except ImportError:
    HAVE_REFERENCE = False

CHO2017 = Path(os.environ.get("CHO2017_DIR", "cho2017"))


def save_fif(recording, directory, name="mi_raw.fif"):
    """`recording` as a FIF file with its events as annotations: the Recording block's input."""
    import mne
    info = mne.create_info(recording.channel_names, recording.fs, "eeg")
    raw = mne.io.RawArray(recording.data * 1e-6, info, verbose="error")
    events = recording.events
    raw.set_annotations(mne.Annotations(events.onsets, events.durations, events.labels))
    path = directory / name
    raw.save(str(path), verbose="error")
    return path


def imagery(path):
    """Load → filter 8-30 Hz → epochs 2-5 s → CV(CSP → LDA)."""
    graph = Graph("test")
    _, filt, epochs = graph.chain(("load", {"path": str(path)}), ("filter", {"low": 8.0, "high": 30.0}),
                                  ("epochs", {"tmin": 2.0, "tmax": 5.0}))
    _, lda = graph.chain(("csp", {"n_components": 2}), "lda")
    cv = graph.add("cross_validate", {"folds": 4})
    graph.connect(epochs, "epochs", cv, "data")
    graph.connect(lda, "model", cv, "model")
    return graph, filt, cv


class GraphTests(unittest.TestCase):
    def test_a_wire_must_carry_a_type_its_input_accepts(self):
        graph = Graph()
        load, psd = graph.add("load"), graph.add("psd")
        with self.assertRaisesRegex(ValueError, "needs epochs"):
            graph.connect(load, "raw", psd, "epochs")
        filt = graph.add("filter")
        with self.assertRaisesRegex(ValueError, "no port"):
            graph.connect(psd, "spectrum", filt, "spectrum")

    def test_loops_and_unknown_parameters_are_refused(self):
        graph = Graph()
        a, b = graph.chain("filter", "filter")
        with self.assertRaisesRegex(ValueError, "loop"):
            graph.connect(b, "raw", a, "raw")
        with self.assertRaisesRegex(ValueError, "no parameter"):
            graph.add("filter", {"cutoff": 3})

    def test_a_new_wire_replaces_the_old_one_on_the_same_input(self):
        graph = Graph()
        a, b, c = graph.add("load"), graph.add("load"), graph.add("filter")
        graph.connect(a, "raw", c, "raw")
        graph.connect(b, "raw", c, "raw")
        self.assertEqual([w.source for w in graph.wires], [b])

    def test_missing_inputs_are_reported_per_node(self):
        graph = Graph()
        cv = graph.add("cross_validate")
        problems = graph.problems()
        self.assertEqual({node for node, _ in problems}, {cv})
        self.assertEqual(len(problems), 2)
        with self.assertRaises(PipelineError) as caught:
            Runner(graph).run()
        self.assertEqual(caught.exception.node, cv)

    def test_file_round_trip_keeps_everything(self):
        graph = templates.csp_lda("x.mat")
        graph.nodes["filter1"].label = "Mu and beta"
        with scratch() as directory:
            path = directory / ("t" + ".pipeline.json")
            graph.save(path)
            again = Graph.load(path)
        self.assertEqual(again.to_dict(), graph.to_dict())

    def test_methods_put_the_signal_path_before_the_model(self):
        text = templates.csp_lda("s01.mat").methods()
        self.assertTrue(text[0].startswith("Recording s01.mat"))
        self.assertIn("band-pass 8–30 Hz", text[1])
        self.assertTrue(text[-1].startswith("Stratified 5-fold"))

    def test_every_block_has_a_title_category_and_valid_defaults(self):
        from eeg.pipeline import REGISTRY
        for name, kind in REGISTRY.items():
            self.assertTrue(kind.title and kind.category, name)
            for param in kind.params:
                if param.choices:
                    self.assertIn(param.default, param.choices, (name, param.name))


class RunTests(unittest.TestCase):
    def test_lateralised_alpha_is_classified_and_only_changed_steps_rerun(self):
        recording = lateralised(fs=250, trials=40)
        with scratch() as directory:
            path = save_fif(recording, directory)
            graph, filt, cv = imagery(path)
            runner = Runner(graph)
            scores = runner.run()[cv]["scores"]
            self.assertGreaterEqual(scores.accuracy, 0.9)
            self.assertEqual(scores.counts, {"left": 20, "right": 20})
            self.assertEqual(scores.confusion.sum(), 40)
            filtered = runner.run()[filt]["raw"]
            graph.set(cv, folds=5)
            again = runner.run()
            self.assertIs(again[filt]["raw"], filtered)          # unchanged: not recomputed
            self.assertEqual(again[cv]["scores"].n_folds, 5)

    def test_overrides_run_one_pipeline_on_another_file(self):
        with scratch() as directory:
            a = save_fif(lateralised(trials=20, seed=1), directory, "a_raw.fif")
            b = save_fif(lateralised(trials=20, seed=2), directory, "b_raw.fif")
            graph, _, cv = imagery(a)
            runner = Runner(graph)
            first = runner.run()[cv]["scores"]
            second = runner.run(overrides={"load1": {"path": str(b)}})[cv]["scores"]
        self.assertNotEqual(first.predictions, second.predictions)

    def test_block_errors_name_the_block(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=10), directory)
            graph, _, _ = imagery(path)
            graph.set("epochs1", conditions=["up"])
            with self.assertRaises(PipelineError) as caught:
                Runner(graph).run()
        self.assertEqual(caught.exception.node, "epochs1")
        self.assertIn("left, right", str(caught.exception))

    def test_band_power_features_and_window(self):
        recording = lateralised(trials=10)
        raw = {"raw": recording}
        context = __import__("eeg.pipeline.graph", fromlist=["Context"]).Context()
        epochs = block("epochs").run(raw, {**block("epochs").defaults(), "tmin": 0.0, "tmax": 7.0}, context)
        cut = block("window").run(epochs, {"start": 2.0, "end": 5.0}, context)["epochs"]
        self.assertEqual(cut.data.shape[-1], 3 * 250)
        self.assertAlmostEqual(cut.tmin, 2.0)
        spectrum = block("psd").run({"epochs": cut}, block("psd").defaults(), context)
        features = block("band_power").run(spectrum, {"bands": [["alpha", 8, 13]], "log": False}, context)
        features = features["features"]
        self.assertEqual(features.names, ["C3 alpha", "Cz alpha", "C4 alpha"])
        left = np.array(features.labels) == "left"
        # alpha drops on C4 in left trials and on C3 in right trials
        self.assertLess(features.values[left, 2].mean(), features.values[~left, 2].mean())
        self.assertLess(features.values[~left, 0].mean(), features.values[left, 0].mean())


class LearnTests(unittest.TestCase):
    def test_folds_are_balanced_and_cover_every_trial_once(self):
        labels = ["a"] * 23 + ["b"] * 17
        folds = learn.stratified_folds(labels, 5, seed=3)
        joined = np.sort(np.concatenate(folds))
        np.testing.assert_array_equal(joined, np.arange(40))
        for fold in folds:
            self.assertIn(sum(labels[i] == "a" for i in fold), (4, 5))

    def test_noise_scores_at_chance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 8, 100))
        labels = np.repeat(["a", "b"], 100)
        scores = learn.cross_validate(x, labels, learn.Model().then("csp", n_components=4).then("lda"))
        self.assertFalse(scores.better_than_random)
        self.assertAlmostEqual(scores.chance, 0.5)

    def test_random_limit_is_the_papers(self):
        # Müller-Putz et al. (2008), section 2.1: 100 trials, alpha 0.05 -> 40.39% to 59.61%
        low, high = learn.random_interval(100, 2)
        self.assertAlmostEqual(low, 0.4039, places=4)
        self.assertAlmostEqual(high, 0.5961, places=4)
        # their simulation, table 1 (alpha 5% / 1%), trials per class -> upper limit in %
        for per_class, classes, five, one in ((40, 3, 41.7, 45.0), (40, 4, 31.9, 34.4), (80, 4, 29.7, 31.3),
                                              (20, 8, 18.1, 19.4), (160, 3, 37.7, 39.0)):
            labels = np.repeat(np.arange(classes), per_class)
            step = 100.0 / labels.size            # one trial more or fewer
            self.assertLessEqual(abs(100 * learn.random_limit(labels, 0.05) - five), step + 0.05, (per_class, classes))
            self.assertLessEqual(abs(100 * learn.random_limit(labels, 0.01) - one), step + 0.05, (per_class, classes))

    def test_a_model_must_end_with_a_classifier(self):
        with self.assertRaisesRegex(ValueError, "end with a classifier"):
            learn.cross_validate(np.zeros((10, 3)), ["a", "b"] * 5, learn.Model().then("standardize"), 2)
        with self.assertRaisesRegex(ValueError, "after the classifier"):
            learn.Model().then("lda").then("standardize")

    @unittest.skipUnless(HAVE_REFERENCE, "scikit-learn / mne.decoding not installed")
    def test_lda_matches_scikit_learn(self):
        rng = np.random.default_rng(1)
        x = np.r_[rng.normal(0, 1, (30, 12)), rng.normal(0.4, 1.3, (25, 12))]
        x[:, 3] = x[:, 2] * 2 + rng.normal(0, 0.01, 55)       # nearly collinear: shrinkage matters
        y = np.r_[["a"] * 30, ["b"] * 25]
        for ours, theirs in (("none", None), ("auto", "auto"), (0.3, 0.3)):
            mine = learn.LDA(ours).fit(x, y)
            reference = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=theirs).fit(x, y)
            np.testing.assert_allclose(mine.decision(x), reference.decision_function(x), rtol=1e-6, atol=1e-8)
            np.testing.assert_array_equal(mine.predict(x), reference.predict(x))
        self.assertAlmostEqual(learn.ledoit_wolf_shrinkage(x), ledoit_wolf_shrinkage(x), places=10)

    @unittest.skipUnless(HAVE_REFERENCE, "scikit-learn / mne.decoding not installed")
    def test_lda_matches_scikit_learn_for_three_conditions(self):
        rng = np.random.default_rng(2)
        x = np.r_[rng.normal(0, 1, (20, 5)), rng.normal(1, 1, (20, 5)), rng.normal(-1, 1, (20, 5))]
        y = np.repeat(["a", "b", "c"], 20)
        mine = learn.LDA("auto").fit(x, y)
        reference = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x, y)
        np.testing.assert_allclose(mine.decision(x), reference.decision_function(x), rtol=1e-6)

    def _lateral_trials(self):
        recording = lateralised(trials=40, seed=4)
        epochs = block("epochs").run({"raw": recording}, {**block("epochs").defaults(), "tmin": 2.0, "tmax": 5.0},
                                     __import__("eeg.pipeline.graph", fromlist=["Context"]).Context())["epochs"]
        return epochs.data, np.array(epochs.labels)

    def test_csp_works_in_the_rank_of_average_referenced_channels(self):
        x, y = self._lateral_trials()
        n = x.shape[1]
        full = learn.CSP(3).fit(x, y)
        self.assertEqual(full.rank, n)
        covs = [np.cov(x[y == c].transpose(1, 0, 2).reshape(n, -1)) for c in np.unique(y)]
        # full rank: exactly the generalised eigenproblem as before, so earlier results do not move
        for got, expected in zip(learn.csp_solve(*covs), __import__("scipy").linalg.eigh(covs[0], covs[0] + covs[1])):
            np.testing.assert_array_equal(got, expected)
        averaged = x - x.mean(axis=1, keepdims=True)                     # an average reference: rank n - 1
        csp = learn.CSP(2).fit(averaged, y)
        self.assertEqual((csp.rank, csp.filters.shape), (n - 1, (n - 1, n)))
        self.assertTrue(np.isfinite(csp.transform(averaged)).all())
        # the lost direction (all channels alike) is in no filter, and the filters still separate the hands
        np.testing.assert_allclose(csp.filters @ np.ones(n), 0, atol=1e-6 * np.abs(csp.filters).max())
        self.assertGreater(abs(csp.eigenvalues[0] - 0.5), 0.2)
        with self.assertRaisesRegex(ValueError, "rank"):
            learn.CSP(n).fit(averaged, y)
        with self.assertRaisesRegex(ValueError, "no usable signal"):
            learn.CSP(2).fit(np.zeros_like(x), y)
        bank = np.concatenate([averaged, averaged], axis=1)              # two bands, each average-referenced
        self.assertTrue(np.isfinite(learn.FBCSP(n_bands=2, n_pairs=1).fit(bank, y).transform(bank)).all())

    @unittest.skipUnless(HAVE_REFERENCE, "scikit-learn / mne.decoding not installed")
    def test_csp_matches_mne_after_an_average_reference(self):
        x, y = self._lateral_trials()
        averaged = x - x.mean(axis=1, keepdims=True)
        mine = learn.CSP(2).fit(averaged, y).transform(averaged)
        # MNE must be told the rank: left full, its generalised eigensolver fails on these trials too
        # (plain arrays are "mag" channels to it)
        theirs = MNECSP(n_components=2, log=True, rank={"mag": x.shape[1] - 1}).fit(averaged, y).transform(averaged)
        for k in range(2):
            np.testing.assert_allclose(mine[:, k] - mine[:, k].mean(), theirs[:, k] - theirs[:, k].mean(),
                                       atol=1e-3)

    @unittest.skipUnless(HAVE_REFERENCE, "scikit-learn / mne.decoding not installed")
    def test_csp_matches_mne(self):
        recording = lateralised(trials=40, seed=4)
        epochs = block("epochs").run({"raw": recording}, {**block("epochs").defaults(), "tmin": 2.0, "tmax": 5.0},
                                     __import__("eeg.pipeline.graph", fromlist=["Context"]).Context())["epochs"]
        x, y = epochs.data, np.array(epochs.labels)
        mine = learn.CSP(3).fit(x, y).transform(x)
        theirs = MNECSP(n_components=3, log=True).fit(x, y).transform(x)
        # A filter's scale and sign are arbitrary: log power then differs by a constant per filter.
        # MNE's covariance differs from ours in the fifth digit (it does not remove the mean), and
        # the third filter (eigenvalue ~0.5) separates nothing, so only the first two are compared.
        for k in range(2):
            np.testing.assert_allclose(mine[:, k] - mine[:, k].mean(), theirs[:, k] - theirs[:, k].mean(),
                                       atol=1e-3)


@unittest.skipUnless((CHO2017 / "s01.mat").is_file(), "Cho2017 not on this computer")
class Cho2017Tests(unittest.TestCase):
    def test_subject_one_csp_lda(self):
        graph = templates.csp_lda(str(CHO2017 / "s01.mat"))
        outputs = Runner(graph).run()
        epochs = outputs["epochs1"]["epochs"]
        self.assertEqual(epochs.counts(), {"left": 100, "right": 100})
        self.assertEqual(epochs.data.shape, (200, 64, 3 * 512))
        scores = outputs["cross_validate1"]["scores"]
        self.assertTrue(scores.better_than_random)
        if HAVE_REFERENCE:
            # The same trials and folds through MNE's CSP and scikit-learn's LDA.
            from sklearn.pipeline import make_pipeline
            x, y = epochs.data, np.array(epochs.labels)
            folds = learn.stratified_folds(y, 5, 42)
            agree = []
            for test in folds:
                train = np.setdiff1d(np.arange(len(y)), test)
                model = make_pipeline(MNECSP(n_components=6, log=True),
                                      LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
                agree.append(model.fit(x[train], y[train]).predict(x[test]))
            reference = np.mean(np.concatenate(agree) == y[np.concatenate(folds)])
            self.assertAlmostEqual(scores.accuracy, reference, delta=0.02)

    def test_two_pipelines_side_by_side(self):
        files = [CHO2017 / "s01.mat", CHO2017 / "s02.mat"]
        result = batch.compare([templates.csp_lda(), templates.band_power_lda()], files)
        self.assertEqual(result.accuracy.shape, (2, 2))
        self.assertTrue(np.isfinite(result.accuracy).all(), result.errors)
        self.assertEqual([row[0] for row in result.summary()], ["CSP + LDA", "Band power + LDA"])
        with scratch() as directory:
            result.save_csv(directory / "compare.csv")
            lines = (directory / "compare.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split(",")[:3], ["subject", "CSP + LDA accuracy", "Band power + LDA accuracy"])
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
