"""The literature library (pipeline/literature.py): each profile checked as its `source` says.

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
from pyqtgraph.Qt import QtWidgets  # noqa: E402
from scipy import linalg  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_pipeline import save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.pipeline import Graph, Runner, batch, learn, literature  # noqa: E402
from eeg.pipeline.export import to_python  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def imagery_trials(n=60, channels=8, seed=0):
    """Two conditions whose power differs on different channels: what CSP-like methods separate."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, channels, 200))
    y = np.repeat(["left", "right"], n // 2)
    x[y == "left", 0] *= 2.5
    x[y == "right", 1] *= 2.5
    return x, y


def wide(recording, extra=5, seed=3):
    """`recording` with `extra` noise channels, so 3 + 3 spatial filters fit."""
    rng = np.random.default_rng(seed)
    data = np.vstack([recording.data, rng.normal(0, 5, (extra, recording.n_samples))])
    names = list(recording.channel_names) + ["N{}".format(k) for k in range(extra)]
    return Recording(data, recording.fs, names, events=recording.events)


class MdrmTests(unittest.TestCase):
    def test_same_predictions_as_an_independent_pyriemann_chain(self):
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        x, y = imagery_trials()
        ours = learn.cross_validate(x, y, learn.Model().then("covariance", estimator="scm").then("mdm"))
        predicted = {}
        for test in learn.stratified_folds(y, 5, 42):
            train = np.setdiff1d(np.arange(len(y)), test)
            model = MDM(metric="riemann").fit(Covariances("scm").fit_transform(x[train]), y[train])
            for i, p in zip(test, model.predict(Covariances("scm").transform(x[test]))):
                predicted[str(i + 1)] = str(p)
        self.assertEqual({trial: p for trial, _t, p, _s in ours.predictions}, predicted)
        self.assertGreater(ours.accuracy, 0.85)


class TrcspTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = imagery_trials()
        self.covs = [learn.normalised_covariance(self.x[self.y == c]) for c in ("left", "right")]

    def ratio(self, w, alpha, own=0):
        a, b = self.covs[own], self.covs[1 - own]
        return (w @ a @ w) / (w @ b @ w + alpha * w @ w)

    def test_each_filter_maximises_the_penalised_ratio(self):
        alpha = 1e-2
        step = learn.TRCSP(3, alpha).fit(self.x, self.y)
        best = linalg.eigh(self.covs[0], self.covs[1] + alpha * np.eye(8), eigvals_only=True).max()
        self.assertAlmostEqual(self.ratio(step.filters[0], alpha), best, places=8)
        rng = np.random.default_rng(1)
        self.assertTrue(all(self.ratio(w, alpha) <= best + 1e-9 for w in rng.normal(size=(500, 8))))
        self.assertGreater(self.ratio(step.filters[3], alpha, own=1), self.ratio(step.filters[0], alpha, own=1))
        self.assertEqual(step.transform(self.x).shape, (60, 6))

    def test_a_vanishing_alpha_spans_plain_csp(self):
        step = learn.TRCSP(3, 1e-12).fit(self.x, self.y)
        values, vectors = linalg.eigh(self.covs[0], self.covs[0] + self.covs[1])
        csp = vectors[:, np.argsort(values)[::-1][:3]]
        # the same subspace: each TRCSP filter is a combination of plain CSP's three
        for w in step.filters[:3]:
            residual = w - csp @ np.linalg.lstsq(csp, w, rcond=None)[0]
            self.assertLess(np.linalg.norm(residual), 1e-6)

    def test_alpha_is_tuned_inside_the_fold_with_ten_inner_folds(self):
        model = learn.Model().then("trcsp", n_filters=2, tune=True, inner_folds=10).then("lda", shrinkage="none")
        scores = learn.cross_validate(self.x, self.y, model)
        self.assertTrue(all("alpha" in c and c["alpha"] in learn.TRCSP_ALPHAS for c in scores.chosen))
        self.assertGreater(scores.accuracy, 0.85)
        self.assertEqual(learn._trcsp(tune=True).inner_folds, 10)


class FbcspTests(unittest.TestCase):
    def test_the_bank_inside_trials_matches_filtering_the_whole_recording(self):
        from scipy.signal import sosfilt
        from eeg.pipeline import api as ee, filters as fl
        from eeg.pipeline.data import Epochs
        fs, rng = 128.0, np.random.default_rng(2)
        t = np.arange(int(60 * fs)) / fs
        # noise, a 10 Hz rhythm and a slow drift, as an unfiltered recording has
        signal = (rng.normal(0, 5, (2, t.size)) + 20 * np.sin(2 * np.pi * 10 * t)
                  + 0.2 * rng.normal(0, 1, (2, t.size)).cumsum(axis=1))
        onsets = np.arange(5, 50, 6.0)
        start, end = 2.5, 4.5
        tmin = start - literature.SETTLE
        trials = np.stack([signal[:, int((o + tmin) * fs):int((o + end) * fs)] for o in onsets])
        epochs = Epochs(trials, fs, ["C3", "C4"], ["a", "b"] * (len(onsets) // 2) + ["a"] * (len(onsets) % 2),
                        tmin, [str(k + 1) for k in range(len(onsets))])
        bank = ee.filter_bank(epochs, start=start, end=end)
        self.assertEqual(bank.data.shape[1], 9 * 2)
        self.assertEqual(bank.names[2], "C3 · 8–12 Hz")
        self.assertAlmostEqual(bank.tmin, start)
        whole = signal - signal.mean(axis=1, keepdims=True)
        for k, (low, high) in enumerate(blocks_bands()):
            _, sos = fl.design(fs, low, high, fl.PASS, "chebyshev2", 4, 1.0, 40.0)
            continuous = sosfilt(sos, whole, axis=-1)
            for i, onset in enumerate(onsets):
                a = int(round((onset + start) * fs))
                reference = continuous[:, a:a + bank.data.shape[2]]
                reference = reference - reference.mean(axis=1, keepdims=True)
                ours = bank.data[i, 2 * k:2 * k + 2]
                # after SETTLE s what is left of each trial's start is under 1% of the band's signal
                self.assertLess(np.abs(ours - reference).max(), 0.01 * np.sqrt(np.mean(reference ** 2)), (low, onset))
        # the 10 Hz sine lives in 8-12 Hz; 4-8 Hz holds only noise
        power = (bank.data ** 2).mean(axis=(0, 2))
        self.assertGreater(power[2], 10 * power[0])

    def test_mutual_information_matches_scipy_parzen_windows(self):
        from scipy.stats import gaussian_kde
        rng = np.random.default_rng(4)
        y = np.repeat(["a", "b"], [30, 20])
        f = rng.normal(size=50) + 1.5 * (y == "b")
        priors = np.array([0.6, 0.4])
        joint = np.array([p * gaussian_kde(f[y == c], bw_method=(4 / (3 * np.sum(y == c))) ** 0.2)(f)
                          for p, c in zip(priors, ("a", "b"))])
        posterior = joint / joint.sum(axis=0)
        expected = -np.sum(priors * np.log2(priors)) + np.sum(posterior * np.log2(posterior)) / f.size
        self.assertAlmostEqual(learn.mutual_information(f, y), expected, places=10)
        noise = rng.normal(size=50)
        self.assertGreater(learn.mutual_information(f, y), learn.mutual_information(noise, y))

    def test_nbpw_matches_scipy_parzen_windows(self):
        from scipy.stats import gaussian_kde
        rng = np.random.default_rng(5)
        y = np.repeat(["a", "b"], 25)
        x = rng.normal(size=(50, 3)) + np.outer(y == "b", [1.0, 0.0, -0.5])
        model = learn.NBPW().fit(x, y)
        test = rng.normal(size=(7, 3))
        expected = np.column_stack([np.log(0.5) + sum(gaussian_kde(x[y == c, j], bw_method=(4 / 75) ** 0.2)
                                                      .logpdf(test[:, j]) for j in range(3)) for c in ("a", "b")])
        np.testing.assert_allclose(model.log_posterior(test), expected, rtol=1e-9)
        self.assertEqual(list(model.predict(test)), list(np.array(["a", "b"])[expected.argmax(axis=1)]))

    def test_mibif_keeps_the_best_and_their_csp_partners(self):
        rng = np.random.default_rng(6)
        y = np.repeat(["a", "b"], 40)
        x = rng.normal(size=(80, 8))                       # two bands of four FBCSP features
        x[:, 1] += 2.0 * (y == "b")                         # band 1, feature 1: its partner is feature 2
        x[:, 4] -= 1.5 * (y == "b")                         # band 2, feature 0: its partner is feature 7
        step = learn.MIBIF(k=2, pair_size=4).fit(x, y)
        self.assertEqual(list(step.selected), [1, 2, 4, 7])
        self.assertEqual(step.transform(x).shape, (80, 4))

    def test_fbcsp_features_come_band_by_band_in_pairs(self):
        x, y = imagery_trials(channels=4)
        bank = np.concatenate([x, x[:, ::-1]], axis=1)       # two "bands" of four channels
        features = learn.FBCSP(n_bands=2, n_pairs=2).fit(bank, y).transform(bank)
        self.assertEqual(features.shape, (60, 8))
        with self.assertRaisesRegex(ValueError, "Filter bank"):
            learn.FBCSP(n_bands=3).fit(bank, y)


def blocks_bands():
    from eeg.pipeline.blocks import FBCSP_BANDS
    return FBCSP_BANDS


class LibraryTests(unittest.TestCase):
    def test_every_profile_says_where_it_comes_from_and_survives_saving(self):
        for title, build in literature.LIBRARY.items():
            with self.subTest(title):
                graph = build()
                for field in ("id", "citation", "doi", "status"):
                    self.assertTrue(graph.source.get(field), field)
                self.assertTrue(graph.source["adaptations"])
                again = Graph.from_dict(graph.to_dict())
                self.assertEqual(again.source, graph.source)
                self.assertFalse(again.changed_from_source())
                again.nodes[again.sources("load")[0]].params["path"] = "elsewhere.fif"   # another file: same method
                self.assertFalse(again.changed_from_source())
                cv = batch.score_nodes(again)[0]
                again.nodes[cv].params["folds"] = 3
                self.assertTrue(again.changed_from_source())

    def test_profiles_run_and_report_their_source(self):
        from eeg.ui.workbench.report import pipeline_report
        with scratch() as folder:
            path = save_fif(wide(lateralised(trials=40)), folder)
            for build in literature.LIBRARY.values():
                graph = build(str(path), "")
                outputs = Runner(graph).run()
                scores = outputs[batch.score_nodes(graph)[0]]["scores"]
                with self.subTest(graph.name):
                    self.assertGreater(scores.accuracy, 0.8)
                    text = pipeline_report(graph, outputs).to_html()
                    self.assertIn("Literature profile", text)
                    self.assertIn(graph.source["doi"], text)
                    self.assertIn("not the paper", text)
                    self.assertIn(graph.source["citation"][:30], to_python(graph))

    def test_holm(self):
        self.assertEqual(batch.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        adjusted = batch.holm([0.2, float("nan"), 0.01])
        self.assertEqual(adjusted[0], 0.2)
        self.assertTrue(np.isnan(adjusted[1]))
        self.assertAlmostEqual(adjusted[2], 0.02)


if __name__ == "__main__":
    unittest.main()
