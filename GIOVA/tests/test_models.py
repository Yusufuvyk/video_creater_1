"""Filter designs, classifiers, cross-validation schemes, ICA and baseline correction.

Every classifier is run on the same folds as the scikit-learn / pyRiemann
pipeline it stands for and must predict the same; the FIR is checked against
MNE's own design rule; ICA must take a blink out of the frontal channels.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
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
from eeg.core.processing import filter_finite, mains_harmonics  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.pipeline import block, learn  # noqa: E402
from eeg.pipeline import filters as fl  # noqa: E402
from eeg.pipeline.graph import Context  # noqa: E402


def run(type_name, inputs, **params):
    kind = block(type_name)
    return kind.run(inputs, {**kind.defaults(), **params}, Context())


def separable(n=120, features=8, seed=0, shift=0.8):
    """Two conditions, some features informative."""
    rng = np.random.default_rng(seed)
    y = np.repeat(["a", "b"], n // 2)
    x = rng.normal(size=(n, features))
    x[y == "b", :3] += shift
    return x, y


def trials(seed=0):
    recording = lateralised(fs=250, trials=60, seed=seed)
    epochs = run("epochs", {"raw": recording}, tmin=2.0, tmax=5.0)["epochs"]
    return epochs.data, np.array(epochs.labels)


class FilterTests(unittest.TestCase):
    fs = 512.0

    def gain(self, params, hz):
        freqs, db = fl.response(self.fs, {**block("filter").defaults(), **params})
        return float(np.interp(hz, freqs, db))

    def test_butterworth_is_the_filter_the_other_pages_use(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=8000).cumsum()
        x[3000:3050] = np.nan
        params = {**block("filter").defaults(), "low": 8.0, "high": 30.0, "notch": 60.0}
        ours = fl.apply(x, self.fs, fl.steps(self.fs, params))
        core = filter_finite(x, self.fs, low=8.0, high=30.0, order=4, notches=mains_harmonics(60.0, self.fs))
        np.testing.assert_array_equal(np.isnan(ours), np.isnan(core))
        np.testing.assert_allclose(ours[np.isfinite(ours)], core[np.isfinite(core)], atol=1e-12)

    def test_every_design_passes_the_band_and_stops_outside_it(self):
        for design in fl.DESIGNS:
            params = {"low": 8.0, "high": 30.0, "notch": 0.0, "design": design}
            self.assertGreater(self.gain(params, 18), -3.5, design)
            self.assertLess(self.gain(params, 2), -30, design)
            self.assertLess(self.gain(params, 60), -30, design)

    def test_the_drawn_response_is_the_one_applied(self):
        """A sine through the chain as it is applied loses what the drawn response says, for a notch with an
        FIR or ideal band (the notch runs forward and backward, the band once) as for an IIR one."""
        fs, hz = 200.0, 49.194
        t = np.arange(int(60 * fs)) / fs
        x = np.sin(2 * np.pi * hz * t)
        for design in ("butterworth", "fir", "ideal"):
            params = {**block("filter").defaults(), "low": 1.0, "high": 70.0, "notch": 50.0, "design": design}
            y = fl.apply(x, fs, fl.steps(fs, params))
            middle = slice(len(t) // 4, 3 * len(t) // 4)
            measured = 20 * np.log10(np.std(y[middle]) / np.std(x[middle]))
            freqs, db = fl.response(fs, params, n=8192)
            self.assertAlmostEqual(measured, float(np.interp(hz, freqs, db)), delta=0.5, msg=design)

    def test_fir_follows_mnes_design(self):
        import mne
        taps = mne.filter.create_filter(None, self.fs, 8.0, 30.0, method="fir", fir_design="firwin",
                                        fir_window="hamming", verbose="error")
        kind, ours = fl.design(self.fs, 8.0, 30.0, design="fir")
        self.assertEqual(kind, "fir")
        self.assertEqual(ours.size, taps.size)
        np.testing.assert_allclose(ours, taps, atol=1e-10)

    def test_band_stop_removes_its_band_only(self):
        t = np.arange(int(10 * self.fs)) / self.fs
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 10 * t)
        params = {**block("filter").defaults(), "low": 45.0, "high": 55.0, "kind": "stop", "notch": 0.0}
        y = fl.apply(x, self.fs, fl.steps(self.fs, params))
        middle = slice(int(self.fs), -int(self.fs))
        residual = y[middle] - np.sin(2 * np.pi * 10 * t[middle])
        self.assertLess(np.sqrt(np.mean(residual ** 2)), 0.05)

    def test_causal_filters_delay_and_zero_phase_does_not(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=int(20 * self.fs))
        base = {**block("filter").defaults(), "low": 8.0, "high": 30.0, "notch": 0.0}
        for design in ("butterworth", "fir"):
            zero = fl.apply(x, self.fs, fl.steps(self.fs, {**base, "design": design}), "zero")
            causal = fl.apply(x, self.fs, fl.steps(self.fs, {**base, "design": design}), "causal")
            reference = fl.apply(x, self.fs, fl.steps(self.fs, {**base, "low": 7.0, "high": 31.0,
                                                               "design": design}), "zero")
            lag = lambda a: int(np.argmax(np.correlate(a[2000:-2000], reference[2000:-2000], "same"))) - \
                (len(a) - 4000) // 2
            self.assertEqual(lag(zero), 0, design)
            self.assertGreater(lag(causal), 0, design)

    def test_cut_offs_above_nyquist_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            fl.design(250.0, 1.0, 200.0)


class MeasureTests(unittest.TestCase):
    def test_kappa_and_f1_are_scikit_learns(self):
        from sklearn.metrics import cohen_kappa_score, f1_score
        rng = np.random.default_rng(3)
        for _ in range(400):
            k, n = int(rng.integers(2, 5)), int(rng.integers(4, 60))
            truth = rng.integers(0, k, n).astype(str)
            predicted = np.where(rng.random(n) < rng.random(), truth, rng.integers(0, k, n).astype(str))
            if len(np.union1d(truth, predicted)) > 1:
                self.assertAlmostEqual(learn.cohen_kappa(truth, predicted), cohen_kappa_score(truth, predicted))
            self.assertAlmostEqual(learn.macro_f1(truth, predicted),
                                   f1_score(truth, predicted, labels=np.unique(truth), average="macro",
                                            zero_division=0))

    def test_kappa_by_hand(self):
        # Cohen (1960): 10 trials, 7 right; true 5 a / 5 b, predicted 6 a / 4 b: p_e = .5 * .6 + .5 * .4 = .5
        truth = list("aaaaabbbbb")
        predicted = list("aaaabaabbb")
        self.assertAlmostEqual(learn.cohen_kappa(truth, predicted), (0.7 - 0.5) / (1 - 0.5))
        self.assertTrue(np.isnan(learn.cohen_kappa(list("aaa"), list("aaa"))))     # p_e = 1: undefined
        self.assertAlmostEqual(learn.cohen_kappa(list("abab"), list("abab")), 1.0)
        # F1: a 4 right, 1 taken for b; b 3 right, 2 taken for a. a: 8 / (8 + 2 + 1); b: 6 / (6 + 1 + 2)
        self.assertAlmostEqual(learn.macro_f1(truth, predicted), (8 / 11 + 6 / 9) / 2)

    def test_cross_validation_averages_kappa_and_f1_over_its_folds(self):
        rng = np.random.default_rng(0)
        y = np.repeat(["left", "right"], 40)
        x = rng.normal(size=(80, 4)) + (y == "right")[:, None] * 1.2
        folds = learn.stratified_folds(y, 5, 1)
        scores = learn.cross_validate(x, y, learn.Model().then("lda"), folds=folds)
        self.assertEqual(len(scores.fold_kappa), 5)
        self.assertEqual(len(scores.fold_f1), 5)
        self.assertAlmostEqual(scores.kappa, np.mean(scores.fold_kappa))
        self.assertAlmostEqual(scores.f1, np.mean(scores.fold_f1))
        # two balanced conditions in every fold: kappa is 2 accuracy - 1 fold by fold only when the predictions
        # are balanced too, so check the general formula on the first fold's own trials instead
        predicted = {row[0]: row[2] for row in scores.predictions}
        first = folds[0]
        truth = y[first]
        guess = np.array([predicted[str(i + 1)] for i in first])
        self.assertAlmostEqual(scores.fold_kappa[0], learn.cohen_kappa(truth, guess))
        self.assertIn("Cohen's kappa", scores.describe())


class ClassifierTests(unittest.TestCase):
    def same_as(self, model, reference, x, y, folds=None):
        """Our model and a scikit-learn pipeline, the same folds: the same predictions."""
        from sklearn.base import clone
        folds = folds or learn.stratified_folds(y, 5, 3)
        ours = learn.cross_validate(x, y, model, folds=folds)
        predicted = {}
        for test in folds:
            train = np.setdiff1d(np.arange(len(y)), test)
            fitted = clone(reference).fit(x[train], y[train])
            predicted.update(zip(test, fitted.predict(x[test])))
        theirs = np.mean([predicted[i] == y[i] for i in range(len(y))])
        mine = {row[0]: row[2] for row in ours.predictions}
        agree = np.mean([mine[str(i + 1)] == predicted[i] for i in range(len(y))])
        return ours, theirs, agree

    def test_each_classifier_predicts_as_scikit_learn_does(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        x, y = separable()
        cases = [
            (learn.Model().then("standardize").then("svm", kernel="linear", C=1.0),
             make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))),
            (learn.Model().then("standardize").then("svm", kernel="rbf", C=10.0),
             make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))),
            (learn.Model().then("logistic", penalty="l1", C=0.5),
             LogisticRegression(l1_ratio=1.0, C=0.5, solver="liblinear", max_iter=2000)),
            (learn.Model().then("forest", trees=50),
             RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)),
            (learn.Model().then("knn", k=7), KNeighborsClassifier(n_neighbors=7)),
        ]
        for model, reference in cases:
            ours, theirs, agree = self.same_as(model, reference, x, y)
            self.assertEqual(agree, 1.0, model.describe())
            self.assertAlmostEqual(ours.accuracy, theirs, places=10, msg=model.describe())

    def test_riemannian_steps_predict_as_pyriemann_does(self):
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        x, y = trials()
        ours, theirs, agree = self.same_as(
            learn.Model().then("covariance", estimator="oas").then("tangent").then("logistic"),
            make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(max_iter=2000)),
            x, y)
        self.assertEqual(agree, 1.0)
        self.assertGreater(ours.accuracy, 0.85)
        ours, theirs, agree = self.same_as(learn.Model().then("covariance").then("mdm"),
                                           make_pipeline(Covariances("oas"), MDM(metric="riemann")), x, y)
        self.assertEqual(agree, 1.0)

    def test_tuning_is_nested_and_matches_a_grid_search(self):
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.svm import SVC
        x, y = separable(seed=4)
        inner = StratifiedKFold(3, shuffle=True, random_state=42)
        reference = GridSearchCV(SVC(kernel="linear", gamma="scale"), {"C": learn.C_GRID}, cv=inner)
        ours, theirs, agree = self.same_as(learn.Model().then("svm", tune=True), reference, x, y)
        self.assertEqual(agree, 1.0)
        self.assertEqual(len(ours.chosen), 5)
        self.assertIn("C = ", ours.tuning_text())

    def test_the_values_folds_and_score_of_a_search_are_the_users(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.svm import SVC
        x, y = separable(seed=4)
        inner = StratifiedKFold(4, shuffle=True, random_state=42)
        values = {"C": [0.5, 2.0, 8.0], "gamma": ["scale", 0.05]}
        reference = GridSearchCV(SVC(kernel="rbf"), values, cv=inner, scoring="balanced_accuracy")
        model = learn.Model().then("svm", kernel="rbf", tune=True, C_values=[0.5, 2, 8], gamma_values=["scale", "0.05"],
                                   inner_folds=4, tune_score=learn.BALANCED)
        ours, theirs, agree = self.same_as(model, reference, x, y)
        self.assertEqual(agree, 1.0)
        # overlapping, unequal conditions, where the score decides: every fold picks what GridSearchCV picks
        rng = np.random.default_rng(8)
        x = np.r_[rng.normal(0, 1, (60, 4)), rng.normal(0.6, 1, (20, 4))]
        y = np.repeat(["a", "b"], [60, 20])
        folds = learn.stratified_folds(y, 5, 3)
        for score, name in ((learn.BALANCED, "balanced_accuracy"), (learn.ACCURACY, "accuracy")):
            tuned = learn.cross_validate(x, y, learn.Model().then("svm", kernel="rbf", tune=True, C_values=[0.5, 2, 8],
                                                                  gamma_values=["scale", "0.05"], inner_folds=4,
                                                                  tune_score=score), folds=folds)
            for test, chosen in zip(folds, tuned.chosen):
                train = np.setdiff1d(np.arange(len(y)), test)
                search = GridSearchCV(SVC(kernel="rbf"), values, cv=inner, scoring=name).fit(x[train], y[train])
                self.assertEqual(chosen, search.best_params_, score)
        # not tuned: the forest's depth and leaf, the SVM's gamma, are taken as given
        forest = learn.Model().then("forest", trees=20, depth=3, leaf=4).build(1)[-1].estimator
        self.assertEqual((forest.max_depth, forest.min_samples_leaf), (3, 4))
        self.assertIsNone(learn.Model().then("forest").build(1)[-1].estimator.max_depth)
        self.assertEqual(learn.Model().then("svm", kernel="rbf", gamma="0.2").build(1)[-1].estimator.gamma, 0.2)
        grid = learn.Model().then("forest", tune=True, depth_values=["none", "4"], leaf_values=[2.0]).build(1)[-1].grid
        self.assertEqual(grid, {"max_depth": [None, 4], "min_samples_leaf": [2]})
        self.assertIsInstance(RandomForestClassifier(**{k: v[0] for k, v in grid.items()}), RandomForestClassifier)
        for bad, reason in (({"C_values": []}, "at least one"), ({"C_values": ["x"]}, "not a number"),
                            ({"k_values": [2.5]}, "whole number"), ({"inner_folds": 1}, "2 inner folds")):
            kind = "knn" if "k_values" in bad else "svm"
            with self.assertRaisesRegex(ValueError, reason):
                learn.Model().then(kind, tune=True, **bad).build(1)

    def test_tuning_on_noise_stays_at_chance(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(160, 20)), np.repeat(["a", "b"], 80)
        scores = learn.cross_validate(x, y, learn.Model().then("standardize").then("svm", kernel="rbf", tune=True))
        self.assertFalse(scores.better_than_random)

    def test_feature_selection_happens_inside_the_folds(self):
        rng = np.random.default_rng(6)
        x, y = rng.normal(size=(60, 2000)), np.repeat(["a", "b"], 30)
        # Selecting the "best" of 2000 noise features on all trials first would score far above chance.
        scores = learn.cross_validate(x, y, learn.Model().then("select", k=10).then("lda"))
        self.assertFalse(scores.better_than_random)

    def test_model_order_is_checked(self):
        with self.assertRaisesRegex(ValueError, "after the classifier"):
            learn.Model().then("svm").then("standardize")
        with self.assertRaisesRegex(ValueError, "end with a classifier"):
            learn.Model().then("covariance").build()


class AugmentTests(unittest.TestCase):
    def test_copies_are_made_of_training_trials_only(self):
        x, y = trials()
        augment = learn.Augment(copies=3, seed=1)
        bigger, labels = augment.augment(x, y)
        self.assertEqual(len(bigger), 4 * len(x))
        self.assertEqual(list(labels[:len(y)]), list(y))
        np.testing.assert_array_equal(bigger[:len(x)], x)                 # the originals are kept as they are
        self.assertFalse(np.allclose(bigger[len(x):2 * len(x)], x))       # the copies differ
        self.assertIs(augment.transform(x), x)                            # test trials pass untouched

    def test_test_trials_are_never_augmented(self):
        seen = []

        class Spy(learn.LDA):
            def predict(self, x):
                seen.append(len(x))
                return super().predict(x)

        x, y = separable()
        learn.STEPS["spy"] = lambda seed=42, **p: Spy()
        learn.CLASSIFIERS = learn.CLASSIFIERS + ("spy",)
        try:
            folds = learn.stratified_folds(y, 4, 0)
            learn.cross_validate(x, y, learn.Model().then("augment", copies=5).then("spy"), folds=folds)
        finally:
            del learn.STEPS["spy"]
            learn.CLASSIFIERS = tuple(c for c in learn.CLASSIFIERS if c != "spy")
        self.assertEqual(seen, [len(f) for f in folds])                   # each fold tested on its own trials only

    def test_augmenting_noise_does_not_make_it_look_learnable(self):
        rng = np.random.default_rng(9)
        x, y = rng.normal(size=(120, 6, 100)), np.repeat(["a", "b"], 60)
        scores = learn.cross_validate(x, y, learn.Model().then("augment", copies=3).then("csp", n_components=4)
                                      .then("lda"))
        self.assertFalse(scores.better_than_random)

    def test_features_are_augmented_without_a_time_shift(self):
        x, y = separable()
        bigger, _ = learn.Augment(copies=1, shift=0.2).augment(x, y)
        self.assertEqual(bigger.shape, (2 * len(x), x.shape[1]))
        with self.assertRaisesRegex(ValueError, "1 to 20"):
            learn.Augment(copies=0)


class SplitTests(unittest.TestCase):
    def test_repeated_k_fold_runs_every_round_and_reports_the_first(self):
        x, y = separable()
        scores = learn.cross_validate(x, y, learn.Model().then("lda"), 4, scheme=learn.REPEATED, repeats=3)
        self.assertEqual(len(scores.fold_accuracy), 12)
        self.assertEqual(scores.confusion.sum(), len(y))
        self.assertEqual(len(scores.predictions), len(y))
        self.assertEqual(scores.split_text(), "3 × 4-fold")

    def test_chronological_split_tests_the_last_trials(self):
        x, y = separable()
        order = np.random.default_rng(0).permutation(len(y))
        x, y = x[order], y[order]
        scores = learn.cross_validate(x, y, learn.Model().then("lda"), scheme=learn.CHRONOLOGICAL,
                                      train_fraction=0.75)
        tested = sorted(int(row[0]) for row in scores.predictions)
        self.assertEqual(tested, list(range(91, 121)))
        self.assertAlmostEqual(scores.random_limit, learn.random_interval(30, 2)[1])

    def test_chronological_split_refuses_blocks_of_one_condition(self):
        x, y = separable()          # all "a" first, then all "b"
        with self.assertRaisesRegex(ValueError, "one block per condition"):
            learn.cross_validate(x, y, learn.Model().then("lda"), scheme=learn.CHRONOLOGICAL)

    def test_permutation_test_separates_signal_from_noise(self):
        x, y = separable(n=80, shift=1.2)
        model = learn.Model().then("lda")
        real = learn.cross_validate(x, y, model, 4, permutations=49)
        self.assertEqual(real.permutations, 49)
        self.assertAlmostEqual(real.permutation_p, 1 / 50)      # no shuffle did as well
        noise = learn.cross_validate(np.random.default_rng(3).normal(size=x.shape), y, model, 4, permutations=49)
        self.assertGreater(noise.permutation_p, 0.05)
        again = learn.cross_validate(x, y, model, 4, permutations=49)
        self.assertEqual(again.permutation_p, real.permutation_p)   # same seed, same shuffles
        self.assertIn("permutation test p = ", real.random_text())

    def test_balanced_accuracy_is_fair_to_the_small_condition(self):
        y = np.array(["a"] * 90 + ["b"] * 10)
        x = np.random.default_rng(1).normal(size=(100, 3))
        scores = learn.cross_validate(x, y, learn.Model().then("knn", k=15), 5)
        self.assertGreater(scores.accuracy, scores.balanced_accuracy)
        self.assertAlmostEqual(scores.chance, 0.9)


class IcaAndBaselineTests(unittest.TestCase):
    def test_ica_takes_a_blink_out_of_the_frontal_channels(self):
        fs, seconds = 250.0, 60
        rng = np.random.default_rng(7)
        n = int(fs * seconds)
        t = np.arange(n) / fs
        names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz",
                 "P4", "P8", "O1", "Oz", "O2"]
        sources = rng.normal(size=(len(names), n)) * 5
        sources[0] = 10 * np.sin(2 * np.pi * 10 * t)
        blink = np.zeros(n)
        for onset in rng.choice(n - 100, 30, replace=False):
            blink[onset:onset + 100] += 150 * np.hanning(100)
        mixing = rng.normal(size=(len(names), len(names))) * 0.3 + np.eye(len(names))
        data = mixing @ sources
        weights = np.array([1.0, 0.9, 0.5, 0.5, 0.45, 0.45, 0.5] + [0.2] * 5 + [0.05] * 5 + [0.02] * 3)
        data += weights[:, None] * blink
        recording = Recording(data, fs, names)
        out = run("ica", {"raw": recording}, components=len(names) - 1)
        ica, cleaned = out["ica"], out["raw"].data
        self.assertTrue(ica.found, "the blink component was not found")
        before = abs(np.corrcoef(data[0], blink)[0, 1])
        after = abs(np.corrcoef(cleaned[0], blink)[0, 1])
        self.assertGreater(before, 0.8)
        self.assertLess(after, 0.3)
        self.assertLess(abs(np.corrcoef(cleaned[-1], data[-1])[0, 1] - 1), 0.05)   # O2 barely touched

    def test_ica_refuses_component_numbers_it_does_not_have(self):
        recording = Recording(np.random.default_rng(8).normal(size=(4, 5000)), 250.0, ["C3", "Cz", "C4", "Pz"])
        with self.assertRaisesRegex(ValueError, "no component 9"):
            run("ica", {"raw": recording}, components=3, exclude=["9"], eog_channels=[])

    def test_baseline_removes_each_trials_mean_in_the_window(self):
        recording = lateralised(trials=6)
        epochs = run("epochs", {"raw": recording}, tmin=-1.0, tmax=3.0)["epochs"]
        shifted = epochs.with_data(epochs.data + 40.0)
        out = run("baseline", {"epochs": shifted}, start=-1.0, end=0.0)["epochs"]
        window = (out.times >= -1.0) & (out.times < 0.0)
        np.testing.assert_allclose(out.data[:, :, window].mean(axis=2), 0, atol=1e-9)
        with self.assertRaisesRegex(ValueError, "inside the trials"):
            run("baseline", {"epochs": shifted}, start=5.0, end=6.0)


if __name__ == "__main__":
    unittest.main()
