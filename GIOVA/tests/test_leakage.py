"""No learning step learns from the test trials (data leakage; Kapoor & Narayanan, 2023, Patterns 4(9), 100804).

Leakage most often enters through preprocessing or feature selection fitted on the training and test data together
(their L1.2 and L1.3). A step fitted on the training trials alone treats each test trial on its own: what it
predicts for one test trial cannot depend on which other trials are tested beside it. So for every learning step,
in a chain that runs it, the predictions and scores for some test trials must stay exactly the same when the
other test trials are replaced by something else entirely, and when each is tested alone.

A check that cannot fail proves nothing: the positive control is a step fitted on what it transforms (the test
trials, at test time), which the same check must catch. And every step the Workbench offers must be in a chain
here, so a step added later is checked too.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eeg.pipeline import learn  # noqa: E402

SEED = 7
KEPT = 4          # test trials whose predictions are watched; the rest are swapped


def data(kind, n=40, n_test=10, seed=0):
    """(training trials, their labels, test trials): two conditions that differ a little, as real ones do.

    trials: (n, 4 channels, 64 samples), condition b with more power on the first channel; bands: the same for
    two bands of a filter bank (8 channels); features: (n, 6), condition b shifted on the first two.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(["a", "b"] * ((n + n_test) // 2))
    if kind == "features":
        x = rng.normal(size=(labels.size, 6))
        x[labels == "b", :2] += 0.8
    else:
        x = rng.normal(size=(labels.size, 8 if kind == "bands" else 4, 64))
        x[labels == "b", 0] *= 1.6
        if kind == "bands":
            x[labels == "b", 4] *= 1.6
    return x[:n], labels[:n], x[n:]


#: every learning step in a chain that runs it, and the data it takes
CHAINS = {
    "csp": (learn.Model().then("csp", n_components=2).then("lda"), "trials"),
    "augment": (learn.Model().then("augment", copies=2).then("csp", n_components=2).then("lda"), "trials"),
    "trcsp": (learn.Model().then("trcsp", n_filters=1, tune=True, inner_folds=3).then("lda"), "trials"),
    "fbcsp": (learn.Model().then("fbcsp", n_bands=2, n_pairs=1).then("mibif", k=1, pair_size=2).then("nbpw"),
              "bands"),
    "riemann": (learn.Model().then("covariance").then("tangent").then("lda"), "trials"),
    "mdm": (learn.Model().then("covariance").then("mdm"), "trials"),
    "standardize": (learn.Model().then("standardize").then("lda"), "features"),
    "select": (learn.Model().then("select", k=2).then("logistic", tune=True), "features"),
    "svm": (learn.Model().then("standardize").then("svm", kernel="rbf", tune=True), "features"),
    "forest": (learn.Model().then("forest", trees=25), "features"),
    "knn": (learn.Model().then("knn", tune=True), "features"),
}


def changes_with_the_other_test_trials(model, x_train, y_train, x_test):
    """What differs for the first KEPT test trials when the others change, or when each is tested alone."""
    _, _, predicted, scores = learn.fit_and_test(model, SEED, x_train, y_train, x_test)
    other = x_test.copy()
    other[KEPT:] = np.random.default_rng(99).normal(3.0, 4.0, size=other[KEPT:].shape)
    _, _, predicted_other, scores_other = learn.fit_and_test(model, SEED, x_train, y_train, other)
    found = []
    if not (np.array_equal(predicted[:KEPT], predicted_other[:KEPT])
            and np.allclose(scores[:KEPT], scores_other[:KEPT], rtol=1e-9, atol=1e-12)):
        found.append("the other test trials")
    for i in range(2):
        _, _, alone, alone_scores = learn.fit_and_test(model, SEED, x_train, y_train, x_test[i:i + 1])
        if alone[0] != predicted[i] or not np.allclose(alone_scores[0], scores[i], rtol=1e-9, atol=1e-12):
            found.append("testing trial {} alone".format(i + 1))
    return found


class FitsOnWhatItTransforms:
    """Standardises whatever it is given by that batch's own mean and spread: at test time, fitted on the test
    trials. The leak this file looks for, on purpose."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)


class LeakageTests(unittest.TestCase):
    def test_every_step_is_checked(self):
        used = {kind for model, _ in CHAINS.values() for kind, _ in model.steps}
        self.assertEqual(used, set(learn.STEPS), "a learning step with no chain in CHAINS is not checked")

    def test_a_test_trial_is_judged_on_its_own(self):
        for model, kind in CHAINS.values():
            with self.subTest(chain=model.describe()):
                self.assertEqual(changes_with_the_other_test_trials(model, *data(kind)), [])

    def test_the_check_catches_a_step_fitted_on_the_test_trials(self):
        leaky = {"leaky": lambda seed=42, **params: FitsOnWhatItTransforms()}
        with patch.dict(learn.STEPS, leaky):
            model = learn.Model().then("leaky").then("lda")
            found = changes_with_the_other_test_trials(model, *data("features"))
        self.assertIn("the other test trials", found)
        self.assertTrue(any("alone" in f for f in found))

    def test_the_tuning_search_never_sees_the_test_trials(self):
        # the setting a tuned step chooses comes from the training trials alone
        model, kind = CHAINS["svm"]
        x_train, y_train, x_test = data(kind)
        steps, _, _, _ = learn.fit_and_test(model, SEED, x_train, y_train, x_test)
        other_steps, _, _, _ = learn.fit_and_test(model, SEED, x_train, y_train, x_test[::-1] * 5 + 2)
        self.assertEqual([getattr(s, "chosen", None) for s in steps], [getattr(s, "chosen", None) for s in other_steps])
        self.assertTrue(steps[-1].chosen)


if __name__ == "__main__":
    unittest.main()
