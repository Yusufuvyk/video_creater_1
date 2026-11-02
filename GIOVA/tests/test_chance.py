"""How often "better than chance" is said of pure noise: a test's false positive rate, measured on many datasets.

One noise dataset scoring at chance (test_pipeline, test_models) shows little: a verdict must be wrong at most as
often as it claims, over many. Here 200 noise datasets, each cross-validated, are put through the permutation test
(Ojala & Garriga, 2010): under the null hypothesis its p value is at most alpha with probability at most alpha,
whatever the cross-validation does. Noirhomme et al. (2014, NeuroImage: Clinical 4, 687-694) found it unaffected
by the cross-validation scheme, unlike the binomial test, whose assumption of independent test trials k-fold
cross-validation breaks.

A harness that could not fail would prove nothing: the positive control is a deliberately lenient rule (accuracy
above chance by one binomial standard deviation), which the same bound must catch.

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

import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402

from eeg.pipeline import learn  # noqa: E402

DATASETS = 200
TRIALS = 20
PERMUTATIONS = 19          # p values 1/20, 2/20, ... 1


def at_most(alpha, n=DATASETS):
    """The largest share of n datasets a rate of alpha reaches by chance, 999 times in 1000."""
    return binom.ppf(0.999, n, alpha) / n


class NoiseRuns:
    """Cross-validated LDA with a permutation test, on DATASETS noise datasets: computed once for the class."""
    p = accuracy = None

    @classmethod
    def get(cls):
        if cls.p is None:
            rng = np.random.default_rng(2014)
            labels = np.repeat(["a", "b"], TRIALS // 2)
            model = learn.Model().then("lda")
            runs = [learn.cross_validate(rng.normal(size=(TRIALS, 2)), labels, model, 5, seed=k,
                                         permutations=PERMUTATIONS, limits=False) for k in range(DATASETS)]
            cls.p = np.array([r.permutation_p for r in runs])
            cls.accuracy = np.array([r.accuracy for r in runs])
        return cls


class FalsePositiveTests(unittest.TestCase):
    def test_the_permutation_test_is_wrong_at_most_as_often_as_it_says(self):
        runs = NoiseRuns.get()
        for alpha in (0.05, 0.25, 0.5):
            with self.subTest(alpha=alpha):
                self.assertLessEqual(np.mean(runs.p <= alpha + 1e-9), at_most(alpha))

    def test_the_bound_catches_a_lenient_rule(self):
        runs = NoiseRuns.get()
        lenient = np.mean(runs.accuracy > 0.5 + np.sqrt(0.25 / TRIALS))       # nominally 16%, as a z of 1
        self.assertGreater(lenient, at_most(0.05))

    def test_p_is_never_below_one_over_the_shuffles_plus_one(self):
        self.assertGreaterEqual(NoiseRuns.get().p.min(), 1 / (PERMUTATIONS + 1))


class RandomLimitTests(unittest.TestCase):
    """The random limit under its own assumption, trials tested independently: exact for two conditions."""

    def test_a_random_classifier_passes_it_at_most_alpha_half_the_time(self):
        for alpha in (0.05, 0.01):
            for n in range(4, 401):
                limit = learn.random_interval(n, 2, alpha)[1]
                correct = round(limit * n)                       # the limit is an accuracy one can reach
                self.assertAlmostEqual(limit * n, correct, places=9)
                self.assertLessEqual(binom.sf(correct, n, 0.5), alpha / 2, (alpha, n))
                # and the lowest such: one correct trial fewer would be passed too often
                self.assertGreater(binom.sf(correct - 1, n, 0.5), alpha / 2, (alpha, n))

    def test_an_accuracy_at_the_limit_is_not_past_it(self):
        # 40 of 60 is the alpha 0.01 limit for 60 trials; the mean of five folds' accuracies (8, 5, 9, 11 and 7
        # correct of 12) is 40/60 too, but comes out a last digit larger, which read "clearly above chance"
        folds = [8 / 12, 5 / 12, 9 / 12, 11 / 12, 7 / 12]
        limit = learn.random_interval(60, 2, 0.01)[1]
        self.assertEqual(limit, 40 / 60)
        self.assertGreater(np.mean(folds), limit)
        self.assertFalse(learn.above(np.mean(folds), limit))
        self.assertTrue(learn.above(41 / 60, limit))

    def test_the_papers_normal_approximation_was_passed_too_often(self):
        # Müller-Putz et al.'s 59.61% for 100 trials let 60 correct through: 2.8% of the time, above alpha/2
        self.assertGreater(binom.sf(59, 100, 0.5), 0.025)
        self.assertEqual(learn.random_interval(100, 2)[1], 0.60)

    def test_a_verdict_on_the_limit_alone_says_it_is_approximate(self):
        x, y = np.random.default_rng(1).normal(size=(40, 2)), np.repeat(["a", "b"], 20)
        alone = learn.cross_validate(x, y, learn.Model().then("lda"))
        self.assertIn(learn.APPROXIMATE, alone.random_text())
        tested = learn.cross_validate(x, y, learn.Model().then("lda"), permutations=9)
        self.assertNotIn(learn.APPROXIMATE, tested.random_text())


if __name__ == "__main__":
    unittest.main()
