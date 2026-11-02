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


if __name__ == "__main__":
    unittest.main()
