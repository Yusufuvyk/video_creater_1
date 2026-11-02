"""Classification: the learning steps a Model chains, and cross-validation.

CSP, standardising and shrinkage LDA are written here in NumPy / SciPy and
checked against MNE and scikit-learn in the tests. SVM, logistic regression,
random forest, kNN, feature selection and the Riemannian steps (covariance,
tangent space, MDM) come from scikit-learn and pyRiemann, imported only when
a model uses them, so a pipeline without them starts no faster or slower.

Nothing is ever fitted outside `cross_validate`. A Model is only a list of
unfitted steps; every fold fits all of them again on its own training trials
and only then touches its test trials. Fitting CSP or a scaler once on all
trials and cross-validating afterwards lets the test trials shape the
features, and the accuracy comes out higher than any new recording would give.

No Qt.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg
from scipy.stats import norm, rankdata


# ----------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------


#: A direction of C_a + C_b with less variance than this share of the largest carries no signal: an average
#: reference leaves one such direction (the channels sum to zero), a re-reference or bridged channels may too.
RANK_TOL = 1e-7


def csp_solve(c_a, c_b):
    """(λ, w) of C_a w = λ (C_a + C_b) w with w'(C_a + C_b) w = 1, in the data's true rank.

    Full rank: the generalised eigenproblem as it stands. Below it, C_a + C_b is not invertible
    and the problem has no unique answer; the directions without variance are dropped first, the
    rest whitened, and C_a solved there - as MNE does with its rank - which gives fewer filters
    than channels (the rank). Raises ValueError when no direction carries signal."""
    composite = c_a + c_b
    scale, basis = linalg.eigh(composite)
    keep = scale > RANK_TOL * scale.max()
    if keep.all():
        return linalg.eigh(c_a, composite)
    if not keep.any():
        raise ValueError("no channel carries signal")
    whiten = basis[:, keep] / np.sqrt(scale[keep])
    values, inner = linalg.eigh(whiten.T @ c_a @ whiten)
    return values, whiten @ inner


class CSP:
    """Common spatial patterns for two conditions (Ramoser et al., 2000).

    Each condition's covariance is taken over all its trials laid end to end;
    the filters solve C_a w = λ (C_a + C_b) w, and those with λ farthest from
    0.5 - most variance for one condition, least for the other - are kept.
    A trial's features are the log of the mean power of each filtered signal.
    After an average reference the channels have one direction less (the rank):
    the filters are found in that rank (`csp_solve`), one fewer than channels.
    """

    def __init__(self, n_components=6, log=True):
        self.n_components = int(n_components)
        self.log = bool(log)

    def fit(self, x, y):
        x = _trials(x, "CSP")
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("CSP separates two conditions; these trials have {}".format(classes.size))
        if not 1 <= self.n_components <= x.shape[1]:
            raise ValueError("CSP components must be between 1 and the number of channels ({})".format(x.shape[1]))
        covs = [np.cov(x[y == c].transpose(1, 0, 2).reshape(x.shape[1], -1)) for c in classes]
        try:
            values, vectors = csp_solve(covs[0], covs[1])
        except (linalg.LinAlgError, ValueError) as exc:
            raise ValueError("CSP found no usable signal in these channels (flat, or not finite)") from exc
        #: independent directions among the channels: one fewer than channels after an average reference
        self.rank = values.size
        if self.n_components > self.rank:
            raise ValueError("CSP components must be at most the channels' rank ({} of {} channels: an average "
                             "reference or channels that are sums of others remove some)".format(self.rank,
                                                                                                 x.shape[1]))
        order = np.argsort(np.abs(values - 0.5))[::-1]
        self.eigenvalues = values[order]
        self.filters = vectors[:, order].T
        #: columns are the scalp patterns of the filters (Haufe et al., 2014)
        self.patterns = linalg.pinv(self.filters)
        return self

    def transform(self, x):
        x = _trials(x, "CSP")
        signals = np.einsum("fc,nct->nft", self.filters[:self.n_components], x)
        power = (signals ** 2).mean(axis=2)
        return np.log(power) if self.log else power


class TRCSP:
    """Tikhonov-regularised CSP (Lotte & Guan, 2011, IEEE TBME 58:355), two conditions.

    The filters of condition a maximise w'C_a w / (w'C_b w + alpha w'w): the leading eigenvectors
    of (C_b + alpha I)^-1 C_a, and the same with a and b swapped for b's. The penalty keeps the
    weights small, which keeps a filter from resting on a few noisy channels. alpha 0 is plain CSP.
    Each condition's covariance is the mean of its trials' covariances divided by their trace, so
    alpha does not depend on the signal's unit or gain. Features: log variance of each filtered signal.

    scikit-learn style (get_params / set_params) so Tune can search alpha.
    """

    def __init__(self, n_filters=3, alpha=1e-5):
        self.n_filters, self.alpha = int(n_filters), float(alpha)

    def get_params(self, deep=True):
        return {"n_filters": self.n_filters, "alpha": self.alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, int(value) if key == "n_filters" else float(value))
        return self

    def fit(self, x, y):
        x, y = _trials(x, "TRCSP"), np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("TRCSP separates two conditions; these trials have {}".format(self.classes_.size))
        if not 1 <= 2 * self.n_filters <= x.shape[1]:
            raise ValueError("TRCSP: 2 × filters per condition must be at most the channels ({})".format(x.shape[1]))
        covs = [normalised_covariance(x[y == c]) for c in self.classes_]
        eye = np.eye(x.shape[1])
        filters = []
        for own, other in ((covs[0], covs[1]), (covs[1], covs[0])):
            try:
                values, vectors = linalg.eigh(own, other + self.alpha * eye)
            except (linalg.LinAlgError, ValueError) as exc:
                raise ValueError("TRCSP could not separate these channels; raise alpha or use fewer "
                                 "channels") from exc
            order = np.argsort(values)[::-1][:self.n_filters]
            filters.append(vectors[:, order].T)
        self.filters = np.vstack(filters)
        self.filters /= np.linalg.norm(self.filters, axis=1, keepdims=True)
        return self

    def transform(self, x):
        signals = np.einsum("fc,nct->nft", self.filters, _trials(x, "TRCSP"))
        return np.log(signals.var(axis=2))


def normalised_covariance(trials):
    """Mean over trials of X X' / trace(X X'), each trial's channels centred."""
    x = trials - trials.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", x, x)
    return (covs / np.trace(covs, axis1=1, axis2=2)[:, None, None]).mean(axis=0)


#: The alphas Lotte & Guan (2011) searched: 10^-10 to 10^-1.
TRCSP_ALPHAS = [10.0 ** k for k in range(-10, 0)]


def _trcsp(n_filters=3, alpha=1e-5, tune=True, inner_folds=10, tune_score="accuracy", seed=42):
    return Library(TRCSP(n_filters, alpha), {"alpha": TRCSP_ALPHAS} if tune else None, seed, "TRCSP",
                   inner_folds=inner_folds, scoring=tune_score)


class FBCSP:
    """Filter bank CSP (Ang et al., 2012, Front Neurosci 6:39), two conditions.

    The trials' channels come band after band, as the Filter bank block lays them out:
    `n_bands` equal groups. In each band CSP is fitted on the training trials and its first and
    last `n_pairs` filters kept; a trial's features are the log of each filtered signal's
    variance over the sum of the variances of that band's kept filters. Features come band by
    band, each band's as [first n_pairs, last n_pairs], so feature j of a band pairs with
    2 n_pairs - 1 - j: the pairing MIBIF adds partners by.
    """

    def __init__(self, n_bands=9, n_pairs=2):
        self.n_bands, self.n_pairs = int(n_bands), int(n_pairs)

    def fit(self, x, y):
        x, y = _trials(x, "FBCSP"), np.asarray(y)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("FBCSP separates two conditions; these trials have {}".format(classes.size))
        if self.n_bands < 1 or x.shape[1] % self.n_bands:
            raise ValueError("FBCSP expects the channels of {} bands; these trials have {} channels. Put a "
                             "Filter bank block with {} bands before it".format(self.n_bands, x.shape[1],
                                                                                self.n_bands))
        per = x.shape[1] // self.n_bands
        if not 1 <= 2 * self.n_pairs <= per:
            raise ValueError("FBCSP: 2 × pairs must be at most the channels per band ({})".format(per))
        self.filters = []
        for band in range(self.n_bands):
            part = x[:, band * per:(band + 1) * per]
            covs = [np.cov(part[y == c].transpose(1, 0, 2).reshape(per, -1)) for c in classes]
            try:
                values, vectors = csp_solve(covs[0], covs[1])     # in the band's rank, as CSP
            except (linalg.LinAlgError, ValueError) as exc:
                raise ValueError("FBCSP: band {} has no usable signal (flat, or not finite)".format(band + 1)) from exc
            if 2 * self.n_pairs > values.size:
                raise ValueError("FBCSP: 2 × pairs must be at most the channels' rank per band ({})".format(
                    values.size))
            order = np.argsort(values)[::-1]           # most variance for the first condition first
            keep = np.concatenate([order[:self.n_pairs], order[-self.n_pairs:]])
            self.filters.append(vectors[:, keep].T)
        return self

    def transform(self, x):
        x = _trials(x, "FBCSP")
        per = x.shape[1] // self.n_bands
        out = []
        for band, filters in enumerate(self.filters):
            variance = np.einsum("fc,nct->nft", filters, x[:, band * per:(band + 1) * per]).var(axis=2)
            out.append(np.log(variance / variance.sum(axis=1, keepdims=True)))
        return np.concatenate(out, axis=1)


def silverman(values):
    """The Parzen bandwidth of Ang et al. (2012): (4 / 3n)^(1/5) σ, σ with n - 1."""
    values = np.asarray(values, dtype=float)
    spread = values.std(ddof=1) if values.size > 1 else 0.0
    return (4.0 / (3.0 * values.size)) ** 0.2 * (spread if spread > 0 else 1.0)


def _parzen(points, samples, bandwidth):
    """Gaussian Parzen density of `samples` at `points`."""
    z = (np.asarray(points, dtype=float)[:, None] - np.asarray(samples, dtype=float)[None, :]) / bandwidth
    return np.exp(-0.5 * z ** 2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def mutual_information(feature, labels):
    """I(f; ω) = H(ω) - H(ω | f) in bits, the class densities by Parzen windows (Ang et al., 2012).

    H(ω | f) is averaged over the trials (the paper sums; the ranking is the same).
    """
    feature, labels = np.asarray(feature, dtype=float), np.asarray(labels)
    classes = np.unique(labels)
    priors = np.array([np.mean(labels == c) for c in classes])
    joint = np.array([p * _parzen(feature, feature[labels == c], silverman(feature[labels == c]))
                      for p, c in zip(priors, classes)])                     # (classes, trials)
    posterior = joint / np.maximum(joint.sum(axis=0, keepdims=True), 1e-300)
    conditional = -np.sum(posterior * np.log2(np.maximum(posterior, 1e-300))) / feature.size
    return float(-np.sum(priors * np.log2(priors)) - conditional)


class MIBIF:
    """Mutual information based best individual features (Ang et al., 2012).

    The `k` features with the largest mutual information with the condition, chosen on the training
    trials, then the CSP partner of each that is not chosen already: k to 2k features. `pair_size`
    is the features per band FBCSP gives (2 × its pairs); 0 adds no partners.
    """

    def __init__(self, k=4, pair_size=4):
        self.k, self.pair_size = int(k), int(pair_size)

    def fit(self, x, y):
        x = _rows(x, "MIBIF")
        y = np.asarray(y)
        self.information = np.array([mutual_information(x[:, j], y) for j in range(x.shape[1])])
        best = list(np.argsort(self.information, kind="stable")[::-1][:min(self.k, x.shape[1])])
        chosen = list(best)
        if self.pair_size:
            if x.shape[1] % self.pair_size:
                raise ValueError("MIBIF: {} features do not make bands of {}; give it FBCSP's features"
                                 .format(x.shape[1], self.pair_size))
            for j in best:
                start, local = j - j % self.pair_size, j % self.pair_size
                partner = start + self.pair_size - 1 - local
                if partner not in chosen:
                    chosen.append(partner)
        self.selected = np.array(sorted(chosen))
        return self

    def transform(self, x):
        return _rows(x, "MIBIF")[:, self.selected]


class NBPW:
    """Naive Bayesian Parzen window classifier (Ang et al., 2012).

    P(ω | x) ∝ P(ω) Π_j p(x_j | ω), each p(x_j | ω) a Gaussian Parzen density of that condition's
    training values with the bandwidth `silverman` gives them; combined as sums of logarithms.
    """

    def fit(self, x, y):
        x, y = _rows(x, "NBPW"), np.asarray(y)
        self.classes = np.unique(y)
        if self.classes.size < 2:
            raise ValueError("NBPW needs trials of at least two conditions")
        self.samples = [x[y == c] for c in self.classes]
        self.bandwidths = [np.array([silverman(s[:, j]) for j in range(x.shape[1])]) for s in self.samples]
        self.log_priors = np.log([np.mean(y == c) for c in self.classes])
        return self

    def log_posterior(self, x):
        """(trials, conditions): log P(ω) + Σ_j log p(x_j | ω), up to one constant per trial."""
        x = _rows(x, "NBPW")
        out = np.empty((x.shape[0], self.classes.size))
        for k, (samples, widths) in enumerate(zip(self.samples, self.bandwidths)):
            logs = [np.log(np.maximum(_parzen(x[:, j], samples[:, j], widths[j]), 1e-300))
                    for j in range(x.shape[1])]
            out[:, k] = self.log_priors[k] + np.sum(logs, axis=0)
        return out

    def decision(self, x):
        scores = self.log_posterior(x)
        return scores[:, 1] - scores[:, 0] if self.classes.size == 2 else scores

    def predict(self, x):
        return self.classes[self.log_posterior(x).argmax(axis=1)]


class Augment:
    """More training trials, made from the ones there are: noisy, rescaled, slightly shifted copies.

    Only ever the training trials. `fit_and_test` calls `augment` on the training set of each fold
    and leaves the test trials as they were (`transform` returns them unchanged): a copy of a test
    trial among the training ones would let the model recognise it rather than generalise.

    noise: Gaussian noise, as a share of each channel's (or feature's) standard deviation.
    scale: each copy's amplitude multiplied by a factor from 1 - scale to 1 + scale.
    shift: trials (channels × time) moved by up to this share of their length, the edge sample held.
    """

    def __init__(self, copies=2, noise=0.1, scale=0.1, shift=0.05, seed=42):
        self.copies, self.noise, self.scale, self.shift = int(copies), float(noise), float(scale), float(shift)
        if self.copies < 1 or self.copies > 20:
            raise ValueError("Augment makes 1 to 20 copies of each training trial")
        if min(self.noise, self.scale, self.shift) < 0 or self.scale >= 1 or self.shift > 0.5:
            raise ValueError("Augment: noise and scale from 0, scale below 1, shift at most 0.5")
        self.seed = seed

    def augment(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        spread = x.std(axis=(0, 2), keepdims=True) if x.ndim == 3 else x.std(axis=0, keepdims=True)
        out = [x]
        for _ in range(self.copies):
            z = x.copy()
            if self.shift and x.ndim == 3:
                most = int(round(self.shift * x.shape[-1]))
                for i, k in enumerate(rng.integers(-most, most + 1, size=len(z))):
                    z[i] = _shifted(z[i], int(k))
            if self.scale:
                factor = rng.uniform(1 - self.scale, 1 + self.scale, size=(len(z),) + (1,) * (x.ndim - 1))
                z = z * factor
            if self.noise:
                z = z + rng.normal(size=z.shape) * spread * self.noise
            out.append(z)
        return np.concatenate(out), np.tile(y, self.copies + 1)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x


def _shifted(trial, k):
    """A trial (channels × time) moved k samples later (earlier when negative), the edge sample held."""
    if k == 0:
        return trial
    out = np.empty_like(trial)
    if k > 0:
        out[:, k:], out[:, :k] = trial[:, :-k], trial[:, :1]
    else:
        out[:, :k], out[:, k:] = trial[:, -k:], trial[:, -1:]
    return out


class Standardize:
    """Each feature to mean 0 and standard deviation 1, measured on the training trials."""

    def fit(self, x, y=None):
        x = _rows(x, "Standardize")
        self.mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, x):
        return (_rows(x, "Standardize") - self.mean) / self.scale


class LDA:
    """Linear discriminant analysis with a pooled covariance.

    shrinkage "auto": Ledoit-Wolf (2004) shrinkage of each condition's
    covariance, which keeps the matrix usable with many features and few
    trials; "none": the plain estimate; a number in [0, 1]: that amount.
    Same estimator as scikit-learn's LinearDiscriminantAnalysis(solver="lsqr").
    """

    def __init__(self, shrinkage="auto"):
        if shrinkage not in ("auto", "none") and not 0 <= float(shrinkage) <= 1:
            raise ValueError("LDA shrinkage is 'auto', 'none' or a number from 0 to 1")
        self.shrinkage = shrinkage

    def fit(self, x, y):
        x = _rows(x, "LDA")
        y = np.asarray(y)
        self.classes = np.unique(y)
        if self.classes.size < 2:
            raise ValueError("LDA needs trials of at least two conditions")
        priors = np.array([np.mean(y == c) for c in self.classes])
        means = np.array([x[y == c].mean(axis=0) for c in self.classes])
        covariance = sum(p * _covariance(x[y == c], self.shrinkage) for p, c in zip(priors, self.classes))
        coef = linalg.lstsq(covariance, means.T)[0].T
        intercept = -0.5 * np.sum(means * coef, axis=1) + np.log(priors)
        if self.classes.size == 2:
            coef, intercept = coef[1:] - coef[:1], intercept[1:] - intercept[:1]
        self.coef, self.intercept = coef, intercept
        return self

    def decision(self, x):
        """Binary: one score per trial, positive for the second condition. Else one column per condition."""
        scores = _rows(x, "LDA") @ self.coef.T + self.intercept
        return scores[:, 0] if self.classes.size == 2 else scores

    def predict(self, x):
        scores = self.decision(x)
        return self.classes[(scores > 0).astype(int)] if scores.ndim == 1 else self.classes[scores.argmax(axis=1)]


def _covariance(x, shrinkage):
    """One condition's covariance, as scikit-learn's LDA estimates it (biased)."""
    centred = x - x.mean(axis=0)
    empirical = centred.T @ centred / x.shape[0]
    if shrinkage == "none":
        return empirical
    if shrinkage != "auto":
        return _shrunk(empirical, float(shrinkage))
    # Ledoit-Wolf is applied to standardised features, then scaled back.
    scale = np.sqrt(np.diag(empirical))
    scale = np.where(scale > 0, scale, 1.0)
    z = centred / scale
    shrunk = _shrunk(z.T @ z / z.shape[0], ledoit_wolf_shrinkage(z))
    return scale[:, None] * shrunk * scale[None, :]


def _shrunk(covariance, amount):
    """Towards a multiple of the identity with the same trace."""
    mu = np.trace(covariance) / covariance.shape[0]
    return (1 - amount) * covariance + amount * mu * np.eye(covariance.shape[0])


def ledoit_wolf_shrinkage(x):
    """The Ledoit-Wolf (2004) shrinkage amount for the rows of `x` (trials × features)."""
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    if p == 1:
        return 0.0
    x = x - x.mean(axis=0)
    x2 = x ** 2
    trace = x2.sum(axis=0) / n
    mu = trace.sum() / p
    beta_ = np.sum(x2.T @ x2)
    delta_ = np.sum((x.T @ x) ** 2) / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * trace.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    return 0.0 if beta == 0 else float(beta / delta)


def _trials(x, step):
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError("{} works on trials (channels × time); connect it to Epochs".format(step))
    return x


def _rows(x, step):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("{} works on one row of numbers per trial; put CSP or a feature block before it"
                         .format(step))
    if not np.isfinite(x).all():
        raise ValueError("{} received missing values (NaN): a feature could not be computed for "
                         "some trials".format(step))
    return x


# ----------------------------------------------------------------------
# steps from scikit-learn and pyRiemann
# ----------------------------------------------------------------------

#: Values tried when a classifier's "Tune" is on. Each is searched with a 3-fold
#: cross-validation inside the training trials of every outer fold (nested), so
#: the test trials never take part in choosing them.
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
GAMMA_GRID = ["scale", 0.001, 0.01, 0.1, 1.0]
TREE_GRID = {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5]}
NEIGHBOUR_GRID = [3, 5, 7, 9, 15]
INNER_FOLDS = 3
#: What the inner search keeps the best of: plain accuracy, or the mean recall over the conditions
#: (fair when they have unequal trials, as the results' balanced accuracy).
ACCURACY, BALANCED = "accuracy", "balanced accuracy"
TUNE_SCORES = (ACCURACY, BALANCED)


def search_values(values, what, kind=float, words=()):
    """The values a block's Options give a search, read: numbers as `kind` (int: whole numbers), or one of
    `words` ("scale", "none") as they are. ValueError saying which value does not read."""
    out = []
    for value in values or []:
        text = str(value).strip().lower()
        if text in words:
            out.append(text)
            continue
        try:
            number = float(text)
        except ValueError:
            raise ValueError("{}: '{}' is not a number{}".format(
                what, value, " or " + " / ".join(words) if words else "")) from None
        if not np.isfinite(number) or number <= 0 or (kind is int and number != int(number)):
            raise ValueError("{}: '{}' must be a positive {}".format(
                what, value, "whole number" if kind is int else "number"))
        out.append(kind(number))
    if not out:
        raise ValueError("{}: give at least one value to search".format(what))
    return list(dict.fromkeys(out))


def _gamma(value):
    """An SVM's gamma: "scale" (1 / (features × variance), scikit-learn's default) or a positive number."""
    return search_values([value], "Gamma", words=("scale",))[0]


def _depth(value):
    """A tree depth: 0 or "none" for trees grown until their leaves are pure, else a whole number."""
    return None if str(value).strip().lower() in ("0", "none", "") else search_values([value], "Depth", int)[0]


def svm_grid(kernel="linear", C_values=C_GRID, gamma_values=GAMMA_GRID):
    grid = {"C": search_values(C_values, "C values")}
    if kernel == "rbf":
        grid["gamma"] = search_values(gamma_values, "Gamma values", words=("scale",))
    return grid


def forest_grid(depth_values=("none", 5, 10), leaf_values=(1, 3, 5)):
    return {"max_depth": [None if v == "none" else v
                          for v in search_values(depth_values, "Depth values", int, words=("none",))],
            "min_samples_leaf": search_values(leaf_values, "Leaf values", int)}


class Library:
    """A scikit-learn style estimator as one step: fit, then transform or predict.

    A grid is the settings "Tune" tries. `fit_and_test` searches it with the whole
    chain refitted in every inner fold, sets the best on the estimator and records
    it in `chosen`, for the report; this step itself then fits only that setting.
    `inner_folds` and `scoring` shape that search.
    """

    def __init__(self, estimator, grid=None, seed=42, name="", inner_folds=INNER_FOLDS, scoring=ACCURACY):
        self.estimator, self.grid, self.seed, self.name = estimator, grid, seed, name
        self.chosen = None
        if grid and int(inner_folds) < 2:
            raise ValueError("Tuning needs at least 2 inner folds")
        if scoring not in TUNE_SCORES:
            raise ValueError("Tune by {} or {}".format(*TUNE_SCORES))
        #: folds of the inner search when tuned, and what it keeps the best of
        self.inner_folds, self.scoring = int(inner_folds), scoring

    def fit(self, x, y):
        if self.grid:
            raise ValueError("A tuned step is searched by fit_and_test, with the steps before it")
        self.fitted = self.estimator.fit(x, np.asarray(y))
        if hasattr(self.fitted, "classes_"):
            self.classes = np.asarray(self.fitted.classes_)
        return self

    def transform(self, x):
        return self.fitted.transform(x)

    def predict(self, x):
        return self.fitted.predict(x)

    def decision(self, x):
        """Binary: one score per trial, higher for the second condition. Else one column per condition."""
        if hasattr(self.fitted, "decision_function"):
            scores = self.fitted.decision_function(x)
        else:
            scores = self.fitted.predict_proba(x)
            if scores.shape[1] == 2:
                scores = scores[:, 1] - 0.5
        return np.asarray(scores, dtype=float)


def _svm(kernel="linear", C=1.0, tune=False, gamma="scale", C_values=C_GRID, gamma_values=GAMMA_GRID,
         inner_folds=INNER_FOLDS, tune_score=ACCURACY, seed=42):
    from sklearn.svm import SVC
    grid = svm_grid(kernel, C_values, gamma_values) if tune else None
    # gamma is read only where it counts: a linear kernel has none
    return Library(SVC(kernel=kernel, C=float(C), gamma=_gamma(gamma) if kernel == "rbf" else "scale"), grid, seed,
                   inner_folds=inner_folds,
                   scoring=tune_score)


def _logistic(penalty="l2", C=1.0, tune=False, C_values=C_GRID, inner_folds=INNER_FOLDS, tune_score=ACCURACY,
              seed=42):
    from sklearn.linear_model import LogisticRegression
    # l1_ratio 1 is the lasso (few features kept), 0 the ridge (all shrunk); scikit-learn 1.8+ wording.
    # liblinear shuffles with random_state: unseeded, two runs of the same folds could differ
    model = LogisticRegression(C=float(C), l1_ratio=1.0 if penalty == "l1" else 0.0,
                               solver="liblinear" if penalty == "l1" else "lbfgs", max_iter=2000,
                               random_state=seed)
    grid = {"C": search_values(C_values, "C values")} if tune else None
    return Library(model, grid, seed, inner_folds=inner_folds, scoring=tune_score)


def _forest(trees=200, tune=False, depth=0, leaf=1, depth_values=("none", 5, 10), leaf_values=(1, 3, 5),
            inner_folds=INNER_FOLDS, tune_score=ACCURACY, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=int(trees), max_depth=_depth(depth),
                                   min_samples_leaf=search_values([leaf], "Leaf size", int)[0],
                                   random_state=seed, n_jobs=1)
    return Library(model, forest_grid(depth_values, leaf_values) if tune else None, seed, inner_folds=inner_folds,
                   scoring=tune_score)


def _neighbours(k=5, tune=False, k_values=NEIGHBOUR_GRID, inner_folds=INNER_FOLDS, tune_score=ACCURACY, seed=42):
    from sklearn.neighbors import KNeighborsClassifier
    grid = {"n_neighbors": search_values(k_values, "Neighbour values", int)} if tune else None
    return Library(KNeighborsClassifier(n_neighbors=int(k)), grid, seed, inner_folds=inner_folds,
                   scoring=tune_score)


class SelectBest(Library):
    """The k features that best separate the conditions (ANOVA F), chosen on the training trials."""

    def __init__(self, k=10, seed=42):
        from sklearn.feature_selection import SelectKBest, f_classif
        super().__init__(SelectKBest(f_classif, k="all"), None, seed)
        self.k = int(k)

    def fit(self, x, y):
        self.estimator.set_params(k=min(self.k, np.asarray(x).shape[1]))
        return super().fit(_rows(x, "Select features"), y)


def _covariances(estimator="oas", seed=42):
    from pyriemann.estimation import Covariances
    return Library(Covariances(estimator=estimator), None, seed)


def _tangent(seed=42):
    from pyriemann.tangentspace import TangentSpace
    return Library(TangentSpace(metric="riemann"), None, seed)


def _mdm(seed=42):
    from pyriemann.classification import MDM
    return Library(MDM(metric="riemann"), None, seed)


# ----------------------------------------------------------------------
# a model: steps in order, unfitted
# ----------------------------------------------------------------------

#: kind -> factory(seed=..., **params)
STEPS = {
    "augment": lambda seed=42, **p: Augment(seed=seed, **p),
    "csp": lambda seed=42, **p: CSP(**p),
    "trcsp": _trcsp,
    "fbcsp": lambda seed=42, **p: FBCSP(**p),
    "mibif": lambda seed=42, **p: MIBIF(**p),
    "nbpw": lambda seed=42, **p: NBPW(),
    "standardize": lambda seed=42, **p: Standardize(),
    "select": lambda seed=42, **p: SelectBest(seed=seed, **p),
    "covariance": _covariances,
    "tangent": _tangent,
    "lda": lambda seed=42, **p: LDA(**p),
    "svm": _svm,
    "logistic": _logistic,
    "forest": _forest,
    "knn": _neighbours,
    "mdm": _mdm,
}
CLASSIFIERS = ("lda", "svm", "logistic", "forest", "knn", "mdm", "nbpw")
NAMES = {"augment": "Augment", "csp": "CSP", "trcsp": "TRCSP", "standardize": "Standardize", "select": "Select", "covariance": "Covariance",
         "tangent": "Tangent space", "lda": "LDA", "svm": "SVM", "logistic": "Logistic regression",
         "forest": "Random forest", "knn": "kNN", "mdm": "MDM", "fbcsp": "FBCSP", "mibif": "MIBIF", "nbpw": "NBPW"}


class Model:
    """Unfitted steps in order, e.g. CSP then LDA. Immutable: `then` returns a new one."""

    def __init__(self, steps=()):
        self.steps = tuple((kind, dict(params)) for kind, params in steps)

    def then(self, kind, **params):
        if kind not in STEPS:
            raise ValueError("Unknown learning step '{}'".format(kind))
        if self.steps and self.steps[-1][0] in CLASSIFIERS:
            raise ValueError("Nothing can come after the classifier")
        return Model(self.steps + ((kind, params),))

    def build(self, seed=42):
        if not self.steps or self.steps[-1][0] not in CLASSIFIERS:
            raise ValueError("The model must end with a classifier (LDA, SVM, logistic regression, random forest, "
                             "kNN, MDM or NBPW)")
        return [STEPS[kind](seed=seed, **params) for kind, params in self.steps]

    def describe(self):
        def name(kind, params):
            text = NAMES.get(kind, kind)
            if kind == "svm":
                text += " ({})".format(params.get("kernel", "linear"))
            return text + (" tuned" if params.get("tune") else "")
        return " → ".join(name(k, p) for k, p in self.steps) or "empty"


# ----------------------------------------------------------------------
# cross-validation
# ----------------------------------------------------------------------

#: How trials are split into training and test.
KFOLD, REPEATED, CHRONOLOGICAL = "k-fold", "repeated k-fold", "chronological"
SCHEMES = (KFOLD, REPEATED, CHRONOLOGICAL)
#: Not a split of one recording: trained on the other recordings, tested on this one.
ACROSS = "leave one recording out"


def stratified_folds(labels, n_folds=5, seed=42):
    """Test indices of each fold: every condition spread evenly, order shuffled by `seed`."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    position = 0
    for c in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == c))
        for i, index in enumerate(members):
            folds[(position + i) % n_folds].append(index)
        position += members.size
    return [np.sort(np.asarray(f, dtype=int)) for f in folds]


def split(labels, scheme=KFOLD, n_folds=5, seed=42, repeats=5, train_fraction=0.8):
    """[(repeat, test indices)] for a scheme.

    chronological: the first `train_fraction` of the trials in recording order train,
    the rest test - how a BCI meets new data, with no shuffling across time.
    """
    labels = np.asarray(labels)
    if scheme == CHRONOLOGICAL:
        if not 0.1 <= train_fraction <= 0.95:
            raise ValueError("The training share must be between 0.1 and 0.95")
        first = int(round(train_fraction * labels.size))
        test = np.arange(first, labels.size)
        if np.unique(labels[:first]).size < np.unique(labels).size:
            raise ValueError("The first {:.0%} of the trials miss a condition; a chronological split cannot "
                             "train on it".format(train_fraction))
        if np.unique(labels[first:]).size < np.unique(labels).size:
            raise ValueError("The last {:.0%} of the trials hold only some conditions (a file with one "
                             "block per condition?); a chronological split would not test them all"
                             .format(1 - train_fraction))
        return [(0, test)]
    smallest = min(np.unique(labels, return_counts=True)[1])
    if not 2 <= int(n_folds) <= smallest:
        raise ValueError("Folds must be at least 2 and at most the smallest condition's trials ({})".format(smallest))
    rounds = int(repeats) if scheme == REPEATED else 1
    if rounds < 1:
        raise ValueError("Repeats must be at least 1")
    return [(r, fold) for r in range(rounds) for fold in stratified_folds(labels, int(n_folds), seed + r)]


@dataclass
class Scores:
    classes: list
    #: condition -> number of trials
    counts: dict
    fold_accuracy: list
    #: area under the ROC curve per fold; empty for more than two conditions
    fold_auc: list
    #: rows: true condition, columns: predicted, over the first round of folds
    confusion: np.ndarray
    #: (trial id, true, predicted, score) for every tested trial, from the first round of folds
    predictions: list
    model: str
    n_folds: int
    seed: int
    #: accuracy of always guessing the largest condition
    chance: float = 0.0
    #: the upper edge of what a random classifier reaches with this many tested trials,
    #: at alpha 0.05 and 0.01 (Müller-Putz et al., 2008); above it is better than random
    random_limit: float = 0.0
    random_limit_01: float = 0.0
    notes: list = field(default_factory=list)
    #: mean recall over the conditions, per fold: fair when they have unequal trials
    fold_balanced: list = field(default_factory=list)
    scheme: str = KFOLD
    repeats: int = 1
    #: per fold, the setting a tuned step chose ({} when nothing was tuned)
    chosen: list = field(default_factory=list)
    #: permutation test: how many label shuffles were run (0: none) and the p value
    permutations: int = 0
    permutation_p: float = float("nan")
    #: when the tested conditions have unequal trials: the upper edge of a random classifier's
    #: balanced accuracy for them (alpha 0.05, by simulation); NaN when they are equal
    balanced_limit: float = float("nan")
    #: Cohen's kappa and macro F1 per fold, averaged as accuracy is (`cohen_kappa`, `macro_f1`)
    fold_kappa: list = field(default_factory=list)
    fold_f1: list = field(default_factory=list)

    @property
    def kappa(self):
        return float(np.nanmean(self.fold_kappa)) if np.isfinite(self.fold_kappa).any() else float("nan")

    @property
    def f1(self):
        return float(np.nanmean(self.fold_f1)) if np.isfinite(self.fold_f1).any() else float("nan")

    @property
    def unequal(self):
        """Unequal tested conditions: accuracy alone can then be high by always guessing the largest."""
        return bool(np.isfinite(self.balanced_limit))

    @property
    def accuracy(self):
        return float(np.mean(self.fold_accuracy))

    @property
    def accuracy_sd(self):
        return float(np.std(self.fold_accuracy, ddof=1)) if len(self.fold_accuracy) > 1 else 0.0

    @property
    def balanced_accuracy(self):
        return float(np.mean(self.fold_balanced)) if self.fold_balanced else self.accuracy

    @property
    def better_than_random(self):
        """Accuracy above the random limit; with unequal conditions balanced accuracy above its own
        limit too; and when a permutation test was run, p at most 0.05 as well."""
        return (self.accuracy > self.random_limit and not self._balanced_fails()
                and not (self.permutations and self.permutation_p > 0.05))

    @property
    def chance_level(self):
        """Where the accuracy stands against chance, in words and in three steps rather than a yes or no
        (GIOVA, 31 October 2026): "clearly above chance" past the alpha 0.01 limit, "above chance" past the 0.05
        one, else "within what chance can reach"."""
        if not self.better_than_random:
            return "within what chance can reach"
        return "clearly above chance" if self.accuracy > self.random_limit_01 else "above chance"

    def _balanced_fails(self):
        return self.unequal and not self.balanced_accuracy > self.balanced_limit

    @property
    def largest(self):
        """(condition, share of all trials) of the largest condition."""
        name = max(self.counts, key=self.counts.get)
        return name, self.counts[name] / sum(self.counts.values())

    def random_text(self):
        """The verdict in words, with the limits it was judged against."""
        if self.better_than_random:
            text = "{}: {:.1%} is above the {:.1%} a random classifier stays within".format(
                self.chance_level, self.accuracy, self.random_limit)
            if self.unequal:
                text += ", and balanced accuracy {:.1%} above its {:.1%}".format(self.balanced_accuracy,
                                                                                 self.balanced_limit)
        else:
            text = "within what chance can reach: a random classifier reaches up to {:.1%}".format(self.random_limit)
            if self._balanced_fails():
                text += ("; the conditions have unequal trials (always guessing {} scores {:.1%}) and balanced "
                         "accuracy {:.1%} is not above the {:.1%} a random classifier reaches".format(
                             *self.largest, self.balanced_accuracy, self.balanced_limit))
            elif self.permutations and self.permutation_p > 0.05 and self.accuracy > self.random_limit:
                text += "; the permutation test does not confirm the accuracy"
        text += " (alpha 0.05, {} tested trials; Müller-Putz et al., 2008; Combrisson & Jerbi, 2015)".format(
            self.tested)
        if self.permutations:
            text += "; permutation test p = {:.3g} ({} shuffles)".format(self.permutation_p, self.permutations)
        return text

    @property
    def tested(self):
        return len(self.predictions)

    @property
    def auc(self):
        return float(np.mean(self.fold_auc)) if self.fold_auc else float("nan")

    def split_text(self):
        if self.scheme == ACROSS:
            return "trained on the other recordings"
        if self.scheme == CHRONOLOGICAL:
            return "chronological split"
        if self.scheme == REPEATED:
            return "{} × {}-fold".format(self.repeats, self.n_folds)
        return "{} folds".format(self.n_folds)

    def tuning_text(self):
        """How often each tuned setting was chosen, e.g. "C = 1 in 4 of 5 folds"."""
        picks = [tuple(sorted(c.items())) for c in self.chosen if c]
        if not picks:
            return ""
        counts = {}
        for pick in picks:
            counts[pick] = counts.get(pick, 0) + 1
        best, n = max(counts.items(), key=lambda item: item[1])
        return "{} in {} of {} folds".format(", ".join("{} = {}".format(k, v) for k, v in best), n, len(picks))

    def describe(self):
        spread = "" if len(self.fold_accuracy) < 2 else " ± {:.1%}".format(self.accuracy_sd)
        text = "Accuracy {:.1%}{} ({}, {}), {}".format(
            self.accuracy, spread, self.split_text(), self.model,
            "better than random (limit {:.1%})".format(self.random_limit) if self.better_than_random
            else "not better than random (limit {:.1%})".format(self.random_limit))
        text += ", ROC AUC {:.3f}".format(self.auc) if self.fold_auc else ""
        return text + "".join(", {} {:.3f}".format(name, value) for name, value in
                              (("Cohen's kappa", self.kappa), ("macro F1", self.f1)) if np.isfinite(value))


def cross_validate(x, labels, model, n_folds=5, seed=42, ids=None, folds=None, scheme=KFOLD, repeats=5,
                   train_fraction=0.8, permutations=0, progress=None, limits=True):
    """Scores of `model`. `folds` (test indices) overrides the split; trials are in recording order.

    `permutations` > 0 also runs the whole cross-validation that many times with the labels shuffled
    (Ojala & Garriga, 2010): p is the share of shuffles that score at least as well as the real labels.
    `limits` False leaves out the random classifier's limits (NaN), which take up to 30 000 simulated
    shuffles: a permutation run is read for its accuracy alone.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray([str(label) for label in labels])
    classes = list(np.unique(labels))
    if len(classes) < 2:
        raise ValueError("Cross-validation needs trials of at least two conditions")
    counts = {c: int(np.sum(labels == c)) for c in classes}
    rounds = [(0, f) for f in folds] if folds is not None else split(labels, scheme, n_folds, seed, repeats,
                                                                      train_fraction)
    model.build(seed)  # fail before any fitting when the model is incomplete
    ids = list(ids) if ids is not None else [str(i + 1) for i in range(len(labels))]

    accuracy, balanced, auc, predictions, chosen, kappa, f1 = [], [], [], [], [], [], []
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for k, (repeat, test) in enumerate(rounds):
        if progress:
            # "k/n" is read as the share done; before a permutation test the folds are only its start
            progress("Real labels, fold {} of {}".format(k + 1, len(rounds)) if permutations else
                     "Fold {}/{}".format(k + 1, len(rounds)))
        train = np.setdiff1d(np.arange(len(labels)), test)
        steps, classifier, predicted, scores = fit_and_test(model, seed + repeat, x[train], labels[train], x[test])
        truth = labels[test]
        accuracy.append(float(np.mean(predicted == truth)))
        balanced.append(float(np.mean([np.mean(predicted[truth == c] == c) for c in np.unique(truth)])))
        kappa.append(cohen_kappa(truth, predicted))
        f1.append(macro_f1(truth, predicted))
        if len(classes) == 2:
            auc.append(roc_auc(truth == str(classifier.classes[1]), scores))
        chosen.append({k: v for s in steps if getattr(s, "chosen", None) for k, v in s.chosen.items()})
        if repeat == 0:
            for t, p in zip(truth, predicted):
                confusion[classes.index(t), classes.index(p)] += 1
            for i, p, s in zip(test, predicted, scores if scores.ndim == 1 else scores.max(axis=1)):
                predictions.append((ids[i], labels[i], str(p), float(s)))

    first_round = [t for r, t in rounds if r == 0]
    tested_labels = labels[np.concatenate(first_round)]
    chance = max(counts.values()) / len(labels)
    order = {trial: k for k, trial in enumerate(ids)}
    if limits:
        limit_05, limit_01 = random_limit(tested_labels, 0.05), random_limit(tested_labels, 0.01)
        limit_balanced = balanced_limit(tested_labels)
    else:
        limit_05 = limit_01 = limit_balanced = float("nan")
    scores = Scores(classes, counts, accuracy, auc, confusion, sorted(predictions, key=lambda r: order[r[0]]),
                    model.describe(), len(first_round), seed, chance, limit_05, limit_01, fold_balanced=balanced,
                    scheme=scheme if folds is None else KFOLD,
                    repeats=int(repeats) if scheme == REPEATED and folds is None else 1, chosen=chosen,
                    balanced_limit=limit_balanced, fold_kappa=kappa, fold_f1=f1)
    if permutations:
        rng = np.random.default_rng(seed)
        better = 0
        for k in range(int(permutations)):
            if progress:
                progress("Permutation test {}/{}".format(k + 1, permutations))
            shuffled = rng.permutation(labels)
            if scheme == CHRONOLOGICAL and folds is None:
                # keep the split; a shuffle that empties a condition from one side is drawn again
                while True:
                    try:
                        split(shuffled, scheme, n_folds, seed, repeats, train_fraction)
                        break
                    except ValueError:
                        shuffled = rng.permutation(labels)
            other = cross_validate(x, shuffled, model, n_folds, seed, ids, folds, scheme, repeats, train_fraction,
                                   limits=False)
            better += other.accuracy >= scores.accuracy
        scores.permutations = int(permutations)
        scores.permutation_p = (better + 1) / (int(permutations) + 1)
    return scores


def roc_auc(positive, scores):
    """Area under the ROC curve: the chance a positive trial outscores a negative one."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cohen_kappa(truth, predicted):
    """Cohen's kappa (Cohen, 1960): (p_o - p_e) / (1 - p_e), p_o the share of trials predicted right and p_e
    the agreement expected by chance from the confusion matrix's row and column sums (sum over the conditions
    of true share times predicted share), as Schlögl et al. (2007) recommend for BCI results. 0 is chance,
    1 every trial right; NaN when p_e is 1 (one condition only, predicted every time)."""
    truth, predicted = np.asarray(truth).astype(str), np.asarray(predicted).astype(str)
    n = truth.size
    if not n:
        return float("nan")
    observed = float(np.mean(truth == predicted))
    expected = float(sum(np.sum(truth == c) * np.sum(predicted == c)
                         for c in np.union1d(truth, predicted))) / (n * n)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else float("nan")


def macro_f1(truth, predicted):
    """F1 per tested condition, 2 TP / (2 TP + FP + FN) (the harmonic mean of precision and recall), averaged
    without weighting over the conditions in `truth`: a condition never predicted counts 0."""
    truth, predicted = np.asarray(truth).astype(str), np.asarray(predicted).astype(str)
    scores = []
    for c in np.unique(truth):
        tp = np.sum((predicted == c) & (truth == c))
        fp = np.sum((predicted == c) & (truth != c))
        fn = np.sum((predicted != c) & (truth == c))
        scores.append(2.0 * tp / (2.0 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


#: Shuffles simulated for the random limit of more than two conditions, as in Müller-Putz et al. (2008).
SIMULATIONS = 10000


def random_interval(n_trials, n_classes=2, alpha=0.05, labels=None, simulations=SIMULATIONS, seed=0):
    """(low, high): the accuracy a random classifier stays within (Müller-Putz et al., 2008).

    Two conditions: the paper's binomial interval around 0.5 with the Agresti & Caffo (2000)
    adjustment, p ± z(1-alpha/2) sqrt(p (1 - p) / (n + 4)) with p = (n/2 + 2) / (n + 4). More
    conditions: as in the paper, by simulation - the labels in a random order scored against
    themselves `simulations` times - and the alpha/2 and 1 - alpha/2 quantiles of that accuracy.
    """
    n = int(n_trials)
    if n < 1:
        raise ValueError("No tested trials")
    if n_classes == 2:
        p = (n * 0.5 + 2) / (n + 4)
        half = norm.ppf(1 - alpha / 2) * np.sqrt(p * (1 - p) / (n + 4))
        return float(p - half), float(p + half)
    rng = np.random.default_rng(seed)
    if labels is None:
        labels = np.arange(n) % n_classes
    labels = np.asarray(labels)
    shuffled = rng.permuted(np.tile(labels, (simulations, 1)), axis=1)
    accuracy = (shuffled == labels).mean(axis=1)
    return float(np.quantile(accuracy, alpha / 2)), float(np.quantile(accuracy, 1 - alpha / 2))


def random_limit(labels, alpha=0.05):
    """The upper edge of `random_interval` for these tested labels."""
    labels = np.asarray(labels)
    return random_interval(labels.size, np.unique(labels).size, alpha, labels)[1]


def balanced_limit(labels, alpha=0.05, simulations=SIMULATIONS, seed=0):
    """The 1 - alpha/2 quantile of a random classifier's balanced accuracy on these tested labels; NaN
    when every condition has the same number of trials.

    With 90 trials of one condition and 10 of the other, always answering the first scores 90%,
    well above the 59.6% limit of `random_interval`, while its balanced accuracy is 50%. The
    random classifier is simulated as `random_interval` does for more conditions: the labels in
    a random order scored against themselves, here by their mean recall over the conditions.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() == counts.max():
        return float("nan")
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(labels, (simulations, 1)), axis=1)
    recall = [((shuffled == c) & (labels == c)).sum(axis=1) / n for c, n in zip(classes, counts)]
    return float(np.quantile(np.mean(recall, axis=0), 1 - alpha / 2))


def fit_and_test(model, seed, x_train, y_train, x_test):
    """(steps, classifier, predicted labels, scores): every step fitted on the training trials only.

    A tuned step's setting is chosen first by `_search`, on these training trials alone.
    """
    steps = model.build(seed)
    tuned = [k for k, step in enumerate(steps) if getattr(step, "grid", None)]
    if tuned:
        for k, setting in _search(model, seed, tuned, x_train, y_train).items():
            steps[k].estimator.set_params(**setting)
            steps[k].grid, steps[k].chosen = None, dict(setting)
    return (steps, *_fit_chain(steps, x_train, y_train, x_test))


def _search(model, seed, tuned, x, y):
    """{step index: setting}: every combination of the tuned steps' grids, each scored by an inner
    stratified cross-validation of the training trials, the best mean score kept (the first of a tie).
    The first tuned step's inner folds and score (accuracy or balanced accuracy) are the search's.

    Each inner fold fits the whole chain again - feature selection, CSP, scaling and augmentation
    included - on its own training part, as a grid search over a scikit-learn Pipeline does.
    Searching only the classifier on features already fitted to all training trials would let
    the inner validation trials shape those features, and augmented copies of them sit among the
    inner training trials.
    """
    from itertools import product
    from sklearn.model_selection import ParameterGrid, StratifiedKFold
    y = np.asarray(y)
    template = model.build(seed)
    # the steps before the first tuned one do not depend on the setting: fitted once per inner fold,
    # every setting continues from there (what a cached Pipeline does)
    first = min(tuned)
    smallest = min(np.unique(y, return_counts=True)[1])
    folds = getattr(template[first], "inner_folds", INNER_FOLDS)
    balanced = getattr(template[first], "scoring", ACCURACY) == BALANCED
    inner = StratifiedKFold(max(2, min(folds, smallest)), shuffle=True, random_state=seed)
    parts = []
    for train, test in inner.split(np.zeros(len(y)), y):
        try:
            parts.append((*_fit_steps(model.build(seed)[:first], x[train], y[train], x[test]), y[test].astype(str)))
        except ValueError:
            parts.append(None)
    best, best_score = None, -np.inf
    for combination in product(*(list(ParameterGrid(template[k].grid)) for k in tuned)):
        scores = []
        for part in parts:
            if part is None:
                scores.append(np.nan)
                continue
            a, ya, b, truth = part
            steps = model.build(seed)
            for k, setting in zip(tuned, combination):
                steps[k].estimator.set_params(**setting)
                steps[k].grid = None
            try:
                _, predicted, _ = _fit_chain(steps[first:], a, ya, b)
            except ValueError:
                scores.append(np.nan)       # as GridSearchCV: a setting that cannot be fitted scores nothing
                continue
            scores.append(float(np.mean([np.mean(predicted[truth == c] == c) for c in np.unique(truth)])
                                if balanced else np.mean(predicted == truth)))
        score = float(np.mean(scores))
        if np.isfinite(score) and score > best_score:
            best, best_score = combination, score
    if best is None:
        raise ValueError("Tuning: no setting could be fitted on the training trials of this fold")
    return dict(zip(tuned, best))


def _fit_steps(steps, x_train, y_train, x_test):
    """(training trials, their labels, test trials) after `steps`, each fitted on the training trials."""
    a, b = x_train, x_test
    for step in steps:
        if hasattr(step, "augment"):          # more training trials; the test trials stay as they are
            a, y_train = step.augment(a, y_train)
            continue
        a = step.fit(a, y_train).transform(a)
        b = step.transform(b)
    return a, y_train, b


def _fit_chain(steps, x_train, y_train, x_test):
    """(classifier, predicted labels, scores) of built, unfitted steps fitted on the training trials."""
    a, y_train, b = _fit_steps(steps[:-1], x_train, y_train, x_test)
    classifier = steps[-1].fit(a, y_train)
    return classifier, np.asarray(classifier.predict(b)).astype(str), classifier.decision(b)


def leave_one_out(groups, model, seed=42, progress=None, stop=None):
    """Scores per group: the model trained on every other group's trials, tested on this one's.

    `groups` is [(name, values, labels, ids)], one per recording or subject. This is the
    question a BCI meets with a new user: does what was learned from others carry over? It is
    usually much harder than cross-validation within one recording. Every step is fitted on the
    training groups only, as in cross_validate.
    """
    if len(groups) < 2:
        raise ValueError("Leaving one recording out needs at least two recordings")
    shapes = {np.asarray(v).shape[1:] for _, v, _, _ in groups}
    if len(shapes) > 1:
        raise ValueError("The recordings give trials of different shapes ({}): give them the same channels, "
                         "rate and window".format(", ".join(str(s) for s in shapes)))
    results = []
    for k, (name, x, labels, ids) in enumerate(groups):
        if stop is not None and stop():
            break
        if progress:
            progress("Testing on {} ({}/{}), trained on the others".format(name, k + 1, len(groups)))
        train = [g for j, g in enumerate(groups) if j != k]
        x_train = np.concatenate([np.asarray(g[1]) for g in train])
        y_train = np.concatenate([np.asarray(g[2]).astype(str) for g in train])
        truth = np.asarray(labels).astype(str)
        steps, classifier, predicted, scores = fit_and_test(model, seed, x_train, y_train, np.asarray(x))
        classes = list(np.unique(np.concatenate([y_train, truth])))
        confusion = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(truth, predicted):
            confusion[classes.index(t), classes.index(p)] += 1
        counts = {c: int(np.sum(truth == c)) for c in np.unique(truth)}
        flat = scores if scores.ndim == 1 else scores.max(axis=1)
        auc = [roc_auc(truth == str(classifier.classes[1]), scores)] if len(classes) == 2 and \
            np.unique(truth).size == 2 else []
        results.append(Scores(
            classes, counts, [float(np.mean(predicted == truth))], auc, confusion,
            [(str(i), t, str(p), float(s)) for i, t, p, s in zip(ids, truth, predicted, flat)],
            model.describe(), 1, seed, max(counts.values()) / len(truth), random_limit(truth, 0.05),
            random_limit(truth, 0.01),
            fold_balanced=[float(np.mean([np.mean(predicted[truth == c] == c) for c in np.unique(truth)]))],
            scheme=ACROSS,
            chosen=[{k2: v for s in steps if getattr(s, "chosen", None) for k2, v in s.chosen.items()}],
            balanced_limit=balanced_limit(truth), fold_kappa=[cohen_kappa(truth, predicted)],
            fold_f1=[macro_f1(truth, predicted)]))
    return results
