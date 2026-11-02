"""Editable feature definitions and trial-level descriptive comparisons."""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

import numpy as np

from .config import BANDS
from .spectral import band_integral, welch_psd, hjorth, spectral_entropy, sample_index


DEFAULT_FEATURES = ["band:alpha", "relative:alpha", "ratio:alpha/beta", "rms", "entropy"]


@dataclass
class FeatureSpec:
    id: str
    name: str
    unit: str
    description: str


@dataclass
class FeatureProfile:
    name: str = "Eyes open / closed"
    bands: list = field(default_factory=lambda: [list(b) for b in BANDS])
    ratios: list = field(default_factory=lambda: [["alpha/beta", "alpha", "beta"], ["theta/alpha", "theta", "alpha"]])
    selected: list = field(default_factory=lambda: list(DEFAULT_FEATURES))
    total_range: list = field(default_factory=lambda: [1.0, 45.0])
    time_window: float = 2.0
    time_step: float = 1.0
    version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload=None):
        try:
            profile = cls(**(payload or {}))
            profile.validate()
            return profile
        except (TypeError,AttributeError,IndexError,KeyError) as exc:
            raise ValueError("Malformed feature profile") from exc

    def validate(self):
        if self.version != 1 or not self.name.strip():
            raise ValueError("Unsupported or unnamed feature profile")
        names = [b[0] for b in self.bands]
        if len(names) != len(set(names)) or not names:
            raise ValueError("Band names must be unique and nonempty")
        for name, low, high in self.bands:
            if not str(name).strip() or not np.isfinite([low, high]).all() or not 0 <= low < high:
                raise ValueError("Each band needs a name and 0 <= low < high")
        lo, hi = self.total_range
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
            raise ValueError("Total power range needs 0 <= low < high")
        if not np.isfinite([self.time_window, self.time_step]).all() or min(self.time_window, self.time_step) <= 0:
            raise ValueError("Time window and step must be positive")
        ratio_names = []
        for name, numerator, denominator in self.ratios:
            if not str(name).strip() or numerator not in names or denominator not in names:
                raise ValueError("Ratios must reference existing bands")
            ratio_names.append(name)
        if len(ratio_names) != len(set(ratio_names)):
            raise ValueError("Ratio names must be unique")
        available = {spec.id for spec in self.catalog()}
        if not self.selected or any(key not in available for key in self.selected):
            raise ValueError("Choose at least one available feature")
        if len(self.selected)!=len(set(self.selected)):
            raise ValueError("Visible features must be unique")

    def catalog(self):
        specs = []
        for name, low, high in self.bands:
            span = f"{low:g}–{high:g} Hz"
            specs.extend([
                FeatureSpec("band:" + name, name + " power", "uV^2", "Welch PSD integrated over " + span),
                FeatureSpec("relative:" + name, "Relative " + name, "%", "Band power / total-range PSD integral × 100"),
                FeatureSpec("peak:" + name, name + " maximum frequency", "Hz", "Frequency of the largest PSD bin in " + span + "; not proof of an oscillation"),
            ])
        specs += [FeatureSpec("ratio:" + name, name + " ratio", "ratio", numerator + " power / " + denominator + " power")
                  for name, numerator, denominator in self.ratios]
        specs += [FeatureSpec("total", "Total range power", "uV^2", "Direct PSD integral over the total frequency range"),
                  FeatureSpec("rms", "RMS", "uV", "Root mean square after removing DC"),
                  FeatureSpec("ptp", "Peak-to-peak", "uV", "Maximum minus minimum"),
                  FeatureSpec("variance", "Variance / Hjorth activity", "uV^2", "Variance after removing DC; identical to squared RMS"),
                  FeatureSpec("mobility", "Hjorth mobility", "s^-1", "sqrt(var(dx/dt) / var(x))"),
                  FeatureSpec("complexity", "Hjorth complexity", "ratio", "Mobility of the derivative / mobility of the signal"),
                  FeatureSpec("crossings", "Zero crossings", "s^-1", "Sign changes per second after removing DC"),
                  FeatureSpec("entropy", "Spectral entropy", "0–1", "Shannon entropy / log2(number of bins) over the total range")]
        return specs

    def visible_specs(self):
        catalog = {s.id: s for s in self.catalog()}
        return [catalog[key] for key in self.selected]


def band_power_profile():
    profile = FeatureProfile(name="Band powers")
    profile.selected = ["variance"] + ["band:" + b[0] for b in profile.bands]
    return profile


def save_profile(path, profile):
    profile.validate()
    Path(path).write_text(json.dumps(profile.to_dict(), indent=2, allow_nan=False), encoding="utf-8")


def load_profile(path):
    try:
        return FeatureProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid feature profile") from exc


@dataclass
class FeatureValues:
    values: dict
    reasons: dict


def extract(signal, fs, profile, window=2.0, overlap=0.5, psd=None):
    """One feature vector from one finite, already filtered signal."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        return FeatureValues({s.id: np.nan for s in profile.catalog()},
                             {s.id: "At least three finite samples required" for s in profile.catalog()})
    x = x - x.mean()
    freqs, power = welch_psd(x, fs, window, overlap) if psd is None else psd
    values, reasons = {}, {}

    def integral(key, low, high):
        if high > fs / 2 or low < freqs[0] or high > freqs[-1]:
            reasons[key] = "Range exceeds available spectrum / Nyquist"
            return np.nan
        if np.count_nonzero((freqs >= low) & (freqs <= high)) < 2:
            reasons[key] = "Frequency resolution too coarse for this range"
            return np.nan
        return band_integral(freqs, power, low, high)

    lo, hi = profile.total_range
    total = integral("total", lo, hi)
    values["total"] = total
    for name, low, high in profile.bands:
        key = "band:" + name
        value = integral(key, low, high)
        values[key] = value
        relative = "relative:" + name
        if low < lo or high > hi:
            values[relative] = np.nan
            reasons[relative] = "Band is outside the total-power range"
        elif np.isfinite(total) and total > 0 and np.isfinite(value):
            values[relative] = value / total * 100
        else:
            values[relative] = np.nan
            reasons[relative] = "Total power or band is unavailable / zero"
        keep = (freqs >= low) & (freqs <= high)
        peak = "peak:" + name
        if np.isfinite(value) and value > 0 and np.ptp(power[keep]) > 0:
            values[peak] = float(freqs[keep][np.argmax(power[keep])])
        else:
            values[peak] = np.nan
            reasons[peak] = "No resolvable maximum in this band"
    for name, numerator, denominator in profile.ratios:
        a, b = values["band:" + numerator], values["band:" + denominator]
        key = "ratio:" + name
        values[key] = a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan
        if not np.isfinite(values[key]):
            reasons[key] = "Band unavailable or denominator is zero"
    activity, mobility, complexity = hjorth(x, fs)
    keep = (freqs >= lo) & (freqs <= hi)
    values.update(rms=float(np.sqrt(np.mean(x ** 2))), ptp=float(np.ptp(x)),
                  variance=float(activity), mobility=float(mobility), complexity=float(complexity),
                  crossings=float(np.count_nonzero(np.diff(np.signbit(x))) / (len(x) / fs)),
                  entropy=float(spectral_entropy(power[keep])) if np.isfinite(total) else np.nan)
    for key, value in values.items():
        if not np.isfinite(value) and key not in reasons:
            reasons[key] = "Undefined for this signal / frequency range"
    return FeatureValues(values, reasons)


@dataclass
class TrialFeatures:
    profile: FeatureProfile
    trials: dict
    mean_signals: dict


def mean_values(rows):
    """Average feature vectors over channels; a value missing anywhere is missing."""
    if len(rows) == 1:
        return rows[0]
    values, reasons = {}, {}
    for key in rows[0].values:
        column = np.array([row.values[key] for row in rows], dtype=float)
        values[key] = float(column.mean()) if np.isfinite(column).all() else np.nan
        found = sorted({row.reasons[key] for row in rows if key in row.reasons})
        if found:
            reasons[key] = "; ".join(found)
    return FeatureValues(values, reasons)


def _roi_extract(signals, fs, profile, settings, metrics=None):
    """Features per channel, then averaged: the ROI never mixes waveforms."""
    if signals is None:
        return None
    return mean_values([extract(row, fs, profile, settings.window, settings.overlap,
                                (m.psd_freqs, m.psd) if m is not None else None)
                        for row, m in zip(signals, metrics or [None] * len(signals))])


def extract_trials(result, profile):
    profile.validate()
    settings = result.settings
    rows = {t.trial.id: _roi_extract(t.channel_signals, result.fs, profile, settings, t.channel_metrics)
            or extract(t.signal, result.fs, profile, settings.window,
                       settings.overlap, (t.metrics.psd_freqs, t.metrics.psd)) for t in result.trials}
    means = {c.name: _roi_extract(c.channel_signals, result.fs, profile, settings)
             or extract(c.signal, result.fs, profile, settings.window,
                        settings.overlap, (c.mean_signal.psd_freqs, c.mean_signal.psd)) for c in result.conditions}
    return TrialFeatures(profile, rows, means)


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan)
    return dict(n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                median=float(np.median(values)), q1=float(np.percentile(values, 25)), q3=float(np.percentile(values, 75)))


def cliffs_delta(a, b):
    """Descriptive superiority of B over A, NOT classification accuracy or a p-value."""
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 3:
        return np.nan
    sorted_a = np.sort(a)
    greater = np.searchsorted(sorted_a, b, side="left").sum()
    lower = (len(a) - np.searchsorted(sorted_a, b, side="right")).sum()
    return float(greater - lower) / (len(a) * len(b))


def condition_values(result, condition, key):
    return [result.features.trials[t.trial.id].values[key]
            for t in result.trials if t.trial.condition == condition]


def time_features(result, profile):
    """Complete windows inside each trial; window count never changes trial n."""
    n = sample_index(profile.time_window, result.fs)
    step = sample_index(profile.time_step, result.fs)
    if n < 3 or step < 1:
        raise ValueError("Time detail needs at least three samples per window and a positive step")
    rows = {}
    for trial in result.trials:
        starts = np.arange(0, len(trial.signal) - n + 1, step, dtype=int)
        signals = trial.channel_signals if trial.channel_signals is not None else trial.signal[None, :]
        values = [_roi_extract(signals[:, s:s + n], result.fs, profile, result.settings) for s in starts]
        rows[trial.trial.id] = ((starts + n / 2) / result.fs, values)
    return rows
