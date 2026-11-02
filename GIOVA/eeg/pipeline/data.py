"""What flows between blocks, besides a Recording and an EpochResult.

Each container is a plain record: arrays plus the labels needed to read them.
Blocks never change one they were given; they return a new one, which is what
makes keeping outputs between runs safe.

Pure numpy - no Qt.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.sources import Recording


def derive(recording, data=None, fs=None, names=None):
    """A copy of `recording` with new samples, rate or channel names; everything else kept."""
    out = Recording(recording.data if data is None else data, recording.fs if fs is None else fs,
                    recording.channel_names if names is None else names,
                    recording.timestamps, recording.path, recording.rail_uv, recording.events)
    for slot in ("gap_samples", "events_are_raw_markers", "channel_log", "motion",
                 "motion_names", "motion_units", "original_fs", "unit_note", "bad_spans"):
        setattr(out, slot, getattr(recording, slot))
    out.reading = list(recording.reading)       # a copy: a note added to the new one is not the old one's
    return out


@dataclass
class Epochs:
    """Trials of equal length: (n_trials, n_channels, n_samples) in µV."""
    data: np.ndarray
    fs: float
    names: list
    #: one condition label per trial
    labels: list
    #: time of the first sample, s from each event
    tmin: float
    #: trial ids, 1-based positions among the recording's events
    ids: list
    #: trials left out, [(id, label, reason)]
    rejected: list = field(default_factory=list)
    #: one dict per row of `data`, where the trial came from: id, condition, start / end (s in the recording,
    #: the trial as defined), included, sample (first sample used), count (samples used), zero (its time zero,
    #: s in the recording); empty when not known
    meta: list = field(default_factory=list)
    #: channel names not measured during these trials: switched off (the amplifier's log) or flat; a region,
    #: picked channels and head maps leave them out, and say so
    unmeasured: list = field(default_factory=list)

    @property
    def times(self):
        return self.tmin + np.arange(self.data.shape[-1]) / self.fs

    def counts(self):
        return {label: self.labels.count(label) for label in dict.fromkeys(self.labels)}

    def describe(self):
        counts = ", ".join("{} ×{}".format(k, v) for k, v in self.counts().items())
        return "{} trials ({}), {} channels, {:g} to {:g} s{}{}".format(
            len(self.labels), counts, len(self.names), self.tmin, self.tmin + self.data.shape[-1] / self.fs,
            " · {} rejected".format(len(self.rejected)) if self.rejected else "",
            " · not measured (switched off or flat): " + ", ".join(self.unmeasured) if self.unmeasured else "")

    def with_data(self, data, **changes):
        return replace(self, data=data, **changes)


#: What a Spectrum holds: a Welch power density, or the FFT amplitude of each trial.
DENSITY, AMPLITUDE = "density", "amplitude"


@dataclass
class Spectrum:
    """Per trial and channel, (n_trials, n_channels, n_freqs): a Welch density in µV²/Hz, or
    with kind AMPLITUDE the FFT amplitude in µV (a sine of amplitude A reads A)."""
    freqs: np.ndarray
    power: np.ndarray
    names: list
    labels: list
    ids: list
    kind: str = DENSITY
    #: condition -> (n_channels, n_freqs): the spectrum of the condition's mean waveform, channel by channel
    #: (the Analysis page's "analysis of the mean signal"); empty unless asked for. `power` stays per trial,
    #: so its mean over a condition's trials is the page's "mean of trial analyses".
    signal_means: dict = field(default_factory=dict)

    @property
    def is_power(self):
        return self.kind == DENSITY

    @property
    def unit(self):
        return "µV²/Hz" if self.is_power else "µV"

    @property
    def title(self):
        return "PSD" if self.is_power else "FFT amplitude"

    def describe(self):
        return "{}: {} trials × {} channels, {:g}–{:g} Hz in {:g} Hz steps".format(
            self.title, self.power.shape[0], self.power.shape[1], self.freqs[0], self.freqs[-1],
            self.freqs[1] - self.freqs[0] if self.freqs.size > 1 else 0)


@dataclass
class Features:
    """One row per trial: (n_trials, n_features)."""
    values: np.ndarray
    #: column names, e.g. "C3 alpha power"
    names: list
    labels: list
    ids: list
    units: list = field(default_factory=list)
    #: one dict per trial, {column name: why that value is missing}; empty when nothing is missing or not known
    reasons: list = field(default_factory=list)

    def describe(self):
        return "{} trials × {} features".format(*self.values.shape)

    def table(self):
        """Rows for a CSV: header, then id, label and the values of each trial."""
        rows = [["trial", "condition", *self.names]]
        rows += [[i, label, *row] for i, label, row in zip(self.ids, self.labels, self.values.tolist())]
        return rows


@dataclass
class IcaResult:
    """Independent components of a recording and which were removed."""
    names: list
    #: (n_channels, n_components): each component's weight on each electrode
    patterns: np.ndarray
    #: components removed, 0-based
    excluded: list
    #: |correlation| of each component with the EOG reference channels; empty without them
    eog_scores: np.ndarray
    eog_channels: list
    method: str
    #: components left out because they matched the EOG channels, and those named by hand
    found: list = field(default_factory=list)
    manual: list = field(default_factory=list)

    @property
    def n_components(self):
        return self.patterns.shape[1]

    def describe(self):
        return "{} components ({}); removed {}".format(
            self.n_components, self.method, ", ".join(str(c) for c in self.excluded) if self.excluded else "none")


@dataclass
class Contrast:
    """Band power of two windows of the same trials, channel by channel, and a paired test across the trials.

    rows: one per channel and band: channel, band, low, high, mean_a, mean_b (µV²), change (% of A from B,
    pooled: the mean powers compared), trials_up (share of trials with more power in A), statistic, p, p_holm
    (Holm over every channel and band).
    """
    windows: list            # [[start, end], [start, end]] s
    names: list             # what the windows are called, e.g. ["odour", "air"]
    bands: list
    rows: list
    n_trials: int
    test: str
    segment: float

    def significant(self, alpha=0.05):
        return [r for r in self.rows if r["p_holm"] == r["p_holm"] and r["p_holm"] <= alpha]

    def describe(self):
        found = self.significant()
        return "{} vs {}: {} trials, {} channel-band pairs; {} with p (Holm) ≤ 0.05".format(
            self.names[0], self.names[1], self.n_trials, len(self.rows), len(found))

    def table(self):
        """Rows for a CSV: a header, then one row per channel and band."""
        keys = ["channel", "band", "low", "high", "mean_a", "mean_b", "change", "trials_up", "statistic", "p",
                "p_holm"]
        head = ["channel", "band", "low_hz", "high_hz", "mean_{}_uV2".format(self.names[0]),
                "mean_{}_uV2".format(self.names[1]), "change_percent", "share_of_trials_up", "statistic", "p",
                "p_holm"]
        return [head] + [[r[k] for k in keys] for r in self.rows]
