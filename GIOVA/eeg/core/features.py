"""Shared feature adapter, bounded history and versioned live CSV logging (no Qt)."""
import csv
import time
from pathlib import Path

import numpy as np

from .config import BANDS


class FeatureSet:
    """Live adapter over the same feature catalog used by offline trials."""

    def __init__(self, fs, window_seconds=2.0, fmin=1.0, fmax=45.0, bands=BANDS, profile=None):
        from .feature_profiles import FeatureProfile
        self.fs = float(fs)
        self.window_seconds = float(window_seconds)
        self.profile = profile or FeatureProfile()
        if profile is None:
            self.profile.total_range = [fmin, fmax]
            self.profile.bands = [list(b) for b in bands]
        self.profile.validate()

    @property
    def names(self):
        return [s.name + " (" + s.unit + ")" for s in self.profile.visible_specs()]

    def compute(self, block):
        from .feature_profiles import extract
        block = np.atleast_2d(np.asarray(block, dtype=float))
        if block.shape[-1] < 3:
            raise ValueError("Window too short")
        self.last_values = [extract(x,self.fs,self.profile,self.window_seconds) for x in block]
        return np.array([[row.values[key] for key in self.profile.selected] for row in self.last_values])


class FeatureHistory:
    """A fixed-length rolling record of one feature per channel over time.

    Deliberately bounded: a session left running for an hour must not grow a
    plot buffer until the machine swaps. Old rows fall off the front.
    """

    def __init__(self, n_channels, n_features, capacity=600):
        self.capacity = int(capacity)
        self.times = np.zeros(self.capacity)
        self.values = np.zeros((self.capacity, int(n_channels), int(n_features)))
        self.filled = 0

    def append(self, timestamp, row):
        if self.filled == self.capacity:
            self.times[:-1] = self.times[1:]
            self.values[:-1] = self.values[1:]
            self.filled -= 1
        self.times[self.filled] = timestamp
        self.values[self.filled] = row
        self.filled += 1

    def series(self, feature_index):
        """(times, values) with values shaped (n_samples, n_channels)."""
        return (self.times[:self.filled],
                self.values[:self.filled, :, int(feature_index)])

    def clear(self):
        self.filled = 0


class FeatureLog:
    """Append feature rows to a CSV, one line per channel per window.

    Long format rather than one wide row per window: it survives the channel
    selection changing mid-session, and it is what every analysis tool wants to
    be given anyway.
    """

    def __init__(self, path, names, feature_names, metadata=None):
        self.path = Path(path)
        self.rows = 0
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._feature_names = list(feature_names)
        self._metadata = __import__("json").dumps(metadata or {}, sort_keys=True)
        self._writer.writerow(["source_seconds", "elapsed_seconds", "channel", "feature", "value", "settings", "reason"])
        self._names = list(names)
        self._t0 = None

    def append(self, timestamp, channels, row, reasons=None):
        if self._t0 is None:
            self._t0 = timestamp
        elapsed = timestamp - self._t0
        for slot, channel in enumerate(channels):
            for index,(feature, value) in enumerate(zip(self._feature_names, row[slot])):
                self._writer.writerow([timestamp, elapsed, self._names[channel], feature,
                                       value if np.isfinite(value) else "", self._metadata,
                                       reasons[slot][index] if reasons is not None else ""])
                self.rows += 1
        self._handle.flush()  # a crashed session should keep what it recorded

    def close(self):
        if not self._handle.closed:
            self._handle.close()


def default_log_path(directory, prefix="features"):
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(time.time_ns() % 1000000000)
    return Path(directory) / "{}_{}.csv".format(prefix, stamp)
