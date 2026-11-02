"""Filtering, electrode quality, and display auto-scaling.

Pure numpy/scipy - no Qt here, so this is the part that is easy to test.
"""
import math
import time

import numpy as np
from scipy.signal import (butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt,
                          tf2sos, welch)

from .config import BANDS, NYQUIST, RAIL_UV
from .spectral import welch_psd, band_integral

# ----------------------------------------------------------------------
# filtering
# ----------------------------------------------------------------------


def design_filter(fs, low=None, high=None, order=4, notches=(), notch_q=30.0):
    """Combine notch and band filters into one SOS chain.

    Frequencies at or above Nyquist are silently dropped - at 125 Hz the
    100 Hz mains harmonic simply cannot be represented, and refusing to
    build the whole chain over that would be unhelpful.

    Returns None when nothing is enabled.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    nyq = fs / 2.0
    if low is not None and (not np.isfinite(low) or not 0 < low < nyq):
        raise ValueError("Low cut must be positive and below Nyquist")
    if high is not None and (not np.isfinite(high) or not 0 < high < nyq):
        raise ValueError("High cut must be positive and below Nyquist")
    if low is not None and high is not None and low >= high:
        raise ValueError("Low cut must be below high cut")
    stages = [tf2sos(*iirnotch(f, notch_q, fs)) for f in notches if 0 < f < nyq]

    low = low if low and low > 0 else None
    high = high if high and high < nyq else None
    if low and high and low < high:
        stages.append(butter(order, [low / nyq, high / nyq], btype="band", output="sos"))
    elif low:
        stages.append(butter(order, low / nyq, btype="high", output="sos"))
    elif high:
        stages.append(butter(order, high / nyq, btype="low", output="sos"))

    return np.vstack(stages) if stages else None


def apply_filter(sos, x):
    """Zero-phase filter. Passes short buffers through rather than raising."""
    if sos is None or x.shape[1] < 3 * (2 * sos.shape[0] + 1):
        return x
    return sosfiltfilt(sos, x, axis=1)


def finite_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def filter_finite(signal, fs, **kwargs):
    """Filter continuous finite runs independently; preserve gaps and sample clock."""
    x = np.asarray(signal, dtype=float)
    sos = design_filter(fs, **kwargs)
    out = np.full(x.shape, np.nan)
    minimum = 3 * (2 * sos.shape[0] + 1) + 1 if sos is not None else 2
    for start, end in finite_runs(np.isfinite(x)):
        if end - start >= minimum:
            run = x[start:end] - x[start:end].mean()
            out[start:end] = apply_filter(sos, run[None, :])[0]
    return out


class StreamFilter:
    """A causal filter that carries its state from one chunk to the next.

    `sosfiltfilt` cannot filter a live stream: it runs the filter backwards as
    well as forwards, so it needs samples that have not arrived. Applying it to
    a sliding window instead - re-filtering the whole display every frame - is
    what made the monitor shimmer: filtfilt leaves a transient at each end of
    whatever window it is handed, the window moves every frame, and those
    transients sweep across the screen. Measured on a 3-45 Hz chain at 125 Hz,
    the same sample was redrawn up to 16 mV away from its previous value on a
    signal whose rms is 40 uV.

    Here each sample passes through the filter exactly once, and its value never
    changes again. The price is phase delay, which zero-phase filtering avoided;
    for a live monitor that is the right trade and it is what streaming EEG
    viewers do. Offline analysis keeps `apply_filter`, where the window is fixed
    and nothing slides.
    """

    def __init__(self, sos, n_channels):
        self.sos = sos
        self.n_channels = int(n_channels)
        self._zi = None

    def reset(self, first_sample=None):
        """Start again, optionally settled on the level of `first_sample`.

        Without this the state starts at zero and the high-pass has to ring
        down from whatever DC offset the electrodes carry - seconds of garbage
        on a 3 Hz corner. Seeded with the first sample it starts settled.
        """
        if self.sos is None:
            self._zi = None
            return
        zi = np.repeat(sosfilt_zi(self.sos)[:, np.newaxis, :], self.n_channels,
                       axis=1)
        if first_sample is not None:
            zi = zi * np.asarray(first_sample, dtype=float)[np.newaxis, :,
                                                            np.newaxis]
        self._zi = zi

    def process(self, chunk):
        """Filter one chunk of (n_channels, n) and advance the state."""
        if self.sos is None or chunk.shape[1] == 0:
            return chunk
        if self._zi is None:
            self.reset(chunk[:, 0])
        if not np.isfinite(chunk).all() or not np.isfinite(self._zi).all():
            out = np.full(chunk.shape, np.nan)
            for channel, row in enumerate(chunk):
                for start, end in finite_runs(np.isfinite(row)):
                    zi = self._zi[:, channel, :]
                    if start > 0 or not np.isfinite(zi).all():
                        zi = sosfilt_zi(self.sos) * row[start]
                    out[channel, start:end], state = sosfilt(self.sos, row[start:end], zi=zi)
                    self._zi[:, channel, :] = state
                if not np.isfinite(row[-1]):
                    self._zi[:, channel, :] = np.nan
            return out
        out, self._zi = sosfilt(self.sos, chunk, axis=-1, zi=self._zi)
        return out


def decimate_minmax(block, bins, return_indices=False):
    """Two points per bin - the min and the max - for every channel at once.

    A screen a thousand pixels wide cannot show five thousand samples, and
    drawing them anyway is most of what makes the monitor slow. Plain
    subsampling would be cheaper still and would drop the spikes between the
    samples it keeps; taking both extremes of each bin keeps the envelope, so
    an artefact one sample wide is still visible.

    Returns (values, n_points), or (values, sample_indices) when requested.
    Extrema retain their sample order; no sample is dropped and none is left
    to a bin much wider than the rest.
    """
    block = np.atleast_2d(block)
    n_channels, n = block.shape
    bins = max(int(bins), 1)
    if n <= 2 * bins:
        return (block, np.broadcast_to(np.arange(n), block.shape)) if return_indices else (block, n)

    # Every bin holds per_bin or per_bin + 1 samples. Leftover samples must be
    # spread over the bins: gathered into one last bin, 2500 samples in 540
    # bins left 340 samples (1.4 s at 250 Hz) drawn as two points - a straight
    # "V" at the right end of every trace.
    per_bin, longer = divmod(n, bins)
    split = longer * (per_bin + 1)
    parts = []
    for start, stop, size in ((0, split, per_bin + 1), (split, n, per_bin)):
        if stop <= start:
            continue
        shaped = block[:, start:stop].reshape(n_channels, -1, size)
        part = np.sort(np.stack((shaped.argmin(axis=2), shaped.argmax(axis=2)), axis=-1), axis=-1)
        part += start + np.arange(shaped.shape[1])[None, :, None] * size
        parts.append(part.reshape(n_channels, -1))
    indices = np.concatenate(parts, axis=1)
    out = np.take_along_axis(block, indices, axis=1)
    return (out, indices) if return_indices else (out, out.shape[1])


def mains_harmonics(base, fs):
    """Mains frequency and the harmonics that fit below Nyquist."""
    return [base * k for k in (1, 2, 3) if base * k < fs / 2.0]


# ----------------------------------------------------------------------
# electrode quality
# ----------------------------------------------------------------------


class Quality:
    """Verdict for one electrode."""

    __slots__ = ("verdict", "railed", "std")

    def __init__(self, verdict, railed, std):
        self.verdict = verdict
        self.railed = railed
        self.std = std

    @property
    def is_good(self):
        return self.verdict == "ok"

    @property
    def label(self):
        if self.verdict == "railed":
            return "railed {:.0%}".format(self.railed)
        return self.verdict


def assess_channels(raw_uv, rail_uv=RAIL_UV):
    """Classify each electrode from UNFILTERED microvolts.

    Filtering hides the problem - a railed channel is pinned to the rail, so
    after a bandpass it looks like a quiet flat line rather than a broken one.

    `rail_uv` is the amplifier's saturation level and is specific to the
    hardware. Pass None for recordings from another system, where a Cyton
    threshold would flag perfectly good channels.
    """
    results = []
    scale = np.nanmedian([np.nanstd(x) for x in raw_uv]) if len(raw_uv) else 0.0
    for x in raw_uv:
        railed = (float(np.mean(np.abs(x) > rail_uv * 0.95))
                  if rail_uv else 0.0)
        std = float(np.nanstd(x))
        # railed is checked first: an electrode pinned hard at the rail has
        # no variance either, and calling that "dead" hides the real cause
        if not np.isfinite(x).all():
            verdict = "missing samples"
        elif railed > 0.15:
            verdict = "railed"
        elif std < 1e-6:
            verdict = "dead"
        elif scale > 0 and std > 20 * scale:
            # Relative to the other channels, so it works whatever the
            # recording's units and amplitude happen to be.
            verdict = "noisy"
        else:
            verdict = "ok"
        results.append(Quality(verdict, railed, std))
    return results


# ----------------------------------------------------------------------
# display scaling
# ----------------------------------------------------------------------


def nice_ceiling(value):
    """Round up to the next 1/2/5 x 10^n, so the axis stops jittering."""
    if value <= 0:
        return 1.0
    decade = 10.0 ** math.floor(math.log10(value))
    for step in (1.0, 2.0, 5.0):
        if value <= step * decade:
            return step * decade
    return 10.0 * decade


class AutoScaler:
    """Tracks a readable amplitude for the time-series view.

    Uses a high percentile rather than the max so one spike does not shrink
    everything else to a flat line, and eases toward the target so the axis
    does not flicker frame to frame.
    """

    def __init__(self, percentile=99.0, smoothing=0.2, floor=5.0):
        self.percentile = percentile
        self.smoothing = smoothing
        self.floor = floor
        self._value = None

    def reset(self):
        self._value = None

    def update(self, x):
        """Feed the currently displayed samples; returns amplitude in uV."""
        if x.size == 0:
            return nice_ceiling(self._value or self.floor)

        finite = np.abs(x[np.isfinite(x)])
        target = max(float(np.percentile(finite, self.percentile)), self.floor) if finite.size else self.floor
        if self._value is None:
            self._value = target
        else:
            self._value += self.smoothing * (target - self._value)
        return nice_ceiling(self._value)


class ChannelAutoScaler:
    """Independent robust scales, refreshed at most four times/second/channel."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.scalers = {}
        self.values = {}
        self.updated = {}

    def update(self, data, channels, now=None):
        now = time.perf_counter() if now is None else now
        for channel in channels:
            if channel not in self.scalers:
                self.scalers[channel] = AutoScaler()
            if channel not in self.updated or now - self.updated[channel] >= .25:
                self.values[channel] = self.scalers[channel].update(data[channel])
                self.updated[channel] = now
        return np.array([self.values[channel] for channel in channels], dtype=float)


# ----------------------------------------------------------------------
# spectrum
# ----------------------------------------------------------------------


def spectrum(signal, fs, fmax=60.0):
    """Welch PSD over roughly 2-second segments."""
    nperseg = min(len(signal), int(fs * 2))
    if nperseg < 32:
        return np.array([]), np.array([])
    if not np.isfinite(signal).all():
        return np.array([]), np.array([])
    freqs, power = welch_psd(signal, fs)
    keep = freqs <= fmax
    return freqs[keep], power[keep]


def band_powers(freqs, power):
    """Integrate the PSD over each named band."""
    out = []
    for _, low, high in BANDS:
        out.append(band_integral(freqs, power, low, high))
    return out
