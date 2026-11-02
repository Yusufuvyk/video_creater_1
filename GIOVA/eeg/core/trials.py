"""Trial analysis of one channel or a channel ROI, independent of Qt and plotting.

Times are recording seconds; intervals are half-open sample slices. Every
accepted trial contributes the same number of samples and the same weight.

An ROI (several channels) is combined in the power domain: every spectrum,
band power and feature is computed per channel and then averaged. Averaging the
waveforms first would cancel oscillations whose phase differs across the scalp.
The Signal view still shows the channel-mean waveform, and says so.
"""
from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.signal import detrend

from .config import BANDS
from .processing import filter_finite
from .spectral import sample_index, band_integral, welch_sizes, welch_psd


@dataclass
class Trial:
    id: str
    condition: str
    start: float
    end: float
    included: bool = True


@dataclass
class Settings:
    duration: float = 30.0
    window: float = 2.0
    overlap: float = 0.5
    filters: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)
    quality_mad: float = 6.0


@dataclass
class Metrics:
    fft_freqs: np.ndarray
    fft: np.ndarray
    psd_freqs: np.ndarray
    psd: np.ndarray
    power: float
    bands: dict


@dataclass
class TrialResult:
    trial: Trial
    signal: np.ndarray
    metrics: Metrics
    #: (n_channels, n_samples) and one Metrics per channel; None for one channel
    channel_signals: np.ndarray = None
    channel_metrics: list = None


@dataclass
class ConditionResult:
    name: str
    trial_ids: list
    signal: np.ndarray
    mean_signal: Metrics
    mean_trials: Metrics
    channel_signals: np.ndarray = None


@dataclass
class AnalysisResult:
    fs: float
    channel: str
    settings: Settings
    times: np.ndarray
    trials: list
    conditions: list
    rejected: dict
    quality: dict = field(default_factory=dict)
    features: object = None
    channels: list = field(default_factory=list)


def validate_settings(settings, fs):
    if not np.isfinite(settings.quality_mad) or settings.quality_mad < 0:
        raise ValueError("Amplitude MAD multiplier must be non-negative")
    from .feature_profiles import FeatureProfile
    FeatureProfile.from_dict(settings.profile)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    if not np.isfinite(settings.duration) or settings.duration <= 0:
        raise ValueError("Common duration must be positive and finite")
    if sample_index(settings.duration, fs) < 2:
        raise ValueError("Common duration must contain at least two samples")
    if not np.isfinite(settings.window) or settings.window <= 0:
        raise ValueError("Welch window must be positive and finite")
    if not np.isfinite(settings.overlap) or not 0 <= settings.overlap < 1:
        raise ValueError("Welch overlap must be between 0 and 1 (exclusive)")
    if settings.filters.get("detrend", "constant") not in DETRENDS:
        raise ValueError("Detrend must be 'constant' or 'linear'")
    if (settings.filters.get("reference") or "none") not in REFERENCES:
        raise ValueError("Reference must be none, average or channels")
    if settings.filters.get("reference") == "channels" and not settings.filters.get("reference_channels"):
        raise ValueError("Choose at least one reference channel")
    order = settings.filters.get("order", 4)
    if not isinstance(order, (int, float)) or int(order) != order or not 1 <= order <= 10:
        raise ValueError("Filter order must be an integer between 1 and 10")
    low, high = settings.filters.get("low"), settings.filters.get("high")
    if low is not None and (not np.isfinite(low) or not 0 < low < fs / 2):
        raise ValueError("Low cut must be below Nyquist")
    if high is not None and (not np.isfinite(high) or not 0 < high < fs / 2):
        raise ValueError("High cut must be below Nyquist")
    if low is not None and high is not None and low >= high:
        raise ValueError("Low cut must be below high cut")


def trial_status(trial, fs, n_samples, duration):
    """Return (start sample, sample count, reason); no silent padding/clipping."""
    if not trial.id.strip() or not trial.condition.strip():
        return 0, 0, "ID and condition are required"
    if not np.isfinite([trial.start, trial.end]).all():
        return 0, 0, "Invalid time"
    if trial.start < 0 or trial.end <= trial.start:
        return 0, 0, "Require 0 <= start < end"
    start, end = sample_index(trial.start, fs), sample_index(trial.end, fs)
    count = sample_index(duration, fs)
    if count < 2:
        return start, 0, "Common duration needs at least two samples"
    if trial.end > n_samples / fs + 1e-9 or start >= n_samples:
        return start, 0, "Outside recording"
    if end - start < count or start + count > n_samples:
        return start, max(end - start, 0), "Too short for common duration"
    return start, count, "Ready" if trial.included else "Excluded by selection"


def periodic_trials(start, phases, repeats, recording_duration):
    """Generate a labelled sequence; retain an incomplete tail as an invalid trial."""
    if not np.isfinite(start) or start < 0 or int(repeats) != repeats or repeats < 1:
        raise ValueError("Use a non-negative start and a positive repeat count")
    if not phases or any(not str(label).strip() or not np.isfinite(length) or length <= 0
                         for label, length in phases):
        raise ValueError("Each phase needs a condition and positive duration")
    trials, time = [], float(start)
    for _ in range(int(repeats)):
        for label, length in phases:
            if time >= recording_duration:
                return trials
            end = min(time + length, recording_duration)
            trials.append(Trial(str(len(trials) + 1), str(label).strip(), time, end))
            time += length
    return trials


#: Keys of Settings.filters that design the IIR chain; the rest are handled here.
DESIGN_KEYS = ("low", "high", "order", "notches", "notch_q")
DETRENDS = ("constant", "linear")
REFERENCES = ("none", "average", "channels")


def design_kwargs(filters):
    """The part of a filter dict that `processing.design_filter` takes."""
    return {key: filters[key] for key in DESIGN_KEYS if key in filters}


def reference_signal(data, settings, names=None):
    """The signal subtracted from every channel before filtering, or None.

    `average` is the common average of all channels in the file; `channels` the
    mean of the named reference channels (for example linked mastoids). Missing
    samples are ignored per time point so one damaged channel does not blank
    the whole reference.
    """
    mode = settings.filters.get("reference") or "none"
    if mode == "none":
        return None
    data = np.asarray(data, dtype=float)
    if mode == "average":
        rows = data
    else:
        wanted = list(settings.filters.get("reference_channels") or [])
        names = list(names or [])
        index = [names.index(name) for name in wanted if name in names]
        if not index:
            raise ValueError("Reference channels are not in this recording")
        rows = data[index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(rows, axis=0)


def prepare_channel(signal, fs, settings, reference=None):
    validate_settings(settings, fs)
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 2 or np.isfinite(signal).sum() < 2:
        raise ValueError("Channel must contain at least two finite samples")
    if reference is not None:
        signal = signal - reference
    return filter_finite(signal, fs, **design_kwargs(settings.filters))


def welch_plan(n_samples, fs, settings):
    return welch_sizes(n_samples, fs, settings.window, settings.overlap)


def measure(signal, fs, settings):
    signal = np.asarray(signal, dtype=float)
    if settings.filters.get("detrend", "constant") == "linear":
        x = detrend(signal, type="linear")
    else:
        x = signal - signal.mean()
    n = x.size
    amplitude = np.abs(np.fft.rfft(x)) / n
    # DC and, for even n, the Nyquist bin have no negative-frequency partner.
    amplitude[1:-1 if n % 2 == 0 else None] *= 2
    freqs, density = welch_psd(x, fs, settings.window, settings.overlap)
    return Metrics(np.fft.rfftfreq(n, 1 / fs), amplitude, freqs, density,
                   float(np.mean(x ** 2)),
                   {name: band_integral(freqs, density, low, high)
                    for name, low, high in BANDS})


def mean_metrics(items):
    """Equal-weight average of several Metrics on the same grids."""
    first = items[0]
    if len(items) == 1:
        return first
    return Metrics(first.fft_freqs, np.mean([m.fft for m in items], axis=0),
                   first.psd_freqs, np.mean([m.psd for m in items], axis=0),
                   float(np.mean([m.power for m in items])),
                   {key: float(np.mean([m.bands[key] for m in items])) for key in first.bands})


def channel_list(channel_index, n_channels):
    """One index or a list of them, validated, as a list of ints."""
    indices = [channel_index] if np.ndim(channel_index) == 0 else list(channel_index)
    if not indices or len(set(indices)) != len(indices) or any(
            int(i) != i or not 0 <= i < n_channels for i in indices):
        raise ValueError("Select one or more valid channels")
    return [int(i) for i in indices]


def analyze(data, fs, channel_index, channel_name, trials, settings, rail_uv=None, names=None):
    """`channel_index` is one index or a list of indices (an ROI)."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Select one valid channel")
    indices = channel_list(channel_index, data.shape[0])
    validate_settings(settings, fs)
    ids = [trial.id for trial in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("Trial IDs must be unique")
    reference = reference_signal(data, settings, names)
    continuous = np.vstack([prepare_channel(data[i], fs, settings, reference) for i in indices])
    roi = len(indices) > 1
    accepted, rejected = [], {}
    for trial in trials:
        start, count, reason = trial_status(trial, fs, data.shape[1], settings.duration)
        if reason != "Ready":
            rejected[trial.id] = reason
            continue
        block = continuous[:, start:start + count].copy()
        if not np.isfinite(block).all():
            rejected[trial.id] = "Missing samples or finite run too short for filtering"
            continue
        if roi:
            per_channel = [measure(row, fs, settings) for row in block]
            accepted.append(TrialResult(trial, block.mean(axis=0), mean_metrics(per_channel),
                                        block, per_channel))
        else:
            accepted.append(TrialResult(trial, block[0], measure(block[0], fs, settings)))
    if not accepted:
        raise ValueError("No selected, valid trials. Check duration and trial boundaries.")
    conditions = []
    for name in dict.fromkeys(item.trial.condition for item in accepted):
        group = [item for item in accepted if item.trial.condition == name]
        signal = np.mean([item.signal for item in group], axis=0)
        averaged = mean_metrics([r.metrics for r in group])
        if roi:
            channel_means = np.mean([item.channel_signals for item in group], axis=0)
            of_mean = mean_metrics([measure(row, fs, settings) for row in channel_means])
        else:
            channel_means, of_mean = None, measure(signal, fs, settings)
        conditions.append(ConditionResult(name, [r.trial.id for r in group], signal,
                                          of_mean, averaged, channel_means))
    result = AnalysisResult(float(fs), channel_name, settings,
                            np.arange(accepted[0].signal.size) / fs,
                            accepted, conditions, rejected,
                            channels=list(indices))
    peaks = np.array([np.ptp(t.channel_signals, axis=1).max() if roi else np.ptp(t.signal)
                      for t in accepted])
    median = np.median(peaks)
    threshold = median + settings.quality_mad * np.median(np.abs(peaks - median))
    labels = list(names) if names is not None else None
    for index, entry in enumerate(accepted):
        start = sample_index(entry.trial.start, fs)
        flags = []
        for channel in indices:
            raw = data[channel, start:start + len(entry.signal)]
            where = " ({})".format(labels[channel]) if roi and labels else ""
            if np.std(raw) < 1e-6:
                flags.append("Flat signal" + where)
            if rail_uv is not None and np.mean(np.abs(raw) >= .95 * rail_uv) > .15:
                flags.append("Amplifier saturation" + where)
        if len(accepted) >= 5 and peaks[index] > threshold:
            flags.append("High amplitude (median + {:g} MAD)".format(settings.quality_mad))
        result.quality[entry.trial.id] = flags
    from .feature_profiles import FeatureProfile, extract_trials
    result.features = extract_trials(result, FeatureProfile.from_dict(settings.profile))
    return result
