"""Event-locked epochs across many channels: ERP, PSD, time-frequency, ERD/ERS,
topography and condition statistics.

The Analysis page answers "how do these labelled blocks differ?" for one
channel or an ROI. This module answers the question motor-imagery and other
cued datasets are built for: what happens, channel by channel, relative to each
event - and how does it differ between conditions (left vs right hand)?

Pipeline, in order, so nothing leaks between trials:

    continuous data -> re-reference -> zero-phase filters      (whole recording)
    -> cut epochs [onset + tmin, onset + tmax)                  (equal length)
    -> reject (outside recording, missing samples, amplitude)   (reported)
    -> per epoch: STFT or Morlet power -> smoothing in time     (never across epochs)
    -> baseline power per epoch from a window inside the epoch
    -> ERD% = 100 (P - R) / R, pooled (mean P vs mean R) by default

Pooled is the default because ERD% is bounded below at -100% and unbounded
above: averaging per-trial percentages is pulled up by trials with a small
baseline (on Cho2017 s01/C3: -50.6% pooled vs -12.3% as a mean of ratios).

Pure numpy/scipy (+ MNE for Morlet) - no Qt.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.signal import get_window, spectrogram
from scipy.stats import ttest_ind

from .processing import filter_finite
from .spectral import sample_index, welch_psd
from .trials import Settings, design_kwargs, reference_signal, validate_settings

STFT, MORLET, CWT = "stft", "morlet", "cwt"
#: How the Workbench offers them. Morlet and CWT are one transform, complex Morlet wavelets, and differ only in how a
#: wavelet's width is given: so there is one "wavelet" method, its width either some cycles per Hz of each
#: frequency (MORLET) or one complex Morlet cmorB-C for every frequency (CWT).
WAVELET = "wavelet"
PER_HZ, FIXED = "cycles per Hz", "one wavelet"
WAVELET_WIDTHS = (PER_HZ, FIXED)


def tf_method(method, width=PER_HZ):
    """The computation a method and a wavelet width name: STFT, MORLET or CWT (the older names pass as they are)."""
    if method == WAVELET:
        if width not in WAVELET_WIDTHS:
            raise ValueError("A wavelet's width is given as {} or {}".format(*WAVELET_WIDTHS))
        return MORLET if width == PER_HZ else CWT
    return method


#: Frequencies a CWT is computed at, evenly from the lowest to the highest (as the lab's scalogram).
CWT_FREQS = 128
POOLED, MEAN_OF_TRIALS = "pooled", "mean_of_trials"
#: What the head map and the statistics are window means of: a separate STFT of each electrode (fast on many
#: electrodes), or the maps' own transform (the method chosen, not smoothed).
SCALP_STFT, SCALP_MAPS = "stft", "maps"
SCALP_SOURCES = (SCALP_STFT, SCALP_MAPS)
#: What ERD is relative to: a window inside every trial (cued designs), or the
#: mean power of a reference condition (block designs: eyes closed vs open).
WINDOW, CONDITION = "window", "condition"


# ----------------------------------------------------------------------
# events
# ----------------------------------------------------------------------


class EventTable:
    """Onsets (s from the first sample), durations (s) and labels."""

    SUFFIX = ".events.json"

    def __init__(self, onsets=(), durations=None, labels=()):
        self.onsets = np.asarray(onsets, dtype=float).reshape(-1)
        self.durations = (np.zeros(self.onsets.size) if durations is None
                          else np.asarray(durations, dtype=float).reshape(-1))
        self.labels = [str(label) for label in labels]
        if not (self.onsets.size == self.durations.size == len(self.labels)):
            raise ValueError("Event onsets, durations and labels must have the same length")
        if not np.isfinite(self.onsets).all() or not np.isfinite(self.durations).all():
            raise ValueError("Event times must be finite")
        order = np.argsort(self.onsets, kind="stable")
        self.onsets, self.durations = self.onsets[order], self.durations[order]
        self.labels = [self.labels[i] for i in order]

    def __len__(self):
        return len(self.labels)

    def __eq__(self, other):
        return (isinstance(other, EventTable) and self.labels == other.labels
                and np.allclose(self.onsets, other.onsets) and np.allclose(self.durations, other.durations))

    def unique_labels(self):
        return list(dict.fromkeys(self.labels))

    def counts(self):
        return {label: self.labels.count(label) for label in self.unique_labels()}

    #: labels named one by one before the summary falls back to a count
    MAX_DESCRIBED = 6

    def describe(self):
        if not len(self):
            return "no events"
        counts = self.counts()
        named = ", ".join("{} (×{})".format(label, n)
                          for label, n in list(counts.items())[:self.MAX_DESCRIBED])
        more = len(counts) - self.MAX_DESCRIBED
        return "{} {} · {}{}".format(len(counts), "code" if len(counts) == 1 else "codes",
                                     named, ", …" if more > 0 else "")

    def relabel(self, mapping):
        """Rename labels; a label mapped to '' or None is dropped."""
        keep = [i for i, label in enumerate(self.labels) if mapping.get(label, label)]
        return EventTable(self.onsets[keep], self.durations[keep],
                          [mapping.get(self.labels[i], self.labels[i]) for i in keep])

    def to_dict(self):
        return dict(version=1, onsets=self.onsets.tolist(), durations=self.durations.tolist(),
                    labels=list(self.labels))

    @classmethod
    def from_dict(cls, raw):
        try:
            return cls(raw["onsets"], raw.get("durations"), raw["labels"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed event table") from exc

    @classmethod
    def from_blocks(cls, blocks):
        return cls([b.start for b in blocks], [b.end - b.start for b in blocks], [b.label for b in blocks])

    @classmethod
    def path_for(cls, recording_path):
        path = Path(recording_path)
        return path.with_name(path.stem + cls.SUFFIX)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read events: {}".format(exc)) from exc


def events_for(recording, path):
    """(EventTable, source text) with the saved code-to-name map applied.

    Sources, in order: events the loader read from the file itself, an
    `.events.json` sidecar, a unix-time marker CSV, a written-down paradigm
    (one event per whole trial).
    """
    table, source = raw_events_for(recording, path)
    if table is not None and path:
        mapping = load_label_map(path)
        if mapping:
            table = table.relabel(mapping)
    return table, source


MAP_SUFFIX = ".eventmap.json"


def map_path(recording_path):
    """Where the event code -> condition name map of a recording lives."""
    path = Path(recording_path)
    return path.with_name(path.stem + MAP_SUFFIX)


def save_label_map(recording_path, mapping):
    map_path(recording_path).write_text(json.dumps({"labels": dict(mapping)}, indent=2), encoding="utf-8")


def load_label_map(recording_path):
    path = map_path(recording_path)
    if not path.is_file():
        return None
    try:
        mapping = json.loads(path.read_text(encoding="utf-8")).get("labels")
    except (OSError, ValueError, AttributeError):
        return None  # a corrupt sidecar should not block the recording
    return {str(k): str(v) for k, v in mapping.items()} if isinstance(mapping, dict) else None


def raw_events_for(recording, path):
    """(EventTable with the labels as stored, source text), else (None, reason)."""
    from . import config, events as ev, paradigm as pdm
    table = getattr(recording, "events", None)
    if table is not None and not len(table):
        table = None
    if table is not None and not getattr(recording, "events_are_raw_markers", False):
        return table, "events stored in the file"
    # Marker-channel codes are placed correctly but unlabelled: a sidecar that
    # names the same events wins, and the codes are the fallback.
    if path:
        sidecar = EventTable.path_for(path)
        if sidecar.is_file():
            try:
                return EventTable.load(sidecar), sidecar.name
            except ValueError:
                pass
        markers = config.events_for(path)
        if markers is not None and recording.start_time is not None:
            relative = [(recording.marker_seconds(t), label) for t, label in ev.load_events(markers)
                        if np.isfinite(t)]
            blocks = ev.to_blocks(relative, 0, recording.duration)
            if blocks:
                return EventTable.from_blocks(blocks), markers.name
        paradigm = pdm.load_for(path)
        if paradigm is not None:
            blocks = ev.blocks_by_cycle(paradigm, recording.duration)
            if blocks:
                return EventTable.from_blocks(blocks), pdm.path_for(path).name
    if table is not None:
        return table, "marker channel in the file"
    return None, "no events: load a file with events, or a marker / paradigm sidecar"


# ----------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------


@dataclass
class EpochSettings:
    conditions: list = field(default_factory=list)   # empty = every label
    tmin: float = -1.0
    tmax: float = 4.0
    baseline: list = field(default_factory=lambda: [-1.0, 0.0])
    active: list = field(default_factory=lambda: [0.5, 2.5])
    reject_uv: float = 0.0                           # 0 = off; |x - epoch mean| limit
    method: str = STFT
    fmin: float = 4.0
    fmax: float = 40.0
    window: float = 0.5                              # STFT window, s
    overlap: float = 0.9
    n_cycles: float = 0.5                            # Morlet cycles per Hz, clipped to 2..12
    wavelet: str = "cmor1.5-1.0"                     # the CWT's complex Morlet (pywt name)
    smooth: float = 0.5                              # s of temporal smoothing, 0 = off
    erd_mode: str = POOLED
    bands: list = field(default_factory=lambda: [["mu", 8.0, 13.0], ["beta", 13.0, 30.0]])
    filters: dict = field(default_factory=dict)
    topography: bool = True
    baseline_mode: str = WINDOW
    reference: str = ""                              # condition used when baseline_mode == CONDITION
    scalp: str = SCALP_STFT                          # what the head map and statistics are window means of

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        known = set(cls.__dataclass_fields__)
        try:
            return cls(**{k: v for k, v in (raw or {}).items() if k in known})
        except TypeError as exc:
            raise ValueError("Malformed epoch settings") from exc

    def validate(self, fs):
        numbers = [self.tmin, self.tmax, *self.baseline, *self.active, self.reject_uv,
                   self.fmin, self.fmax, self.window, self.overlap, self.n_cycles, self.smooth]
        if not np.isfinite(numbers).all():
            raise ValueError("Epoch settings must be finite numbers")
        if self.tmax <= self.tmin:
            raise ValueError("Epoch end must be after its start")
        if self.baseline_mode not in (WINDOW, CONDITION):
            raise ValueError("Baseline must be a window or a reference condition")
        if self.baseline_mode == CONDITION and not str(self.reference).strip():
            raise ValueError("Choose the reference condition")
        windows = (("Active window", self.active),) if self.baseline_mode == CONDITION else (
            ("Baseline", self.baseline), ("Active window", self.active))
        for name, (low, high) in windows:
            if not self.tmin <= low < high <= self.tmax:
                raise ValueError("{} must lie inside the epoch ({:g} to {:g} s)".format(name, self.tmin, self.tmax))
        if self.method not in (STFT, MORLET, CWT):
            raise ValueError("Method must be STFT, Morlet or CWT")
        if self.method == CWT and not str(self.wavelet).startswith("cmor"):
            raise ValueError("The CWT needs a complex Morlet, e.g. cmor1.5-1.0")
        if self.erd_mode not in (POOLED, MEAN_OF_TRIALS):
            raise ValueError("ERD mode must be pooled or mean of trials")
        if self.scalp not in SCALP_SOURCES:
            raise ValueError("The head map is computed from {} or {}".format(*SCALP_SOURCES))
        if not 0 <= self.fmin < self.fmax or self.fmax > fs / 2:
            raise ValueError("Frequency range must be 0 <= low < high <= Nyquist ({:g} Hz)".format(fs / 2))
        if self.window <= 0 or not 0 <= self.overlap < 1 or self.n_cycles <= 0 or self.smooth < 0 or self.reject_uv < 0:
            raise ValueError("Window, cycles and smoothing must be positive; overlap in [0, 1)")
        names = [b[0] for b in self.bands]
        if not self.bands or len(set(names)) != len(names):
            raise ValueError("Give at least one band, with unique names")
        for name, low, high in self.bands:
            if not str(name).strip() or not 0 <= low < high:
                raise ValueError("Each band needs a name and 0 <= low < high")
        validate_settings(Settings(duration=self.tmax - self.tmin, filters=dict(self.filters)), fs)


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------


@dataclass
class ChannelTF:
    name: str
    index: int
    freqs: np.ndarray
    times: np.ndarray            # frame centres, s from event onset
    power: dict                  # condition -> (n_epochs, n_freqs, n_times), smoothed
    baseline: dict               # condition -> (n_epochs, n_freqs)
    units: str
    #: condition -> (n_freqs, n_times): amplitude in µV per trial, averaged over trials, not smoothed
    amplitude: dict = field(default_factory=dict)


@dataclass
class EpochResult:
    fs: float
    settings: EpochSettings
    names: list                  # every channel name in the recording
    channels: list               # analysed channel indices
    conditions: list
    times: np.ndarray            # sample times, s from onset
    epochs: list                 # (id, label, onset, status)
    epoch_ids: dict              # condition -> [id]
    evoked: dict                 # condition -> (mean, sem) each (n_channels, n_times)
    psd_freqs: np.ndarray
    psd: dict                    # condition -> (mean, sem) each (n_channels, n_freqs), active window
    tf: list                     # ChannelTF per analysed channel
    topo_channels: list = field(default_factory=list)   # channel indices in `topo`
    topo: dict = field(default_factory=dict)             # band -> condition -> (n_epochs, n_topo) ERD%
    topo_pooled: dict = field(default_factory=dict)      # band -> condition -> (n_topo,) pooled ERD%
    topo_power: dict = field(default_factory=dict)       # band -> condition -> (n_topo,) mean band power
    topo_units: str = "µV²/Hz"                           # of topo_power: the scalp transform's units

    @property
    def rejected(self):
        return [(i, label, status) for i, label, _, status in self.epochs if status != "Used"]


# ----------------------------------------------------------------------
# computation
# ----------------------------------------------------------------------


def _percent(power, reference):
    reference = np.where(reference > 0, reference, np.nan)
    return 100.0 * (power - reference) / reference


def smooth_power(power, step, smooth_seconds):
    """Smooth power along time, within the block it is given and no further.

    Square first, then smooth - the other order averages the waveform rather
    than its envelope, and an oscillation averages to nothing.

    The caller decides what "the block" is, and for epochs that must be one
    epoch at a time. Smoothing the whole continuous recording instead lets an
    epoch's first column borrow from the end of the previous trial, and on
    Cho2017 that inflated the 0.5-1.5 s baseline enough to read -68% where the
    honest answer is -51%.
    """
    if not smooth_seconds or step <= 0 or power.shape[-1] < 2:
        return power
    width = max(int(round(smooth_seconds / step)), 1)
    if width <= 1:
        return power
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(power, size=width, axis=-1, mode="nearest")


def _frames(times, window):
    mask = (times >= window[0] - 1e-9) & (times <= window[1] + 1e-9)
    if not mask.any():
        raise ValueError("The {:g}-{:g} s window holds no time-frequency frame; widen it, "
                         "or use a shorter window / more overlap".format(*window))
    return mask


def _stft_nperseg(n_samples, fs, settings):
    return min(n_samples, max(8, sample_index(settings.window, fs)))


def stft_power(epochs, fs, settings):
    """(freqs, times from onset, power (n_epochs, n_freqs, n_times)) in µV²/Hz."""
    nperseg = _stft_nperseg(epochs.shape[-1], fs, settings)
    noverlap = min(int(nperseg * settings.overlap), nperseg - 1)
    freqs, times, power = spectrogram(epochs, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                                      scaling="density", mode="psd", detrend="constant", axis=-1)
    keep = (freqs >= settings.fmin) & (freqs <= settings.fmax)
    if keep.sum() < 1:
        raise ValueError("A {:g} s window resolves {:g} Hz; no bin falls in {:g}-{:g} Hz".format(
            nperseg / fs, fs / nperseg, settings.fmin, settings.fmax))
    return freqs[keep], times + settings.tmin, power[:, keep, :], "µV²/Hz"


def morlet_power(epochs, fs, settings):
    """Morlet power in µV², decimated to about 50 frames per second.

    Scaled to mean-square power: a sine of amplitude A at a wavelet's frequency
    reads A²/2, whatever the sampling rate. MNE's wavelets have unit energy
    instead, so its raw output grows with the wavelet's length in samples (a
    10 µV sine read 1763 at 125 Hz and 7222 at 512 Hz); the factor 2 / (Σ|w|)²
    per frequency removes that. Ratios (ERD %) are the same either way.
    """
    from mne.time_frequency import morlet, tfr_array_morlet
    step = max(1, (settings.fmax - max(settings.fmin, 1.0)) / 40)
    freqs = np.arange(max(settings.fmin, 1.0), settings.fmax + step / 2, step)
    cycles = np.clip(freqs * settings.n_cycles, 2.0, 12.0)
    duration = epochs.shape[-1] / fs
    fits = 5.0 * cycles / (np.pi * freqs) <= duration
    if not fits.any():
        raise ValueError("The epoch is too short for any wavelet in this range")
    freqs, cycles = freqs[fits], cycles[fits]
    decim = max(1, int(fs // 50))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        power = tfr_array_morlet(epochs[:, None, :], sfreq=fs, freqs=freqs, n_cycles=cycles,
                                 output="power", decim=decim, verbose="error")[:, 0]
        gain = np.array([np.abs(w).sum() for w in morlet(fs, freqs, n_cycles=cycles)])
    power = power * (2.0 / gain ** 2)[None, :, None]
    times = settings.tmin + np.arange(epochs.shape[-1])[::decim] / fs
    return freqs, times, power, "µV²"


def cwt_power(epochs, fs, settings):
    """Complex Morlet CWT power, µV², at CWT_FREQS frequencies, decimated to about 50 frames per second.

    Mean-square power A²/2 of a sine of amplitude A, as Morlet's; the amplitude
    itself is core.wavelets.cwt_amplitude, the lab scalogram's arithmetic.
    """
    from .wavelets import cwt_amplitude
    freqs = np.linspace(max(settings.fmin, 1.0), settings.fmax, CWT_FREQS)
    decim = max(1, int(fs // 50))
    power = np.stack([cwt_amplitude(x, fs, freqs, settings.wavelet)[:, ::decim] for x in epochs]) ** 2 / 2.0
    times = settings.tmin + np.arange(epochs.shape[-1])[::decim] / fs
    return freqs, times, power, "µV²"


def amplitude_from_power(power, fs, settings, n_samples):
    """Amplitude in µV from unsmoothed power: a sine of amplitude A reads A.

    STFT frames are a one-sided Hann density P = 2 |Σ x w|² / (fs Σw²), so the
    amplitude 2 |Σ x w| / Σw is sqrt(2 P fs Σw²) / Σw. Morlet and CWT power are
    already the mean square A²/2.
    """
    if settings.method in (MORLET, CWT):
        return np.sqrt(2.0 * power)
    window = get_window("hann", _stft_nperseg(n_samples, fs, settings))
    return np.sqrt(2.0 * power * fs * np.sum(window ** 2)) / np.sum(window)


def _time_frequency(epochs, fs, settings):
    return {MORLET: morlet_power, CWT: cwt_power}.get(settings.method, stft_power)(epochs, fs, settings)


def wavelet_reach(fs, settings):
    """Samples a wavelet method reaches beyond each end of a trial, at its lowest frequency (0 for STFT).

    A wavelet convolves the whole trial: without these samples it sees zeros past the ends and the edges
    swell (tests/test_tf_edges.py). Morlet: MNE's wavelets span 5 standard deviations each side, sigma =
    cycles / (2 pi f). CWT: the complex Morlet's Gaussian exp(-u^2/B) is below 1 % past 3 sqrt(B/2) units of
    C / f seconds. Rounded up to whole decimation steps, so the frames keep the times they had.
    """
    low = max(settings.fmin, 1.0)
    if settings.method == MORLET:
        seconds = 5.0 * float(np.clip(low * settings.n_cycles, 2.0, 12.0)) / (2 * np.pi * low)
    elif settings.method == CWT:
        from .wavelets import cmor_parameters
        bandwidth, centre = cmor_parameters(settings.wavelet)
        seconds = 3.0 * np.sqrt(bandwidth / 2.0) * centre / low
    else:
        return 0
    decim = max(1, int(fs // 50))
    return int(np.ceil(seconds * fs / decim)) * decim


def padded_trials(x, starts, n, reach):
    """Each trial x[start:start + n] with `reach` samples more on each side: the recording's own samples where it
    has them (finite), the trial mirrored where it does not."""
    out = np.empty((len(starts), n + 2 * reach))
    for row, start in enumerate(starts):
        trial = x[start:start + n]
        before = x[max(0, start - reach):start]
        after = x[start + n:start + n + reach]
        if before.size < reach or not np.isfinite(before).all():
            before = np.pad(trial, (reach, 0), mode="reflect")[:reach] if reach < n else np.zeros(reach)
        if after.size < reach or not np.isfinite(after).all():
            after = np.pad(trial, (0, reach), mode="reflect")[-reach:] if reach < n else np.zeros(reach)
        out[row] = np.concatenate([before, trial, after])
    return out


def window_mean_settings(settings):
    """The transform every electrode's window means (the head map, the statistics) come from.

    By default (scalp SCALP_STFT) an STFT with the block's window, at most 50 % overlap and no smoothing, whatever
    the method of the detail maps: only means over the baseline and active windows are taken, it is fast on 64
    electrodes, and a band ratio does not depend on the PSD-vs-power unit difference. With SCALP_MAPS, the maps'
    own transform, not smoothed: one transform for everything shown, slower with wavelets on many electrodes."""
    if settings.scalp == SCALP_MAPS:
        return EpochSettings(**{**settings.to_dict(), "smooth": 0.0})
    return EpochSettings(**{**settings.to_dict(), "method": STFT, "overlap": min(settings.overlap, 0.5),
                            "smooth": 0.0})


def window_means_text(settings, topography=None):
    """Where the window means come from, in words: one sentence for the methods, the captions and Help."""
    stft = window_mean_settings(settings)
    where = "every electrode" if (settings.topography if topography is None else topography) else \
        "each analysed channel"
    if settings.scalp == SCALP_MAPS:
        return ("The head map and the statistics are window means of the maps' own transform ({}), not smoothed, "
                "on {}.".format("STFT" if settings.method == STFT else "complex Morlet wavelets", where))
    same = settings.method == STFT and stft.overlap == settings.overlap and not settings.smooth
    if same:
        return "The head map and the statistics are window means of the same STFT, on {}.".format(where)
    return ("The head map and the statistics are window means of a separate STFT of {} ({:g} s Hann window, "
            "{:.0%} overlap, not smoothed), whatever the method of the maps.".format(where, stft.window, stft.overlap))


#: How the head maps place the electrodes: by name, on a template, never on this head's measured positions.
TEMPLATE_POSITIONS = ("Electrodes are drawn at the standard 10-05 positions of their names (MNE's template), "
                      "not at positions measured on this head.")


def _band_rows(freqs, low, high):
    rows = (freqs >= low) & (freqs <= high)
    if not rows.any():
        raise ValueError("No frequency bin between {:g} and {:g} Hz at this resolution".format(low, high))
    return rows


def compute_epochs(data, fs, names, channels, events, settings, progress=None, ids=None, prepared=False):
    """Time-frequency, ERD, topography and statistics for one set of settings.

    `events` holds each trial's time zero; `ids` (optional) names the trials,
    e.g. the IDs of the analysis page's trial table. `prepared`: the data are
    already referenced and filtered as wanted, so they are used as they are.
    """
    data = np.asarray(data, dtype=float)
    fs = float(fs)
    if data.ndim != 2:
        raise ValueError("Data must be channels x samples")
    if not channels:
        raise ValueError("Select at least one channel")
    if any(not 0 <= c < data.shape[0] for c in channels):
        raise ValueError("Invalid channel selection")
    if events is None or not len(events):
        raise ValueError("This recording has no events to epoch")
    settings.validate(fs)
    wanted = list(settings.conditions) or events.unique_labels()
    conditions = [c for c in events.unique_labels() if c in wanted]
    if not conditions:
        raise ValueError("None of the chosen conditions occur in the events")

    n = sample_index(settings.tmax - settings.tmin, fs)
    offset = sample_index(settings.tmin, fs)
    times = settings.tmin + np.arange(n) / fs

    topo_channels = list(range(data.shape[0])) if settings.topography else list(channels)
    needed = sorted(set(topo_channels) | set(channels))
    common = None if prepared else reference_signal(data, Settings(filters=settings.filters), names)
    design = design_kwargs(settings.filters)

    # Cut first on raw indices so rejection is decided once for all channels.
    candidates = []
    for i, (onset, label) in enumerate(zip(events.onsets, events.labels)):
        if label in conditions:
            candidates.append((str(ids[i]) if ids is not None else str(i + 1), label, float(onset),
                               sample_index(onset, fs) + offset))

    def filtered(c):
        if prepared:
            return data[c]
        x = data[c] if common is None else data[c] - common
        return filter_finite(x, fs, **design)

    # Decide every epoch's fate once. Missing samples are judged on the raw data
    # of every channel involved; amplitude only on the analysed channels, which
    # are the only ones filtered and kept here. Topography-only electrodes are
    # filtered one at a time later, so a 64-electrode file is never held
    # filtered in memory and each channel is filtered exactly once.
    status = {eid: "Used" for eid, _, _, _ in candidates}
    finite_rows = np.isfinite(data[needed])
    for eid, _, _, start in candidates:
        if start < 0 or start + n > data.shape[1]:
            status[eid] = "Outside recording"
        elif not finite_rows[:, start:start + n].all():
            bad = needed[int(np.argmin(finite_rows[:, start:start + n].all(axis=1)))]
            status[eid] = "Missing samples ({})".format(names[bad])
    del finite_rows
    kept = {}
    for count, c in enumerate(channels):
        if progress:
            progress("Filtering {} ({}/{})".format(names[c], count + 1, len(channels)))
        x = kept[c] = filtered(c)
        for eid, _, _, start in candidates:
            if status[eid] != "Used":
                continue
            segment = x[start:start + n]
            if not np.isfinite(segment).all():
                status[eid] = "Missing samples ({})".format(names[c])
            elif settings.reject_uv > 0 and np.max(np.abs(segment - segment.mean())) > settings.reject_uv:
                status[eid] = "Amplitude > {:g} µV ({})".format(settings.reject_uv, names[c])
    used = {cond: [(eid, start) for eid, label, _, start in candidates if label == cond and status[eid] == "Used"]
            for cond in conditions}
    conditions = [c for c in conditions if used[c]]
    if not conditions:
        reasons = sorted(set(status.values()) - {"Used"})
        raise ValueError("No usable epochs. " + "; ".join(reasons[:4]))
    by_condition = settings.baseline_mode == CONDITION
    if by_condition and settings.reference not in conditions:
        raise ValueError("The reference condition '{}' has no usable trials".format(settings.reference))

    def stack(x, cond):
        return np.stack([x[start:start + n] for _, start in used[cond]])

    evoked, psd = {}, {}
    active = (sample_index(settings.active[0] - settings.tmin, fs),
              sample_index(settings.active[1] - settings.tmin, fs))
    psd_freqs = None
    for cond in conditions:
        blocks = np.stack([stack(kept[c], cond) for c in channels], axis=1)       # (n_ep, n_ch, n_t)
        k = blocks.shape[0]
        evoked[cond] = (blocks.mean(axis=0),
                        blocks.std(axis=0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(blocks.shape[1:]))
        segment = blocks[:, :, active[0]:active[1]]
        if segment.shape[-1] < 4:
            raise ValueError("The active window is too short for a PSD")
        psd_freqs, spectra = welch_psd(segment, fs, min(2.0, segment.shape[-1] / fs), 0.5)
        psd[cond] = (spectra.mean(axis=0),
                     spectra.std(axis=0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(spectra.shape[1:]))

    scalp_settings = window_mean_settings(settings)

    def decompose(x, cond, method_settings, amplitudes=None):
        reach = wavelet_reach(fs, method_settings)
        if reach:
            # wavelets see the samples around each trial, then are cut back to it
            wide = EpochSettings(**{**method_settings.to_dict(), "tmin": method_settings.tmin - reach / fs})
            freqs, frame_times, power, units = _time_frequency(
                padded_trials(x, [start for _, start in used[cond]], n, reach), fs, wide)
            inside = (frame_times >= settings.tmin - 1e-9) & (frame_times < settings.tmin + n / fs - 1e-9)
            frame_times, power = frame_times[inside], power[:, :, inside]
        else:
            freqs, frame_times, power, units = _time_frequency(stack(x, cond), fs, method_settings)
        if amplitudes is not None:
            # Per trial, then averaged: the induced response. Averaging the
            # signals first would cancel anything not phase-locked to the event.
            amplitudes[cond] = amplitude_from_power(power, fs, method_settings, n).mean(axis=0).astype(np.float32)
        step = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.0
        power = smooth_power(power, step, method_settings.smooth)
        base = None if by_condition else power[:, :, _frames(frame_times, settings.baseline)].mean(axis=2)
        return freqs, frame_times, power, base, units

    def with_reference(spectra):
        """Condition baseline: every trial is measured against the reference
        condition's mean power in the active window (per frequency)."""
        if not by_condition:
            return spectra
        freqs, frame_times, power, _, _ = spectra[settings.reference]
        reference = power[:, :, _frames(frame_times, settings.active)].mean(axis=(0, 2))
        return {cond: (f, t, p, np.broadcast_to(reference, p.shape[:2]).copy(), u)
                for cond, (f, t, p, _, u) in spectra.items()}

    tf = []
    topo_units = "µV²/Hz"
    topo_values = {band[0]: {cond: [] for cond in conditions} for band in settings.bands}
    topo_pooled = {band[0]: {cond: [] for cond in conditions} for band in settings.bands}
    topo_power = {band[0]: {cond: [] for cond in conditions} for band in settings.bands}
    for count, c in enumerate(needed):
        if progress:
            progress("Time-frequency {} ({}/{})".format(names[c], count + 1, len(needed)))
        x = kept[c] if c in kept else filtered(c)
        if c in channels:
            amplitudes = {}
            detail = with_reference({cond: decompose(x, cond, settings, amplitudes) for cond in conditions})
            first = detail[conditions[0]]
            tf.append(ChannelTF(names[c], c, first[0], first[1],
                                {cond: d[2].astype(np.float32) for cond, d in detail.items()},
                                {cond: d[3] for cond, d in detail.items()}, first[4], amplitudes))
        if c not in topo_channels:
            continue
        spectra = with_reference({cond: decompose(x, cond, scalp_settings) for cond in conditions})
        for cond in conditions:
            freqs, frame_times, power, base, topo_units = spectra[cond]
            act = _frames(frame_times, settings.active)
            for name, low, high in settings.bands:
                rows = _band_rows(freqs, low, high)
                p_band = power[:, rows][:, :, act].mean(axis=(1, 2))
                r_band = base[:, rows].mean(axis=1)
                topo_values[name][cond].append(_percent(p_band, r_band))
                topo_pooled[name][cond].append(float(_percent(p_band.mean(), r_band.mean())))
                topo_power[name][cond].append(float(p_band.mean()))
    topo = {name: {cond: np.stack(values[cond], axis=1) for cond in conditions}
            for name, values in topo_values.items()}
    pooled = {name: {cond: np.asarray(values[cond]) for cond in conditions}
              for name, values in topo_pooled.items()}

    epochs = [(eid, label, onset, status[eid]) for eid, label, onset, _ in candidates]
    return EpochResult(fs, settings, list(names), list(channels), conditions, times, epochs,
                       {cond: [eid for eid, _ in used[cond]] for cond in conditions},
                       evoked, psd_freqs, psd, tf, topo_channels, topo, pooled,
                       {name: {cond: np.asarray(v[cond]) for cond in conditions} for name, v in topo_power.items()},
                       topo_units)


def compute_trials(trials, fs, names, labels, tmin, channels, settings, ids=None, rejected=(), progress=None):
    """`compute_epochs` for trials already cut (n_trials, n_channels, n_samples), used as they are.

    Whatever was done to the trials before (baseline, a filter applied trial by
    trial, trials left out by hand) is kept: nothing is filtered or referenced
    again. The trials are laid end to end with an event at each one's first
    sample; every STFT or wavelet still sees one trial only, because each trial
    is cut out again before it is transformed. `settings.tmin` / `tmax` are
    replaced by the trials' own span. `rejected` [(id, label, reason)] are trials
    left out before, listed in the result's trial table.
    """
    trials = np.asarray(trials, dtype=float)
    if trials.ndim != 3 or not trials.shape[0]:
        raise ValueError("Trials must be trials x channels x samples")
    fs = float(fs)
    n = trials.shape[-1]
    settings = EpochSettings(**{**settings.to_dict(), "tmin": float(tmin), "tmax": float(tmin) + n / fs,
                                "filters": {}})
    data = trials.transpose(1, 0, 2).reshape(trials.shape[1], -1)
    # an event at onset - tmin puts each trial's first sample at k * n exactly
    starts = np.arange(trials.shape[0]) * n
    events = EventTable(starts / fs - settings.tmin, np.full(trials.shape[0], n / fs), labels)
    offset = sample_index(settings.tmin, fs)
    if [sample_index(t, fs) + offset for t in events.onsets] != starts.tolist():
        raise ValueError("The trials' start ({:g} s) is not a whole sample at {:g} Hz".format(settings.tmin, fs))
    result = compute_epochs(data, fs, names, channels, events, settings, progress=progress,
                            ids=list(ids) if ids is not None else None, prepared=True)
    result.epochs += [(str(eid), label, float("nan"), reason) for eid, label, reason in rejected]
    return result


# ----------------------------------------------------------------------
# views derived from a result
# ----------------------------------------------------------------------


def power_map(channel, condition, db=False):
    """Mean power (n_freqs, n_times); dB re 1 unit when asked."""
    mean = channel.power[condition].mean(axis=0)
    return 10 * np.log10(np.maximum(mean, 1e-12)) if db else mean


def amplitude_map(channel, condition):
    """Amplitude in µV, (n_freqs, n_times): each trial's amplitude, averaged over trials."""
    return channel.amplitude[condition]


#: What a time-frequency map shows.
ERD_MAP, POWER_MAP, AMPLITUDE_MAP = "erd", "power", "amplitude"


def trial_map(result, channel, condition, trial, kind=ERD_MAP):
    """One trial's map (n_freqs, n_times); `trial` counts that condition's used trials from 0.

    ERD against the trial's own baseline (or the reference condition), power in
    dB, or amplitude in µV. Power is the smoothed power every view uses, so the
    amplitude of one trial is the raw |STFT| (or |wavelet|) only when the
    result was computed without smoothing.
    """
    power = channel.power[condition][trial]
    if kind == ERD_MAP:
        return _percent(power, channel.baseline[condition][trial][:, None])
    if kind == POWER_MAP:
        return 10 * np.log10(np.maximum(power, 1e-12))
    return amplitude_from_power(power, result.fs, result.settings, result.times.size)


def condition_map(result, channel, condition, kind=ERD_MAP):
    """The map averaged over a condition's trials, as `trial_map` draws one trial."""
    if kind == ERD_MAP:
        return erd_map(channel, condition, result.settings.erd_mode)
    if kind == POWER_MAP:
        return power_map(channel, condition, db=True)
    return amplitude_map(channel, condition)


def erd_map(channel, condition, mode=POOLED):
    """ERD/ERS in percent, (n_freqs, n_times)."""
    power, reference = channel.power[condition], channel.baseline[condition][:, :, None]
    if mode == POOLED:
        return _percent(power.mean(axis=0), reference.mean(axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(_percent(power, reference), axis=0)


def band_course(channel, condition, band, mode=POOLED):
    """(mean ERD% over time, SEM across epochs, n) for one band."""
    rows = _band_rows(channel.freqs, band[0], band[1])
    power = channel.power[condition][:, rows].mean(axis=1)                 # (n_ep, n_t)
    reference = channel.baseline[condition][:, rows].mean(axis=1)[:, None]  # (n_ep, 1)
    curves = _percent(power, reference)
    n = curves.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(curves.shape[1])
        mean = (_percent(power.mean(axis=0), reference.mean()) if mode == POOLED
                else np.nanmean(curves, axis=0))
    return mean, sem, n


def band_trace(result, channel, condition, band, trial=-1, kind=ERD_MAP):
    """One band over time (n_times,), as the maps draw it: ERD% of the band's power (pooled or a mean of
    trials, as `erd_map`), its power in dB, or its amplitude in µV; `trial` -1: the condition's trials."""
    rows = _band_rows(channel.freqs, band[0], band[1])
    if kind == ERD_MAP:
        power = channel.power[condition][:, rows].mean(axis=1)              # (n_ep, n_t)
        reference = channel.baseline[condition][:, rows].mean(axis=1)       # (n_ep,)
        if trial >= 0:
            return _percent(power[trial], reference[trial])
        if result.settings.erd_mode == POOLED:
            return _percent(power.mean(axis=0), reference.mean())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(_percent(power, reference[:, None]), axis=0)
    values = trial_map(result, channel, condition, trial, kind) if trial >= 0 else \
        condition_map(result, channel, condition, kind)
    return np.asarray(values)[rows].mean(axis=0)


def cliffs_delta(a, b):
    from .feature_profiles import cliffs_delta as delta
    return delta(a, b)


def statistics(result, condition_a, condition_b):
    """Rows per analysed channel and band: per-epoch active-window ERD% of A and B.

    Descriptive: mean, SD, n, B - A, Cliff's delta and an uncorrected Welch
    t-test p. With many channels and bands, a p below 0.05 is expected by
    chance alone; the column is a screen, not a finding.
    """
    rows = []
    order = {c: i for i, c in enumerate(result.topo_channels)}
    for c in result.channels:
        if c not in order:
            continue
        for name, low, high in result.settings.bands:
            a = result.topo[name][condition_a][:, order[c]] if condition_a in result.topo[name] else np.array([])
            b = result.topo[name][condition_b][:, order[c]] if condition_b in result.topo[name] else np.array([])
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            p = np.nan
            if len(a) > 1 and len(b) > 1 and condition_a != condition_b:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p = float(ttest_ind(a, b, equal_var=False).pvalue)
            rows.append(dict(channel=result.names[c], band=name, band_hz="{:g}-{:g}".format(low, high),
                             n_a=len(a), mean_a=float(a.mean()) if len(a) else np.nan,
                             sd_a=float(a.std(ddof=1)) if len(a) > 1 else np.nan,
                             pooled_a=(float(result.topo_pooled[name][condition_a][order[c]])
                                       if condition_a in result.topo_pooled[name] else np.nan),
                             n_b=len(b), mean_b=float(b.mean()) if len(b) else np.nan,
                             sd_b=float(b.std(ddof=1)) if len(b) > 1 else np.nan,
                             pooled_b=(float(result.topo_pooled[name][condition_b][order[c]])
                                       if condition_b in result.topo_pooled[name] else np.nan),
                             difference=(float(b.mean() - a.mean()) if len(a) and len(b) else np.nan),
                             cliffs_delta=cliffs_delta(a, b) if condition_a != condition_b else np.nan,
                             p_welch=p))
    return rows


def export_csv(path, result, source, condition_a, condition_b):
    """Long-form CSV of every number the page shows."""
    import csv
    common = dict(recording=str(source), fs_hz=result.fs,
                  settings=json.dumps(result.settings.to_dict(), sort_keys=True))
    headers = ["recording", "fs_hz", "settings", "section", "condition", "channel", "band", "x", "x_unit",
               "y", "value", "unit", "sem", "n", "epoch_id", "status", "onset_s",
               "mean_a", "sd_a", "pooled_a", "n_a", "mean_b", "sd_b", "pooled_b", "n_b",
               "difference", "cliffs_delta", "p_welch", "condition_a", "condition_b"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()

        def finite(value):
            return value if np.isfinite(value) else ""

        for eid, label, onset, status in result.epochs:
            writer.writerow(dict(common, section="epoch", condition=label, epoch_id=eid, onset_s=onset, status=status))
        for cond in result.conditions:
            n = len(result.epoch_ids[cond])
            for slot, c in enumerate(result.channels):
                mean, sem = result.evoked[cond]
                for t, v, e in zip(result.times, mean[slot], sem[slot]):
                    writer.writerow(dict(common, section="evoked", condition=cond, channel=result.names[c], x=t,
                                         x_unit="s", value=v, unit="uV", sem=e, n=n))
                mean, sem = result.psd[cond]
                for f, v, e in zip(result.psd_freqs, mean[slot], sem[slot]):
                    writer.writerow(dict(common, section="psd_active_window", condition=cond, channel=result.names[c],
                                         x=f, x_unit="Hz", value=v, unit="uV^2/Hz", sem=e, n=n))
            for channel in result.tf:
                for name, low, high in result.settings.bands:
                    mean, sem, count = band_course(channel, cond, (low, high), result.settings.erd_mode)
                    for t, v, e in zip(channel.times, mean, sem):
                        writer.writerow(dict(common, section="band_erd_course", condition=cond, channel=channel.name,
                                             band=name, x=t, x_unit="s", value=finite(v), unit="%", sem=finite(e),
                                             n=count))
            for name in result.topo_pooled:
                for slot, c in enumerate(result.topo_channels):
                    writer.writerow(dict(common, section="topography_pooled_erd", condition=cond,
                                         channel=result.names[c], band=name,
                                         value=finite(result.topo_pooled[name][cond][slot]), unit="%", n=n))
        for row in statistics(result, condition_a, condition_b):
            writer.writerow(dict(common, section="statistics", channel=row["channel"], band=row["band"],
                                 condition_a=condition_a, condition_b=condition_b, unit="%",
                                 **{k: finite(row[k]) if isinstance(row[k], float) else row[k]
                                    for k in ("mean_a", "sd_a", "pooled_a", "n_a", "mean_b", "sd_b", "pooled_b", "n_b",
                                              "difference", "cliffs_delta", "p_welch")}))
