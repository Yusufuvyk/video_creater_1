"""Live display measurements. No Qt and no changes to analysis/export data.

Channel quality, per channel, over the last 4 s of raw samples:

  rms         EEG-band amplitude: sqrt of the Welch PSD integrated over 1-45 Hz.
              Band-limited on purpose - raw RMS is mostly electrode drift and
              mains, which the display filters remove, so it flagged channels
              whose on-screen EEG was fine.
  mains       49-51 Hz power / 1-60 Hz power. Reported as its own status,
              because a notch filter removes it and it says "cable/room
              pick-up", not "no EEG".
  saturation  share of samples within 5% of the Cyton rail (±187,500 µV).

  same_as     another channel carrying the same waveform and offset, or -1.

Order of verdicts: Missing data > Saturated > Flat > Check signal (EEG band
too large) > Identical signal > Line noise > No issues. Defaults follow common practice rather
than a single standard: ±100 µV is the usual amplitude beyond which scalp EEG
is treated as artefact; the other three are this app's operating limits.
"""
from dataclasses import dataclass, replace
import numpy as np
from .spectral import welch_psd, band_integral

EEG_BAND = (1.0, 45.0)

#: Two scalp electrodes centimetres apart never record the same waveform:
#: when two channels' 1-45 Hz signals differ by less than 2 % of their size and
#: their offsets by less than 1 mV, both inputs sit on one electrical node -
#: electrodes not touching the head (floating together), or bridged by gel or
#: sweat. The difference is Tenke & Kayser's (2001) "electrical distance"
#: idea, normalised. The two limits are this lab's, not a published standard:
#: set on 18 Sept 2026 Unicorn recordings where Pz/PO7/Oz/PO8 floated
#: together (difference 0.3-1.4 %, same offset), and checked against the
#: lab's Cyton recordings, where only channels already railed or >100 µV -
#: which keep their own verdicts - came that close.
IDENTICAL_DIFFERENCE = 0.02
IDENTICAL_DC_UV = 1000.0


@dataclass(frozen=True)
class QualityLimits:
    flat_uv: float = 0.1
    high_uv: float = 100.0
    mains_ratio: float = 0.2
    saturation_ratio: float = 0.15


@dataclass(frozen=True)
class ChannelQuality:
    channel: int
    status: str
    rms: float = float('nan')
    saturation: float = float('nan')
    mains: float = float('nan')
    data_time: float = 0.0
    same_as: int = -1


def quality_snapshot(raw, fs, data_time, rail_uv=None, limits=QualityLimits()):
    raw = np.asarray(raw, dtype=float)
    if raw.shape[1] < round(4 * fs):
        return [ChannelQuality(i, 'Collecting data', data_time=data_time)
                for i in range(len(raw))]
    # One Welch call for every channel. Per-channel calls cost ~16 ms on the GUI
    # thread once a second, which showed as a regular stutter in the monitor.
    finite = np.isfinite(raw).all(axis=1)
    # Linear detrend per segment: a dry electrode's offset ramps by millivolts
    # per second, and a mean-only detrend leaks that ramp into 1-3 Hz.
    f, psd = welch_psd(np.where(finite[:, None], raw, 0.0), fs, detrend="linear")
    totals = np.atleast_1d(band_integral(f, psd, 1, min(60, fs / 2)))
    eeg_power = np.atleast_1d(band_integral(f, psd, EEG_BAND[0], min(EEG_BAND[1], fs / 2)))
    line = (np.atleast_1d(band_integral(f, psd, 49, min(51, fs / 2)))
            if fs / 2 > 49 else np.full(len(raw), np.nan))
    partners = identical_partners(raw, finite, fs)
    result = []
    for i, row in enumerate(raw):
        if not finite[i]:
            result.append(ChannelQuality(i, 'Missing data', data_time=data_time))
            continue
        rms = float(np.sqrt(max(eeg_power[i], 0.0)))
        sat = float(np.mean(np.abs(row) > .95 * rail_uv)) if rail_uv else float('nan')
        total = float(totals[i])
        mains = (float(line[i]) / total
                 if fs / 2 > 49 and total > 1e-20 else float('nan'))
        status = ('Saturated' if sat > limits.saturation_ratio else
                  'Flat' if rms < limits.flat_uv else
                  'Check signal' if rms > limits.high_uv else
                  'Identical signal' if partners[i] >= 0 else
                  'Line noise' if mains > limits.mains_ratio else
                  'No issues detected')
        result.append(ChannelQuality(i, status, rms, sat, mains, data_time,
                                     int(partners[i]) if status == 'Identical signal' else -1))
    return result


def flat_channels(data, flat_uv=QualityLimits.flat_uv):
    """Channels whose raw signal does not move (SD below `flat_uv` µV): powered down or unwired.

    A channel switched off on the Cyton records exactly 0 µV (checked on the
    lab's live recordings); a real input always moves by far more. Broadband on
    purpose: a slow but real signal with nothing in 1-45 Hz is still measured.
    NaN samples are left out.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    flat = set()
    for i, row in enumerate(data):
        row = row[np.isfinite(row)]
        if row.size >= 2 and float(np.std(row)) < flat_uv:
            flat.add(i)
    return flat


def identical_partners(raw, finite, fs):
    """For each channel, another whose signal it duplicates, or -1.

    Compared in the EEG band only, through one FFT: the difference's power is
    |X_i|^2 + |X_j|^2 - 2 Re(X_i X_j*), for every pair at once.
    """
    partners = np.full(len(raw), -1)
    rows = np.flatnonzero(finite)
    if rows.size < 2:
        return partners
    x = raw[rows]
    t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
    x = x - x.mean(axis=1, keepdims=True)
    x = x - np.outer((x @ t) / (t @ t), t)                # linear drift out
    freqs = np.fft.rfftfreq(x.shape[1], 1 / fs)
    spectrum = np.fft.rfft(x, axis=1)[:, (freqs >= EEG_BAND[0]) & (freqs <= EEG_BAND[1])]
    power = np.sum(np.abs(spectrum) ** 2, axis=1)
    cross = np.real(spectrum @ spectrum.conj().T)
    larger = np.maximum(power[:, None], power[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        difference = np.sqrt(np.maximum(power[:, None] + power[None, :] - 2 * cross, 0) / larger)
    difference[~(larger > 0)] = np.inf
    np.fill_diagonal(difference, np.inf)
    offsets = np.median(raw[rows], axis=1)
    same = (difference < IDENTICAL_DIFFERENCE) & (np.abs(offsets[:, None] - offsets[None, :]) < IDENTICAL_DC_UV)
    for k in np.flatnonzero(same.any(axis=1)):
        partners[rows[k]] = rows[np.argmin(np.where(same[k], difference[k], np.inf))]
    return partners


class QualityTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = {}
        self.pending = {}

    def update(self, values):
        result = []
        for value in values:
            i, wanted = value.channel, value.status
            current = self.states.get(i, 'Collecting data')
            if wanted in ('Missing data', 'Saturated', 'Collecting data'):
                current = wanted
                self.pending.pop(i, None)
            elif wanted == current:
                self.pending.pop(i, None)
            else:
                previous, count = self.pending.get(i, (None, 0))
                count = count + 1 if previous == wanted else 1
                self.pending[i] = (wanted, count)
                required = 3 if wanted == 'No issues detected' else 2
                # A first healthy window needs no recovery debounce.
                if current == 'Collecting data' and wanted == 'No issues detected':
                    required = 1
                if count >= required:
                    current = wanted
                    self.pending.pop(i, None)
            self.states[i] = current
            result.append(replace(value, status=current))
        return result


def spectra_snapshot(data, fs, welch_window=2.0, welch_overlap=0.5):
    """Batch FFT amplitude + Welch density; invalid rows cannot poison peers.

    The FFT uses the whole block; the PSD averages `welch_window`-second Hann
    segments overlapping by `welch_overlap` (a fraction, 0 <= overlap < 1).
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[1]
    if n < 2:
        return None
    valid = np.isfinite(data).all(axis=1)
    safe = np.where(valid[:, None], data, 0.0)
    centered = safe - safe.mean(axis=1, keepdims=True)
    window = np.hanning(n)
    amplitude = np.abs(np.fft.rfft(centered * window, axis=1)) / max(window.sum(), 1e-20)
    amplitude[:, 1: -1 if n % 2 == 0 else None] *= 2
    frequencies, density = welch_psd(safe, fs, welch_window, welch_overlap)
    amplitude[~valid] = np.nan
    density[~valid] = np.nan
    return dict(fft_freqs=np.fft.rfftfreq(n, 1 / fs), fft=amplitude,
                psd_freqs=frequencies, psd=density, valid=valid)


#: Movement above this is the head or the headset moving rather than the
#: sensor's own noise. Measured on a seated Cyton SSVEP session (18 Sept,
#: 3 minutes): median 3 mg, 99th percentile 21 mg, largest 54 mg - so 30 mg
#: marks the top third of a percent of the samples, which is what a nod is.
MOVEMENT_MG = 30.0


def movement_mg(accel):
    """How much the headset moves, in mg, from its three accelerometer axes.

    The accelerometer always reads gravity - about 1 g in whatever direction
    the head is tilted - so the raw axes say where the head points, not whether
    it moved. Taking |a| and removing its median leaves the movement itself,
    and zero means still.
    """
    accel = np.asarray(accel, dtype=float)
    if accel.ndim != 2 or accel.shape[0] < 3 or accel.shape[1] == 0:
        return np.zeros(0 if accel.ndim != 2 else accel.shape[1])
    magnitude = np.linalg.norm(accel[:3], axis=0)
    finite = magnitude[np.isfinite(magnitude)]
    rest = float(np.median(finite)) if finite.size else 0.0
    return 1000.0 * (magnitude - rest)


def movement_spans(mg, fs, threshold=MOVEMENT_MG, join_seconds=0.25):
    """[(start s, end s)] where the movement passes `threshold`.

    Bursts closer together than `join_seconds` become one span: a head shake is
    one event, not the dozen crossings its individual swings would give.
    """
    mg = np.abs(np.asarray(mg, dtype=float))
    fs = float(fs)
    if mg.ndim != 1 or not mg.size or not np.isfinite(fs) or fs <= 0:
        return []
    over = np.isfinite(mg) & (mg > float(threshold))
    if not over.any():
        return []
    edges = np.flatnonzero(np.diff(over.astype(np.int8)))
    starts = list(edges[::2] + 1) if not over[0] else [0] + list(edges[1::2] + 1)
    ends = list(edges[1::2] + 1) if not over[0] else list(edges[::2] + 1)
    if len(ends) < len(starts):
        ends.append(mg.size)
    spans = []
    for start, end in zip(starts, ends):
        seconds = (start / fs, end / fs)
        if spans and seconds[0] - spans[-1][1] <= join_seconds:
            spans[-1] = (spans[-1][0], seconds[1])
        else:
            spans.append(seconds)
    return spans
