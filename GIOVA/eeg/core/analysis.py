"""Offline analysis of a selected segment.

Three methods, each producing one kind of result:

    PSD (Welch)  -> Spectrum        frequency only, frames averaged
    STFT         -> TimeFrequency   fixed window, time and frequency
    Morlet       -> TimeFrequency   per-frequency window, time and frequency

Welch is literally the average of the STFT's frames, and the code says so: it
calls the STFT and averages. Everything else - band powers, features, ERD - is
derived from whichever result you asked for, so the numbers on screen always
agree with each other.

Pure numpy/scipy/mne, no Qt, so all of this is testable headless. Inputs are
microvolts shaped (n_samples,) for one channel.
"""
import numpy as np
from scipy.signal import spectrogram

from .config import BANDS
from .spectral import band_integral, welch_psd

WELCH = "welch"
STFT = "stft"
MORLET = "morlet"
METHODS = (WELCH, STFT, MORLET)

#: Methods that produce a time axis, and therefore support ERD.
TIME_RESOLVED = (STFT, MORLET)


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------


class Spectrum:
    """A frequency-domain result: no time axis, frames already averaged."""

    __slots__ = ("method", "freqs", "mean", "sd", "n_frames", "units",
                 "window_seconds")

    def __init__(self, method, freqs, mean, sd, n_frames, units, window_seconds):
        self.method = method
        self.freqs = freqs
        self.mean = mean
        self.sd = sd
        self.n_frames = n_frames
        self.units = units
        self.window_seconds = window_seconds

    @property
    def resolution(self):
        return _bin_spacing(self.freqs)

    def describe(self):
        return "{}: {} bins from {} frames, {:.2f} Hz resolution".format(
            self.method, self.freqs.size, self.n_frames, self.resolution)


class TimeFrequency:
    """A time-frequency decomposition and the things computed from it.

    `power` is (n_freqs, n_times). The STFT gives a power spectral density per
    frame; Morlet gives instantaneous power. Units differ, so they travel in
    `units`, and every ratio (ERD, relative band power) is unaffected.

    `span` is the full extent of the analysed signal. The STFT's `times` are
    frame *centres*, which sit half a window inside that extent - drawing an
    image against `times` instead of `span` is what makes a spectrogram look
    shifted relative to the trace above it.
    """

    __slots__ = ("method", "freqs", "times", "power", "units", "frame_seconds",
                 "span")

    def __init__(self, method, freqs, times, power, units, frame_seconds, span):
        self.method = method
        self.freqs = freqs
        self.times = times
        self.power = power
        self.units = units
        self.frame_seconds = frame_seconds
        self.span = span

    @property
    def n_frames(self):
        return self.power.shape[1]

    @property
    def resolution(self):
        """Frequency bin spacing in Hz.

        For the STFT this is 1 / window_seconds: a shorter window averages more
        frames but resolves frequency more coarsely. The two trade directly.
        """
        return _bin_spacing(self.freqs)

    def shift(self, offset):
        """Move onto recording time. Mutates and returns self."""
        self.times = self.times + offset
        self.span = (self.span[0] + offset, self.span[1] + offset)
        return self

    def frames_in(self, window, allow_nearest=True):
        """Mask of frames whose centre falls inside `window`.

        A window shorter than one frame contains no centre at all. Rather than
        failing, fall back to the single nearest frame - the caller can ask
        `covers_frames` first if it wants to warn about the approximation.
        """
        start, end = window
        mask = (self.times >= start) & (self.times <= end)
        if mask.any() or not allow_nearest or self.times.size == 0:
            return mask
        centre = (start + end) / 2.0
        mask = np.zeros(self.times.shape, dtype=bool)
        mask[int(np.argmin(np.abs(self.times - centre)))] = True
        return mask

    def covers_frames(self, window):
        """True when `window` genuinely contains at least one frame centre."""
        start, end = window
        return bool(((self.times >= start) & (self.times <= end)).any())

    def describe(self):
        return "{}: {} freqs x {} frames, {:.1f}-{:.1f} Hz, {:.2f} Hz resolution".format(
            self.method, self.freqs.size, self.n_frames,
            self.freqs[0], self.freqs[-1], self.resolution)


def _bin_spacing(freqs):
    return float(freqs[1] - freqs[0]) if freqs.size > 1 else float("nan")


# ----------------------------------------------------------------------
# decompositions
# ----------------------------------------------------------------------


def frame_plan(duration, fs, window_seconds, overlap):
    """What a window/overlap choice will actually produce.

    Returns (n_frames, resolution_hz, window_seconds_used). Used by the UI to
    show the cost of a setting before it is applied.
    """
    nperseg = _clamp_window(duration * fs, window_seconds, fs)
    step = max(nperseg - int(nperseg * _clamp_overlap(overlap)), 1)
    total = int(duration * fs)
    frames = max((total - nperseg) // step + 1, 1)
    return frames, fs / nperseg, nperseg / fs


def _clamp_window(n_samples, window_seconds, fs):
    """Window in samples, kept between 16 samples and the signal length."""
    nperseg = int(round(window_seconds * fs))
    return int(np.clip(nperseg, 16, max(int(n_samples), 16)))


def _clamp_overlap(overlap):
    return float(np.clip(overlap, 0.0, 0.95))


def stft_tfr(signal, fs, window_seconds=2.0, overlap=0.5, fmin=1.0, fmax=45.0):
    """Short-time Fourier transform with an explicit window and overlap.

    Frequency resolution is 1 / window_seconds; overlap only adds frames, it
    buys no extra frequency detail.
    """
    if signal.size < 16:
        raise ValueError("segment too short to transform")

    from .spectral import welch_sizes
    nperseg,noverlap = welch_sizes(signal.size,fs,window_seconds,overlap)

    freqs, times, power = spectrogram(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window="hann", scaling="density", mode="psd")

    keep = (freqs >= fmin) & (freqs <= fmax)
    if not keep.any():
        raise ValueError("Window is too short to resolve the selected frequency range")
    return TimeFrequency(STFT, freqs[keep], times, power[keep],
                         "uV^2/Hz", nperseg / fs, (0.0, signal.size / fs))


def morlet_tfr(signal, fs, fmin=2.0, fmax=45.0, n_freqs=40, n_cycles=None):
    """Continuous Morlet wavelet transform.

    Adapts its window per frequency - long for slow rhythms, short for fast -
    which one fixed STFT window cannot do.
    """
    from mne.time_frequency import tfr_array_morlet

    freqs = np.linspace(max(fmin, 1.0), fmax, int(n_freqs))
    if n_cycles is None:
        # Fewer cycles low down, or the wavelet outgrows the segment.
        n_cycles = np.clip(freqs / 2.0, 2.0, 12.0)
    n_cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), freqs.shape)

    # MNE truncates each wavelet at five standard deviations, making it about
    # 5 / pi times longer than the bare n_cycles / f. Low frequencies therefore
    # need long wavelets that will not fit a short segment - drop those bands
    # rather than refusing the whole transform. A short chunk simply cannot
    # resolve slow rhythms, and saying so beats failing.
    wavelet_samples = 5.0 * n_cycles * fs / (np.pi * freqs)
    fits = wavelet_samples <= signal.size
    if not fits.any():
        raise ValueError(
            "segment is {:.2f} s; too short for any wavelet up to {:g} Hz"
            .format(signal.size / fs, fmax))
    freqs, n_cycles = freqs[fits], n_cycles[fits]

    data = (signal * 1e-6)[np.newaxis, np.newaxis, :]  # (epoch, channel, time)
    power = tfr_array_morlet(data, sfreq=fs, freqs=freqs,
                             n_cycles=n_cycles, output="power")[0, 0] * 1e12

    times = np.arange(signal.size) / fs
    return TimeFrequency(MORLET, freqs, times, power, "uV^2", 1.0 / fs,
                         (0.0, signal.size / fs))


def welch_spectrum(signal, fs, window_seconds=2.0, overlap=0.5, fmin=1.0,
                   fmax=45.0):
    """Welch's estimate: the STFT's frames, averaged.

    Averaging N frames cuts the standard error of each bin by about sqrt(N),
    which is why a single long FFT of a noisy signal is a poor spectrum no
    matter how long it is.
    """
    freqs, power = welch_psd(signal, fs, window_seconds, overlap)
    tfr = stft_tfr(signal, fs, window_seconds, overlap, fmin, fmax)
    keep = (freqs >= fmin) & (freqs <= fmax)
    return Spectrum(WELCH, freqs[keep], power[keep], tfr.power.std(axis=1),
                    tfr.n_frames, "uV^2/Hz", tfr.frame_seconds)



def compute(signal, fs, method=WELCH, window_seconds=2.0, overlap=0.5,
            fmin=1.0, fmax=45.0, n_freqs=40):
    """Run one method. Single entry point for the whole page."""
    if method == WELCH:
        return welch_spectrum(signal, fs, window_seconds, overlap, fmin, fmax)
    if method == STFT:
        return stft_tfr(signal, fs, window_seconds, overlap, fmin, fmax)
    if method == MORLET:
        return morlet_tfr(signal, fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs)
    raise ValueError("unknown method: {}".format(method))


# ----------------------------------------------------------------------
# averaging
# ----------------------------------------------------------------------


def average_results(results):
    """Combine several results by averaging power.

    Used both for averaging channels and for averaging the chunks of one
    channel. Raw waveforms are deliberately not averaged: anything not locked
    to the split points cancels through phase differences, so ongoing EEG would
    average towards zero and look like no signal at all.
    """
    if not results:
        raise ValueError("nothing to average")
    first = results[0]
    if len(results) == 1:
        return first

    if isinstance(first, Spectrum):
        return Spectrum(first.method, first.freqs,
                        np.mean([r.mean for r in results], axis=0),
                        np.sqrt(np.maximum(0, np.mean([r.sd ** 2 + r.mean ** 2 for r in results], axis=0) - np.mean([r.mean for r in results], axis=0) ** 2)),
                        sum(r.n_frames for r in results), first.units, first.window_seconds)

    return TimeFrequency(first.method, first.freqs, first.times,
                         np.mean([r.power for r in results], axis=0),
                         first.units, first.frame_seconds, first.span)


# ----------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------


#: An epoch needs at least this many frame centres to be resampled at all -
#: `np.interp` cannot draw a line through one point.
MIN_EPOCH_FRAMES = 2

#: Seconds of temporal smoothing applied to the power before any ratio is
#: taken. Instantaneous power swings through a full cycle every 1/f seconds and
#: the ERD is buried in that picket fence; 0.5 s is about five cycles of mu,
#: long enough to average the oscillation away and short enough to keep the
#: drop where it happens.
DEFAULT_SMOOTH_SECONDS = 0.5


def frame_step(tfr):
    """Seconds between frame centres, or 0 when there is only one frame."""
    return float(np.median(np.diff(tfr.times))) if tfr.times.size > 1 else 0.0


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




def average_spectrum(tfr, window=None):
    """Mean spectrum across frames, with the spread between them.

    Returns (freqs, mean, sd, n_frames).
    """
    power = tfr.power
    if window is not None:
        mask = tfr.frames_in(window)
        if not mask.any():
            raise ValueError("window contains no frames")
        power = power[:, mask]
    return tfr.freqs, power.mean(axis=1), power.std(axis=1), power.shape[1]


def as_spectrum(result, window=None):
    """(freqs, mean) from either result type, so callers need not branch."""
    if isinstance(result, Spectrum):
        return result.freqs, result.mean
    freqs, mean, _, _ = average_spectrum(result, window)
    return freqs, mean


def band_powers(result, window=None):
    """Integrate the (averaged) spectrum over each named band."""
    freqs, mean = as_spectrum(result, window)
    if isinstance(result, TimeFrequency) and result.units != "uV^2/Hz":
        raise ValueError("Wavelet power is not PSD; use its time-frequency power view")
    return {name: band_integral(freqs, mean, low, high) for name, low, high in BANDS}



# ----------------------------------------------------------------------
# trial averaging
# ----------------------------------------------------------------------

def _epoch_grid(epochs, n_points, margin=0.0):
    """Shared time-from-onset axis, as long as the shortest epoch.

    The axis keeps its offset rather than starting at zero, so a cue two
    seconds into the trial is drawn at two seconds. Trimming `margin` from both
    ends drops the frames whose window straddles the epoch boundary and so
    contains the neighbouring trial.
    """
    shortest = min(end - start for (start, end), _ in epochs)
    low, high = margin, shortest - margin
    if high <= low:
        raise ValueError(
            "epochs are {:.2f} s but {:.2f} s is dropped at each end for the "
            "analysis window - nothing is left".format(shortest, margin))
    return np.linspace(low, high, int(n_points))


def epoch_problem(tfr, epochs, edge_margin=None):
    """Why trial averaging cannot work here, in the caller's terms, or None.

    Worth its own function because the honest answer is almost never the one
    the failure looks like. An epoch that yields no usable frames is nearly
    always an analysis window longer than the epoch, not a mislabelled event.
    """
    if not epochs:
        return "no epochs - define a trial structure first"
    if edge_margin is None:
        edge_margin = tfr.frame_seconds / 2.0

    shortest = min(end - start for (start, end), _ in epochs)
    counts = _frame_counts(tfr, epochs, edge_margin)
    step = frame_step(tfr)

    if counts.max() < MIN_EPOCH_FRAMES:
        suggested = max(round(shortest / 7.0, 2), 0.25)
        # The suggested window brings its own edge margin with it; quoting the
        # current window's margin here would advertise the fix as no better
        # than the problem.
        gain = frames_per_epoch(shortest, suggested, 0.75, suggested / 2.0)
        return (
            "the {:.2f} s analysis window steps every {:.2f} s, so a {:.1f} s "
            "epoch holds {} frame{} and {} are needed. A window longer than "
            "the epoch also measures mostly its neighbouring trials. Try a "
            "window near {:.2f} s at 75% overlap, which gives {} frames per "
            "epoch.".format(
                tfr.frame_seconds, step, shortest, int(counts.max()),
                "" if counts.max() == 1 else "s", MIN_EPOCH_FRAMES,
                suggested, gain))

    if not _any_baseline_frames(tfr, epochs):
        return ("no frames fall inside the baseline window - move it inside "
                "the recording, or widen it past one frame step ({:.2f} s)"
                .format(step))
    return None


def _frame_counts(tfr, epochs, edge_margin):
    return np.array([
        int(((tfr.times >= start + edge_margin) &
             (tfr.times <= end - edge_margin)).sum())
        for (start, end), _ in epochs])


def frames_per_epoch(epoch_seconds, window, overlap, margin=0.0):
    """How many frame centres a window/overlap choice puts inside one epoch.

    The number that decides whether trial averaging is possible at all, and the
    one the whole-recording frame count hides: 198 frames over 700 s is still
    only one frame inside a 3 s epoch.
    """
    step = max(window * (1.0 - overlap), 1e-9)
    return max(int((epoch_seconds - 2 * margin) // step), 0) + 1


def _any_baseline_frames(tfr, epochs):
    return any(((tfr.times >= low) & (tfr.times <= high)).any()
               for _, (low, high) in epochs)


def epoch_stack(tfr, epochs, n_points=240, smooth_seconds=DEFAULT_SMOOTH_SECONDS,
                edge_margin=None):
    """Everything the epoch views need, resampled onto one axis.

    Returns (grid, power, reference):

        grid       (n_points,)                       time from epoch onset
        power      (n_epochs, n_freqs, n_points)     each epoch, resampled
        reference  (n_epochs, n_freqs, 1)            each epoch's baseline power

    The baseline is read from the recording's own time axis, so it works
    whether it sits inside the epoch, in a preceding block, or somewhere fixed
    at the start of the session. Nothing here knows which.
    """
    problem = epoch_problem(tfr, epochs, edge_margin)
    if problem:
        raise ValueError(problem)
    if edge_margin is None:
        edge_margin = tfr.frame_seconds / 2.0

    grid = _epoch_grid(epochs, n_points, edge_margin)
    step = frame_step(tfr)

    blocks, references = [], []
    for (start, end), (low, high) in epochs:
        inside = (tfr.times >= start + edge_margin) & (tfr.times <= end - edge_margin)
        in_baseline = (tfr.times >= low) & (tfr.times <= high)
        if inside.sum() < MIN_EPOCH_FRAMES or not in_baseline.any():
            continue  # this epoch cannot contribute; others still can

        # Smoothed per slice, so the filter never reaches across an epoch
        # boundary into a neighbouring trial.
        block = smooth_power(tfr.power[:, inside], step, smooth_seconds)
        relative = tfr.times[inside] - start
        blocks.append(np.vstack([np.interp(grid, relative, row)
                                 for row in block]))
        references.append(smooth_power(tfr.power[:, in_baseline], step,
                                       smooth_seconds).mean(axis=1))

    if not blocks:
        raise ValueError(epoch_problem(tfr, epochs, edge_margin)
                         or "no usable epochs")
    return grid, np.stack(blocks), np.stack(references)[:, :, np.newaxis]



def _percent(power, reference):
    """ERD% = 100 * (A - R) / R, with a dead reference left as NaN."""
    reference = np.where(reference <= 0, np.nan, reference)
    return 100.0 * (power - reference) / reference


def epoch_power_map(tfr, epochs, **kwargs):
    """Trial-averaged power on time from epoch onset.

    One continuous image of a whole session hides every trial boundary, so
    whatever is locked to the cue averages away against everything that is not.

    Returns (freqs, grid, power) with power (n_freqs, n_points).
    """
    grid, power, _ = epoch_stack(tfr, epochs, **kwargs)
    return tfr.freqs, grid, power.mean(axis=0)


def epoch_erd_map(tfr, epochs, pooled=True, **kwargs):
    """Trial-averaged ERD/ERS across all frequencies, in percent.

    `pooled` decides WHERE the division happens, and it is not a presentation
    choice. ERD% is bounded at -100% below but unbounded above, so averaging
    per-trial ratios is dragged upward by the trials whose baseline came out
    small. On Cho2017 s01/C3 the same data reads -50.6% pooled and -12.3% as a
    mean of ratios: a four-fold understatement. Average the power first for any
    map reported as one number, and keep the per-trial rows only where each
    trial must stay separate - which is what a classifier needs.

    Returns (freqs, grid, percent) with percent (n_freqs, n_points).
    """
    grid, power, reference = epoch_stack(tfr, epochs, **kwargs)
    if pooled:
        return tfr.freqs, grid, _percent(power.mean(axis=0),
                                         reference.mean(axis=0))
    return tfr.freqs, grid, np.nanmean(_percent(power, reference), axis=0)


def epoch_erd_trials(tfr, epochs, **kwargs):
    """Per-trial ERD maps, nothing averaged: (freqs, grid, (n_epochs, n_f, n_t)).

    The classifier's view. Kept separate from `epoch_erd_map` so that the
    display path cannot accidentally take this one.
    """
    grid, power, reference = epoch_stack(tfr, epochs, **kwargs)
    return tfr.freqs, grid, _percent(power, reference)


def epoch_band_erd(tfr, band, epochs, pooled=True, **kwargs):
    """Trial-averaged band ERD over time. Returns (grid, mean, sem, n_epochs).

    The curve is the band average of the pooled map; the spread is the spread
    of the individual trials' curves, which is a different and honest question -
    how consistent the trials were, not how uncertain the pooled estimate is.
    """
    grid, power, reference = epoch_stack(tfr, epochs, **kwargs)
    rows = (tfr.freqs >= band[0]) & (tfr.freqs <= band[1])
    if not rows.any():
        raise ValueError(
            "resolution is {:.1f} Hz, too coarse for the {:.0f}-{:.0f} Hz band "
            "- use a longer window or a wider band".format(
                tfr.resolution, band[0], band[1]))

    power, reference = power[:, rows], reference[:, rows]
    curves = np.nanmean(_percent(power, reference), axis=1)      # per trial
    n = curves.shape[0]
    sem = curves.std(axis=0) / np.sqrt(n) if n > 1 else np.zeros(grid.size)

    mean = (np.nanmean(_percent(power.mean(axis=0), reference.mean(axis=0)),
                       axis=0) if pooled else np.nanmean(curves, axis=0))
    return grid, mean, sem, n

# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------


from .spectral import hjorth, spectral_entropy


# ----------------------------------------------------------------------
# display helpers
# ----------------------------------------------------------------------


def envelope(signal, times, target_points):
    """Min/max decimation for drawing long signals.

    Plain striding drops the peaks between kept samples, so the overview ends
    up showing a different waveform from the one being analysed. Keeping both
    extremes of every bin preserves the envelope.
    """
    n = signal.size
    if n <= target_points or target_points < 4:
        return times, signal

    bins = max(target_points // 2, 2)
    per_bin = n // bins
    usable = per_bin * bins
    block = signal[:usable].reshape(bins, per_bin)
    time_block = times[:usable].reshape(bins, per_bin)

    lows = block.min(axis=1)
    highs = block.max(axis=1)
    low_at = time_block[np.arange(bins), block.argmin(axis=1)]
    high_at = time_block[np.arange(bins), block.argmax(axis=1)]

    out_t = np.empty(bins * 2)
    out_v = np.empty(bins * 2)
    first_is_low = low_at <= high_at
    out_t[0::2] = np.where(first_is_low, low_at, high_at)
    out_v[0::2] = np.where(first_is_low, lows, highs)
    out_t[1::2] = np.where(first_is_low, high_at, low_at)
    out_v[1::2] = np.where(first_is_low, highs, lows)

    if usable < n:  # keep the tail rather than silently dropping it
        out_t = np.append(out_t, times[usable:])
        out_v = np.append(out_v, signal[usable:])
    return out_t, out_v
