"""Shared, explicitly scaled spectral and temporal measurements (no UI)."""
import numpy as np
from scipy.signal import welch


def sample_index(seconds, fs):
    return int(np.floor(float(seconds) * fs + 0.5))


def welch_sizes(n_samples, fs, window=2.0, overlap=0.5):
    if n_samples < 2 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("At least two samples and a positive sampling rate are required")
    if not np.isfinite(window) or window <= 0 or not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Invalid Welch window or overlap")
    nperseg = min(n_samples, max(2, sample_index(window, fs)))
    return nperseg, min(int(nperseg * overlap), nperseg - 1)


def welch_psd(signal, fs, window=2.0, overlap=0.5, detrend="constant"):
    x = np.asarray(signal, dtype=float)
    nperseg, noverlap = welch_sizes(x.shape[-1], fs, window, overlap)
    if not np.isfinite(x).all():
        raise ValueError("PSD needs finite samples")
    x = x - x.mean(axis=-1, keepdims=True)
    return welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                 detrend=detrend, scaling="density", axis=-1)


#: How a Welch density is drawn (Settings → Graphs). Display only: features,
#: band powers and exports always use the density itself.
POWER_DENSITY, AMPLITUDE_DENSITY, AMPLITUDE = "power", "amplitude_density", "amplitude"
PSD_UNITS = {POWER_DENSITY: "µV²/Hz", AMPLITUDE_DENSITY: "µV/√Hz", AMPLITUDE: "µV"}
PSD_TITLES = {POWER_DENSITY: "PSD", AMPLITUDE_DENSITY: "Amplitude density", AMPLITUDE: "Amplitude"}
#: Equivalent noise bandwidth of the Hann window, in bins (Harris, 1978).
HANN_ENBW_BINS = 1.5


def psd_in_unit(freqs, density, unit):
    """A Hann Welch density (µV²/Hz) in the unit chosen for the graphs.

    amplitude_density: √density, µV/√Hz. amplitude: √(2 · density · 1.5 Δf),
    µV - the density times the window's noise bandwidth is the power in one
    bin, and a sine of amplitude A carries A²/2, so that sine reads A, as it
    does in the FFT view. Anything else returns the density unchanged.
    """
    density = np.asarray(density, dtype=float)
    if unit == AMPLITUDE_DENSITY:
        return np.sqrt(np.maximum(density, 0))
    if unit == AMPLITUDE:
        freqs = np.asarray(freqs, dtype=float)
        if freqs.size < 2:
            return np.full(density.shape, np.nan)
        return np.sqrt(np.maximum(2 * density * HANN_ENBW_BINS * (freqs[1] - freqs[0]), 0))
    return density


def band_integral(freqs, density, low, high):
    """Integrate exact edges on the last axis; adjacent bands share an edge."""
    freqs, density = np.asarray(freqs), np.asarray(density)
    if freqs.size < 2:
        return np.zeros(density.shape[:-1]) if density.ndim > 1 else 0.0
    low, high = max(low, float(freqs[0])), min(high, float(freqs[-1]))
    if high <= low:
        return np.zeros(density.shape[:-1]) if density.ndim > 1 else 0.0
    inside = (freqs > low) & (freqs < high)
    x = np.r_[low, freqs[inside], high]
    flat = density.reshape(-1, len(freqs))
    values = [np.trapezoid(np.interp(x, freqs, row), x) for row in flat]
    result = np.asarray(values).reshape(density.shape[:-1])
    return float(result) if result.ndim == 0 else result


def spectral_entropy(power):
    """Normalized Shannon entropy, fixed bin count; no power is undefined."""
    power = np.asarray(power, dtype=float)
    n = power.shape[-1]
    total = power.sum(axis=-1, keepdims=True)
    p = power / np.where(total > 0, total, 1)
    terms = p * np.log2(np.where(p > 0, p, 1))
    return np.where((total[..., 0] > 0) & (n > 1),
                    -terms.sum(axis=-1) / np.log2(max(n, 2)), np.nan)


def hjorth(signal, fs=1.0):
    """Variance (uV²), mobility (s^-1), complexity (dimensionless)."""
    x = np.asarray(signal, dtype=float)
    var0 = x.var(axis=-1)
    if x.shape[-1] < 3:
        missing = np.full_like(var0, np.nan)
        return var0, missing, missing
    d1 = np.diff(x, axis=-1) * fs
    d2 = np.diff(d1, axis=-1) * fs
    var1, var2 = d1.var(axis=-1), d2.var(axis=-1)
    mobility = np.sqrt(var1 / np.where(var0 > 0, var0, 1))
    complexity = np.sqrt(var2 / np.where(var1 > 0, var1, 1)) / np.where(mobility > 0, mobility, 1)
    return var0, np.where(var0 > 0, mobility, np.nan), np.where(var1 > 0, complexity, np.nan)


#: Neighbour-bin SNR: power at f over the median power 0.75-2.5 Hz away on both
#: sides, the usual steady-state (SSVEP) measure. 1 = no peak at f.
SNR_NEIGHBOURS_HZ = (0.75, 2.5)


def line_frequencies(fs, mains=50.0, highest=None):
    """Where mains interference shows up in a spectrum sampled at `fs`.

    Its harmonics, and - because a board sampling at 125 Hz cannot show
    anything above 62.5 Hz - the frequencies those fold back to. On the
    Cyton+Daisy the 100 and 150 Hz harmonics both land on 25 Hz, which is
    inside the beta band and taller than any EEG.
    """
    fs = float(fs)
    nyquist = fs / 2
    top = float(highest) if highest else 4 * float(mains)
    found = []
    harmonic = float(mains)
    while harmonic <= max(top, nyquist):
        folded = abs(harmonic - fs * round(harmonic / fs))     # what it is seen as
        if 0 < folded <= nyquist and not any(abs(folded - f) < 1e-6 for f in found):
            found.append(folded)
        harmonic += float(mains)
    return tuple(sorted(found))


def neighbour_snr(freqs, power, neighbours=SNR_NEIGHBOURS_HZ):
    """(..., n_freqs) power -> the same shape of peak-over-neighbour ratios."""
    power = np.asarray(power, dtype=float)
    shape = power.shape
    power = power.reshape(-1, shape[-1]) if power.ndim else power.reshape(1, 1)
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size < 2:
        return np.full(shape, np.nan)
    step = float(np.median(np.diff(freqs)))
    lo, hi = neighbours
    offsets = [k for k in range(1, int(hi / step) + 2) if lo - 1e-9 <= k * step <= hi + 1e-9]
    if not offsets:
        return np.full(shape, np.nan)
    offsets = [-k for k in offsets] + offsets
    n = freqs.size
    index = np.arange(n)[:, None] + np.array(offsets)[None, :]
    inside = (index >= 0) & (index < n)
    around = np.where(inside[None], power[:, np.clip(index, 0, n - 1)], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        base = np.nanmedian(np.where(around > 0, around, np.nan), axis=2)
        return (power / base).reshape(shape)


def condition_contrast(psd_a, psd_b, freqs=None, sd_span_hz=1.0):
    """Cohen's d of log power per frequency, A minus B, over trials.

    psd_a, psd_b: (n_trials, n_freqs). The log makes a doubling count the same
    at every frequency; the pooled SD scales each difference by how much single
    trials vary there. Positive: more power in A.

    With `freqs`, the SD is the median of the pooled SDs within +/- sd_span_hz:
    from five trials a single bin's SD is itself noisy, and on 19 Sept test data
    it moved the largest d off a 15 Hz flicker line onto its 14.5 Hz neighbour.
    """
    a = np.log(np.maximum(np.asarray(psd_a, dtype=float), 1e-30))
    b = np.log(np.maximum(np.asarray(psd_b, dtype=float), 1e-30))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Both conditions need (trials, frequencies) spectra on the same frequencies")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each condition needs at least two trials")
    pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2)
    if freqs is not None:
        freqs = np.asarray(freqs, dtype=float)
        near = np.abs(freqs[:, None] - freqs[None, :]) <= sd_span_hz + 1e-9
        with np.errstate(invalid="ignore"):
            pooled = np.nanmedian(np.where(near, pooled[None, :], np.nan), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a.mean(axis=0) - b.mean(axis=0)) / pooled


def strongest_frequencies(freqs, values, low, high, count=3, min_gap=0.6):
    """The `count` largest values between low and high Hz, at least min_gap Hz apart: [(f, value)]."""
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.flatnonzero((freqs >= low) & (freqs <= high) & np.isfinite(values))
    picked = []
    for i in keep[np.argsort(values[keep])[::-1]]:
        if all(abs(freqs[i] - f) >= min_gap for f, _ in picked):
            picked.append((float(freqs[i]), float(values[i])))
        if len(picked) == count:
            break
    return picked
