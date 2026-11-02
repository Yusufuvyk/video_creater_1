"""Wavelets: denoising, dyadic bands (DWT) and the complex Morlet scalogram (CWT).

The same arithmetic as a lab's reference code (`lib/wavelet.py` of its Python conversion, not
public), so a Workbench pipeline and the lab script give the same numbers; the lab's
reasons are kept beside each function.

* Denoising: `pywt.wavedec`, every detail level thresholded (hard or soft), the
  approximation kept, `waverec`. Without a threshold given, each signal gets the
  universal threshold sigma * sqrt(2 ln n), sigma from the finest detail's median
  absolute value / 0.6745 (Donoho & Johnstone, 1994).
* Bands: each coefficient array reconstructed on its own. The DWT is orthogonal,
  so band energies add up to the signal's energy: "31% of the power is in
  8-16 Hz" is a statement the redundant CWT cannot make.
* CWT: a complex Morlet ("cmor1.5-1.0", pywt's definition), so |coefficient| is
  the amplitude envelope, smooth; a real wavelet's magnitude collapses every half
  cycle. Computed from the wavelet's formula rather than with `pywt.cwt` (see
  `cwt_amplitude`). Here a sine of amplitude A reads A, as everywhere in
  GIOVA; the lab (like scipy's STFT) gives A/2.

Pure numpy/scipy + PyWavelets (denoising, bands) - no Qt. pywt is imported when first used.
"""
import numpy as np

#: The complex Morlet the lab uses: bandwidth 1.5, centre frequency 1.0.
CWT_WAVELET = "cmor1.5-1.0"
#: Discrete wavelets offered for denoising and bands.
DISCRETE = ("db4", "db2", "db6", "db8", "sym4", "sym8", "coif3", "haar")


def max_level(n, wavelet):
    """The deepest level that still makes sense for `n` samples and this wavelet."""
    import pywt
    return pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)


def decompose(x, wavelet="db4", level=0):
    """(coefficients [cA_L, cD_L, ..., cD_1], level); level 0: the deepest."""
    import pywt
    level = int(level) or max_level(len(x), wavelet)
    return pywt.wavedec(np.asarray(x, dtype=float), wavelet, level=level), level


def universal_threshold(finest):
    """sigma * sqrt(2 ln n), sigma estimated from the finest detail level."""
    sigma = np.median(np.abs(finest)) / 0.6745
    return sigma * np.sqrt(2.0 * np.log(len(finest)))


def denoise(x, wavelet="db4", level=4, mode="hard", threshold=0.0):
    """(denoised signal, threshold used, level used). Threshold 0: the universal one for this signal."""
    import pywt
    coefficients, level = decompose(x, wavelet, level)
    used = float(threshold) if threshold else universal_threshold(coefficients[-1])
    if used > 0:
        coefficients = [coefficients[0]] + [pywt.threshold(c, used, mode=mode) for c in coefficients[1:]]
    # a threshold of 0 (a flat signal has no noise to estimate) removes nothing; pywt's soft rule would
    # divide 0 by 0 on its zero coefficients and fill the signal with NaN
    return pywt.waverec(coefficients, wavelet)[:len(x)], used, level


def bands(x, fs, wavelet="db4", level=6):
    """[(name, low Hz, high Hz, signal)] slow to fast: each coefficient array rebuilt on its own.

    cA_L holds 0 to fs / 2^(L+1); cD_j holds fs / 2^(j+1) to fs / 2^j.
    """
    import pywt
    coefficients, level = decompose(x, wavelet, level)
    nyquist = fs / 2.0
    out = []
    for index in range(len(coefficients)):
        parts = [np.zeros_like(c) for c in coefficients]
        parts[index] = coefficients[index]
        signal = pywt.waverec(parts, wavelet)[:len(x)]
        if index == 0:
            out.append(("cA{}".format(level), 0.0, nyquist / 2 ** level, signal))
        else:
            j = level - index + 1
            out.append(("cD{}".format(j), nyquist / 2 ** j, nyquist / 2 ** (j - 1), signal))
    return sorted(out, key=lambda band: band[1])


def cmor_parameters(wavelet):
    """(bandwidth B, centre frequency C) of a complex Morlet named as pywt names it, "cmorB-C"."""
    try:
        bandwidth, centre = (float(v) for v in str(wavelet).removeprefix("cmor").split("-"))
    except ValueError:
        raise ValueError("A complex Morlet is written cmorB-C, e.g. cmor1.5-1.0") from None
    if bandwidth <= 0 or centre <= 0:
        raise ValueError("A complex Morlet's bandwidth and centre frequency must be above 0")
    return bandwidth, centre


def cmor_cycles(wavelet):
    """How many cycles a complex Morlet cmorB-C spans, as a Morlet wavelet's n_cycles counts them: 2 pi C sqrt(B / 2).

    Its Gaussian exp(-u^2 / B), with u = t f / C, has the standard deviation C sqrt(B / 2) / f seconds, and a
    Morlet of n cycles has n / (2 pi f): cmor1.5-1.0 is a Morlet of 5.44 cycles at every frequency.
    """
    bandwidth, centre = cmor_parameters(wavelet)
    return 2.0 * np.pi * centre * np.sqrt(bandwidth / 2.0)


def cwt_amplitude(x, fs, freqs, wavelet=CWT_WAVELET):
    """|CWT| as amplitude, (n_freqs, n_samples): a sine of amplitude A at one of `freqs` reads A.

    pywt's complex Morlet, psi(u) = exp(-u²/B) exp(2πiCu) / sqrt(πB) on -8 <= u <= 8,
    stretched to s = C fs / f samples, sampled at every sample and convolved (FFT)
    with the signal; dividing by s gives the amplitude normalisation the lab applies
    to pywt's output. Not `pywt.cwt` itself: it differentiates an integrated wavelet
    tabulated at 2^precision points, which at large scales (low frequencies) is a
    staircase that lets broadband noise in. On white noise its 1 Hz row read 4 to 11
    times this exact value, and differed between two installs of the same pywt 1.8.0;
    on a pure 4 Hz sine both read 1.2% low, where this reads it exactly.
    """
    from scipy.signal import fftconvolve
    bandwidth, centre = cmor_parameters(wavelet)
    x = np.asarray(x, dtype=float)
    out = np.empty((len(freqs), x.size))
    for row, f in enumerate(np.asarray(freqs, dtype=float)):
        s = centre * fs / f
        u = np.arange(-np.ceil(8 * s), np.ceil(8 * s) + 1) / s
        psi = np.exp(-u ** 2 / bandwidth) * np.exp(2j * np.pi * centre * u) / np.sqrt(np.pi * bandwidth)
        out[row] = 2.0 * np.abs(fftconvolve(x, np.conj(psi[::-1]), mode="same")) / s
    return out
