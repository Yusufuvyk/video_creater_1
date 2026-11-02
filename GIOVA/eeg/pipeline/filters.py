"""Filter design for the Workbench's Filter block: IIR families, FIR, zero-phase or causal.

The Monitor and Analysis pages keep `core.processing` (Butterworth, zero-phase
offline). This module adds the choices a researcher reporting a method needs:

* IIR: Butterworth, Chebyshev I (pass-band ripple), Chebyshev II (stop-band
  attenuation), elliptic (both), each as second-order sections.
* FIR: designed by MNE-Python itself (`mne.filter.create_filter`, Hamming
  window, firwin, automatic transition bands; Widmann et al., 2015), so a
  pipeline here and an MNE script use the same taps.
* Ideal: the FFT of each run, bins outside the band set to zero and back
  (zero-phase, a brick wall that rings in time), as teaching code often does.
* Zero-phase (IIR forward and backward; FIR applied centred) or causal
  (forward only, as a live system would).

Butterworth, zero-phase, band-pass: the same coefficients and the same
`sosfiltfilt` as `core.processing`, so earlier pipelines give the same numbers.

Pure numpy/scipy - no Qt.
"""
import numpy as np
from scipy import signal

from ..core.processing import finite_runs, mains_harmonics

IIR = ("butterworth", "chebyshev1", "chebyshev2", "elliptic")
DESIGNS = IIR + ("fir", "ideal")
PHASES = ("zero", "causal")
PASS, STOP = "pass", "stop"
_FTYPE = {"butterworth": "butter", "chebyshev1": "cheby1", "chebyshev2": "cheby2", "elliptic": "ellip"}


def band(low, high, kind=PASS):
    """(btype, edges) for scipy, from the block's low / high (0 = off)."""
    if low and high:
        if low >= high:
            raise ValueError("Low cut must be below high cut")
        return ("bandpass" if kind == PASS else "bandstop"), [low, high]
    if kind == STOP:
        raise ValueError("A band-stop needs both a low and a high edge")
    if low:
        return "highpass", [low]
    if high:
        return "lowpass", [high]
    return None, []


def design(fs, low=0.0, high=0.0, kind=PASS, design="butterworth", order=4, ripple=1.0,
           attenuation=40.0, transition_hz=0.0):
    """("sos", array) for IIR, ("fir", taps) for FIR, ("ideal", (fs, btype, edges)) for the FFT mask,
    or (None, None) when there is nothing to do."""
    nyq = fs / 2.0
    for edge in (low, high):
        if edge and not 0 < edge < nyq:
            raise ValueError("Cut-offs must be above 0 and below Nyquist ({:g} Hz)".format(nyq))
    btype, edges = band(low, high, kind)
    if btype is None:
        return None, None
    if design in IIR:
        if int(order) < 1:
            raise ValueError("Filter order must be at least 1")
        # a high- or low-pass has one edge, which scipy wants as a number, not a list of one
        sos = signal.iirfilter(int(order), edges[0] if len(edges) == 1 else edges, rp=ripple, rs=attenuation,
                               btype=btype,
                               ftype=_FTYPE[design], fs=fs, output="sos")
        return "sos", sos
    if design == "ideal":
        return "ideal", (float(fs), btype, list(edges))
    if design != "fir":
        raise ValueError("Unknown filter design '{}'".format(design))
    import mne
    l_freq, h_freq = (low or None), (high or None)
    if btype == "bandstop":
        l_freq, h_freq = high, low          # MNE's convention: l_freq > h_freq means band-stop
    width = transition_hz or "auto"
    try:
        taps = mne.filter.create_filter(None, fs, l_freq, h_freq, method="fir", fir_design="firwin",
                                        fir_window="hamming", phase="zero", l_trans_bandwidth=width,
                                        h_trans_bandwidth=width, verbose="error")
    except ValueError as exc:
        raise ValueError("FIR design: {}".format(exc)) from None
    return "fir", np.asarray(taps)


def notch_sos(fs, mains, q, harmonics=True):
    """Notches at `mains` and its harmonics below Nyquist; with `harmonics` False, at each frequency of the
    list `mains` only, as `core.processing.design_filter` builds them."""
    freqs = mains_harmonics(mains, fs) if harmonics else [float(f) for f in mains]
    stages = [signal.tf2sos(*signal.iirnotch(f, q, fs)) for f in freqs if 0 < f < fs / 2]
    return np.vstack(stages) if stages else None


def shortest(kind, coefficients):
    """Fewest samples a run needs for this filter; shorter runs are left missing, as in core.processing."""
    if kind == "ideal":
        return 2
    return 3 * (2 * coefficients.shape[0] + 1) + 1 if kind == "sos" else coefficients.size


def ideal_gain(freqs, btype, edges):
    """1 where the ideal filter keeps a frequency, 0 where it removes it; edges are kept."""
    if btype == "lowpass":
        return (freqs <= edges[0]).astype(float)
    if btype == "highpass":
        return (freqs >= edges[0]).astype(float)
    inside = (freqs >= edges[0]) & (freqs <= edges[1])
    return (inside if btype == "bandpass" else ~inside).astype(float)


def _run(kind, coefficients, x, phase):
    if kind == "ideal":
        fs, btype, edges = coefficients
        gain = ideal_gain(np.fft.rfftfreq(x.size, 1.0 / fs), btype, edges)
        return np.fft.irfft(np.fft.rfft(x) * gain, n=x.size)
    if kind == "sos":
        return signal.sosfilt(coefficients, x) if phase == "causal" else signal.sosfiltfilt(coefficients, x)
    if phase == "causal":
        return signal.lfilter(coefficients, 1.0, x)
    # Linear phase, applied centred: no delay, the FIR's own zero-phase form. The run is
    # mirrored at both ends first (as MNE pads), so its edges do not ring from a step to zero.
    pad = min(coefficients.size, x.size - 1)
    padded = np.pad(x, pad, mode="reflect")
    return signal.oaconvolve(padded, coefficients, mode="same")[pad:pad + x.size]


def apply(x, fs, steps, phase="zero"):
    """`x` (one channel) through each (kind, coefficients), every run of finite samples on its own.

    Missing samples stay missing and no filter rings across them, as in core.processing.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    minimum = max((shortest(k, c) for k, c in steps), default=2)
    for start, end in finite_runs(np.isfinite(x)):
        if end - start < minimum:
            continue
        run = x[start:end] - x[start:end].mean()
        for kind, coefficients in steps:
            run = _run(kind, coefficients, run, phase)
        out[start:end] = run
    return out


def steps(fs, params):
    """The filters a Filter block's parameters ask for, in order: notch, then the band."""
    chain = []
    if params.get("notches"):
        # the frequencies listed, each once: no harmonics are added
        sos = notch_sos(fs, params["notches"], params.get("notch_q", 30.0), harmonics=False)
        if sos is not None:
            chain.append(("sos", sos))
    elif params.get("notch"):
        sos = notch_sos(fs, params["notch"], params.get("notch_q", 30.0))
        if sos is not None:
            chain.append(("sos", sos))
    kind, coefficients = design(fs, params.get("low") or 0.0, params.get("high") or 0.0,
                                params.get("kind", PASS), params.get("design", "butterworth"),
                                params.get("order", 4), params.get("ripple", 1.0),
                                params.get("attenuation", 40.0), params.get("transition", 0.0))
    if kind == "ideal" and params.get("phase", "zero") == "causal":
        raise ValueError("The ideal filter works on the whole run at once: zero-phase only")
    if kind is not None:
        chain.append((kind, coefficients))
    if len(chain) == 2 and chain[0][0] == chain[1][0] == "sos":
        # one chain, notch first, as core.processing builds it: the same numbers as before
        chain = [("sos", np.vstack([chain[0][1], chain[1][1]]))]
    return chain


def response(fs, params, n=2048):
    """(frequencies, gain in dB) of the whole chain as it is applied (zero-phase squares the gain)."""
    freqs = np.linspace(0, fs / 2, n)
    gain = np.ones(n)
    zero = params.get("phase", "zero") == "zero"
    for kind, coefficients in steps(fs, params):
        if kind == "ideal":
            h = ideal_gain(freqs, coefficients[1], coefficients[2])
        elif kind == "sos":
            _, h = signal.sosfreqz(coefficients, worN=freqs, fs=fs)
        else:
            _, h = signal.freqz(coefficients, worN=freqs, fs=fs)
        # as `_run` applies each: an IIR (the notch too) forward and backward, its magnitude twice;
        # the FIR centred and the ideal filter once
        gain *= np.abs(h) ** (2 if zero and kind == "sos" else 1)
    return freqs, 20 * np.log10(np.maximum(gain, 1e-12))


def describe(params):
    """The methods sentence for a Filter block."""
    low, high, kind = params.get("low") or 0, params.get("high") or 0, params.get("kind", PASS)
    name = {"butterworth": "Butterworth", "chebyshev1": "Chebyshev type I", "chebyshev2": "Chebyshev type II",
            "elliptic": "elliptic", "fir": "FIR (Hamming-windowed, designed by MNE-Python)",
            "ideal": "ideal FFT"}[
        params.get("design", "butterworth")]
    if low and high:
        what = "band-{} {:g}–{:g} Hz".format("pass" if kind == PASS else "stop", low, high)
    elif low:
        what = "high-pass {:g} Hz".format(low)
    elif high:
        what = "low-pass {:g} Hz".format(high)
    else:
        what = ""
    parts = []
    if what:
        detail = []
        if params.get("design") == "ideal":
            detail.append("every frequency bin outside the band set to zero in each run's FFT, zero-phase")
        elif params.get("design", "butterworth") != "fir":
            detail.append("order {}".format(params.get("order", 4)))
            if params.get("design") in ("chebyshev1", "elliptic"):
                detail.append("{:g} dB pass-band ripple".format(params.get("ripple", 1.0)))
            if params.get("design") in ("chebyshev2", "elliptic"):
                detail.append("{:g} dB stop-band attenuation".format(params.get("attenuation", 40.0)))
        else:
            detail.append("transition {}".format("{:g} Hz".format(params["transition"])
                                                  if params.get("transition") else "automatic as in MNE"))
        if params.get("design") != "ideal":
            phase = ("zero-phase, applied forward and backward" if params.get("design", "butterworth") != "fir"
                     else "zero-phase, linear-phase taps applied centred") if params.get("phase", "zero") == "zero" \
                else "causal, forward only"
            detail.append(phase)
        parts.append("{} {} ({})".format(name, what, "; ".join(detail)))
    if params.get("notches"):
        parts.append("notch at {} Hz only (Q = {:g})".format(
            ", ".join("{:g}".format(f) for f in params["notches"]), params.get("notch_q", 30.0)))
    elif params.get("notch"):
        parts.append("notch at {:g} Hz and its harmonics (Q = {:g})".format(params["notch"], params.get("notch_q", 30.0)))
    return ("Filtered: " + "; ".join(parts) + ".") if parts else "No filtering."
