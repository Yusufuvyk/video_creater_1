"""ADS1299 internal-input-short diagnostic for the Cyton+Daisy.

The ADS1299 CHnSET MUX value 001 shorts a channel's inputs internally.  With
the electrode inputs disconnected by that mux, a recording measures the
ADC/PGA chain rather than electrode contact, reference wiring or room pickup.

All channels are tested together at gain 24, outside BIAS and SRB.  The caller
must restore the operator's channel settings when the measurement ends.
"""
from dataclasses import dataclass

import numpy as np

from .hardware import CHANNEL_CHARS
from .spectral import band_integral, welch_psd

SETTLE_SECONDS = 1.5
MEASURE_SECONDS = 20.0


def channel_command(index):
    """Channel on, gain 24, input shorted, outside BIAS/SRB2/SRB1."""
    if not 0 <= index < len(CHANNEL_CHARS):
        raise ValueError("Cyton+Daisy has channels 1-16, not {}".format(index + 1))
    return "x{}061000X".format(CHANNEL_CHARS[index])


@dataclass(frozen=True)
class Result:
    channel: int
    mean_uv: float
    noise_uv: float
    mains_uv: float
    mains_ratio: float
    status: str


def analyse(samples_uv, fs):
    """Return per-channel DC, centred noise RMS and 50 Hz power measurements.

    ``status`` is an intentionally conservative screening aid, not an ADS1299
    production-test limit.  A tiny absolute 50 Hz component is accepted even
    when its percentage is unstable because the total noise is nearly zero.
    """
    data = np.atleast_2d(np.asarray(samples_uv, dtype=float))
    if data.ndim != 2 or data.shape[1] < round(float(fs)):
        raise ValueError("not enough internal-short samples")
    results = []
    for channel, row in enumerate(data):
        finite = row[np.isfinite(row)]
        if finite.size != row.size:
            results.append(Result(channel, float("nan"), float("nan"),
                                  float("nan"), float("nan"), "Missing samples"))
            continue
        mean = float(np.mean(row))
        noise = float(np.std(row - mean))
        freqs, psd = welch_psd(row, fs, detrend="linear")
        high = min(60.0, float(fs) / 2)
        total = float(band_integral(freqs, psd, 1.0, high))
        line_power = (float(band_integral(freqs, psd, 49.0, min(51.0, high)))
                      if high > 49 else float("nan"))
        mains = float(np.sqrt(max(line_power, 0.0))) if np.isfinite(line_power) else float("nan")
        ratio = line_power / total if total > 1e-20 and np.isfinite(line_power) else float("nan")
        clean_mains = not np.isfinite(ratio) or ratio <= 0.20 or mains <= 0.5
        status = "Clean" if abs(mean) <= 50.0 and noise <= 5.0 and clean_mains else "Check"
        results.append(Result(channel, mean, noise, mains, ratio, status))
    return results


class Capture:
    """Collect settling plus a fixed-length raw internal-short recording."""

    def __init__(self, channels, fs, settle=SETTLE_SECONDS, measure=MEASURE_SECONDS):
        self.channels = list(channels)
        self.fs = float(fs)
        self.settle = int(round(settle * fs))
        self.measure = int(round(measure * fs))
        self.seen = 0
        self.parts = []
        self.results = None

    @property
    def progress(self):
        return min(max(self.seen - self.settle, 0) / max(self.measure, 1), 1.0)

    def feed(self, chunk):
        if self.results is not None or chunk.shape[1] == 0:
            return self.results is not None
        start = max(self.settle - self.seen, 0)
        self.seen += chunk.shape[1]
        if start < chunk.shape[1]:
            self.parts.append(chunk[:, start:])
        if sum(part.shape[1] for part in self.parts) < self.measure:
            return False
        data = np.concatenate(self.parts, axis=1)[:, :self.measure]
        all_results = analyse(data, self.fs)
        wanted = set(self.channels)
        self.results = [result for result in all_results if result.channel in wanted]
        return True


def summary(results, names):
    return ", ".join(
        "{} {}: mean {:+.2f} uV, noise {:.2f} uV RMS, 50 Hz {:.1f}%".format(
            names[r.channel], r.status, r.mean_uv, r.noise_uv,
            100 * r.mains_ratio if np.isfinite(r.mains_ratio) else float("nan"))
        for r in results)


def summary_html(results, names):
    colour = {"Clean": "#1a7f37", "Check": "#cf222e", "Missing samples": "#6e7781"}
    if not results:
        return "No result"
    clean = sum(r.status == "Clean" for r in results)
    finite_noise = [r for r in results if np.isfinite(r.noise_uv)]
    finite_mains = [r for r in results if np.isfinite(r.mains_ratio)]
    worst_noise = max(finite_noise, key=lambda r: r.noise_uv) if finite_noise else None
    worst_mains = max(finite_mains, key=lambda r: r.mains_ratio) if finite_mains else None
    header_colour = colour["Clean"] if clean == len(results) else colour["Check"]
    header = '<span style="color:{}"><b>{}/{} channels clean</b></span>'.format(
        header_colour, clean, len(results))
    details = []
    if worst_noise is not None:
        details.append("worst noise <b>{}</b> {:.2f} µV RMS".format(
            names[worst_noise.channel], worst_noise.noise_uv))
    if worst_mains is not None:
        details.append("worst 50 Hz <b>{}</b> {:.1f}%".format(
            names[worst_mains.channel], 100 * worst_mains.mains_ratio))
    rows = [header + (" · " + " · ".join(details) if details else "")]
    # Keep the 300 px settings panel usable: clean channels are represented by
    # the aggregate above; only channels needing attention get their own row.
    for r in (item for item in results if item.status != "Clean"):
        ratio = "—" if not np.isfinite(r.mains_ratio) else "{:.1f}%".format(100 * r.mains_ratio)
        rows.append('<span style="color:{}">● <b>{}</b> {} · mean {:+.2f} µV · '
                    'noise {:.2f} µV RMS · 50 Hz {}</span>'.format(
                        colour[r.status], names[r.channel], r.status,
                        r.mean_uv, r.noise_uv, ratio))
    return "<br>".join(rows)
