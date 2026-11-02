"""Electrode impedance on the Cyton+Daisy, from the ADS1299 AC lead-off test.

Method (the one the OpenBCI GUI uses, with two of its failure modes closed):
the chip drives a 6 nA current at 31.25 Hz (fCLK / 2^16) into one channel's
N input, where the electrode is plugged in. The current flows through that
electrode, the head and the BIAS electrode back to the board, so the channel
sees a 31.25 Hz tone whose peak voltage is I x Z. The board adds a 2.2 kOhm
series resistor per input, which is subtracted.

    Z = peak(31.25 Hz tone) / 6 nA - 2.2 kOhm

During the test the channel runs at gain 1 (a dry electrode's offset would
rail it at gain 24), with SRB2 disconnected and in BIAS, as the GUI does.
BrainFlow tracks the gain sent through config_board and scales accordingly.

What the GUI gets wrong, and this module does not:
  * it reads the standard deviation of the whole channel, so noise on a lead
    with no current flowing reads as a plausible impedance. Here only the
    31.25 Hz tone counts, and it must stand clearly above the neighbouring
    frequencies, or the result is "no test signal".
  * without a return path (normally the BIAS electrode on the skin) the
    current cannot flow; a channel with no clear tone is reported as such
    instead of as a number.
  * it restores the board default afterwards (every channel on, in BIAS);
    the caller here restores the operator's own settings.

The number is the whole current loop: this electrode + head + the BIAS
electrode. It is approximate (the ADS1299 datasheet describes the current as
alternately sourced and sunk, i.e. a square wave, and Daisy mode samples at
125 Hz): good for telling touching / poor / open apart, not a certified value.

Colour limits and where they come from (no published standard exists for
passive dry electrodes; Help → "Impedance check" gives the sources):
  green  < 750 kOhm, yellow < 2500 kOhm, red above - the OpenBCI GUI defaults
         (CytonElectrodeStatus.pde), which use this same lead-off measurement
         on this same board, so the numbers are directly comparable.
  "Open?" >= 4 MOhm - OpenBCI docs: ~5000 kOhm with no electrode attached;
         14 Sept data: apparently disconnected leads 4.6-5.4 MOhm.
Signal attenuation is not the limit at these values: with the ADS1299's 1 GOhm
input impedance, 1 MOhm costs ~0.1% (Ferree et al. 2001 analysis). What grows
with impedance, and with the mismatch between an electrode and the reference,
is mains and movement pick-up - hence the need to also watch the 50 Hz column.
"""
from dataclasses import dataclass

import numpy as np

from .hardware import CHANNEL_CHARS

LEAD_OFF_HZ = 31.25
DRIVE_AMPS = 6e-9
SERIES_OHMS = 2200.0
#: The test runs at gain 1, so the rail is the ADS1299's full ±4.5 V, not the
#: ±187.5 mV of gain 24 (verified on 14 Sept GUI lead-off data: live readings
#: up to ~2 V, railed channels pinned at 4,500 mV).
RAIL_UV = 4_500_000.0
MIN_TONE_UV = 1.0        # below this nothing is driving the channel (off / zeros)

SETTLE_SECONDS = 1.5     # gain change and DC step after the command
MEASURE_SECONDS = 4.0    # 0.25 Hz bins: 31.25 Hz falls on a bin
MIN_SNR = 4.0            # tone amplitude over the median neighbouring bin
GREEN_OHMS = 750e3       # OpenBCI GUI impedanceGreenCutoff (kOhm 750)
YELLOW_OHMS = 2.5e6      # OpenBCI GUI impedanceYellowCuttoff (kOhm 2500)
#: OpenBCI docs give ~5000 kOhm for a channel with no electrode attached; on the
#: 14 Sept GUI lead-off data such leads read 4.6-5.4 MOhm with an unusually
#: clean tone (current leaking through cable capacitance), electrodes on the
#: head 0.8-2 MOhm. So >= 4 MOhm is reported as "Open?", not as a number.
OPEN_OHMS = 4e6

COLOURS = {"green": "#1a7f37", "yellow": "#b08800", "red": "#cf222e", "grey": "#6e7781"}


def test_command(index):
    """Channel on, gain 1, normal input, in BIAS, SRB2 off; N-side lead-off on."""
    c = CHANNEL_CHARS[index]
    return "x{}000100Xz{}01Z".format(c, c)


def stop_command(index):
    """Lead-off current off. The channel's own settings are restored separately."""
    c = CHANNEL_CHARS[index]
    return "z{}00Z".format(c)


def format_ohms(ohms):
    return ("{:.0f} kΩ".format(ohms / 1e3) if ohms < 1e6 else "{:.1f} MΩ".format(ohms / 1e6))


@dataclass(frozen=True)
class ImpedanceResult:
    channel: int
    status: str                  # Measured / Open? / No test signal / Railed / Missing samples
    ohms: float = float("nan")
    tone_uv: float = float("nan")
    snr: float = float("nan")
    peak_hz: float = float("nan")
    rail: float = float("nan")

    @property
    def text(self):
        """The value, or what prevented one. No verbal grade: the colour carries it."""
        if self.status == "Measured":
            return format_ohms(self.ohms)
        if self.status == "Open?":
            return "Open? ({})".format(format_ohms(self.ohms))
        return self.status

    def level(self, green=GREEN_OHMS, yellow=YELLOW_OHMS):
        """green / yellow / red for a number, grey when there is none."""
        if self.status == "Open?":
            return "red"
        if self.status != "Measured":
            return "grey"
        return "green" if self.ohms < green else "yellow" if self.ohms < yellow else "red"


def estimate(channel, samples_uv, fs):
    """Impedance of one channel from its samples recorded during the test."""
    x = np.asarray(samples_uv, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        return ImpedanceResult(channel, "Missing samples")
    if not np.isfinite(fs) or fs / 2 <= LEAD_OFF_HZ + 1:
        return ImpedanceResult(channel, "No test signal")
    if x.size < fs:
        return ImpedanceResult(channel, "No test signal")
    rail = float(np.mean(np.abs(x) >= 0.95 * RAIL_UV))
    if rail > 0.1:
        return ImpedanceResult(channel, "Railed", rail=rail)
    x = x - np.polyval(np.polyfit(np.arange(x.size), x, 1), np.arange(x.size))
    window = np.hanning(x.size)
    amplitude = np.abs(np.fft.rfft(x * window)) * 2 / window.sum()   # peak µV per bin
    freqs = np.fft.rfftfreq(x.size, 1 / fs)
    band = (freqs >= LEAD_OFF_HZ - 1) & (freqs <= LEAD_OFF_HZ + 1)
    peak = int(np.flatnonzero(band)[np.argmax(amplitude[band])])
    # Hann spreads a tone over three bins; summing them in power recovers its
    # amplitude whether or not it sits exactly on a bin.
    tone = float(np.sqrt(np.sum(amplitude[peak - 1:peak + 2] ** 2) / 1.5))
    side = ((freqs >= LEAD_OFF_HZ - 6) & (freqs <= LEAD_OFF_HZ - 2)) | \
           ((freqs >= LEAD_OFF_HZ + 2) & (freqs <= LEAD_OFF_HZ + 6))
    noise = float(np.median(amplitude[side])) if side.any() else 0.0
    snr = tone / noise if noise > 0 else (float("inf") if tone > 0 else 0.0)
    if snr < MIN_SNR or tone < MIN_TONE_UV:
        return ImpedanceResult(channel, "No test signal", tone_uv=tone, snr=snr,
                               peak_hz=float(freqs[peak]), rail=rail)
    ohms = max(tone * 1e-6 / DRIVE_AMPS - SERIES_OHMS, 0.0)
    status = "Open?" if ohms >= OPEN_OHMS else "Measured"
    return ImpedanceResult(channel, status, ohms, tone, snr, float(freqs[peak]), rail)


def _hint(results):
    if results and all(r.status in ("No test signal", "Railed") for r in results):
        return "no channel carried the test current: is the BIAS electrode on the skin?"
    return ""


def summary(results, names):
    """One plain line for the status bar and the hardware log."""
    line = ", ".join("{} {}".format(names[r.channel], r.text) for r in results)
    hint = _hint(results)
    return line + (" · " + hint if hint else "")


def summary_html(results, names, green=GREEN_OHMS, yellow=YELLOW_OHMS):
    """Coloured dot and value per channel, plus whether every channel is green."""
    cells = ['<span style="color:{}">● <b>{}</b> {}</span>'.format(
        COLOURS[r.level(green, yellow)], names[r.channel], r.text) for r in results]
    ready = bool(results) and all(r.level(green, yellow) == "green" for r in results)
    hint = _hint(results)
    verdict = ('<span style="color:{}"><b>All green</b></span>'.format(COLOURS["green"]) if ready
               else "Not all green · adjust the non-green electrodes and check again")
    return "&nbsp;&nbsp;".join(cells) + "<br>" + (hint if hint else verdict), ready


class Sweep:
    """Tests channels one at a time. Driven by sample counts, not wall time.

    `feed` takes the board's chunks as they arrive and returns what the caller
    must do next: ("switch", previous, next), ("done", previous, None) or None.
    """

    def __init__(self, channels, fs, settle=SETTLE_SECONDS, measure=MEASURE_SECONDS):
        self.channels = list(channels)
        self.fs = float(fs)
        self.settle = int(round(settle * fs))
        self.measure = int(round(measure * fs))
        self.position = 0
        self.results = []
        self._seen = 0
        self._samples = []

    @property
    def current(self):
        return self.channels[self.position] if self.position < len(self.channels) else None

    @property
    def finished(self):
        return self.current is None

    def feed(self, chunk):
        if self.finished or chunk.shape[1] == 0:
            return None
        channel = self.current
        start = max(self.settle - self._seen, 0)
        self._seen += chunk.shape[1]
        if start < chunk.shape[1]:
            self._samples.append(chunk[channel, start:])
        if sum(s.size for s in self._samples) < self.measure:
            return None
        data = np.concatenate(self._samples)[:self.measure]
        self.results.append(estimate(channel, data, self.fs))
        self.position += 1
        self._seen = 0
        self._samples = []
        return ("done", channel, None) if self.finished else ("switch", channel, self.current)
