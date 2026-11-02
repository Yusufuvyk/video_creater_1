"""The blocks a pipeline is built from.

Each block is a thin wrapper around a function `core` already has and has been
checked against (the tests under tests/): the Workbench adds a way to join
steps, not new ways to compute them. The exceptions are the learning blocks,
whose arithmetic is in `pipeline.learn`. What a block computes is a plain
function in `pipeline.compute`, which an exported script calls too; the block
adds its settings, its ports and its words.

`describe` returns the sentence a report's methods section needs, with the
numbers actually used.

Pure numpy - no Qt.
"""
import hashlib
from pathlib import Path

from ..core import epochs as ep
from ..core.wavelets import CWT_WAVELET, DISCRETE
from .graph import (Block, CONTRAST, EPOCHS, FEATURES, ICA, MODEL, Param, Port, RAW, SCORES, SPECTRUM, TF,
                    applies_when, block, register)
from . import compute
from .compute import LEFT_OUT, UNIT_FACTORS, trial_numbers, trial_text  # noqa: F401  (used by the inspector)
from . import filters as fl
from . import learn

INPUT, PREPROCESS, TRIALS, SPECTRAL, FEATURE, STATISTICS = ("Input", "Preprocessing", "Trials", "Spectral",
                                                            "Feature extraction", "Statistics")
#: The steps a model is made of, and the block that evaluates it, under one heading by what they do.
#: Only a classifier ends a model; the steps before it are fitted on each fold's training trials.
LEARNING = "Machine learning"
AUGMENT, SPATIAL, RIEMANN, SCALING, CLASSIFIERS, EVALUATION = (
    "Data augmentation", "Spatial filters", "Riemannian geometry", "Scaling & selection", "Classifiers", "Evaluation")
CATEGORIES = (INPUT, PREPROCESS, TRIALS, SPECTRAL, FEATURE, STATISTICS, AUGMENT, SPATIAL, RIEMANN, SCALING, CLASSIFIERS,
              EVALUATION)
#: category -> the heading it sits under in the block list
GROUPS = {c: LEARNING for c in (AUGMENT, SPATIAL, RIEMANN, SCALING, CLASSIFIERS, EVALUATION)}

_names = compute.names_of

#: When a setting applies (graph.Param.when): the Workbench shows each one only where it means something.
TUNED, UNTUNED = applies_when(tune=True), applies_when(tune=False)
#: Filter designs and the settings each one reads: the order (IIR), the ripple, the attenuation, the transition.
IIR_ONLY = applies_when(design=fl.IIR)
RIPPLE = applies_when(design=("chebyshev1", "elliptic"))
ATTENUATION = applies_when(design=("chebyshev2", "elliptic"))


def _port_type(kind, output):
    """The kind of data `kind`'s output `output` gives."""
    return next(p.types[0] for p in kind.outputs if p.name == output)


def _hz(value):
    return "{:g} Hz".format(value)


# ----------------------------------------------------------------------
# input
# ----------------------------------------------------------------------


@register
class Load(Block):
    type = "load"
    title = "Recording"
    category = INPUT
    outputs = (Port("raw", (RAW,), "Recording"),)
    params = (
        Param("path", "", "File", "Any file GIOVA opens: BrainFlow CSV, EDF/BDF, BrainVision, FIF, "
              "EEGLAB, MATLAB."),
        Param("condition", "", "Part of the file",
              "For a dataset that keeps several conditions in one file (Cho2017): which to read. "
              "Empty: both imagery sides together when the file has them (imagery_left + "
              "imagery_right), else the first part it offers.", advanced=True),
        Param("rename", [], "Rename events", "old=new each, e.g. Primary/P 1=odour: the name the file gives "
              "(case and repeated spaces aside) and the one to use from here on.", advanced=True),
        Param("shift_ms", 0.0, "Move events later (ms)", "Every event this much later; negative: earlier. For "
              "the delay between a trigger and the stimulus itself (an olfactometer's air reaching the nose).",
              advanced=True),
    )
    #: 2: one side of a Cho2017 file read alone has an event per trial
    version = 2

    def fingerprint(self, params):
        """Size and date of the file and of every sidecar it is read with; a missing one counts too."""
        from ..formats.recordings import dependencies
        if not params["path"]:
            return ""
        path = Path(params["path"])
        stat = path.stat()          # the recording itself must be there
        parts = ["{}:{}".format(stat.st_size, stat.st_mtime_ns)]
        for other in dependencies(path)[1:]:
            try:
                stat = other.stat()
                # sidecars are small: their content, so an edit within the same clock tick still counts
                mark = (hashlib.sha1(other.read_bytes()).hexdigest() if stat.st_size < 1_000_000
                        else "{}:{}".format(stat.st_size, stat.st_mtime_ns))
                parts.append("{}={}".format(other.name, mark))
            except OSError:
                parts.append("{}=-".format(other.name))
        return "|".join(parts)

    def run(self, inputs, params, context):
        return {"raw": compute.read_recording(params["path"], params["condition"], params["rename"],
                                              params["shift_ms"], progress=context.progress)}

    def short(self, params):
        return Path(params["path"]).name if params["path"] else "no file chosen"

    def describe(self, params):
        name = Path(params["path"]).name if params["path"] else "(no file)"
        text = "Recording {}{}.".format(name, " ({})".format(params["condition"]) if params["condition"] else "")
        if _names(params["rename"]):
            text += " Events renamed: {}.".format(", ".join(n.replace("=", " → ", 1) for n in _names(params["rename"])))
        if params["shift_ms"]:
            text += " Every event moved {:g} ms {}.".format(abs(params["shift_ms"]),
                                                           "later" if params["shift_ms"] > 0 else "earlier")
        return text


_preferred = compute.preferred_part


# ----------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------


def unit_scale(params):
    """The factor a Units block multiplies the samples by."""
    return compute.unit_scale(params["stored"], params["gain"], params["factor"])


@register
class Units(Block):
    """Samples to µV when the file holds other units or amplified values."""
    type = "units"
    title = "Units / gain"
    category = PREPROCESS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("raw", (RAW,), "Recording"),)
    params = (
        Param("stored", "µV", "The numbers are in", "What the file's numbers really are. EEG is tens of µV: "
              "values near 0.00001 are volts, near 0.01 millivolts.", choices=tuple(UNIT_FACTORS)),
        Param("gain", 1.0, "Amplifier gain", "Divide by the gain the amplifier applied before recording "
              "(×1000 recorded: 1000). 1: none."),
        Param("factor", 1.0, "Other factor", "Any other multiplier, e.g. µV per ADC count from the device's "
              "manual.", advanced=True),
    )

    def run(self, inputs, params, context):
        return {"raw": compute.convert_units(inputs["raw"], **params)}

    def short(self, params):
        parts = [] if params["stored"] == "µV" else ["{} → µV".format(params["stored"])]
        if float(params["gain"]) != 1:
            parts.append("÷ {:g}".format(float(params["gain"])))
        if float(params["factor"]) != 1:
            parts.append("× {:g}".format(float(params["factor"])))
        return ", ".join(parts) or "as read"

    def describe(self, params):
        steps = []
        if params["stored"] != "µV":
            steps.append("converted from {} to µV (× {:g})".format(params["stored"], UNIT_FACTORS[params["stored"]]))
        if float(params["gain"]) != 1:
            steps.append("divided by an amplifier gain of {:g}".format(float(params["gain"])))
        if float(params["factor"]) != 1:
            steps.append("multiplied by {:g}".format(float(params["factor"])))
        return "Samples {}.".format(", ".join(steps)) if steps else "Samples kept in µV as read."


@register
class Channels(Block):
    type = "channels"
    title = "Channels"
    category = PREPROCESS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("raw", (RAW,), "Recording"),)
    params = (Param("channels", [], "Channels", "Names to keep, in this order. Empty: all."),)

    def run(self, inputs, params, context):
        return {"raw": compute.pick_channels(inputs["raw"], **params)}

    def short(self, params):
        wanted = _names(params["channels"])
        return ", ".join(wanted) if 0 < len(wanted) <= 4 else "{} channels".format(len(wanted)) if wanted else "all"

    def describe(self, params):
        wanted = _names(params["channels"])
        return "Channels {}.".format(", ".join(wanted)) if wanted else "All channels."


@register
class TrialChannels(Channels):
    """The Channels block's choice made on trials: the trials are cut once from every electrode (a head map
    still has them all), and only the spectra or features of the channels named are taken further."""
    type = "trial_channels"
    title = "Channels of trials"
    category = TRIALS
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)

    def run(self, inputs, params, context):
        return {"epochs": compute.pick_trial_channels(inputs["epochs"], **params)}

    def describe(self, params):
        wanted = _names(params["channels"])
        return "Channels of the trials: {}.".format(", ".join(wanted)) if wanted else "All channels of the trials."


@register
class BadChannels(Block):
    """Channels that did not record EEG (a loose electrode), rebuilt from their neighbours or taken out."""
    type = "bad_channels"
    title = "Bad channels"
    category = PREPROCESS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("raw", (RAW,), "Recording"),)
    params = (
        Param("channels", [], "Bad channels", "Those to rebuild or take out, e.g. P8. Empty: none."),
        Param("action", "interpolate", "Then", "interpolate: rebuilt from the other channels by spherical splines "
              "(MNE-Python, standard 10-05 positions), so every later step keeps the full set; drop: taken out.",
              choices=("interpolate", "drop")),
    )

    def run(self, inputs, params, context):
        return {"raw": compute.repair_channels(inputs["raw"], **params)}

    def short(self, params):
        bad = _names(params["channels"])
        return "{} {}".format(params["action"], ", ".join(bad)) if bad else "none"

    def describe(self, params):
        bad = _names(params["channels"])
        if not bad:
            return "No channel marked bad."
        if params["action"] == "drop":
            return "Bad channel{} {} left out.".format("s" if len(bad) > 1 else "", ", ".join(bad))
        return ("Bad channel{} {} interpolated from the others by spherical splines (MNE-Python, standard 10-05 "
                "electrode positions).".format("s" if len(bad) > 1 else "", ", ".join(bad)))


@register
class Reference(Block):
    type = "reference"
    title = "Re-reference"
    category = PREPROCESS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("raw", (RAW,), "Recording"),)
    params = (
        Param("mode", "average", "Reference", "average: the mean of all channels; channels: the mean of "
              "the channels named below (e.g. mastoids); laplacian: each channel minus the mean of its four "
              "nearest electrodes (standard 10-05 positions), which sharpens local rhythms such as mu over C3 "
              "and C4.", choices=("average", "channels", "laplacian")),
        Param("channels", [], "Reference channels", when=applies_when(mode="channels")),
    )

    def run(self, inputs, params, context):
        return {"raw": compute.rereference(inputs["raw"], **params)}

    def short(self, params):
        if params["mode"] in ("average", "laplacian"):
            return params["mode"]
        return ", ".join(_names(params["channels"]))

    def describe(self, params):
        if params["mode"] == "average":
            return "Re-referenced to the common average."
        if params["mode"] == "laplacian":
            return ("Re-referenced by a nearest-neighbour (small) Laplacian: each channel minus the mean of its "
                    "four nearest electrodes on standard 10-05 positions; less exact at the edge of "
                    "the montage, where the neighbours lie on one side.")
        return "Re-referenced to the mean of {}.".format(", ".join(_names(params["channels"])))


@register
class Filter(Block):
    type = "filter"
    title = "Filter"
    category = PREPROCESS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("raw", (RAW,), "Recording"),)
    params = (
        Param("low", 1.0, "Low cut (Hz)", "0: no high-pass."),
        Param("high", 40.0, "High cut (Hz)", "0: no low-pass."),
        Param("notch", 50.0, "Mains (Hz)", "Notch at this frequency and its harmonics below Nyquist. 0: off. "
              "Europe 50, the Americas and Korea 60."),
        Param("design", "butterworth", "Design", "IIR families, or a windowed FIR as MNE-Python uses by default. "
              "Chebyshev II: the cut-offs are where the stop-band attenuation is reached.", choices=fl.DESIGNS),
        Param("kind", fl.PASS, "Band", "pass: keep between the cut-offs; stop: remove between them.",
              choices=(fl.PASS, fl.STOP), advanced=True),
        Param("phase", "zero", "Phase", "zero: no delay, for offline analysis; causal: forward only, as a live "
              "system would filter. The ideal filter is zero-phase only.", choices=fl.PHASES, advanced=True,
              when=applies_when(design=fl.IIR + ("fir",))),
        Param("order", 4, "Order", "IIR only. A band-pass of order 4 has 8 poles.", advanced=True, when=IIR_ONLY),
        Param("ripple", 1.0, "Pass-band ripple (dB)", "Chebyshev I and elliptic.", advanced=True, when=RIPPLE),
        Param("attenuation", 40.0, "Stop-band attenuation (dB)", "Chebyshev II and elliptic.", advanced=True,
              when=ATTENUATION),
        Param("transition", 0.0, "Transition (Hz)", "FIR only. 0: automatic, as in MNE.", advanced=True,
              when=applies_when(design="fir")),
        Param("notch_q", 30.0, "Notch Q", advanced=True),
        Param("notches", [], "Notch only at (Hz)", "Notch exactly these frequencies, e.g. 50, 75, with no "
              "harmonics added (as the Analysis page). When given, Mains is not used. Empty: Mains and its "
              "harmonics.", advanced=True),
    )
    version = 2

    def run(self, inputs, params, context):
        return {"raw": compute.filter_recording(inputs["raw"], **params)}

    def short(self, params):
        low, high = params["low"], params["high"]
        band = ("{:g}–{:g} Hz".format(low, high) if low and high else
                "> {:g} Hz".format(low) if low else "< {:g} Hz".format(high) if high else "")
        if band and params["kind"] == fl.STOP:
            band = "stop " + band
        design = "" if params["design"] == "butterworth" else params["design"].replace("chebyshev", "cheby ")
        notch = ("notch " + ", ".join("{:g}".format(f) for f in params["notches"]) if params["notches"] else
                 "notch {:g}".format(params["notch"]) if params["notch"] else "")
        return " · ".join(p for p in (band, design, notch) if p) or "off"

    def describe(self, params):
        return fl.describe(params)


@register
class Resample(Block):
    type = "resample"
    title = "Downsample"
    category = PREPROCESS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("raw", (RAW,), "Recording"),)
    params = (Param("target", 250.0, "At least (Hz)",
                    "Divides the rate by a whole number, keeping it at or above this."),)

    def run(self, inputs, params, context):
        return {"raw": compute.downsample_recording(inputs["raw"], **params)}

    def short(self, params):
        return "≥ {:g} Hz".format(params["target"])

    def describe(self, params):
        return ("Downsampled by a whole factor to at least {} after a zero-phase anti-alias low-pass."
                .format(_hz(params["target"])))


@register
class IcaBlock(Block):
    type = "ica"
    title = "ICA"
    category = PREPROCESS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("raw", (RAW,), "Cleaned"), Port("ica", (ICA,), "Components"))
    params = (
        Param("eog_channels", ["Fp1", "Fp2"], "EOG reference",
              "Channels that carry blinks and eye movements. Components that follow them are removed. "
              "Empty: none found automatically. Blinks are below 4 Hz: put ICA before a filter that "
              "removes them."),
        Param("exclude", [], "Also remove", "Component numbers to remove by hand, e.g. 0, 3."),
        Param("components", 0.99, "Components", "Below 1: enough components to explain this share of the "
              "variance. 1 or more: that many."),
        Param("method", "fastica", "Method", choices=("fastica", "infomax"), advanced=True),
        Param("threshold", 3.0, "EOG threshold (z)", "How far a component's EOG correlation must stand out "
              "from the others (MNE find_bads_eog). With n components no z can exceed (n-1)/sqrt(n): "
              "with fewer than 12, lower it or name components by hand.", advanced=True),
        Param("fit_highpass", 1.0, "Fit on high-pass (Hz)", "ICA is fitted on a copy high-passed at this "
              "frequency, which makes it far more stable, and applied to the recording as it is. 0: off.",
              advanced=True),
        Param("seed", 42, "Seed", advanced=True),
    )

    def run(self, inputs, params, context):
        cleaned, result = compute.run_ica(inputs["raw"], **params, progress=context.progress)
        return {"raw": cleaned, "ica": result}

    def short(self, params):
        eog = _names(params["eog_channels"])
        return "EOG: " + ", ".join(eog) if eog else "manual"

    def describe(self, params):
        eog = _names(params["eog_channels"])
        size = ("components explaining {:.0%} of the variance".format(params["components"])
                if params["components"] < 1 else "{:g} components".format(params["components"]))
        text = ("Independent component analysis ({}, {}) fitted on a {:g} Hz high-passed copy"
                .format(params["method"], size, params["fit_highpass"]) if params["fit_highpass"] else
                "Independent component analysis ({}, {})".format(params["method"], size))
        if eog:
            text += "; components correlating with {} beyond z = {:g} removed (MNE find_bads_eog)".format(
                ", ".join(eog), params["threshold"])
        if _names(params["exclude"]):
            text += "; components {} removed by hand".format(", ".join(_names(params["exclude"])))
        return text + "."


# ----------------------------------------------------------------------
# trials
# ----------------------------------------------------------------------


@register
class Epoch(Block):
    type = "epochs"
    title = "Epochs"
    category = TRIALS
    inputs = (Port("raw", (RAW,), "Recording"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)
    params = (
        Param("conditions", [], "Conditions", "Event labels to cut around. Empty: every label."),
        Param("tmin", 0.0, "From (s)", "Start of each trial relative to its event."),
        Param("tmax", 4.0, "To (s)"),
        Param("reject_uv", 0.0, "Reject above (µV)",
              "Leave out a trial whose signal strays further than this from its mean on any channel. 0: off.",
              advanced=True),
        Param("every", 0.0, "Without markers: every (s)",
              "For a paradigm repeated back to back with no markers: a trial starts every this many seconds "
              "(from 'First at'), and the markers are not used. 0: cut at the markers.", advanced=True,
              when=applies_when(source=compute.EVENTS, event_length=False)),
        Param("first", 0.0, "First at (s)", "Where the first trial starts, when trials repeat without markers.",
              advanced=True, when=applies_when(source=compute.EVENTS, event_length=False)),
        Param("motion_limit", 0.0, "Reject motion above", "Leave out a trial whose accelerometer (LiveAmp, Unicorn) "
              "swings more than this, peak to peak on any axis, in the sensor's unit (mg). 0: off.", advanced=True),
        Param("source", compute.EVENTS, "Trials from", "events: around each event (or every so many seconds "
              "without markers). manual: the trial table. sequence: labelled phases repeated from a start. With "
              "manual, sequence or Each event's length, every trial is used for To − From s from its start, "
              "and From is the time of that start from the trial's time zero (0: the start itself).",
              choices=compute.TRIAL_SOURCES),
        Param("event_length", False, "Each event's length", "Trials from events: each event of non-zero length "
              "from its onset to its end, used for To − From s; events without a length make no trial.",
              advanced=True, when=applies_when(source=compute.EVENTS)),
        Param("trials", [], "Trial table", "Trials from manual: id | condition | start-end s | out (when left "
              "out of the selection), one per entry.", advanced=True, when=applies_when(source=compute.MANUAL)),
        Param("sequence", "", "Sequence", "Trials from sequence: Condition:seconds; ... e.g. Eyes open:30; Eyes "
              "closed:30.", advanced=True, when=applies_when(source=compute.SEQUENCE)),
        Param("sequence_start", 0.0, "Sequence starts at (s)", advanced=True,
              when=applies_when(source=compute.SEQUENCE)),
        Param("repeats", 1, "Repeats", "How many times the sequence is repeated.", advanced=True,
              when=applies_when(source=compute.SEQUENCE)),
        Param("quality_mad", 6.0, "Review above (MAD)", "Trials from manual, sequence or Each event's length: one "
              "whose peak-to-peak is beyond the median plus this many median absolute deviations is marked for "
              "review in the trial table (five trials or more), and so is a flat channel. A mark never leaves a "
              "trial out.", advanced=True),
    )
    version = 2

    def run(self, inputs, params, context):
        return {"epochs": compute.cut_epochs(inputs["raw"], **params)}

    def short(self, params):
        conditions = _names(params["conditions"])
        if params["source"] == compute.MANUAL:
            what = "{} trials by hand".format(len(params["trials"]))
        elif params["source"] == compute.SEQUENCE:
            what = "sequence × {}".format(params["repeats"])
        elif params["event_length"]:
            what = "event lengths"
        else:
            what = ""
        return " · ".join(p for p in ("{:g} to {:g} s".format(params["tmin"], params["tmax"]), what,
                                      ", ".join(conditions)) if p)

    def describe(self, params):
        conditions = _names(params["conditions"])
        defined = params["source"] != compute.EVENTS or params["event_length"]
        if defined:
            origin = {compute.MANUAL: "{} trials defined by hand".format(len(params["trials"])),
                      compute.SEQUENCE: "a sequence ({}) repeated {} time{} from {:g} s".format(
                          params["sequence"], params["repeats"], "" if params["repeats"] == 1 else "s",
                          params["sequence_start"])}.get(params["source"], "each event's own length")
            text = ("Trials from {}{}; each used for {:g} s from its start (its start at {:g} s from time zero), "
                    "trials shorter than that or left out of the selection excluded with their reason{}.".format(
                        origin, " (" + "/".join(conditions) + ")" if conditions else "",
                        params["tmax"] - params["tmin"], params["tmin"],
                        "; trials beyond ±{:g} µV left out".format(params["reject_uv"]) if params["reject_uv"]
                        else ""))
            if params["motion_limit"]:
                text += (" Trials whose accelerometer swung more than {:g} (its unit) on any axis left out."
                         .format(params["motion_limit"]))
            return text
        if params["every"] > 0:
            return "Trials of {:g} s repeated back to back from {:g} s, without markers{}.".format(
                params["tmax"] - params["tmin"], params["first"],
                "; trials beyond ±{:g} µV left out".format(params["reject_uv"]) if params["reject_uv"] else "")
        text = "Trials from {:g} to {:g} s around each {}event{}.".format(
            params["tmin"], params["tmax"], "/".join(conditions) + " " if conditions else "",
            "; trials beyond ±{:g} µV left out".format(params["reject_uv"]) if params["reject_uv"] else "")
        if params["motion_limit"]:
            text += (" Trials whose accelerometer swung more than {:g} (its unit) on any axis left out."
                     .format(params["motion_limit"]))
        return text


@register
class Window(Block):
    type = "window"
    title = "Time window"
    category = TRIALS
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)
    params = (Param("start", 0.5, "From (s)"), Param("end", 2.5, "To (s)"))

    def run(self, inputs, params, context):
        return {"epochs": compute.time_window(inputs["epochs"], **params)}

    def short(self, params):
        return "{:g} to {:g} s".format(params["start"], params["end"])

    def describe(self, params):
        return "Analysed {:g} to {:g} s of each trial.".format(params["start"], params["end"])


@register
class TrialFilter(Filter):
    """The Filter block's settings, applied to each trial and channel on its own.

    No trial can ring into the next through the filter, and a join between
    trials stored end to end (Cho2017) is never filtered across.
    """
    type = "trial_filter"
    title = "Filter trials"
    category = TRIALS
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)
    version = 1

    def run(self, inputs, params, context):
        return {"epochs": compute.filter_trials(inputs["epochs"], **params)}

    def describe(self, params):
        return fl.describe(params).replace("Filtered:", "Each trial filtered on its own:", 1)


@register
class SelectTrials(Block):
    """Keep the trials named by number: after looking at them, the bad ones go."""
    type = "select_trials"
    title = "Select trials"
    category = TRIALS
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)
    params = (
        Param("keep", [], "Keep", "Trial numbers to keep, e.g. 1-3, 5, 7. Empty: all. After a run, click the "
              "trials below to keep or leave them out."),
        Param("leave_out", [], "Leave out", "Trial numbers to leave out of those kept, e.g. 4, 6.", advanced=True),
    )

    def run(self, inputs, params, context):
        return {"epochs": compute.select_trials(inputs["epochs"], **params)}

    def short(self, params):
        keep, leave = _names(params["keep"]), _names(params["leave_out"])
        text = "keep " + ", ".join(keep) if keep else "all"
        return text + (" · not " + ", ".join(leave) if leave else "")

    def describe(self, params):
        keep, leave = _names(params["keep"]), _names(params["leave_out"])
        if not keep and not leave:
            return "Every trial kept."
        return "Trials chosen by hand after inspection: {}{}.".format(
            "kept " + ", ".join(keep) if keep else "all kept", "; left out " + ", ".join(leave) if leave else "")


@register
class WaveletDenoise(Block):
    """Each trial and channel decomposed, its detail levels thresholded, and rebuilt (core.wavelets)."""
    type = "wavelet_denoise"
    title = "Wavelet denoise"
    category = TRIALS
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)
    params = (
        Param("wavelet", "db4", "Wavelet", choices=DISCRETE),
        Param("level", 4, "Levels", "How many times the signal is halved. At 512 Hz, 4 keeps 0-16 Hz (mu) in the "
              "approximation, which is never thresholded. 0: as deep as the trial allows."),
        Param("mode", "hard", "Threshold", "hard: coefficients below the threshold set to zero, the rest kept. "
              "soft: every coefficient also shrunk by it.", choices=("hard", "soft")),
        Param("threshold", 0.0, "Threshold (µV)", "0: the universal threshold, worked out for each trial and "
              "channel from its finest level (Donoho & Johnstone)."),
    )

    def run(self, inputs, params, context):
        return {"epochs": compute.wavelet_denoise(inputs["epochs"], **params)}

    def short(self, params):
        return "{} L{} {} · {}".format(params["wavelet"], params["level"] or "max", params["mode"],
                                       "{:g} µV".format(params["threshold"]) if params["threshold"] else "universal")

    def describe(self, params):
        return ("Each trial denoised on its own: {} wavelet, {} levels, {} thresholding of every detail level at {} "
                "(approximation kept).".format(params["wavelet"], params["level"] or "the most",
                                               params["mode"], "{:g} µV".format(params["threshold"])
                                               if params["threshold"] else "the universal threshold"))


@register
class DwtBands(Block):
    """Each trial split into the DWT's dyadic bands: their amplitude envelopes, and each band's share of energy."""
    type = "dwt_bands"
    title = "DWT bands"
    category = SPECTRAL
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Band envelopes"), Port("features", (FEATURES,), "Energy share"))
    params = (
        Param("wavelet", "db4", "Wavelet", choices=DISCRETE),
        Param("level", 6, "Levels", "At 512 Hz, 6 gives 0-4, 4-8, 8-16, 16-32, 32-64, 64-128 and 128-256 Hz. "
              "0: as deep as the trial allows."),
        Param("envelope", True, "Envelope", "Each band as its amplitude envelope (|Hilbert|), so a rhythm that is "
              "not locked to the cue survives averaging. Off: the band's own waveform."),
    )

    def run(self, inputs, params, context):
        envelopes, energy = compute.dwt_bands(inputs["epochs"], **params, progress=context.progress)
        return {"epochs": envelopes, "features": energy}

    def short(self, params):
        return "{} L{}{}".format(params["wavelet"], params["level"] or "max",
                                 " · envelopes" if params["envelope"] else "")

    def describe(self, params):
        return ("Discrete wavelet transform of each trial ({} wavelet, {} levels), every band rebuilt on its own{}; "
                "each band's share of the trial's energy per channel.".format(
                    params["wavelet"], params["level"] or "the most",
                    " as its amplitude envelope (|Hilbert transform|)" if params["envelope"] else ""))


#: Ang et al. (2012): nine 4 Hz bands from 4 to 40 Hz.
FBCSP_BANDS = [list(band) for band in compute.FBCSP_BANDS]


@register
class FilterBank(Block):
    type = "filter_bank"
    title = "Filter bank"
    category = SPECTRAL
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)
    params = (
        Param("bands", FBCSP_BANDS, "Bands (Hz)", "[low, high] each. Out come the channels band after band, "
              "for FBCSP."),
        Param("start", 0.5, "Keep from (s)", "Trials are filtered forward from their first sample and kept from "
              "here: the time before lets each filter settle, as on a continuous recording. Order-4 Chebyshev II "
              "bands need about 4 s (then within 0.3% of continuous filtering)."),
        Param("end", 2.5, "Keep to (s)"),
        Param("design", "chebyshev2", "Design", "Chebyshev II, as Ang et al. (2012): flat pass band, the "
              "cut-offs where the stop-band attenuation is reached.", choices=fl.IIR, advanced=True),
        Param("order", 4, "Order", advanced=True),
        Param("attenuation", 40.0, "Stop-band attenuation (dB)", "Chebyshev II and elliptic.", advanced=True,
              when=ATTENUATION),
        Param("ripple", 1.0, "Pass-band ripple (dB)", "Chebyshev I and elliptic.", advanced=True, when=RIPPLE),
    )

    def run(self, inputs, params, context):
        return {"epochs": compute.filter_bank(inputs["epochs"], **params)}

    def short(self, params):
        bands = params["bands"]
        return "{} bands {:g}–{:g} Hz".format(len(bands), bands[0][0], bands[-1][1]) if bands else "no bands"

    def describe(self, params):
        return ("Filter bank of {} {} band-passes (order {}, {:g} dB stop band) from {:g} to {:g} Hz, applied "
                "forward (causal) from each trial's first sample and kept from {:g} to {:g} s.".format(
                    len(params["bands"]), params["design"].replace("chebyshev", "Chebyshev type ").replace(
                        "2", "II").replace("1", "I"), params["order"], params["attenuation"],
                    params["bands"][0][0], params["bands"][-1][1], params["start"], params["end"]))


@register
class Baseline(Block):
    type = "baseline"
    title = "Baseline"
    category = TRIALS
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("epochs", (EPOCHS,), "Trials"),)
    params = (Param("start", -0.5, "From (s)"), Param("end", 0.0, "To (s)"))

    def run(self, inputs, params, context):
        return {"epochs": compute.subtract_baseline(inputs["epochs"], **params)}

    def short(self, params):
        return "{:g} to {:g} s".format(params["start"], params["end"])

    def describe(self, params):
        return "Each trial's mean from {:g} to {:g} s subtracted, per channel.".format(params["start"], params["end"])


# ----------------------------------------------------------------------
# spectral
# ----------------------------------------------------------------------

SIGNAL_MEAN_HELP = ("Also the spectrum of each condition's mean waveform (the Analysis page's analysis of the mean "
                    "signal), shown beside the mean of the trials' spectra. Oscillations not locked in phase across "
                    "trials cancel in it. Band power and models still read every trial's spectrum.")
SIGNAL_MEAN_TEXT = (" Also the spectrum of each condition's mean waveform, channel by channel (trials averaged "
                    "first).")


@register
class PSD(Block):
    type = "psd"
    title = "Welch PSD"
    category = SPECTRAL
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("spectrum", (SPECTRUM,), "Spectrum"),)
    params = (Param("window", 1.0, "Segment (s)"), Param("overlap", 0.5, "Overlap", advanced=True),
              Param("detrend", "constant", "Remove", "constant: each trial's mean. linear: its straight-line drift "
                    "over the whole trial first (as the Analysis page's linear detrend).",
                    choices=("constant", "linear"), advanced=True),
              Param("signal_mean", False, "Also of the mean signal", SIGNAL_MEAN_HELP, advanced=True))

    def run(self, inputs, params, context):
        return {"spectrum": compute.welch_spectrum(inputs["epochs"], **params)}

    def short(self, params):
        return "{:g} s segments".format(params["window"])

    def describe(self, params):
        return "Welch power spectral density per trial and channel ({:g} s Hann segments, {:.0%} overlap{}).{}".format(
            params["window"], params["overlap"],
            "; each trial's linear trend removed first" if params["detrend"] == "linear" else "",
            SIGNAL_MEAN_TEXT if params["signal_mean"] else "")


@register
class Fft(Block):
    type = "fft"
    title = "FFT"
    category = SPECTRAL
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("spectrum", (SPECTRUM,), "Spectrum"),)
    params = (Param("detrend", "constant", "Remove", "constant: each trial's mean. linear: its straight-line "
                    "drift, which otherwise leaks into the lowest frequencies.", choices=("constant", "linear")),
              Param("signal_mean", False, "Also of the mean signal", SIGNAL_MEAN_HELP, advanced=True))

    def run(self, inputs, params, context):
        return {"spectrum": compute.fft_spectrum(inputs["epochs"], **params)}

    def short(self, params):
        return "amplitude"

    def describe(self, params):
        return ("FFT amplitude spectrum of each whole trial and channel, one-sided, its {} removed first "
                "(no window).{}".format("mean" if params["detrend"] == "constant" else "linear trend",
                                        SIGNAL_MEAN_TEXT if params["signal_mean"] else ""))


@register
class BandPower(Block):
    type = "band_power"
    title = "Band power"
    category = FEATURE
    inputs = (Port("spectrum", (SPECTRUM,), "Spectrum"),)
    outputs = (Port("features", (FEATURES,), "Features"),)
    params = (
        Param("bands", [["mu", 8.0, 13.0], ["beta", 13.0, 30.0]], "Bands", "[name, low Hz, high Hz] each."),
        Param("log", True, "Logarithm", "log10 of each power: closer to normal, better for LDA."),
    )

    def run(self, inputs, params, context):
        return {"features": compute.band_power(inputs["spectrum"], **params)}

    def short(self, params):
        return ", ".join(b[0] for b in params["bands"]) + (" · log" if params["log"] else "")

    def describe(self, params):
        bands = ", ".join("{} {:g}–{:g} Hz".format(*b) for b in params["bands"])
        return "Band power ({}) integrated from the PSD{}.".format(bands, ", log10" if params["log"] else "")


@register
class FeatureSet(Block):
    type = "features"
    title = "Signal features"
    category = FEATURE
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("features", (FEATURES,), "Features"),)
    params = (
        Param("selected", ["rms", "mobility", "complexity", "entropy"], "Features",
              "Ids from the feature catalogue: band:alpha, relative:alpha, ratio:alpha/beta, rms, ptp, "
              "variance, mobility, complexity, crossings, entropy, peak:alpha, total."),
        Param("window", 1.0, "PSD segment (s)", advanced=True),
        Param("overlap", 0.5, "PSD overlap", advanced=True),
        Param("detrend", "constant", "PSD after removing", "constant: each trial's mean. linear: its straight-line "
              "drift first (as the Analysis page's linear detrend). Time-domain features always lose only the mean.",
              choices=("constant", "linear"), advanced=True),
        Param("bands", [list(b) for b in compute.FEATURE_BANDS], "Bands", "The bands band:, relative: and peak: "
              "features use, [name, low Hz, high Hz] each.", advanced=True),
        Param("ratios", [list(r) for r in compute.FEATURE_RATIOS], "Ratios", "name=band/band each, e.g. "
              "alpha/beta=alpha/beta.", advanced=True),
        Param("total_range", [1.0, 45.0], "Total range (Hz)", "What relative: features divide by, and the range of "
              "total and entropy.", advanced=True),
        Param("roi", False, "One region", "Average each feature over the channels, each channel measured on its "
              "own first, as the Analysis page combines a region of several channels.", advanced=True),
    )

    def run(self, inputs, params, context):
        return {"features": compute.signal_features(inputs["epochs"], **params, progress=context.progress)}

    def short(self, params):
        return "{} features{}".format(len(params["selected"]), " · region" if params["roi"] else "")

    def describe(self, params):
        text = self._describe(params)
        defaults = self.defaults()
        if any(params[k] != defaults[k] for k in ("bands", "ratios", "total_range", "detrend", "overlap")):
            text += (" Bands {}; ratios {}; total range {:g}–{:g} Hz; Welch {:g} s, {:.0%} overlap{}.".format(
                ", ".join("{} {:g}–{:g} Hz".format(*b) for b in params["bands"]) or "none",
                ", ".join("{} = {}/{}".format(*r) for r in params["ratios"]) or "none", *params["total_range"],
                params["window"], params["overlap"],
                ", linear trend removed first" if params["detrend"] == "linear" else ""))
        if params["roi"]:
            text += " Each feature averaged over the channels, each measured on its own."
        return text

    def _describe(self, params):
        return "Per-trial features: {}.".format(", ".join(params["selected"]))


@register
class CompareWindows(Block):
    """Two windows of the same trials compared: band power channel by channel, a paired test across trials."""
    type = "compare_windows"
    title = "Compare windows"
    category = STATISTICS
    inputs = (Port("epochs", (EPOCHS,), "Trials"),)
    outputs = (Port("contrast", (CONTRAST,), "Comparison"),)
    params = (
        Param("window_a", [0.0, 2.0], "Window A (s)", "From and to, in each trial, e.g. the stimulus."),
        Param("window_b", [4.0, 10.0], "Window B (s)", "The window A is compared with, e.g. the rest after it."),
        Param("name_a", "A", "Name of A", "As the results call it, e.g. odour."),
        Param("name_b", "B", "Name of B", "e.g. air."),
        Param("bands", [list(b) for b in compute.THETA_ALPHA_BETA], "Bands", "[name, low Hz, high Hz] each."),
        Param("segment", 1.0, "Welch segment (s)", "Both windows' power from pieces this long (or the shorter "
              "window), so both have the same frequency resolution.", advanced=True),
        Param("test", "wilcoxon", "Test", "wilcoxon: signed-rank test of the log power, trial by trial, no "
              "assumption of normality; t-test: paired t-test of the log power.", choices=("wilcoxon", "t-test"),
              advanced=True),
    )

    def run(self, inputs, params, context):
        return {"contrast": compute.compare_windows(inputs["epochs"], **params)}

    def short(self, params):
        return "{} {:g}–{:g} s vs {} {:g}–{:g} s".format(params["name_a"], *params["window_a"], params["name_b"],
                                                       *params["window_b"])

    def describe(self, params):
        bands = ", ".join("{} {:g}–{:g} Hz".format(*b) for b in params["bands"])
        return ("Band power ({}) of {} ({:g}–{:g} s) compared with {} ({:g}–{:g} s) in every trial and channel: "
                "Welch density of {:g} s Hann segments integrated over each band; {} of the log10 power across "
                "trials, p values adjusted by Holm's method over every channel and band.".format(
                    bands, params["name_a"], *params["window_a"], params["name_b"], *params["window_b"],
                    params["segment"], "paired Wilcoxon signed-rank test" if params["test"] == "wilcoxon"
                    else "paired t-test"))


#: How each time-frequency method is named on a block.
METHOD_NAMES = {ep.STFT: "STFT", ep.MORLET: "Morlet", ep.CWT: "CWT"}


@register
class TimeFrequency(Block):
    type = "erd"
    title = "Time-frequency / ERD"
    category = SPECTRAL
    #: a recording is cut into trials here; trials (after Baseline, Filter trials, Select trials) are used as
    #: they come, so whatever was done to them before reaches the map
    inputs = (Port("raw", (RAW, EPOCHS), "Recording or trials"),)
    outputs = (Port("tf", (TF,), "Time-frequency"),)
    params = (
        Param("channels", ["C3", "Cz", "C4"], "Channels", "Empty: every channel of the trials (with trials in)."),
        Param("conditions", [], "Conditions", "Empty: every event label."),
        Param("tmin", -1.0, "From (s)", "From a recording only: trials keep their own span."),
        Param("tmax", 4.0, "To (s)", "From a recording only: trials keep their own span."),
        Param("baseline", [-1.0, 0.0], "Baseline (s)", when=applies_when(baseline_mode=ep.WINDOW)),
        Param("active", [0.5, 2.5], "Active window (s)"),
        Param("method", ep.STFT, "Method", "stft: one window for every frequency. morlet: MNE's Morlet wavelets. "
              "cwt: complex Morlet scalogram (pywt's cmor1.5-1.0), {} frequencies from Lowest to Highest, as "
              "the lab scripts draw it.".format(ep.CWT_FREQS), choices=(ep.STFT, ep.MORLET, ep.CWT)),
        Param("fmin", 4.0, "Lowest (Hz)", advanced=True), Param("fmax", 40.0, "Highest (Hz)", advanced=True),
        Param("window", 0.5, "STFT window (s)", advanced=True, when=applies_when(method=ep.STFT)),
        Param("overlap", 0.9, "Overlap", advanced=True, when=applies_when(method=ep.STFT)),
        Param("smooth", 0.5, "Smoothing (s)", "Power smoothed over this long in time, per trial, before ERD. "
              "0: none, so one trial's amplitude is its raw |STFT|.", advanced=True),
        Param("bands", [["mu", 8.0, 13.0], ["beta", 13.0, 30.0]], "Bands", advanced=True),
        Param("topography", False, "Scalp map", "ERD of every electrode, for the head map. Slower.",
              advanced=True),
        Param("baseline_mode", ep.WINDOW, "Reference", "window: each trial's own baseline window; condition: "
              "the mean power of a reference condition (eyes open for eyes closed).",
              choices=(ep.WINDOW, ep.CONDITION), advanced=True),
        Param("reference", "", "Reference condition", "For Reference = condition. Empty: the first condition of "
              "the trials (as the Analysis page chose it).", advanced=True,
              when=applies_when(baseline_mode=ep.CONDITION)),
        Param("erd_mode", ep.POOLED, "Average", "pooled: the trials' mean power against their mean baseline "
              "(robust). mean_of_trials: each trial's % change, then averaged. Used by every map, band course "
              "and trace.", choices=(ep.POOLED, ep.MEAN_OF_TRIALS), advanced=True),
        Param("n_cycles", 0.5, "Morlet cycles per Hz", "Morlet only: cycles of each wavelet per Hz of its "
              "frequency, kept between 2 and 12.", advanced=True, when=applies_when(method=ep.MORLET)),
        Param("wavelet", CWT_WAVELET, "CWT wavelet", "CWT only: a complex Morlet, cmor<bandwidth>-<centre>.",
              advanced=True, when=applies_when(method=ep.CWT)),
        Param("reject_uv", 0.0, "Reject above (µV)", "Leave out a trial whose signal strays further than this "
              "from its mean on an analysed channel. 0: off.", advanced=True),
    )

    version = 2

    def run(self, inputs, params, context):
        return {"tf": compute.time_frequency(inputs["raw"], **params, progress=context.progress)}

    def short(self, params):
        return "{} · {}".format(METHOD_NAMES[params["method"]],
                                ", ".join(_names(params["channels"])) or "all channels")

    def describe(self, params, trials=False):
        method = {ep.STFT: "STFT ({:g} s Hann window, {:.0%} overlap)".format(params["window"], params["overlap"]),
                  ep.MORLET: "Morlet wavelet ({:g} cycles per Hz, 2 to 12 cycles)".format(params["n_cycles"]),
                  ep.CWT: "Complex Morlet CWT ({}, {} frequencies)".format(
                      params["wavelet"], ep.CWT_FREQS)}[params["method"]]
        span = ("of each whole trial" if trials else
                "of each trial from {:g} to {:g} s around each event".format(params["tmin"], params["tmax"]))
        smooth = "smoothed over {:g} s".format(params["smooth"]) if params["smooth"] else "not smoothed"
        average = (" Each trial's % change averaged over the trials." if params["erd_mode"] == ep.MEAN_OF_TRIALS
                   else "")
        reject = (" Trials beyond ±{:g} µV on an analysed channel left out.".format(params["reject_uv"])
                  if params["reject_uv"] else "")
        if params["baseline_mode"] == ep.CONDITION:
            reference = ("'{}'".format(params["reference"]) if str(params["reference"]).strip()
                         else "the first condition of the trials")
            return ("{} time-frequency power {}, {}; ERD/ERS as % change from the mean power of "
                    "{} in the {:g}–{:g} s window.{}{}".format(method, span, smooth, reference,
                                                              *params["active"], average, reject))
        return ("{} time-frequency power {}, {}; ERD/ERS as % change from the {:g}–{:g} s "
                "baseline.{}{}".format(method, span, smooth, *params["baseline"], average, reject))

    def describe_in(self, graph, node_id):
        wire = graph.feeds(node_id).get("raw")
        source = block(graph.nodes[wire.source].type) if wire else None
        trials = source is not None and _port_type(source, wire.output) == EPOCHS
        return self.describe(graph.params(node_id), trials)


# ----------------------------------------------------------------------
# machine learning
# ----------------------------------------------------------------------


def _tune_options(folds=learn.INNER_FOLDS):
    """The search's own settings, behind Options on every block that can Tune."""
    return (Param("inner_folds", folds, "Inner folds", "Tune only: folds of the cross-validation inside each "
                  "training fold that compares the values.", advanced=True, when=TUNED),
            Param("tune_score", learn.ACCURACY, "Tune by", "Tune only: what the search keeps the best of. "
                  "balanced accuracy: the mean recall over the conditions, fair when they have unequal trials.",
                  choices=learn.TUNE_SCORES, advanced=True, when=TUNED))


def _listed(values):
    return ", ".join("{:g}".format(v) if isinstance(v, (int, float)) else str(v) for v in values)


def _tuned(params, searched):
    """The methods sentence's tail for a tuned step: what was searched, how, and by which score."""
    return (" Its {} chosen by {}-fold cross-validation inside each training fold, keeping the best {}."
            .format(searched, params["inner_folds"], params["tune_score"]) if params["tune"] else "")


class _Step(Block):
    """A learning step: takes the model so far (or nothing) and returns it one step longer."""
    category = SCALING
    inputs = (Port("model", (MODEL,), "Model so far", optional=True),)
    outputs = (Port("model", (MODEL,), "Model"),)
    step = ""

    def run(self, inputs, params, context):
        return {"model": inputs.get("model", learn.Model()).then(self.step, **self.arguments(params))}

    def arguments(self, params):
        return dict(params)


@register
class AugmentStep(_Step):
    type = "augment"
    title = "Augment"
    category = AUGMENT
    step = "augment"
    params = (Param("copies", 2, "Copies", "New trials made from each training trial."),
              Param("noise", 0.1, "Noise (× SD)", "Gaussian noise, as a share of each channel's standard deviation."),
              Param("scale", 0.1, "Scale ±", "Each copy's amplitude multiplied by 1 ± up to this much."),
              Param("shift", 0.05, "Shift ±", "Trials moved in time by up to this share of their length (trials "
                                              "only, not features).", advanced=True))

    def short(self, params):
        return "× {}".format(int(params["copies"]) + 1)

    def describe(self, params):
        return ("Training trials augmented in every fold: {} copies of each with Gaussian noise at {:g} of each "
                "channel's SD, amplitude scaled by up to ±{:.0%} and shifted by up to ±{:.0%} of the trial; the "
                "test trials were never augmented.".format(int(params["copies"]), params["noise"], params["scale"],
                                                           params["shift"]))


@register
class CSPStep(_Step):
    type = "csp"
    title = "CSP"
    category = SPATIAL
    step = "csp"
    params = (Param("n_components", 6, "Filters", "How many spatial filters to keep (the most "
                                                  "discriminative, from both ends)."),
              Param("log", True, "Log power", advanced=True))

    def short(self, params):
        return "{} filters".format(params["n_components"])

    def describe(self, params):
        return "Common spatial patterns, {} filters, log-variance features.".format(params["n_components"])


@register
class TRCSPStep(_Step):
    type = "trcsp"
    title = "TRCSP"
    category = SPATIAL
    step = "trcsp"
    params = (Param("n_filters", 3, "Filters per condition", "Spatial filters kept for each of the two "
                    "conditions (Lotte & Guan, 2011: 3)."),
              Param("tune", True, "Tune", "Choose alpha from 1e-10 to 1e-1 by a cross-validation inside each "
                    "training fold, as the paper does."),
              Param("alpha", "1e-05", "Alpha", "Penalty on large filter weights when not tuned. 0 would be "
                    "plain CSP.", choices=tuple("{:.0e}".format(a) for a in learn.TRCSP_ALPHAS), advanced=True,
                    when=UNTUNED),
              Param("inner_folds", 10, "Inner folds", "Folds of the search for alpha (the paper: 10).",
                    advanced=True, when=TUNED),
              _tune_options()[1])

    def arguments(self, params):
        return {**params, "alpha": float(params["alpha"])}

    def short(self, params):
        return "{} + {} filters · {}".format(params["n_filters"], params["n_filters"],
                                            "alpha tuned" if params["tune"] else "alpha " + params["alpha"])

    def describe(self, params):
        alpha = ("alpha chosen from 1e-10 to 1e-1 by {}-fold cross-validation inside each training fold "
                 "(best {})".format(params["inner_folds"], params["tune_score"]) if params["tune"]
                 else "alpha = {}".format(params["alpha"]))
        return ("Tikhonov-regularised CSP (Lotte & Guan, 2011), {} filters per condition, trial covariances "
                "normalised by their trace, {}; log-variance features.".format(params["n_filters"], alpha))


@register
class FBCSPStep(_Step):
    type = "fbcsp"
    title = "FBCSP"
    category = SPATIAL
    step = "fbcsp"
    params = (Param("n_bands", 9, "Bands", "As many as the Filter bank before the trials gives."),
              Param("n_pairs", 2, "Filter pairs per band", "The first and last this many CSP filters of each band "
                    "(Ang et al., 2012: 2)."))

    def short(self, params):
        return "{} bands × {} pairs".format(params["n_bands"], params["n_pairs"])

    def describe(self, params):
        return ("Filter bank CSP (Ang et al., 2012): in each of {} bands, the first and last {} CSP filters, fitted "
                "on the training trials; features the log of each filtered signal's variance over the band's "
                "total.".format(params["n_bands"], params["n_pairs"]))


@register
class StandardizeStep(_Step):
    type = "standardize"
    title = "Standardize"
    category = SCALING
    step = "standardize"

    def describe(self, params):
        return "Features standardised to zero mean and unit variance on the training trials."


@register
class LDAStep(_Step):
    type = "lda"
    title = "LDA"
    category = CLASSIFIERS
    step = "lda"
    params = (Param("shrinkage", "auto", "Shrinkage", "auto: Ledoit-Wolf; none: plain covariance.",
                    choices=("auto", "none")),)

    def short(self, params):
        return "shrinkage " + params["shrinkage"]

    def describe(self, params):
        return "Linear discriminant analysis{}.".format(
            " with Ledoit-Wolf shrinkage" if params["shrinkage"] == "auto" else "")


@register
class SelectStep(_Step):
    type = "select"
    title = "Select features"
    category = SCALING
    step = "select"
    params = (Param("k", 10, "Keep", "How many features to keep: those whose ANOVA F between the conditions "
                                     "is largest on the training trials."),)

    def short(self, params):
        return "best {}".format(params["k"])

    def describe(self, params):
        return "The {} features with the largest ANOVA F, chosen on the training trials.".format(params["k"])


@register
class MIBIFStep(_Step):
    type = "mibif"
    title = "MIBIF"
    category = SCALING
    step = "mibif"
    params = (Param("k", 4, "Best features", "Kept by their mutual information with the condition, on the "
                    "training trials (Ang et al., 2012: 4)."),
              Param("pair_size", 4, "Features per band", "2 × FBCSP's pairs: each kept feature brings its CSP "
                    "partner. 0: no partners.", advanced=True))

    def short(self, params):
        return "best {}{}".format(params["k"], " + partners" if params["pair_size"] else "")

    def describe(self, params):
        return ("The {} features with the largest mutual information with the condition (Parzen window "
                "estimate), chosen on the training trials{} (MIBIF; Ang et al., 2012).".format(
                    params["k"], ", each with its CSP partner" if params["pair_size"] else ""))


@register
class CovarianceStep(_Step):
    type = "covariance"
    title = "Covariance"
    category = RIEMANN
    step = "covariance"
    params = (Param("estimator", "oas", "Estimator", "oas and lwf shrink the estimate, which keeps it usable "
                                                     "with many channels; scm is the plain one.",
                    choices=("oas", "lwf", "scm")),)

    def short(self, params):
        return params["estimator"]

    def describe(self, params):
        return "Spatial covariance matrix of each trial ({} estimator).".format(params["estimator"].upper())


@register
class TangentStep(_Step):
    type = "tangent"
    title = "Tangent space"
    category = RIEMANN
    step = "tangent"

    def describe(self, params):
        return ("Covariance matrices projected to the Riemannian tangent space at the training trials' mean "
                "(Barachant et al., 2012).")


@register
class SVMStep(_Step):
    type = "svm"
    title = "SVM"
    category = CLASSIFIERS
    step = "svm"
    params = (Param("kernel", "linear", "Kernel", choices=("linear", "rbf")),
              Param("C", 1.0, "C", "Larger: fits the training trials more closely.", when=UNTUNED),
              Param("tune", False, "Tune", "Search C (and gamma for rbf) inside each training fold."),
              Param("gamma", "scale", "Gamma", "rbf only, when not tuned: the kernel's width. scale: "
                    "1 / (features × their variance), scikit-learn's default; or a number.", advanced=True,
                    when=applies_when(kernel="rbf", tune=False)),
              Param("C_values", list(learn.C_GRID), "C values", "Tune only: the C values searched.", advanced=True,
                    when=TUNED),
              Param("gamma_values", [str(v) for v in learn.GAMMA_GRID], "Gamma values", "Tune with rbf only: the "
                    "gamma values searched (scale or numbers).", advanced=True,
                    when=applies_when(kernel="rbf", tune=True)),
              *_tune_options())

    def short(self, params):
        return params["kernel"] + (" · tuned" if params["tune"] else " · C {:g}".format(params["C"]))

    def describe(self, params):
        rbf = params["kernel"] == "rbf"
        fixed = "" if params["tune"] else ", C = {:g}{}".format(params["C"], ", gamma = {}".format(
            params["gamma"]) if rbf else "")
        searched = "C (from {})".format(_listed(params["C_values"])) + (
            " and gamma (from {})".format(_listed(params["gamma_values"])) if rbf else "")
        return "Support vector machine, {} kernel{}.".format(params["kernel"], fixed) + \
            _tuned(params, searched + (" were" if rbf else " was"))


@register
class LogisticStep(_Step):
    type = "logistic"
    title = "Logistic regression"
    category = CLASSIFIERS
    step = "logistic"
    params = (Param("penalty", "l2", "Penalty", "l1 keeps few features, l2 shrinks all.", choices=("l2", "l1")),
              Param("C", 1.0, "C", "Larger: weaker penalty.", when=UNTUNED),
              Param("tune", False, "Tune", "Search C inside each training fold."),
              Param("C_values", list(learn.C_GRID), "C values", "Tune only: the C values searched.", advanced=True,
                    when=TUNED),
              *_tune_options())

    def short(self, params):
        return params["penalty"] + (" · tuned" if params["tune"] else " · C {:g}".format(params["C"]))

    def describe(self, params):
        fixed = "" if params["tune"] else ", C = {:g}".format(params["C"])
        return "Logistic regression, {} penalty{}.".format(params["penalty"], fixed) + \
            _tuned(params, "C (from {}) was".format(_listed(params["C_values"])))


@register
class ForestStep(_Step):
    type = "forest"
    title = "Random forest"
    category = CLASSIFIERS
    step = "forest"
    params = (Param("trees", 200, "Trees"),
              Param("tune", False, "Tune", "Search tree depth and leaf size inside each training fold."),
              Param("depth", 0, "Depth", "When not tuned: the deepest a tree grows. 0: until its leaves are pure "
                    "(scikit-learn's default).", advanced=True, when=UNTUNED),
              Param("leaf", 1, "Leaf size", "When not tuned: the fewest training trials a leaf holds.",
                    advanced=True, when=UNTUNED),
              Param("depth_values", ["none" if v is None else str(v) for v in learn.TREE_GRID["max_depth"]],
                    "Depth values",
                    "Tune only: the depths searched (none: no limit).", advanced=True, when=TUNED),
              Param("leaf_values", list(learn.TREE_GRID["min_samples_leaf"]), "Leaf size values",
                    "Tune only: the leaf sizes searched.", advanced=True, when=TUNED),
              *_tune_options())

    def short(self, params):
        return "{} trees{}".format(params["trees"], " · tuned" if params["tune"] else "")

    def describe(self, params):
        fixed = "" if params["tune"] else "".join(
            [", depth at most {}".format(params["depth"]) if int(params["depth"] or 0) else "",
             ", leaves of at least {} trials".format(params["leaf"]) if int(params["leaf"]) > 1 else ""])
        return "Random forest of {} trees (Breiman, 2001){}.".format(params["trees"], fixed) + \
            _tuned(params, "depth (from {}) and leaf size (from {}) were".format(
                _listed(params["depth_values"]), _listed(params["leaf_values"])))


@register
class KNNStep(_Step):
    type = "knn"
    title = "kNN"
    category = CLASSIFIERS
    step = "knn"
    params = (Param("k", 5, "Neighbours", when=UNTUNED),
              Param("tune", False, "Tune", "Search the number of neighbours inside each training fold."),
              Param("k_values", list(learn.NEIGHBOUR_GRID), "Neighbour values", "Tune only: the numbers of "
                    "neighbours searched.", advanced=True, when=TUNED),
              *_tune_options())

    def short(self, params):
        return "k = {}{}".format(params["k"], " · tuned" if params["tune"] else "")

    def describe(self, params):
        fixed = "" if params["tune"] else ", k = {}".format(params["k"])
        return "k-nearest neighbours{}.".format(fixed) + \
            _tuned(params, "k (from {}) was".format(_listed(params["k_values"])))


@register
class MDMStep(_Step):
    type = "mdm"
    title = "MDM"
    category = CLASSIFIERS
    step = "mdm"

    def describe(self, params):
        return ("Minimum distance to the Riemannian mean of each condition's covariance matrices "
                "(Barachant et al., 2012).")


@register
class NBPWStep(_Step):
    type = "nbpw"
    title = "NBPW"
    category = CLASSIFIERS
    step = "nbpw"

    def describe(self, params):
        return ("Naive Bayesian Parzen window classifier: each feature's density per condition by Gaussian "
                "Parzen windows, (4/3n)^(1/5)·σ wide, multiplied (Ang et al., 2012).")


@register
class CrossValidate(Block):
    type = "cross_validate"
    title = "Cross-validate"
    category = EVALUATION
    inputs = (Port("data", (EPOCHS, FEATURES), "Trials or features"),
              Port("model", (MODEL,), "Model"))
    outputs = (Port("scores", (SCORES,), "Scores"),)
    params = (
        Param("folds", 5, "Folds", when=applies_when(scheme=(learn.KFOLD, learn.REPEATED))),
        Param("conditions", [], "Conditions", "Which conditions to tell apart. Empty: all."),
        Param("scheme", learn.KFOLD, "Split", "k-fold: shuffled, stratified folds. repeated k-fold: the same "
              "again with other shuffles, averaged. chronological: train on the first trials, test on the "
              "last, as a BCI meets new data.", choices=learn.SCHEMES),
        Param("repeats", 5, "Repeats", "For repeated k-fold.", advanced=True,
              when=applies_when(scheme=learn.REPEATED)),
        Param("train_fraction", 0.8, "Training share", "For the chronological split.", advanced=True,
              when=applies_when(scheme=learn.CHRONOLOGICAL)),
        Param("permutations", 0, "Permutation test", "0: off. Otherwise the whole cross-validation is run this "
              "many times again with the labels shuffled, and p is the share that scores as well (Ojala & "
              "Garriga, 2010). Stricter than the random limit, and slow: 200 shuffles take 200 runs.",
              advanced=True),
        Param("seed", 42, "Shuffle seed", "Same seed, same folds: results can be repeated exactly.",
              advanced=True),
    )

    def run(self, inputs, params, context):
        return {"scores": compute.evaluate(inputs["data"], inputs["model"], **params, progress=context.progress)}

    def short(self, params):
        if params["scheme"] == learn.CHRONOLOGICAL:
            return "first {:.0%} train".format(params["train_fraction"])
        if params["scheme"] == learn.REPEATED:
            return "{} × {} folds".format(params["repeats"], params["folds"])
        return "{} folds".format(params["folds"])

    def describe(self, params):
        if params["scheme"] == learn.CHRONOLOGICAL:
            return ("Chronological split: the first {:.0%} of the trials trained the model, the rest tested it; "
                    "every step fitted on the training trials only.".format(params["train_fraction"]))
        repeated = ("{} times repeated ".format(params["repeats"]) if params["scheme"] == learn.REPEATED else "")
        text = ("{}stratified {}-fold cross-validation (seed {}); every step fitted on the training "
                "folds only.".format(repeated, params["folds"], params["seed"])).capitalize()
        text += (" Accuracy judged against the upper limit of a random classifier's accuracy for the number "
                 "of tested trials (alpha 0.05; Müller-Putz et al., 2008); with unequal conditions, balanced "
                 "accuracy also against a random classifier's, simulated for the same trials. Balanced accuracy, "
                 "Cohen's kappa (Cohen, 1960) and macro F1 are, like accuracy, the mean of the folds' own.")
        if params["permutations"]:
            text += (" Permutation test with {} label shuffles (Ojala & Garriga, 2010); better than random "
                     "also needs p ≤ 0.05.".format(int(params["permutations"])))
        return text
