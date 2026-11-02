"""What each block computes, as a plain function of its inputs and settings.

A block's `run` calls its function here, and an exported script calls the very
same function (copied into it by `bundle`): one place for the arithmetic, so
the sheet and the script cannot give different numbers. The keyword defaults
are the block's own defaults (tests check that they stay equal), so a call
that names only the settings changed on the sheet means what the sheet means.

Inputs and outputs are the containers blocks pass along: a core Recording,
`data.Epochs`, `Spectrum`, `Features`, a core EpochResult, `learn.Scores`.
`progress`, when given, is told what is being done.

Pure numpy - no Qt.
"""
from pathlib import Path
import warnings

import numpy as np

from ..core.analysis_profiles import parse_sequence
from ..core.epochs import (CONDITION, EpochSettings, EventTable, PER_HZ, POOLED, SCALP_STFT, STFT, WINDOW,
                           compute_epochs, compute_trials, events_for, load_label_map, tf_method)
from ..core.config import BANDS
from ..core.feature_profiles import FeatureProfile, extract, mean_values
from ..core.sources import downsample
from ..core.spectral import band_integral, sample_index, welch_psd
from ..core.trials import Settings, Trial, periodic_trials, reference_signal, trial_status
from ..core.wavelets import CWT_WAVELET
from ..formats import channels as channel_names, matfile, presets
from .data import AMPLITUDE, Contrast, Epochs, Features, IcaResult, Spectrum, derive
from . import filters as fl
from . import learn

#: Ang et al. (2012): nine 4 Hz bands from 4 to 40 Hz.
FBCSP_BANDS = tuple((float(low), float(low + 4)) for low in range(4, 40, 4))
#: Bands a spectrum or a time-frequency map is summarised in, unless others are given.
MU_BETA = (("mu", 8.0, 13.0), ("beta", 13.0, 30.0))
#: Bands two windows are compared in, unless others are given.
THETA_ALPHA_BETA = (("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0))
#: What the numbers of a file may be in, and the factor that makes them µV.
UNIT_FACTORS = {"µV": 1.0, "mV": 1e3, "V": 1e6, "nV": 1e-3}
#: The feature profile's bands and ratios unless others are given (core.feature_profiles.FeatureProfile's).
FEATURE_BANDS = tuple(tuple(b) for b in BANDS)
FEATURE_RATIOS = (("alpha/beta", "alpha", "beta"), ("theta/alpha", "theta", "alpha"))
#: Why a trial left out by Select trials is missing.
LEFT_OUT = "Left out by hand"


def names_of(value):
    """Channel or condition names from a list or a comma-separated text."""
    if isinstance(value, str):
        value = value.split(",")
    return [str(v).strip() for v in value or () if str(v).strip()]


def _say(progress, text):
    if progress:
        progress(text)


def summary(value):
    """One line about a block's output, as a script prints it."""
    if hasattr(value, "describe"):
        return value.describe()
    if hasattr(value, "tf") and hasattr(value, "epoch_ids"):        # a time-frequency result
        return "Time-frequency ({}) of {}: {}".format(
            value.settings.method, ", ".join(c.name for c in value.tf),
            ", ".join("{} ×{}".format(c, len(value.epoch_ids[c])) for c in value.conditions))
    return repr(value)


# ----------------------------------------------------------------------
# input
# ----------------------------------------------------------------------


def read_recording(path, condition="", rename=(), shift_ms=0.0, progress=None):
    """Any file GIOVA opens, with its events: the Recording block. `rename` and `shift_ms`: see
    `adjust_events`."""
    if not path:
        raise ValueError("Choose a recording file")
    path = Path(path)
    _say(progress, "Reading {}".format(path.name))
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return read_mat_recording(path, condition, rename, shift_ms)
    if suffix in MNE_SUFFIXES:
        return read_mne_recording(path, rename, shift_ms)
    return read_brainflow_recording(path, rename, shift_ms)


def _label_key(label):
    """A label as it is matched for renaming: case and runs of spaces do not matter ("P  1" is "p 1")."""
    return " ".join(str(label).split()).casefold()


def adjust_events(raw, rename=(), shift_ms=0.0):
    """The recording with its events renamed and moved.

    rename: "old=new" each, e.g. "Primary/P 1=odour"; the old name as the file gives it, case and
    repeated spaces aside. shift_ms: every event moved this much later (negative: earlier), e.g. the
    delay between a stimulator's trigger and the stimulus reaching the subject.
    """
    pairs = []
    for item in names_of(rename):
        old, equals, new = item.partition("=")
        if not equals or not old.strip() or not new.strip():
            raise ValueError("Write each renaming as old=new, e.g. Primary/P 1=odour ('{}' is not)".format(item))
        pairs.append((_label_key(old), new.strip(), old.strip()))
    if not pairs and not shift_ms:
        return raw
    events = raw.events
    if events is None or not len(events):
        raise ValueError("This recording has no events to rename or shift")
    known = {_label_key(label) for label in events.labels}
    unknown = [typed for old, _, typed in pairs if old not in known]
    if unknown:
        raise ValueError("No event named {}; the events are {}".format(
            ", ".join(unknown), ", ".join(events.unique_labels())))
    mapping = {old: new for old, new, _ in pairs}
    out = derive(raw)
    out.events = EventTable(events.onsets + float(shift_ms) / 1000.0, events.durations,
                            [mapping.get(_label_key(label), label) for label in events.labels])
    if pairs:
        out.reading.append("Events renamed (Recording block): {}.".format(
            ", ".join("{} to {}".format(typed, new) for _, new, typed in pairs)))
    if shift_ms:
        out.reading.append("Every event moved {:g} ms {} (Recording block).".format(
            abs(float(shift_ms)), "later" if shift_ms > 0 else "earlier"))
    return out


#: Files MNE-Python reads for GIOVA.
MNE_SUFFIXES = (".edf", ".bdf", ".gdf", ".fif", ".set", ".vhdr")


def preferred_part(labels):
    """The part of a multi-condition file to read when none is named: joined imagery first.

    Motor imagery is what such files are mostly opened for; reading the movement
    trials because they happen to be listed first would mislead quietly.
    """
    joined = [label for label in labels if "+" in label]
    return next((label for label in joined if "imagery" in label), joined[0] if joined else labels[0])


#: The start of the message when GIOVA cannot tell how a MATLAB file holds its signal (the Workbench offers
#: to choose its layout on seeing it), and what to do about it.
MAT_LAYOUT = "GIOVA cannot tell how this MATLAB file holds its signal."
MAT_LAYOUT_HELP = ("Choose how to read it: which variable is the signal, its sampling rate, and where the events "
                   "are. The choice is kept for this file and, if you like, for every file laid out the same way.")


def read_mat_recording(path, condition="", rename=(), shift_ms=0.0):
    """A MATLAB file: a dataset recognised on sight (Cho2017: `condition` picks the part), else its saved or
    detected layout."""
    path = Path(path)
    variables = matfile.read_variables(path)
    preset = presets.detect(variables)
    if preset is not None:
        wanted = condition or preferred_part(preset.labels)
        mapping = preset.mapping_for(wanted)
        if mapping is None:
            raise ValueError("'{}' is not in this {} file; it has {}".format(
                wanted, preset.name, ", ".join(preset.labels)))
    else:
        try:
            mapping, origin = matfile.mapping_origin(path, variables)
        except matfile.MatLayoutUnknown as exc:
            raise ValueError("{} {} {}".format(MAT_LAYOUT, exc, MAT_LAYOUT_HELP)) from exc
    try:
        recording = matfile.apply_mapping(variables, mapping, path)
    except matfile.MatError as exc:
        raise ValueError("{} {} {}".format(MAT_LAYOUT, exc, MAT_LAYOUT_HELP)) from exc
    recording.reading.insert(0, "Read as the {} dataset, '{}'.".format(preset.name, wanted) if preset is not None
                             else "Read with {}.".format(origin))
    if preset is not None and recording.events is None:
        recording.events = presets.trial_events(variables, mapping)
    return _finish(recording, path, rename, shift_ms)


def read_mne_recording(path, rename=(), shift_ms=0.0):
    """EDF/BDF/GDF, FIF, EEGLAB or BrainVision, read with MNE-Python."""
    from ..formats import mne_io
    path = Path(path)
    recording = mne_io.load(path)
    return _finish(recording, path, rename, shift_ms)


def read_brainflow_recording(path, rename=(), shift_ms=0.0):
    """A BrainFlow CSV as GIOVA records it."""
    from ..core.sources import load_brainflow_csv
    path = Path(path)
    recording = load_brainflow_csv(path)
    recording.reading = ["Read as a BrainFlow CSV ({}).".format(getattr(recording, "unit_note", "") or "µV")]
    if recording.gap_samples:
        recording.reading.append("{} missing samples kept as gaps; no filter runs across them.".format(
            recording.gap_samples))
    return _finish(recording, path, rename, shift_ms)


def _finish(recording, path, rename, shift_ms):
    """What every reader does last: channel names and events from sidecars when there are, events renamed or
    moved as asked; each said in the recording's reading notes."""
    names = channel_names.load_names(path, recording.n_channels)
    if names:
        recording.channel_names = names
        recording.reading.append("Channel names: from the sidecar beside the file ({}).".format(
            Path(path).stem + ".channels.json"))
    recording.events, source = events_for(recording, path)
    if recording.events is not None and len(recording.events):
        named = load_label_map(path)
        recording.reading.append("Events: {} ({}), {}{}.".format(
            len(recording.events), recording.events.describe(), source,
            "; names from {}".format(Path(path).stem + ".eventmap.json") if named else ""))
    else:
        recording.reading.append("Events: none in the file.")
    return adjust_events(recording, rename, shift_ms)


# ----------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------


def unit_scale(stored="µV", gain=1.0, factor=1.0):
    """The factor a Units block multiplies the samples by."""
    gain = float(gain)
    if gain <= 0:
        raise ValueError("The amplifier gain must be above 0 (1: none)")
    if float(factor) == 0:
        raise ValueError("A factor of 0 would leave no signal")
    return UNIT_FACTORS[stored] * float(factor) / gain


def convert_units(raw, stored="µV", gain=1.0, factor=1.0):
    """Samples to µV when the file holds other units or amplified values."""
    scale = unit_scale(stored, gain, factor)
    if scale == 1.0:
        return raw
    out = derive(raw, raw.data * scale)
    if out.rail_uv is not None:
        out.rail_uv = out.rail_uv * abs(scale)
    out.unit_note = "{}; then × {:g} (Units / gain)".format(raw.unit_note or "As read", scale)
    return out


def pick_channels(raw, channels=()):
    """The channels named, in this order; none named: all."""
    wanted = names_of(channels)
    if not wanted:
        return raw
    missing = [n for n in wanted if n not in raw.channel_names]
    if missing:
        raise ValueError("Not in this recording: {}. It has {}".format(
            ", ".join(missing), ", ".join(raw.channel_names[:12]) + (", …" if raw.n_channels > 12 else "")))
    index = [raw.channel_names.index(n) for n in wanted]
    out = derive(raw, raw.data[index], names=wanted)
    if raw.channel_log:
        # the amplifier's on/off states are by position: they follow their channels
        out.channel_log = [(when, [on[i] if i < len(on) else True for i in index]) for when, on in raw.channel_log]
    return out


def pick_trial_channels(epochs, channels=()):
    """The trials' channels named, in this order; none named: all. A channel not measured during the trials
    (switched off or flat, `Epochs.unmeasured`) is left out and stays named there, as the Analysis page left it
    out of a region; when every channel named was off, nothing is guessed in their place."""
    wanted = names_of(channels)
    if not wanted:
        return epochs
    missing = [n for n in wanted if n not in epochs.names]
    if missing:
        raise ValueError("Not in these trials: {}. They have {}".format(
            ", ".join(missing), ", ".join(epochs.names[:12]) + (", …" if len(epochs.names) > 12 else "")))
    measured = [n for n in wanted if n not in epochs.unmeasured]
    if not measured:
        raise ValueError("Every channel picked was switched off or flat during these trials ({}). Choose "
                         "channels that were on".format(", ".join(wanted)))
    wanted = measured
    index = [epochs.names.index(n) for n in wanted]
    return epochs.with_data(epochs.data[:, index], names=wanted)


def repair_channels(raw, channels=(), action="interpolate"):
    """Bad channels rebuilt from the others (MNE-Python's spherical splines on standard 10-05 positions), or
    taken out with `action` "drop". Missing samples stay missing where any channel used had them."""
    bad = names_of(channels)
    if not bad:
        return raw
    names = list(raw.channel_names)
    missing = [n for n in bad if n not in names]
    if missing:
        raise ValueError("Not in this recording: {}. It has {}".format(", ".join(missing), ", ".join(names[:12])))
    if action == "drop":
        kept = [n for n in names if n not in bad]
        if not kept:
            raise ValueError("Dropping {} leaves no channel".format(", ".join(bad)))
        return pick_channels(raw, kept)
    if action != "interpolate":
        raise ValueError("Bad channels are interpolated or dropped")
    if len(names) - len(bad) < 3:
        raise ValueError("Interpolating needs at least three good channels around the bad ones")
    import mne
    montage = mne.channels.make_standard_montage("standard_1005")
    known = {n.casefold() for n in montage.ch_names}
    unplaced = [n for n in names if n.casefold() not in known]
    if unplaced:
        raise ValueError("Interpolating needs every channel's position, and {} {} not a 10-20/10-05 name; "
                         "drop the bad channels instead".format(", ".join(unplaced[:6]),
                                                                "is" if len(unplaced) == 1 else "are"))
    info = mne.create_info(names, raw.fs, "eeg")
    data = mne.io.RawArray(raw.data * 1e-6, info, verbose="error")
    data.set_montage(montage, match_case=False, verbose="error")
    data.info["bads"] = bad
    data.interpolate_bads(reset_bads=True, mode="accurate", verbose="error")
    # only the bad rows are replaced: the good channels stay exactly as they were
    rows = [names.index(n) for n in bad]
    fixed = raw.data.copy()
    fixed[rows] = data.get_data(picks=rows) * 1e6
    return derive(raw, fixed)


def rereference(raw, mode="average", channels=()):
    """Every channel minus the common average, or minus the mean of the channels named, or (`laplacian`)
    minus the mean of its four nearest electrodes."""
    if mode == "laplacian":
        return derive(raw, laplacian(raw.data, raw.channel_names))
    settings = Settings(filters={"reference": mode, "reference_channels": names_of(channels)})
    common = reference_signal(raw.data, settings, raw.channel_names)
    return derive(raw, raw.data - common)


#: how many nearest electrodes a channel's Laplacian subtracts (Hjorth's four on a 10-20 grid)
LAPLACIAN_NEIGHBOURS = 4


def laplacian_neighbours(names, count=LAPLACIAN_NEIGHBOURS):
    """For each channel, the rows of its `count` nearest other channels, on standard 10-05 positions."""
    import mne
    names = list(names)
    if len(names) < count + 1:
        raise ValueError("A Laplacian needs at least {} channels; this recording has {}".format(
            count + 1, len(names)))
    montage = mne.channels.make_standard_montage("standard_1005")
    places = {n.casefold(): p for n, p in montage.get_positions()["ch_pos"].items()}
    unplaced = [n for n in names if n.casefold() not in places]
    if unplaced:
        raise ValueError("A Laplacian needs every channel's position, and {} {} not a 10-20/10-05 name".format(
            ", ".join(unplaced[:6]), "is" if len(unplaced) == 1 else "are"))
    xyz = np.array([places[n.casefold()] for n in names])
    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    return [list(np.argsort(row, kind="stable")[:count]) for row in distance]


def laplacian(data, names, count=LAPLACIAN_NEIGHBOURS):
    """Each channel minus the mean of its `count` nearest electrodes (a nearest-neighbour, or small, Laplacian).
    Missing samples among the neighbours are left out of their mean, time point by time point."""
    data = np.asarray(data, dtype=float)
    out = np.empty_like(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, rows in enumerate(laplacian_neighbours(names, count)):
            out[i] = data[i] - np.nanmean(data[rows], axis=0)
    return out


def filter_recording(raw, low=1.0, high=40.0, notch=50.0, design="butterworth", kind=fl.PASS, phase="zero",
                     order=4, ripple=1.0, attenuation=40.0, transition=0.0, notch_q=30.0, notches=()):
    """Each channel through the filters `filters.steps` designs, every run of finite samples on its own.
    `notches`: notch exactly these frequencies (no harmonics) instead of `notch` and its harmonics."""
    settings = dict(low=low, high=high, notch=notch, design=design, kind=kind, phase=phase, order=order,
                    ripple=ripple, attenuation=attenuation, transition=transition, notch_q=notch_q,
                    notches=list(notches))
    chain = fl.steps(raw.fs, settings)
    data = np.vstack([fl.apply(row, raw.fs, chain, phase) for row in raw.data])
    return derive(raw, data)


def downsample_recording(raw, target=250.0):
    """Divided by a whole factor to at least `target` Hz, after a zero-phase anti-alias low-pass."""
    return downsample(raw, target)


def run_ica(raw, eog_channels=("Fp1", "Fp2"), exclude=(), components=0.99, method="fastica", threshold=3.0,
            fit_highpass=1.0, seed=42, progress=None):
    """(cleaned recording, IcaResult): MNE-Python's ICA, components that follow the EOG channels removed."""
    import mne
    names = list(raw.channel_names)
    finite = np.isfinite(raw.data).all(axis=0)
    if finite.sum() < 10 * raw.fs:
        raise ValueError("ICA needs at least 10 s of samples without gaps")
    try:
        manual = sorted({int(v) for v in names_of(exclude)})
    except ValueError:
        raise ValueError("'Also remove' takes component numbers, e.g. 0, 3") from None
    components = int(components) if components >= 1 else float(components)
    if not 0 < components <= len(names):
        raise ValueError("Components must be a share below 1 or a count up to the channels ({})".format(len(names)))
    info = mne.create_info(names, raw.fs, "eeg")
    fit = mne.io.RawArray(raw.data[:, finite] * 1e-6, info, verbose="error")
    if fit_highpass:
        _say(progress, "High-pass copy for ICA")
        fit.filter(fit_highpass, None, verbose="error")
    _say(progress, "Fitting ICA ({})".format(method))
    ica = mne.preprocessing.ICA(n_components=components, method=method, random_state=int(seed), max_iter="auto",
                                verbose="error")
    ica.fit(fit, verbose="error")
    eog = [n for n in names_of(eog_channels) if n in names]
    found, scores = [], np.zeros(0)
    if eog:
        found, raw_scores = ica.find_bads_eog(fit, ch_name=eog, threshold=threshold, verbose="error")
        scores = np.max(np.abs(np.atleast_2d(raw_scores)), axis=0)
    bad = [c for c in manual if c >= ica.n_components_]
    if bad:
        raise ValueError("There is no component {} (0 to {})".format(bad[0], ica.n_components_ - 1))
    ica.exclude = sorted(set(found) | set(manual))
    cleaned = raw.data.copy()
    if ica.exclude:
        whole = mne.io.RawArray(raw.data[:, finite] * 1e-6, info, verbose="error")
        cleaned[:, finite] = ica.apply(whole, verbose="error").get_data() * 1e6
    result = IcaResult(names, ica.get_components(), list(ica.exclude), scores, eog, method,
                       [int(c) for c in found], manual)
    return derive(raw, cleaned), result


# ----------------------------------------------------------------------
# trials
# ----------------------------------------------------------------------


#: Where an Epochs block's trials come from: the recording's events (or every so many seconds without them),
#: a table of trials typed or dragged by hand, or a labelled sequence of phases repeated from a start.
EVENTS, MANUAL, SEQUENCE = "events", "manual", "sequence"
TRIAL_SOURCES = (EVENTS, MANUAL, SEQUENCE)


def _trial_reason(raw, segment, start, n, reject_uv, motion_limit):
    """Why a cut trial is left out by a span the file marks as bad, amplitude or motion, or None."""
    marked = raw.marked_bad(start / raw.fs, (start + n) / raw.fs)
    if marked:
        return "Marked {} in the file".format(marked)
    if reject_uv > 0:
        worst = np.max(np.abs(segment - segment.mean(axis=1, keepdims=True)), axis=1)
        if worst.max() > reject_uv:
            return "Amplitude > {:g} µV ({})".format(reject_uv, raw.channel_names[int(worst.argmax())])
    if motion_limit > 0:
        motion = raw.motion[:, start:start + n]
        swing = np.nanmax(motion, axis=1) - np.nanmin(motion, axis=1)
        if np.nanmax(swing) > motion_limit:
            axis = int(np.nanargmax(swing))
            return "Motion > {:g} {} ({})".format(motion_limit, raw.motion_units[axis], raw.motion_names[axis])
    return None


def defined_trials(raw, source=MANUAL, event_length=False, trials=(), sequence="", sequence_start=0.0, repeats=1):
    """The trials as defined (core.trials.Trial), each with its own start and end in the recording.

    manual: rows [id, condition, start s, end s, included] as typed (included may be left out: yes);
    sequence: "Condition:seconds; ..." phases repeated `repeats` times from `sequence_start` s, an
    incomplete last phase kept and later refused as too short (`core.trials.periodic_trials`);
    events with `event_length`: each event of non-zero length from its onset to its end, as the Analysis
    page made them (events of zero length make no trial).
    """
    if source == MANUAL:
        rows = list(trials or ())
        if not rows:
            raise ValueError("The trial table is empty: add trials, or take them from the events or a sequence")
        out = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) not in (4, 5):
                raise ValueError("Each trial is [id, condition, start, end, included] ({!r} is not)".format(row))
            included = bool(row[4]) if len(row) == 5 else True
            try:
                out.append(Trial(str(row[0]).strip(), str(row[1]).strip(), float(row[2]), float(row[3]), included))
            except (TypeError, ValueError):
                raise ValueError("Trial {}: start and end must be numbers of seconds".format(row[0])) from None
    elif source == SEQUENCE:
        out = periodic_trials(float(sequence_start), parse_sequence(sequence), repeats, raw.duration)
        if not out:
            raise ValueError("The sequence starts after the recording ends")
    elif source == EVENTS and event_length:
        events = raw.events
        if events is None or not len(events):
            raise ValueError("This recording has no events to take trials from")
        out = [Trial(str(i + 1), label, float(onset), float(onset + length))
               for i, (onset, length, label) in enumerate(zip(events.onsets, events.durations, events.labels))
               if length > 0]
        if not out:
            raise ValueError("No event of this recording has a length: cut a window around each event instead")
    else:
        raise ValueError("Trials come from events, a table ({}) or a sequence ({})".format(MANUAL, SEQUENCE))
    ids = [t.id for t in out]
    if len(ids) != len(set(ids)):
        raise ValueError("Trial IDs must be unique")
    return out


def unmeasured_channels(raw, spans):
    """Names of the channels not measured during the trials used, [(first sample, count)]: switched off at any
    moment (the amplifier's own log, `Recording.switched_off`) or flat (`core.monitoring.flat_channels`), as
    the Analysis page found them before it measured a region."""
    from ..core.monitoring import flat_channels
    off = set()
    for start, count in spans:
        off |= raw.switched_off(start / raw.fs, (start + count) / raw.fs)
    if spans:
        off |= flat_channels(np.concatenate([raw.data[:, a:a + n] for a, n in spans], axis=1))
    return [raw.channel_names[i] for i in sorted(off) if i < raw.n_channels]


def review_flags(raw, segments, quality_mad=6.0):
    """[flags] per trial, as `core.trials.analyze` marks trials for review - never to leave one out: a channel
    whose signal does not move (flat), and, with five trials or more, a peak-to-peak larger than the median plus
    `quality_mad` median absolute deviations. Judged on the samples the trials have (after any filter), so an
    amplifier's saturation, which shows in the raw level only, is not judged here."""
    flags = [["Flat signal ({})".format(raw.channel_names[c]) for c in np.flatnonzero(np.std(s, axis=1) < 1e-6)]
             for s in segments]
    if len(segments) >= 5:
        peaks = np.array([np.ptp(s, axis=1).max() for s in segments])
        median = np.median(peaks)
        threshold = median + quality_mad * np.median(np.abs(peaks - median))
        for k in np.flatnonzero(peaks > threshold):
            flags[k].append("High amplitude (median + {:g} MAD)".format(quality_mad))
    return flags


def cut_defined(raw, trials, conditions=(), tmin=0.0, tmax=4.0, reject_uv=0.0, motion_limit=0.0, quality_mad=6.0):
    """Trials as defined (start, end), each `tmax - tmin` s from its start: the Analysis page's common duration.

    The first sample of a trial is at `tmin` s from its time zero (an event `-tmin` s after the start, or the
    start itself when `tmin` is 0). Each trial is judged by `core.trials.trial_status`, as the Analysis page
    judges it: no padding and no quiet clipping, a trial shorter than the common duration is left out with
    its reason, and so is one left out of the selection ("Excluded by selection"). Then, on the samples used,
    missing samples, amplitude and motion as `cut_epochs`. Trials used may carry review marks in their
    `meta["flags"]` (`review_flags`); a mark never leaves a trial out.
    """
    if tmax <= tmin:
        raise ValueError("'To' must be after 'From'")
    fs, duration = raw.fs, tmax - tmin
    labels_there = list(dict.fromkeys(t.condition for t in trials))
    wanted = names_of(conditions) or labels_there
    absent = [c for c in wanted if c not in labels_there]
    if absent:
        raise ValueError("No trials of {}; the conditions are {}".format(", ".join(absent), ", ".join(labels_there)))
    first = sample_index(tmin, fs) / fs
    kept, labels, ids, rejected, meta = [], [], [], [], []
    for trial in trials:
        if trial.condition not in wanted:
            continue
        start, count, reason = trial_status(trial, fs, raw.n_samples, duration)
        if reason == "Ready":
            segment = raw.data[:, start:start + count]
            reason = ("Missing samples or finite run too short for filtering" if not np.isfinite(segment).all()
                      else _trial_reason(raw, segment, start, count, reject_uv, motion_limit))
        if reason and reason != "Ready":
            rejected.append((trial.id, trial.condition, reason))
            continue
        kept.append(segment)
        labels.append(trial.condition)
        ids.append(trial.id)
        meta.append(dict(id=trial.id, condition=trial.condition, start=trial.start, end=trial.end,
                         included=trial.included, sample=start, count=count, zero=start / fs - first))
    if not kept:
        raise ValueError("No usable trials: " + "; ".join(sorted({r for _, _, r in rejected})))
    for entry, flags in zip(meta, review_flags(raw, kept, quality_mad)):
        entry["flags"] = flags
    return Epochs(np.stack(kept), fs, list(raw.channel_names), labels, first, ids, rejected, meta,
                  unmeasured_channels(raw, [(m["sample"], m["count"]) for m in meta]))


def cut_epochs(raw, conditions=(), tmin=0.0, tmax=4.0, reject_uv=0.0, every=0.0, first=0.0, motion_limit=0.0,
               source=EVENTS, event_length=False, trials=(), sequence="", sequence_start=0.0, repeats=1,
               quality_mad=6.0):
    """Trials from `tmin` to `tmax` s around each event of the conditions named (none named: every label),
    or every `every` s from `first` when the trials repeat without markers.

    With `source` manual or sequence, or events with `event_length`, the trials are defined first
    (`defined_trials`) and cut by `cut_defined`: `tmax - tmin` is then the common duration used from each
    trial's start, and `tmin` the time of that start from the trial's time zero.

    reject_uv: a trial straying further than this from its mean on any channel is left out. motion_limit:
    one whose accelerometer (a LiveAmp's, a Unicorn's) swings more than this, peak to peak on any axis,
    in the sensor's own unit (mg), is left out too: the head moved.
    """
    motion = getattr(raw, "motion", None)
    if motion_limit > 0 and motion is None:
        raise ValueError("This recording has no motion channels to judge movement by; set the motion limit to 0")
    if source not in TRIAL_SOURCES:
        raise ValueError("Trials come from {}".format(", ".join(TRIAL_SOURCES)))
    if source != EVENTS or event_length:
        defined = defined_trials(raw, source, event_length, trials, sequence, sequence_start, repeats)
        return cut_defined(raw, defined, conditions, tmin, tmax, reject_uv, motion_limit, quality_mad)
    if every > 0:
        # a trial may start wherever the window cut from it (to 'To') still ends inside the recording;
        # how often trials repeat says nothing about how long each one is
        last = raw.duration - max(tmax, 0.0) + 0.5 / raw.fs
        starts = np.arange(first, last, every)
        if starts.size > 10000:
            raise ValueError("This would make more than 10,000 trials: a longer period, please")
        events = EventTable(starts, [every] * starts.size, ["Trial"] * starts.size)
    else:
        events = raw.events
    if events is None or not len(events):
        raise ValueError("This recording has no events to cut trials around. For trials repeated without "
                         "markers, set 'Without markers: every' in Options")
    if tmax <= tmin:
        raise ValueError("'To' must be after 'From'")
    wanted = names_of(conditions) or events.unique_labels()
    absent = [c for c in wanted if c not in events.unique_labels()]
    if absent:
        raise ValueError("No events labelled {}; the labels are {}".format(
            ", ".join(absent), ", ".join(events.unique_labels())))
    fs = raw.fs
    n = sample_index(tmax - tmin, fs)
    offset = sample_index(tmin, fs)
    kept, labels, ids, rejected, meta = [], [], [], [], []
    for i, (onset, label) in enumerate(zip(events.onsets, events.labels)):
        if label not in wanted:
            continue
        start = sample_index(onset, fs) + offset
        if start < 0 or start + n > raw.n_samples:
            reason = "Outside recording"
        else:
            segment = raw.data[:, start:start + n]
            reason = ("Missing samples" if not np.isfinite(segment).all()
                      else _trial_reason(raw, segment, start, n, reject_uv, motion_limit))
        if reason:
            rejected.append((str(i + 1), label, reason))
        else:
            kept.append(segment)
            labels.append(label)
            ids.append(str(i + 1))
            meta.append(dict(id=str(i + 1), condition=label, start=start / fs, end=(start + n) / fs, included=True,
                             sample=start, count=n, zero=float(onset)))
    if not kept:
        raise ValueError("No usable trials: " + "; ".join(sorted({r for _, _, r in rejected})))
    return Epochs(np.stack(kept), fs, list(raw.channel_names), labels, sample_index(tmin, fs) / fs, ids, rejected,
                  meta, unmeasured_channels(raw, [(m["sample"], m["count"]) for m in meta]))


def time_window(epochs, start=0.5, end=2.5):
    """The part of each trial from `start` to `end` s."""
    times = epochs.times
    last = times[-1] + 1 / epochs.fs
    if not times[0] - 1e-9 <= start < end <= last + 1e-9:
        raise ValueError("The window must lie inside the trials ({:g} to {:g} s)".format(times[0], last))
    a = sample_index(start - epochs.tmin, epochs.fs)
    b = sample_index(end - epochs.tmin, epochs.fs)
    return epochs.with_data(epochs.data[:, :, a:b], tmin=epochs.tmin + a / epochs.fs)


def filter_trials(epochs, low=1.0, high=40.0, notch=50.0, design="butterworth", kind=fl.PASS, phase="zero",
                  order=4, ripple=1.0, attenuation=40.0, transition=0.0, notch_q=30.0, notches=()):
    """`filter_recording`'s filters applied to each trial and channel on its own.

    No trial can ring into the next through the filter, and a join between
    trials stored end to end (Cho2017) is never filtered across.
    """
    settings = dict(low=low, high=high, notch=notch, design=design, kind=kind, phase=phase, order=order,
                    ripple=ripple, attenuation=attenuation, transition=transition, notch_q=notch_q,
                    notches=list(notches))
    chain = fl.steps(epochs.fs, settings)
    rows = epochs.data.reshape(-1, epochs.data.shape[-1])
    data = np.vstack([fl.apply(row, epochs.fs, chain, phase) for row in rows])
    return epochs.with_data(data.reshape(epochs.data.shape))


def _numeric_id(text):
    """True for an id written as a plain whole number ("12", not "012" or "T1")."""
    return text.isdigit() and str(int(text)) == text


def trial_numbers(value, known=()):
    """Trial numbers from "1-3, 5, 7" (or a list of such parts), in order, as the ids trials carry.

    A part that is itself one of the `known` ids (a trial typed by hand as "T1" or "rest-2") is taken as it is.
    """
    numbers = []
    for part in names_of(value):
        if part in known:
            numbers += [part] if part not in numbers else []
            continue
        low, dash, high = part.partition("-")
        try:
            first, last = int(low), int(high if dash else low)
        except ValueError:
            raise ValueError("Write trial numbers and ranges, e.g. 1-3, 5, 7 ('{}' is neither)".format(part)) from None
        if first > last:
            raise ValueError("In {} the first trial must come before the last".format(part))
        numbers += [str(n) for n in range(first, last + 1) if str(n) not in numbers]
    return numbers


def trial_text(numbers):
    """Trial numbers written short, runs as ranges: ["1", "2", "3", "5"] -> ["1-3", "5"]; ids that are not plain
    numbers follow as they are, in their order."""
    values = sorted({int(n) for n in numbers if _numeric_id(str(n))})
    parts, start = [], None
    for k, n in enumerate(values):
        start = n if start is None else start
        if k + 1 == len(values) or values[k + 1] != n + 1:
            parts.append(str(n) if n == start else "{}-{}".format(start, n))
            start = None
    return parts + list(dict.fromkeys(str(n) for n in numbers if not _numeric_id(str(n))))


def select_trials(epochs, keep=(), leave_out=()):
    """The trials named by number or id (none named: all), less those left out."""
    keep, leave = trial_numbers(keep, epochs.ids), trial_numbers(leave_out, epochs.ids)
    unknown = [n for n in keep + leave if n not in epochs.ids]
    if unknown:
        raise ValueError("No trial {} among these: they are numbered {}".format(
            ", ".join(unknown[:6]), ", ".join(trial_text(epochs.ids))))
    chosen = [k for k, i in enumerate(epochs.ids) if (not keep or i in keep) and i not in leave]
    if not chosen:
        raise ValueError("Every trial is left out")
    dropped = [(i, label, LEFT_OUT) for k, (i, label) in enumerate(zip(epochs.ids, epochs.labels)) if k not in chosen]
    return epochs.with_data(epochs.data[chosen], labels=[epochs.labels[k] for k in chosen],
                            ids=[epochs.ids[k] for k in chosen], rejected=list(epochs.rejected) + dropped,
                            meta=[epochs.meta[k] for k in chosen] if epochs.meta else [])


def wavelet_denoise(epochs, wavelet="db4", level=4, mode="hard", threshold=0.0):
    """Each trial and channel decomposed, its detail levels thresholded, and rebuilt (core.wavelets)."""
    from ..core.wavelets import denoise
    if level < 0 or threshold < 0:
        raise ValueError("Levels and threshold cannot be negative")
    rows = epochs.data.reshape(-1, epochs.data.shape[-1])
    data = np.vstack([denoise(row, wavelet, level, mode, threshold)[0] for row in rows])
    return epochs.with_data(data.reshape(epochs.data.shape))


def dwt_bands(epochs, wavelet="db4", level=6, envelope=True, progress=None):
    """(band envelopes, energy share): each trial split into the DWT's dyadic bands."""
    from scipy.signal import hilbert
    from ..core.wavelets import bands
    if level < 0:
        raise ValueError("Levels cannot be negative")
    signals, energies, names = [], [], []
    for k, trial in enumerate(epochs.data):
        if k % 20 == 0:
            _say(progress, "DWT {}/{}".format(k + 1, len(epochs.data)))
        row, energy = [], []
        for c, x in enumerate(trial):
            split = bands(x, epochs.fs, wavelet, level)
            if k == 0:
                names += ["{} · {} {:g}–{:g} Hz".format(epochs.names[c], name, low, high)
                          for name, low, high, _ in split]
            row += [np.abs(hilbert(band[3])) if envelope else band[3] for band in split]
            total = sum(float(np.sum(band[3] ** 2)) for band in split) or 1.0
            energy += [100.0 * float(np.sum(band[3] ** 2)) / total for band in split]
        signals.append(row)
        energies.append(energy)
    units = ["% of the channel's energy"] * len(names)
    return (epochs.with_data(np.asarray(signals), names=names),
            Features(np.asarray(energies), [n + " energy" for n in names], list(epochs.labels), list(epochs.ids),
                     units))


def filter_bank(epochs, bands=FBCSP_BANDS, start=0.5, end=2.5, design="chebyshev2", order=4, attenuation=40.0,
                ripple=1.0):
    """The trials band-passed in each band, kept from `start` to `end` s; the channels come band after band."""
    from scipy.signal import sosfilt
    if not np.isfinite(epochs.data).all():
        raise ValueError("The filter bank needs trials without missing samples")
    times = epochs.times
    if not times[0] <= start < end <= times[-1] + 1.5 / epochs.fs:
        raise ValueError("Keep from/to must lie inside the trials ({:g} to {:g} s)".format(
            times[0], times[-1] + 1 / epochs.fs))
    a = sample_index(start - epochs.tmin, epochs.fs)
    b = sample_index(end - epochs.tmin, epochs.fs)
    x = epochs.data - epochs.data.mean(axis=2, keepdims=True)
    parts, names = [], []
    for low, high in bands:
        _, sos = fl.design(epochs.fs, float(low), float(high), fl.PASS, design, order, ripple, attenuation)
        kept = sosfilt(sos, x, axis=-1)[:, :, a:b]
        # a 40 dB stop band passes 1% of each trial's leftover offset: a constant, taken out
        parts.append(kept - kept.mean(axis=2, keepdims=True))
        names += ["{} · {:g}–{:g} Hz".format(n, low, high) for n in epochs.names]
    return epochs.with_data(np.concatenate(parts, axis=1), names=names, tmin=epochs.tmin + a / epochs.fs)


def subtract_baseline(epochs, start=-0.5, end=0.0):
    """Each trial's mean from `start` to `end` s subtracted, per channel."""
    times = epochs.times
    last = times[-1] + 1 / epochs.fs
    rows = (times >= start - 1e-9) & (times < end - 1e-9)
    # the whole window inside the trials: a part outside would be dropped quietly, and the mean of
    # what is left (perhaps after the cue) taken off while the methods name the window as asked
    if not (times[0] - 1e-9 <= start < end <= last + 1e-9) or not rows.any():
        raise ValueError("The baseline must lie inside the trials ({:g} to {:g} s)".format(times[0], last))
    offset = epochs.data[:, :, rows].mean(axis=2, keepdims=True)
    return epochs.with_data(epochs.data - offset)


# ----------------------------------------------------------------------
# spectral and features
# ----------------------------------------------------------------------


def _condition_waveforms(epochs):
    """{condition: (n_channels, n_samples)}: each condition's trials averaged sample by sample, channel by channel."""
    labels = np.asarray(epochs.labels)
    return {c: epochs.data[labels == c].mean(axis=0) for c in dict.fromkeys(epochs.labels)}


def _welch(x, fs, window, overlap, detrend):
    from scipy.signal import detrend as remove
    if detrend == "linear":
        if not np.isfinite(x).all():
            raise ValueError("PSD needs finite samples")
        x = remove(x, axis=-1, type="linear")
    return welch_psd(x, fs, window, overlap)


def welch_spectrum(epochs, window=1.0, overlap=0.5, detrend="constant", signal_mean=False):
    """Welch power spectral density per trial and channel (Hann segments of `window` s).

    detrend: "constant" removes each trial's mean; "linear" its straight-line drift over the whole trial
    first, as the Analysis page's linear detrend (core.trials.measure); each segment loses its mean either way.
    signal_mean: also the density of each condition's mean waveform (`Spectrum.signal_means`); oscillations
    not phase-locked across trials cancel in it, unlike in the mean of the trials' spectra.
    """
    if detrend not in ("constant", "linear"):
        raise ValueError("Remove the mean (constant) or the linear trend (linear)")
    freqs, power = _welch(epochs.data, epochs.fs, window, overlap, detrend)
    means = ({c: _welch(x, epochs.fs, window, overlap, detrend)[1] for c, x in _condition_waveforms(epochs).items()}
             if signal_mean else {})
    return Spectrum(freqs, power, epochs.names, epochs.labels, epochs.ids, signal_means=means)


def _amplitude(x, detrend):
    from scipy.signal import detrend as remove
    x = remove(x, axis=-1, type=detrend)
    n = x.shape[-1]
    # as the Analysis page's FFT (core.trials.measure): one-sided, a sine of amplitude A reads A
    amplitude = np.abs(np.fft.rfft(x, axis=-1)) / n
    amplitude[..., 1:-1 if n % 2 == 0 else None] *= 2
    return amplitude


def fft_spectrum(epochs, detrend="constant", signal_mean=False):
    """FFT amplitude of each whole trial and channel, one-sided: a sine of amplitude A reads A.
    signal_mean: also the amplitude of each condition's mean waveform (`Spectrum.signal_means`)."""
    if not np.isfinite(epochs.data).all():
        raise ValueError("FFT needs trials without missing samples")
    n = epochs.data.shape[-1]
    means = {c: _amplitude(x, detrend) for c, x in _condition_waveforms(epochs).items()} if signal_mean else {}
    return Spectrum(np.fft.rfftfreq(n, 1 / epochs.fs), _amplitude(epochs.data, detrend), epochs.names,
                    epochs.labels, epochs.ids, kind=AMPLITUDE, signal_means=means)


def band_power(spectrum, bands=MU_BETA, log=True):
    """Power in each [name, low, high] band per channel, integrated from a Welch PSD; log10 when asked."""
    if not spectrum.is_power:
        raise ValueError("Band power integrates a power density: connect a Welch PSD, not the FFT amplitude")
    columns, names = [], []
    for name, low, high in bands:
        columns.append(band_integral(spectrum.freqs, spectrum.power, low, high))     # (n_trials, n_channels)
        names += ["{} {}".format(channel, name) for channel in spectrum.names]
    values = np.concatenate(columns, axis=1)
    if log:
        if not (values > 0).all():
            raise ValueError("A band has zero power, so its logarithm is undefined")
        values = np.log10(values)
    unit = "log10 µV²" if log else "µV²"
    return Features(values, names, spectrum.labels, spectrum.ids, [unit] * len(names))


def signal_features(epochs, selected=("rms", "mobility", "complexity", "entropy"), window=1.0, overlap=0.5,
                    detrend="constant", bands=FEATURE_BANDS, ratios=FEATURE_RATIOS, total_range=(1.0, 45.0), roi=False,
                    progress=None):
    """Per trial and channel, the features of GIOVA's feature catalogue named in `selected`.

    bands, ratios, total_range: the feature profile (`core.feature_profiles.FeatureProfile`) the band, relative,
    peak and ratio features are defined by. The PSD they read is a Welch density of `window` s Hann segments,
    `overlap`; with detrend "linear" it is taken after removing each trial's linear trend, as the Analysis page
    does. roi: each feature averaged over the channels, every channel measured on its own first (a value missing
    on any channel is missing), as the Analysis page combines a region. Why a value is missing is kept in
    `Features.reasons`.
    """
    profile = FeatureProfile(name="Workbench", bands=[list(b) for b in bands], ratios=[list(r) for r in ratios],
                             selected=list(selected), total_range=list(total_range))
    profile.validate()
    if detrend not in ("constant", "linear"):
        raise ValueError("Remove the mean (constant) or the linear trend (linear)")
    specs = profile.visible_specs()
    rows, reasons = [], []
    for k, trial in enumerate(epochs.data):
        if k % 20 == 0:
            _say(progress, "Features {}/{}".format(k + 1, len(epochs.data)))
        measured = []
        for signal in trial:
            psd = _welch(signal, epochs.fs, window, overlap, "linear") if detrend == "linear" else None
            measured.append(extract(signal, epochs.fs, profile, window=window, overlap=overlap, psd=psd))
        if roi:
            # a channel not measured during the trials (switched off or flat) does not count in the region
            on = [m for m, name in zip(measured, epochs.names) if name not in epochs.unmeasured] or measured
            measured = [mean_values(on)]
        row, why = [], {}
        for c, values in enumerate(measured):
            for s in specs:
                row.append(values.values[s.id])
                if s.id in values.reasons and not np.isfinite(values.values[s.id]):
                    why[len(row) - 1] = values.reasons[s.id]
        rows.append(row)
        reasons.append(why)
    region = [n for n in epochs.names if n not in epochs.unmeasured] or epochs.names
    channels = ["ROI " + "+".join(region)] if roi else epochs.names
    names = ["{} {}".format(c, s.name) for c in channels for s in specs]
    units = [s.unit for _ in channels for s in specs]
    return Features(np.asarray(rows, dtype=float), names, epochs.labels, epochs.ids, units,
                    reasons=[{names[i]: text for i, text in why.items()} for why in reasons])


def time_frequency(source, channels=("C3", "Cz", "C4"), conditions=(), tmin=-1.0, tmax=4.0, baseline=(-1.0, 0.0),
                   active=(0.5, 2.5), method=STFT, fmin=4.0, fmax=40.0, window=0.5, overlap=0.9, smooth=0.5,
                   bands=MU_BETA, topography=False, baseline_mode=WINDOW, reference="", erd_mode=POOLED,
                   n_cycles=0.5, wavelet=CWT_WAVELET, reject_uv=0.0, width=PER_HZ, scalp=SCALP_STFT,
                   progress=None):
    """Time-frequency power and ERD/ERS of a recording (cut into trials from `tmin` to `tmax` s around each
    event) or of trials as they come (`tmin` / `tmax` then unused).

    method: "stft" or "wavelet" (complex Morlet wavelets), its `width` either `n_cycles` cycles per Hz of each
    frequency (clipped to 2..12) or one complex Morlet `wavelet` (cmorB-C) for every frequency; "morlet" and
    "cwt", the older names of those two, are read as well. erd_mode: "pooled" (mean power against mean
    baseline) or "mean_of_trials" (each trial's % averaged), for every map, band course and trace; reject_uv: a
    trial straying further than this from its mean on an analysed channel is left out (0: off); scalp: what the
    head map and the statistics are window means of, a separate STFT ("stft") or the maps' own transform
    ("maps"). All as `core.epochs.EpochSettings`."""
    trials = isinstance(source, Epochs)
    available = list(source.names if trials else source.channel_names)
    names = names_of(channels)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError("Not in the {}: {}. {} {}".format(
            "trials" if trials else "recording", ", ".join(missing), "They have" if trials else "It has",
            ", ".join(available[:12]) + (", …" if len(available) > 12 else "")))
    if baseline_mode == CONDITION and not str(reference).strip():
        # none named: the first condition of the trials, as the Analysis page chose it (the result's settings say
        # which one was used)
        labels = list(source.labels) if trials else list(source.events.unique_labels()) if source.events else []
        wanted = names_of(conditions)
        reference = next((c for c in dict.fromkeys(labels) if not wanted or c in wanted), "")
    settings = EpochSettings(
        conditions=names_of(conditions), tmin=tmin, tmax=tmax, baseline=list(baseline), active=list(active),
        method=tf_method(method, width), fmin=fmin, fmax=fmax, window=window, overlap=overlap, smooth=smooth,
        bands=[list(b) for b in bands], topography=bool(topography), filters={}, baseline_mode=baseline_mode,
        reference=reference, erd_mode=erd_mode, n_cycles=float(n_cycles), wavelet=wavelet,
        reject_uv=float(reject_uv), scalp=scalp)
    if trials:
        off = list(source.unmeasured)
        if names and not [n for n in names if n not in off]:
            raise ValueError("Every channel chosen was switched off or flat during these trials ({}). Choose "
                             "channels that were on".format(", ".join(names)))
        index = ([available.index(n) for n in names if n not in off] or
                 [k for k, n in enumerate(available) if n not in off] or list(range(len(available))))
        result = compute_trials(source.data, source.fs, available, source.labels, source.tmin, index, settings,
                                ids=source.ids, rejected=source.rejected, progress=progress)
        result.unmeasured = off                 # no colour for them on a head map, as on the Analysis page
        return result
    index = [available.index(n) for n in names] or [0]
    return compute_epochs(source.data, source.fs, available, index, source.events, settings, progress=progress)


def compare_windows(epochs, window_a=(0.0, 2.0), window_b=(4.0, 10.0), name_a="A", name_b="B",
                    bands=THETA_ALPHA_BETA, segment=1.0, test="wilcoxon"):
    """Band power in two windows of every trial, channel by channel, and a paired test across the trials.

    Each window's power: a Welch density of `segment` s Hann pieces (the same length in both windows, so
    both have the same frequency resolution), integrated over each band. The test compares the log10 power
    of A and B trial by trial: Wilcoxon's signed-rank test, or a paired t-test; its p values are adjusted
    by Holm's method over every channel and band, as many tests are made at once.
    """
    from scipy.stats import ttest_rel, wilcoxon
    from .batch import holm
    if test not in ("wilcoxon", "t-test"):
        raise ValueError("The test is wilcoxon or t-test")
    times = epochs.times
    last = times[-1] + 1 / epochs.fs
    spans = []
    for name, (start, end) in ((name_a, window_a), (name_b, window_b)):
        if not times[0] - 1e-9 <= start < end <= last + 1e-9:
            raise ValueError("The {} window must lie inside the trials ({:g} to {:g} s)".format(name, times[0], last))
        spans.append((sample_index(start - epochs.tmin, epochs.fs), sample_index(end - epochs.tmin, epochs.fs)))
    shortest = min(b - a for a, b in spans) / epochs.fs
    if len(epochs.labels) < 6:
        raise ValueError("A paired test needs at least 6 trials; these are {}".format(len(epochs.labels)))
    length = min(float(segment), shortest)
    power = []
    for a, b in spans:
        freqs, density = welch_psd(epochs.data[:, :, a:b], epochs.fs, length, 0.5)
        power.append(np.stack([band_integral(freqs, density, low, high) for _, low, high in bands], axis=-1))
    first, second = power                                   # (trials, channels, bands)
    if not ((first > 0).all() and (second > 0).all()):
        raise ValueError("A band has no power in some trial, so its logarithm is undefined")
    rows, p_values = [], []
    for c, channel in enumerate(epochs.names):
        for k, (band, low, high) in enumerate(bands):
            x, y = np.log10(first[:, c, k]), np.log10(second[:, c, k])
            result = wilcoxon(x, y) if test == "wilcoxon" else ttest_rel(x, y)
            mean_a, mean_b = float(first[:, c, k].mean()), float(second[:, c, k].mean())
            rows.append(dict(channel=channel, band=band, low=float(low), high=float(high), mean_a=mean_a,
                             mean_b=mean_b, change=100.0 * (mean_a - mean_b) / mean_b,
                             trials_up=float(np.mean(x > y)), statistic=float(result.statistic),
                             p=float(result.pvalue)))
            p_values.append(float(result.pvalue))
    for row, adjusted in zip(rows, holm(p_values)):
        row["p_holm"] = adjusted
    return Contrast([list(window_a), list(window_b)], [name_a, name_b], [list(b) for b in bands], rows,
                    len(epochs.labels), test, length)


# ----------------------------------------------------------------------
# machine learning
# ----------------------------------------------------------------------


class Chain:
    """A model as a script writes it: `build(seed)` returns its steps, unfitted, as `learn.Model.build` does.

    Every fold of `evaluate` builds the steps again and fits them on its own
    training trials only; `seed` is the fold's (the split's seed plus the repeat).
    """

    def __init__(self, build, description):
        self._build, self.description = build, description

    def build(self, seed=42):
        return self._build(seed)

    def describe(self):
        return self.description


def evaluate(data, model, folds=5, conditions=(), scheme=learn.KFOLD, repeats=5, train_fraction=0.8,
             permutations=0, seed=42, progress=None):
    """Cross-validated scores of `model` on trials (Epochs) or features (Features): the Cross-validate block."""
    values = data.data if isinstance(data, Epochs) else data.values
    labels = np.asarray(data.labels)
    wanted = names_of(conditions)
    keep = np.isin(labels, wanted) if wanted else np.ones(labels.size, dtype=bool)
    if not keep.any():
        raise ValueError("No trials of {}".format(", ".join(wanted)))
    ids = [i for i, k in zip(data.ids, keep) if k]
    _say(progress, "Cross-validating {}".format(model.describe()))
    return learn.cross_validate(values[keep], labels[keep], model, folds, seed, ids=ids, scheme=scheme,
                                repeats=repeats, train_fraction=train_fraction, permutations=int(permutations),
                                progress=progress)
