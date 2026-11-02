"""EDF, BDF, GDF, FIF, EEGLAB and BrainVision recordings, read through MNE.

Most public BCI datasets ship in one of these: BCI Competition IV 2a (GDF),
PhysioNet motor movement/imagery (EDF), MNE's own FIF, EEGLAB .set. MNE reads
them all; this module only turns what it reads into the project's Recording:

    - EEG channels only by default (EOG, EMG, stimulus channels left out),
    - volts converted to microvolts,
    - channel names cleaned so they meet the 10-20 montage ('EEG-C3', 'C3..'
      and 'Fc5.' all become the electrode name),
    - events from annotations or a stimulus channel, as raw codes. What a code
      *means* (769 = left hand) is the user's answer, given once in the event
      dialog and saved beside the file.

An epochs file (-epo.fif) is laid out as a continuous recording, epochs end to
end with a gap of missing samples between them, one event per epoch at its
original time zero - so the Epoch page can cut it again exactly.
"""
from pathlib import Path
import re

import numpy as np

from ..core.epochs import EventTable
from ..core.sources import Recording

MNE_SUFFIXES = (".edf", ".bdf", ".gdf", ".fif", ".set", ".vhdr")

#: Annotations that mark bad or boundary segments rather than trials: any description starting with one of
#: these, whatever its case (BAD_blink, bad_motion, EDGE, boundary), as MNE itself reads them.
IGNORED_LABELS = ("BAD", "EDGE", "boundary")
#: Matches an annotation that is not one of those (mne.events_from_annotations' `regexp`).
_KEPT = "(?i)^(?!({}))".format("|".join(re.escape(b) for b in IGNORED_LABELS))


def _is_ignored(description):
    return str(description).upper().startswith(tuple(b.upper() for b in IGNORED_LABELS))


def is_epochs_file(path):
    name = Path(path).name.lower()
    return name.endswith("-epo.fif") or name.endswith("_epo.fif")


def clean_name(name):
    """'EEG-C3' -> 'C3', 'Fc5.' -> 'FC5', 'EEG Fz-Ref' -> 'Fz'. Unknown stays as is."""
    text = str(name).strip()
    text = re.sub(r"^(EEG|eeg)[\s:_-]+", "", text)
    text = re.sub(r"[-_ ](REF|Ref|ref|LE|A1|A2|M1|M2|AVG)$", "", text)
    return _standard_case(text.strip(". "))


def _standard_case(text):
    """10-20 spelling: upper case except 'Fp' and a trailing 'z' or 'h' (Fz, Fp1, FCz, AFp1h)."""
    if not re.fullmatch(r"[A-Za-z]{1,4}[0-9zZ]{1,2}[hH]?", text):
        return text
    upper = text.upper()
    if upper.endswith("H") and len(upper) > 2:
        upper = upper[:-1] + "h"
    if upper.rstrip("h").endswith("Z"):
        head = upper.rstrip("h")
        upper = head[:-1] + "z" + upper[len(head):]
    return re.sub(r"FP", "Fp", upper)


def _unique(names):
    seen, out = {}, []
    for name in names:
        if name in seen:
            seen[name] += 1
            name = "{}_{}".format(name, seen[name])
        else:
            seen[name] = 0
        out.append(name)
    return out


def _quiet():
    import mne
    mne.set_log_level("ERROR")
    return mne


def read_raw(path):
    mne = _quiet()
    if is_epochs_file(path):
        return mne.read_epochs(str(path), preload=True)
    return mne.io.read_raw(str(path), preload=True)


def raw_events(raw):
    """EventTable with raw labels: annotation descriptions, else stim channel codes."""
    mne = _quiet()
    fs = raw.info["sfreq"]
    annotations = raw.annotations
    if annotations is not None and len(annotations):
        # events_from_annotations resolves orig_time / first_samp correctly for every format.
        try:
            found, codes = mne.events_from_annotations(raw, regexp=_KEPT, verbose="error")
        except ValueError:
            found, codes = np.zeros((0, 3), int), {}
        if len(found):
            names = {code: label for label, code in codes.items()}
            # one event per annotation kept, in the annotations' order: their lengths go with them
            kept = [not _is_ignored(d) for d in annotations.description]
            durations = np.asarray(annotations.duration, dtype=float)[kept]
            return EventTable((found[:, 0] - raw.first_samp) / fs,
                              durations if durations.size == len(found) else None,
                              [names[code] for code in found[:, 2]])
    stim = mne.pick_types(raw.info, stim=True, eeg=False)
    if len(stim):
        found = mne.find_events(raw, stim_channel=raw.ch_names[stim[0]], shortest_event=1, verbose="error")
        if len(found):
            onsets = (found[:, 0] - raw.first_samp) / fs
            return EventTable(onsets, None, [str(code) for code in found[:, 2]])
    return EventTable()


def bad_spans(raw):
    """[(start s, end s, label)] of the annotations that mark data as not to be used (BAD_..., EDGE,
    boundary), on the same time axis as `raw_events`: s from the first sample."""
    annotations = raw.annotations
    if annotations is None or not len(annotations):
        return []
    bad = [_is_ignored(d) for d in annotations.description]
    if not any(bad):
        return []
    # as mne.events_from_annotations places an annotation, so the spans line up with the events
    samples = raw.time_as_index(annotations.onset, use_rounding=True, origin=annotations.orig_time)
    if annotations.orig_time is not None:
        samples = samples + raw.first_samp
    fs = raw.info["sfreq"]
    return [(float((sample - raw.first_samp) / fs), float((sample - raw.first_samp) / fs + duration), str(label))
            for sample, duration, label, b in zip(samples, annotations.duration, annotations.description, bad) if b]


def load(path, include_non_eeg=False):
    """Recording with raw-labelled events (no label mapping applied yet)."""
    mne = _quiet()
    path = Path(path)
    source = read_raw(path)
    if isinstance(source, mne.BaseEpochs):
        return _from_epochs(source, path, include_non_eeg)
    raw = source
    events = raw_events(raw)
    picks = (mne.pick_types(raw.info, eeg=True, eog=True, emg=True, ecg=True, misc=True, stim=False)
             if include_non_eeg else mne.pick_types(raw.info, eeg=True, stim=False))
    if not len(picks):
        # Some EDF writers type every channel as EEG, others none at all.
        picks = [i for i, kind in enumerate(raw.get_channel_types()) if kind != "stim"]
    if not len(picks):
        raise ValueError("No signal channels in {}".format(path.name))
    data = raw.get_data(picks=picks) * 1e6
    names = _unique([clean_name(raw.ch_names[i]) for i in picks])
    recording = Recording(data, raw.info["sfreq"], names, None, path, events=events)
    recording.bad_spans = bad_spans(raw)
    recording.unit_note = unit_note(raw, picks, path)
    recording.reading = reading_notes(raw, picks, recording, path)
    _add_motion(recording, raw)
    return recording


#: What a file format calls the calibration MNE applies, for `unit_note`.
FORMAT_NAMES = {".vhdr": "BrainVision header", ".edf": "EDF header", ".bdf": "BDF header", ".gdf": "GDF header",
                ".set": "EEGLAB file", ".fif": "FIF file"}


def unit_note(raw, picks, path):
    """How MNE turned the file's stored numbers into µV, in words: the factor per stored unit, from the header.

    MNE multiplies each stored value by the channel's calibration (BrainVision: its "resolution";
    EDF/BDF: physical range over digital range) and its unit's factor to volts; GIOVA then
    multiplies by 1e6. For a FIF file the numbers are stored in volts already.
    """
    factors = sorted({float("{:.6g}".format(raw.info["chs"][i]["cal"] * raw.info["chs"][i]["range"] * 1e6))
                      for i in picks})
    where = FORMAT_NAMES.get(Path(path).suffix.lower(), "File header")
    if not factors:
        return ""
    if len(factors) == 1:
        factor = factors[0]
        if factor == 1.0:
            return "{}: the stored numbers are µV (× 1)".format(where)
        return "{}: each stored number × {:g} µV".format(where, factor)
    return "{}: each stored number × {:g} to {:g} µV, channel by channel".format(where, factors[0], factors[-1])


#: Accelerometer channels as BrainVision (LiveAmp) names them, and the units they come in.
MOTION_NAMES = ("x_dir", "y_dir", "z_dir", "acc_x", "acc_y", "acc_z", "accx", "accy", "accz")
MOTION_UNITS = ("mg", "g", "m/s²", "m/s^2")


def _add_motion(recording, raw):
    """A LiveAmp's (or any file's) accelerometer channels as the recording's motion, in their own unit.

    They are not EEG: kept apart from the channels, as the Unicorn's are, so a trial that moved can be
    found (Epochs' motion limit) without them ever being filtered or averaged as brain signal.
    """
    units = getattr(raw, "_orig_units", {}) or {}
    rows = [i for i, name in enumerate(raw.ch_names)
            if name.lower() in MOTION_NAMES or str(units.get(name, "")).strip() in MOTION_UNITS]
    if not rows or len(raw.times) != recording.n_samples:
        return
    # MNE keeps a unit it does not know as it is (factor 1): the numbers stay the file's own, in its unit
    values = raw.get_data(picks=rows)
    for k, i in enumerate(rows):
        values[k] /= raw.info["chs"][i]["range"] or 1.0
    recording.motion = values
    recording.motion_names = [raw.ch_names[i] for i in rows]
    header = _brainvision_units(recording.path) if recording.path is not None else {}
    recording.motion_units = [header.get(raw.ch_names[i]) or (units.get(raw.ch_names[i], "") if
                              units.get(raw.ch_names[i], "") not in ("", "n/a") else "a.u.") for i in rows]


def _brainvision_units(path):
    """{channel name: unit} from a BrainVision header's channel lines; {} for any other file."""
    if Path(path).suffix.lower() != ".vhdr":
        return {}
    found = {}
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^Ch\d+=(.*)$", line.strip())
            if match:
                fields = match.group(1).split(",")
                if len(fields) >= 4:
                    found[fields[0].replace(r"\1", ",")] = fields[3].strip()
    except OSError:
        return {}
    return found


def _from_epochs(epochs, path, include_non_eeg):
    mne = _quiet()
    picks = (mne.pick_types(epochs.info, eeg=True, eog=True, emg=True, misc=True, stim=False)
             if include_non_eeg else mne.pick_types(epochs.info, eeg=True, stim=False))
    if not len(picks):
        raise ValueError("No EEG channels in {}".format(path.name))
    values = epochs.get_data(picks=picks) * 1e6          # (n_epochs, n_channels, n_times)
    fs = epochs.info["sfreq"]
    gap = int(round(fs))
    code_to_label = {code: label for label, code in epochs.event_id.items()}
    blocks, onsets, labels, position = [], [], [], 0
    for index, epoch in enumerate(values):
        if blocks:
            blocks.append(np.full((values.shape[1], gap), np.nan))
            position += gap
        blocks.append(epoch)
        onsets.append((position - epochs.times[0] * fs) / fs)
        labels.append(code_to_label.get(epochs.events[index, 2], str(epochs.events[index, 2])))
        position += epoch.shape[1]
    names = _unique([clean_name(epochs.ch_names[i]) for i in picks])
    duration = values.shape[2] / fs
    recording = Recording(np.hstack(blocks), fs, names, None, path,
                          events=EventTable(onsets, [duration] * len(onsets), labels))
    recording.reading = ["Read with MNE-Python (an epochs file): {} trials laid one after another, each its own "
                         "event.".format(len(onsets))]
    return recording


def reading_notes(raw, picks, recording, path):
    """What `load` did with a file, in words: the reader, the channels left out, the spans marked bad."""
    notes = ["Read with MNE-Python ({}).".format(FORMAT_NAMES.get(Path(path).suffix.lower(), Path(path).suffix)
                                                 .replace(" header", "").replace(" file", ""))]
    kept = {raw.ch_names[i] for i in picks}
    others = [n for i, n in enumerate(raw.ch_names) if n not in kept and raw.get_channel_types()[i] != "stim"]
    if others:
        notes.append("Not EEG, left out of the signal: {}{}.".format(
            ", ".join(others[:8]), " and {} more".format(len(others) - 8) if len(others) > 8 else ""))
    if recording.bad_spans:
        labels = list(dict.fromkeys(label for _, _, label in recording.bad_spans))
        notes.append("Marked bad in the file: {} span(s) ({}), {:.1f} s; trials over them are left out.".format(
            len(recording.bad_spans), ", ".join(labels[:4]), sum(b - a for a, b, _ in recording.bad_spans)))
    notes.append("Units: {}.".format(recording.unit_note))
    return notes
