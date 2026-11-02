"""Datasets we recognise on sight.

Auto-detection in `matfile` answers "which array is the data?" from shape alone.
That is enough for a file with one array in it and hopeless for a public dataset,
where the answer depends on facts the file does not state: which rows are EEG
rather than EMG, what the stored integers mean in microvolts, what the electrodes
are called, how long a trial is.

A preset supplies those facts for a dataset we have identified by its variable
names. It never picks the *condition* - left hand or right hand is a scientific
choice, not a layout detail - so it offers the conditions and the user chooses.

Pure numpy/stdlib - no Qt.
"""
import numpy as np

from .matfile import Mapping
from ..core.montages import CHO2017_EEG
from ..core.paradigm import Paradigm

#: Cho 2017 stores raw Biosemi A/D counts, not microvolts: 1 count = 31.25 nV.
#: Skipping this makes every amplitude 32x too big, which quietly breaks the
#: paper's own bad-trial rule ("|x| > 100 uV") and every absolute number on the
#: page. Derived in the Lab project's cho2017.py.
CHO2017_COUNTS_TO_UV = 1.0 / 32.0

#: 64 EEG electrodes, then 4 EMG. Nothing in the file marks the boundary.
CHO2017_EEG_ROWS = "1-64"

#: The paradigm behind one 7 s trial, from the paper: 2 s pre-cue rest, the cue
#: at 2 s, imagery until 5 s, then the rest of the trial.
CHO2017_PHASES = (("rest", 2.0), ("imagery", 3.0), ("post", 2.0))


class Preset:
    """A recognised dataset: what it is, and the ways it can be read."""

    __slots__ = ("name", "note", "conditions")

    def __init__(self, name, note, conditions):
        self.name = name
        self.note = note
        #: [(label, Mapping)] - one entry per condition stored in the file.
        self.conditions = list(conditions)

    @property
    def labels(self):
        return [label for label, _ in self.conditions]

    def mapping_for(self, label):
        for name, mapping in self.conditions:
            if name == label:
                return mapping
        return None


# ----------------------------------------------------------------------
# Cho 2017 - "EEG datasets for motor imagery brain-computer interface"
# ----------------------------------------------------------------------

#: Every condition the file may hold, with the trial-count field that goes with
#: it. `rest` is continuous, so it has no trials and gets no paradigm.
CHO2017_CONDITIONS = (
    ("imagery_left", "eeg.n_imagery_trials"),
    ("imagery_right", "eeg.n_imagery_trials"),
    ("movement_left", "eeg.n_movement_trials"),
    ("movement_right", "eeg.n_movement_trials"),
    ("rest", None),
)

#: The signature. Any file with all four is this dataset; no other layout we
#: have seen names its arrays this way.
CHO2017_SIGNATURE = ("eeg.srate", "eeg.imagery_left", "eeg.imagery_right",
                     "eeg.n_imagery_trials")


def _scalar(variables, key):
    """A numeric scalar variable, or None if it is missing or not a number."""
    if key not in variables:
        return None
    array = np.asarray(variables[key])
    if array.size != 1 or array.dtype.kind not in "fiu":
        return None
    return float(array.reshape(-1)[0])


def _cho2017(variables):
    if not all(key in variables for key in CHO2017_SIGNATURE):
        return None
    fs = _scalar(variables, "eeg.srate")
    if not fs or fs <= 0:
        return None

    conditions = []
    for field, _ in CHO2017_CONDITIONS:
        key = "eeg." + field
        if key not in variables:
            continue
        conditions.append((field, Mapping(
            key, fs=fs, transpose=False, scale=CHO2017_COUNTS_TO_UV,
            rows=CHO2017_EEG_ROWS, names=CHO2017_EEG, preset="Cho2017")))

    # Pairs of conditions read together, each trial labelled by its side. The
    # comparison the dataset exists for (left vs right) needs both at once.
    for first, second, count_key in (("imagery_left", "imagery_right", "eeg.n_imagery_trials"),
                                     ("movement_left", "movement_right", "eeg.n_movement_trials")):
        keys = ("eeg." + first, "eeg." + second)
        if not all(key in variables for key in keys):
            continue
        n_trials = _scalar(variables, count_key) or 0
        parts = [[keys[0], first.split("_")[1], n_trials], [keys[1], second.split("_")[1], n_trials]]
        conditions.insert(0, ("{} + {}".format(first, second), Mapping(
            keys[0], fs=fs, transpose=False, scale=CHO2017_COUNTS_TO_UV,
            rows=CHO2017_EEG_ROWS, names=CHO2017_EEG, preset="Cho2017", parts=parts)))

    if not conditions:
        return None
    return Preset(
        "Cho2017",
        "64 EEG electrodes then 4 EMG, stored as A/D counts at {:g} Hz. The "
        "preset keeps rows 1-64, scales by 1/32 to microvolts and names the "
        "electrodes; which condition to analyse is yours to choose. A joined "
        "pair (e.g. imagery_left + imagery_right) loads both sides as one "
        "recording with one 'left' / 'right' event per trial, for the "
        "Epoch / ERD page.".format(fs),
        conditions)


#: Every detector, tried in order. A detector returns a Preset or None.
DETECTORS = (_cho2017,)


def detect(variables):
    """The preset for this file's variables, or None if we do not know it."""
    for detector in DETECTORS:
        preset = detector(variables)
        if preset is not None:
            return preset
    return None


def trial_events(variables, mapping):
    """One event per trial of a single condition, labelled by its side ('right'), else None.

    What a joined pair gives each trial (`matfile.join_parts`), for one side read
    alone: trial 1 is then the side's first trial, as the dataset numbers it.
    """
    from ..core.epochs import EventTable
    if getattr(mapping, "parts", None):
        return None
    field = str(mapping.data_key).split(".")[-1]
    count_key = next((key for name, key in CHO2017_CONDITIONS if name == field), None)
    n_trials = int(_scalar(variables, count_key) or 0) if count_key else 0
    array = np.asarray(variables[mapping.data_key])
    if n_trials < 1 or array.ndim != 2:
        return None
    length = array.shape[1] / n_trials / float(mapping.fs)
    return EventTable(np.arange(n_trials) * length, np.full(n_trials, length), [field.split("_")[-1]] * n_trials)


def paradigm_for(variables, mapping):
    """The trial structure behind one condition, or None if it has none.

    Cho 2017 stores a condition as one long array with its trials laid end to
    end and only a trial *count* to say where the joins are. Turning that count
    into labelled blocks is what lets the analysis page average across trials
    at all - without it the file reads as one 700 s stretch, and the trial
    structure that the whole dataset exists for is invisible.
    """
    if getattr(mapping, "parts", None):
        return None  # a joined recording carries its own per-trial events
    field = str(mapping.data_key).split(".")[-1]
    count_key = next((key for name, key in CHO2017_CONDITIONS if name == field),
                     None)
    n_trials = _scalar(variables, count_key) if count_key else None
    if not n_trials or n_trials < 2:
        return None

    array = np.asarray(variables[mapping.data_key])
    if array.ndim != 2:
        return None
    n_trials = int(n_trials)
    trial_seconds = array.shape[1] / float(n_trials) / float(mapping.fs)
    if trial_seconds <= 0:
        return None

    # The paper's own timing, but only claimed for a trial long enough to hold
    # it. A condition with some other trial length gets one unlabelled block
    # per trial, which still enables trial averaging without inventing a cue.
    if abs(trial_seconds - sum(s for _, s in CHO2017_PHASES)) < 0.5:
        phases = [(label, seconds) for label, seconds in CHO2017_PHASES]
    else:
        phases = [("trial", trial_seconds)]

    return Paradigm("{} ({})".format(mapping.preset or "preset", field),
                    phases, repeats=n_trials, fs=mapping.fs)
