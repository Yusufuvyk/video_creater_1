"""Electrode names and where they sit on a head.

A recording carries channel *names*; a head map needs channel *positions*. The
two are joined here, and nowhere else, so both pages draw the same head.

Positions are not hand-typed. MNE ships the standard 10-05 montage, which
contains every 10-20/10-10 label we use - including all 64 Cho 2017 electrodes -
so a name is looked up rather than guessed. Names that are not in it (EMG rows,
`CH37`, a bare `EEG5`) simply have no position, and callers must cope: an
electrode that cannot be placed is listed beside the head, never dropped.

Pure numpy/mne - no Qt, no I/O.
"""
import numpy as np

# ----------------------------------------------------------------------
# named montages
# ----------------------------------------------------------------------

#: Cho 2017's own channel order inside imagery_left / imagery_right. The stored
#: arrays have 68 rows: these 64 EEG electrodes followed by 4 EMG.
CHO2017_EEG = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)

CHO2017_EMG = ("EMG1", "EMG2", "EMG3", "EMG4")

#: OpenBCI's default Cyton+Daisy head plot order. Offered, never assumed: which
#: electrode is on which pin is a wiring choice each lab makes for itself, so
#: this is a starting point the user confirms, not a fact about the hardware.
CYTON_DAISY_1020 = (
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",
)

#: g.tec Unicorn Hybrid Black, fixed electrode positions in channel order.
UNICORN_8 = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")

#: The classic clinical 19.
TEN_TWENTY_19 = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)

#: BNCI 2014-001 (BCI Competition IV 2a): 22 EEG electrodes, then 3 EOG, in
#: the files' column order (MOABB's BNCI2014_001 names them so).
BNCI2014_001_EEG = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
BNCI2014_001_EOG = ("EOG1", "EOG2", "EOG3")

#: Selectable in the mapping dialog's "Channel names" combo, in this order.
MONTAGES = (
    ("Cho2017 64 (EEG only)", CHO2017_EEG),
    ("Cho2017 68 (64 EEG + 4 EMG)", CHO2017_EEG + CHO2017_EMG),
    ("BNCI 2014-001 22 (EEG only)", BNCI2014_001_EEG),
    ("BNCI 2014-001 25 (22 EEG + 3 EOG)", BNCI2014_001_EEG + BNCI2014_001_EOG),
    ("10-20 clinical 19", TEN_TWENTY_19),
    ("Cyton+Daisy 16 (OpenBCI default)", CYTON_DAISY_1020),
    ("Unicorn Hybrid Black 8", UNICORN_8),
)


# ----------------------------------------------------------------------
# selection presets
# ----------------------------------------------------------------------

#: Pfurtscheller (2006): C3 and C4 carry most of the left-vs-right information.
MOTOR = ("C3", "Cz", "C4")

#: Cho 2017's own CSP set - the three central rows.
MOTOR_EXTENDED = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)

OCCIPITAL = ("O1", "Oz", "O2", "POz", "PO3", "PO4", "PO7", "PO8")
FRONTAL = ("Fp1", "Fpz", "Fp2", "AF3", "AFz", "AF4", "F3", "Fz", "F4")
MIDLINE = ("Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz")

#: Shown in the channel picker. Names absent from the recording are ignored,
#: so the same preset works on a 16- and a 64-electrode cap.
PRESETS = (
    ("Motor (C3, Cz, C4)", MOTOR),
    ("Motor extended (21)", MOTOR_EXTENDED),
    ("Midline", MIDLINE),
    ("Frontal", FRONTAL),
    ("Occipital", OCCIPITAL),
)


def generic_names(n_channels):
    """`CH1..CHn` - what a file with no labels of its own gets."""
    return ["CH{}".format(i + 1) for i in range(int(n_channels))]


def guess_montage(names, n_channels=None):
    """The montage these names look like, or None.

    Matches on length first, then on how many labels actually overlap. A file
    that already names its channels sensibly keeps its own names; this only
    answers "which cap is this?" for files that do not.
    """
    n_channels = int(n_channels if n_channels is not None else len(names))
    upper = {str(name).strip().upper() for name in names}

    best, best_score = None, 0.0
    for label, montage in MONTAGES:
        if len(montage) != n_channels:
            continue
        overlap = len({m.upper() for m in montage} & upper)
        score = overlap / float(len(montage))
        if score > best_score:
            best, best_score = label, score

    # Below half the labels it is a coincidence, not a montage.
    return best if best_score >= 0.5 else None


def montage_named(label):
    """The electrode list behind a MONTAGES label, or None."""
    for name, montage in MONTAGES:
        if name == label:
            return list(montage)
    return None


# ----------------------------------------------------------------------
# positions
# ----------------------------------------------------------------------

_STANDARD = None


def _standard_positions():
    """{NAME: (x, y, z)} from MNE's 10-05 montage, loaded once.

    Building the montage costs a moment and pulls in MNE, so it is deferred
    until a head is actually drawn - importing this module stays cheap.
    """
    global _STANDARD
    if _STANDARD is None:
        from mne.channels import make_standard_montage
        found = make_standard_montage("standard_1005").get_positions()["ch_pos"]
        _STANDARD = {name.upper(): np.asarray(xyz, dtype=float)
                     for name, xyz in found.items()}
    return _STANDARD


def _sphere_centre(points):
    """Least-squares centre of the sphere through `points`.

    The montage's origin is the fiducial frame, not the centre of the head, so
    it sits a few millimetres low. Projecting from there tilts the whole map -
    Cz drifts off centre and the front of the head crowds together. Fitting the
    centre first removes that.
    """
    if len(points) < 4:
        return np.zeros(3)
    design = np.column_stack([2.0 * points, np.ones(len(points))])
    target = (points ** 2).sum(axis=1)
    solution = np.linalg.lstsq(design, target, rcond=None)[0]
    return solution[:3]


def _ear_radius(flat):
    """Mean distance of T7/T8 from the centre, or None if neither is present."""
    ears = [np.linalg.norm(xy) for name, xy in flat.items()
            if str(name).strip().upper() in ("T7", "T8", "T3", "T4")]
    return float(np.mean(ears)) if ears else None


def positions_2d(names):
    """{name: (x, y)} for the names that can be placed on a head.

    Azimuthal equidistant projection, the standard EEG topographic layout:
    distance from the vertex is preserved along every radius, so the back of the
    head is not crushed the way a straight overhead view crushes it. Scaled so
    the ears (T7/T8) land on the unit circle, which is where the head outline is
    drawn.

    Names with no known position are absent from the result - deliberately, so
    the caller has to decide what to do with them rather than silently placing
    them at the origin.
    """
    standard = _standard_positions()
    placed = [(name, standard[str(name).strip().upper()])
              for name in names if str(name).strip().upper() in standard]
    if not placed:
        return {}

    points = np.vstack([xyz for _, xyz in placed])
    centred = points - _sphere_centre(points)

    radius = np.linalg.norm(centred, axis=1)
    radius[radius == 0] = 1.0
    # Angle down from the vertex becomes distance out from the centre.
    theta = np.arccos(np.clip(centred[:, 2] / radius, -1.0, 1.0))
    phi = np.arctan2(centred[:, 1], centred[:, 0])
    # x to the right, y forward: the head is drawn nose-up.
    flat = np.column_stack([theta * np.cos(phi), theta * np.sin(phi)])

    out = {name: flat[i] for i, (name, _) in enumerate(placed)}
    scale = _ear_radius(out) or float(np.linalg.norm(flat, axis=1).max()) or 1.0
    return {name: (float(xy[0] / scale), float(xy[1] / scale))
            for name, xy in out.items()}
