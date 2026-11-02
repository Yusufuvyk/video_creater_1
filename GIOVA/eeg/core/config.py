"""Board constants and paths. No Qt, no I/O - safe to import anywhere."""
import sys
from pathlib import Path

from brainflow.board_shim import BoardIds, BoardShim

from .montages import CYTON_DAISY_1020, generic_names

BOARD = BoardIds.CYTON_DAISY_BOARD
FS = BoardShim.get_sampling_rate(BOARD)
EEG_ROWS = BoardShim.get_eeg_channels(BOARD)
N_CH = len(EEG_ROWS)
#: Unix timestamp column. Needed to line event markers up with samples.
TIMESTAMP_ROW = BoardShim.get_timestamp_channel(BOARD)

#: LIVE CAPTURE AND BRAINFLOW CSV ONLY. A loaded .mat or a recording with a
#: names sidecar carries its own channel names and its own count, and those win
#: everywhere: a 64-electrode file labelled EEG1..EEG16 is not a cosmetic
#: problem, it renames the electrodes and silently discards three quarters of
#: them. Anything reading these constants to describe such a *file* is a bug.
#:
#: The electrodes are OpenBCI's own default 16-channel layout, in pin order -
#: Cyton 1-8 then Daisy 9-16. Verified against the GUI shipped in this repo:
#: `OpenBCI_GUI/data/electrode_positions_default.txt` gives the same sixteen
#: positions in the same order, and each one lands within a fifth of a
#: half-head-width of its true 10-20 location.
#:
#: This is the lab's declared standard, not a fact about the hardware. If a
#: headset is wired differently, change it here, or override one recording at a
#: time with the Channel names dialog (which writes a `.channels.json` sidecar).
CH_NAMES = (list(CYTON_DAISY_1020) if N_CH == len(CYTON_DAISY_1020)
            else generic_names(N_CH))

RAIL_UV = 187_500.0  # Cyton amplifier saturation
NYQUIST = FS / 2.0

BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)

_PKG = Path(__file__).resolve().parent
_SCRIPTS = _PKG.parent          # eegstudio
_PROJECT = _SCRIPTS.parent      # the folder above

#: The recordings folder has moved around and is spelled both ways. Try the
#: likely spots rather than hard-coding one, so moving the project again does
#: not silently empty the file list.
_DATA_NAMES = ("Data", "data")


def find_data_dir():
    # Packaged exe: the code lives in a bundle folder that may be replaced on
    # the next build. An exe still in the project's dist folder records into the
    # project's own Data (a rebuild recreates dist); one copied anywhere else
    # records next to itself.
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve().parent
        if exe.parent.name.lower() == "dist":
            for name in _DATA_NAMES:
                candidate = exe.parent.parent / name
                if candidate.is_dir():
                    return candidate
        return exe / "Data"
    for parent in (_SCRIPTS, _PROJECT):
        for name in _DATA_NAMES:
            candidate = parent / name
            if candidate.is_dir():
                return candidate
    return _SCRIPTS / "Data"


DATA_DIR = find_data_dir()

#: Game folders (each with an eegstudio_game.json) found automatically: next to
#: the exe when packaged, else eegstudio/Games. More folders can be added in the app.
GAMES_DIR = (Path(sys.executable).resolve().parent / "Games" if getattr(sys, "frozen", False)
             else _SCRIPTS / "Games")
VENV_PYTHON = _PROJECT / ".venv" / "Scripts" / "python.exe"

#: Marker sidecars sit next to their recording and share the .csv extension,
#: but hold (unix_time, label) rows rather than samples. Listing one as a
#: recording just produces a confusing failure at load time.
EVENTS_SUFFIX = "_events.csv"


#: Extensions that hold a recording. EDF/BDF/GDF/FIF/SET/VHDR are read with MNE.
RECORDING_SUFFIXES = (".csv", ".mat", ".edf", ".bdf", ".gdf", ".fif", ".set", ".vhdr")
#: File dialog filter for every readable recording.
RECORDING_FILTER = ("Recordings (*.csv *.mat *.edf *.bdf *.gdf *.fif *.set *.vhdr);;"
                    "BrainFlow CSV (*.csv);;MATLAB (*.mat);;EDF/BDF/GDF (*.edf *.bdf *.gdf);;"
                    "MNE FIF (*.fif);;EEGLAB (*.set);;BrainVision (*.vhdr)")


def list_recordings(directory=None):
    """Recordings in `directory`, excluding marker and mapping sidecars."""
    directory = directory or DATA_DIR
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.suffix.lower() in RECORDING_SUFFIXES
        and not path.name.endswith(EVENTS_SUFFIX)
        and not path.name.startswith("features_")
        and not path.name.endswith(".mapping.json"))


def events_for(recording_path):
    """The marker sidecar belonging to a recording, if there is one.

    Two spellings are in use: `<recording>_events.csv`, and a per-subject
    `<subject>_events.csv` shared by that subject's recordings (which is how
    alpha.csv pairs with alpha_events.csv).
    """
    path = Path(recording_path)
    stems = [path.stem]
    if "_" in path.stem:
        stems.append(path.stem.split("_")[0])
    for stem in stems:
        candidate = path.with_name(stem + EVENTS_SUFFIX)
        if candidate.is_file():
            return candidate
    return None
