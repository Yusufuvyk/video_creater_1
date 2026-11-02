"""Electrode names for a recording that does not carry its own.

A BrainFlow CSV stores sixteen columns and no labels, so every tool in this
project has called them EEG1..EEG16 - `alpha_ball_game_v1/plot_data.py` included.
That is honest but useless: which electrode is on channel 3 is decided when the
headset goes on, and nothing in the file records it.

So let the user say it once, here, in a sidecar beside the recording. It is
deliberately separate from the .mat mapping: a CSV has no layout to argue about,
only names, and a name assignment applies to any recording of any format.

Pure stdlib - no Qt.
"""
import json
from pathlib import Path

SUFFIX = ".channels.json"


def path_for(recording_path):
    path = Path(recording_path)
    return path.with_name(path.stem + SUFFIX)


def save_names(recording_path, names):
    path_for(recording_path).write_text(
        json.dumps({"names": list(names)}, indent=2), encoding="utf-8")


def load_names(recording_path, n_channels=None):
    """The names beside a recording, or None.

    A list of the wrong length is ignored rather than padded: half-renamed
    electrodes are worse than none, because the wrong half looks right.
    """
    sidecar = path_for(recording_path)
    if not sidecar.is_file():
        return None
    try:
        names = json.loads(sidecar.read_text(encoding="utf-8")).get("names")
    except (ValueError, TypeError, OSError):
        return None  # a corrupt sidecar should not block the recording
    if not isinstance(names, list) or not names:
        return None
    if n_channels is not None and len(names) != int(n_channels):
        return None
    return [str(name) for name in names]
