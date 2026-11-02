"""Opening a recording file, whatever kind it is.

The dispatcher lives here rather than in `core.sources` on purpose. Deciding
how to read a file needs to know about .mat layouts, mappings and dataset
presets - all of which already depend on `core`. Putting the choice in `core`
made the dependency run both ways, which is how a package turns into a knot.

`core` now knows what a Recording *is*; `formats` knows how to get one.
"""
import functools
import re
from pathlib import Path

from ..core.sources import load_brainflow_csv
from . import channels


def load_recording(path, mapping=None):
    """Open any supported file. `mapping` is only used by .mat files."""
    path = Path(path)
    if path.suffix.lower() == ".mat":
        from .matfile import load_mat
        recording = load_mat(path, mapping)
    elif path.suffix.lower() in (".edf", ".bdf", ".gdf", ".fif", ".set", ".vhdr"):
        from . import mne_io
        recording = mne_io.load(path)
    else:
        recording = load_brainflow_csv(path)

    # A names sidecar wins over whatever the file or the mapping produced: it
    # is the most recent thing a human actually asserted about this recording.
    names = channels.load_names(path, recording.n_channels)
    if names:
        recording.channel_names = names
    return recording


def dependencies(path):
    """Every file a reading of `path` may depend on, the recording first, whether or not each exists.

    The sidecars (channel names, events, event map, mapping, board, paradigm, a subject's
    marker CSV) and the data files of formats split over several files (BrainVision .eeg/.vmrk,
    EEGLAB .fdt, split FIF). A cache keyed on these sees a sidecar created, edited or deleted.
    """
    from ..core.config import EVENTS_SUFFIX
    from ..core.sources import SIDECAR_SUFFIXES
    path = Path(path)
    found = [path] + [path.with_name(path.stem + suffix) for suffix in SIDECAR_SUFFIXES]
    if "_" in path.stem:
        found.append(path.with_name(path.stem.split("_")[0] + EVENTS_SUFFIX))
    suffix = path.suffix.lower()
    if suffix == ".vhdr":
        found += _brainvision_files(path)
    elif suffix == ".set":
        found += _eeglab_files(path)
    elif suffix == ".fif":
        found += [path.with_name("{}-{}.fif".format(path.stem, k)) for k in range(1, 4)]
    return list(dict.fromkeys(found))


def _brainvision_files(header):
    """The .eeg and .vmrk files a BrainVision header names, else the ones beside it with its stem.

    Named paths are relative to the header's folder and may go into a sub-folder (data/signal.eeg), as
    MNE reads them.
    """
    named = []
    try:
        for line in _brainvision_text(header).splitlines():
            key, _, value = line.partition("=")
            if key.strip() in ("DataFile", "MarkerFile") and value.strip():
                named.append(header.parent / value.strip())
    except OSError:
        pass
    return named or [header.with_suffix(".eeg"), header.with_suffix(".vmrk")]


def _brainvision_text(header):
    """A BrainVision header's text decoded as MNE decodes it, so a file name with ö or ç is the one MNE opens:
    in the code page it states (ANSI: cp1252), UTF-8 when it states none, Latin-1 when that fails."""
    raw = header.read_bytes()
    stated = re.search(r"Codepage=(.+)", raw.decode("ascii", "ignore"), re.IGNORECASE)
    codepage = stated.group(1).strip() if stated else "utf-8"
    codepage = "cp1252" if codepage == "ANSI" else codepage
    try:
        return raw.decode(codepage)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("latin-1")


def _eeglab_files(path):
    """The data file an EEGLAB .set reads its samples from, as MNE finds it.

    EEG.data names it, relative to the .set's folder, when it is not stored in the
    .set itself; when that file is missing MNE falls back to the .set's own stem
    with .fdt, so both count.
    """
    beside = path.with_suffix(".fdt")
    try:
        stat = path.stat()
    except OSError:
        return [beside]
    name = _eeglab_data_name(str(path), stat.st_mtime_ns, stat.st_size)
    return [path.parent / name, beside] if name else [beside]


@functools.lru_cache(maxsize=64)
def _eeglab_data_name(path, _mtime, _size):
    """EEG.data of a .set when it names a file, else ""; read once per version of the file (cache keys are
    computed often, and a .set may hold all its samples)."""
    from scipy.io import loadmat
    from .matfile import is_v73, v73_text
    if is_v73(path):
        data = v73_text(path, ("data", "EEG/data"))      # only the name: the samples stay on disk
    else:
        try:
            variables = loadmat(path, variable_names=["EEG", "data"], squeeze_me=True, struct_as_record=False)
        except (OSError, ValueError, NotImplementedError, TypeError):
            return ""        # an unreadable .set: the reader says what is wrong
        data = getattr(variables.get("EEG"), "data", variables.get("data"))
    return str(data) if isinstance(data, str) and str(data).lower().endswith((".fdt", ".set")) else ""
