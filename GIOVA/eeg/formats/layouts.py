"""MATLAB layouts kept for every file laid out the same way.

A `matfile.Mapping` says how one file holds its signal: which variable, which
way round, at what rate, with which names and events. Chosen once in the layout
dialog it is kept beside that file (its .mapping.json). A public dataset comes
as many files of one layout - BNCI 2014-001 is nine people times two sessions -
and nobody should answer the same questions eighteen times; a lab's own export
script writes every file alike too. So a layout can also be kept by name, with
the variables it needs as its signature, and any file that has them and the same
number of channels is read with it.

Built-in layouts are written out here, as a saved one would be; the user's are
JSON files in Data/layouts, tried first. Only a file that has every variable a
layout names is read with it, so a layout never takes over a file it was not
made for.

Pure stdlib/numpy - no Qt.
"""
import json
import re
from pathlib import Path

import numpy as np

from ..core.config import DATA_DIR
from ..core.montages import BNCI2014_001_EEG
from .matfile import SAMPLES_1, Mapping, channels_of, runs

SUFFIX = ".layout.json"


class Layout:
    """A named way of reading MATLAB files, and how to recognise the files it fits."""

    __slots__ = ("name", "note", "mapping", "signature", "channels", "path")

    def __init__(self, name, note, mapping, signature=None, channels=None, path=None):
        self.name = name
        self.note = note
        self.mapping = mapping
        #: the variables a file must have ({*} keys: at least one run)
        self.signature = list(signature) if signature else signature_of(mapping)
        #: rows of the signal before any are left out (the file's own channel count); None: any
        self.channels = int(channels) if channels else None
        #: the file it was read from; None for a built-in one
        self.path = path

    def fits(self, variables):
        """True when the file has every variable this layout needs, its signal as many channels as it expects."""
        if not all(runs(variables, key) for key in self.signature):
            return False
        if self.channels is None:
            return True
        array = np.asarray(variables[runs(variables, self.mapping.data_key)[0][1]])
        if array.ndim != (3 if self.mapping.axes else 2):
            return False
        return channels_of(array, self.mapping) == self.channels

    def to_dict(self):
        return {"name": self.name, "note": self.note, "mapping": self.mapping.to_dict(),
                "signature": self.signature, "channels": self.channels}

    @classmethod
    def from_dict(cls, raw, path=None):
        return cls(str(raw["name"]), str(raw.get("note", "")), Mapping.from_dict(raw["mapping"]),
                   raw.get("signature"), raw.get("channels"), path)


def signature_of(mapping):
    """The variables a mapping reads: what a file needs to be read with it."""
    keys = [mapping.data_key, mapping.fs_key if not mapping.fs else None, mapping.names_key, mapping.onsets_key,
            mapping.codes_key, mapping.labels_key, mapping.bad_key]
    return list(dict.fromkeys(k for k in keys if k))


# ----------------------------------------------------------------------
# built in
# ----------------------------------------------------------------------

BNCI2014_001 = Layout(
    "BNCI 2014-001 (BCI Competition IV 2a)",
    "Nine runs per file (data{1} ... data{9}): three eye-movement runs without trials, left out, then six "
    "runs of 48 motor imagery trials (left hand, right hand, feet, tongue), joined with a 1 s gap. 22 EEG "
    "electrodes at 250 Hz in microvolts; the 3 EOG columns are left out. Each trial's start is 'trial' "
    "(samples from 1), its class 'y' (1-4, named by 'classes'). Trials the dataset's experts marked as "
    "artefacts ('artifacts') are left out: 2 to 6 s after the trial's start is marked, as MOABB marks it "
    "(MOABB keeps them unless asked). Tangermann et al. (2012), doi 10.3389/fnins.2012.00055.",
    Mapping("data{*}.X", fs=None, fs_key="data{*}.fs", transpose=True, scale=1.0, rows="1-22",
            names=list(BNCI2014_001_EEG), preset="BNCI 2014-001", onsets_key="data{*}.trial",
            onset_unit=SAMPLES_1, codes_key="data{*}.y", labels_key="data{*}.classes",
            bad_key="data{*}.artifacts", bad_window=[2.0, 6.0]),
    channels=25)

BUILT_IN = (BNCI2014_001,)


# ----------------------------------------------------------------------
# kept by the user
# ----------------------------------------------------------------------


def layouts_dir(root=None):
    return Path(root or DATA_DIR) / "layouts"


def slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name).strip()).strip("-.") or "layout"


def saved(root=None):
    """The user's layouts, by name; a file that cannot be read is passed over."""
    found = []
    directory = layouts_dir(root)
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*" + SUFFIX)):
        try:
            found.append(Layout.from_dict(json.loads(path.read_text(encoding="utf-8")), path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def save(layout, root=None):
    """Keep `layout` in Data/layouts under its name (an older one of that name is replaced); its path."""
    directory = layouts_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (slug(layout.name) + SUFFIX)
    path.write_text(json.dumps(layout.to_dict(), indent=2), encoding="utf-8")
    layout.path = path
    return path


def every(root=None):
    """The user's layouts, then the built-in ones: a layout someone chose wins over one that came with the program."""
    return saved(root) + list(BUILT_IN)


def matching(variables, root=None):
    """The first layout that fits a file's variables, or None."""
    return next((layout for layout in every(root) if layout.fits(variables)), None)
