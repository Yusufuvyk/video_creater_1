"""Reading MATLAB .mat recordings.

There is no single layout for EEG in a .mat file, so this guesses: the largest
2-D numeric array is the data, a scalar called something like `fs` is the
sampling rate, and a list of strings as long as the channel count are the
channel names. EEGLAB's `EEG.data` / `EEG.srate` / `EEG.chanlocs` falls out of
the same rules once nested structs are flattened.

When the guess fails the caller is handed the variable list and asks the user,
rather than picking something and being quietly wrong.

Pure scipy/numpy (+ h5py for v7.3 files) - no Qt.
"""
import json
from pathlib import Path
import re

import numpy as np

from ..core.sources import Recording

#: Field names that carry a sampling rate, in the order they are trusted.
FS_KEYS = ("fs", "srate", "Fs", "SamplingRate", "sampling_rate", "samplingRate",
           "sfreq", "Fsample", "sample_rate")

#: Field names that tend to hold channel labels.
NAME_KEYS = ("labels", "label", "chanlocs", "channels", "channel_names",
             "ch_names", "clab")

#: Below this a signal is almost certainly in volts rather than microvolts:
#: real EEG never peaks at a hundredth of a microvolt.
VOLT_THRESHOLD = 0.01

MAPPING_SUFFIX = ".mapping.json"


class MatError(Exception):
    """Base for anything that stops a .mat file being read."""


class MatVersionUnsupported(MatError):
    """A MATLAB v7.3 file (HDF5) and no h5py to read it with."""


class MatLayoutUnknown(MatError):
    """Auto-detection failed. `candidates` describes what the file holds."""

    def __init__(self, message, candidates):
        super().__init__(message)
        self.candidates = candidates


class Mapping:
    """How to read one file. Small enough to store beside it as JSON."""

    __slots__ = ("data_key", "fs", "fs_key", "transpose", "names_key", "scale",
                 "rows", "names", "preset", "parts", "onsets_key", "onset_unit",
                 "codes_key", "labels_key", "bad_key", "bad_window", "axes", "trial_start")

    def __init__(self, data_key, fs=None, fs_key=None, transpose=False,
                 names_key=None, scale=1.0, rows=None, names=None, preset=None,
                 parts=None, onsets_key=None, onset_unit=None, codes_key=None,
                 labels_key=None, bad_key=None, bad_window=None, axes=None, trial_start=None):
        self.data_key = data_key
        self.fs = fs
        self.fs_key = fs_key
        self.transpose = bool(transpose)
        self.names_key = names_key
        self.scale = float(scale)
        #: Which rows of the oriented array are EEG, as a 1-based range string
        #: like "1-64". Cho 2017 stores 64 EEG electrodes followed by 4 EMG in
        #: one array; analysing the EMG rows as brain signal is not a subtle
        #: error, but nothing in the file marks where the EEG stops.
        self.rows = rows or None
        #: Labels supplied by a montage rather than read from the file. Many
        #: datasets, Cho 2017 among them, carry no channel names at all.
        self.names = list(names) if names else None
        #: Which recognised dataset this came from, for display only.
        self.preset = preset
        #: Several arrays read as one recording: [[data_key, label, n_trials]].
        #: Each array is a run of equal trials laid end to end (Cho 2017 keeps
        #: left- and right-hand imagery apart); the parts are joined in time
        #: with a short gap of missing samples, and every trial becomes an event
        #: carrying its part's label. That is what lets left and right be
        #: compared in one session instead of two files that never meet.
        self.parts = [list(part) for part in parts] if parts else None
        #: Events kept in variables of their own, one value per event: when
        #: (`onset_unit`: SAMPLES_1, MATLAB's numbering from 1; SAMPLES_0; or
        #: SECONDS), which kind (`codes_key`, numbers or names) and what the
        #: numbers are called (`labels_key`, a list: code 1 is its first name).
        #: BNCI 2014-001 keeps them so, per run: data{4}.trial, data{4}.y.
        self.onsets_key = onsets_key or None
        self.onset_unit = onset_unit or None
        self.codes_key = codes_key or None
        self.labels_key = labels_key or None
        #: One value per event, true for a trial the dataset marks as bad; the
        #: span it marks is `bad_window` (s after the event, [from, to]), else
        #: up to the next event. Trials over it are left out, as MNE's BAD_.
        self.bad_key = bad_key or None
        self.bad_window = list(bad_window) if bad_window else None
        #: A signal already cut into trials (a 3-D array): what each of its
        #: axes is, in the order stored, one letter each: t (trials), c
        #: (channels), s (samples), e.g. "cst" for EEGLAB's channels x samples
        #: x trials. Each trial is then laid after the one before, apart, with
        #: one event; `codes_key` holds one label per trial. `trial_start`: the
        #: time of a trial's first sample from its event, s (-0.5: half a
        #: second before the cue).
        self.axes = axes or None
        self.trial_start = float(trial_start) if trial_start is not None else None

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, raw):
        return cls(**{k: raw.get(k) for k in cls.__slots__ if k in raw})


# ----------------------------------------------------------------------
# reading and flattening
# ----------------------------------------------------------------------


class Unread:
    """A MATLAB variable that is not read: a string array, a table, a function handle or another MATLAB object.
    MATLAB keeps their contents in an undocumented part of the file (MCOS) in v7 and v7.3 alike; they are named,
    with what to do in MATLAB, rather than left out in silence."""

    def __init__(self, kind):
        self.kind = str(kind or "object")

    def __repr__(self):
        return "Unread({})".format(self.kind)

    def advice(self):
        return {"string": "cellstr(x)", "table": "table2struct(x, 'ToScalar', true)",
                "categorical": "cellstr(x)", "datetime": "datenum(x)"}.get(
            self.kind, "convert it to numbers, char or a struct")


def _unread(value):
    """scipy's MATLAB objects (v7) as Unread, else None."""
    try:
        from scipy.io.matlab import MatlabFunction, MatlabOpaque
    except ImportError:
        return None
    if isinstance(value, MatlabFunction):
        return Unread("function_handle")
    if isinstance(value, MatlabOpaque):
        try:
            kind = value["_Class"].reshape(-1)[0] if "_Class" in (value.dtype.names or ()) else value[0][2]
        except (IndexError, KeyError, TypeError, ValueError):
            kind = "object"
        return Unread(kind.decode() if isinstance(kind, bytes) else kind)
    return None


def unread(variables):
    """{name: Unread} of the variables a file holds but GIOVA does not read."""
    return {k: v for k, v in variables.items() if isinstance(v, Unread)}


def _walk(value, prefix, out, depth=0):
    """Flatten nested MATLAB structs into 'EEG.data' style keys."""
    if depth > 4:
        return
    marker = _unread(value)
    if marker is not None:
        out[prefix] = marker
        return
    if hasattr(value, "_fieldnames"):
        for field in value._fieldnames:
            _walk(getattr(value, field), "{}.{}".format(prefix, field), out,
                  depth + 1)
        return
    if isinstance(value, np.ndarray) and value.dtype == object and value.size:
        if _holds_signal(value):
            # A cell (or struct) array whose elements carry signals: each one
            # its own entry, numbered from 1 as MATLAB does - data{4}.X.
            for number, item in enumerate(value.flat, 1):
                _walk(item, "{}{{{}}}".format(prefix, number), out, depth + 1)
            return
        # A struct array (chanlocs is one). Only the first element's fields
        # matter for naming; the labels are gathered separately.
        first = value.flat[0]
        if hasattr(first, "_fieldnames"):
            out[prefix] = value
            # Each field that holds one number or one text per element is also
            # a list of its own: EEG.event(i).latency is EEG.event.latency, so
            # events kept this way can be chosen like any other variable.
            for field in first._fieldnames:
                column = _column(value, field)
                if column is not None:
                    out.setdefault("{}.{}".format(prefix, field), column)
            return
    out[prefix] = value


def _column(array, field):
    """One field of every element of a struct array, as a 1-D array (numbers) or object array (texts); None when
    an element lacks it or holds more than one value."""
    values = []
    for item in array.flatten(order="F"):       # MATLAB's own order, as [s.field] gives it
        if not hasattr(item, field):
            return None
        value = getattr(item, field)
        if isinstance(value, str):
            values.append(value)
            continue
        number = np.asarray(value)
        if number.size != 1 or number.dtype.kind not in "fiub":
            return None
        values.append(float(number.reshape(-1)[0]))
    if all(isinstance(v, str) for v in values):
        column = np.empty(len(values), dtype=object)
        column[:] = values
        return column
    if any(isinstance(v, str) for v in values):
        return None
    return np.asarray(values, dtype=float)


def _signal(value):
    array = value if isinstance(value, np.ndarray) else None
    return array is not None and array.ndim == 2 and array.dtype.kind in "fiu" and array.size > 16


def _holds_signal(array):
    """True when an element of an object array is, or has a field that is, a 2-D numeric array."""
    for item in array.flat:
        if _signal(item):
            return True
        if hasattr(item, "_fieldnames") and any(_signal(getattr(item, f)) for f in item._fieldnames):
            return True
    return False


#: In a key, "every element": data{*}.X is data{1}.X, data{2}.X, ... read one
#: after another as the runs of one recording.
EVERY = "{*}"


def runs(variables, pattern):
    """[(number, key)] of the variables a key with {*} names, in MATLAB's order."""
    if EVERY not in str(pattern):
        return [(None, pattern)] if pattern in variables else []
    found = re.compile(re.escape(str(pattern)).replace(re.escape(EVERY), r"\{(\d+)\}") + "$")
    hits = [(int(m.group(1)), key) for key, m in ((k, found.match(k)) for k in variables) if m]
    return sorted(hits)


def run_key(pattern, number):
    """The key a {*} key gives for one run: data{*}.y for run 4 is data{4}.y."""
    return str(pattern).replace(EVERY, "{{{}}}".format(number)) if number is not None else pattern


def patterns(keys):
    """The {*} keys that name two or more numbered variables: data{*}.X for data{1}.X ... data{9}.X."""
    counts = {}
    for key in keys:
        general = re.sub(r"\{\d+\}", EVERY, key)
        if general != key:
            counts[general] = counts.get(general, 0) + 1
    return sorted(k for k, n in counts.items() if n > 1)


def read_variables(path):
    """Every variable in the file, with nested structs flattened. MATLAB v7.3 files (HDF5) are read with h5py
    into the same shapes and kinds scipy gives a v7 file, so everything after this reads both alike."""
    from scipy.io import loadmat   # here, not at the top: scipy.io costs 0.2 s at every start
    if is_v73(path):
        raw = read_v73(path)
    else:
        try:
            raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            raw = read_v73(path)      # an HDF5 file whose header did not say so

    out = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        _walk(value, key, out)
    return out


# ----------------------------------------------------------------------
# MATLAB v7.3: HDF5, read with h5py
# ----------------------------------------------------------------------
#
# MATLAB writes a v7.3 file as HDF5 with a 512-byte header. Each variable is a
# dataset or group whose MATLAB_class attribute says what it is. Arrays are
# stored in MATLAB's column-major order, so HDF5 gives every dimension reversed
# (a 3 x 1000 signal reads 1000 x 3). Text is UTF-16 code units. A cell array
# is a dataset of references into the #refs# group. A struct is a group of its
# fields (their order in MATLAB_fields); a struct array is a group whose fields
# are reference datasets without a MATLAB_class, one reference per element.
# An empty array is a dataset holding its dimensions, marked MATLAB_empty.
# These rules were checked against files MATLAB R2025b wrote (tests/test_mat_v73.py).
#
# The values come out as scipy's loadmat(squeeze_me=True, struct_as_record=False)
# gives them for a v7 file: dimensions of 1 dropped, a 1 x 1 value as a plain
# number or string, a cell array as an object array, a struct as an object with
# _fieldnames, a char matrix as an array of row strings. One difference cannot
# be helped: v7 may store whole numbers as small integer types, v7.3 keeps them
# double; every reader here takes numbers as float.

#: Nesting deeper than this is not EEG data; it stops a cycle of references.
V73_DEPTH = 64


def is_v73(path):
    """True when the file's header says MATLAB 7.3 (HDF5)."""
    try:
        with open(path, "rb") as handle:
            return handle.read(128).startswith(b"MATLAB 7.3")
    except OSError:
        return False


class MatStruct:
    """A MATLAB struct from a v7.3 file, read as scipy reads one from v7: its fields as attributes, named in
    `_fieldnames` in MATLAB's order."""

    def __init__(self, fields):
        self._fieldnames = list(fields)
        for name, value in fields.items():
            setattr(self, name, value)

    def __repr__(self):
        return "MatStruct({})".format(", ".join(self._fieldnames))


def read_v73(path):
    """{name: value} of a MATLAB v7.3 file, as scipy gives a v7 file's (see the notes above)."""
    try:
        import h5py
    except ImportError as exc:
        raise MatVersionUnsupported(
            "This is a MATLAB v7.3 file (HDF5), and h5py, which reads it, is not installed. In MATLAB, "
            "save it again with save(name, '-v7'), or install h5py.") from exc
    try:
        with h5py.File(str(path), "r") as handle:
            return {key: _v73_value(handle, handle[key], h5py, 0) for key in handle if not key.startswith("#")}
    except (OSError, KeyError, ValueError) as exc:
        raise MatError("Could not read {} as a MATLAB v7.3 file: {}".format(Path(path).name, exc)) from exc


def v73_text(path, names):
    """The first of `names` (HDF5 paths: "data", "EEG/data") that is a char variable in a v7.3 file, as text;
    None when none is, the file is not HDF5 or h5py is missing. Reads that variable only."""
    try:
        import h5py
        with h5py.File(str(path), "r") as handle:
            for name in names:
                item = handle.get(name)
                if isinstance(item, h5py.Dataset) and _v73_class(item) == "char":
                    return _v73_text(np.asarray(item[()]).T, item)
    except (ImportError, OSError, KeyError, ValueError):
        return None
    return None


def _v73_class(item):
    value = item.attrs.get("MATLAB_class", b"")
    return value.decode("ascii", "replace") if isinstance(value, bytes) else str(value)


def _v73_squeezed(array):
    """As scipy's squeeze_me: dimensions of 1 dropped, a single value as itself."""
    array = np.squeeze(array)
    if array.ndim == 0:
        item = array[()]
        return item.item() if isinstance(item, np.generic) else item
    return array


def _v73_objects(shape, items):
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out.reshape(shape)


def _v73_value(handle, item, h5py, depth):
    if depth > V73_DEPTH:
        raise MatError("'{}' is nested more than {} levels deep".format(item.name, V73_DEPTH))
    kind = _v73_class(item)
    if isinstance(item, h5py.Group):
        if "MATLAB_sparse" in item.attrs:
            from scipy.sparse import csc_matrix
            jc = np.asarray(item["jc"][()]) if "jc" in item else np.zeros(1, dtype=np.int64)
            data = np.asarray(item["data"][()]) if "data" in item else np.zeros(0)
            ir = np.asarray(item["ir"][()]) if "ir" in item else np.zeros(0, dtype=np.int64)
            return csc_matrix((data, ir, jc), shape=(int(item.attrs["MATLAB_sparse"]), len(jc) - 1))
        if kind != "struct":
            return Unread(kind)             # a MATLAB object (table, function handle ...): named, not read
        names = [b"".join(np.asarray(n).reshape(-1)).decode("utf-8", "replace")
                 for n in item.attrs.get("MATLAB_fields", [])] or list(item)
        names = [n for n in names if n in item]
        fields = {n: item[n] for n in names}
        array_fields = [f for f in fields.values() if isinstance(f, h5py.Dataset)
                        and h5py.check_dtype(ref=f.dtype) is not None and "MATLAB_class" not in f.attrs]
        if names and len(array_fields) == len(names):
            # a struct array: element k of each field is its k-th reference, MATLAB's shape reversed in HDF5
            refs = {n: np.asarray(f[()]).T for n, f in fields.items()}
            shape = next(iter(refs.values())).shape
            down = {n: r.flatten(order="F") for n, r in refs.items()}
            elements = [MatStruct({n: _v73_value(handle, handle[down[n][k]], h5py, depth + 1) for n in names})
                        for k in range(int(np.prod(shape)))]
            # MATLAB counts the elements down the columns: lay them out in that order
            return _v73_squeezed(_v73_objects(shape[::-1], elements).T)
        return MatStruct({n: _v73_value(handle, f, h5py, depth + 1) for n, f in fields.items()})
    if "MATLAB_empty" in item.attrs and int(np.asarray(item.attrs["MATLAB_empty"]).reshape(-1)[0]):
        if kind == "char":
            return np.zeros(0, dtype="<U1")
        if kind in ("cell", "struct"):
            return np.zeros(0, dtype=object)
        return np.zeros(0)
    if h5py.check_dtype(ref=item.dtype) is not None:
        refs = np.asarray(item[()]).T            # a cell array, in MATLAB's shape
        values = [_v73_value(handle, handle[ref], h5py, depth + 1) for ref in refs.flatten(order="F")]
        return _v73_squeezed(_v73_objects(refs.shape[::-1], values).T)
    data = item[()]
    if kind == "char":
        return _v73_text(np.asarray(data).T, item)
    if kind not in ("", "double", "single", "logical") and not kind.startswith(("int", "uint")):
        return Unread(kind)                     # string arrays and other MATLAB objects: named, not read
    data = np.asarray(data)
    if data.dtype.names and {"real", "imag"} <= set(data.dtype.names):
        data = data["real"] + 1j * data["imag"]
    return _v73_squeezed(data.T)


def _v73_text(codes, item):
    """A char array (in MATLAB's shape) as scipy gives it: one row a str, several an array of row strings."""
    codes = np.atleast_2d(codes)
    if codes.ndim > 2:
        return None
    wide = codes.dtype.itemsize == 2 or int(np.asarray(item.attrs.get("MATLAB_int_decode", 2)).reshape(-1)[0]) == 2

    def row(values):
        if wide:
            return np.asarray(values, dtype="<u2").tobytes().decode("utf-16-le", "replace")
        return "".join(chr(int(c)) for c in values)
    rows = [row(r) for r in codes]
    if len(rows) == 1:
        return rows[0]
    return np.array(rows)


def describe_variables(variables):
    """[(key, shape, kind)] for showing the user what the file holds."""
    rows = []
    for key, value in sorted(variables.items()):
        if isinstance(value, Unread):
            rows.append((key, (), "MATLAB {}: not read".format(value.kind)))
            continue
        array = np.asarray(value)
        if array.dtype.kind in "fiu" and array.ndim == 2:
            kind = "numeric 2-D"
        elif array.dtype.kind in "fiu" and array.size == 1:
            kind = "scalar"
        elif array.dtype.kind in "fiu":
            kind = "numeric {}-D".format(array.ndim)
        else:
            kind = "text/struct"
        rows.append((key, tuple(array.shape), kind))
    return rows


# ----------------------------------------------------------------------
# guessing
# ----------------------------------------------------------------------


def _numeric_2d(variables):
    out = {}
    for key, value in variables.items():
        array = np.asarray(value)
        if array.ndim == 2 and array.dtype.kind in "fiu" and array.size > 16:
            out[key] = array
    return out


def _find_fs(variables):
    for wanted in FS_KEYS:
        for key, value in variables.items():
            if key.split(".")[-1] == wanted:
                array = np.asarray(value)
                if array.size == 1 and array.dtype.kind in "fiu":
                    rate = float(array.reshape(-1)[0])
                    if 1.0 < rate <= 100000.0:
                        return key, rate
    return None, None


def _labels_from(value, n_channels):
    """Pull channel labels out of a cell array, char array or struct array."""
    array = np.asarray(value)
    if array.dtype == object and array.size and hasattr(array.flat[0], "_fieldnames"):
        for field in ("labels", "label", "name"):
            if field in array.flat[0]._fieldnames:
                names = [str(getattr(item, field)) for item in array.flat]
                return names if len(names) == n_channels else None
        return None
    if array.dtype.kind in "OUS":
        names = [str(item) for item in array.reshape(-1)]
        return names if len(names) == n_channels else None
    return None


def _find_names(variables, n_channels):
    for key, value in variables.items():
        if key.split(".")[-1] in NAME_KEYS:
            names = _labels_from(value, n_channels)
            if names:
                return key, names
    for key, value in variables.items():  # any list of the right length
        names = _labels_from(value, n_channels)
        if names:
            return key, names
    return None, None


def parse_rows(text, n_rows):
    """A 1-based range string like "1-64" or "1-32,65" as row indices.

    Returns None when the text is empty or covers everything, so callers can
    tell "all rows" from "a selection that happens to be all of them" and skip
    the slice entirely. Out-of-range entries are dropped rather than raising:
    the field is edited live in a dialog and passes through half-typed states.
    """
    if not text:
        return None
    wanted = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        try:
            if "-" in part[1:]:
                low, high = part.split("-", 1)
                wanted.extend(range(int(low) - 1, int(high)))
            else:
                wanted.append(int(part) - 1)
        except ValueError as exc:
            raise MatError("Invalid channel range: {}".format(text)) from exc
    kept = [i for i in dict.fromkeys(wanted) if 0 <= i < int(n_rows)]
    if not kept or len(kept) != len(set(wanted)):
        raise MatError("Channel range must contain valid rows between 1 and {}".format(n_rows))
    if len(kept) == int(n_rows):
        return None
    return kept


def guess_scale(data):
    """1e6 when the data looks like volts, else 1. EEG in volts is tiny."""
    peak = float(np.nanmax(np.abs(data))) if data.size else 0.0
    return 1e6 if 0 < peak < VOLT_THRESHOLD else 1.0


def autodetect(variables):
    """Best guess at how to read the file, or raise MatLayoutUnknown."""
    candidates = _numeric_2d(variables)
    # A signal inside a cell array (data{4}.X) is a guess only when the file has
    # no other: a file that read before reads as it did.
    plain = {k: v for k, v in candidates.items() if "{" not in k}
    candidates = plain or candidates
    if not candidates:
        raise MatLayoutUnknown("No 2-D numeric array found.",
                               describe_variables(variables))

    data_key = max(candidates, key=lambda k: candidates[k].size)
    array = candidates[data_key]

    # Several arrays of the same shape means several conditions in one file
    # (Cho 2017 holds imagery_left, imagery_right, movement_* and rest). Picking
    # the first silently would analyse the wrong condition, so ask instead.
    rivals = [key for key, value in candidates.items()
              if key != data_key and value.shape == array.shape]
    if rivals:
        raise MatLayoutUnknown(
            "This file holds {} arrays of the same shape ({}), so which one to "
            "analyse is a choice, not a guess: {}.".format(
                len(rivals) + 1, tuple(array.shape),
                ", ".join(sorted([data_key] + rivals))),
            describe_variables(variables))

    # EEG is long in time and narrow in channels; orient to (channels, samples).
    transpose = array.shape[0] > array.shape[1]
    oriented = array.T if transpose else array

    fs_key, fs = _find_fs(variables)
    if fs is None:
        raise MatLayoutUnknown(
            "Found data in '{}' {} but no sampling rate. It matters: without "
            "it every duration and frequency would be wrong.".format(
                data_key, tuple(array.shape)),
            describe_variables(variables))

    names_key, _ = _find_names(variables, oriented.shape[0])
    return Mapping(data_key, fs=fs, fs_key=fs_key, transpose=transpose,
                   names_key=names_key, scale=guess_scale(oriented))


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------


def mapping_path(path):
    return Path(path).with_suffix("").with_name(Path(path).stem + MAPPING_SUFFIX)


def save_mapping(path, mapping):
    mapping_path(path).write_text(json.dumps(mapping.to_dict(), indent=2),
                                  encoding="utf-8")


def load_mapping(path):
    sidecar = mapping_path(path)
    if not sidecar.is_file():
        return None
    try:
        return Mapping.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
    except (ValueError, TypeError):
        return None  # a corrupt sidecar should not block the file


#: Missing samples between joined parts, so zero-phase filters never run
#: across the join (each part has its own electrode offset).
PART_GAP_SECONDS = 1.0


def _oriented(variables, key, mapping):
    if EVERY in str(key):
        found = runs(variables, key)
        if not found:
            raise MatError("No variable matches '{}'".format(key))
        key = found[0][1]
    if key not in variables:
        raise MatError("'{}' is not in this file".format(key))
    data = np.asarray(variables[key], dtype=float)
    if data.ndim != 2:
        raise MatError("'{}' is not a 2-D array".format(key))
    if mapping.transpose:
        data = data.T
    rows = parse_rows(mapping.rows, data.shape[0])
    if rows is not None:
        data = data[rows]
    return data * mapping.scale


def join_parts(variables, mapping, fs):
    """(data, EventTable) for a multi-part mapping."""
    from ..core.epochs import EventTable
    blocks, onsets, durations, labels = [], [], [], []
    gap = int(round(PART_GAP_SECONDS * fs))
    position = 0
    for key, label, n_trials in mapping.parts:
        part = _oriented(variables, key, mapping)
        if blocks and part.shape[0] != blocks[0].shape[0]:
            raise MatError("'{}' has {} rows, the other parts {}".format(key, part.shape[0], blocks[0].shape[0]))
        if blocks:
            blocks.append(np.full((part.shape[0], gap), np.nan))
            position += gap
        n_trials = int(n_trials or 0)
        if n_trials >= 1:
            length = part.shape[1] / n_trials
            for k in range(n_trials):
                onsets.append((position + k * length) / fs)
                durations.append(length / fs)
                labels.append(str(label))
        blocks.append(part)
        position += part.shape[1]
    return np.hstack(blocks), EventTable(onsets, durations, labels)


#: How event times are counted (Mapping.onset_unit).
SAMPLES_1, SAMPLES_0, SECONDS = "samples from 1", "samples from 0", "seconds"
ONSET_UNITS = (SAMPLES_1, SAMPLES_0, SECONDS)


def _values(variables, key):
    """A variable as a flat array; empty when it is missing or empty (a run without trials)."""
    if key not in variables:
        return np.zeros(0)
    return np.atleast_1d(np.asarray(variables[key], dtype=object)).reshape(-1)


def _names(variables, key):
    """A list of names (a cell array of strings, a char array), or []."""
    if not key or key not in variables:
        return []
    value = np.asarray(variables[key])
    if value.dtype.kind in "OUS":
        return [str(v).strip() for v in value.reshape(-1)]
    return []


def field_events(variables, mapping, fs, number=None, n_samples=None):
    """(onsets s, labels, bad spans) of one run's events kept in variables, its time counted from its first sample."""
    onsets = _values(variables, run_key(mapping.onsets_key, number))
    if not onsets.size:
        return np.zeros(0), [], []
    try:
        onsets = onsets.astype(float)
    except (TypeError, ValueError) as exc:
        raise MatError("'{}' does not hold numbers: it cannot be the event times".format(
            run_key(mapping.onsets_key, number))) from exc
    unit = mapping.onset_unit or SAMPLES_1
    seconds = (onsets - 1) / fs if unit == SAMPLES_1 else onsets / fs if unit == SAMPLES_0 else onsets
    if n_samples is not None and seconds.size and (seconds.min() < 0 or seconds.max() > n_samples / fs):
        raise MatError("The event times in '{}' run past the signal ({:.1f} s against {:.1f} s): are they in {} "
                       "rather than {}?".format(run_key(mapping.onsets_key, number), float(seconds.max()),
                                                n_samples / fs, "seconds" if unit != SECONDS else "samples", unit))
    if mapping.codes_key:
        codes = _values(variables, run_key(mapping.codes_key, number))
        if codes.size != onsets.size:
            raise MatError("'{}' has {} values and '{}' {}: each event needs one time and one code".format(
                run_key(mapping.onsets_key, number), onsets.size, run_key(mapping.codes_key, number), codes.size))
    else:
        codes = np.array(["event"] * onsets.size, dtype=object)
    names = _names(variables, run_key(mapping.labels_key, number)) if mapping.labels_key else []
    labels = [_label(code, names) for code in codes]
    bad = []
    if mapping.bad_key:
        marks = _values(variables, run_key(mapping.bad_key, number))
        if marks.size and marks.size != onsets.size:
            raise MatError("'{}' has {} values for {} events: it must mark each event".format(
                run_key(mapping.bad_key, number), marks.size, onsets.size))
        order = np.argsort(seconds)
        end = n_samples / fs if n_samples is not None else float(seconds.max()) + 1.0
        label = "BAD_" + str(mapping.bad_key).split(".")[-1]
        for rank, i in enumerate(order):
            if marks.size and _truthy(marks[i]):
                if mapping.bad_window:
                    first, last = seconds[i] + mapping.bad_window[0], seconds[i] + mapping.bad_window[1]
                else:
                    first = seconds[i]
                    last = seconds[order[rank + 1]] if rank + 1 < len(order) else end
                bad.append((float(first), float(last), label))
    return seconds, labels, bad


def _truthy(value):
    try:
        return bool(float(value))
    except (TypeError, ValueError):
        return bool(value)


def _label(code, names):
    """An event's name: code n is the n-th of `names` (MATLAB counts from 1), else the code itself."""
    try:
        number = float(code)
    except (TypeError, ValueError):
        return str(code).strip()
    if number.is_integer() and 1 <= number <= len(names):
        return names[int(number) - 1]
    return "{:g}".format(number)


#: Missing samples between trials laid one after another, so no filter runs from one into the next.
TRIAL_GAP_SECONDS = 0.2


def guess_axes(shape, n_names=None):
    """Which axis of a 3-D signal is trials, channels and samples ("tcs" letters in the stored order): samples
    the longest, channels the one as long as the names when known, else the shorter of the other two. A guess the
    dialog shows for the user to confirm."""
    order = sorted(range(3), key=lambda i: shape[i])
    samples = order[2]
    rest = [i for i in range(3) if i != samples]
    if n_names and [shape[i] for i in rest].count(n_names) == 1:
        channels = next(i for i in rest if shape[i] == n_names)
    else:
        channels = min(rest, key=lambda i: shape[i])
    letters = ["t", "t", "t"]
    letters[samples], letters[channels] = "s", "c"
    return "".join(letters)


def as_trials(array, axes):
    """A 3-D signal as (trials, channels, samples), from its axes letters."""
    if sorted(str(axes)) != ["c", "s", "t"]:
        raise MatError("Say which axis is trials, channels and samples")
    return np.transpose(np.asarray(array), [axes.index(r) for r in "tcs"])


def channels_of(array, mapping):
    """How many channels a signal has as the mapping reads it, before rows are left out."""
    array = np.asarray(array)
    if array.ndim == 3 and mapping.axes:
        return array.shape[mapping.axes.index("c")]
    return array.shape[1] if mapping.transpose else array.shape[0]


def is_trials(variables, mapping):
    """True when the mapping's signal is 3-D: trials already cut."""
    found = runs(variables, mapping.data_key) if mapping.data_key else []
    return bool(found) and np.asarray(variables[found[0][1]]).ndim == 3


def join_trials(variables, mapping, fs):
    """(data, EventTable, bad spans) of a signal already cut into trials (every run of a {*} key in turn): the
    trials one after another, a short gap of missing samples between, one event per trial at its time zero."""
    from ..core.epochs import EventTable
    gap = max(1, int(round(TRIAL_GAP_SECONDS * fs)))
    start = float(mapping.trial_start or 0.0)
    blocks, onsets, durations, labels, bad = [], [], [], [], []
    position = 0
    for number, key in runs(variables, mapping.data_key):
        trials = as_trials(np.asarray(variables[key], dtype=float), mapping.axes)
        rows = parse_rows(mapping.rows, trials.shape[1])
        if rows is not None:
            trials = trials[:, rows]
        trials = trials * mapping.scale
        n = trials.shape[2]
        if mapping.codes_key:
            codes = _values(variables, run_key(mapping.codes_key, number))
            if codes.size != trials.shape[0]:
                raise MatError("'{}' has {} values for {} trials: it must label each trial".format(
                    run_key(mapping.codes_key, number), codes.size, trials.shape[0]))
        else:
            codes = np.array(["trial"] * trials.shape[0], dtype=object)
        names = _names(variables, run_key(mapping.labels_key, number)) if mapping.labels_key else []
        marks = _values(variables, run_key(mapping.bad_key, number)) if mapping.bad_key else np.zeros(0)
        if marks.size and marks.size != trials.shape[0]:
            raise MatError("'{}' has {} values for {} trials: it must mark each trial".format(
                run_key(mapping.bad_key, number), marks.size, trials.shape[0]))
        for k, trial in enumerate(trials):
            if blocks and blocks[0].shape[0] != trial.shape[0]:
                raise MatError("'{}' has {} channels, the trials before it {}".format(
                    key, trial.shape[0], blocks[0].shape[0]))
            if blocks:
                blocks.append(np.full((trial.shape[0], gap), np.nan))
                position += gap
            onsets.append(position / fs - start)
            durations.append(n / fs + start)
            labels.append(_label(codes[k], names))
            if marks.size and _truthy(marks[k]):
                bad.append((position / fs, (position + n) / fs, "BAD_" + str(mapping.bad_key).split(".")[-1]))
            blocks.append(trial)
            position += n
    if not blocks:
        raise MatError("No variable matches '{}'".format(mapping.data_key))
    return np.hstack(blocks), EventTable(onsets, np.maximum(durations, 0.0), labels), bad


def join_runs(variables, mapping, fs):
    """(data, EventTable, bad spans, runs left out) for a key with {*}: every run in order, joined in time with a
    short gap of missing samples. With events in variables, a run without any (a calibration run) is left out."""
    from ..core.epochs import EventTable
    found = runs(variables, mapping.data_key)
    if not found:
        raise MatError("No variable matches '{}'".format(mapping.data_key))
    gap = int(round(PART_GAP_SECONDS * fs))
    blocks, onsets, labels, bad, skipped = [], [], [], [], []
    position = 0
    for number, key in found:
        part = _oriented(variables, key, mapping)
        if mapping.onsets_key:
            seconds, names, marked = field_events(variables, mapping, fs, number, part.shape[1])
            if not seconds.size:
                skipped.append(number)
                continue
        else:
            seconds, names, marked = np.zeros(0), [], []
        if blocks and part.shape[0] != blocks[0].shape[0]:
            raise MatError("'{}' has {} channels, the runs before it {}".format(key, part.shape[0], blocks[0].shape[0]))
        if blocks:
            blocks.append(np.full((part.shape[0], gap), np.nan))
            position += gap
        offset = position / fs
        onsets.extend(seconds + offset)
        labels.extend(names)
        bad.extend((a + offset, b + offset, label) for a, b, label in marked)
        blocks.append(part)
        position += part.shape[1]
    if not blocks:
        raise MatError("None of the {} runs of '{}' has events in '{}'".format(
            len(found), mapping.data_key, mapping.onsets_key))
    events = EventTable(onsets, None, labels) if mapping.onsets_key else None
    return np.hstack(blocks), events, bad, skipped


def _rate(variables, mapping):
    fs = mapping.fs
    if fs is None and mapping.fs_key:
        found = runs(variables, mapping.fs_key)
        if found:
            fs = float(np.asarray(variables[found[0][1]]).reshape(-1)[0])
    return fs


def apply_mapping(variables, mapping, path=None):
    """Build a Recording from a file's variables and a chosen mapping."""
    fs = _rate(variables, mapping)
    if not fs or fs <= 0:
        raise MatError("no usable sampling rate in the mapping")

    events, bad, skipped = None, [], []
    if mapping.parts:
        data, events = join_parts(variables, mapping, fs)
    elif is_trials(variables, mapping):
        data, events, bad = join_trials(variables, mapping, fs)
    elif EVERY in str(mapping.data_key):
        data, events, bad, skipped = join_runs(variables, mapping, fs)
    else:
        data = _oriented(variables, mapping.data_key, mapping)
        if mapping.onsets_key:
            from ..core.epochs import EventTable
            seconds, labels, bad = field_events(variables, mapping, fs, None, data.shape[1])
            events = EventTable(seconds, None, labels)

    # An explicit montage wins over a variable in the file: it is the more
    # deliberate statement, and it is the only source there is for a dataset
    # that stores no labels of its own.
    names = None
    if mapping.names:
        names = list(mapping.names)[:data.shape[0]]
        if len(names) != data.shape[0]:
            names = None  # the montage does not fit this many rows; stay honest
    if names is None and mapping.names_key and mapping.names_key in variables:
        names = _labels_from(variables[mapping.names_key], data.shape[0])

    recording = Recording(data, fs, names, None, path, events=events)
    recording.unit_note = unit_note(mapping)
    recording.bad_spans = bad
    recording.reading = reading_notes(variables, mapping, recording, skipped)
    return recording


def _numbers(values):
    return ", ".join(str(v) for v in values)


def reading_notes(variables, mapping, recording, skipped=()):
    """What `apply_mapping` did with a file, in words: the signal and any runs or trials joined or left out, the
    channels kept, where names, events and bad marks came from, the units, what was not read."""
    from ..core.montages import guess_montage
    notes = []
    found = runs(variables, mapping.data_key) if mapping.data_key else []
    words = {"t": "trials", "c": "channels", "s": "samples"}
    n_events = len(recording.events) if recording.events is not None else 0
    if mapping.parts:
        notes.append("Signal: {} joined in time, {:g} s of missing samples between them; each trial is an event "
                     "named by its part ({}).".format(", ".join(p[0] for p in mapping.parts), PART_GAP_SECONDS,
                                                      ", ".join("{} x{}".format(p[1], int(p[2] or 0))
                                                                for p in mapping.parts)))
    elif is_trials(variables, mapping):
        notes.append("Signal: {} holds trials already cut ({}); {} trials laid one after another, {:g} s of "
                     "missing samples between them, each its own event {:g} s after its first sample.".format(
                         mapping.data_key, " x ".join(words[r] for r in mapping.axes), n_events,
                         TRIAL_GAP_SECONDS, -float(mapping.trial_start or 0.0)))
    elif EVERY in str(mapping.data_key):
        used = [n for n, _ in found if n not in set(skipped)]
        notes.append("Signal: runs {} of {} read one after another, {:g} s of missing samples between them.".format(
            _numbers(used), mapping.data_key, PART_GAP_SECONDS))
        if skipped:
            notes.append("Left out: run(s) {} of {}, which hold no events (calibration or rest).".format(
                _numbers(skipped), mapping.data_key))
    else:
        notes.append("Signal: {}{}.".format(mapping.data_key, ", its rows taken as samples" if mapping.transpose
                                            else ""))
    if mapping.rows and found:
        total = channels_of(variables[found[0][1]], mapping)
        if total > recording.n_channels:
            notes.append("Channels: rows {} kept; {} of {} left out.".format(
                mapping.rows, total - recording.n_channels, total))
    if mapping.names and recording.channel_names[:1] == list(mapping.names)[:1]:
        montage = guess_montage(mapping.names, len(mapping.names))
        notes.append("Channel names: {}.".format("the montage '{}'".format(montage) if montage else
                                                 "as the layout lists them"))
    elif mapping.names_key:
        notes.append("Channel names: from '{}'.".format(mapping.names_key))
    else:
        notes.append("Channel names: none in the file; called CH1 ... CH{}.".format(recording.n_channels))
    if mapping.onsets_key and n_events:
        notes.append("Events: {} from '{}' ({}){}{}.".format(
            n_events, mapping.onsets_key, mapping.onset_unit or SAMPLES_1,
            ", their kinds from '{}'".format(mapping.codes_key) if mapping.codes_key else "",
            ", named by '{}'".format(mapping.labels_key) if mapping.labels_key else ""))
    elif is_trials(variables, mapping) and mapping.codes_key:
        notes.append("Trial labels: from '{}'{}.".format(mapping.codes_key, ", named by '{}'".format(
            mapping.labels_key) if mapping.labels_key else ""))
    if mapping.bad_key:
        span = ("the whole trial" if is_trials(variables, mapping) else
                "{:g} to {:g} s after the trial's start".format(*mapping.bad_window) if mapping.bad_window else
                "up to the next event")
        notes.append("Marked bad in the file: {} trial(s) in '{}' ({} marked); trials over them are left "
                     "out.".format(len(recording.bad_spans), mapping.bad_key, span))
    notes.append("Units: {}.".format(recording.unit_note))
    skipped_vars = unread(variables)
    if skipped_vars:
        notes.append("Not read: {} (MATLAB objects GIOVA cannot open).".format(
            ", ".join("{} ({})".format(k, v.kind) for k, v in sorted(skipped_vars.items()))))
    return notes


def unit_note(mapping):
    """How a mapping turns a .mat file's numbers into µV, in words."""
    if mapping.preset:
        return "{} preset: the stored numbers × {:g} µV".format(mapping.preset, mapping.scale)
    if mapping.scale == 1.0:
        return "MATLAB file: the numbers taken as µV (× 1)"
    return "MATLAB file: the numbers × {:g} to µV (its layout's scale)".format(mapping.scale)


def load_mat(path, mapping=None):
    """Open a .mat file. Uses the saved sidecar mapping, then auto-detection."""
    path = Path(path)
    variables = read_variables(path)
    origin = "the layout given"
    if mapping is None:
        mapping, origin = mapping_origin(path, variables)  # may raise MatLayoutUnknown
    recording = apply_mapping(variables, mapping, path)
    recording.reading.insert(0, "Read with {}.".format(origin))
    return recording


def mapping_for(path, variables, root=None):
    """How to read a file: the layout chosen for it (its .mapping.json), else a saved or built-in layout for
    files laid out like it (formats.layouts), else a guess. MatLayoutUnknown when none of them works."""
    return mapping_origin(path, variables, root)[0]


def mapping_origin(path, variables, root=None):
    """(mapping, where it came from in words), as mapping_for chooses it."""
    from . import layouts
    mapping = load_mapping(path)
    if mapping is not None:
        return mapping, "the layout chosen for this file ({})".format(mapping_path(path).name)
    layout = layouts.matching(variables, root)
    if layout is not None:
        where = "built into GIOVA" if layout.path is None else "kept in Data/layouts/" + layout.path.name
        return layout.mapping, "the layout '{}' ({}), for files holding {}".format(
            layout.name, where, ", ".join(layout.signature))
    mapping = autodetect(variables)
    return mapping, "a guess: the largest 2-D array, '{}'{}".format(
        mapping.data_key, ", its rate from '{}'".format(mapping.fs_key) if mapping.fs_key else "")


# ----------------------------------------------------------------------
# preview
# ----------------------------------------------------------------------

#: Seconds sampled from the middle of the recording to judge a mapping by.
PREVIEW_SECONDS = 10.0


def preview(variables, mapping):
    """One line describing what a mapping would produce. Cheap to recompute.

    The amplitude quoted is *band-limited*, and that is the whole point. Raw
    EEG is dominated by a DC electrode offset and slow drift - Cho 2017's is
    tens of thousands of microvolts - so a raw peak says nothing about whether
    the units are right. Passed through the same 1-45 Hz band the analysis uses,
    real EEG lands at roughly 5-50 uV rms, and a scale factor that is out by 32
    becomes obvious at a glance instead of silently wrong.

    Only a few seconds from the middle are used, so this can run on every
    keystroke in a dialog without stalling it.
    """
    key = mapping.data_key
    if EVERY in str(key):
        found = runs(variables, key)
        key = found[0][1] if found else key
    try:
        array = np.asarray(variables[key])
    except KeyError:
        return "'{}' is not in this file".format(mapping.data_key)
    if array.ndim == 3:
        # trials already cut: the first few, one after another, judged like a signal
        try:
            trials = as_trials(array, mapping.axes)
        except MatError as exc:
            return str(exc)
        shown = np.concatenate(list(trials[:max(1, min(len(trials), 10))]), axis=1)
        flat = Mapping(mapping.data_key, fs=_rate(variables, mapping), rows=mapping.rows, scale=mapping.scale)
        return "{} trials  |  ".format(trials.shape[0]) + _single_preview(variables, flat, shown)
    if mapping.parts:
        if any(key not in variables for key, _, _ in mapping.parts):
            return "a part of this preset is missing from the file"
        counts = ", ".join("{} x{}".format(label, int(n or 0)) for _, label, n in mapping.parts)
        prefix = "{} parts joined, {} trials ({})  |  ".format(
            len(mapping.parts), sum(int(n or 0) for _, _, n in mapping.parts), counts)
        return prefix + _single_preview(variables, mapping, array)
    return _single_preview(variables, mapping, array)


def _single_preview(variables, mapping, array):
    if array.ndim != 2:
        return "{} is {}-D; pick a 2-D array".format(mapping.data_key, array.ndim)

    fs = float(mapping.fs or 0.0)
    if fs <= 0:
        return "no sampling rate yet"

    # Slice before anything else: these arrays run to hundreds of megabytes.
    n_samples = array.shape[0] if mapping.transpose else array.shape[1]
    width = min(int(PREVIEW_SECONDS * fs), int(n_samples))
    start = max((int(n_samples) - width) // 2, 0)
    block = (array[start:start + width, :].T if mapping.transpose
             else array[:, start:start + width])

    block = np.asarray(block, dtype=float)
    rows = parse_rows(mapping.rows, block.shape[0])
    if rows is not None:
        block = block[rows]
    if block.size == 0:
        return "that row range selects nothing"
    block = (block - block.mean(axis=1, keepdims=True)) * mapping.scale

    return "{} ch x {:.1f} s @ {:g} Hz  |  {}".format(
        block.shape[0], n_samples / fs, fs, _amplitude_note(block, fs))


def _amplitude_note(block, fs):
    """"in band: rms N uV" plus a verdict, or why no verdict is possible."""
    from ..core.processing import apply_filter, design_filter

    high = min(45.0, fs / 2.0 - 1.0)
    if high <= 1.0 or block.shape[1] < 32:
        return "too short to judge the amplitude"
    try:
        filtered = apply_filter(design_filter(fs, low=1.0, high=high, order=4),
                                block)
    except Exception:
        return "rms {:,.0f} uV unfiltered".format(block.std())

    rms = float(filtered.std())
    return "in band: rms {:,.1f} uV  ({})".format(rms, _verdict(rms))


def _verdict(rms):
    """Whether a band-limited rms looks like EEG. The scale check, in words."""
    if rms < 0.5:
        return "far too small - are these volts? try a larger unit scale"
    if rms > 200.0:
        return "far too large - are these A/D counts? try a smaller unit scale"
    if rms > 80.0:
        return "high for EEG - check the unit scale"
    return "plausible EEG"


# ----------------------------------------------------------------------
# guidance: what a layout would give, and what to try
# ----------------------------------------------------------------------

#: Field names that tend to hold, beside a signal, its events' times, their codes, the codes' names, the trials
#: marked bad, and the sampling rate (BNCI: trial, y, classes, artifacts, fs).
ONSET_NAMES = ("trial", "trials", "pos", "onset", "onsets", "latency", "event_times", "times", "time")
CODE_NAMES = ("y", "type", "types", "code", "codes", "label", "labels", "class", "event")
LABEL_NAMES = ("classes", "class_names", "classnames", "names", "event_names")
BAD_NAMES = ("artifacts", "artifact", "artefacts", "bad", "reject", "rejected")

#: (state, text) states of a checklist line.
READY, MISSING, OPTIONAL = "ready", "missing", "optional"


def suggest(variables, data_key):
    """{slot: key} for the fields beside `data_key` that look like its events, their names, its bad trials and its
    rate: what a dialog fills in for the user to check, never applied on its own."""
    prefix = str(data_key).rsplit(".", 1)[0] + "." if "." in str(data_key) else ""
    probe = data_key
    if EVERY in str(data_key):
        found = runs(variables, data_key)
        probe = found[0][1] if found else data_key
    probe_prefix = str(probe).rsplit(".", 1)[0] + "." if "." in str(probe) else ""
    fields = {k[len(probe_prefix):]: k for k in variables if probe_prefix and k.startswith(probe_prefix)} \
        if probe_prefix else {k: k for k in variables if "." not in k}

    def pick(names, length=None):
        """The first field named like one of `names` (its own name, after any struct it is in: event.latency is
        latency), events' own (event, marker, trigger) before others, of `length` values when one is given."""
        found = []
        for rank, name in enumerate(names):
            for field in fields:
                if field.split(".")[-1].lower() != name.lower():
                    continue
                value = np.asarray(variables[fields[field]], dtype=object)
                if value.size and hasattr(value.reshape(-1)[0], "_fieldnames"):
                    continue                    # a struct array itself, not one of its fields
                if length is not None and value.size != length:
                    continue
                eventish = any(w in field.lower() for w in ("event", "mrk", "marker", "trig"))
                found.append((not eventish, rank, field))
        return prefix + min(found)[2] if found else None
    onsets = pick(ONSET_NAMES)
    # codes and bad marks go with the times only when they are as many (runs: each run is checked when read)
    length = np.asarray(variables[onsets], dtype=object).size if onsets in variables else None
    out = {"onsets_key": onsets, "codes_key": pick(CODE_NAMES, length), "labels_key": pick(LABEL_NAMES),
           "bad_key": pick(BAD_NAMES, length), "fs_key": pick(FS_KEYS)}
    if out["codes_key"] == out["labels_key"]:
        out["labels_key"] = None
    return {k: v for k, v in out.items() if v}


def checklist(variables, mapping):
    """[(state, text)]: what this layout gives - the signal, its rate, events, bad trials, names - each line saying
    what is still missing and what to try. Cheap: it reads shapes and the small event variables only."""
    lines = _checklist(variables, mapping)
    skipped = unread(variables)
    if skipped:
        lines.append((OPTIONAL, "Not read: {}. MATLAB keeps these objects in a form GIOVA cannot open; if one "
                                "is needed, in MATLAB use {} and save again.".format(
                                    ", ".join("{} (MATLAB {})".format(k, v.kind) for k, v in sorted(skipped.items())),
                                    "; ".join(sorted({"{} for a {}".format(v.advice(), v.kind)
                                                      for v in skipped.values()})))))
    return lines


def _checklist(variables, mapping):
    lines = []
    found = runs(variables, mapping.data_key) if mapping.data_key else []
    if not found:
        lines.append((MISSING, "Signal: choose the variable that holds it, usually the largest 2-D array in the "
                               "list above (channels x samples, or samples x channels). Several numbered ones "
                               "(data{1}.X, data{2}.X ...) are runs: choose the one with {*} to read them all."))
        return lines
    if np.asarray(variables[found[0][1]]).ndim == 3:
        return _trial_checklist(variables, mapping, found)
    shapes = []
    for _, key in found:
        array = np.asarray(variables[key])
        if array.ndim != 2:
            lines.append((MISSING, "Signal: '{}' is {}-D; choose a 2-D array.".format(key, array.ndim)))
            return lines
        shapes.append(array.shape[::-1] if mapping.transpose else array.shape)
    channels, samples = shapes[0][0], sum(shape[1] for shape in shapes)
    fs = _rate(variables, mapping)
    if channels > samples / max(len(shapes), 1):
        lines.append((MISSING, "Signal: {} channels and {} samples looks the wrong way round: tick 'Rows are "
                               "samples, columns are channels'.".format(channels, samples)))
    elif len(shapes) > 1:
        lines.append((READY, "Signal: {}, {} runs read one after another, {} channels.".format(
            mapping.data_key, len(shapes), channels)))
    else:
        lines.append((READY, "Signal: {}, {} channels.".format(found[0][1], channels)))
    if not fs or fs <= 0:
        lines.append((MISSING, "Sampling rate: type it (often in the paper or a variable called fs or srate)."))
        return lines
    lines.append((READY, "Sampling rate: {:g} Hz{}, {:.1f} s of signal.".format(
        fs, " (from {})".format(mapping.fs_key) if mapping.fs_key and not mapping.fs else "", samples / fs)))
    if not mapping.onsets_key:
        hint = suggest(variables, mapping.data_key).get("onsets_key")
        lines.append((OPTIONAL, "Events: none chosen. Trials can then be cut every few seconds or typed in a "
                                "table. {}".format("'{}' looks like event times: choose it under Event "
                                                   "times.".format(hint) if hint else
                                                   "If a variable holds the event times (trial, pos, onset, "
                                                   "latency ...), choose it under Event times.")))
    else:
        try:
            labels, marked, skipped = [], 0, 0
            for number, key in found:
                n = (np.asarray(variables[key]).shape[0] if mapping.transpose else np.asarray(variables[key]).shape[1])
                seconds, names, bad = field_events(variables, mapping, fs, number if EVERY in str(
                    mapping.data_key) else None, n)
                if not seconds.size and EVERY in str(mapping.data_key):
                    skipped += 1
                labels += names
                marked += len(bad)
        except MatError as exc:
            lines.append((MISSING, "Events: {}".format(exc)))
            return lines
        if not labels:
            lines.append((MISSING, "Events: '{}' is empty: choose another variable for the event times.".format(
                mapping.onsets_key)))
        else:
            counts = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            named = ", ".join("{} x{}".format(k, v) for k, v in list(counts.items())[:6])
            plain = all(k.replace(".", "", 1).isdigit() for k in counts)
            lines.append((READY, "Events: {} ({}){}.".format(len(labels), named,
                                                             "; {} run(s) without events left out".format(skipped)
                                                             if skipped else "")))
            if plain and not mapping.labels_key:
                lines.append((OPTIONAL, "Event names: the events are numbers. If a variable lists their names "
                                        "(classes ...), choose it under Code names: code 1 is its first name."))
        if mapping.bad_key:
            lines.append((READY, "Bad trials: {} marked in {}; trials over them are left out.".format(
                marked, mapping.bad_key)))
    return lines


def _trial_checklist(variables, mapping, found):
    """The checklist of a signal already cut into trials (3-D)."""
    first = np.asarray(variables[found[0][1]])
    if not mapping.axes or sorted(mapping.axes) != ["c", "s", "t"]:
        return [(MISSING, "Signal: {} is cut into trials ({} values). Under Axes, say which is trials, channels "
                          "and samples.".format(found[0][1], " x ".join(str(n) for n in first.shape)))]
    counts = [as_trials(variables[key], mapping.axes).shape for _, key in found]
    n_trials, channels, samples = sum(c[0] for c in counts), counts[0][1], counts[0][2]
    fs = _rate(variables, mapping)
    lines = []
    if channels > samples:
        lines.append((MISSING, "Signal: {} channels of {} samples looks the wrong way round: check Axes.".format(
            channels, samples)))
    else:
        lines.append((READY, "Signal: {}, {} trials of {} channels x {} samples, laid one after another.".format(
            mapping.data_key, n_trials, channels, samples)))
    if not fs or fs <= 0:
        lines.append((MISSING, "Sampling rate: type it (often in the paper or a variable called fs or srate)."))
        return lines
    lines.append((READY, "Sampling rate: {:g} Hz, {:.3g} s per trial.".format(fs, samples / fs)))
    if mapping.codes_key:
        labels = []
        try:
            for number, key in found:
                n = as_trials(variables[key], mapping.axes).shape[0]
                codes = _values(variables, run_key(mapping.codes_key, number))
                if codes.size != n:
                    raise MatError("'{}' has {} values for {} trials: choose a variable with one label per "
                                   "trial".format(run_key(mapping.codes_key, number), codes.size, n))
                names = _names(variables, run_key(mapping.labels_key, number)) if mapping.labels_key else []
                labels += [_label(c, names) for c in codes]
        except MatError as exc:
            lines.append((MISSING, "Trial labels: {}".format(exc)))
            return lines
        tally = {}
        for label in labels:
            tally[label] = tally.get(label, 0) + 1
        lines.append((READY, "Trial labels: {}.".format(", ".join("{} x{}".format(k, v)
                                                                  for k, v in list(tally.items())[:6]))))
    else:
        lines.append((OPTIONAL, "Trial labels: none chosen, every trial is called 'trial'. If a variable holds one "
                                "label per trial (y, labels, class ...), choose it under Event codes."))
    start = float(mapping.trial_start or 0.0)
    lines.append((READY, "Each trial's zero is {:g} s after its first sample: in Epochs, From {:g} s To {:g} s "
                         "gives the trials as stored.".format(-start, start, start + samples / fs)))
    if mapping.bad_key:
        marked = 0
        for number, _ in found:
            marked += int(sum(_truthy(m) for m in _values(variables, run_key(mapping.bad_key, number))))
        lines.append((READY, "Bad trials: {} marked in {}; they are left out.".format(marked, mapping.bad_key)))
    return lines
