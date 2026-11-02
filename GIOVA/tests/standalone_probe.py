"""Run an exported script where GIOVA cannot be imported, and keep everything its run() returns.

    python -I standalone_probe.py SCRIPT OUT

writes OUT.json (every number, text and shape, by where it sits: "epochs1.data",
"cross_validate1.predictions[3][1]") and OUT.npz (the arrays). The test that
starts it flattens the Workbench's outputs with the same `flatten` and compares.
-I keeps the user's site-packages and the current folder out; the import hook
below also refuses `eeg` outright, so the script can only work on its own.

Stdlib + numpy only: this file is not part of GIOVA.
"""
import dataclasses
import importlib.abc
import json
from pathlib import Path
import runpy
import sys

import numpy as np


class NoEEGStudio(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "eeg" or name.startswith("eeg."):
            raise ImportError("the exported script imported GIOVA's {}".format(name))
        return None


def _fields(value):
    if dataclasses.is_dataclass(value):
        return [f.name for f in dataclasses.fields(value)]
    names = []
    for cls in type(value).__mro__:
        names += [s for s in getattr(cls, "__slots__", ()) if s not in names]
    return names + [n for n in getattr(value, "__dict__", {}) if n not in names]


def flatten(value, prefix="", scalars=None, arrays=None, seen=None):
    """({place: number or text}, {place: array}) of everything inside `value`."""
    scalars = {} if scalars is None else scalars
    arrays = {} if arrays is None else arrays
    seen = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for i, item in enumerate(value.tolist()):
                flatten(item, "{}[{}]".format(prefix, i), scalars, arrays, seen)
        else:
            arrays[prefix] = np.asarray(value)
    elif isinstance(value, np.generic):
        flatten(value.item(), prefix, scalars, arrays, seen)
    elif isinstance(value, float):
        scalars[prefix] = ["float", repr(value)]
    elif value is None or isinstance(value, (bool, int, str)):
        scalars[prefix] = value
    elif isinstance(value, Path):
        scalars[prefix] = str(value)
    elif isinstance(value, (list, tuple)):
        scalars[prefix + "#len"] = len(value)
        for i, item in enumerate(value):
            flatten(item, "{}[{}]".format(prefix, i), scalars, arrays, seen)
    elif isinstance(value, dict):
        scalars[prefix + "#keys"] = sorted(str(k) for k in value)
        for key in sorted(value, key=str):
            flatten(value[key], "{}[{!r}]".format(prefix, key), scalars, arrays, seen)
    elif callable(value) or type(value).__module__.split(".")[0] in ("sklearn", "pyriemann", "mne"):
        scalars[prefix] = "<{}>".format(type(value).__name__)
    else:
        if id(value) in seen:
            return scalars, arrays
        seen.add(id(value))
        scalars[prefix + "#type"] = type(value).__name__
        for name in _fields(value):
            if not name.startswith("_") and hasattr(value, name):
                flatten(getattr(value, name), "{}.{}".format(prefix, name), scalars, arrays, seen)
    return scalars, arrays


def main(script, out):
    sys.meta_path.insert(0, NoEEGStudio())
    import matplotlib
    matplotlib.use("Agg")
    # as `python run.py` does: the script's folder is searched first, where a zip's eeg_studio_code.py sits
    sys.path.insert(0, str(Path(script).resolve().parent))
    namespace = runpy.run_path(script, run_name="exported")
    results = namespace["run"]()
    scalars, arrays = {}, {}
    for name, value in results.items():
        flatten(value, name, scalars, arrays)
    Path(out + ".json").write_text(json.dumps(scalars), encoding="utf-8")
    np.savez(out + ".npz", **{"a{}".format(k): v for k, v in enumerate(arrays.values())})
    Path(out + ".keys.json").write_text(json.dumps(list(arrays)), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
