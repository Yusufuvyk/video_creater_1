"""Trial structure defined up front, instead of read from markers.

Many paradigms are simply a fixed cycle repeated - motor imagery might be
2 s rest, 2 s cue, 2 s rest, forty times over. Writing that down once gives the
same labelled blocks a marker file would (`events.blocks_by_cycle`: one block
per repeat), so everything downstream works unchanged.

Durations are in seconds. They are converted to samples using the *recording's*
own sampling rate; the `fs` field only records the rate the paradigm was
written for and never reinterprets the data.

Pure stdlib - no Qt.
"""
import json
from pathlib import Path

SUFFIX = ".paradigm.json"


class Paradigm:
    """A repeating cycle of labelled phases."""

    __slots__ = ("name", "phases", "repeats", "start_offset", "fs")

    def __init__(self, name="paradigm", phases=(), repeats=1, start_offset=0.0,
                 fs=None):
        self.name = name
        #: [(label, seconds)]
        self.phases = [(str(label), float(seconds)) for label, seconds in phases]
        self.repeats = int(repeats)
        self.start_offset = float(start_offset)
        self.fs = float(fs) if fs else None

    @property
    def cycle_seconds(self):
        return sum(seconds for _, seconds in self.phases)

    def to_dict(self):
        return {
            "name": self.name,
            "fs": self.fs,
            "start_offset": self.start_offset,
            "repeats": self.repeats,
            "phases": [{"label": label, "seconds": seconds}
                       for label, seconds in self.phases],
        }

    @classmethod
    def from_dict(cls, raw):
        phases = [(p.get("label", "phase"), p.get("seconds", 0.0))
                  for p in raw.get("phases", [])]
        return cls(raw.get("name", "paradigm"), phases, raw.get("repeats", 1),
                   raw.get("start_offset", 0.0), raw.get("fs"))


def path_for(recording_path):
    """Where this recording's paradigm file would live."""
    path = Path(recording_path)
    return path.with_name(path.stem + SUFFIX)


def load(path):
    return Paradigm.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_for(recording_path):
    """The paradigm beside a recording, or None."""
    candidate = path_for(recording_path)
    if not candidate.is_file():
        return None
    try:
        return load(candidate)
    except (ValueError, TypeError):
        return None  # a corrupt file should not block the recording


def save(path, paradigm):
    Path(path).write_text(json.dumps(paradigm.to_dict(), indent=2),
                          encoding="utf-8")
