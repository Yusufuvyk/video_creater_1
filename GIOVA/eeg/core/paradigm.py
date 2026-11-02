"""Trial structure defined up front, instead of read from markers.

Many paradigms are simply a fixed cycle repeated - motor imagery might be
2 s rest, 2 s cue, 2 s rest, forty times over. Writing that down once produces
the same labelled blocks a marker file would, so everything downstream
(`events.pair_epochs`, `analysis.epoch_band_erd`) works unchanged.

Durations are in seconds. They are converted to samples using the *recording's*
own sampling rate; the `fs` field here is only ever used to warn about a
mismatch, never to reinterpret the data.

Pure stdlib - no Qt.
"""
import json
from pathlib import Path

from .events import Block

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

    @property
    def total_seconds(self):
        return self.start_offset + self.cycle_seconds * self.repeats

    def labels(self):
        seen = []
        for label, _ in self.phases:
            if label not in seen:
                seen.append(label)
        return seen

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

    def describe(self):
        if not self.phases:
            return "no phases"
        cycle = " + ".join("{} {:g}s".format(label, seconds)
                           for label, seconds in self.phases)
        return "{} x [{}] = {:.1f} s".format(self.repeats, cycle,
                                             self.total_seconds)


DEFAULT = Paradigm("motor imagery",
                   [("rest", 2.0), ("cue", 2.0), ("rest", 2.0)],
                   repeats=40)


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


def to_blocks(paradigm, duration):
    """Expand the cycle into labelled blocks, clipped to the recording.

    Returns the same `events.Block` type a marker file produces, so callers
    cannot tell the two apart - which is the point.
    """
    blocks = []
    time = paradigm.start_offset
    if paradigm.cycle_seconds <= 0:
        return blocks

    for _ in range(max(paradigm.repeats, 0)):
        for label, seconds in paradigm.phases:
            if seconds <= 0 or time >= duration:
                continue
            end = min(time + seconds, duration)
            if end > time:
                blocks.append(Block(time, end, label))
            time += seconds
        if time >= duration:
            break
    return blocks


def check_rate(paradigm, fs):
    """Warning text when the paradigm's stated rate is not the file's, else None."""
    if paradigm.fs and abs(paradigm.fs - fs) > 0.5:
        return ("paradigm says {:g} Hz but the recording is {:g} Hz - "
                "durations follow the recording".format(paradigm.fs, fs))
    return None
