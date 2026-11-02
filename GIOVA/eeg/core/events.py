"""Event markers and the labelled blocks between them.

A marker file sits next to a recording and holds `unix_time,label` rows. Markers
are instants; the interval from one marker to the next is a *block* carrying the
first marker's label. A paradigm written down instead (`paradigm.py`) gives the
same blocks, one per repeat of its cycle.

Nothing here is specific to one protocol: labels are read from the file, so any
recording with a marker sidecar works.

Pure stdlib - no Qt.
"""
from collections import namedtuple
from pathlib import Path

#: An interval carrying one label. Times are seconds from the recording start.
Block = namedtuple("Block", "start end label")


def load_events(path):
    """Read a marker file into [(unix_time, label)], sorted by time.

    Blank lines and `#` comments are skipped. Rows whose first field is not a
    number are skipped too, which quietly tolerates a header line. A byte order
    mark (a spreadsheet's "CSV UTF-8") is not part of the first time.
    """
    events = []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            timestamp = float(parts[0])
        except ValueError:
            continue  # header row
        events.append((timestamp, parts[1]))
    return sorted(events)


def to_blocks(events, t0, duration):
    """Turn markers into labelled intervals covering the recording.

    `t0` is the recording's first unix timestamp; `duration` its length in
    seconds. Each marker runs until the next one, and the last runs to the end
    of the recording. Markers outside the recording are dropped.
    """
    relative = [(t - t0, label) for t, label in events]
    relative = [(t, label) for t, label in relative if 0.0 <= t < duration]

    blocks = []
    for i, (start, label) in enumerate(relative):
        end = relative[i + 1][0] if i + 1 < len(relative) else duration
        if end > start:
            blocks.append(Block(start, end, label))
    return blocks


def blocks_by_cycle(paradigm, duration):
    """One block per repeat of the cycle - the whole trial, not one phase.

    This is the shape almost every cued experiment has: a trial that contains
    its own rest, its cue and its task. Splitting it into phases and analysing
    one of them throws away the cue, which is the moment the figure exists to
    show.
    """
    cycle = paradigm.cycle_seconds
    if cycle <= 0:
        return []
    blocks, time = [], paradigm.start_offset
    for _ in range(max(paradigm.repeats, 0)):
        end = min(time + cycle, duration)
        if end - time < cycle:
            break  # a truncated final trial is not a trial
        blocks.append(Block(time, end, paradigm.name or "trial"))
        time += cycle
    return blocks
