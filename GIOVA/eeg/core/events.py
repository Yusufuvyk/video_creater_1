"""Event markers and the epochs built from them.

A marker file sits next to a recording and holds `unix_time,label` rows. Markers
are instants; the interval from one marker to the next is a *block* carrying the
first marker's label. Pairing a block with the preceding block of another label
gives an epoch and its baseline, which is what ERD is measured against.

Nothing here is specific to one protocol: labels are read from the file, so any
recording with a marker sidecar works.

Pure stdlib/numpy - no Qt.
"""
from collections import namedtuple
from pathlib import Path

#: An interval carrying one label. Times are seconds from the recording start.
Block = namedtuple("Block", "start end label")

#: One epoch and the baseline it is measured against, both as Blocks.
Epoch = namedtuple("Epoch", "block baseline")


def load_events(path):
    """Read a marker file into [(unix_time, label)], sorted by time.

    Blank lines and `#` comments are skipped. Rows whose first field is not a
    number are skipped too, which quietly tolerates a header line.
    """
    events = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
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


def labels_of(blocks):
    """Distinct labels, in first-seen order, for populating a dropdown."""
    seen = []
    for block in blocks:
        if block.label not in seen:
            seen.append(block.label)
    return seen


def summarise(blocks):
    """[(label, count, shortest, longest)] in first-seen order.

    What a recording's structure actually is, in the terms the person who
    recorded it used. A bare block count answers none of the questions you have
    when you open a session you did not label yesterday: which conditions are in
    here, how many of each, how long do they run.
    """
    out = []
    for label in labels_of(blocks):
        lengths = [b.end - b.start for b in blocks if b.label == label]
        out.append((label, len(lengths), min(lengths), max(lengths)))
    return out


def describe_labels(blocks, separator=" | "):
    """One line naming every label with its count and duration."""
    parts = []
    for label, count, shortest, longest in summarise(blocks):
        span = ("{:.1f} s".format(shortest) if abs(longest - shortest) < 0.05
                else "{:.1f}-{:.1f} s".format(shortest, longest))
        parts.append("{} {} x {}".format(label, count, span))
    return separator.join(parts)


def pair_epochs(blocks, epoch_label, baseline_label):
    """Pair every `epoch_label` block with the nearest preceding baseline.

    An epoch with no baseline before it is dropped rather than silently
    measured against something arbitrary.
    """
    epochs = []
    previous_baseline = None
    for block in blocks:
        if block.label == baseline_label:
            previous_baseline = block
        elif block.label == epoch_label and previous_baseline is not None:
            epochs.append(Epoch(block, previous_baseline))
    return epochs


def describe(epochs):
    """One-line summary for the status bar."""
    if not epochs:
        return "no epochs"
    lengths = [e.block.end - e.block.start for e in epochs]
    return "{} epochs, {:.1f}-{:.1f} s each".format(
        len(epochs), min(lengths), max(lengths))


# ----------------------------------------------------------------------
# building epochs: where they come from, and what they are measured against
# ----------------------------------------------------------------------
#
# These two questions used to be one. "Epoch" meant "a block with this label"
# and "baseline" meant "the preceding block with that label", which suited an
# alternating rest/task marker file and nothing else. A cued design - a trial
# that contains its own pre-cue rest - could not be expressed at all, because
# the reference is a window *inside* the epoch rather than a separate block.
#
# So they are separate now, and every combination is legal. What produced the
# blocks - a marker file, a written-down paradigm, or nothing at all - stops
# mattering past this point: all six roads end at the same Epoch list.

#: Where epochs come from.
BY_LABEL = "label"        # every block carrying one label
BY_CYCLE = "cycle"        # every repeat of a paradigm cycle: the whole trial
BY_PERIOD = "period"      # fixed length at a fixed period, for markerless data
EPOCH_SOURCES = (BY_LABEL, BY_CYCLE, BY_PERIOD)

#: What each epoch is measured against.
RELATIVE = "relative"     # a window measured from the epoch's own onset
PRECEDING = "preceding"   # the nearest preceding block of a chosen label
ABSOLUTE = "absolute"     # one fixed window in recording time
BASELINE_MODES = (RELATIVE, PRECEDING, ABSOLUTE)


def blocks_by_label(blocks, label):
    """Every block carrying `label`. Marker files and paradigm phases alike."""
    return [block for block in blocks if block.label == label]


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


def blocks_by_period(duration, length, period=None, offset=0.0, label="epoch"):
    """Fixed-length blocks at a fixed period. For recordings with no events.

    A resting recording still has structure worth averaging over - it is just
    structure the experimenter imposes rather than reads back.
    """
    period = float(period or length)
    if length <= 0 or period <= 0:
        return []
    blocks, time = [], float(offset)
    while time + length <= duration:
        blocks.append(Block(time, time + length, label))
        time += period
    return blocks


def default_relative_baseline(epoch_seconds, margin=0.0):
    """(start, end) from epoch onset - derived, not a magic pair of numbers.

    Start after the frames that straddle the epoch's leading edge, then take up
    to a second. On a 7 s trial analysed with a 1 s window this is 0.5-1.5 s,
    which is the value Cho 2017 uses; on any other design it is the same idea
    in that design's terms.
    """
    start = max(float(margin), 0.0)
    end = start + min(1.0, max(epoch_seconds / 4.0, 0.05))
    return start, min(end, epoch_seconds)


def with_relative_baseline(epoch_blocks, start, end):
    """Measure each epoch against a window inside itself.

    The baseline is a synthetic Block on the recording's own time axis, so
    everything downstream treats it exactly like one read from a marker file.
    """
    epochs = []
    for block in epoch_blocks:
        low = block.start + float(start)
        high = block.start + float(end)
        if high > low:
            epochs.append(Epoch(block, Block(low, min(high, block.end),
                                             "baseline")))
    return epochs


def with_absolute_baseline(epoch_blocks, start, end):
    """Measure every epoch against one fixed stretch of the recording."""
    if end <= start:
        return []
    reference = Block(float(start), float(end), "baseline")
    return [Epoch(block, reference) for block in epoch_blocks]


def build_epochs(blocks, source=BY_LABEL, baseline=PRECEDING, label=None,
                 baseline_label=None, window=(0.0, 1.0), paradigm=None,
                 duration=0.0, length=0.0, period=None, offset=0.0):
    """One entry point: pick where epochs come from and what they reference.

    Returns the same `Epoch` list whatever the combination, which is the whole
    point - `analysis.epoch_stack` and the views below it never learn which
    road was taken.
    """
    if source == BY_CYCLE:
        found = blocks_by_cycle(paradigm, duration) if paradigm else []
    elif source == BY_PERIOD:
        found = blocks_by_period(duration, length, period, offset)
    else:
        found = blocks_by_label(blocks, label)

    if not found:
        return []
    if baseline == RELATIVE:
        return with_relative_baseline(found, *window)
    if baseline == ABSOLUTE:
        return with_absolute_baseline(found, *window)
    return pair_epochs(blocks, found[0].label if label is None else label,
                       baseline_label)
