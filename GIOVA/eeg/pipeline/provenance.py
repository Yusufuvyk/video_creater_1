"""What the program did on its own in a run, said plainly: how each file was read, which trials were used, and
which were left out and why.

A result is only as trustworthy as what went into it, and much of that is decided without the user typing it:
a layout that joined some runs and left others out, channels dropped as not EEG, spans the file marks as bad,
trials past the recording's end, a motion threshold. Each reader writes its own notes on the Recording
(`Recording.reading`) and each trial cutter its reasons (`Epochs.rejected`); this gathers them per block, for the
report of one pipeline and for a comparison over many recordings.

Pure stdlib - no Qt.
"""
from pathlib import Path

from .graph import block

#: Blocks that leave trials out, each with its own reasons (Select trials adds to what came in).
CUTTERS = ("epochs", "select_trials")


def _counts(labels):
    tally = {}
    for label in labels:
        tally[label] = tally.get(label, 0) + 1
    return ", ".join("{} {}".format(n, label) for label, n in tally.items())


def _reasons(rejected):
    """'Marked BAD_artifacts in the file x15, Outside recording x1': the reasons, most common first."""
    tally = {}
    for _, _, reason in rejected:
        tally[reason] = tally.get(reason, 0) + 1
    return ", ".join("{} x{}".format(reason, n) for reason, n in sorted(tally.items(), key=lambda kv: -kv[1]))


def summary(graph, outputs):
    """[(heading, [lines], [(trial, condition, reason)])] for a run's `outputs` ({node: {port: value}}): each
    Recording block with its reading notes, each block that cuts or selects trials with the trials it used and the
    ones it left out (its own, not those left out before it)."""
    from ..core.sources import Recording
    from .data import Epochs
    out = []
    for node_id in graph.order():
        values = outputs.get(node_id) or {}
        node = graph.nodes[node_id]
        title = node.label or block(node.type).title
        for value in values.values():
            if node.type == "load" and isinstance(value, Recording):
                name = Path(value.path).name if value.path else "recording"
                out.append(("{} · {}".format(title, name), list(getattr(value, "reading", [])) or
                            ["Read as it is: nothing was left out or changed."], []))
            if node.type in CUTTERS and isinstance(value, Epochs):
                before = 0
                feed = graph.feeds(node_id).get("epochs")
                if node.type == "select_trials" and feed is not None:
                    given = (outputs.get(feed.source) or {}).get(feed.output)
                    before = len(getattr(given, "rejected", []) or [])
                own = list(value.rejected[before:])
                lines = ["{} trials used: {}.".format(len(value.labels), _counts(value.labels) or "none")]
                if own:
                    lines.append("{} left out: {}.".format(len(own), _reasons(own)))
                else:
                    lines.append("None left out here.")
                flagged = sum(1 for entry in (value.meta or []) if entry.get("flags"))
                if flagged:
                    lines.append("{} trials marked for review (flat channel or unusually large swing) and "
                                 "kept.".format(flagged))
                if getattr(value, "unmeasured", None):
                    lines.append("Not measured during these trials (switched off or flat), left out of regions: "
                                 "{}.".format(", ".join(value.unmeasured)))
                out.append((title, lines, [(str(i), str(c), str(r)) for i, c, r in own]))
    return out


def trial_counts(graph, outputs):
    """(used, left out, reasons) over the blocks that cut or select trials: what a comparison table says of one
    recording; None when the pipeline makes no trials."""
    from .data import Epochs
    used, left = None, []
    for node_id in graph.order():
        node = graph.nodes[node_id]
        if node.type not in CUTTERS:
            continue
        for value in (outputs.get(node_id) or {}).values():
            if isinstance(value, Epochs):
                used = len(value.labels)
                left = list(value.rejected)
    if used is None:
        return None
    return used, len(left), _reasons(left)


def reading(graph, outputs):
    """The reading notes of the pipeline's first Recording block that ran, or []."""
    from ..core.sources import Recording
    for node_id in graph.sources("load"):
        for value in (outputs.get(node_id) or {}).values():
            if isinstance(value, Recording):
                return list(getattr(value, "reading", []))
    return []
