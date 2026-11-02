"""One pipeline, or several side by side, over many recordings.

A pipeline names one file in its Recording block; here that file is swapped
for each recording in turn. Every pipeline sees the same recordings with the
same folds, so their accuracies can be compared subject by subject - the
paired comparison is what tells two methods apart, not two separate means.

Steps the pipelines share (loading, filtering) run once per recording. Results
are kept only for the recording being processed, so fifty 200 MB files do not
pile up in memory.

Pure numpy/scipy - no Qt.
"""
from dataclasses import dataclass, field
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .graph import PipelineError, Runner


def source_node(graph):
    loads = graph.sources("load")
    if len(loads) != 1:
        raise ValueError("'{}' must have exactly one Recording block to run over many files; it has {}"
                         .format(graph.name, len(loads)))
    return loads[0]


def score_nodes(graph):
    return [n for n in graph.order() if graph.nodes[n].type == "cross_validate"]


def recording_names(paths):
    """A short name per recording that tells it apart from the others: its file name, with as
    many parent folders as it takes when two share one (participant_a/session, participant_b/session).

    The names key a Comparison's results, so two files called session.csv cannot overwrite each other.
    """
    paths = [Path(p) for p in paths]
    depth = [0] * len(paths)

    def name(k):
        parents = paths[k].parent.parts
        return "/".join((*parents[len(parents) - depth[k]:], paths[k].stem)) if depth[k] else paths[k].stem

    while True:
        names = [name(k) for k in range(len(paths))]
        clash = [k for k, n in enumerate(names) if depth[k] < len(paths[k].parent.parts)
                 and any(m == n and paths[j] != paths[k] for j, m in enumerate(names))]
        if not clash:
            break
        for k in clash:
            depth[k] += 1
    total = {n: names.count(n) for n in names}
    seen = {}
    for k, n in enumerate(list(names)):
        if total[n] > 1:                # the same file listed twice
            seen[n] = seen.get(n, 0) + 1
            names[k] = "{} ({})".format(n, seen[n])
    return names


#: What Compare can show per recording: (title, Scores attribute). Each a mean of the folds' own values.
ACCURACY = "accuracy"
MEASURES = (("Accuracy", ACCURACY), ("Balanced accuracy", "balanced_accuracy"), ("Cohen's κ", "kappa"),
            ("F1 (macro)", "f1"), ("ROC AUC", "auc"))
PERCENT = (ACCURACY, "balanced_accuracy")


def shown(value, measure):
    """A measure's value as the tables write it: accuracies in %, the others to three decimals; "-" for none."""
    if not np.isfinite(value):
        return "-"
    return "{:.1%}".format(value) if measure in PERCENT else "{:.3f}".format(value)


#: Bytes of trials a leave-one-out run may hold at once; above it, the pipeline must reduce them first.
MEMORY_LIMIT = 3e9
WITHIN, ACROSS = "within", "across"


@dataclass
class Comparison:
    #: one column per (pipeline name, cross-validation node)
    columns: list
    subjects: list
    #: (n_subjects, n_columns); NaN where that pipeline failed on that subject
    accuracy: np.ndarray
    auc: np.ndarray
    #: (subject, column) -> Scores
    scores: dict = field(default_factory=dict)
    #: (subject, pipeline name) -> message
    errors: dict = field(default_factory=dict)
    #: WITHIN: cross-validation inside each recording; ACROSS: trained on the others, tested on each
    evaluation: str = WITHIN
    #: copies of the pipelines as they were run: what a report of these numbers describes, whatever is
    #: edited in the window afterwards
    graphs: list = field(default_factory=list)
    #: recording -> how its file was read (Recording.reading), as the first pipeline to read it said
    reading: dict = field(default_factory=dict)
    #: (recording, pipeline name) -> (trials used, trials left out, why): pipeline.provenance.trial_counts
    trials: dict = field(default_factory=dict)

    def column_name(self, column):
        name, node = column
        return name if sum(c[0] == name for c in self.columns) == 1 else "{} / {}".format(name, node)

    def better(self, i, j):
        """Was recording i's accuracy under column j better than random? None when it failed."""
        scores = self.scores.get((self.subjects[i], self.columns[j]))
        return None if scores is None else scores.better_than_random

    def better_counts(self):
        """Per column: recordings whose accuracy was better than random."""
        return [sum(1 for i in range(len(self.subjects)) if self.better(i, j)) for j in range(len(self.columns))]

    def values(self, measure=ACCURACY):
        """(n_subjects, n_columns) of one of MEASURES; NaN where that pipeline failed or the measure has no value
        (ROC AUC of more than two conditions)."""
        if measure == ACCURACY:
            return np.asarray(self.accuracy, dtype=float)
        out = np.full((len(self.subjects), len(self.columns)), np.nan)
        for i, subject in enumerate(self.subjects):
            for j, column in enumerate(self.columns):
                scores = self.scores.get((subject, column))
                if scores is not None:
                    out[i, j] = getattr(scores, measure)
        return out

    def cell(self, i, j, measure=ACCURACY):
        """The table's text for one recording and column: the measure, * when its accuracy was better than random
        (the verdict is accuracy's, whichever measure is shown)."""
        value = self.values(measure)[i, j] if measure != ACCURACY else self.accuracy[i, j]
        if not np.isfinite(self.accuracy[i, j]):
            return "failed"
        return "{}{}".format(shown(value, measure), " *" if self.better(i, j) else "")

    def summary(self):
        """Per column: (name, mean, sd, median, subjects counted)."""
        rows = []
        for j, column in enumerate(self.columns):
            values = self.accuracy[:, j][np.isfinite(self.accuracy[:, j])]
            rows.append((self.column_name(column), float(np.mean(values)) if values.size else np.nan,
                         float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                         float(np.median(values)) if values.size else np.nan, int(values.size)))
        return rows

    def paired(self):
        """Every pair of columns: (a, b, mean difference a - b, Wilcoxon p, subjects in both)."""
        out = []
        for i, j in combinations(range(len(self.columns)), 2):
            both = np.isfinite(self.accuracy[:, i]) & np.isfinite(self.accuracy[:, j])
            diff = self.accuracy[both, i] - self.accuracy[both, j]
            p = float(wilcoxon(diff).pvalue) if both.sum() >= 6 and np.any(diff != 0) else np.nan
            out.append((self.column_name(self.columns[i]), self.column_name(self.columns[j]),
                        float(diff.mean()) if diff.size else np.nan, p, int(both.sum())))
        return out

    def paired_holm(self):
        """`paired()` with each p also Holm-adjusted over all the pairs: (a, b, diff, p, Holm p, n)."""
        rows = self.paired()
        adjusted = holm([p for *_, p, _n in rows])
        return [(a, b, d, p, q, n) for (a, b, d, p, n), q in zip(rows, adjusted)]

    def save_csv(self, path):
        names = [self.column_name(c) for c in self.columns]
        # the measures added in 1.0.0 go after the columns there were before, so older readers still find theirs
        extra = [("balanced accuracy", "balanced_accuracy"), ("Cohen's kappa", "kappa"), ("macro F1", "f1")]
        extra_values = [self.values(measure) for _, measure in extra]
        number = lambda v: "" if not np.isfinite(v) else round(float(v), 6)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["subject"] + ["{} accuracy".format(n) for n in names] + ["{} AUC".format(n) for n in names]
                            + ["{} better than random".format(n) for n in names]
                            + ["{} {}".format(n, title) for title, _ in extra for n in names])
            for i, (subject, acc, auc) in enumerate(zip(self.subjects, self.accuracy, self.auc)):
                verdicts = ["" if self.better(i, j) is None else int(self.better(i, j)) for j in range(len(names))]
                writer.writerow([subject] + [number(v) for v in (*acc, *auc)] + verdicts
                                + [number(values[i, j]) for values in extra_values for j in range(len(names))])


def holm(p_values):
    """Holm (1979) step-down adjustment: with three pipelines there are three pairs, and one of three
    p values below 0.05 by chance is far likelier than one. NaN stays NaN and is not counted."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite], kind="stable")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (order.size - rank) * p[index]))
        out[index] = running
    return out.tolist()


def compare(graphs, paths, progress=None, stop=None):
    """Run every graph on every recording. `graphs` is a list; each graph's name heads its column.

    `stop()` is asked before each recording; when it says True the result holds the recordings done so far.
    """
    graphs = list(graphs)
    names = [g.name for g in graphs]
    if len(set(names)) != len(names):
        raise ValueError("Give each pipeline its own name")
    sources = [source_node(g) for g in graphs]
    columns = [(g.name, n) for g in graphs for n in score_nodes(g)]
    if not columns:
        raise ValueError("None of the pipelines has a Cross-validate block")
    paths = [Path(p) for p in paths]
    accuracy = np.full((len(paths), len(columns)), np.nan)
    auc = np.full_like(accuracy, np.nan)
    labels = recording_names(paths)
    result = Comparison(columns, list(labels), accuracy, auc, graphs=[g.copy() for g in graphs])

    for i, (path, label) in enumerate(zip(paths, labels)):
        if stop is not None and stop():
            keep = slice(0, i)
            result.subjects, result.accuracy, result.auc = result.subjects[keep], accuracy[keep], auc[keep]
            break
        cache = {}
        for graph, source in zip(graphs, sources):
            def report(node, message, _graph=graph, _i=i):
                if progress:
                    progress("{} ({}/{}) · {} · {}".format(path.name, _i + 1, len(paths), _graph.name, message))
            try:
                outputs = Runner(graph, cache).run(score_nodes(graph), {source: {"path": str(path)}}, report)
            except PipelineError as exc:
                result.errors[(label, graph.name)] = str(exc)
                # what did run is still said: how the file was read, which trials went before it stopped
                _remember_handling(result, label, graph, _finished(graph, source, path, cache))
                continue
            _remember_handling(result, label, graph, outputs)
            for node in score_nodes(graph):
                j = columns.index((graph.name, node))
                scores = outputs[node]["scores"]
                result.scores[(label, columns[j])] = scores
                accuracy[i, j] = scores.accuracy
                auc[i, j] = scores.auc
    return result


def _finished(graph, source, path, cache):
    """{node: outputs} of the blocks a run on `path` finished before it stopped, from the run's cache."""
    copy = graph.copy()
    copy.set(source, path=str(path))
    return {n: cache[k] for n, k in Runner(copy, cache).current_keys().items() if k in cache}


def _remember_handling(result, label, graph, outputs):
    """What the program did with this recording under this pipeline, for the comparison's report."""
    from . import provenance
    counts = provenance.trial_counts(graph, outputs)
    if counts is not None:
        result.trials[(label, graph.name)] = counts
    if label not in result.reading:
        notes = provenance.reading(graph, outputs)
        if notes:
            result.reading[label] = notes


def _cv_feeds(graph, node):
    """(data node, its output, model node, its output) wired into a Cross-validate block."""
    feeds = graph.feeds(node)
    if "data" not in feeds or "model" not in feeds:
        raise ValueError("'{}': its Cross-validate block needs both its inputs wired".format(graph.name))
    return feeds["data"].source, feeds["data"].output, feeds["model"].source, feeds["model"].output


def across(graphs, paths, progress=None, stop=None, limit=MEMORY_LIMIT):
    """Leave one recording out, for every pipeline: trained on the others, tested on each.

    Each recording is run up to what its Cross-validate block would receive - the trials or
    features and the model - and those are kept (as 32-bit floats) until every recording is in.
    The Cross-validate block's own condition choice and seed apply; its folds do not.
    """
    from .data import Epochs
    from . import learn
    graphs = list(graphs)
    names = [g.name for g in graphs]
    if len(set(names)) != len(names):
        raise ValueError("Give each pipeline its own name")
    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("Leaving one recording out needs at least two recordings")
    columns = [(g.name, n) for g in graphs for n in score_nodes(g)]
    if not columns:
        raise ValueError("None of the pipelines has a Cross-validate block")
    accuracy = np.full((len(paths), len(columns)), np.nan)
    labels = recording_names(paths)
    result = Comparison(columns, list(labels), accuracy, np.full_like(accuracy, np.nan), evaluation=ACROSS,
                        graphs=[g.copy() for g in graphs])
    for graph in graphs:
        source = source_node(graph)
        for node in score_nodes(graph):
            data_node, data_port, model_node, model_port = _cv_feeds(graph, node)
            params = graph.params(node)
            wanted = [c.strip() for c in (params["conditions"] if isinstance(params["conditions"], list)
                                         else str(params["conditions"]).split(",")) if c.strip()]
            groups, held, model, reference = [], 0, None, None
            for i, (path, label) in enumerate(zip(paths, labels)):
                if stop is not None and stop():
                    break
                if progress:
                    progress("{} ({}/{}) · {} · reading".format(path.name, i + 1, len(paths), graph.name))
                cache = {}
                try:
                    out = Runner(graph, cache).run([data_node, model_node], {source: {"path": str(path)}})
                except PipelineError as exc:
                    result.errors[(label, graph.name)] = str(exc)
                    _remember_handling(result, label, graph, _finished(graph, source, path, cache))
                    continue
                _remember_handling(result, label, graph, out)
                value, model = out[data_node][data_port], out[model_node][model_port]
                values = value.data if isinstance(value, Epochs) else value.values
                schema = _schema(value)
                reference = reference or schema
                try:
                    values = _aligned(values, schema, reference)
                except ValueError as exc:
                    result.errors[(label, graph.name)] = "{} {}".format(label, exc)
                    continue
                conditions = np.asarray(value.labels).astype(str)
                keep = np.isin(conditions, wanted) if wanted else np.ones(conditions.size, dtype=bool)
                values = np.asarray(values[keep], dtype=np.float32)
                held += values.nbytes
                if held * 2 > limit:
                    raise ValueError("Holding every recording's trials needs over {:.1f} GB. Reduce them first: "
                                     "Downsample, a Time window, fewer Channels, or features instead of raw "
                                     "trials".format(held * 2 / 1e9))
                groups.append((label, values, conditions[keep], [i for i, k in zip(value.ids, keep) if k]))
            if model is None or len(groups) < 2:
                continue
            j = columns.index((graph.name, node))
            for (name, *_), scores in zip(groups, learn.leave_one_out(groups, model, params["seed"], progress,
                                                                      stop)):
                i = result.subjects.index(name)
                result.scores[(name, columns[j])] = scores
                accuracy[i, j] = scores.accuracy
                result.auc[i, j] = scores.auc
    return result


def _schema(value):
    """What the columns of a recording's trials or features mean, for comparing recordings."""
    from .data import Epochs
    if isinstance(value, Epochs):
        return dict(kind="trials", names=[str(n) for n in value.names], fs=float(value.fs),
                    tmin=round(float(value.tmin), 6), samples=int(value.data.shape[-1]))
    return dict(kind="features", names=[str(n) for n in value.names], units=[str(u) for u in value.units])


def _aligned(values, schema, reference):
    """`values` with their channels (trials) or features in the reference recording's order.

    Equal shapes are not enough to train on one recording and test on another: C3 and C4
    swapped, or another rate or window, give arrays of the same size that mean something
    else. Names in another order are put in the reference's order; anything else is refused.
    """
    what = "channels" if schema["kind"] == "trials" else "features"
    if schema["kind"] != reference["kind"]:
        raise ValueError("gives {} where the first recording gave {}".format(schema["kind"], reference["kind"]))
    if schema["kind"] == "trials":
        if schema["fs"] != reference["fs"]:
            raise ValueError("is sampled at {:g} Hz, the first recording at {:g} Hz; add Downsample to give them "
                             "one rate".format(schema["fs"], reference["fs"]))
        if (schema["tmin"], schema["samples"]) != (reference["tmin"], reference["samples"]):
            raise ValueError("gives trials of {} samples from {:g} s, the first recording {} samples from {:g} s"
                             .format(schema["samples"], schema["tmin"], reference["samples"], reference["tmin"]))
    names, wanted = schema["names"], reference["names"]
    order = None
    if names != wanted:
        if sorted(names) != sorted(wanted) or len(set(names)) != len(names):
            missing = [n for n in wanted if n not in names]
            extra = [n for n in names if n not in wanted]
            raise ValueError("has other {} than the first recording{}{}; give them the same {} (a Channels "
                             "block)".format(what, "; missing " + ", ".join(missing[:8]) if missing else "",
                                             "; not in the first: " + ", ".join(extra[:8]) if extra else "", what))
        order = [names.index(n) for n in wanted]
    if schema["kind"] == "features" and schema["units"] and reference["units"]:
        units = [schema["units"][k] for k in order] if order is not None else schema["units"]
        if units != reference["units"]:
            raise ValueError("has its features in other units than the first recording")
    values = np.asarray(values)
    return values if order is None else values[:, order]
