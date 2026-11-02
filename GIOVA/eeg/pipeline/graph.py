"""Analysis pipelines as graphs of blocks: the engine behind the Workbench.

A pipeline is a set of blocks joined by wires. Every block has typed inputs, a
few parameters and typed outputs; a wire may only join an output to an input
that accepts its type, so a spectrum can never reach a filter. The Workbench
draws a Graph and edits it; it never computes. Everything here runs without a
window, which is what lets a pipeline be tested, run over fifty subjects, or
saved as a `.pipeline.json` and opened by someone else.

Running keeps every block's outputs together with a key made from the block's
type, its parameters and the keys of whatever feeds it. Change one parameter
and only that block and the ones after it run again.

Pure numpy/stdlib - no Qt.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

SUFFIX = ".pipeline.json"
VERSION = 1

# ----------------------------------------------------------------------
# what flows along the wires
# ----------------------------------------------------------------------

#: A continuous recording (core.sources.Recording), events attached.
RAW = "raw"
#: Trials cut from a recording (pipeline.data.Epochs).
EPOCHS = "epochs"
#: One spectrum per trial and channel (pipeline.data.Spectrum).
SPECTRUM = "spectrum"
#: One row of numbers per trial (pipeline.data.Features).
FEATURES = "features"
#: Unfitted learning steps, in order (pipeline.learn.Model).
MODEL = "model"
#: Cross-validated classification (pipeline.learn.Scores).
SCORES = "scores"
#: Time-frequency and ERD / ERS (core.epochs.EpochResult).
TF = "tf"
#: What an ICA found and removed (pipeline.data.IcaResult).
ICA = "ica"

#: band power of two windows of the same trials compared, with a paired test
CONTRAST = "contrast"
TYPES = (RAW, EPOCHS, SPECTRUM, FEATURES, MODEL, SCORES, TF, ICA, CONTRAST)


class PipelineError(Exception):
    """A block could not run. `node` names it, so the Workbench can mark it."""

    def __init__(self, node, message):
        super().__init__(message)
        self.node = node


@dataclass(frozen=True)
class Port:
    name: str
    #: an output produces its first type; an input accepts any of them
    types: tuple
    label: str = ""
    optional: bool = False


@dataclass(frozen=True)
class Param:
    name: str
    default: object
    label: str = ""
    help: str = ""
    #: allowed values, or None for free input
    choices: tuple = None
    #: shown only behind the block's Options: rarely changed
    advanced: bool = False


class Block:
    """One kind of step. Subclasses describe themselves and implement `run`."""

    #: registry key, stored in pipeline files; never rename one
    type = ""
    title = ""
    category = ""
    inputs = ()
    outputs = ()
    params = ()
    #: raise when a change alters results, so old cached outputs are not reused
    version = 1

    def defaults(self):
        return {p.name: p.default for p in self.params}

    def run(self, inputs, params, context):
        """{output name: value} from {input name: value}."""
        raise NotImplementedError

    def describe(self, params):
        """One sentence for a report's methods section."""
        return self.title

    def describe_in(self, graph, node_id):
        """`describe` for the node as it is wired: a block whose sentence depends on what feeds it says so."""
        return self.describe(graph.params(node_id))

    def short(self, params):
        """A few words for the Workbench's block: the setting that matters most."""
        return ""

    def fingerprint(self, params):
        """Anything outside the parameters that changes the result (a file's date)."""
        return ""


REGISTRY = {}


def register(cls):
    REGISTRY[cls.type] = cls()
    return cls


def block(type_name):
    try:
        return REGISTRY[type_name]
    except KeyError:
        raise ValueError("Unknown block '{}'".format(type_name)) from None


# ----------------------------------------------------------------------
# the graph
# ----------------------------------------------------------------------


@dataclass
class Node:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    #: where the Workbench draws it; no meaning to the computation
    position: tuple = (0.0, 0.0)
    #: the user's own name for this node, shown instead of the block's title
    label: str = ""


@dataclass(frozen=True)
class Wire:
    source: str
    output: str
    target: str
    input: str


#: What a literature profile's `source` holds; every field text except `adaptations`, a list.
SOURCE_FIELDS = ("id", "citation", "doi", "status", "adaptations", "fingerprint")


class Graph:
    def __init__(self, name="Pipeline", description="", source=None):
        self.name = name
        self.description = description
        #: for a pipeline taken from a paper (pipeline.literature): where it comes from, how it was
        #: checked, and every way it differs from the paper; None for any other pipeline
        self.source = _source(source)
        self.nodes = {}
        self.wires = []
        #: what each node's previews had chosen on screen, {node id: {output: {"choosers": [[name, value]],
        #: "mean_of": [channel names]}}}: kept in the file and in copies, never part of what is computed (not in
        #: the cache keys nor the method fingerprint)
        self.views = {}

    # building --------------------------------------------------------

    def add(self, type_name, params=None, node_id=None, position=(0.0, 0.0), label=""):
        kind = block(type_name)
        unknown = set(params or {}) - set(kind.defaults())
        if unknown:
            raise ValueError("{} has no parameter {}".format(kind.title, ", ".join(sorted(unknown))))
        if node_id is None:
            node_id = self._free_id(type_name)
        elif node_id in self.nodes:
            raise ValueError("There is already a node '{}'".format(node_id))
        self.nodes[node_id] = Node(node_id, type_name, dict(params or {}), tuple(position), label)
        return node_id

    def _free_id(self, type_name):
        n = 1
        while "{}{}".format(type_name, n) in self.nodes:
            n += 1
        return "{}{}".format(type_name, n)

    def connect(self, source, output, target, input):
        """Join `source.output` to `target.input`; the input's old wire is replaced."""
        for node_id in (source, target):
            if node_id not in self.nodes:
                raise ValueError("There is no node '{}'".format(node_id))
        out_port = _port(block(self.nodes[source].type).outputs, output, source)
        in_port = _port(block(self.nodes[target].type).inputs, input, target)
        if out_port.types[0] not in in_port.types:
            raise ValueError("{} gives {}, but {} needs {}".format(
                _name(self, source, output), out_port.types[0],
                _name(self, target, input), " or ".join(in_port.types)))
        if source == target or target in self.upstream(source):
            raise ValueError("That wire would make a loop")
        self.wires = [w for w in self.wires if not (w.target == target and w.input == input)]
        self.wires.append(Wire(source, output, target, input))

    def chain(self, *steps):
        """Add blocks one after another, each fed by the first output of the one before.

        Steps are type names or (type name, params). Returns the node ids.
        """
        ids, previous = [], None
        for step in steps:
            type_name, params = (step, None) if isinstance(step, str) else step
            node_id = self.add(type_name, params)
            if previous is not None:
                out = block(self.nodes[previous].type).outputs[0]
                accepting = [p for p in block(type_name).inputs if out.types[0] in p.types]
                if not accepting:
                    raise ValueError("{} cannot follow {}".format(block(type_name).title,
                                                                  block(self.nodes[previous].type).title))
                self.connect(previous, out.name, node_id, accepting[0].name)
            ids.append(node_id)
            previous = node_id
        return ids

    def remove(self, node_id):
        self.nodes.pop(node_id)
        self.views.pop(node_id, None)
        self.wires = [w for w in self.wires if node_id not in (w.source, w.target)]

    def disconnect(self, target, input):
        self.wires = [w for w in self.wires if not (w.target == target and w.input == input)]

    def set(self, node_id, **params):
        kind = block(self.nodes[node_id].type)
        unknown = set(params) - set(kind.defaults())
        if unknown:
            raise ValueError("{} has no parameter {}".format(kind.title, ", ".join(sorted(unknown))))
        self.nodes[node_id].params.update(params)

    # reading ---------------------------------------------------------

    def params(self, node_id):
        """The node's parameters with every default filled in."""
        node = self.nodes[node_id]
        return {**block(node.type).defaults(), **node.params}

    def feeds(self, node_id):
        """{input name: Wire} for the node."""
        return {w.input: w for w in self.wires if w.target == node_id}

    def upstream(self, node_id):
        seen, todo = set(), [node_id]
        while todo:
            current = todo.pop()
            for w in self.wires:
                if w.target == current and w.source not in seen:
                    seen.add(w.source)
                    todo.append(w.source)
        return seen

    def order(self, targets=None):
        """Node ids in an order where every node comes after what feeds it."""
        wanted = set(self.nodes) if targets is None else set(targets).union(
            *(self.upstream(t) for t in targets))
        done, ordered = set(), []
        while len(ordered) < len(wanted):
            ready = [n for n in self.nodes if n in wanted and n not in done
                     and all(w.source in done for w in self.wires if w.target == n)]
            if not ready:
                raise ValueError("The pipeline has a loop")
            for n in ready:
                done.add(n)
                ordered.append(n)
        return ordered

    def sources(self, type_name="load"):
        return [n for n, node in self.nodes.items() if node.type == type_name]

    def ends(self):
        """Nodes nothing else reads from: the pipeline's results."""
        used = {w.source for w in self.wires}
        return [n for n in self.order() if n not in used]

    def problems(self):
        """What stops the graph from running, as (node id, message); empty when it can run."""
        found = []
        for node_id, node in self.nodes.items():
            kind = block(node.type)
            feeds = self.feeds(node_id)
            for port in kind.inputs:
                if not port.optional and port.name not in feeds:
                    found.append((node_id, "{} needs an input: {}".format(
                        node.label or kind.title, port.label or port.name)))
        try:
            self.order()
        except ValueError as exc:
            found.append(("", str(exc)))
        return found

    # files -----------------------------------------------------------

    def to_dict(self):
        return dict(
            version=VERSION, name=self.name, description=self.description,
            nodes=[dict(id=n.id, type=n.type, params=n.params, position=list(n.position),
                        **({"label": n.label} if n.label else {}))
                   for n in self.nodes.values()],
            wires=[dict(source=w.source, output=w.output, target=w.target, input=w.input)
                   for w in self.wires],
            **({"source": dict(self.source)} if self.source else {}),
            **({"views": json.loads(json.dumps(self.views))} if self.views else {}))

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("version") != VERSION:
            raise ValueError("Not a pipeline this version of GIOVA can read")
        try:
            graph = cls(raw.get("name", "Pipeline"), raw.get("description", ""), raw.get("source"))
            for n in raw["nodes"]:
                graph.add(n["type"], n.get("params"), n["id"], tuple(n.get("position", (0, 0))), n.get("label", ""))
            for w in raw["wires"]:
                graph.connect(w["source"], w["output"], w["target"], w["input"])
            views = raw.get("views") or {}
            # a view of a node no longer there, or not a mapping, is dropped: it never stops a pipeline opening
            graph.views = {k: v for k, v in views.items() if k in graph.nodes and isinstance(v, dict)} \
                if isinstance(views, dict) else {}
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed pipeline file") from exc
        return graph

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read pipeline: {}".format(exc)) from exc

    def copy(self):
        return Graph.from_dict(self.to_dict())

    def arrange(self, dx=220.0, dy=130.0):
        """Place nodes in columns, each as close as it can be to the block it feeds; for templates.

        A model chain (CSP -> LDA) then ends right beside the Cross-validate block it plugs into.
        """
        height = {}
        for n in reversed(self.order()):
            height[n] = max((height[w.target] + 1 for w in self.wires if w.source == n), default=0)
        top = max(height.values(), default=0)
        depth = {n: top - h for n, h in height.items()}
        rows = {}
        # the signal's path along the top row, model chains below it
        for n in sorted(self.order(), key=self._learning):
            row = rows.get(depth[n], 0)
            rows[depth[n]] = row + 1
            self.nodes[n].position = (depth[n] * dx, row * dy)
        return self

    def method_fingerprint(self):
        """A hash of the blocks, their settings and wires: what a method is, without the file it reads,
        the part of the file, the names given or where the blocks sit on the sheet."""
        nodes = sorted((n.id, n.type, json.dumps({k: v for k, v in self.params(n.id).items()
                                                  if not (n.type == "load" and k in ("path", "condition"))},
                                                 sort_keys=True, default=str))
                       for n in self.nodes.values())
        wires = sorted((w.source, w.output, w.target, w.input) for w in self.wires)
        return hashlib.sha1(json.dumps([nodes, wires]).encode("utf-8")).hexdigest()[:16]

    def changed_from_source(self):
        """True when a literature profile's blocks or settings were changed after it was taken."""
        return bool(self.source and self.source.get("fingerprint")
                    and self.source["fingerprint"] != self.method_fingerprint())

    def methods(self, instead=None):
        """The report's methods text: one sentence per block, the signal's path first, then the model.
        `instead`: {node: text} for blocks run otherwise than they say (Cross-validate under Across)."""
        instead = instead or {}
        steps = sorted(self.order(), key=lambda n: self._learning(n) or block(self.nodes[n].type).outputs[0].types[0] == SCORES)
        return [instead[n] if n in instead else block(self.nodes[n].type).describe_in(self, n) for n in steps]

    def _learning(self, node_id):
        """True for a step of a model (CSP, LDA): it describes how to learn, not the signal."""
        return any(p.types[0] == MODEL for p in block(self.nodes[node_id].type).outputs)


def _source(raw):
    """A literature profile's source with known fields only; None when there is none."""
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("A pipeline's source must be a mapping")
    out = {k: str(raw[k]) for k in SOURCE_FIELDS if k != "adaptations" and raw.get(k)}
    out["adaptations"] = [str(a) for a in raw.get("adaptations") or ()]
    return out


def _port(ports, name, node_id):
    for port in ports:
        if port.name == name:
            return port
    raise ValueError("'{}' has no port '{}'".format(node_id, name))


def _name(graph, node_id, port):
    node = graph.nodes[node_id]
    return "{} ({})".format(node.label or block(node.type).title, port)


# ----------------------------------------------------------------------
# running
# ----------------------------------------------------------------------


class Stopped(Exception):
    """A run stopped on request: the blocks finished before keep their outputs."""


class Context:
    """What a block may use besides its inputs: progress messages, the node's id.

    `stop`, when given, is asked at every progress message and before every block;
    once it answers True the run ends there with Stopped. A block's long loops
    report progress (trials, folds, channels), so a stop takes effect within one step.
    """

    def __init__(self, progress=None, stop=None):
        self._progress = progress
        self._stop = stop
        self.node = ""

    def check(self):
        if self._stop is not None and self._stop():
            raise Stopped("Stopped")

    def progress(self, message):
        self.check()
        if self._progress:
            self._progress(self.node, message)


class Runner:
    """Runs a graph and keeps each node's outputs until something before it changes.

    `cache` may be shared between runners: two pipelines that start with the
    same loading and filtering then compute those steps once.
    """

    def __init__(self, graph, cache=None):
        self.graph = graph
        self.cache = {} if cache is None else cache
        #: node id -> key of its latest outputs
        self.keys = {}

    def run(self, targets=None, overrides=None, progress=None, stop=None):
        """{node id: {output: value}} for `targets` (default: every node).

        `overrides` replaces parameters for this run only, {node id: {name: value}};
        that is how one pipeline is run on many recordings. `stop()` returning True
        ends the run with Stopped at the next block or progress message.
        """
        overrides = overrides or {}
        problems = [p for p in self.graph.problems()
                    if targets is None or p[0] in set(targets).union(*(self.graph.upstream(t) for t in targets))]
        if problems:
            raise PipelineError(problems[0][0], problems[0][1])
        context = Context(progress, stop)
        results = {}
        for node_id in self.graph.order(targets):
            context.check()
            node = self.graph.nodes[node_id]
            kind = block(node.type)
            params = {**self.graph.params(node_id), **overrides.get(node_id, {})}
            feeds = self.graph.feeds(node_id)
            try:
                key = _key(kind, params, {name: self.keys[w.source] for name, w in feeds.items()})
            except (OSError, TypeError, ValueError) as exc:
                raise PipelineError(node_id, str(exc)) from exc
            if key not in self.cache:
                inputs = {name: results[w.source][w.output] for name, w in feeds.items()}
                context.node = node_id
                context.progress(node.label or kind.title)
                try:
                    self.cache[key] = kind.run(inputs, params, context)
                except (PipelineError, Stopped):
                    raise
                except (ValueError, KeyError, IndexError, ArithmeticError, OSError, MemoryError) as exc:
                    raise PipelineError(node_id, "{}: {}".format(node.label or kind.title, exc)) from exc
                except Exception as exc:
                    # anything else still belongs to this block: marked on it, not only in the status line
                    raise PipelineError(node_id, "{}: {}: {}".format(node.label or kind.title, type(exc).__name__,
                                                                     exc)) from exc
            self.keys[node_id] = key
            results[node_id] = self.cache[key]
        return results

    def forget(self):
        self.cache.clear()
        self.keys.clear()

    def current_keys(self):
        """{node id: key} the graph has now, without running anything.

        A node missing an input, or fed by one, has no key. A key found in the
        cache means that node's outputs are already up to date.
        """
        keys = {}
        for node_id in self.graph.order():
            kind = block(self.graph.nodes[node_id].type)
            feeds = self.graph.feeds(node_id)
            if any(not p.optional and p.name not in feeds for p in kind.inputs) or \
                    any(w.source not in keys for w in feeds.values()):
                continue
            try:
                keys[node_id] = _key(kind, self.graph.params(node_id),
                                     {name: keys[w.source] for name, w in feeds.items()})
            except (OSError, TypeError, ValueError):
                continue
        return keys

    def runnable(self):
        """Nodes that have everything they need: the ones Run can compute."""
        return list(self.current_keys())


def _key(kind, params, upstream):
    text = json.dumps([kind.type, kind.version, params, kind.fingerprint(params), sorted(upstream.items())],
                      sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
