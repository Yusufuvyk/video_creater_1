"""A pipeline written out as Python that runs on its own: a zip folder, or one script.

The zip (`save_package`) holds `run.py` - the pipeline, short enough to read at a
glance -, `eeg_studio_code.py` beside it with GIOVA's code it needs,
`requirements.txt` (the libraries, at the versions GIOVA used) and a
README. The one-file script (`save_python`) is the same code in one file.

The script needs Python and the libraries the pipeline uses (numpy, scipy,
and MNE, scikit-learn, pyRiemann, PyWavelets or BrainFlow where a step needs
them) - not GIOVA. It reads like this:

* `run()`: one line per block, in an order where everything a block
  needs comes before it; the signal's path first, then the model, then
  cross-validation. Each line names only the settings the sheet changed from
  the block's defaults; the methods sentence above it says what they all are.
* A model is the list of its steps. Where a library has the step itself -
  scikit-learn's LDA, SVM, logistic regression, random forest, kNN, scaler;
  pyRiemann's covariances, tangent space, MDM - the script calls it directly.
* Below, GIOVA's own functions for everything else (reading the file,
  filters, epochs, CSP, cross-validation, the random-classifier limit, ...),
  copied from the program's source by `bundle` with what they use. The script
  computes with the same code as the Workbench, so the numbers match exactly
  (tests/test_standalone_export.py runs scripts where GIOVA is not
  importable and compares every output).

Pure stdlib - no Qt.
"""
import ast
from datetime import date
from pathlib import Path
import sys
import textwrap

from . import block
from .bundle import bundle
from .graph import EPOCHS, FEATURES, MODEL, SCORES, SPECTRUM, TF
from . import learn

COMPUTE = "eeg.pipeline.compute"
#: the module of a zip that carries GIOVA's code, beside run.py
SUPPORT = "eeg_studio_code"
LEARN = "eeg.pipeline.learn"

#: block type -> the function in pipeline.compute it runs
FUNCTIONS = {
    "units": "convert_units", "channels": "pick_channels", "bad_channels": "repair_channels",
    "reference": "rereference",
    "filter": "filter_recording", "resample": "downsample_recording", "ica": "run_ica", "epochs": "cut_epochs",
    "window": "time_window", "trial_filter": "filter_trials", "select_trials": "select_trials",
    "wavelet_denoise": "wavelet_denoise", "dwt_bands": "dwt_bands", "filter_bank": "filter_bank",
    "baseline": "subtract_baseline", "psd": "welch_spectrum", "fft": "fft_spectrum", "band_power": "band_power",
    "features": "signal_features", "erd": "time_frequency", "compare_windows": "compare_windows",
    "cross_validate": "evaluate", "trial_channels": "pick_trial_channels",
}
#: what a last block gives -> the figure main() draws of it (pipeline.api)
SHOW = {SCORES: "show_scores", SPECTRUM: "show_spectrum", EPOCHS: "show_epochs", FEATURES: "show_features",
        TF: "show_tf"}
#: file suffix -> the reader a Recording block of that file uses
MNE_SUFFIXES = (".edf", ".bdf", ".gdf", ".fif", ".set", ".vhdr")
#: import name -> what pip installs it as
PIP = {"sklearn": "scikit-learn", "pywt": "PyWavelets", "pyriemann": "pyriemann", "mne": "mne",
       "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib", "pandas": "pandas",
       "brainflow": "brainflow"}
#: the order requirements are listed in
PIP_ORDER = ("numpy", "scipy", "mne", "scikit-learn", "pyriemann", "PyWavelets", "brainflow", "pandas",
             "matplotlib")


def package_folder():
    """The folder holding the `eeg` package's readable source, which `bundle` copies from.

    Packaged, the exe carries a readable copy of the package in `_internal/python`
    (PyInstaller's folder for data that travels with it).
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)) / "python"
    return Path(__file__).resolve().parents[2]


def _ordered(graph):
    order = graph.order()
    kind = lambda n: block(graph.nodes[n].type).outputs[0].types[0]
    return ([n for n in order if kind(n) not in (MODEL, SCORES)] + [n for n in order if kind(n) == MODEL] +
            [n for n in order if kind(n) == SCORES])


def _names(graph, node_id):
    """Variable names for each output: the node id, then node id + port for the others."""
    outputs = block(graph.nodes[node_id].type).outputs
    return {p.name: node_id if k == 0 else "{}_{}".format(node_id, p.name) for k, p in enumerate(outputs)}


def _value(value):
    """A literal for the script; Windows paths as raw strings, readable as typed."""
    if isinstance(value, str) and "\\" in value and '"' not in value and not value.endswith("\\"):
        return 'r"{}"'.format(value)
    return repr(value)


def _changed(graph, node_id, skip=()):
    """keyword=value for each setting the node does not keep at its block's default."""
    node = graph.nodes[node_id]
    defaults = block(node.type).defaults()
    return ["{}={}".format(k, _value(v)) for k, v in node.params.items() if v != defaults.get(k) and k not in skip]


def reader(path):
    """The pipeline.compute function that reads a file of this kind (only its code goes into the script)."""
    suffix = Path(path).suffix.lower() if path else ""
    if suffix == ".mat":
        return "read_mat_recording"
    if suffix in MNE_SUFFIXES:
        return "read_mne_recording"
    if suffix == ".csv":
        return "read_brainflow_recording"
    return "read_recording"


# ----------------------------------------------------------------------
# a model's steps as library calls or GIOVA's own classes
# ----------------------------------------------------------------------


def step_code(kind, params):
    """(expression building the step with `seed` in scope, [(module, name) to bundle], [import statements]).

    `params` are the step's arguments as `learn.STEPS[kind]` takes them. The expression builds the same
    step as that factory (tests/test_standalone_export.py checks every kind and setting).
    """
    p = dict(params)
    own = {"augment": "Augment", "csp": "CSP", "fbcsp": "FBCSP", "mibif": "MIBIF", "nbpw": "NBPW",
           "select": "SelectBest"}
    if kind in own:
        if kind == "augment":
            arguments = "copies={!r}, noise={!r}, scale={!r}, shift={!r}, seed=seed".format(
                p["copies"], p["noise"], p["scale"], p["shift"])
        elif kind == "csp":
            arguments = "n_components={!r}, log={!r}".format(p["n_components"], p["log"])
        elif kind == "fbcsp":
            arguments = "n_bands={!r}, n_pairs={!r}".format(p["n_bands"], p["n_pairs"])
        elif kind == "mibif":
            arguments = "k={!r}, pair_size={!r}".format(p["k"], p["pair_size"])
        elif kind == "select":
            arguments = "k={!r}, seed=seed".format(p["k"])
        else:
            arguments = ""
        return "{}({})".format(own[kind], arguments), [(LEARN, own[kind])], []
    if kind == "trcsp":
        return ("_trcsp(n_filters={!r}, alpha={!r}, tune={!r}, inner_folds={!r}, tune_score={!r}, seed=seed)".format(
            p["n_filters"], float(p["alpha"]), p["tune"], p["inner_folds"], p.get("tune_score", learn.ACCURACY)),
            [(LEARN, "_trcsp")], [])
    library = [(LEARN, "Library")]
    if kind == "lda":
        shrinkage = "None" if p["shrinkage"] == "none" else repr(p["shrinkage"])
        return ('Library(LinearDiscriminantAnalysis(solver="lsqr", shrinkage={}))'.format(shrinkage), library,
                ["from sklearn.discriminant_analysis import LinearDiscriminantAnalysis"])
    if kind == "standardize":
        return "Library(StandardScaler())", library, ["from sklearn.preprocessing import StandardScaler"]
    if kind == "covariance":
        return ("Library(Covariances(estimator={!r}))".format(p["estimator"]), library,
                ["from pyriemann.estimation import Covariances"])
    if kind == "tangent":
        return 'Library(TangentSpace(metric="riemann"))', library, ["from pyriemann.tangentspace import TangentSpace"]
    if kind == "mdm":
        return 'Library(MDM(metric="riemann"))', library, ["from pyriemann.classification import MDM"]
    tune = bool(p.get("tune"))
    # the grids come from the same functions the Workbench builds its steps with, so the values match
    if kind == "svm":
        grid = learn.svm_grid(p["kernel"], p.get("C_values", learn.C_GRID), p.get("gamma_values", learn.GAMMA_GRID))
        estimator = "SVC(kernel={!r}, C={!r}, gamma={!r})".format(p["kernel"], float(p["C"]),
                                                                  learn._gamma(p.get("gamma", "scale"))
                                                                  if p["kernel"] == "rbf" else "scale")
        imports = ["from sklearn.svm import SVC"]
    elif kind == "logistic":
        l1 = p["penalty"] == "l1"
        grid = {"C": learn.search_values(p.get("C_values", learn.C_GRID), "C values")}
        estimator = "LogisticRegression(C={!r}, l1_ratio={!r}, solver={!r}, max_iter=2000, random_state=seed)".format(
            float(p["C"]), 1.0 if l1 else 0.0, "liblinear" if l1 else "lbfgs")
        imports = ["from sklearn.linear_model import LogisticRegression"]
    elif kind == "forest":
        grid = learn.forest_grid(p.get("depth_values", ("none", 5, 10)), p.get("leaf_values", (1, 3, 5)))
        estimator = ("RandomForestClassifier(n_estimators={!r}, max_depth={!r}, min_samples_leaf={!r}, "
                     "random_state=seed, n_jobs=1)".format(int(p["trees"]), learn._depth(p.get("depth", 0)),
                                                           int(p.get("leaf", 1))))
        imports = ["from sklearn.ensemble import RandomForestClassifier"]
    elif kind == "knn":
        grid = {"n_neighbors": learn.search_values(p.get("k_values", learn.NEIGHBOUR_GRID), "Neighbour values", int)}
        estimator = "KNeighborsClassifier(n_neighbors={!r})".format(int(p["k"]))
        imports = ["from sklearn.neighbors import KNeighborsClassifier"]
    else:
        raise ValueError("No script form for the learning step '{}'".format(kind))
    if tune:
        # the values "Tune" tries, each scored by a cross-validation inside the training trials
        return "Library({}, {!r}, seed, inner_folds={!r}, scoring={!r})".format(
            estimator, grid, int(p.get("inner_folds", learn.INNER_FOLDS)), p.get("tune_score", learn.ACCURACY)), \
            library, imports
    return "Library({})".format(estimator), library, imports


def model_nodes(graph, node_id):
    """The step nodes of the model a node gives, first to last: those wired into it, then itself."""
    nodes, current = [], node_id
    while current is not None:
        nodes.append(current)
        wire = graph.feeds(current).get("model")
        current = wire.source if wire else None
    return nodes[::-1]


def model_steps(graph, node_id):
    """[(kind, arguments)] of the model a node gives, as `learn.Model` holds them."""
    return [(block(graph.nodes[n].type).step, block(graph.nodes[n].type).arguments(graph.params(n)))
            for n in model_nodes(graph, node_id)]


# ----------------------------------------------------------------------
# the script
# ----------------------------------------------------------------------


def _output_type(graph, node_id):
    return block(graph.nodes[node_id].type).outputs[0].types[0]


def _comment(text, indent):
    """`text` as comment lines of at most about 100 characters."""
    return [indent + "# " + line for line in textwrap.wrap(text, 100 - len(indent) - 2)] or [indent + "#"]


def _parts(graph):
    """What a script is made of: (imports run() needs, run() and main() lines, the helpers bundle, the names
    run() and main() call from it)."""
    variables = {n: _names(graph, n) for n in graph.nodes}
    roots = [(COMPUTE, "summary")]
    imports = set()
    body = []
    models = {n for n in graph.nodes if _output_type(graph, n) == MODEL}
    # models the script needs: those cross-validated, and complete ones nothing uses yet
    used = {w.source for w in graph.wires if w.source in models and w.target not in models}
    used |= {n for n in models if n in graph.ends() and model_steps(graph, n)[-1][0] in learn.CLASSIFIERS}
    for node_id in _ordered(graph):
        node = graph.nodes[node_id]
        kind = block(node.type)
        if node_id in models and node_id not in used:
            continue
        targets = ", ".join(variables[node_id][p.name] for p in kind.outputs)
        if node_id in models:
            # the model's steps, each built anew (seed: the fold's) and fitted by every fold on its training trials
            body += _comment("The model ({}): its steps, built anew for every fold and fitted on that fold's "
                             "training trials only.".format(node.label or kind.title), "    ")
            body.append("    {} = Chain(lambda seed: [".format(targets))
            for step_node in model_nodes(graph, node_id):
                step = block(graph.nodes[step_node].type)
                expression, needed, imported = step_code(step.step, step.arguments(graph.params(step_node)))
                roots += needed
                imports.update(imported)
                label = graph.nodes[step_node].label
                body += _comment(step.describe_in(graph, step_node) + (" ({})".format(label) if label else ""),
                                 " " * 8)
                body.append("        " + expression + ",")
            body.append("    ], {!r})".format(learn.Model(model_steps(graph, node_id)).describe()))
            body.append("")
            roots.append((COMPUTE, "Chain"))
            continue
        body += _comment(kind.describe_in(graph, node_id) + (" ({})".format(node.label) if node.label else ""),
                         "    ")
        unwired = []
        if node.type == "load":
            path = graph.params(node_id)["path"]
            function = reader(path)
            # the part of a file (condition) only means something to MATLAB files
            skip = ("path",) if function in ("read_mat_recording", "read_recording") else ("path", "condition")
            args = [_value(path)] + _changed(graph, node_id, skip=skip)
        else:
            function = FUNCTIONS[node.type]
            feeds = graph.feeds(node_id)
            args = []
            for port in kind.inputs:
                if port.name in feeds:
                    wire = feeds[port.name]
                    args.append(variables[wire.source][wire.output])
                elif not port.optional:
                    args.append("None")
                    unwired.append(port.name)
            args += _changed(graph, node_id)
        roots.append((COMPUTE, function))
        body.append("    {} = {}({}){}".format(targets, function, ", ".join(args),
                                               "  # not wired on the sheet: " + ", ".join(unwired) if unwired else ""))
        body.append("")
    every = [name for n in _ordered(graph) if n not in models or n in used for name in variables[n].values()]
    body.append("    return dict({})".format(", ".join("{0}={0}".format(v) for v in every)))

    ends = [n for n in graph.ends() if n not in models]
    main = ["def main():", '    """Run the pipeline, print what each last block gives, and draw it."""',
            "    # a console or file without µ, → or ± still gets everything else",
            '    sys.stdout.reconfigure(errors="replace")',
            "    out = run()"]
    imports.add("import sys")
    for node_id in ends:
        name = variables[node_id][block(graph.nodes[node_id].type).outputs[0].name]
        main.append("    print({!r}, summary(out[{!r}]))".format(node_id + ":", name))
    for node_id in ends:
        name = variables[node_id][block(graph.nodes[node_id].type).outputs[0].name]
        draw = SHOW.get(_output_type(graph, node_id))
        if draw:
            roots.append(("eeg.pipeline.api", draw))
            main.append("    {}(out[{!r}])".format(draw, name))

    helpers = bundle(roots)
    clashes = sorted(v for v in every if v in helpers.names)
    if clashes:
        raise ValueError("Rename the block(s) {}: the script's functions use that name".format(", ".join(clashes)))
    code = ["def run():",
            '    """Every step of the pipeline, in order. Returns each step\'s output by its name on the sheet."""'] + \
        body + ["", ""] + main
    return imports, code, helpers, sorted({name for _module, name in roots})


def _sorted_imports(statements):
    return sorted(statements, key=lambda s: (not s.startswith("import "), s))


def to_python(graph):
    """The one-file script's text."""
    imports, code, helpers, _called = _parts(graph)
    every_import = _sorted_imports(set(helpers.imports) | imports)
    return "\n".join(
        [_header(graph, helpers, every_import), ""] + every_import + ["", ""] + code +
        ["", "", "# " + "=" * 70,
         "# GIOVA's own code from here on, copied from the program with everything it uses,",
         "# so this script computes exactly what the Workbench computes.",
         "# " + "=" * 70, "", "", helpers.text(), "", "",
         'if __name__ == "__main__":', "    main()"]) + "\n"


def requirements(script_text):
    """What pip must install for a script: every non-standard module it imports, anywhere in it."""
    found = set()
    for node in ast.walk(ast.parse(script_text)):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    names = {PIP.get(m, m) for m in found if m not in sys.stdlib_module_names}
    return sorted(names, key=lambda n: (PIP_ORDER.index(n) if n in PIP_ORDER else len(PIP_ORDER), n))


def _source_lines(graph):
    """What a literature profile rests on, for a script's or a README's opening."""
    if not graph.source:
        return []
    lines = ["Literature profile {}: {}".format(graph.source.get("id", ""), graph.source.get("citation", "")),
             "doi {}. Checked: {}.".format(graph.source.get("doi", "-"), graph.source.get("status", "design"))]
    lines += ["Differs from the paper: " + a for a in graph.source.get("adaptations", [])]
    if graph.changed_from_source():
        lines.append("Changed after it was taken from the library: an adaptation of the profile.")
    return lines


def _header(graph, helpers, imports):
    needed = requirements("\n".join(imports) + "\n" + helpers.text())
    lines = ['"""{}{}'.format(graph.name, ": " + graph.description if graph.description else ""), "",
             "Written by GIOVA's Workbench on {}. It runs on its own: GIOVA's code it".format(
                 date.today().isoformat()),
             "needs is copied below, and it computes the same numbers as the Workbench.",
             "Needs Python 3.11 or later with {}{}:".format(
                 ", ".join(needed[:-1]) + (" and " if len(needed) > 1 else ""), needed[-1]),
             "    pip install " + " ".join(needed),
             "",
             "run() is the pipeline, step by step; main() runs it, prints the results and draws them.",
             "To run it on another file of the same kind, change the path in run()."]
    if graph.source:
        lines += [""] + _source_lines(graph)
    return "\n".join(lines + ['"""'])


def save_python(graph, path):
    Path(path).write_text(to_python(graph), encoding="utf-8")
    return Path(path)


def pinned(names):
    """`name==version` for each library, at the version installed here (the one GIOVA computed with);
    a bare name when the version cannot be read."""
    import importlib
    from importlib import metadata
    modules = {pip: module for module, pip in PIP.items()}
    out = []
    for name in names:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            # packaged as an exe, the libraries travel without their metadata: ask the library itself
            try:
                version = getattr(importlib.import_module(modules.get(name, name)), "__version__", None)
            except ImportError:
                version = None
        # a library that cannot find its own metadata may call itself 0.0.0 (MNE does): pip would fail on it
        known = version and str(version).strip("0.") != ""
        out.append("{}=={}".format(name, version) if known else name)
    return out


def to_package(graph):
    """{file name: text} of the zip: run.py, eeg_studio_code.py, requirements.txt, README.md."""
    imports, code, helpers, called = _parts(graph)
    support = "\n".join(
        ['"""GIOVA\'s own code that run.py uses, copied from the program with everything it needs, so',
         "run.py computes exactly what the Workbench computes. Grouped by the GIOVA file it comes from.",
         "Change run.py, not this file, to keep the numbers equal to the Workbench's.",
         '"""', ""] + _sorted_imports(helpers.imports) + ["", "", helpers.text()]) + "\n"
    needed = requirements(support + "\n" + "\n".join(imports))
    head = ['"""{}{}'.format(graph.name, ": " + graph.description if graph.description else ""), "",
            "Written by GIOVA's Workbench on {}. The pipeline, step by step: run() runs it,".format(
                date.today().isoformat()),
            "main() prints what its last blocks give and draws them. GIOVA's code it needs is in",
            "{}.py beside this file; it runs without GIOVA. Install what requirements.txt lists:".format(
                SUPPORT),
            "    pip install -r requirements.txt",
            "To run it on another file of the same kind, change the path in run()."]
    if graph.source:
        head += [""] + _source_lines(graph)
    run_py = "\n".join(
        head + ['"""'] + _sorted_imports(imports) +
        ["from {} import (".format(SUPPORT)] + _wrapped(called) + [")", "", ""] + code +
        ["", "", 'if __name__ == "__main__":', "    main()"]) + "\n"
    readme = "\n".join(
        ["# {}".format(graph.name), ""] + ([graph.description, ""] if graph.description else []) +
        ["Written by GIOVA's Workbench on {}. It runs without GIOVA and computes the".format(
            date.today().isoformat()),
         "same numbers as the Workbench.", "",
         "## Run it", "", "```", "pip install -r requirements.txt", "python run.py", "```", "",
         "Python 3.11 or later. To run it on another recording of the same kind, change the path in",
         "`run()` in `run.py`.", "",
         "## Files", "",
         "| File | What it is |", "|---|---|",
         "| `run.py` | The pipeline: one step per block, in order, with what each step does |",
         "| `{}.py` | GIOVA's code the steps use (reading the file, filters, trials, models, ...) |".format(
             SUPPORT),
         "| `requirements.txt` | The libraries, at the versions GIOVA used |", ""] +
        (["## Source", ""] + _source_lines(graph) + [""] if graph.source else []))
    return {"run.py": run_py, SUPPORT + ".py": support, "requirements.txt": "\n".join(pinned(needed)) + "\n",
            "README.md": readme}


def _wrapped(names, width=100):
    """`names` as the lines of a parenthesised import, as many to a line as fit."""
    lines, line = [], "   "
    for name in names:
        if len(line) + len(name) + 2 > width:
            lines.append(line)
            line = "   "
        line += " " + name + ","
    return lines + [line] if line.strip() else lines


def save_package(graph, path):
    """The zip at `path`: its files in a folder named after it."""
    import zipfile
    path = Path(path)
    folder = path.stem
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in to_package(graph).items():
            archive.writestr("{}/{}".format(folder, name), text)
    return path
