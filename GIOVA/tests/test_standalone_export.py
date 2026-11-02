"""The exported Python script runs without GIOVA and computes what the Workbench computes.

Each pipeline is run on the sheet and as its exported script, the script in
another Python process (-I) where `eeg` cannot be imported (standalone_probe.py).
Every output of every block is compared: texts, labels, ids and counts exactly,
numbers to 1e-9 of their size. The script's code is GIOVA's own, copied by
pipeline.bundle, so nearly everything is bitwise equal; the exception is a
model step the script takes from scikit-learn (LDA, the scaler) where EEG
Studio has its own, which agree to about 1e-15.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from standalone_probe import flatten  # noqa: E402
from test_epochs import lateralised  # noqa: E402
from test_pipeline import save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from test_trial_tf import cho_file  # noqa: E402
from eeg.pipeline import Graph, Runner, block, compute, learn, literature, templates  # noqa: E402
from eeg.pipeline.bundle import BundleError, bundle  # noqa: E402
from eeg.pipeline.export import FUNCTIONS, SUPPORT, requirements, save_package, step_code, to_python  # noqa: E402
from eeg.pipeline.graph import MODEL  # noqa: E402

PROBE = ROOT / "tests" / "standalone_probe.py"
#: numbers agree to this share of the largest value they are compared among
TOLERANCE = 1e-9


def run_alone(script_text, directory, name="pipeline"):
    """(scalars, arrays) of the script's run(), in a Python that cannot import GIOVA."""
    script = directory / (name + ".py")
    script.write_text(script_text, encoding="utf-8")
    alone = directory / "alone"
    alone.mkdir(exist_ok=True)
    out = alone / name
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    done = subprocess.run([sys.executable, "-I", str(PROBE), str(script), str(out)], cwd=str(alone),
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, env=env)
    if done.returncode:
        raise AssertionError("The script failed on its own:\n" + done.stderr[-3000:])
    scalars = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    keys = json.loads(Path(str(out) + ".keys.json").read_text(encoding="utf-8"))
    with np.load(str(out) + ".npz") as stored:
        arrays = {key: stored["a{}".format(k)] for k, key in enumerate(keys)}
    return scalars, arrays


def on_the_sheet(graph):
    """(scalars, arrays) of every block's output when the Workbench runs `graph`, named as the script names them."""
    outputs = Runner(graph).run()
    scalars, arrays = {}, {}
    for node_id, ports in outputs.items():
        kind = block(graph.nodes[node_id].type)
        if kind.outputs[0].types[0] == MODEL:
            continue
        for k, port in enumerate(kind.outputs):
            flatten(ports[port.name], node_id if k == 0 else "{}_{}".format(node_id, port.name), scalars, arrays)
    return scalars, arrays


class Matching:
    def assert_same(self, sheet, alone, label=""):
        (s_scalars, s_arrays), (a_scalars, a_arrays) = sheet, alone
        models = {k[:-len("#type")] for k, v in a_scalars.items() if k.endswith("#type") and v == "Chain"}
        a_scalars = {k: v for k, v in a_scalars.items() if k.split(".")[0].split("[")[0].split("#")[0] not in models}
        self.assertEqual(sorted(s_arrays), sorted(a_arrays), label)
        missing = sorted(set(s_scalars) ^ set(a_scalars))
        self.assertEqual(missing, [], label)
        for key, value in s_scalars.items():
            other = a_scalars[key]
            if isinstance(value, list) and value[:1] == ["float"]:
                x, y = float(value[1]), float(other[1])
                if np.isnan(x) or np.isnan(y):
                    self.assertTrue(np.isnan(x) and np.isnan(y), "{} {}".format(label, key))
                else:
                    self.assertLessEqual(abs(x - y), TOLERANCE * max(abs(x), 1e-300) + 1e-300, "{} {}".format(label, key))
            else:
                self.assertEqual(value, other, "{} {}".format(label, key))
        for key, a in s_arrays.items():
            b = a_arrays[key]
            self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), "{} {}".format(label, key))
            if a.dtype.kind in "fc":
                np.testing.assert_array_equal(np.isnan(a), np.isnan(b), "{} {}".format(label, key))
                finite = np.isfinite(a)
                scale = float(np.max(np.abs(a[finite]))) if finite.any() else 0.0
                np.testing.assert_allclose(b[finite], a[finite], rtol=0, atol=TOLERANCE * scale,
                                           err_msg="{} {}".format(label, key))
            else:
                np.testing.assert_array_equal(a, b, "{} {}".format(label, key))


def run_package_alone(graph, directory):
    """(scalars, arrays) of the zip's run.py, unzipped, in a Python that cannot import GIOVA; and the folder."""
    import zipfile
    archive = save_package(graph, directory / "pipeline_zip.zip")
    with zipfile.ZipFile(archive) as opened:
        opened.extractall(directory / "unzipped")
    folder = directory / "unzipped" / "pipeline_zip"
    return run_alone((folder / "run.py").read_text(encoding="utf-8"), folder, "run"), folder


class PackageTests(Matching, unittest.TestCase):
    def test_the_zip_runs_alone_and_matches_the_sheet(self):
        builds = {"Motor imagery: CSP + LDA": templates.TEMPLATES["Motor imagery: CSP + LDA"],
                  "Band power per condition": templates.TEMPLATES["Band power per condition"],
                  **literature.LIBRARY}
        with scratch() as directory:
            path = cho_file(directory, trials=20)
            for name, build in builds.items():
                with self.subTest(name):
                    graph = build(str(path))
                    for old in (directory / "unzipped",):
                        if old.exists():
                            import shutil
                            shutil.rmtree(old)
                    alone, folder = run_package_alone(graph, directory)
                    self.assert_same(on_the_sheet(graph), alone, name)
                    files = sorted(p.name for p in folder.iterdir() if p.is_file())
                    self.assertEqual(files, ["README.md", SUPPORT + ".py", "requirements.txt", "run.py"])
                    run_py = (folder / "run.py").read_text(encoding="utf-8")
                    # the pipeline reads at a glance: GIOVA's code is beside it, not in it
                    self.assertLess(len(run_py.splitlines()), 200, name)
                    self.assertNotIn("import eeg", run_py)
                    self.assertIn("from {} import (".format(SUPPORT), run_py)
                    support = (folder / (SUPPORT + ".py")).read_text(encoding="utf-8")
                    self.assertNotIn("def run(", support)
                    pins = (folder / "requirements.txt").read_text(encoding="utf-8").split()
                    self.assertTrue(any(p.startswith("numpy==") for p in pins), pins)
                    if graph.source:
                        self.assertIn(graph.source["citation"][:30], run_py)
                        self.assertIn(graph.source["citation"][:30],
                                      (folder / "README.md").read_text(encoding="utf-8"))


class PinTests(unittest.TestCase):
    def test_versions_without_metadata_come_from_the_libraries_and_never_read_0_0_0(self):
        # packaged as an exe, the libraries travel without their metadata
        from importlib import metadata
        from unittest import mock
        from eeg.pipeline.export import pinned
        with mock.patch.object(metadata, "version", side_effect=metadata.PackageNotFoundError):
            pins = pinned(["numpy", "mne", "nothing-of-that-name"])
        import numpy
        self.assertEqual(pins[0], "numpy==" + numpy.__version__)
        self.assertNotIn("==0.0.0", " ".join(pins))
        self.assertEqual(pins[2], "nothing-of-that-name")


class TemplateTests(Matching, unittest.TestCase):
    def test_every_template_and_literature_profile_matches_the_sheet(self):
        builds = {**templates.TEMPLATES, **literature.LIBRARY}
        with scratch() as directory:
            path = cho_file(directory, trials=20)
            for name, build in builds.items():
                with self.subTest(name):
                    graph = build(str(path))
                    script = to_python(graph)
                    self.assertNotIn("import eeg", script)
                    self.assertNotIn("from eeg", script)
                    self.assert_same(on_the_sheet(graph), run_alone(script, directory, "t"), name)


def _branches(path):
    """Every block the templates leave out, on a 3-channel FIF recording."""
    graph = Graph("Everything else", "Blocks no template uses")
    load, units, channels, filt, resample, epochs, baseline, window = graph.chain(
        ("load", {"path": str(path), "shift_ms": 12.0}), ("units", {"stored": "mV", "gain": 1000.0}),
        ("channels", {"channels": ["C3", "Cz", "C4"]}),
        ("filter", {"low": 1.0, "high": 40.0, "notch": 50.0, "design": "fir"}), ("resample", {"target": 125.0}),
        ("epochs", {"tmin": 0.0, "tmax": 7.0, "reject_uv": 500.0}), ("baseline", {"start": 0.0, "end": 0.5}),
        ("window", {"start": 2.0, "end": 5.0}))
    fft = graph.add("fft", {"detrend": "linear"})
    graph.connect(window, "epochs", fft, "epochs")
    features = graph.add("features", {"selected": ["band:alpha", "relative:alpha", "rms", "mobility"]})
    graph.connect(window, "epochs", features, "epochs")
    # features -> standardise -> select -> tuned SVM, repeated k-fold
    _, _, svm = graph.chain("standardize", ("select", {"k": 6}), ("svm", {"kernel": "rbf", "tune": True}))
    cv1 = graph.add("cross_validate", {"folds": 4, "scheme": learn.REPEATED, "repeats": 2})
    graph.connect(features, "features", cv1, "data")
    graph.connect(svm, "model", cv1, "model")
    # trials -> augment -> CSP -> random forest
    _, _, forest = graph.chain(("augment", {"copies": 1}), ("csp", {"n_components": 2}), ("forest", {"trees": 20}))
    cv2 = graph.add("cross_validate", {"folds": 4})
    graph.connect(window, "epochs", cv2, "data")
    graph.connect(forest, "model", cv2, "model")
    # features -> kNN (tuned), chronological; logistic l1 with a permutation test
    knn = graph.add("knn", {"tune": True})
    cv3 = graph.add("cross_validate", {"scheme": learn.CHRONOLOGICAL})
    graph.connect(features, "features", cv3, "data")
    graph.connect(knn, "model", cv3, "model")
    logistic = graph.add("logistic", {"penalty": "l1", "C": 10.0})
    cv4 = graph.add("cross_validate", {"folds": 4, "permutations": 3, "conditions": ["left", "right"]})
    graph.connect(features, "features", cv4, "data")
    graph.connect(logistic, "model", cv4, "model")
    # ICA (a component named by hand), then a Welch PSD and LDA without shrinkage
    ica, ica_epochs, psd, power = graph.chain(
        ("ica", {"eog_channels": [], "exclude": [0], "components": 3.0}), ("epochs", {"tmin": 2.0, "tmax": 5.0}),
        ("psd", {"window": 0.5}), ("band_power", {"log": False}))
    graph.connect(channels, "raw", ica, "raw")
    lda = graph.add("lda", {"shrinkage": "none"})
    cv5 = graph.add("cross_validate", {"folds": 4})
    graph.connect(power, "features", cv5, "data")
    graph.connect(lda, "model", cv5, "model")
    # re-referenced to Cz, a Morlet map from the recording; trials re-referenced to the average
    # a bad channel taken out, then a Welch PSD of its own trials
    bad, bad_epochs, bad_psd = graph.chain(("bad_channels", {"channels": ["Cz"], "action": "drop"}),
                                           ("epochs", {"tmin": 1.0, "tmax": 6.0}), "psd")
    graph.connect(load, "raw", bad, "raw")
    compare = graph.add("compare_windows", {"window_a": [2.0, 4.0], "window_b": [4.0, 6.0], "name_a": "early"})
    graph.connect(bad_epochs, "epochs", compare, "epochs")
    reference, erd = graph.chain(("reference", {"mode": "channels", "channels": ["Cz"]}),
                                 ("erd", {"method": "morlet", "tmin": 0.0, "tmax": 7.0, "baseline": [0.5, 1.5],
                                          "active": [2.5, 4.5], "topography": True}))
    graph.connect(load, "raw", reference, "raw")
    # the Analysis page's trial analysis: explicit notches, trials typed by hand (an id that is not a number, one
    # left out, one too short), its channels picked from the trials, both averages, the whole feature profile as
    # a region, and the time-frequency settings it had
    notched, manual = graph.chain(
        ("filter", {"low": 1.0, "high": 40.0, "notch": 0.0, "notches": [50.0, 75.0]}),
        ("epochs", {"source": "manual", "tmin": -0.5, "tmax": 4.5,
                    "trials": [["a1", "left", 1.0, 6.5, True], ["2", "right", 8.0, 13.5, True],
                               ["3", "left", 15.0, 20.5, False], ["4", "right", 22.0, 24.0, True],
                               ["5", "left", 29.0, 34.5, True], ["6", "right", 36.0, 41.5, True]]}))
    microvolts = graph.add("units", {"stored": "mV"})       # in µV: a signal of nanovolts would read as flat
    graph.connect(load, "raw", microvolts, "raw")
    graph.connect(microvolts, "raw", notched, "raw")
    picked, both = graph.chain(("trial_channels", {"channels": ["C4", "C3"]}),
                               ("psd", {"window": 1.0, "detrend": "linear", "signal_mean": True}))
    graph.connect(manual, "epochs", picked, "epochs")
    fft_mean = graph.add("fft", {"signal_mean": True})
    graph.connect(picked, "epochs", fft_mean, "epochs")
    region = graph.add("features", {"selected": ["band:mu", "relative:mu", "ratio:mu/beta", "peak:beta", "rms"],
                                    "bands": [["mu", 8.0, 13.0], ["beta", 13.0, 30.0]],
                                    "ratios": [["mu/beta", "mu", "beta"]], "total_range": [2.0, 40.0],
                                    "detrend": "linear", "roi": True})
    graph.connect(picked, "epochs", region, "epochs")
    trial_tf = graph.add("erd", {"channels": ["C3", "C4"], "baseline": [-0.5, 0.0], "active": [1.0, 3.0],
                                 "method": "morlet", "n_cycles": 0.4, "erd_mode": "mean_of_trials",
                                 "reject_uv": 400.0})
    graph.connect(manual, "epochs", trial_tf, "raw")
    return graph


class BlockTests(Matching, unittest.TestCase):
    def test_blocks_no_template_uses(self):
        with scratch() as directory:
            recording = lateralised(trials=40)
            recording.data = recording.data * 1e-3          # stored in mV
            graph = _branches(save_fif(recording, directory))
            used = {n.type for n in graph.nodes.values()}
            for build in {**templates.TEMPLATES, **literature.LIBRARY}.values():
                used |= {n.type for n in build("x.mat").nodes.values()}
            from eeg.pipeline import REGISTRY
            self.assertEqual(sorted(set(REGISTRY) - used), [])            # every block is tested somewhere
            script = to_python(graph)
            self.assert_same(on_the_sheet(graph), run_alone(script, directory, "b"), "branches")

    def test_a_brainflow_recording(self):
        from eeg.core import boards
        board = boards.CYTON_DAISY
        with scratch() as directory:
            rng = np.random.default_rng(3)
            n = int(board.fs * 64)
            t = np.arange(n) / board.fs
            rows = np.zeros((n, board.n_rows))
            rows[:, list(board.eeg_rows)] = rng.normal(0, 5, (n, len(board.eeg_rows)))
            rows[:, list(board.eeg_rows)[0]] += 20 * np.sin(2 * np.pi * 10 * t) * (np.floor(t / 4) % 2)
            rows[:, board.timestamp_row] = 1700000000 + t
            rows[:, 0] = np.arange(n) % 256
            for k in range(15):
                rows[int((k * 4 + 1) * board.fs), board.marker_row] = 1 + k % 2
            path = directory / "session.csv"
            np.savetxt(path, rows, delimiter="\t")
            graph = Graph("CSV")
            *_, power = graph.chain(("load", {"path": str(path)}), ("filter", {"low": 1.0, "high": 30.0}),
                                    ("epochs", {"tmin": 0.0, "tmax": 3.0}), "psd", "band_power")
            lda = graph.add("lda")
            cv = graph.add("cross_validate", {"folds": 3})
            graph.connect(power, "features", cv, "data")
            graph.connect(lda, "model", cv, "model")
            script = to_python(graph)
            self.assertIn("read_brainflow_recording(", script)
            self.assertIn("brainflow", requirements(script))
            self.assert_same(on_the_sheet(graph), run_alone(script, directory, "c"), "csv")


class ScriptTests(unittest.TestCase):
    def test_the_script_reads_as_the_pipeline(self):
        graph = templates.csp_lda(r"C:\data\s01.mat")
        script = to_python(graph)
        run = script[script.index("def run():"):script.index("def main():")]
        self.assertIn('read_mat_recording(r"C:\\data\\s01.mat"', run)
        self.assertIn("filter_recording(load1, low=8.0, high=30.0, notch=0.0)", run)
        self.assertIn("CSP(n_components=6, log=True)", run)
        self.assertIn('LinearDiscriminantAnalysis(solver="lsqr", shrinkage=\'auto\')', run)
        self.assertIn("evaluate(epochs1, lda1)", run)
        self.assertLess(script.index("def run():"), script.index("# from GIOVA's"))   # the pipeline first
        self.assertIn("pip install numpy scipy mne scikit-learn matplotlib", script)
        self.assertNotIn("brainflow", script.lower().split('"""')[1])      # a .mat file does not need BrainFlow
        compile(script, "csp_lda.py", "exec")
        graph.disconnect("cross_validate1", "data")
        script = to_python(graph)
        self.assertIn("evaluate(None, lda1)  # not wired on the sheet: data", script)
        compile(script, "unwired.py", "exec")

    def test_steps_are_the_learning_steps_the_workbench_builds(self):
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        namespace = dict(vars(learn), LinearDiscriminantAnalysis=LinearDiscriminantAnalysis, SVC=SVC,
                         RandomForestClassifier=RandomForestClassifier, LogisticRegression=LogisticRegression,
                         KNeighborsClassifier=KNeighborsClassifier, StandardScaler=StandardScaler,
                         Covariances=Covariances, TangentSpace=TangentSpace, MDM=MDM, seed=7)
        from eeg.pipeline import REGISTRY
        for kind in REGISTRY.values():
            if not getattr(kind, "step", ""):
                continue
            variants = [kind.defaults()]
            for param in kind.params:
                for choice in param.choices or ((not param.default,) if isinstance(param.default, bool) else ()):
                    variants.append({**kind.defaults(), param.name: choice})
            for params in variants:
                with self.subTest(kind=kind.type, params=params):
                    arguments = kind.arguments(params)
                    expression, _, _ = step_code(kind.step, arguments)
                    built = eval(expression, namespace)
                    expected = learn.STEPS[kind.step](seed=7, **arguments)
                    if kind.step in ("lda", "standardize"):
                        self.assertIsInstance(built, learn.Library)       # scikit-learn's, checked below
                        continue
                    self.assertIs(type(built), type(expected))
                    if isinstance(built, learn.Library):
                        self.assertIs(type(built.estimator), type(expected.estimator))
                        self.assertEqual(built.estimator.get_params(), expected.estimator.get_params())
                        self.assertEqual(built.grid, expected.grid)
                        self.assertEqual(built.inner_folds, expected.inner_folds)
                    else:
                        self.assertEqual(vars(built), vars(expected))

    def test_scikit_learn_lda_and_scaler_are_eeg_studios_own(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(1)
        for n, p in ((40, 3), (30, 25), (200, 6)):
            x = rng.normal(size=(n, p)) * rng.uniform(0.1, 10, p)
            y = np.where(x[:, 0] + rng.normal(size=n) > 0, "right", "left")
            test = rng.normal(size=(50, p)) * 3
            for shrinkage, theirs in (("auto", "auto"), ("none", None)):
                ours = learn.LDA(shrinkage).fit(x, y)
                library = learn.Library(LinearDiscriminantAnalysis(solver="lsqr", shrinkage=theirs)).fit(x, y)
                np.testing.assert_allclose(library.decision(test), ours.decision(test), rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(library.predict(test), ours.predict(test))
            scaled = learn.Library(StandardScaler()).fit(x, y)
            np.testing.assert_allclose(scaled.transform(test), learn.Standardize().fit(x).transform(test),
                                       rtol=1e-12, atol=1e-12)

    def test_compute_defaults_are_the_blocks_defaults(self):
        def plain(value):
            return [plain(v) for v in value] if isinstance(value, (list, tuple)) else value
        from eeg.pipeline import REGISTRY
        for kind in REGISTRY.values():
            name = FUNCTIONS.get(kind.type) or ("read_recording" if kind.type == "load" else None)
            if name is None:
                continue
            signature = inspect.signature(getattr(compute, name))
            for param in kind.params:
                with self.subTest(block=kind.type, param=param.name):
                    self.assertIn(param.name, signature.parameters)
                    default = signature.parameters[param.name].default
                    if param.name != "path":
                        self.assertEqual(plain(default), plain(param.default))

    def test_a_name_two_modules_define_is_renamed_where_only_one_is_meant(self):
        helpers = bundle([("eeg.pipeline.compute", "read_mat_recording")])
        text = helpers.text()
        self.assertIn("def channels_path_for(", text)       # formats.channels.path_for; core.paradigm keeps its name
        self.assertIn("channels_path_for(recording_path)", text)
        compile(text, "bundle", "exec")
        with self.assertRaises(BundleError):
            bundle([("eeg.pipeline.compute", "no_such_function")])


if __name__ == "__main__":
    unittest.main()
