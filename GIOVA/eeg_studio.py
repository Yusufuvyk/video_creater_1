"""Entry point for the packaged GIOVA exe (see build_exe.bat).

run_dashboard.py stays the way to launch from source: it relaunches under the
project venv when started from a foreign interpreter, which a frozen exe never
needs.

    GIOVA.exe              the app
    GIOVA.exe --game DIR   runs a game folder (eegstudio_game.json) on the
                               stream of a running GIOVA (started by its Games tab)
    GIOVA.exe --selftest   checks the bundle without hardware and writes
                               selftest.txt next to the exe (no console in a
                               windowed build); exit code 0 = everything worked
"""
import os
import sys
import tempfile

# MNE imports matplotlib for its file readers, montages and wavelets; nothing is
# drawn with it here, so keep it on the non-GUI backend (no Tk, no second Qt).
os.environ.setdefault("MPLBACKEND", "Agg")
import time
import traceback
from pathlib import Path


def selftest():
    """What a missing DLL or hidden import would break, exercised for real."""
    lines, ok = [], True

    def step(name, function):
        nonlocal ok
        started = time.perf_counter()
        try:
            detail = function()
            lines.append("PASS {:<34} {:6.2f} s  {}".format(name, time.perf_counter() - started, detail or ""))
        except Exception:
            ok = False
            lines.append("FAIL {}\n{}".format(name, traceback.format_exc()))

    def brainflow_stream():
        # Synthetic board through the same streamer path live recording uses.
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        from eeg.core.sources import load_brainflow_csv
        path = Path(tempfile.mkdtemp()) / "selftest.csv"
        board = BoardShim(BoardIds.SYNTHETIC_BOARD, BrainFlowInputParams())
        board.prepare_session()
        board.add_streamer("file://{}:w".format(path))
        board.start_stream(); time.sleep(1.5); board.stop_stream()
        board.release_session()
        rec = load_brainflow_csv(path, board_id=BoardIds.SYNTHETIC_BOARD)
        assert rec.data.shape[1] > 50, rec.data.shape
        assert rec.fs == 250 and rec.n_channels == 16, (rec.fs, rec.n_channels)
        return "{} samples written and read back at {} Hz".format(rec.data.shape[1], rec.fs)

    def board_libraries():
        # Loading these ids pulls in the Cyton and Unicorn board descriptions.
        from brainflow.board_shim import BoardIds, BoardShim
        return "Cyton+Daisy {} Hz, Unicorn {} Hz".format(
            BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD),
            BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD))

    def signal_processing():
        import numpy as np
        from eeg.core.processing import StreamFilter, design_filter
        from eeg.core.spectral import welch_psd
        from eeg.core.impedance import estimate
        t = np.arange(1000) / 125
        x = np.vstack([10 * np.sin(2 * np.pi * 10 * t)] * 2)
        y = StreamFilter(design_filter(125, low=1, high=45, notches=(50,)), 2).process(x)
        f, p = welch_psd(y, 125)
        z = estimate(0, 6e-9 * 1e6 * 500e3 * np.sin(2 * np.pi * 31.25 * t), 125)
        return "alpha peak {:.1f} Hz, impedance {}".format(f[np.argmax(p[0])], z.text)

    def mne_formats():
        # Readers are imported lazily by MNE, and montages are data files: both vanish
        # from a bundle that only follows imports.
        import mne
        import numpy as np
        from eeg.core import montages
        from eeg.core.epochs import EventTable
        from eeg.formats.recordings import load_recording
        mne.set_log_level("ERROR")
        places = montages.positions_2d(["C3", "Cz", "C4", "O1"])
        assert len(places) == 4, places
        fs = 128.0
        raw = mne.io.RawArray(np.random.default_rng(0).normal(0, 1e-5, (2, 1280)),
                              mne.create_info(["C3", "C4"], fs, "eeg"))
        raw.set_annotations(mne.Annotations([1.0, 5.0], [1.0, 1.0], ["769", "770"]))
        path = Path(tempfile.mkdtemp()) / "selftest_raw.fif"
        raw.save(path)
        rec = load_recording(path)
        assert isinstance(rec.events, EventTable) and rec.events.labels == ["769", "770"], rec.events
        from mne.io import read_raw_edf, read_raw_gdf  # noqa: F401  (EDF/GDF readers present)
        return "FIF read back, {} events, montage ok, EDF/GDF readers present".format(len(rec.events))

    def matlab_files():
        # v7 through scipy, v7.3 through h5py, whose HDF5 DLLs a bundle can lose: a 3 x 1000 signal must
        # read 3 x 1000 from both (v7.3 stores it reversed).
        import h5py
        import numpy as np
        from scipy.io import savemat
        from eeg.formats import matfile
        folder = Path(tempfile.mkdtemp())
        signal = np.arange(3000.0).reshape(3, 1000)
        savemat(str(folder / "v7.mat"), {"eeg": signal, "fs": 250.0})
        with h5py.File(str(folder / "v73.mat"), "w", userblock_size=512) as handle:
            handle.create_dataset("eeg", data=signal.T).attrs["MATLAB_class"] = np.bytes_("double")
            handle.create_dataset("fs", data=np.array([[250.0]])).attrs["MATLAB_class"] = np.bytes_("double")
        with open(folder / "v73.mat", "r+b") as raw:
            raw.write(b"MATLAB 7.3 MAT-file, Platform: selftest".ljust(116) + b"\x00" * 8 + b"\x00\x02IM")
        for name in ("v7.mat", "v73.mat"):
            read = matfile.read_variables(folder / name)
            assert np.array_equal(read["eeg"], signal) and float(read["fs"]) == 250.0, name
        # MNE reads EEGLAB .set files through pymatreader, which it imports by name: a bundle can miss it
        import pymatreader
        assert np.array_equal(np.asarray(pymatreader.read_mat(str(folder / "v73.mat"))["eeg"]), signal)
        return "v7 and v7.3 read alike, h5py {}, pymatreader {}".format(h5py.__version__, pymatreader.__version__)

    def game_bridge():
        # A game process talks to the Monitor through this bridge; pygame draws the game.
        import numpy as np
        import pygame  # noqa: F401
        from eeg.core.game_bridge import BridgeServer, SignalTap
        from eeg.games import studio_eeg
        tap = SignalTap()
        tap.push(np.zeros((2, 250)), 125, ["O1", "O2"], "selftest", now=time.time())
        calls = []
        server = BridgeServer(tap, lambda command, args: calls.append(command) or "ok")
        try:
            os.environ[studio_eeg.ADDRESS_ENV] = "{}:{}".format(*server.address)
            os.environ[studio_eeg.KEY_ENV] = server.key.hex()
            eeg = studio_eeg.StudioEEG()
            shape = eeg.last_seconds(1).shape
            eeg.mark(1)
            eeg.close()
        finally:
            server.close()
        assert shape == (2, 125) and "mark" in calls, (shape, calls)
        return "last_seconds {} and markers through the bridge, pygame {}".format(shape, pygame.version.ver)

    def pipeline():
        # The Workbench engine: a FIF through Recording -> Filter -> Epochs -> CV(CSP -> LDA).
        import mne
        import numpy as np
        from eeg.pipeline import Graph, Runner
        mne.set_log_level("ERROR")
        fs, rng = 128.0, np.random.default_rng(1)
        data = rng.normal(0, 1e-6, (3, 40 * 4 * int(fs)))
        onsets = np.arange(40) * 4.0
        labels = ["left" if k % 2 else "right" for k in range(40)]
        for onset, label in zip(onsets, labels):
            a = int((onset + 1) * fs)
            data[0 if label == "left" else 2, a:a + 2 * int(fs)] *= 4
        raw = mne.io.RawArray(data, mne.create_info(["C3", "Cz", "C4"], fs, "eeg"))
        raw.set_annotations(mne.Annotations(onsets, [4.0] * 40, labels))
        path = Path(tempfile.mkdtemp()) / "pipeline_raw.fif"
        raw.save(path)
        graph = Graph("selftest")
        *_, epochs = graph.chain(("load", {"path": str(path)}), ("filter", {"low": 1.0, "high": 40.0}),
                                 ("epochs", {"tmin": 1.0, "tmax": 3.0}))
        _, lda = graph.chain(("csp", {"n_components": 2}), "lda")
        cv = graph.add("cross_validate")
        graph.connect(epochs, "epochs", cv, "data")
        graph.connect(lda, "model", cv, "model")
        scores = Runner(graph).run()[cv]["scores"]
        assert scores.accuracy > 0.9, scores.describe()
        # scikit-learn and pyRiemann are imported only when a model uses them: make sure they came along
        graph.remove(lda)
        graph.remove("csp1")
        *_, svm = graph.chain("covariance", "tangent", ("svm", {"tune": True}))
        graph.connect(svm, "model", cv, "model")
        riemann = Runner(graph).run()[cv]["scores"]
        assert riemann.accuracy > 0.9, riemann.describe()
        ica = graph.add("ica", {"eog_channels": [], "exclude": ["0"], "components": 2})
        graph.connect("filter1", "raw", ica, "raw")
        Runner(graph).run([ica])
        # PyWavelets too: denoising, DWT bands and the complex Morlet CWT
        wavelets = [graph.add("wavelet_denoise"), graph.add("dwt_bands"),
                    graph.add("erd", {"channels": ["C3"], "method": "cwt", "baseline": [1.0, 1.5],
                                      "active": [2.0, 3.0]})]
        for node in wavelets:
            graph.connect(epochs, "epochs", node, "epochs" if node != wavelets[2] else "raw")
        Runner(graph).run(wavelets)
        # Python…: the script is put together from the readable copy of the package the exe carries
        from eeg.pipeline.export import to_python
        script = to_python(graph)
        compile(script, "exported.py", "exec")
        assert "import eeg" not in script and "from eeg" not in script
        exported = path.with_name("selftest_pipeline.py")
        exported.write_text(script, encoding="utf-8")
        return "CSP/LDA {:.0%}, tangent space + tuned SVM {:.0%}, ICA ran, wavelets ran, script {} ({} lines)".format(
            scores.accuracy, riemann.accuracy, exported, len(script.splitlines()))

    def window():
        from pyqtgraph.Qt import QtWidgets
        from eeg.ui.app import MainWindow, use_light_theme
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        use_light_theme(app)
        w = MainWindow(); w.show(); app.processEvents(); w.close()
        from eeg.core.config import DATA_DIR
        return "data folder {}".format(DATA_DIR)

    for name, function in [("board libraries", board_libraries), ("BrainFlow stream + CSV", brainflow_stream),
                           ("filters / spectra / impedance", signal_processing),
                           ("MNE formats / montage", mne_formats), ("MATLAB v7 / v7.3", matlab_files),
                           ("pipeline engine", pipeline),
                           ("game bridge", game_bridge),
                           ("main window", window)]:
        step(name, function)
    report = "\n".join(lines) + "\n\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED") + "\n"
    target = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).parent
    (target / "selftest.txt").write_text(report, encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--game" in sys.argv:
        # A game process: no Qt, the game draws its own (pygame) window.
        if sys.stdout is None:                      # windowed exe started without pipes
            sys.stdout = sys.stderr = open(os.devnull, "w")
        from eeg.games.runner import main as run_game
        sys.exit(run_game(sys.argv))
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "--icon" in sys.argv:            # build step: write the exe icon
        from pyqtgraph.Qt import QtWidgets
        from eeg.ui.splash import write_ico
        app = QtWidgets.QApplication(sys.argv)
        ok = write_ico(sys.argv[sys.argv.index("--icon") + 1])
        sys.exit(0 if ok else 1)
    # Splash first: Qt alone loads in a fraction of a second, the rest takes longer.
    from pyqtgraph.Qt import QtWidgets
    from eeg.ui.splash import show_splash
    app = QtWidgets.QApplication(sys.argv)
    splash = show_splash(app)
    from eeg.ui.app import main
    sys.exit(main(splash))
