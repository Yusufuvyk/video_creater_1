"""Games on the Monitor's stream: tap, bridge, the EEG stand-in, runner, Games tab.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from test_trial_analysis import scratch  # noqa: E402
from eeg.core.game_bridge import BridgeServer, SignalTap, window_reason  # noqa: E402
from eeg.games import manifest as mf  # noqa: E402
from eeg.games.studio_eeg import ADDRESS_ENV, KEY_ENV  # noqa: E402

NAMES = ["O1", "O2", "Cz"]
#: An external game folder to test against (set GIOVA_TEST_GAME); skipped when not given
V2 = Path(os.environ.get("GIOVA_TEST_GAME", "alpha_ball_game_openbci"))


def filled_tap(seconds=5, fs=125):
    tap = SignalTap()
    now = time.time()
    for k in range(seconds * 10):
        chunk = np.full((3, fs // 10), float(k)) + np.arange(3)[:, None] * 100
        tap.push(chunk, fs, NAMES, "file playback test.csv", now=now - seconds + (k + 1) / 10)
    return tap


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, command, args):
        self.calls.append((command, args))
        return "ok " + command


def bridge_env(server):
    host, port = server.address
    return {ADDRESS_ENV: "{}:{}".format(host, port), KEY_ENV: server.key.hex()}


FAKE_ACQUISITION = '''
class EEG:
    def __init__(self):
        raise RuntimeError("the real board must not be opened")
'''
FAKE_CONFIG = '''
DISABLED_CHANNELS = ["Cz"]
USE_BIAS = False
FEATURE_CHANNELS = ["O1"]
'''
FAKE_GAME = '''
import config
from acquisition import EEG
participant = input("Participant name: ")
session = input("Session number [1]: ")
eeg = EEG()
rows = eeg.rows_for(config.FEATURE_CHANNELS)
window = eeg.last_seconds(2)[rows]
eeg.start_recording("recordings/run.csv")
eeg.mark(4)
print("PLAYED", participant, session, eeg.fs, window.shape, flush=True)
eeg.close()
'''


def fake_game(directory):
    for name, text in (("acquisition.py", FAKE_ACQUISITION), ("config.py", FAKE_CONFIG), ("game.py", FAKE_GAME)):
        (directory / name).write_text(textwrap.dedent(text), encoding="utf-8")
    return mf.create(directory, name="Fake game")


class TapAndBridgeTests(unittest.TestCase):
    def test_tap_keeps_newest_samples_in_order(self):
        tap = filled_tap(seconds=40)
        data, times = tap.last(2)
        self.assertEqual(data.shape, (3, 250))
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertEqual(data[0, -1], 399)
        self.assertEqual(tap.last(100)[0].shape[1], 30 * 125)       # capacity
        self.assertEqual(window_reason(times, 250), "")
        self.assertEqual(window_reason(times[:10], 250), "Waiting for samples")
        self.assertEqual(window_reason(times, 250, now=times[-1] + 5), "Stream gap - feedback paused")
        tap.reset()
        self.assertIsNone(tap.info()["fs"])

    def test_studio_eeg_speaks_the_games_interface(self):
        from eeg.games.studio_eeg import StudioEEG
        tap, handler = filled_tap(), Recorder()
        server = BridgeServer(tap, handler)
        config = type(sys)("config")
        config.DISABLED_CHANNELS, config.USE_BIAS = ["Cz"], False
        try:
            with patch.dict(os.environ, bridge_env(server)), patch.dict(sys.modules, {"config": config}):
                eeg = StudioEEG()
                self.assertEqual((eeg.fs, eeg.names, eeg.rows), (125, NAMES, [0, 1, 2]))
                self.assertEqual(eeg.last_seconds(2).shape, (3, 250))
                with self.assertRaisesRegex(ValueError, "check config.py"):
                    eeg.rows_for(["Oz"])
                data, stamp, reason = eeg.window(2)
                self.assertEqual(data.shape, (3, 250))
                with scratch() as directory:
                    eeg.start_recording(str(directory / "run.csv"))
                    self.assertTrue((directory / "run.channels.json").is_file())
                    eeg.mark(4)
                    eeg.close()
                    time.sleep(0.2)
            commands = [c for c, _ in handler.calls]
            self.assertEqual(commands[:3], ["configure", "start_recording", "mark"])
            self.assertEqual(handler.calls[0][1], dict(disabled=["Cz"], use_bias=False))
            self.assertEqual(handler.calls[2][1]["value"], 4.0)
            self.assertIn("closed", commands)
        finally:
            server.close()

    def test_runner_plays_a_game_without_opening_its_board(self):
        tap, handler = filled_tap(), Recorder()
        server = BridgeServer(tap, handler)
        try:
            with scratch() as directory:
                (directory / "recordings").mkdir()
                fake_game(directory)
                env = dict(os.environ, **bridge_env(server), PYTHONIOENCODING="utf-8")
                done = subprocess.run([sys.executable, str(ROOT / "eeg_studio.py"), "--game", str(directory)],
                                      input="Ada\n2\n", capture_output=True, text=True, env=env, timeout=120)
                for path in sorted(directory.rglob("*"), reverse=True):
                    path.rmdir() if path.is_dir() else path.unlink()
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("PLAYED Ada 2 125 (1, 250)", done.stdout)
            self.assertEqual([c for c, _ in handler.calls][:3], ["configure", "start_recording", "mark"])
        finally:
            server.close()


class ManifestTests(unittest.TestCase):
    def test_find_create_and_errors(self):
        with scratch() as root:
            game = root / "Demo_basic_v9"
            game.mkdir()
            fake_game(game)
            broken = root / "broken"
            broken.mkdir()
            (broken / mf.FILENAME).write_text('{"name": "Broken", "entry": "missing.py"}', encoding="utf-8")
            found = mf.find([root])
            self.assertEqual([(m.name if m else None) for m, _, _ in found], [None, "Fake game"])
            self.assertIn("missing.py", found[0][2])
            self.assertEqual(mf.find([game])[0][0].entry, "game.py")
            for path in sorted(root.rglob("*"), reverse=True):
                path.rmdir() if path.is_dir() else path.unlink()

    @unittest.skipUnless((V2 / mf.FILENAME).is_file(), "alpha_ball_game_openbci not on this machine")
    def test_demo_basic_v2_mod_file(self):
        game = mf.load(V2)
        self.assertEqual((game.entry, game.eeg), ("ball_game.py", {"module": "acquisition", "class": "EEG"}))


class GamesPageTests(unittest.TestCase):
    def setUp(self):
        from pyqtgraph.Qt import QtWidgets
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from eeg.ui.monitor import MonitorPage
        from eeg.ui.games_page import GamesPage
        self.monitor = MonitorPage()
        self.page = GamesPage(self.monitor)

    def tearDown(self):
        self.page.shutdown()
        self.page.close()
        self.monitor.stop()
        self.monitor.close()

    def test_start_starts_monitor_and_runs_the_game_process(self):
        from pyqtgraph.Qt import QtCore
        with scratch() as root:
            game_dir = root / "fake"
            game_dir.mkdir()
            (game_dir / "recordings").mkdir()
            fake_game(game_dir)
            self.page.preferences.setValue("folders", [str(game_dir)])
            self.page.reload(select=str(game_dir))
            self.assertEqual(self.page.current_game().name, "Fake game")
            self.page.answers[0][1].setText("Ada")
            self.page.answers[1][1].setText("3")

            def start():                           # stands in for Monitor Start (file playback)
                self.monitor.source = object()
                self.monitor.player = object()
                tap = filled_tap()
                self.monitor.tap = tap
            with patch.object(self.monitor, "start", side_effect=start) as monitor_start, \
                    patch.object(self.monitor, "game_configure", return_value="n/a") as configure, \
                    patch.object(self.monitor, "game_start_recording", return_value="rec") as recording, \
                    patch.object(self.monitor, "game_mark", return_value="marker 4") as mark, \
                    patch.object(self.monitor, "game_stop_recording", return_value="stopped"):
                self.page.start_game()
                monitor_start.assert_called_once()
                self.assertTrue(self.page.wait_until_finished(120), self.page.output.toPlainText())
                self.app.processEvents()
                deadline = time.time() + 5
                while self.page.bridge is not None and time.time() < deadline:
                    self.app.processEvents()
                    time.sleep(0.05)
            text = self.page.output.toPlainText()
            self.assertIn("PLAYED Ada 3 125 (1, 250)", text)
            configure.assert_called_once_with(["Cz"], False, "Fake game")
            recording.assert_called_once()
            self.assertEqual(mark.call_args[0][0], 4.0)
            self.assertTrue(self.page.start_button.isEnabled())
            self.monitor.source = None
            self.monitor.player = None
            for path in sorted(root.rglob("*"), reverse=True):
                path.rmdir() if path.is_dir() else path.unlink()

    def test_monitor_hooks_file_playback_writes_events(self):
        with scratch() as directory:
            self.monitor.source = object()
            self.monitor.player = object()
            try:
                note = self.monitor.game_start_recording(str(directory / "eeg_run01.csv"))
                self.assertIn("eeg_run01_events.csv", note)
                self.monitor.game_mark(1, 1000.5)
                self.monitor.game_mark(2, 1030.5)
                self.assertIn("not applicable", self.monitor.game_configure(["Fp1"], False, "g"))
                lines = (directory / "eeg_run01_events.csv").read_text().splitlines()
                self.assertEqual(lines[1:], ["1000.500000,1", "1030.500000,2"])
                with self.assertRaises(ValueError):
                    self.monitor.game_configure(["Oz9"], False, "g")
                self.monitor.game_stop_recording()
            finally:
                self.monitor.source = None
                self.monitor.player = None


@unittest.skipUnless((V2 / mf.FILENAME).is_file(), "alpha_ball_game_openbci not on this machine")
class RealAlphaBallTests(unittest.TestCase):
    def test_v2_game_connects_through_the_bridge_unchanged(self):
        """The real ball_game.py: it must reach GIOVA and apply its channel settings."""
        from eeg.core.config import CH_NAMES
        tap = SignalTap()
        rng = np.random.default_rng(0)
        stop = threading.Event()

        def feed():
            while not stop.is_set():
                tap.push(rng.normal(0, 10, (16, 12)), 125, list(CH_NAMES), "file playback test")
                time.sleep(0.1)

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        handler = Recorder()
        server = BridgeServer(tap, handler)
        env = dict(os.environ, **bridge_env(server), SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                   PYTHONIOENCODING="utf-8")
        process = subprocess.Popen([sys.executable, str(ROOT / "eeg_studio.py"), "--game", str(V2)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env)
        try:
            process.stdin.write("Test\n1\n")
            process.stdin.close()
            deadline = time.time() + 60
            while not handler.calls and time.time() < deadline and process.poll() is None:
                time.sleep(0.2)
            time.sleep(1.5)                           # a few frames of the game loop
            self.assertIsNone(process.poll(), process.stdout.read() if process.poll() is not None else "")
            self.assertEqual(handler.calls[0][0], "configure")
            # Whatever the game's own config.py asks for, and nothing else:
            # the bridge relays its settings, it does not choose them.
            import importlib.util
            spec = importlib.util.spec_from_file_location("v2_config", V2 / "config.py")
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            self.assertEqual(sorted(handler.calls[0][1]["disabled"]), sorted(config.DISABLED_CHANNELS))
            self.assertEqual(bool(handler.calls[0][1]["use_bias"]), bool(config.USE_BIAS))
        finally:
            process.terminate()
            process.wait(10)
            process.stdout.close()
            stop.set()
            server.close()


if __name__ == "__main__":
    unittest.main()
