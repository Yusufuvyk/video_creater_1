"""Letting a game read the signal GIOVA is already streaming.

A game such as alpha_ball_game_openbci's alpha ball opens the board itself. Inside
GIOVA it cannot: the Monitor holds the serial port, and a port has one owner.
So the game runs in its own process and its `acquisition.EEG` is swapped for a
client of this bridge. The Monitor pushes every raw chunk it receives into a
`SignalTap`; the game asks the bridge for "the last N seconds", sends markers,
starts its recording and asks for its channel settings - exactly the calls it
made on the board before.

Pure stdlib/numpy - no Qt. Requests that must touch the board or the widgets
are handed to a callback, which the UI turns into work on its own thread.
"""
from multiprocessing.connection import Client, Listener
import secrets
import threading
import time

import numpy as np

#: Seconds of raw signal kept for games.
TAP_SECONDS = 30.0


class SignalTap:
    """The newest raw samples (µV), with a wall-clock time for each, thread-safe."""

    def __init__(self, seconds=TAP_SECONDS):
        self.seconds = float(seconds)
        self._lock = threading.Lock()
        self.reset()

    def reset(self, fs=None, names=None, source="stopped"):
        with self._lock:
            self.fs = float(fs) if fs else None
            self.names = list(names or [])
            self.source = source
            n = len(self.names)
            self._data = np.empty((n, 0))
            self._times = np.empty(0)

    def push(self, chunk, fs, names, source, now=None):
        """Append a (n_channels, n_samples) chunk that has just arrived."""
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] == 0:
            return
        now = time.time() if now is None else float(now)
        with self._lock:
            if self.fs != float(fs) or self.names != list(names) or self._data.shape[0] != chunk.shape[0]:
                self.fs, self.names, self.source = float(fs), list(names), source
                self._data, self._times = np.empty((chunk.shape[0], 0)), np.empty(0)
            self.source = source
            n = chunk.shape[1]
            times = now - (n - 1 - np.arange(n)) / self.fs
            capacity = int(round(self.seconds * self.fs))
            self._data = np.concatenate((self._data, chunk), axis=1)[:, -capacity:]
            self._times = np.concatenate((self._times, times))[-capacity:]

    def info(self):
        with self._lock:
            return dict(fs=self.fs, names=list(self.names), source=self.source)

    def last(self, seconds):
        """(data (n_channels, n), times (n,)) for up to the last `seconds`."""
        with self._lock:
            if not self.fs:
                return np.empty((len(self.names), 0)), np.empty(0)
            n = int(seconds * self.fs)
            return self._data[:, -n:].copy() if n > 0 else self._data[:, :0].copy(), self._times[-n:].copy()


def window_reason(times, n, now=None):
    """Why a window cannot be used yet, in the words alpha_ball_game_v5_openbci uses, or ''."""
    now = time.time() if now is None else now
    if len(times) < n:
        return "Waiting for samples"
    if np.any(np.diff(times) < 0) or np.max(np.diff(times)) > 0.25 or now - times[-1] > 1:
        return "Stream gap - feedback paused"
    return ""


class BridgeServer:
    """A local, authenticated request/response server for one game at a time.

    `handler(command, args)` receives mark / start_recording / stop_recording /
    configure / closed requests from the server thread and returns a reply.
    """

    def __init__(self, tap, handler):
        self.tap, self.handler = tap, handler
        self.key = secrets.token_bytes(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=self.key)
        self.address = self._listener.address
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="game-bridge", daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                connection = self._listener.accept()
            except (OSError, EOFError):
                if self._stop.is_set():
                    return
                continue
            with connection:
                if self._stop.is_set():
                    return
                self._handle(connection)

    def _handle(self, connection):
        while not self._stop.is_set():
            try:
                command, args = connection.recv()
            except (EOFError, OSError):
                self.handler("closed", {})
                return
            try:
                reply = ("ok", self._answer(command, args))
            except Exception as exc:          # the game gets the error, the studio keeps running
                reply = ("error", "{}: {}".format(type(exc).__name__, exc))
            try:
                connection.send(reply)
            except (OSError, EOFError):
                return
            if command == "bye":
                self.handler("closed", {})
                return

    def _answer(self, command, args):
        if command == "hello":
            return self.tap.info()
        if command == "last":
            data, _ = self.tap.last(args["seconds"])
            return data
        if command == "window":
            info = self.tap.info()
            n = int(round(args["seconds"] * (info["fs"] or 0)))
            data, times = self.tap.last(args["seconds"])
            reason = window_reason(times, n) if info["fs"] else "Waiting for samples"
            stamp = float(times[-1]) if len(times) else None
            return data, (None if reason == "Waiting for samples" else stamp), reason
        if command == "bye":
            return None
        return self.handler(command, args)

    def close(self):
        """Stop serving without waiting for the server thread.

        The thread may itself be waiting for the caller (a request handled on
        the UI thread), so nothing here blocks on it: the listening socket is
        closed, and a throw-away connection from a daemon thread wakes a
        pending accept().
        """
        if self._stop.is_set():
            return
        self._stop.set()

        def wake():
            try:
                Client(self.address, authkey=self.key).close()
            except Exception:
                pass
            try:
                self._listener.close()
            except Exception:
                pass

        threading.Thread(target=wake, daemon=True).start()


class BridgeClient:
    """The game-side end: one blocking request at a time."""

    def __init__(self, address, key):
        self._connection = Client(tuple(address), authkey=key)
        self._lock = threading.Lock()

    def request(self, command, **args):
        with self._lock:
            self._connection.send((command, args))
            status, value = self._connection.recv()
        if status != "ok":
            raise RuntimeError("GIOVA: " + str(value))
        return value

    def close(self):
        try:
            self.request("bye")
        except Exception:
            pass
        try:
            self._connection.close()
        except Exception:
            pass
