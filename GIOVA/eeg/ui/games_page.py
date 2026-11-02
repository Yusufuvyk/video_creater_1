"""Playing a game (alpha_ball_game_openbci and its copies) on the Monitor's stream.

The game runs as it always did - its own pygame window, its own calibration,
its own recordings folder - in a separate process. Only its board class is
replaced, so the Monitor keeps the board and the signal stays visible while
the game is played.
"""
import os
from pathlib import Path
import sys

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..core.config import GAMES_DIR
from ..core.game_bridge import BridgeServer
from ..games import manifest as mf
from ..games.studio_eeg import ADDRESS_ENV, KEY_ENV
from .detach import ResizableSide
from .profile_box import menu_button, page_settings
from .widgets import NoScrollComboBox
from .theme import color, muted


def heading(text):
    label = QtWidgets.QLabel(text)
    label.setStyleSheet("font-weight: 600; margin-top: 6px;")
    return label


def game_command(folder):
    """(program, arguments) that start `folder`'s game in a new process."""
    if getattr(sys, "frozen", False):
        return sys.executable, ["--game", str(folder)]
    script = Path(__file__).resolve().parents[2] / "eeg_studio.py"
    return sys.executable, ["-u", str(script), "--game", str(folder)]


class _Invoker(QtCore.QObject):
    """Runs a bridge request on the GUI thread and hands the reply back."""

    request = QtCore.Signal(object)
    notify = QtCore.Signal(str)

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.request.connect(self._run, QtCore.Qt.ConnectionType.BlockingQueuedConnection)
        self.notify.connect(lambda command: handler(command, {}), QtCore.Qt.ConnectionType.QueuedConnection)

    def _run(self, box):
        try:
            box["reply"] = self.handler(box["command"], box["args"])
        except Exception as exc:
            box["error"] = exc

    def __call__(self, command, args):
        if command == "closed":
            # A notification, not a request: never make the bridge thread wait for it.
            self.notify.emit(command)
            return None
        box = dict(command=command, args=args)
        if QtCore.QThread.currentThread() is self.thread():
            self._run(box)
        else:
            self.request.emit(box)
        if "error" in box:
            raise box["error"]
        return box.get("reply")


class GamesPage(QtWidgets.QWidget):
    status_changed = QtCore.Signal(str)
    closed_by_game = QtCore.Signal()

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.preferences = page_settings("Games")
        self.process = None
        self.bridge = None
        self.game = None
        self.invoker = _Invoker(self._request)
        self.closed_by_game.connect(self._game_closed_connection)
        self._build()
        self.reload()
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(500)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        body = self.body = ResizableSide("games/side", self.preferences)
        root.addWidget(body)

        panel = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(panel)
        left.setContentsMargins(4, 4, 8, 4)
        left.setSpacing(4)
        left.addWidget(heading("Game"))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.games = NoScrollComboBox()
        self.games.currentIndexChanged.connect(self._game_chosen)
        row.addWidget(self.games, 1)
        row.addWidget(menu_button("Game folders", [
            ("Add game folder…", self.add_folder),
            ("Open game folder", self.open_folder),
            ("Remove from list", self.remove_folder),
            None,
            ("Reload list", self.reload)], self))
        left.addLayout(row)
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet(muted())
        left.addWidget(self.description)
        left.addWidget(heading("Player"))
        self.ask_form = QtWidgets.QFormLayout()
        self.ask_form.setContentsMargins(0, 0, 0, 0)
        left.addLayout(self.ask_form)
        self.answers = []
        left.addWidget(heading("Signal"))
        self.source_note = QtWidgets.QLabel()
        self.source_note.setWordWrap(True)
        self.source_note.setStyleSheet(muted())
        left.addWidget(self.source_note)
        left.addStretch()
        body.add_side(panel, "left", 310, minimum=260, maximum=640)

        right = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(right)
        column.setContentsMargins(4, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.notice = QtWidgets.QLabel("Choose a game.")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: {}; padding: 4px 2px;".format(color("strong")))
        bar.addWidget(self.notice, 1)
        self.start_button = QtWidgets.QPushButton("Start game")
        self.start_button.setProperty("primary", True)
        self.start_button.setDefault(True)
        self.start_button.setMinimumWidth(110)
        self.start_button.setStyleSheet("QPushButton { padding: 6px 14px; font-weight: 600; }")
        self.start_button.clicked.connect(self.start_game)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_game)
        self.stop_button.setEnabled(False)
        bar.addWidget(self.start_button)
        bar.addWidget(self.stop_button)
        column.addLayout(bar)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(5000)
        self.output.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.output.setPlaceholderText("The game's messages appear here (calibration results, errors).")
        split.addWidget(self.output)
        self.events = QtWidgets.QListWidget()
        self.events.setToolTip("Markers the game sent and its recording, newest last")
        split.addWidget(self.events)
        split.setSizes([500, 160])
        column.addWidget(split, 1)
        body.add_main(right)

    # ------------------------------------------------------------------
    # game list
    # ------------------------------------------------------------------

    def folders(self):
        saved = self.preferences.value("folders", []) or []
        if isinstance(saved, str):
            saved = [saved]
        return [str(GAMES_DIR)] + [f for f in saved if f]

    def reload(self, select=None):
        wanted = select or self.games.currentData()
        self.found = [(m, folder) for m, folder, problem in mf.find(self.folders()) if m is not None]
        self.problems = [(folder, problem) for m, folder, problem in mf.find(self.folders()) if m is None]
        self.games.blockSignals(True)
        self.games.clear()
        for game, folder in self.found:
            self.games.addItem(game.name, str(folder))
            self.games.setItemData(self.games.count() - 1, str(folder), QtCore.Qt.ItemDataRole.ToolTipRole)
        index = self.games.findData(wanted)
        self.games.setCurrentIndex(index if index >= 0 else 0)
        self.games.blockSignals(False)
        self._game_chosen()
        if not self.found:
            self.notice.setText("No games yet. Use … → Add game folder (e.g. alpha_ball_game_openbci).")
        for folder, problem in self.problems:
            self._log("Not loaded: {}: {}".format(folder, problem))

    def current_game(self):
        index = self.games.currentIndex()
        return self.found[index][0] if 0 <= index < len(self.found) else None

    def _game_chosen(self, *args):
        game = self.current_game()
        self.description.setText(game.description if game else "")
        while self.ask_form.rowCount():
            self.ask_form.removeRow(0)
        self.answers = []
        for question in (game.asks if game else []):
            field = QtWidgets.QLineEdit(str(self.preferences.value("answer/" + question, "") or ""))
            field.setPlaceholderText(question)
            self.ask_form.addRow(question.replace(" number", "").replace(" name", ""), field)
            self.answers.append((question, field))
        if game:
            self.notice.setText("{} · {}".format(game.name, game.folder))

    def add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Game folder (e.g. alpha_ball_game_openbci)")
        if not folder:
            return
        path = Path(folder)
        if not (path / mf.FILENAME).is_file():
            answer = QtWidgets.QMessageBox.question(
                self, "Make it playable",
                "{} has no {}.\n\nCreate one? The game's own files are not changed.".format(path.name, mf.FILENAME))
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            try:
                mf.create(path)
            except ValueError as exc:
                self._message("Cannot add game: " + str(exc))
                return
        saved = [f for f in self.folders()[1:] if f != str(path)] + [str(path)]
        self.preferences.setValue("folders", saved)
        self.reload(select=str(path))

    def remove_folder(self):
        folder = self.games.currentData()
        saved = [f for f in self.folders()[1:] if Path(f) != Path(folder or "")]
        self.preferences.setValue("folders", saved)
        self.reload()

    def open_folder(self):
        folder = self.games.currentData()
        if folder:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------

    def _message(self, text):
        self.notice.setText(text)
        self.status_changed.emit(text)

    def _log(self, text):
        self.output.appendPlainText(text.rstrip("\n"))

    def _update_status(self):
        info = self.monitor.tap.info()
        if info["fs"]:
            self.source_note.setText("Monitor: {} · {:g} Hz · {} channels".format(
                info["source"], info["fs"], len(info["names"])))
        else:
            self.source_note.setText("Monitor is not streaming. Start game starts it with the source chosen "
                                     "on the Monitor (live board or file playback).")

    @property
    def running(self):
        return self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning

    def start_game(self):
        game = self.current_game()
        if game is None or self.running:
            return
        if self.monitor.source is None and not getattr(self.monitor, "_starting", False):
            self._message("Starting the Monitor…")
            self.monitor.start()
            if self.monitor.source is None and not getattr(self.monitor, "_starting", False):
                self._message("The Monitor could not start - choose a live board or a file there first.")
                return
        for question, field in self.answers:
            self.preferences.setValue("answer/" + question, field.text())
        self.game = game
        self.bridge = BridgeServer(self.monitor.tap, self.invoker)
        self.events.clear()
        self._log("Started: {}".format(game.name))
        process = self.process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        host, port = self.bridge.address
        environment.insert(ADDRESS_ENV, "{}:{}".format(host, port))
        environment.insert(KEY_ENV, self.bridge.key.hex())
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(game.folder)
        process.readyReadStandardOutput.connect(self._read_output)
        process.started.connect(self._write_answers)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(lambda error, p=process: self._process_error(p, error))
        program, arguments = game_command(game.folder)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._message("{} is starting in its own window…".format(game.name))
        process.start(program, arguments)

    def _process_error(self, process, error):
        if process is not self.process:
            return
        self._log("Process error: {} · {}".format(error, process.errorString()))
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            # Qt does not emit finished when there was no process to finish.
            self.monitor.game_stop_recording()
            if self.bridge is not None:
                self.bridge.close()
                self.bridge = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self._message("Could not start the game: {}. You can try again.".format(process.errorString()))
            self.process = None
            process.deleteLater()

    def _write_answers(self):
        text = "".join(field.text().strip() + "\n" for _, field in self.answers)
        self.process.write(text.encode("utf-8"))
        self.process.closeWriteChannel()

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self._log(line)

    def stop_game(self):
        if not self.running:
            return
        self._message("Stopping {}…".format(self.game.name if self.game else "the game"))
        self.process.terminate()
        QtCore.QTimer.singleShot(3000, lambda p=self.process: p.kill() if p is not None and
                                 p.state() != QtCore.QProcess.ProcessState.NotRunning else None)

    def _finished(self, code, status):
        self._read_output()
        self._log("Game ended (exit code {})".format(code))
        self.monitor.game_stop_recording()
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._message("{} ended.".format(self.game.name if self.game else "Game"))

    def _game_closed_connection(self):
        self.monitor.game_stop_recording()

    def _request(self, command, args):
        """Bridge requests, on the GUI thread."""
        name = self.game.name if self.game else "game"
        if command == "configure":
            reply = self.monitor.game_configure(args["disabled"], args["use_bias"], name)
            self.events.addItem("channels: " + reply)
        elif command == "mark":
            reply = self.monitor.game_mark(args["value"], args["time"])
            self.events.addItem(reply)
        elif command == "start_recording":
            reply = self.monitor.game_start_recording(args["path"])
            self.events.addItem(reply)
        elif command == "stop_recording":
            reply = self.monitor.game_stop_recording()
        elif command == "closed":
            self.closed_by_game.emit()
            reply = None
        else:
            raise ValueError("unknown request " + command)
        self.events.scrollToBottom()
        return reply

    def wait_until_finished(self, seconds):
        deadline = QtCore.QDeadlineTimer(int(seconds * 1000))
        while self.running and not deadline.hasExpired():
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            self.process.waitForFinished(20)
        QtWidgets.QApplication.processEvents()
        return not self.running

    def shutdown(self):
        """On application exit: end the game, release the bridge."""
        if self.running:
            self.process.terminate()
            # Keep the event loop turning: a game request waiting for this thread
            # would otherwise block the game from exiting.
            if not self.wait_until_finished(3.0):
                self.process.kill()
                self.wait_until_finished(2.0)
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
