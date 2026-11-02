"""The Workbench's sheets: tabs, a sheet in a window of its own, and how a run shows its progress.

A Sheet is one pipeline on its own canvas, with its file and its run. The page
holds several, as tabs; a tab dragged off the tab bar (or "Move to its own
window") becomes a window, for a second screen, and closing that window puts it
back. At most MAX_RUNNING sheets compute at once; the others wait in line.

While a sheet runs, its RunStrip says which block is being computed, how far
it is (blocks, and within a block the folds, trials or channels it reports),
the time spent and, once enough is done to judge, the time left; Stop ends it
at the next step. A SignalTrace - two mirrored waves filling the line from the
left - shows both how far it is and that it is alive.
"""
import math
import re
import threading
import time

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..theme import color, muted
from ..widgets import ElidedLabel
from ..icons import icon, put

#: Sheets computing at the same time; the rest wait.
MAX_RUNNING = 2
#: Share of a block's own steps (folds, permutations), and seconds of them, before the time left is guessed:
#: earlier guesses mislead.
GUESS_AFTER = (0.1, 3.0)
#: "3/5", "21/100": the progress within a block, read from its messages.
COUNT = re.compile(r"(\d+)\s*/\s*(\d+)")
#: How far a tab must be dragged off the bar to become a window, pixels: well beyond a slip of the hand.
DETACH_DISTANCE = 90

IDLE, QUEUED, RUNNING, STOPPING, DONE, ERROR, STOPPED = (
    "idle", "queued", "running", "stopping", "done", "error", "stopped")


def clock(seconds):
    seconds = int(round(max(seconds, 0)))
    return "{}:{:02d}".format(seconds // 60, seconds % 60) if seconds < 3600 else \
        "{}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)


class Sheet:
    """One tab: a pipeline on its own canvas, the file it came from, its run."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.path = None
        self.modified = False
        self.errors = {}               # node id -> message from the last run
        self.revision = 0
        self.status = IDLE
        self.targets = None
        self.stop = threading.Event()
        #: the blocks this run computes, in order; those finished; the one being computed
        self.nodes, self.finished, self.current = [], set(), None
        self.inner = 0.0               # share of the current block done, from its own messages
        self.message = ""
        self.started = 0.0
        self.phase, self.phase_started = None, 0.0    # the steps the current block counts, and since when
        self.took = None               # seconds the last run took, once it has ended
        self.accuracy = None           # the latest cross-validated accuracy, for the tab
        self.window = None             # its own window, when out of the tabs

    @property
    def graph(self):
        return self.canvas.graph

    @property
    def busy(self):
        return self.status in (QUEUED, RUNNING, STOPPING)

    def begin(self, nodes):
        self.nodes, self.finished, self.current = list(nodes), set(), None
        self.inner, self.message, self.started = 0.0, "", time.monotonic()
        #: the steps the current block counts ("Fold #/#", "Permutation #/#") and when they began
        self.phase, self.phase_started = None, self.started
        self.status = RUNNING
        # a flag of its own for each run: one stopped and replaced must not be woken up again
        self.stop = threading.Event()

    def advance(self, node_id, text, title):
        """A progress message from the run: the block it is in, and its words."""
        if node_id and node_id != self.current:
            # every block before this one is done, said or not: one another tab computed meanwhile
            # comes from the shared cache without a word
            if node_id in self.nodes:
                self.finished |= set(self.nodes[:self.nodes.index(node_id)])
            elif self.current is not None:
                self.finished.add(self.current)
            self.current, self.inner = node_id, 0.0
        self.message = "" if text == title else text
        found = COUNT.findall(text) if text != title else []
        if found:
            done, total = (int(v) for v in found[-1])
            phase = (self.current, COUNT.sub("#", text))
            if phase != self.phase:              # other steps begin (permutations after folds): their own pace
                self.phase, self.phase_started = phase, time.monotonic()
            if total > 0:
                # "fold 3/5" is said as the third starts: two are done
                self.inner = min(max(done - 1, 0) / total, 1.0)

    @property
    def fraction(self):
        if not self.nodes:
            return 0.0
        return min((len(self.finished) + self.inner) / len(self.nodes), 1.0)

    @property
    def elapsed(self):
        return time.monotonic() - self.started if self.started else 0.0

    def remaining(self):
        """Seconds left of the steps the current block is counting (its folds, its permutations), by their own
        pace; None until it has said how far it is and enough time has passed.

        Not by the share of blocks done: a Recording or a Filter takes a second, a Cross-validate minutes, so
        five quick blocks of seven would promise 0:02 with minutes still to go.
        """
        if self.phase is None or self.phase[0] != self.current or self.inner < GUESS_AFTER[0]:
            return None
        seconds = time.monotonic() - self.phase_started
        if seconds < GUESS_AFTER[1]:
            return None
        return seconds * (1 - self.inner) / self.inner

    def later_blocks(self):
        """Blocks still to come after the current one."""
        if self.current not in self.nodes:
            return []
        return [n for n in self.nodes[self.nodes.index(self.current) + 1:] if n not in self.finished]

    def tab_text(self):
        """The tab's title: the pipeline's name, then how it stands, in words."""
        name = ("*" if self.modified else "") + (self.graph.name or "Pipeline")
        if self.status == QUEUED:
            return name + "  ·  queued"
        if self.status in (RUNNING, STOPPING):
            return name + "  ·  {:.0%}".format(self.fraction)
        if self.status == ERROR:
            return name + "  ·  error"
        if self.status == STOPPED:
            return name + "  ·  stopped"
        if self.status == DONE and self.accuracy is not None:
            return name + "  ·  accuracy {:.1%}".format(self.accuracy)
        return name

    def tab_colour(self):
        return {QUEUED: color("subtle"), RUNNING: color("accent"), STOPPING: color("stale"),
                ERROR: color("bad")}.get(self.status)


class SignalTrace(QtWidgets.QWidget):
    """A run in progress, drawn as GIOVA's agreed trace (the 30 October 2026 draft the user picked): two mirrored
    waves along the strip, fullest in the middle and fading to the line at both ends, travelling steadily in the
    accent colour. Each wave is two rectified rhythms added, as a band-passed signal looks. Not a progress bar:
    how far the run is, and how it ended, are said in words beside it. Idle, a faint flat line that keeps its
    place; animated only while running, so an idle window costs nothing.
    """

    FRAME_MS = 16
    #: phase per frame of the two rhythms (radians at 60 frames a second): 90 % of the draft's 0.12 and 0.20,
    #: the pace the user picked
    STEPS = (0.108, 0.18)
    #: the rhythms' spatial frequencies (radians per pixel) and heights (pixels, for a 26 px strip), as drafted
    WAVES = ((0.09, 6.0), (0.23, 3.0))
    #: one point every this many pixels: smooth at the waves' width, cheap at 60 frames a second
    STRIDE = 2

    def __init__(self, parent=None, height=26):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setMinimumWidth(160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.phases = [0.0, 0.0]
        self.fraction = 0.0
        self.state = None           # None idle, "run", "done", "stopped", "error"
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.FRAME_MS)
        self.timer.timeout.connect(self._step)

    def set_running(self, running):
        if running and not self.timer.isActive():
            self.timer.start()
        elif not running:
            self.timer.stop()
        self.update()

    def show_progress(self, fraction, state):
        """`fraction` of the run done (0-1, said in the strip's words) and how it stands: None, "run", "done",
        "stopped" or "error". Only "run" moves the waves."""
        self.fraction = min(max(float(fraction), 0.0), 1.0)
        self.state = state
        self.set_running(state == "run")

    def _step(self):
        self.phases = [p + step for p, step in zip(self.phases, self.STEPS)]
        self.update()

    def points(self, sign=1):
        """One wave now, a point every STRIDE pixels; `sign` -1 its mirror below the line."""
        w, h = self.width(), self.height()
        scale = h / 26.0
        out = []
        for x in range(0, w + 1, self.STRIDE):
            envelope = math.sin(math.pi * x / max(w, 1))
            height = sum(abs(math.sin(k * x - phase)) * size for (k, size), phase in zip(self.WAVES, self.phases))
            out.append(QtCore.QPointF(x, h / 2 - sign * envelope * height * scale))
        return out

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.state == "run":
            painter.setPen(QtGui.QPen(QtGui.QColor(color("accent")), 1.3))
            painter.drawPolyline(self.points(1))
            painter.drawPolyline(self.points(-1))
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor(color("line")), 1))
            painter.drawLine(QtCore.QPointF(0, self.height() / 2), QtCore.QPointF(self.width(), self.height() / 2))
        painter.end()


#: What the one Run button says in each of its two faces.
RUN_TIP = "Compute every block that has its inputs. Unchanged blocks are not computed again."
STOP_TIP = "End the run at its next step: the blocks already computed keep their results, and Run goes on from there"


def show_run_button(button, sheet):
    """Run, or Stop while `sheet` runs or waits its turn: one button in one place, so stopping needs no trip across the window. An icon alone, like the rest of the toolbar
    (the user's choice, 31 October 2026): its name and F5 on hover and as its accessible name."""
    busy = sheet is not None and sheet.busy
    if button.property("face") != busy:
        word = "Stop" if busy else "Run"
        put(button, "square" if busy else "play", "", primary=True)
        button.setAccessibleName(word)
        button.setToolTip("{} (F5)\n{}".format(word, STOP_TIP if busy else RUN_TIP))
        button.setProperty("icon_only", True)
        button.setProperty("face", busy)
    button.setEnabled(not (busy and sheet.status == STOPPING))


def run_or_stop(page, sheet):
    """What the Run button does now: stop `sheet` if it runs or waits, else run it."""
    return page.stop(sheet) if sheet.busy else page.run(sheet=sheet)


class RunStrip(QtWidgets.QWidget):
    """A sheet's run, along the bottom: the trace that fills as it goes, what is being computed, time.

    Always in its place, at one height: Run sets it moving, the end of the run stills it with how it
    ended. Nothing around it opens, closes or changes size. A click on it opens or folds the panel under
    it (Results and Log, page.py); a run that ended with an error says so on its right.
    """

    clicked = QtCore.Signal()

    def __init__(self, parent=None, opens_panel=True):
        super().__init__(parent)
        self.trace = SignalTrace()
        self.text = ElidedLabel()       # one line however long the message: never wider than the window
        self.chevron = QtWidgets.QLabel()
        self.badge = QtWidgets.QLabel()
        self.badge.setStyleSheet("color: {}; padding: 0 6px;".format(color("warning")))
        self.badge.hide()
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(6, 2, 10, 2)
        if opens_panel:
            row.addWidget(self.chevron)
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Results and Log")
        row.addWidget(self.text, 2)
        row.addWidget(self.trace, 3)
        row.addWidget(self.badge)
        self.opens_panel = opens_panel
        self.set_open(False)
        self.setFixedHeight(max(self.sizeHint().height(), 28))
        self.setObjectName("runstrip")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#runstrip {{ background: {}; border-top: 1px solid {}; }}".format(
            color("header"), color("line")))
        self.show_sheet(None)

    def set_open(self, shown):
        """The chevron: up while the panel is folded (a click opens it), down while it is open."""
        self.chevron.setPixmap(icon("chevron-down" if shown else "chevron-up", role="subtle").pixmap(14, 14))

    def set_errors(self, count):
        self.badge.setText("{} error{}".format(count, "s" if count != 1 else "") if count else "")
        self.badge.setVisible(bool(count))

    def mousePressEvent(self, event):
        if self.opens_panel and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def show_sheet(self, sheet):
        """How `sheet` stands now: moving while it runs, still with how it ended otherwise."""
        busy = sheet is not None and sheet.busy
        state = {RUNNING: "run", STOPPING: "run", DONE: "done", STOPPED: "stopped", ERROR: "error"}.get(
            None if sheet is None else sheet.status)
        self.trace.show_progress(sheet.fraction if state in ("run", "stopped", "error") else 0.0, state)
        self.text.setStyleSheet("" if busy else muted())
        self.text.setText(self.describe(sheet))

    @staticmethod
    def describe(sheet):
        if sheet is None or sheet.status == IDLE:
            return "Not run yet: Run (F5) computes every block that has its inputs"
        if sheet.status == DONE:
            return "Done in {}".format(clock(sheet.took or 0)) + (
                "  ·  accuracy {:.1%}".format(sheet.accuracy) if sheet.accuracy is not None else "")
        if sheet.status == STOPPED:
            return "Stopped after {}: the blocks computed keep their results".format(clock(sheet.took or 0))
        if sheet.status == ERROR:
            return "Ended with an error: the block marked red says why"
        if sheet.status == QUEUED:
            return "Queued: starts when one of the {} pipelines computing now is done".format(MAX_RUNNING)
        if sheet.status == STOPPING:
            return "Stopping at the next step…"
        parts = []
        if sheet.current is not None and sheet.current in sheet.graph.nodes:
            from ...pipeline import block
            node = sheet.graph.nodes[sheet.current]
            parts.append(node.label or block(node.type).title)
        elif sheet.current is not None:
            parts.append(sheet.current)
        if sheet.message:
            parts.append(sheet.message)
        parts.append("block {} of {}".format(min(len(sheet.finished) + 1, len(sheet.nodes)), len(sheet.nodes))
                     if sheet.nodes else "")
        parts.append(clock(sheet.elapsed))
        left = sheet.remaining()
        if left is not None:
            # what comes after this block has not said how long it takes: the guess is for this block
            parts.append("about {} left{}".format(clock(left), " in this block" if sheet.later_blocks() else ""))
        return "  ·  ".join(p for p in parts if p)


class SheetTabBar(QtWidgets.QTabBar):
    """Tabs that can be moved along the bar, and dragged off it to become a window of their own.

    Each tab's close mark is a quiet grey cross, shown on the current tab and on the one under the
    mouse: Qt's own turns red on hover, and on GIOVA's sheet red only means an error."""

    detach = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.setExpanding(False)
        self.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.setMouseTracking(True)
        self._pressed = None
        self._hovered = -1
        self.currentChanged.connect(lambda _index: self._show_close_marks())

    def tabInserted(self, index):
        super().tabInserted(index)
        button = QtWidgets.QToolButton(self)
        button.setIcon(icon("x", role="subtle"))
        button.setIconSize(QtCore.QSize(12, 12))
        button.setFixedSize(18, 18)
        button.setAutoRaise(True)
        button.setToolTip("Close")
        button.setAccessibleName("Close")
        button.setStyleSheet("QToolButton { padding: 0; border-radius: 4px; }")
        button.clicked.connect(lambda _checked=False, b=button: self._close_from(b))
        self.setTabButton(index, QtWidgets.QTabBar.ButtonPosition.RightSide, button)
        self._show_close_marks()

    def _close_from(self, button):
        for index in range(self.count()):
            if self.tabButton(index, QtWidgets.QTabBar.ButtonPosition.RightSide) is button:
                self.tabCloseRequested.emit(index)
                return

    def _show_close_marks(self):
        for index in range(self.count()):
            button = self.tabButton(index, QtWidgets.QTabBar.ButtonPosition.RightSide)
            if button is not None:
                # hidden but keeping its room, so the tab does not change width under the mouse
                effect = button.graphicsEffect() or QtWidgets.QGraphicsOpacityEffect(button)
                effect.setOpacity(1.0 if index in (self.currentIndex(), self._hovered) else 0.0)
                button.setGraphicsEffect(effect)

    def leaveEvent(self, event):
        self._hovered = -1
        self._show_close_marks()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = (self.tabAt(event.position().toPoint()), event.position().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        hovered = self.tabAt(event.position().toPoint())
        if hovered != self._hovered:
            self._hovered = hovered
            self._show_close_marks()
        if self._pressed is not None and self._pressed[0] >= 0:
            point = event.position().toPoint()
            outside = point.y() < -DETACH_DISTANCE or point.y() > self.height() + DETACH_DISTANCE
            if outside:
                index = self.currentIndex()
                self._pressed = None
                # the bar's own drag ends first, or the moved tab stays half-way
                release = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonRelease, event.position(),
                                            event.globalPosition(), QtCore.Qt.MouseButton.LeftButton,
                                            QtCore.Qt.MouseButton.NoButton, event.modifiers())
                super().mouseReleaseEvent(release)
                self.detach.emit(index)
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._pressed = None
        super().mouseReleaseEvent(event)


class SheetWindow(QtWidgets.QWidget):
    """A sheet in a window of its own: its name, Run, its progress, and the way back to the tabs.

    Its block settings and results stay on the Workbench beside the tabs: clicking here shows them there.
    """

    def __init__(self, page, sheet):
        super().__init__(None, QtCore.Qt.WindowType.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.page, self.sheet = page, sheet
        self.run_button = QtWidgets.QPushButton()
        self.run_button.setProperty("primary", True)
        self.run_button.clicked.connect(lambda: run_or_stop(page, sheet))
        back = QtWidgets.QToolButton()
        back.setText("Back to the tabs")
        back.setToolTip("Put this pipeline back among the Workbench's tabs (closing the window does too)")
        back.clicked.connect(self.close)
        self.name = QtWidgets.QLabel()
        font = self.name.font()
        font.setBold(True)
        self.name.setFont(font)
        self.note = QtWidgets.QLabel("Settings and results of the block clicked here: on the Workbench")
        self.note.setStyleSheet(muted())
        self.strip = RunStrip(opens_panel=False)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)
        bar.addWidget(self.name)
        bar.addWidget(self.run_button)
        bar.addWidget(self.note, 1)
        bar.addWidget(back)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addLayout(bar)
        column.addWidget(sheet.canvas, 1)
        column.addWidget(self.strip)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("F5"), self)
        shortcut.activated.connect(lambda: run_or_stop(page, sheet))
        self.resize(1000, 700)
        self.refresh()

    def refresh(self):
        self.setWindowTitle("{} - GIOVA Workbench".format(self.sheet.tab_text()))
        self.name.setText(self.sheet.graph.name)
        show_run_button(self.run_button, self.sheet)
        self.strip.show_sheet(self.sheet)

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.Type.ActivationChange and self.isActiveWindow():
            self.page.activate(self.sheet)          # the Workbench's settings follow this window
        super().changeEvent(event)

    def closeEvent(self, event):
        self.page.attach(self.sheet)
        super().closeEvent(event)
