"""Home: where GIOVA starts - choose a mode, continue where you left off, or start a task.

Three parts on one page: the modes as cards (Quick analysis among them, which opens on the Workbench);
"Continue", the recordings, pipelines (with their last accuracy) and reports used last, one click from
reopening; and "Start a task", the usual jobs, each opening the right mode already set up. Nothing on it
moves.

The program opens straight into the mode used last (unless that is switched off here), and
Home stays one click away in the header.
"""
from pathlib import Path
import time

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import recent
from .preferences import setting, user_settings
from .splash import icon_pixmap
from .theme import color

#: The card that is a task, not a page: a quick analysis on the Workbench.
QUICK = "Quick"
#: (page name, title, what it is for, shortcut) in the order shown, left to right
MODES = (
    ("Monitor", "Live monitor", "Stream from an OpenBCI board or a Unicorn headset, check the electrodes, "
                                "record.", "Ctrl+1"),
    ("Workbench", "Workbench", "Build an analysis from blocks, run it on one recording or many, compare, "
                               "report.", "Ctrl+2"),
    (QUICK, "Quick analysis", "One recording with a ready profile, on the Workbench: spectra, features, "
                              "time-frequency, ERD.", "Ctrl+3"),
    ("Games", "Games", "The festival games, played on the live stream.", "Ctrl+4"),
)
TITLES = {name: title for name, title, _, _ in MODES}

#: (key, title, what it opens) of the tasks Home offers
TASKS = (
    ("record", "Record from a headset", "Live monitor: connect, check the electrodes, record"),
    ("analyse", "Analyse a recording", "Quick analysis on the Workbench: a recording and a ready profile"),
    ("classify", "Classify motor imagery", "Workbench: CSP + LDA, cross-validated"),
    ("literature", "Compare methods from the literature", "Compare: CSP + LDA beside MDRM, TRCSP and FBCSP"),
    ("build", "Build a pipeline from blocks", "Workbench, an empty sheet"),
)
#: entries per list under "Continue"
SHOWN = 4


def start_in_last_mode():
    return str(setting("ui/start_last", True)).lower() != "false"


class Card(QtWidgets.QFrame):
    """A clickable box: a bold title, a line under it, an optional note on the right."""

    clicked = QtCore.Signal()

    def __init__(self, title, text, note="", big=False):
        super().__init__()
        self.setObjectName("homeCard")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(title)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(16, 12 if big else 8, 16, 12 if big else 8)
        column.setSpacing(4 if big else 1)
        top = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel(title)
        self.title.setWordWrap(True)          # a long title takes two lines on a narrow window
        self.title.setStyleSheet("font-size: {}pt; font-weight: 600; color: {};".format(
            13 if big else 10, color("strong")))
        top.addWidget(self.title, 1)
        if note:
            hint = QtWidgets.QLabel(note)
            hint.setStyleSheet("color: {};".format(color("subtle")))
            top.addWidget(hint)
        column.addLayout(top)
        self.text = QtWidgets.QLabel(text)
        self.text.setWordWrap(True)
        self.text.setStyleSheet("color: {};".format(color("subtle")))
        column.addWidget(self.text)
        column.addStretch(1)                 # titles line up across a row of cards

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class RecentRow(QtWidgets.QPushButton):
    """One entry under "Continue": its name, what it is, when it was used."""

    def __init__(self, kind, entry):
        super().__init__()
        self.kind, self.path = kind, entry["path"]
        name = entry.get("name") or Path(self.path).name
        detail = ""
        if kind == recent.PIPELINES and entry.get("accuracy") is not None:
            detail = "  ·  {:.1%}".format(float(entry["accuracy"]))
        self.full = "{}{}  ·  {}".format(name, detail, recent.ago(entry.get("when", time.time())))
        self.setText(self.full)
        self.setToolTip(self.path)
        self.setFlat(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setObjectName("recentRow")

    def minimumSizeHint(self):
        # a long file name ends in … rather than widening the window past the screen
        return QtCore.QSize(self.fontMetrics().horizontalAdvance("M" * 8), super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        room = max(self.width() - 2 * self.fontMetrics().horizontalAdvance("M"), 1)
        self.setText(self.fontMetrics().elidedText(self.full, QtCore.Qt.TextElideMode.ElideMiddle, room))


class HomePage(QtWidgets.QWidget):
    #: a mode's page name, on a mode card
    chosen = QtCore.Signal(str)
    #: a task's key (TASKS)
    task = QtCore.Signal(str)
    #: (kind, path) of an entry under "Continue"
    reopen = QtCore.Signal(str, str)

    def __init__(self):
        super().__init__()
        self.hidden = set()
        self.cards = {}
        self.setStyleSheet(self._style())
        # the window's own ground behind the cards, not the tab page's lighter one
        self.setAutoFillBackground(True)
        ground = self.palette()
        ground.setColor(self.backgroundRole(), QtGui.QColor(color("canvas")))
        self.setPalette(ground)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 18, 24, 16)
        body = QtWidgets.QWidget()
        body.setMaximumWidth(1180)
        outer.addWidget(body, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)
        column = QtWidgets.QVBoxLayout(body)
        column.setSpacing(14)

        head = QtWidgets.QHBoxLayout()
        mark = QtWidgets.QLabel()
        from .theme import window_is_dark
        pixmap = icon_pixmap(80, light=not window_is_dark())
        pixmap.setDevicePixelRatio(2.0)
        mark.setPixmap(pixmap)
        head.addWidget(mark)
        words = QtWidgets.QVBoxLayout()
        words.setSpacing(0)
        from .. import NAME, TAGLINE
        name = QtWidgets.QLabel(NAME)
        name.setStyleSheet("font-size: 20pt; font-weight: 700; color: {};".format(color("strong")))
        words.addWidget(name)
        lab = QtWidgets.QLabel(TAGLINE)
        lab.setStyleSheet("color: {}; letter-spacing: 1px;".format(color("subtle")))
        words.addWidget(lab)
        head.addLayout(words)
        head.addStretch(1)
        column.addLayout(head)

        modes = QtWidgets.QHBoxLayout()
        modes.setSpacing(12)
        for name_, title, text, keys in MODES:
            card = Card(title, text, keys, big=True)
            card.clicked.connect(lambda n=name_: self.choose(n))
            self.cards[name_] = card
            modes.addWidget(card, 1)
        column.addLayout(modes)

        lower = QtWidgets.QHBoxLayout()
        lower.setSpacing(24)
        self.continue_box = QtWidgets.QVBoxLayout()
        self.continue_box.setSpacing(2)
        left = QtWidgets.QWidget()
        left.setLayout(self.continue_box)
        lower.addWidget(self._section("Continue", left), 3)
        tasks = QtWidgets.QWidget()
        tasks_column = QtWidgets.QVBoxLayout(tasks)
        tasks_column.setContentsMargins(0, 0, 0, 0)
        tasks_column.setSpacing(6)
        self.tasks = {}
        for key, title, text in TASKS:
            card = Card(title, text)
            card.clicked.connect(lambda k=key: self.task.emit(k))
            self.tasks[key] = card
            tasks_column.addWidget(card)
        tasks_column.addStretch(1)
        lower.addWidget(self._section("Start a task", tasks), 2)
        column.addLayout(lower, 1)

        self.remember = QtWidgets.QCheckBox("Open {} in the mode I used last".format(NAME))
        self.remember.setChecked(start_in_last_mode())
        self.remember.toggled.connect(lambda on: user_settings().setValue("ui/start_last", bool(on)))
        outer.addWidget(self.remember, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.refresh()

    def _style(self):
        return ("QFrame#homeCard {{ background: {card}; border: 1px solid {line}; border-radius: 10px; }}"
                "QFrame#homeCard:hover, QFrame#homeCard:focus {{ border-color: {border}; background: {hover}; }}"
                "QPushButton#recentRow {{ text-align: left; border: none; padding: 5px 8px; color: {text}; "
                "background: transparent; border-radius: 6px; }}"
                "QPushButton#recentRow:hover {{ background: {hover}; }}").format(
            card=color("card"), line=color("line"), border=color("hover_border"), hover=color("hover"),
            text=color("text"))

    @staticmethod
    def _section(title, widget):
        box = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        heading = QtWidgets.QLabel(title)
        heading.setStyleSheet("font-size: 11pt; font-weight: 600; color: {};".format(color("strong")))
        column.addWidget(heading)
        column.addWidget(widget, 1)
        return box

    def refresh(self):
        """The "Continue" lists as they are now."""
        while self.continue_box.count():
            widget = self.continue_box.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)        # gone from the page now, not at the next event loop
                widget.deleteLater()
        for kind, title, empty in ((recent.RECORDINGS, "Recordings", "A recording opened anywhere shows here."),
                                   (recent.PIPELINES, "Pipelines",
                                    "A pipeline saved or opened on the Workbench, with its last accuracy."),
                                   (recent.REPORTS, "Reports", "A report saved from the Workbench or Compare.")):
            label = QtWidgets.QLabel(title)
            label.setStyleSheet("color: {}; font-weight: 600; margin-top: 6px;".format(color("subtle")))
            self.continue_box.addWidget(label)
            entries = recent.entries(kind)[:SHOWN]
            if not entries:
                note = QtWidgets.QLabel(empty)
                note.setWordWrap(True)
                note.setStyleSheet("color: {}; padding: 3px 8px;".format(color("dim")))
                self.continue_box.addWidget(note)
            for entry in entries:
                row = RecentRow(kind, entry)
                row.clicked.connect(lambda _c=False, k=kind, p=entry["path"]: self.reopen.emit(k, p))
                self.continue_box.addWidget(row)
        self.continue_box.addStretch(1)

    def choose(self, name):
        """As a click on that mode's card; Quick analysis is the task of that name."""
        if self.is_shown(name):
            if name == QUICK:
                self.task.emit("analyse")
            else:
                self.chosen.emit(name)
            return True
        return False

    def is_shown(self, name):
        return name in self.cards and name not in self.hidden

    def show_games(self, shown):
        if shown:
            self.hidden.discard("Games")
        else:
            self.hidden.add("Games")
        self.cards["Games"].setVisible(shown)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
