"""A side pane that folds away to a narrow strip and comes back with a click.

Every pane is open the first time, so nothing is hidden from someone who has
not found the button yet; whether it was left folded is remembered.
"""
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .theme import color

#: Width of a folded pane: room for its name written downwards.
STRIP = 24
_ANY_WIDTH = 16777215        # Qt's QWIDGETSIZE_MAX


class Strip(QtWidgets.QAbstractButton):
    """What is left of a folded pane: an arrow and the pane's name, top to bottom."""

    def __init__(self, title, opens_towards_right=True):
        super().__init__()
        self.title = title
        self.arrow = "›" if opens_towards_right else "‹"
        self.setFixedWidth(STRIP)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Show " + title)

    def sizeHint(self):
        return QtCore.QSize(STRIP, 120)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(color("panel_hover") if self.underMouse() else color("panel")))
        painter.setPen(QtGui.QColor(color("panel_text")))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QtCore.QRect(0, 4, self.width(), 18), QtCore.Qt.AlignmentFlag.AlignCenter, self.arrow)
        font.setBold(False)
        painter.setFont(font)
        painter.translate(self.width(), 28)
        painter.rotate(90)
        painter.drawText(QtCore.QRect(0, 0, max(self.height() - 32, 0), self.width()),
                         QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, self.title)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)


def fold_button(title, closes_towards_left=True):
    """The small arrow in a pane's heading that folds it away."""
    button = QtWidgets.QToolButton()
    button.setText("‹" if closes_towards_left else "›")
    button.setAutoRaise(True)
    button.setToolTip("Hide {} (it folds to a strip at the edge; click the strip to show it again)".format(title))
    return button


class Collapsible(QtWidgets.QWidget):
    """`content` under a heading with a fold button, or folded to a `Strip`.

    `side` is where the pane sits in the window, 'left' or 'right': it folds
    towards that edge. `key` is where the folded state is remembered.
    """

    toggled = QtCore.Signal(bool)        # True when folded

    def __init__(self, title, content, key, preferences, side="left", heading=True):
        super().__init__()
        self.title, self.key, self.preferences = title, key, preferences
        self.collapsed = False
        self._limits = None
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.strip = Strip(title, opens_towards_right=side == "left")
        self.strip.clicked.connect(lambda: self.set_collapsed(False))
        self.strip.setVisible(False)
        self.body = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(self.body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        self.fold = fold_button(title, closes_towards_left=side == "left")
        self.fold.clicked.connect(lambda: self.set_collapsed(True))
        if heading:
            top = QtWidgets.QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            self.heading = QtWidgets.QLabel(title)
            top.addWidget(self.heading, 1)
            top.addWidget(self.fold)
            column.addLayout(top)
        column.addWidget(content, 1)
        row.addWidget(self.strip)
        row.addWidget(self.body, 1)

    def restore(self):
        """Fold the pane if it was left folded. Call once it is in its splitter:
        the splitter sets the open pane's width limits, which folding replaces."""
        if self.preferences.value(self.key + "/collapsed", False, type=bool):
            self.set_collapsed(True, remember=False)

    def set_collapsed(self, collapsed, remember=True):
        collapsed = bool(collapsed)
        if collapsed == self.collapsed:
            return
        # A ResizableSide remembers the open width before the pane becomes a strip,
        # and hands the freed (or needed) room to the main area afterwards.
        splitter = self.parentWidget()
        if collapsed and hasattr(splitter, "save"):
            splitter.save()
        self.collapsed = collapsed
        if collapsed:
            self._limits = (self.minimumWidth(), self.maximumWidth())
            self.body.setVisible(False)
            self.strip.setVisible(True)
            self.setFixedWidth(STRIP)
        else:
            low, high = self._limits or (0, _ANY_WIDTH)
            self.setMinimumWidth(low)
            self.setMaximumWidth(high)
            self.strip.setVisible(False)
            self.body.setVisible(True)
        if remember:
            self.preferences.setValue(self.key + "/collapsed", collapsed)
        if hasattr(splitter, "apply_sizes"):
            splitter.apply_sizes()
        self.toggled.emit(collapsed)
