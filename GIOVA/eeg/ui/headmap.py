"""Picking channels off a head instead of out of a list.

Sixty-four checkboxes in a sidebar is not a channel selector, it is a wall. It
also throws away the one thing that makes a channel meaningful: where on the
scalp it sits. `C3` and `C4` matter because they are over the left and right
motor cortex, and a list cannot show that while a head can.

So `Channels` becomes a button that opens a head. Electrodes are drawn where
`montages.positions_2d` puts them; click one to toggle it, drag a box to take a
region, or use a preset. Anything with no known position - EMG rows, a bare
`CH37` - is still listed beside the head and still selectable, because being
unplottable is not a reason to become unreachable.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..core import montages as mo
from .theme import warning as warning_style
from .widgets import NoScrollComboBox, fit_to_screen
from .plots import TRACE_COLOUR
from .theme import color, plot_background

REPLACE, ADD, REMOVE = "replace", "add", "remove"

#: Above this many electrodes, permanent labels turn the head into soup and
#: names move to hover only. A full 10-10 cap (64) still reads at a sensible
#: dialog size, which is the case that matters; a 128-electrode cap does not.
MAX_LABELS = 70

SELECTED_BRUSH = pg.mkBrush(TRACE_COLOUR)
UNSELECTED_BRUSH = pg.mkBrush(color("block"))
OUTLINE_PEN = pg.mkPen(color("subtle"), width=2)
GOOD_PEN = pg.mkPen(color("dim"), width=1)
BAD_PEN = pg.mkPen("#c0392b", width=2)


class _SelectBox(pg.ViewBox):
    """A ViewBox whose left-drag selects a region instead of zooming.

    Zooming a head is meaningless - it is already the whole scalp - so the drag
    gesture is free to do the useful thing.
    """

    sigRegionSelected = QtCore.Signal(object, object)  # QRectF, mode

    def __init__(self):
        super().__init__(lockAspect=True, enableMenu=False)
        self.setMouseEnabled(x=False, y=False)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        if not ev.isFinish():
            return

        self.rbScaleBox.hide()
        rect = QtCore.QRectF(pg.Point(ev.buttonDownPos()), pg.Point(ev.pos()))
        rect = self.childGroup.mapRectFromParent(rect).normalized()

        modifiers = ev.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            mode = ADD
        elif modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            mode = REMOVE
        else:
            mode = REPLACE
        self.sigRegionSelected.emit(rect, mode)


class HeadView(pg.PlotWidget):
    """Electrodes on a scalp. Click to toggle, drag a box to take a region."""

    selection_changed = QtCore.Signal()

    def __init__(self):
        self._box = _SelectBox()
        super().__init__(viewBox=self._box)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setBackground(plot_background())
        self.setMinimumSize(320, 320)

        self._box.sigRegionSelected.connect(self._region_selected)
        self._names = []          # every channel, in recording order
        self._placed = []         # indices that have a position
        self._points = np.zeros((0, 2))
        self._selected = set()
        self._bad = set()

        self._scatter = pg.ScatterPlotItem(
            size=15, hoverable=True, hoverSize=19,
            tip=lambda x, y, data: str(data))
        self._scatter.sigClicked.connect(self._point_clicked)
        self.addItem(self._scatter)
        self._labels = []

    # ---------------- data ----------------

    def set_channels(self, names, bad=()):
        """Lay out `names`; those with no known position are simply not drawn."""
        self._names = list(names)
        self._bad = set(bad)
        positions = mo.positions_2d(self._names)

        self._placed = [i for i, name in enumerate(self._names)
                        if name in positions]
        self._points = (np.array([positions[self._names[i]] for i in self._placed])
                        if self._placed else np.zeros((0, 2)))
        self._draw_outline()
        self._draw_labels()
        self.refresh()

    @property
    def placed_indices(self):
        return list(self._placed)

    def set_selected(self, indices):
        self._selected = set(int(i) for i in indices)
        self.refresh()

    # ---------------- drawing ----------------

    def _radius(self):
        """Outline radius: the ears at 1.0, or further out if electrodes are.

        Iz and P9/P10 sit below the ear line, and a head drawn inside them looks
        broken. The outline follows the montage rather than the other way round.
        """
        if not len(self._points):
            return 1.0
        return max(1.0, float(np.linalg.norm(self._points, axis=1).max()) * 1.06)

    def _draw_outline(self):
        for item in list(self.items()):
            if isinstance(item, pg.PlotCurveItem):
                self.removeItem(item)
        if not len(self._points):
            return

        r = self._radius()
        angle = np.linspace(0, 2 * np.pi, 240)
        self.addItem(pg.PlotCurveItem(r * np.cos(angle), r * np.sin(angle),
                                      pen=OUTLINE_PEN))
        # Nose, so left and right are never in doubt.
        nose = 0.13 * r
        self.addItem(pg.PlotCurveItem(
            [-nose, 0.0, nose], [r * 0.99, r * 1.16, r * 0.99], pen=OUTLINE_PEN))
        # Ears.
        for side in (-1.0, 1.0):
            t = np.linspace(-0.9, 0.9, 40)
            self.addItem(pg.PlotCurveItem(
                side * (r + 0.055 * r * np.cos(t * 1.6)), 0.34 * r * np.sin(t),
                pen=OUTLINE_PEN))

        pad = r * 1.28
        self.setXRange(-pad, pad, padding=0)
        self.setYRange(-pad, pad, padding=0)

    def _draw_labels(self):
        for item in self._labels:
            self.removeItem(item)
        self._labels = []
        if len(self._placed) > MAX_LABELS:
            return  # names live on hover instead
        for slot, index in enumerate(self._placed):
            text = pg.TextItem(self._names[index], color=color("text"), anchor=(0.5, 0))
            text.setFont(QtGui.QFont("", 6 if len(self._placed) > 40 else 8))
            text.setPos(float(self._points[slot][0]),
                        float(self._points[slot][1]) - 0.045)
            self.addItem(text)
            self._labels.append(text)

    def refresh(self):
        """Repaint the electrodes from the current selection."""
        if not len(self._points):
            self._scatter.setData([], [])
            return
        self._scatter.setData(
            x=self._points[:, 0], y=self._points[:, 1],
            data=[self._names[i] for i in self._placed],
            brush=[SELECTED_BRUSH if i in self._selected else UNSELECTED_BRUSH
                   for i in self._placed],
            pen=[BAD_PEN if i in self._bad else GOOD_PEN for i in self._placed])

    # ---------------- interaction ----------------

    def _point_clicked(self, _scatter, points):
        if not len(points):
            return
        slot = int(points[0].index())
        index = self._placed[slot]
        self._selected.symmetric_difference_update({index})
        self.refresh()
        self.selection_changed.emit()

    def _region_selected(self, rect, mode):
        inside = {self._placed[slot] for slot, (x, y) in enumerate(self._points)
                  if rect.contains(float(x), float(y))}
        if mode == REPLACE:
            # An empty box is how you clear the head, which is worth having.
            self._selected = inside
        elif mode == ADD:
            self._selected |= inside
        else:
            self._selected -= inside
        self.refresh()
        self.selection_changed.emit()

    @property
    def selected(self):
        return sorted(self._selected)


class ChannelSelectorDialog(QtWidgets.QDialog):
    """The head, plus the list and presets that the head alone cannot give."""

    def __init__(self, parent, names, selected=(), bad=()):
        super().__init__(parent)
        self.setWindowTitle("Channels")
        # Big enough that a 64-electrode cap's labels are readable on opening;
        # the dialog is resizable for anyone who wants more.
        fit_to_screen(self, 880, 620)
        self.names = list(names)
        self.bad = set(bad)

        self.head = HeadView()
        self.head.set_channels(self.names, self.bad)
        self.head.selection_changed.connect(self._head_changed)

        root = QtWidgets.QHBoxLayout(self)
        root.addWidget(self.head, 1)
        root.addLayout(self._side_panel(), 0)

        self._set_selected(selected)
        if not self.head.placed_indices:
            self.head.setVisible(False)
            self.setMinimumWidth(360)

    def _side_panel(self):
        column = QtWidgets.QVBoxLayout()

        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(warning_style())
        self.note.setVisible(False)
        column.addWidget(self.note)
        self._check_positions()

        self.preset = NoScrollComboBox()
        self.preset.addItem("Presets...")
        for label, _ in mo.PRESETS:
            self.preset.addItem(label)
        self.preset.activated.connect(self._preset_chosen)
        column.addWidget(self.preset)

        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter, e.g. c3 c4  or  fc")
        self.filter.textChanged.connect(self._refill)
        column.addWidget(self.filter)

        self.list = QtWidgets.QListWidget()
        self.list.setFixedWidth(210)
        self.list.itemChanged.connect(self._item_changed)
        column.addWidget(self.list, 1)

        self.counter = QtWidgets.QLabel("-")
        column.addWidget(self.counter)

        row = QtWidgets.QHBoxLayout()
        for label, action in (("All", self._all), ("None", self._none),
                              ("Invert", self._invert)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(action)
            row.addWidget(button)
        column.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)
        return column

    def _check_positions(self):
        """Say plainly when the head is empty or partly empty, and why."""
        placed = set(self.head.placed_indices)
        missing = [n for i, n in enumerate(self.names) if i not in placed]
        if not missing:
            return
        if not placed:
            self.note.setText(
                "No electrode positions for these names, so there is no head to "
                "draw. Give the recording a montage under Mat layout > Channel "
                "names, or pick from the list.")
        else:
            self.note.setText("{} without a position ({}) are in the list "
                              "only.".format(len(missing), ", ".join(missing[:6])))
        self.note.setVisible(True)

    # ---------------- selection ----------------

    def _set_selected(self, indices):
        # The set is the state; the list widget is only a view of it, so
        # filtering the list can never quietly drop a hidden selection.
        self._selected = {int(i) for i in indices if 0 <= int(i) < len(self.names)}
        self._sync()

    def _sync(self):
        self.head.set_selected(self._selected)
        self._refill()
        self.counter.setText("{} / {} selected".format(len(self._selected),
                                                       len(self.names)))

    def _matches(self, name):
        """Space-separated terms, any of which may match anywhere in the name."""
        terms = self.filter.text().lower().split()
        return not terms or any(term in name.lower() for term in terms)

    def _refill(self):
        placed = set(self.head.placed_indices)
        self.list.blockSignals(True)
        self.list.clear()
        for index, name in enumerate(self.names):
            if not self._matches(name):
                continue
            item = QtWidgets.QListWidgetItem(
                name if index in placed else name + "   (no position)")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if index in
                               self._selected else QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
            if index in self.bad:
                item.setForeground(QtGui.QColor(color("bad")))
            self.list.addItem(item)
        self.list.blockSignals(False)

    def _item_changed(self, item):
        index = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self._selected.add(index)
        else:
            self._selected.discard(index)
        self.head.set_selected(self._selected)
        self.counter.setText("{} / {} selected".format(len(self._selected),
                                                       len(self.names)))

    def _head_changed(self):
        self._selected = set(self.head.selected)
        self._sync()

    def _visible_indices(self):
        """Only what the filter shows: All means all of what you can see."""
        return [i for i, name in enumerate(self.names) if self._matches(name)]

    def _all(self):
        self._set_selected(self._selected | set(self._visible_indices()))

    def _none(self):
        self._set_selected(self._selected - set(self._visible_indices()))

    def _invert(self):
        self._set_selected(self._selected ^ set(self._visible_indices()))

    def _preset_chosen(self, row):
        if row <= 0:
            return
        wanted = {name.upper() for name in dict(mo.PRESETS)[self.preset.itemText(row)]}
        self._set_selected([i for i, name in enumerate(self.names)
                            if str(name).strip().upper() in wanted])
        self.preset.setCurrentIndex(0)

    @classmethod
    def ask(cls, parent, names, selected=(), bad=()):
        """The chosen channel indices, or None if cancelled."""
        dialog = cls(parent, names, selected, bad)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return sorted(dialog._selected)
        return None


class ChannelButton(QtWidgets.QPushButton):
    """The one-line summary that stands in for the whole channel panel."""

    changed = QtCore.Signal()

    #: Names listed before falling back to a bare count.
    MAX_SHOWN = 4

    def __init__(self):
        super().__init__()
        self.names = []
        self.bad = set()
        self._selected = []
        self.clicked.connect(self._open)
        self._sync()

    def set_channels(self, names, selected=None, bad=()):
        self.names = list(names)
        self.bad = set(bad)
        if selected is None:
            selected = self._selected
        self._selected = [i for i in selected if i < len(self.names)]
        self._sync()

    @property
    def selected(self):
        return list(self._selected)

    def set_selected(self, indices, notify=True):
        self._selected = sorted(int(i) for i in indices if i < len(self.names))
        self._sync()
        if notify:
            self.changed.emit()

    def _open(self):
        chosen = ChannelSelectorDialog.ask(self, self.names, self._selected,
                                           self.bad)
        if chosen is not None:
            self.set_selected(chosen)

    def _sync(self):
        if not self.names:
            self._full_text = None
            self.setText("Channels: no recording loaded")
            self.setEnabled(False)
            return
        self.setEnabled(True)
        chosen = [self.names[i] for i in self._selected]
        if not chosen:
            shown = "none"
        elif len(chosen) <= self.MAX_SHOWN:
            shown = ", ".join(chosen)
        else:
            shown = "{} +{}".format(", ".join(chosen[:self.MAX_SHOWN]),
                                    len(chosen) - self.MAX_SHOWN)
        # The count first, so it survives when a narrow sidebar cuts the names.
        self._full_text = "{} / {} · {}".format(len(chosen), len(self.names), shown)
        self.setToolTip("Shown on the graphs ({} / {}):\n{}\n\nClick to choose on the head map."
                        .format(len(chosen), len(self.names), ", ".join(chosen) or "none"))
        self._elide()

    def _elide(self):
        """Fit the summary to the button: cut with '…' instead of clipping both ends."""
        text = getattr(self, "_full_text", None)
        if text is None:
            return
        room = max(self.width() - 24, 40)
        QtWidgets.QPushButton.setText(
            self, self.fontMetrics().elidedText(text, QtCore.Qt.TextElideMode.ElideRight, room))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()
