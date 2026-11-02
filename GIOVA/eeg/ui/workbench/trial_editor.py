"""The Epochs block's trial table, edited beside its settings: a timeline to drag a span on, and the table.

The table is the block's `trials` parameter and nothing else: every change is written there at once (so it is
saved with the pipeline, reaches the script, and makes the blocks after it run again), and nothing drawn here
is a trial the pipeline does not have. After a run, each trial's status says what the block made of it:
used, left out with its reason (too short for the common duration, left out of the selection, missing
samples, amplitude), or used and marked for review (flat, unusually large) - a review mark never leaves a
trial out by itself.

Qt widgets only; the trials are judged by `pipeline.compute`, as the block judges them.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..plots import CHANNEL_COLORS
from ..theme import color, muted
from .previews import small_plot

#: The table's columns.
COLUMNS = ("ID", "Condition", "Start (s)", "End (s)", "Included", "Status")
ID, CONDITION, START, END, INCLUDED, STATUS = range(len(COLUMNS))
#: How many samples of the timeline's channel are drawn at most.
TIMELINE_POINTS = 20000


def statuses(epochs):
    """{trial id: what the block made of it} from its output: used (with any review marks) or the reason left out."""
    if epochs is None:
        return {}
    found = {i: "Used" for i in epochs.ids}
    for meta in epochs.meta or ():
        if meta.get("flags"):
            found[meta["id"]] = "Used · review: " + "; ".join(meta["flags"])
    for trial_id, _label, reason in epochs.rejected:
        found[str(trial_id)] = reason
    return found


def next_id(rows):
    """One more than the largest plain-number id, as the Analysis page numbered a trial added by hand."""
    numbers = [int(r[0]) for r in rows if str(r[0]).isdigit()]
    return str(max(numbers + [0]) + 1)


class TrialEditor(QtWidgets.QWidget):
    """`rows` [id, condition, start, end, included]; `raw` the recording reaching the block (a timeline to drag a
    span on), `epochs` its last output (each trial's status). `edited` carries the whole new table."""

    edited = QtCore.Signal(list)

    def __init__(self, rows, raw=None, epochs=None, parent=None):
        super().__init__(parent)
        self.rows = [list(r) + [True] * (5 - len(r)) for r in rows or []]
        self.raw, self.status = raw, statuses(epochs)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self.region = None
        self.timeline = None
        if raw is not None and raw.n_samples > 1:
            self.timeline = small_plot(120)
            self.timeline.setLabel("bottom", "Recording time (s)")
            self.timeline.getPlotItem().hideAxis("left")
            step = max(1, raw.n_samples // TIMELINE_POINTS)
            times = np.arange(0, raw.n_samples, step) / raw.fs
            self.timeline.plot(times, np.nan_to_num(raw.data[0, ::step]), pen=pg.mkPen(color("subtle"), width=1))
            end = min(10.0, raw.duration)
            self.region = pg.LinearRegionItem((0.0, end), bounds=(0.0, raw.duration))
            self.region.setZValue(10)
            self.timeline.addItem(self.region)
            self.shades = []
            column.addWidget(self.timeline)
        add_row = QtWidgets.QHBoxLayout()
        self.condition = QtWidgets.QLineEdit()
        self.condition.setPlaceholderText("Condition of the span")
        self.add_button = QtWidgets.QPushButton("Add span")
        self.add_button.setToolTip("Add the span dragged on the timeline as a trial of this condition")
        self.add_button.clicked.connect(self.add_span)
        self.add_button.setEnabled(self.region is not None)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setToolTip("Remove the trials selected in the table")
        self.remove_button.clicked.connect(self.remove_selected)
        for widget in (self.condition, self.add_button, self.remove_button):
            add_row.addWidget(widget)
        column.addLayout(add_row)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(160)
        self.table.itemChanged.connect(self._cell_changed)
        column.addWidget(self.table)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet(muted())
        column.addWidget(self.message)
        self._fill()

    # ---------------- the table ----------------

    def _fill(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for k, (trial_id, condition, start, end, included) in enumerate(self.rows):
            for col, text in ((ID, str(trial_id)), (CONDITION, str(condition)), (START, "{:g}".format(start)),
                              (END, "{:g}".format(end))):
                self.table.setItem(k, col, QtWidgets.QTableWidgetItem(text))
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled |
                           QtCore.Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(QtCore.Qt.CheckState.Checked if included else QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(k, INCLUDED, check)
            status = QtWidgets.QTableWidgetItem(self.status.get(str(trial_id), "Run to see"))
            status.setFlags(status.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(k, STATUS, status)
        self.table.blockSignals(False)
        self._shade()
        used = sum(1 for r in self.rows if self.status.get(str(r[0]), "").startswith("Used"))
        self.message.setText("{} trials; {}. Review marks never leave a trial out: untick Included to leave it "
                             "out.".format(len(self.rows), "{} used in the last run".format(used) if self.status
                                           else "run to see which are used"))

    def _shade(self):
        """Each trial as a shaded span on the timeline: its condition's colour, grey when left out."""
        if self.timeline is None:
            return
        for item in self.shades:
            self.timeline.removeItem(item)
        self.shades = []
        conditions = list(dict.fromkeys(r[1] for r in self.rows))
        for trial_id, condition, start, end, included in self.rows:
            tint = pg.mkColor(CHANNEL_COLORS[conditions.index(condition) % len(CHANNEL_COLORS)] if included
                              else color("dim"))
            tint.setAlpha(50)
            shade = pg.LinearRegionItem((start, end), movable=False, brush=tint, pen=pg.mkPen(None))
            shade.setZValue(-10)
            self.timeline.addItem(shade)
            self.shades.append(shade)

    def _cell_changed(self, item):
        k, col = item.row(), item.column()
        row = list(self.rows[k])
        try:
            if col == INCLUDED:
                row[4] = item.checkState() == QtCore.Qt.CheckState.Checked
            elif col in (ID, CONDITION):
                text = item.text().strip()
                if not text:
                    raise ValueError("An ID and a condition are needed")
                if col == ID and any(str(r[0]) == text for j, r in enumerate(self.rows) if j != k):
                    raise ValueError("Trial ID {} is taken".format(text))
                row[col] = text
            elif col in (START, END):
                try:
                    row[col] = float(item.text().replace(",", "."))
                except ValueError:
                    row[col] = float("nan")
                if not np.isfinite(row[col]):
                    raise ValueError("A time must be a number of seconds")
            else:
                return
            if not 0 <= row[2] < row[3]:
                raise ValueError("A trial needs 0 ≤ start < end")
        except ValueError as exc:
            self._fill()                         # back to the table as it was: nothing half-typed is kept
            self.message.setText(str(exc))
            return
        self.rows[k] = row
        self._changed()

    def _changed(self):
        # the table changed: what was judged in the last run no longer applies to it
        self.status = {}
        self._fill()
        self.edited.emit([list(r) for r in self.rows])

    # ---------------- adding and removing ----------------

    def add_span(self):
        """The span dragged on the timeline becomes a trial of the condition typed."""
        condition = self.condition.text().strip()
        if self.region is None:
            return False
        if not condition:
            self.message.setText("Type the condition of the span first")
            return False
        start, end = (float(v) for v in self.region.getRegion())
        if not end > start:
            self.message.setText("Drag a span on the timeline first")
            return False
        self.rows.append([next_id(self.rows), condition, round(start, 6), round(end, 6), True])
        self._changed()
        return True

    def remove_selected(self):
        chosen = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        if not chosen:
            self.message.setText("Select the trials to remove in the table")
            return False
        for k in chosen:
            del self.rows[k]
        self._changed()
        return True
