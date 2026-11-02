"""What a block produced, shown under its settings after a run.

One small view per kind of data: a few seconds of the recording, the condition
means of the trials, the spectrum, a feature table, the classification scores,
an ERD map. Enough to see that a step did what was meant before going on;
the Analysis page remains the place for a full look.
"""
import csv
from pathlib import Path
import re

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ...core import epochs as ep
from ...core.spectral import POWER_DENSITY
from ...pipeline import graph as g
from ...pipeline.data import Contrast, Epochs, Features, IcaResult, Spectrum
from ..plots import CHANNEL_COLORS, MARKER_COLOUR, channel_pen, erd_colormap, tf_scale
from ..theme import color, muted, paper, plot_background, warning
from ..widgets import NoScrollComboBox, NoScrollSpinBox
from ..icons import only, put

#: Curves of many points are drawn one pixel wide: Qt draws any other width (1.1, 1.5, 2) with antialiasing
#: through a path a hundred times slower - 50 trials of 5000 samples took 4.5 s to repaint at 1.1, 36 ms at 1.

#: Seconds of the recording drawn in its preview.
RAW_SECONDS = 10.0
#: Channels drawn in the recording preview; in its larger window.
RAW_CHANNELS = 8
RAW_CHANNELS_LARGE = 32
#: Rows a drop-down shows before it scrolls: a 64-channel list must not fill the screen.
LIST_ROWS = 15
#: The channel a preview opens on, the first of these the data has: where most effects are looked for.
FIRST_CHANNELS = ("C3", "Cz", "C4", "Oz", "O1", "Pz", "Fz")
#: A plot sized for the settings column is this much taller in a report, where the page is wide.
REPORT_GROWTH = 1.8


def kind_of(value):
    from ...core.sources import Recording
    from ...pipeline.learn import Model, Scores
    for cls, kind in ((Recording, g.RAW), (Epochs, g.EPOCHS), (Spectrum, g.SPECTRUM), (Features, g.FEATURES),
                      (Model, g.MODEL), (Scores, g.SCORES), (ep.EpochResult, g.TF), (IcaResult, g.ICA),
                      (Contrast, g.CONTRAST)):
        if isinstance(value, cls):
            return kind
    return None


def result_line(value):
    """The few words written under a block on the sheet after it ran."""
    kind = kind_of(value)
    if kind == g.RAW:
        events = len(value.events) if value.events is not None else 0
        return "{} ch · {:.0f} s · {} events".format(value.n_channels, value.duration, events)
    if kind == g.EPOCHS:
        return "{} trials{}".format(len(value.labels), " · {} rejected".format(len(value.rejected))
                                    if value.rejected else "")
    if kind == g.SPECTRUM:
        return "{} trials × {} ch".format(*value.power.shape[:2])
    if kind == g.FEATURES:
        return "{} trials × {} features".format(*value.values.shape)
    if kind == g.SCORES:
        return "accuracy {:.1%}{}".format(value.accuracy, "" if value.better_than_random else " (random)")
    if kind == g.TF:
        return "{} channels · {}".format(len(value.tf), ", ".join(value.conditions))
    if kind == g.MODEL:
        return value.describe()
    if kind == g.ICA:
        return "{} components · {} removed".format(value.n_components, len(value.excluded))
    if kind == g.CONTRAST:
        return "{} trials · {} of {} with p (Holm) ≤ 0.05".format(value.n_trials, len(value.significant()),
                                                                  len(value.rows))
    return ""


def preview(value, large=False):
    """A widget showing `value`, or None when there is nothing to draw. `large`: for its own window."""
    kind = kind_of(value)
    builder = {g.RAW: RawPreview, g.EPOCHS: EpochsPreview, g.SPECTRUM: SpectrumPreview,
               g.FEATURES: FeaturesPreview, g.SCORES: ScoresPreview, g.TF: TFPreview,
               g.ICA: IcaPreview, g.CONTRAST: ContrastPreview}.get(kind)
    if builder is not None:
        return builder(value, large).with_tools()
    if kind == g.MODEL:
        return note("Steps: {}. They are fitted inside Cross-validate, fold by fold.".format(value.describe()))
    return None


def note(text):
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def small_plot(height=200, large=False):
    """A plot `height` pixels high beside the settings; in a larger window it grows with the window."""
    plot = pg.PlotWidget()
    plot.setBackground(plot_background())
    if large:
        plot.setMinimumHeight(int(height * 1.8))
    else:
        plot.setFixedHeight(height)
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.setMenuEnabled(False)
    # the wheel scrolls the settings column; Zoom under the preview unlocks wheel zoom and drag
    plot.setMouseEnabled(x=False, y=False)
    plot.getAxis("left").enableAutoSIPrefix(False)      # 0.4 stays 0.4, not "400 (x0.001)"
    return plot


#: How sharp a saved or reported picture is: device pixels per screen pixel.
PICTURE_SCALE = 2


def picture(widget, scale=PICTURE_SCALE):
    """`widget` as it is drawn now, at `scale` times its pixels: crisp when printed or enlarged."""
    size = widget.size()
    image = QtGui.QImage(size.width() * scale, size.height() * scale, QtGui.QImage.Format.Format_ARGB32)
    image.setDevicePixelRatio(scale)
    image.fill(QtGui.QColor("#ffffff"))
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # QWidget's own render: a plot (QGraphicsView) has another one, for its scene only.
        # Children only: on white, not on the window's grey, as on a page.
        QtWidgets.QWidget.render(widget, painter, QtCore.QPoint(), QtGui.QRegion(),
                                 QtWidgets.QWidget.RenderFlag.DrawChildren)
    finally:
        painter.end()          # a painter left open aborts the program when it is collected
    return image


def legend(plot):
    item = plot.addLegend(offset=(-5, 5))
    item.setLabelTextColor(color("text"))
    return item


def short_list(box):
    """A drop-down that opens as a scrolling list of LIST_ROWS, not a menu as tall as the screen."""
    box.setStyleSheet("combobox-popup: 0;")
    box.setMaxVisibleItems(LIST_ROWS)
    return box


def _choice(widget):
    """What a chooser (a drop-down or a number box) shows, in words."""
    return widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else widget.text()


#: Qt's "no limit" for a widget's size.
QWIDGETSIZE_MAX = 16777215

#: The channel chooser's entries beside the channels themselves.
ALL_CHANNELS, MEAN_OF, MEAN_PREFIX = "All channels", "Mean of channels…", "Mean of "
#: One band's head map, step by step in time (time-frequency only).
OVER_TIME = "Head maps over time"
#: Columns of the all-channels grid; channels drawn in it at most.
GRID_COLUMNS, GRID_MOST = 4, 64


#: 10-20 rows, front to back (FT with FC, T with C, TP with CP: one line across the head), and a name's parts.
HEAD_ROWS = {"FP": 0, "AF": 1, "F": 2, "FT": 3, "FC": 3, "T": 4, "C": 4, "TP": 5, "CP": 5, "P": 6, "PO": 7,
             "O": 8, "I": 9}
TEN_TWENTY = re.compile(r"^(FP|AF|FT|FC|TP|CP|PO|F|T|C|P|O|I)(Z|\d+)$")


def head_order(names, indices):
    """`indices` in the order their electrodes sit on the head: front to back, row by row (Fp, AF, F, FC,
    C, CP, P, PO, O), left to right in a row (odd numbers left, the outermost first, z in the middle, even
    right), so neighbours on the scalp are neighbours in the grid. Other names follow in their own order."""
    def place(name):
        found = TEN_TWENTY.match(str(name).strip().upper())
        if not found:
            return None
        row, number = found.groups()
        x = 0 if number == "Z" else (-int(number) if int(number) % 2 else int(number))
        return HEAD_ROWS[row], x
    known = [i for i in indices if place(names[i]) is not None]
    return sorted(known, key=lambda i: place(names[i])) + [i for i in indices if place(names[i]) is None]


class ChannelGrid(pg.GraphicsLayoutWidget):
    """Every channel side by side, GRID_COLUMNS to a row, the axes linked: what differs across the head."""

    def __init__(self, large=False):
        super().__init__()
        self.setBackground(plot_background())
        self.large = large
        self.items = []
        #: a colour bar beside the last plot, kept with the plots (time-frequency maps)
        self.bar = None
        self._made_for = None

    def fill(self, names, x_label="", link_y=True):
        """One empty plot per name, titled with it; returns them. The same names again: the same plots,
        emptied - making 64 plots takes seconds, emptying them none, so a change of condition or trial is
        drawn at once."""
        made_for = (list(names), x_label, link_y)
        if self.items and made_for == self._made_for:
            for plot in self.items:
                plot.clear()
            return self.items
        self._made_for = made_for
        self.bar = None
        self.clear()
        self.items = []
        rows = int(np.ceil(len(names) / GRID_COLUMNS))
        for k, name in enumerate(names):
            plot = self.addPlot(row=k // GRID_COLUMNS, col=k % GRID_COLUMNS, title=name)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("left").setWidth(48)      # one width for all: the plots line up column by column
            if k // GRID_COLUMNS == rows - 1 and x_label:
                plot.setLabel("bottom", x_label)
            if self.items:
                plot.setXLink(self.items[0])
                if link_y:
                    plot.setYLink(self.items[0])
            self.items.append(plot)
        height = int(rows * (190 if self.large else 135) + 10)
        if self.large:
            # in a window of its own: at least this tall, and as tall as the window allows
            self.setMinimumHeight(height)
            self.setMaximumHeight(QWIDGETSIZE_MAX)
        else:
            self.setFixedHeight(height)
        return self.items


def mean_text(names, index):
    shown = [names[i] for i in index]
    return MEAN_PREFIX + (", ".join(shown) if len(shown) <= 6 else "{} channels".format(len(shown)))


class _Preview(QtWidgets.QWidget):
    def __init__(self, value, large=False):
        super().__init__()
        self.value, self.large = value, large
        #: widgets for working on screen (choosers, save buttons): hidden in a report
        self.controls = []
        self.choosers = []
        self.column = QtWidgets.QVBoxLayout(self)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(6)

    def add_note(self, text, quiet=True):
        label = note(text)
        if quiet:
            label.setStyleSheet(muted())
        self.column.addWidget(label)
        return label

    def _add_grid(self):
        """The all-channels grid, made the first time it is asked for, under the one-channel plot."""
        grid = ChannelGrid(self.large)
        self.column.insertWidget(self.column.indexOf(self.plot) + 1, grid)
        self.fill_height()
        return grid

    def fill_height(self):
        """In a window of its own: the graphs share the window's height, notes and choosers take only what they
        need, so a maximised window is filled by graphs, not by gaps. True when something grows."""
        if not self.large:
            return False
        grows = False
        for k in range(self.column.count()):
            widget = self.column.itemAt(k).widget()
            if isinstance(widget, (pg.PlotWidget, pg.GraphicsLayoutWidget))                     and widget.maximumHeight() > widget.minimumHeight():
                self.column.setStretchFactor(widget, 1)
                grows = True
            elif isinstance(widget, QtWidgets.QLabel):
                widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        return grows

    def channel_box(self, names, draw, several=False, head=False):
        """The channel chooser. `several`: also every channel side by side, and the mean of channels picked
        (a region of interest: their spectra or maps averaged, not their signals summed first). `head`: also
        head maps over time."""
        box = short_list(NoScrollComboBox())
        box.addItems(list(names))
        self.channel_names, self.mean_of = list(names), []
        first = next((n for n in FIRST_CHANNELS if n in names), None)
        if first is not None:
            box.setCurrentIndex(list(names).index(first))
        if several and len(names) > 1:
            box.insertSeparator(box.count())
            box.addItem(ALL_CHANNELS)
            if head:
                box.addItem(OVER_TIME)
            box.addItem(MEAN_OF)
            box.setItemData(box.count() - 1, "Pick channels; their spectra or maps are averaged",
                            QtCore.Qt.ItemDataRole.ToolTipRole)
            box.activated.connect(lambda _i: self._pick_mean(box, draw) if box.currentText() == MEAN_OF else None)
        box.currentIndexChanged.connect(lambda _i: draw() if box.currentText() != MEAN_OF else None)
        label = QtWidgets.QLabel("Channel")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(box, 1)
        self.column.addLayout(row)
        self.controls += [label, box]
        self.choosers.append(("Channel", box))
        return box

    def _pick_mean(self, box, draw):
        """Ask which channels to average; their mean becomes an entry of the chooser, chosen."""
        from .inspector import PickDialog
        names = self.channel_names
        dialog = PickDialog("Mean of channels", names, {names[i] for i in self.mean_of} or set(names[:3]), self)
        chosen = [n for n in dialog.chosen()] if dialog.exec() else []
        index = [names.index(n) for n in chosen if n in names]
        box.blockSignals(True)
        for k in reversed(range(box.count())):
            if box.itemText(k).startswith(MEAN_PREFIX):
                box.removeItem(k)
        if index:
            self.mean_of = index
            box.insertItem(box.count() - 1, mean_text(names, index))
            box.setCurrentIndex(box.count() - 2)
        else:
            box.setCurrentIndex(0)
        box.blockSignals(False)
        draw()

    def channels_chosen(self):
        """("one", [index]), ("all" / "over time", every index) or ("mean", the indices averaged)."""
        text = self.channel.currentText()
        every = {ALL_CHANNELS: "all", OVER_TIME: "over time"}
        if text in every:
            return every[text], list(range(len(self.channel_names)))
        if text.startswith(MEAN_PREFIX) and self.mean_of:
            return "mean", list(self.mean_of)
        index = self.channel.currentIndex()
        return "one", [index if 0 <= index < len(self.channel_names) else 0]

    def with_tools(self):
        """[Zoom] [Reset] [Save image…] under a preview that draws plots; wheel zoom stays off until Zoom."""
        from ..plots import ZoomControls, plot_items
        if not plot_items(self):
            return self
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        zoom = ZoomControls(self)
        zoom.zoom.setToolTip("Allow mouse-wheel zoom and drag-pan on these graphs; Reset fits them again")
        save = QtWidgets.QToolButton()
        only(save, "image-down", "Save image", size=16)
        save.setToolTip("These graphs as a PNG on white, at twice the screen's sharpness")
        save.clicked.connect(lambda: self.save_image())
        row.addWidget(zoom)
        row.addWidget(save)
        row.addStretch(1)
        self.column.addLayout(row)
        self.controls += [zoom, save]
        return self

    def save_image(self, path=None):
        """The preview as it is on screen, without its buttons, as a PNG."""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "figure.png", "PNG image (*.png)")
            if not path:
                return None
        from .report import render
        # drawn again on white, as in a report, whatever the window's theme; the same choices made
        with paper():
            copy = type(self)(self.value, self.large).choose(self.choices())
            image = render(copy, width=self.width())
        image.save(str(path), "PNG")
        return path

    def choices(self):
        """What is chosen on screen now (channel, span, page…), to be chosen again in another copy of this
        preview: the larger window, a picture, the same block shown again."""
        state = []
        for name, box in self.choosers:
            if isinstance(box, QtWidgets.QComboBox):
                state.append((name, box.currentText()))
            else:
                state.append((name, box.value()))
        names = getattr(self, "channel_names", [])
        # channels by name: a pipeline saved and opened on another recording picks the same electrodes or none
        return {"mean_of": [names[i] for i in getattr(self, "mean_of", []) if i < len(names)], "choosers": state}

    def choose(self, state):
        """Choose again what `choices()` gave (or a pipeline file kept), chooser by chooser in this preview's order
        (a condition before its trials), each found by its name; a choice this preview has no place for, or
        whose value it does not offer (a channel not in this recording), is left as it is. Returns self."""
        if not state:
            return self
        if state.get("mean_of") and hasattr(self, "channel_names"):
            names = self.channel_names
            self.mean_of = [names.index(c) if isinstance(c, str) else c for c in state["mean_of"]
                            if (c in names if isinstance(c, str) else 0 <= c < len(names))]
        given = {}
        for same, value in state.get("choosers", ()):
            given.setdefault(same, value)
        for name, box in self.choosers:
            if name not in given:
                continue
            value = given[name]
            if not isinstance(box, QtWidgets.QComboBox):
                box.setValue(value)
            elif box.isEditable():
                box.setCurrentText(value)             # a typed value is not among its items
            elif box.findText(value) >= 0:
                box.setCurrentIndex(box.findText(value))
            elif str(value).startswith(MEAN_PREFIX) and self.mean_of and box.findText(MEAN_OF) >= 0:
                # a mean of channels picked there: the same entry here, before "Mean of channels…"
                box.blockSignals(True)
                box.insertItem(box.findText(MEAN_OF), mean_text(self.channel_names, self.mean_of))
                box.blockSignals(False)
                box.setCurrentIndex(box.findText(MEAN_OF) - 1)
        return self

    def said_choices(self):
        """The (name, chooser) pairs a report caption states: those that shape what is drawn."""
        return self.choosers

    def for_report(self):
        """Ready to be pictured in a report: no controls, taller plots, and what they had chosen said in words."""
        for widget in self.controls:
            widget.hide()
        for plot in self.findChildren(pg.PlotWidget):
            if plot.minimumHeight() == plot.maximumHeight():      # fixed beside the settings: give it room
                plot.setFixedHeight(int(plot.minimumHeight() * REPORT_GROWTH))
        self.report_caption = QtWidgets.QLabel()
        self.report_caption.setStyleSheet("font-weight: 600;")
        self.column.insertWidget(0, self.report_caption)
        self.say_choices()
        return self

    def say_choices(self):
        """The report caption: what is chosen now, in words."""
        chosen = ["{} {}".format(name, _choice(box)) for name, box in self.said_choices() if _choice(box)]
        self.report_caption.setText(" · ".join(chosen))
        self.report_caption.setVisible(bool(chosen))


#: Sample-to-sample change of scalp EEG, µV (robust spread of the first difference): a few µV,
#: a few hundred with mains hum in an unfiltered recording. Far outside, the units are likely wrong.
PLAUSIBLE_UV = (0.05, 2000.0)


def unit_advice(data):
    """What to do when a recording's amplitudes do not look like µV, else "".

    Measured on the change from one sample to the next, so an electrode's DC offset or slow drift,
    tens of thousands of µV in a raw OpenBCI file, does not look like amplified data.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        return ""
    step = np.diff(data[:, :min(data.shape[1], 60000)], axis=1)
    centre = np.nanmedian(step, axis=1, keepdims=True)
    channel = 1.4826 * np.nanmedian(np.abs(step - centre), axis=1)
    channel = channel[np.isfinite(channel) & (channel > 0)]          # flat or empty channels say nothing
    if not channel.size:
        return ""
    spread = float(np.median(channel))
    if spread < PLAUSIBLE_UV[0]:
        guess = "V" if spread < 5e-5 else "mV"
        return ("The channels vary by only {:.2g} µV, where EEG varies by tens of µV: the file's numbers are "
                "probably in {}. Add Units / gain after the Recording and set 'The numbers are in' to {}."
                .format(spread, guess, guess))
    if spread > PLAUSIBLE_UV[1]:
        return ("The channels vary by {:.3g} µV, far more than EEG: the signal was probably amplified before it "
                "was recorded, or the numbers are ADC counts. Add Units / gain after the Recording and enter the "
                "amplifier's gain (or the µV per count as Other factor). Unfiltered drift can look like this too; "
                "a Filter shows which.".format(spread))
    return ""


def _marked_note(spans):
    """A line on the spans the file marks as bad, which leave the trials over them out; "" when none."""
    if not spans:
        return ""
    labels = list(dict.fromkeys(label for _, _, label in spans))
    return "\nMarked bad in the file: {} span{} ({}), {:.1f} s; trials over them are left out".format(
        len(spans), "" if len(spans) == 1 else "s", ", ".join(labels[:4]) + (", …" if len(labels) > 4 else ""),
        sum(end - start for start, end, _ in spans))


#: Seconds a recording's preview can show at once.
RAW_SPANS = (2.0, 5.0, 10.0, 30.0)
STACKED = "All (stacked)"


class RawPreview(_Preview):
    """A stretch of the recording: one channel on a µV axis, or several stacked; where and how long chosen."""

    def __init__(self, recording, large=False):
        super().__init__(recording, large)
        motion = getattr(recording, "motion", None) is not None and recording.motion_names
        self.add_note(recording.describe() + "\nEvents: " +
                      (recording.events.describe() if recording.events is not None else "none") +
                      ("\nUnits: " + recording.unit_note if getattr(recording, "unit_note", "") else "") +
                      ("\nMotion: {} ({}), kept apart from the EEG".format(
                          ", ".join(recording.motion_names), ", ".join(dict.fromkeys(recording.motion_units)))
                       if motion else "") + _marked_note(getattr(recording, "bad_spans", [])))
        reading = getattr(recording, "reading", [])
        if reading:
            # what the program did on its own while reading: in view, and in the report's Data handling
            self.add_note("How it was read:\n" + "\n".join("· " + line for line in reading))
        if recording.events is None or not len(recording.events):
            # said here, where it begins, not only when Epochs stops on it
            self.add_note("No events in this file. Trials can still be cut: in Epochs, 'Without markers: every' "
                          "(its Options) for trials that repeat, or Trials from: manual for a table. A MATLAB "
                          "file may hold its events in variables: 'How this file is read…' in this block.",
                          quiet=False).setStyleSheet(warning())
        names = list(recording.channel_names)
        self.channel = self.channel_box([STACKED] + names, self.draw)
        if not any(n in names for n in FIRST_CHANNELS) and names:
            self.channel.setCurrentIndex(1)          # one channel first: its amplitude can be read
        self.channel.setToolTip("One channel on a µV axis, or the first {} stacked".format(
            RAW_CHANNELS_LARGE if large else RAW_CHANNELS))
        row = self.column.itemAt(self.column.count() - 1).layout()
        # a few lengths to pick, or any typed (7 for one Cho2017 trial) and Enter
        self.span = NoScrollComboBox()
        self.span.setEditable(True)
        self.span.addItems(["{:g} s".format(s) for s in RAW_SPANS])
        self.span.setCurrentIndex(RAW_SPANS.index(RAW_SECONDS))
        self.span.setToolTip("Seconds shown at once: pick one, or type any length and press Enter")
        # picked from the list, or set by the program: drawn at once; typed: when Enter is pressed or it is left
        self.span.activated.connect(lambda _i: self._typed_span())
        self.span.currentTextChanged.connect(lambda _t: self.draw() if not self.span.lineEdit().hasFocus() else None)
        self.span.lineEdit().editingFinished.connect(self._typed_span)
        self.start = NoScrollSpinBox()
        self.start.setDecimals(1)
        self.start.setSuffix(" s")
        self.start.setRange(0.0, max(0.0, recording.duration - 0.1))
        self.start.setKeyboardTracking(False)
        self.start.valueChanged.connect(lambda _v: self.draw())
        labels = [QtWidgets.QLabel("Show"), QtWidgets.QLabel("from")]
        play = QtWidgets.QToolButton()
        play.setText("Play…")
        play.setToolTip("Play the recording like a video: its signal, the spectrum of each moment and the "
                        "spectrogram of the whole recording, in a window of its own")
        play.clicked.connect(self.play)
        for widget in (labels[0], self.span, labels[1], self.start, play):
            row.addWidget(widget)
        self.controls += labels + [self.span, self.start, play]
        self.choosers += [("Show", self.span), ("from", self.start)]
        self.plot = small_plot(260, large)
        self.plot.setLabel("bottom", "Time (s)")
        self.column.addWidget(self.plot)
        self.caption = self.add_note("")
        advice = unit_advice(recording.data)
        if advice:
            self.add_note(advice, quiet=False)
        self.draw()

    def play(self):
        """The recording in a player window: signal, spectrum now, spectrogram of all of it."""
        from .player import PlayerWindow
        window = PlayerWindow(self.value, self.value.path.name if getattr(self.value, "path", None) else "Recording")
        window.show()
        self._player = window
        return window

    def seconds(self):
        """The length typed or picked, within 0.1 s and the whole recording; what cannot be read: the default."""
        try:
            seconds = float(self.span.currentText().replace(",", ".").replace("s", "").strip())
        except ValueError:
            return RAW_SECONDS
        return min(max(seconds, 0.1), max(self.value.duration, 0.1)) if np.isfinite(seconds) else RAW_SECONDS

    def _typed_span(self):
        """The typed length written as the others are ("7 s"), then drawn."""
        text = "{:g} s".format(self.seconds())
        self.span.blockSignals(True)
        self.span.setCurrentText(text)
        self.span.blockSignals(False)
        self.draw()

    def draw(self):
        recording, plot = self.value, self.plot
        plot.clear()
        seconds = self.seconds()
        a = min(int(self.start.value() * recording.fs), max(0, recording.n_samples - 1))
        b = min(recording.n_samples, a + int(seconds * recording.fs))
        t = np.arange(a, b) / recording.fs
        left = plot.getAxis("left")
        if self.channel.currentText() == STACKED:
            rows = list(range(min(RAW_CHANNELS_LARGE if self.large else RAW_CHANNELS, recording.n_channels)))
            block = recording.data[rows, a:b]
            spread = (np.nanpercentile(np.abs(block - np.nanmean(block, axis=1, keepdims=True)), 99)
                      if block.size else 1.0)
            spacing = 2.5 * (spread if np.isfinite(spread) and spread > 0 else 1.0)
            ticks = []
            for k, index in enumerate(rows):
                plot.plot(t, block[k] - np.nanmean(block[k]) - k * spacing, pen=channel_pen(index, width=1),
                          connect="finite")
                ticks.append((-k * spacing, recording.channel_names[index]))
            left.setTicks([ticks])
            plot.setLabel("left", "")
            self.caption.setText("{} of {} channels, each around its mean, {:.0f} µV apart.".format(
                len(rows), recording.n_channels, spacing))
        else:
            index = recording.channel_names.index(self.channel.currentText())
            x = recording.data[index, a:b]
            centre = np.nanmean(x) if np.isfinite(x).any() else 0.0
            plot.plot(t, x - centre, pen=channel_pen(index, width=1), connect="finite")
            left.setTicks(None)
            plot.setLabel("left", "µV (around its mean)")
            finite = x[np.isfinite(x)]
            self.caption.setText("{}: mean {:.4g} µV taken off; {:.4g} µV peak to peak, {:.3g} µV RMS here.".format(
                recording.channel_names[index], centre, np.ptp(finite) if finite.size else 0.0,
                np.std(finite) if finite.size else 0.0))
        if recording.events is not None and b > a:
            for onset, label in zip(recording.events.onsets, recording.events.labels):
                if t[0] <= onset <= t[-1]:
                    plot.addItem(pg.InfiniteLine(onset, pen=pg.mkPen(MARKER_COLOUR, width=1),
                                                 label=label, labelOpts={"position": 0.95, "color": MARKER_COLOUR}))
        plot.enableAutoRange()


#: Trials drawn one above the other at a time: beside the settings, in a larger window.
TRIAL_PAGE = 10
TRIAL_PAGE_LARGE = 25
MEANS, EACH, IMAGE = "Condition means", "Each trial", "All trials (image)"
#: Trials per page offered; any other number can be typed.
TRIAL_PAGES = (10, 25, 50)
#: Trial numbers written beside the image, at most.
IMAGE_TICKS = 20


def draw_trials(plot, epochs, channel, rows, kept=None, toggled=None):
    """Trials `rows` (indices) of `epochs` one above the other, the first on top, on one µV scale.

    `kept`: ids drawn in colour, the others faint and dashed. `toggled(id)`: called
    when a trial is clicked. Returns the spacing between trials, µV.
    """
    plot.clear()
    channels = channel if isinstance(channel, (list, tuple)) else [channel]
    # several channels: their signal averaged, trial by trial (a region of interest)
    block = (epochs.data[rows][:, channels].mean(axis=1) if len(rows) else np.zeros((0, epochs.data.shape[-1])))
    # each trial around its own mean, as the recording's channels are: an offset (unfiltered drift, a file's
    # DC level) is thousands of µV and would set the spacing, leaving every trial a flat line
    if block.size:
        with np.errstate(invalid="ignore"):
            block = block - np.nanmean(block, axis=1, keepdims=True)
    spread = np.nanpercentile(np.abs(block), 99) if block.size else 1.0
    spacing = 2.2 * (spread if np.isfinite(spread) and spread > 0 else 1.0)
    ticks = []
    for k, row in enumerate(rows):
        trial = epochs.ids[row]
        inside = kept is None or trial in kept
        pen = (channel_pen(k, width=1) if inside else
               pg.mkPen(color("dim"), width=1, style=QtCore.Qt.PenStyle.DashLine))
        curve = plot.plot(epochs.times, block[k] - k * spacing, pen=pen, connect="finite")
        if toggled is not None:
            curve.setCurveClickable(True, width=10)
            curve.sigClicked.connect(lambda _c, _e=None, t=trial: toggled(t))
        ticks.append((-k * spacing, "{}{}".format(trial, "" if inside else " ✕")))
    plot.getAxis("left").setTicks([ticks])
    plot.setLabel("left", "Trial")
    plot.enableAutoRange(axis="x")
    # half a gap above the first and below the last, so every trial's number has room on the axis
    plot.setYRange(-(max(len(rows), 1) - 0.5) * spacing, 0.5 * spacing, padding=0)
    return spacing


class EpochsPreview(_Preview):
    """The trials: their condition means, a page of them one above the other, or all of them as an image."""

    def __init__(self, epochs, large=False, kept=None, toggled=None):
        super().__init__(epochs, large)
        self.kept, self.toggled = kept, toggled
        self.picking = toggled is not None
        self.add_note(epochs.describe())
        if epochs.rejected and not self.picking:
            reasons = {}
            for _, _, reason in epochs.rejected:
                reasons[reason] = reasons.get(reason, 0) + 1
            self.add_note("Left out: " + "; ".join("{} ×{}".format(r, n) for r, n in reasons.items()), quiet=False)
        self.channel = self.channel_box(epochs.names, self.draw, several=True)
        self.grid = None
        row = QtWidgets.QHBoxLayout()
        self.column.addLayout(row)
        self.show_box = NoScrollComboBox()
        self.show_box.addItems([MEANS, EACH, IMAGE])
        self.show_box.setItemData(2, "Every trial as one row, its µV in colour: a hundred trials at a glance, an odd "
                                     "one stands out as a bright stripe", QtCore.Qt.ItemDataRole.ToolTipRole)
        self.show_box.setCurrentText(EACH if self.picking else MEANS)
        self.show_box.currentIndexChanged.connect(lambda _i: self.draw())
        n = len(epochs.ids)
        self.page = min(TRIAL_PAGE_LARGE if large else TRIAL_PAGE, max(n, 1))
        # how many trials a page holds: a few to pick, or any typed and Enter
        self.page_box = NoScrollComboBox()
        self.page_box.setEditable(True)
        self.page_box.addItems([str(p) for p in TRIAL_PAGES])
        self.page_box.setCurrentText(str(self.page))
        self.page_box.setToolTip("Trials on one page: pick, or type any number and press Enter")
        self.page_box.lineEdit().setValidator(QtGui.QIntValidator(1, 10000, self.page_box))
        self.page_box.activated.connect(lambda _i: self._page_changed())
        self.page_box.currentTextChanged.connect(
            lambda _t: self._page_changed() if not self.page_box.lineEdit().hasFocus() else None)
        self.page_box.lineEdit().editingFinished.connect(self._page_changed)
        self.start = NoScrollSpinBox()
        self.start.setDecimals(0)
        self.start.setPrefix("from ")
        self.start.setRange(1, max(1, n))
        self.start.setSingleStep(self.page)
        self.start.setKeyboardTracking(False)
        self.start.valueChanged.connect(lambda _v: self.draw())
        self.back, self.forward = QtWidgets.QToolButton(), QtWidgets.QToolButton()
        for button, shape, tip, step in ((self.back, "chevron-left", "The page before", -1),
                                         (self.forward, "chevron-right", "The next page", 1)):
            put(button, shape, "", size=16)
            button.setAccessibleName(tip)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, s=step: self.turn(s))
        labels = [QtWidgets.QLabel("Show"), QtWidgets.QLabel("per page")]
        self.paging = [labels[1], self.page_box, self.back, self.start, self.forward]
        for widget in (labels[0], self.show_box, self.back, self.start, self.forward, self.page_box, labels[1]):
            row.addWidget(widget)
        row.addStretch(1)
        self.controls += [labels[0], self.show_box] + self.paging
        self.choosers += [("Show", self.show_box), ("Per page", self.page_box), ("Trials", self.start)]
        self.plot = small_plot(300 if self.picking else 220, large)
        self.plot.setLabel("bottom", "Time from event (s)")
        self.column.addWidget(self.plot)
        # the image and its means are made the first time they are shown: two more plots cost a selection time
        self.image_plot = self.mean_plot = None
        self._image_at = self.column.count()
        self.caption = self.add_note("")
        self.draw()

    def _image_plots(self):
        """The image of every trial, and under it the condition means on the same time axis."""
        large = self.large
        self.image_plot = small_plot(320 if self.picking else 260, large)
        self.image_plot.setLabel("left", "Trial")
        self.image = pg.ImageItem()
        self.image_plot.addItem(self.image)
        self.bar = pg.ColorBarItem(colorMap=pg.colormap.get("CET-D1"), values=(-1, 1), interactive=False, width=14)
        self.bar.setImageItem(self.image, insert_in=self.image_plot.getPlotItem())
        self.bar.axis.setLabel("µV, around each trial's mean")
        self.image_plot.scene().sigMouseClicked.connect(self._image_clicked)
        self.mean_plot = small_plot(130, False)
        self.mean_plot.setLabel("left", "Mean (µV)")
        self.mean_plot.setLabel("bottom", "Time from event (s)")
        self.mean_plot.setXLink(self.image_plot)
        self.mean_legend = legend(self.mean_plot)
        self.column.insertWidget(self._image_at, self.image_plot)
        self.column.insertWidget(self._image_at + 1, self.mean_plot)

    def page_size(self):
        """Trials per page as typed or picked, at least 1; what cannot be read: as before."""
        try:
            return max(1, int(self.page_box.currentText().strip()))
        except ValueError:
            return self.page

    def _page_changed(self):
        self.page = self.page_size()
        self.page_box.blockSignals(True)
        self.page_box.setCurrentText(str(self.page))
        self.page_box.blockSignals(False)
        self.start.setSingleStep(self.page)
        self.draw()

    def turn(self, step):
        """To the page before (-1) or after (1)."""
        n = len(self.value.ids)
        first = int(self.start.value()) + step * self.page
        self.start.setValue(min(max(first, 1), n))

    def mode(self):
        return self.show_box.currentText()

    def draw(self):
        what, chosen = self.channels_chosen()
        mode, n = self.mode(), len(self.value.ids)
        # every channel side by side: for the means; trials one above the other are of one channel or a mean
        c = chosen if what == "mean" else chosen[:1] if what == "all" and mode != MEANS else chosen
        if self.grid is not None:
            self.grid.setVisible(mode == MEANS and what == "all")
        for widget in self.paging:
            widget.setVisible(mode == EACH and (n > self.page or widget in self.paging[:2]))
        self.plot.setVisible(mode != IMAGE)
        if mode == IMAGE and self.image_plot is None:
            self._image_plots()
        if self.image_plot is not None:
            self.image_plot.setVisible(mode == IMAGE)
            self.mean_plot.setVisible(mode == IMAGE)
        if mode == EACH:
            first = int(self.start.value()) - 1
            rows = list(range(first, min(n, first + self.page)))
            self.back.setEnabled(first > 0)
            self.forward.setEnabled(first + self.page < n)
            spacing = draw_trials(self.plot, self.value, c, rows, self.kept, self.toggled)
            self.caption.setText(self._each_caption(rows, spacing))
            return
        if mode == IMAGE:
            self._draw_image(c)
            self.caption.setText(self._image_caption())
            return
        if what == "all":
            if self.grid is None:
                self.grid = self._add_grid()
            self.plot.hide()
            self.grid.show()
            shown = head_order(self.value.names, chosen)[:GRID_MOST]
            for plot, channel in zip(self.grid.fill([self.value.names[k] for k in shown], "Time from event (s)"),
                                     shown):
                self._draw_means(plot, [channel], named=False)
            self.grid.items[0].setLabel("left", "Mean (µV)")
            self.caption.setText("Each channel's condition means, one µV scale")
            return
        self.plot.show()
        self.plot.clear()
        self.plot.getAxis("left").setTicks(None)
        self.plot.setLabel("left", "Mean (µV)")
        if self.plot.getPlotItem().legend is None:
            legend(self.plot)
        self._draw_means(self.plot, c)
        # the trials' page fixed the µV range; the means have their own, at once and as they change
        self.plot.enableAutoRange()
        self.plot.autoRange()
        self.caption.setText("Mean of {}".format(", ".join(self.value.names[k] for k in c)) if what == "mean" else "")

    def _draw_means(self, plot, c, named=True):
        labels = np.asarray(self.value.labels)
        channels = c if isinstance(c, (list, tuple)) else [c]
        for k, condition in enumerate(dict.fromkeys(self.value.labels)):
            trials = self.value.data[labels == condition][:, channels].mean(axis=1)
            plot.plot(self.value.times, trials.mean(axis=0), pen=pg.mkPen(CHANNEL_COLORS[k], width=1),
                      name="{} (n = {})".format(condition, len(trials)) if named else None)

    def _draw_image(self, c):
        epochs, n = self.value, len(self.value.ids)
        channels = c if isinstance(c, (list, tuple)) else [c]
        block = np.asarray(epochs.data[:, channels].mean(axis=1), dtype=float)
        with np.errstate(invalid="ignore"):
            block = block - np.nanmean(block, axis=1, keepdims=True)
        finite = np.abs(block[np.isfinite(block)])
        self.bound = float(np.percentile(finite, 99)) if finite.size else 1.0
        self.bound = self.bound if self.bound > 0 else 1.0
        # rows top to bottom, trial k at y = -k; ImageItem's second axis runs upwards
        self.image.setImage(np.nan_to_num(block[::-1].T), levels=(-self.bound, self.bound))
        self.bar.setLevels((-self.bound, self.bound))
        times = epochs.times
        dt = 1.0 / epochs.fs
        self.image.setRect(QtCore.QRectF(times[0] - dt / 2, -(n - 0.5), dt * times.size, n))
        plot = self.image_plot
        for item in [i for i in plot.getPlotItem().items if i is not self.image]:
            plot.removeItem(item)
        every = max(1, int(np.ceil(n / IMAGE_TICKS)))
        ticks = []
        for k, trial in enumerate(epochs.ids):
            inside = self.kept is None or trial in self.kept
            if not inside:
                plot.addItem(pg.InfiniteLine(-k, angle=0, pen=pg.mkPen(color("dim"), width=2,
                                                                         style=QtCore.Qt.PenStyle.DashLine)))
            if k % every == 0 or not inside:
                ticks.append((-k, "{}{}".format(trial, "" if inside else " ✕")))
        plot.getAxis("left").setTicks([ticks])
        plot.setXRange(times[0] - dt / 2, times[-1] + dt / 2, padding=0)
        plot.setYRange(-(n - 0.5), 0.5, padding=0)
        self.mean_plot.clear()
        self.mean_legend.clear()
        self._draw_means(self.mean_plot, c)
        self.mean_plot.enableAutoRange(axis="y")
        self.mean_plot.autoRange()
        self.mean_plot.setXRange(times[0] - dt / 2, times[-1] + dt / 2, padding=0)

    def _image_clicked(self, event):
        """A click on a trial's row of the image, when picking: that trial kept or left out."""
        if self.toggled is None or self.mode() != IMAGE:
            return
        view = self.image_plot.getPlotItem().getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        k = int(round(-view.mapSceneToView(event.scenePos()).y()))
        if 0 <= k < len(self.value.ids):
            self.toggled(self.value.ids[k])

    def said_choices(self):
        # the trial page only when trials are drawn, and nothing about them when the means are
        mode = self.mode()
        if mode == EACH:
            return self.choosers
        return self.choosers[:2] if mode == IMAGE else self.choosers[:1]

    def _image_caption(self):
        ids = self.value.ids
        text = ("All {} trials, one row each, first on top; colour: µV around each trial's mean, ±{:.3g} µV at the "
                "ends of the scale. Under it, the condition means.".format(len(ids), self.bound))
        if self.picking:
            from ...pipeline.blocks import trial_text
            kept = [i for i in ids if self.kept is None or i in self.kept]
            text += " Click a row to keep the trial or leave it out. Kept {} of {}: {}.".format(
                len(kept), len(ids), ", ".join(trial_text(kept)) or "none")
        return text

    def _each_caption(self, rows, spacing):
        ids = self.value.ids
        shown = "Trials {}–{} of {}".format(ids[rows[0]], ids[rows[-1]], len(ids)) if rows else "No trials"
        text = "{}, {:.0f} µV apart.".format(shown, spacing)
        if self.picking:
            from ...pipeline.blocks import trial_text
            kept = [i for i in ids if self.kept is None or i in self.kept]
            text += " Click a trial to keep it or leave it out. Kept {} of {}: {}.".format(
                len(kept), len(ids), ", ".join(trial_text(kept)) or "none")
        return text

    def set_kept(self, kept):
        self.kept = kept
        self.draw()


#: How a spectrum's values are drawn: (label, log axis, in decibels).
SCALES = (("Log", True, False), ("Linear", False, False), ("dB", False, True))
#: A density's scale that divides each frequency's power by its neighbours'.
SNR = "SNR"
#: The two averages of a condition (the Analysis page's), and what a spectrum preview shows.
MEAN_TRIALS, MEAN_SIGNAL = "Mean of trials", "Mean signal"
SPECTRUM, CONTRAST = "Spectrum", "Contrast (d)"
#: The frequencies a spectrum is drawn over: from 1 Hz to 70 Hz (or the Nyquist frequency), so mains at 50 or
#: 60 Hz stands clear of the edge (at 60 the 60 Hz peak sat on the plot's border; tests/test_plot_labels.py)
SPECTRUM_HZ = (1.0, 70.0)


class SpectrumPreview(_Preview):
    #: the scale chosen last, kept for the next spectrum shown this session
    scale = "Log"

    def __init__(self, spectrum, large=False):
        super().__init__(spectrum, large)
        self.add_note(spectrum.describe())
        self.channel = self.channel_box(spectrum.names, self.draw, several=True)
        self.scale_box = NoScrollComboBox()
        self.scale_box.addItems([s[0] for s in SCALES])
        if spectrum.is_power:
            # power at each frequency over its neighbours 0.75-2.5 Hz away (core.spectral.neighbour_snr)
            self.scale_box.addItem(SNR)
            self.scale_box.setItemData(self.scale_box.count() - 1, "Power over its neighbours 0.75–2.5 Hz away "
                                       "(1 = no peak), the usual steady-state (SSVEP) measure",
                                       QtCore.Qt.ItemDataRole.ToolTipRole)
        self.scale_box.setCurrentText(SpectrumPreview.scale)
        self.scale_box.setToolTip("Log: a wide range of power at once. Linear: the size of a peak as it is. "
                                  "dB: 10·log10, as amplifiers and many papers give it")
        self.scale_box.currentTextChanged.connect(self._rescale)
        label = QtWidgets.QLabel("Scale")
        row = self.column.itemAt(self.column.count() - 1).layout()      # the channel row
        row.addWidget(label)
        row.addWidget(self.scale_box)
        self.controls += [label, self.scale_box]
        self.choosers.append(("Scale", self.scale_box))
        row = QtWidgets.QHBoxLayout()
        # the Analysis page's two averages: the trials' spectra averaged, or the spectrum of the mean waveform
        self.average = self._chooser(row, "Average", (MEAN_TRIALS, MEAN_SIGNAL) if spectrum.signal_means else
                                     (MEAN_TRIALS,),
                                     "Mean of trials: each trial's spectrum, then averaged. Mean signal: the "
                                     "trials averaged first, then one spectrum (what is not phase-locked cancels)")
        conditions = list(dict.fromkeys(spectrum.labels))
        contrast = spectrum.is_power and len(conditions) > 1
        self.show_box = self._chooser(row, "Show", (SPECTRUM, CONTRAST) if contrast else (SPECTRUM,),
                                      "Contrast: Cohen's d of log power per frequency between two conditions, over "
                                      "their trials (core.spectral.condition_contrast)")
        self.a_box = self._chooser(row, "A", conditions)
        self.b_box = self._chooser(row, "B", conditions)
        if len(conditions) > 1:
            self.b_box.setCurrentIndex(1)
        row.addStretch(1)
        self.column.addLayout(row)
        self.plot = small_plot(200, large)
        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.legend = legend(self.plot)
        self.column.addWidget(self.plot)
        self.grid = None
        self.caption = self.add_note("")
        self.draw()

    def _chooser(self, row, name, items, tip=""):
        """A labelled combo in `row`, kept among the choosers; hidden while it offers a single item."""
        box = NoScrollComboBox()
        box.addItems(list(items))
        box.setToolTip(tip)
        box.currentIndexChanged.connect(lambda _i: self.draw())
        label = QtWidgets.QLabel(name)
        row.addWidget(label)
        row.addWidget(box)
        self.controls += [label, box]
        self.choosers.append((name, box))
        box.label = label
        return box

    @staticmethod
    def density_unit():
        """How a density is drawn (Settings → Graphs): power density, amplitude density or amplitude."""
        from ..preferences import psd_unit
        return psd_unit()

    def _rescale(self, text):
        SpectrumPreview.scale = text
        self.draw()
        self.plot.enableAutoRange()

    def said_choices(self):
        """Only what shapes the drawing: A and B only for a contrast, the average only when there is a choice."""
        contrast = self.show_box.currentText() == CONTRAST
        return [(name, box) for name, box in self.choosers
                if (name not in ("A", "B") or contrast) and (name != "Average" or box.count() > 1)
                and (name != "Show" or box.count() > 1)]

    def draw(self):
        if not hasattr(self, "caption"):
            return                                    # still being built
        contrast = self.show_box.currentText() == CONTRAST
        for box, shown in ((self.average, self.average.count() > 1 and not contrast),
                           (self.show_box, self.show_box.count() > 1), (self.a_box, contrast), (self.b_box, contrast)):
            box.setVisible(shown)
            box.label.setVisible(shown)
        self.scale_box.setEnabled(not contrast)
        if contrast:
            return self._draw_contrast()
        snr = self.scale_box.currentText() == SNR
        name, log, decibels = next((s for s in SCALES if s[0] == self.scale_box.currentText()),
                                   ("SNR", False, False) if snr else SCALES[0])
        from ...core.spectral import PSD_TITLES, PSD_UNITS
        unit = PSD_UNITS[self.density_unit()] if self.value.is_power else self.value.unit
        title = PSD_TITLES[self.density_unit()] if self.value.is_power else self.value.title
        what, channels = self.channels_chosen()
        axis = ("SNR (× neighbours)" if snr else
                "{} ({})".format(title, "dB re 1 " + unit if decibels else unit))
        if what == "all":
            if self.grid is None:
                self.grid = self._add_grid()
            self.plot.hide()
            self.grid.show()
            shown = head_order(self.value.names, channels)[:GRID_MOST]
            for plot, c in zip(self.grid.fill([self.value.names[c] for c in shown], "Frequency (Hz)"), shown):
                plot.setLogMode(x=False, y=log)
                self._curves(plot, [c], decibels)
            self.grid.items[0].setLabel("left", axis)
            self.caption.setText("Every channel, the same scale; {}".format(
                "the first {} of {}".format(GRID_MOST, len(channels)) if len(channels) > GRID_MOST else
                "{} channels".format(len(channels))))
            return
        if self.grid is not None:
            self.grid.hide()
        self.plot.show()
        self.plot.clear()
        self.plot.setLogMode(x=False, y=log)
        self.plot.setLabel("left", axis)
        self._curves(self.plot, channels, decibels, named=True)
        signal = self.average.currentText() == MEAN_SIGNAL
        parts = ["spectrum of each condition's mean waveform" if signal else "each condition's trials' spectra "
                 "averaged"]
        if what == "mean":
            parts.append("then the mean over {} channels, each measured on its own: {}".format(
                len(channels), ", ".join(self.value.names[c] for c in channels)))
        text = "; ".join(parts)
        self.caption.setText(text[0].upper() + text[1:] + ".")       # channel names keep their case

    def _curves(self, plot, channels, decibels, named=False):
        keep = (self.value.freqs >= SPECTRUM_HZ[0]) & (self.value.freqs <= SPECTRUM_HZ[1])
        labels = np.asarray(self.value.labels)
        signal = self.average.currentText() == MEAN_SIGNAL and self.value.signal_means
        snr = self.scale_box.currentText() == SNR
        for k, condition in enumerate(dict.fromkeys(self.value.labels)):
            if signal:
                # the spectrum of the condition's mean waveform, channel by channel, then the channels averaged
                mean = self.value.signal_means[condition][channels].mean(axis=0)
            else:
                # the mean over this condition's trials and the channels chosen: spectra averaged, as a region's are
                mean = self.value.power[labels == condition][:, channels].mean(axis=(0, 1))
            unit = self.density_unit()
            if snr:
                from ...core.spectral import neighbour_snr
                mean = neighbour_snr(self.value.freqs, mean)[keep]
            elif self.value.is_power:
                # the density drawn in the unit of Settings → Graphs (display only), as every PSD graph is
                from ...core.spectral import psd_in_unit
                mean = np.maximum(psd_in_unit(self.value.freqs, mean, unit)[keep], 1e-12)
            else:
                mean = np.maximum(mean[keep], 1e-12)
            if decibels:
                # power is 10·log10; an amplitude 20·log10 (10·log10 of its square), so all read the same in dB
                power_like = self.value.is_power and unit == POWER_DENSITY
                mean = (10 if power_like else 20) * np.log10(mean)
            n = int(np.sum(labels == condition))
            plot.plot(self.value.freqs[keep], mean, pen=pg.mkPen(CHANNEL_COLORS[k], width=1),
                      name="{} (n={})".format(condition, n) if named else None)

    def contrast(self):
        """(freqs, Cohen's d of log power A − B per frequency, n A, n B) for the channels chosen, or a reason.

        Each trial's spectrum is first averaged over the channels chosen, as the Analysis page averages a
        region; every trial that reached this block counts."""
        from ...core.spectral import condition_contrast
        a, b = self.a_box.currentText(), self.b_box.currentText()
        if a == b:
            return "Choose two different conditions (A, B)"
        _, channels = self.channels_chosen()
        labels = np.asarray(self.value.labels)
        per_trial = self.value.power[:, channels].mean(axis=1)
        try:
            d = condition_contrast(per_trial[labels == a], per_trial[labels == b], self.value.freqs)
        except ValueError as exc:
            return str(exc)
        return self.value.freqs, d, int(np.sum(labels == a)), int(np.sum(labels == b))

    def _draw_contrast(self):
        from ...core.spectral import strongest_frequencies
        if self.grid is not None:
            self.grid.hide()
        self.plot.show()
        self.plot.clear()
        self.plot.setLogMode(x=False, y=False)
        self.plot.setLabel("left", "Cohen's d, log power")
        found = self.contrast()
        if isinstance(found, str):
            self.caption.setText(found)
            return
        freqs, d, na, nb = found
        top = min(45.0, float(freqs[-1]))
        keep = freqs <= top
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color("dim"), width=1)))
        self.plot.plot(freqs[keep], d[keep], pen=pg.mkPen(CHANNEL_COLORS[0], width=2))
        a, b = self.a_box.currentText(), self.b_box.currentText()

        def listed(picks):
            return ", ".join("{:g} Hz (d {:.1f})".format(f, v) for f, v in picks if v > 0) or "none"

        what, channels = self.channels_chosen()
        self.caption.setText("{a} > {b}: {up} · {b} > {a}: {down} (trials {na} / {nb}; {where}). Positive: more power "
                             "in {a}.".format(a=a, b=b, up=listed(strongest_frequencies(freqs, d, 1, top)),
                                              down=listed(strongest_frequencies(freqs, -d, 1, top)), na=na, nb=nb,
                                              where=", ".join(self.value.names[c] for c in channels)))


def band_layout(names):
    """(channels, bands) when every feature is "<channel> <band>" for each channel and band (Band power's
    columns, band after band), else None."""
    pairs = [name.rsplit(" ", 1) for name in names]
    if not names or any(len(p) != 2 for p in pairs):
        return None
    channels = list(dict.fromkeys(p[0] for p in pairs))
    bands = list(dict.fromkeys(p[1] for p in pairs))
    if len(bands) < 2 or len(channels) * len(bands) != len(names):
        return None
    if [" ".join(p) for p in pairs] != ["{} {}".format(c, b) for b in bands for c in channels]:
        return None
    return channels, bands


class FeaturesPreview(_Preview):
    """A block's features. Band power is drawn first - each band's power per condition, bars with the SD
    over trials, for one channel or the mean of all - and its numbers follow in the table; other features
    are the table alone."""

    #: Band power's chooser: every channel's mean, then the channels one by one.
    ALL_MEAN = "Mean of all channels"
    #: The share of the plot's height kept free at the top for the legend's row.
    LEGEND_ROOM = 0.2

    def __init__(self, features, large=False):
        super().__init__(features, large)
        self.layout_ = band_layout(features.names)
        if self.layout_ is not None:
            channels, bands = self.layout_
            unit = features.units[0] if features.units else ""
            self.add_note("{}: {} per band and condition, mean ± SD over trials{}.".format(
                features.describe(), "power", " ({})".format(unit) if unit else ""))
            self.band_channel = short_list(NoScrollComboBox())
            self.band_channel.addItems([self.ALL_MEAN] + channels)
            label = QtWidgets.QLabel("Channel")
            row = QtWidgets.QHBoxLayout()
            row.addWidget(label)
            row.addWidget(self.band_channel, 1)
            self.column.addLayout(row)
            self.controls += [label, self.band_channel]
            self.choosers.append(("Channel", self.band_channel))
            self.plot = small_plot(240, large)
            self.plot.setLabel("left", "Power", units=unit or None)
            self.plot.getAxis("bottom").setTicks([list(enumerate(bands))])
            self.legend = legend(self.plot)
            self.column.addWidget(self.plot)
            self.band_channel.currentIndexChanged.connect(lambda _i: self.draw_bands())
            self.draw_bands()
            self.fill_height()
        self.add_note(("Every channel and band, " if self.layout_ is not None else features.describe() + ". ")
                      + "Mean ± SD per condition:")
        conditions = list(dict.fromkeys(features.labels))
        labels = np.asarray(features.labels)
        table = QtWidgets.QTableWidget(len(features.names), len(conditions))
        table.setHorizontalHeaderLabels(conditions)
        table.setVerticalHeaderLabels(features.names)
        for j, condition in enumerate(conditions):
            rows = features.values[labels == condition]
            for i in range(len(features.names)):
                column = rows[:, i]
                text = "{:.4g} ± {:.2g}".format(np.nanmean(column), np.nanstd(column, ddof=1) if len(column) > 1 else 0)
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                table.setItem(i, j, item)
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(min(700 if large else (200 if self.layout_ is not None else 320),
                                   40 + 26 * len(features.names)))
        self.column.addWidget(table)
        save = QtWidgets.QPushButton("Save every trial as CSV…")
        save.clicked.connect(self.save)
        self.column.addWidget(save, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.controls.append(save)

    def band_values(self):
        """{condition: (n_trials, n_bands)} of the chosen channel's band power, or every channel's mean."""
        channels, bands = self.layout_
        values = self.value.values.reshape(len(self.value.labels), len(bands), len(channels))
        chosen = self.band_channel.currentText()
        per_trial = (np.nanmean(values, axis=2) if chosen == self.ALL_MEAN
                     else values[:, :, channels.index(chosen)])
        labels = np.asarray(self.value.labels)
        return {c: per_trial[labels == c] for c in dict.fromkeys(self.value.labels)}

    def draw_bands(self):
        from ..theme import CONDITIONS
        from ..plots import CHANNEL_COLORS
        self.plot.clear()
        if self.legend is not None:
            self.legend.clear()
        groups = self.band_values()
        if self.legend is not None:
            self.legend.setColumnCount(max(len(groups), 1))   # one row across the top, over the headroom below
        colours = list(CONDITIONS) + [c for c in CHANNEL_COLORS if c not in CONDITIONS]
        width = 0.8 / max(len(groups), 1)
        # log power is negative or near zero: there the bars stand on the lowest value shown, not on 0, so
        # differences show; power itself (never below 0) stands on 0, so a bar's height is the power. The top
        # LEGEND_ROOM of the plot is left empty for the legend, so it never covers a bar or its SD.
        # (Before, the bars were drawn from 0 even for log power: off the top of the plot, cut, under the legend.)
        lows = [np.nanmin(np.nanmean(r, axis=0) - np.nanstd(r, axis=0)) for r in groups.values() if r.size]
        highs = [np.nanmax(np.nanmean(r, axis=0) + np.nanstd(r, axis=0)) for r in groups.values() if r.size]
        bottom = top = None
        if lows and highs and np.isfinite(lows + highs).all():
            span = max(highs) - min(lows) or 1.0
            bottom = 0.0 if min(lows) >= 0 else min(lows) - 0.1 * span
            top = max(highs) + 0.05 * span
        for k, (condition, rows) in enumerate(groups.items()):
            x = np.arange(rows.shape[1]) - 0.4 + width * (k + 0.5)
            mean = np.nanmean(rows, axis=0)
            sd = np.nanstd(rows, axis=0, ddof=1) if rows.shape[0] > 1 else np.zeros_like(mean)
            colour = colours[k % len(colours)]
            base = 0.0 if bottom is None else bottom
            bars = pg.BarGraphItem(x=x, y0=base, height=mean - base, width=width * 0.9, brush=colour,
                                   pen=pg.mkPen(None), name="{} (n = {})".format(condition, rows.shape[0]))
            self.plot.addItem(bars)                  # named: the legend takes it by itself
            self.plot.addItem(pg.ErrorBarItem(x=x, y=mean, height=2 * sd, beam=width * 0.3,
                                              pen=pg.mkPen(color("subtle"), width=1.2)))
        if bottom is not None:
            self.plot.setYRange(bottom, top + (top - bottom) * self.LEGEND_ROOM / (1 - self.LEGEND_ROOM), padding=0)

    def save(self, _checked=False, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save features", "features.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(self.value.table())
        return path


def roc_points(positive, scores):
    """(false-positive rates, true-positive rates) of the ROC curve, from (0, 0) to (1, 1): every score taken as
    the threshold in turn, tied scores together."""
    positive = np.asarray(positive, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    scores, positive = scores[order], positive[order]
    # one point where the score changes: tied trials cross the threshold together
    last = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]
    tp = np.cumsum(positive)[last]
    fp = np.cumsum(~positive)[last]
    n_pos, n_neg = max(int(positive.sum()), 1), max(int((~positive).sum()), 1)
    return np.r_[0.0, fp / n_neg], np.r_[0.0, tp / n_pos]


class ScoresPreview(_Preview):
    def __init__(self, scores, large=False):
        super().__init__(scores, large)
        above = scores.better_than_random
        spread = "" if len(scores.fold_accuracy) < 2 else " ± {:.1%}".format(scores.accuracy_sd)
        headline = QtWidgets.QLabel("<span style='font-size:20pt; font-weight:600'>{:.1%}</span>"
                                    "&nbsp;{} accuracy".format(scores.accuracy, spread))
        self.column.addWidget(headline)
        # said, not judged: three steps in neutral colours, no tick or cross, no red (it is not an error)
        badge = QtWidgets.QLabel(scores.chance_level[0].upper() + scores.chance_level[1:])
        badge.setStyleSheet("font-weight: 600; color: {};".format(color("strong") if above else color("subtle")))
        self.column.addWidget(badge)
        limit = self.add_note("With {} tested trials a random classifier reaches up to {:.1%} (alpha 0.05) or {:.1%} "
                              "(alpha 0.01).".format(scores.tested, scores.random_limit, scores.random_limit_01))
        limit.setToolTip("The binomial limit of a random classifier for this many trials: Müller-Putz et al. "
                         "(2008); Combrisson & Jerbi (2015)")
        if scores.unequal:
            name, share = scores.largest
            self.add_note("The conditions have unequal trials: always guessing {} scores {:.1%}. Balanced accuracy "
                          "{:.1%} must also beat the {:.1%} a random classifier reaches.".format(
                              name, share, scores.balanced_accuracy, scores.balanced_limit),
                          quiet=scores.balanced_accuracy > scores.balanced_limit)
        if scores.permutations:
            self.add_note("Permutation test: p = {:.3g} ({} label shuffles); the verdict needs p ≤ 0.05.".format(
                scores.permutation_p, scores.permutations), quiet=scores.permutation_p <= 0.05)
        self.add_note("{} · {}, seed {}".format(scores.model, scores.split_text(), scores.seed))
        # the other measures on one small line, each the mean of the folds' own as accuracy is
        measures = ["balanced accuracy {:.1%}".format(scores.balanced_accuracy)]
        if np.isfinite(scores.kappa):
            measures.append("Cohen's κ {:.2f}".format(scores.kappa))
        if np.isfinite(scores.f1):
            measures.append("F1 {:.2f}".format(scores.f1))
        if scores.fold_auc:
            measures.append("ROC AUC {:.3f}".format(scores.auc))
        line = self.add_note(" · ".join(measures))
        line.setToolTip("Each the mean of the folds' own values. Balanced accuracy: the mean hit rate of the "
                        "conditions. Cohen's κ: agreement beyond chance, 0 chance and 1 every trial right "
                        "(Cohen, 1960; Schlögl et al., 2007). F1: macro, each condition's F1 averaged unweighted. "
                        "ROC AUC: two conditions, how well the scores rank the trials.")
        if scores.tuning_text():
            self.add_note("Tuning chose " + scores.tuning_text() + ".")
        plot = small_plot(150, large)
        plot.setLabel("bottom", "Fold")
        plot.setLabel("left", "Accuracy")
        plot.addItem(pg.BarGraphItem(x=np.arange(1, len(scores.fold_accuracy) + 1), height=scores.fold_accuracy,
                                     width=0.6, brush=color("ink")))
        plot.setYRange(0, 1)
        plot.getAxis("bottom").setTicks([[(k, str(k)) for k in range(1, len(scores.fold_accuracy) + 1)]])
        self.column.addWidget(plot)
        table = QtWidgets.QTableWidget(len(scores.classes), len(scores.classes))
        table.setHorizontalHeaderLabels(["→ " + c for c in scores.classes])
        table.setVerticalHeaderLabels(scores.classes)
        for i in range(len(scores.classes)):
            for j in range(len(scores.classes)):
                item = QtWidgets.QTableWidgetItem(str(int(scores.confusion[i, j])))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, j, item)
        table.setFixedHeight(30 + 28 * len(scores.classes))
        note = self.add_note("Rows: true condition; columns: predicted, over all folds.")
        self.column.addWidget(table)
        self._add_roc(scores)
        save = QtWidgets.QPushButton("Save predictions as CSV…")
        save.clicked.connect(self.save)
        self.column.addWidget(save, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        # a report states these in its own tables
        self.controls += [note, table, save]

    def _add_roc(self, scores):
        """The ROC curve of every tested trial, for two conditions: how well the scores rank one above the other."""
        rows = [r for r in scores.predictions if np.isfinite(r[3])]
        if len(scores.classes) != 2 or not rows:
            return
        positive = [str(r[1]) == str(scores.classes[1]) for r in rows]
        if all(positive) or not any(positive):
            return
        fpr, tpr = roc_points(positive, [r[3] for r in rows])
        plot = small_plot(220, large=self.large)
        plot.setLabel("bottom", "False positive rate ({} called {})".format(scores.classes[0], scores.classes[1]))
        plot.setLabel("left", "True positive rate")
        plot.setAspectLocked(True)
        plot.plot([0, 1], [0, 1], pen=pg.mkPen(color("dim"), width=1, style=QtCore.Qt.PenStyle.DashLine))
        plot.plot(fpr, tpr, pen=pg.mkPen(color("ink"), width=1), stepMode=None)
        plot.setXRange(0, 1, padding=0.02)
        plot.setYRange(0, 1, padding=0.02)
        area = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        self.column.addWidget(plot)
        self.add_note("ROC: every tested trial, scored by the model of the fold that tested it ({} counted "
                      "positive). Area under this curve {:.3f}; the ROC AUC above is the mean of each fold's own, "
                      "which can differ a little as each fold's model scores on its own scale. The dashed line is "
                      "a random classifier.".format(scores.classes[1], area))

    def save(self, _checked=False, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save predictions", "predictions.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["trial", "condition", "predicted", "score"])
                writer.writerows(self.value.predictions)
        return path


#: How long each head map over time is, s: those to pick, and the first shown; maps at most, and to a row.
MAP_STEPS, MAP_STEP, MAPS_MOST, MAP_COLUMNS = (0.1, 0.25, 0.5, 1.0), 0.5, 40, 8


#: A band typed into the head maps' Band box: "9-12", "9–12 Hz", "alpha 9-12" (a comma for the decimal point too).
TYPED_BAND = re.compile(r"^\s*(?:([^\d\s][^\s]*)\s+)?(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*(?:hz)?\s*$",
                        re.IGNORECASE)


def read_band(text):
    """(name, low, high) of a typed band, name "" when none was typed; ValueError saying how to write it."""
    found = TYPED_BAND.match(str(text))
    if not found:
        raise ValueError("Write the band as low-high in Hz, e.g. 6-20 or alpha 9-12")
    name, low, high = found.group(1), float(found.group(2).replace(",", ".")), float(found.group(3).replace(",", "."))
    if not low < high:
        raise ValueError("The band's low edge must be below its high edge")
    return name or "", low, high


def read_seconds(text):
    """Seconds typed as "0.5", "0,5" or "0.5 s"; ValueError otherwise."""
    found = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*s?\s*$", str(text), re.IGNORECASE)
    if not found or float(found.group(1).replace(",", ".")) <= 0:
        raise ValueError("Write each map's length in seconds, e.g. 0.5")
    return float(found.group(1).replace(",", "."))


def _safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text)) or "condition"


#: What a time-frequency preview can show: (label, core.epochs map kind).
TF_KINDS = (("ERD / ERS %", ep.ERD_MAP), ("Power (dB)", ep.POWER_MAP), ("Amplitude (µV)", ep.AMPLITUDE_MAP))
#: What a time-frequency result is shown as, besides its map.
MAP_VIEW, COURSE_VIEW, HEAD_VIEW, STATS_VIEW = "Map", "Band course", "Head map", "Statistics"
TF_VIEWS = (MAP_VIEW, COURSE_VIEW, HEAD_VIEW, STATS_VIEW)
NO_COMPARISON = "none"
#: The statistics table's columns: (heading, key of core.epochs.statistics' rows).
STAT_COLUMNS = (("Channel", "channel"), ("Band", "band"), ("Hz", "band_hz"), ("A mean %", "mean_a"),
                ("A SD", "sd_a"), ("A n", "n_a"), ("A pooled %", "pooled_a"), ("B mean %", "mean_b"),
                ("B SD", "sd_b"), ("B n", "n_b"), ("B pooled %", "pooled_b"), ("B − A (pp)", "difference"),
                ("Cliff's δ", "cliffs_delta"), ("p (Welch)", "p_welch"))
#: Where the active window starts and ends (the cue and the end of imagery, in a cued design).
#: Ochre, not red: in GIOVA red only means an error.
WINDOW_PEN = dict(color="#C49A5A", width=1.3, style=QtCore.Qt.PenStyle.DashLine)


class TFPreview(_Preview):
    """A time-frequency map: ERD, power or amplitude, averaged over a condition's trials or of one trial."""

    #: a band typed for the head maps, to be added to the block's Bands: [name, low, high]
    band_kept = QtCore.Signal(list)

    #: the map chosen last, kept for the next result shown this session
    kind = ep.ERD_MAP

    def __init__(self, result, large=False):
        super().__init__(result, large)
        self.channel = self.channel_box([tf.name for tf in result.tf], self.draw, several=True, head=True)
        row = self.column.itemAt(self.column.count() - 1).layout()
        self.condition = self._chooser(row, "Condition", result.conditions)
        self.condition.currentIndexChanged.connect(lambda _i: self._fill_trials())
        second = QtWidgets.QHBoxLayout()
        self.column.addLayout(second)
        self.kind_box = self._chooser(second, "Show", [k[0] for k in TF_KINDS])
        self.kind_box.setCurrentIndex(next(i for i, k in enumerate(TF_KINDS) if k[1] == TFPreview.kind))
        self.kind_box.setItemData(2, "Each trial's amplitude, averaged over trials, linear from zero: a sine of "
                                     "amplitude A reads A (scipy.signal.stft gives A/2)",
                                  QtCore.Qt.ItemDataRole.ToolTipRole)
        self.kind_box.currentIndexChanged.connect(self._kind_changed)
        self.trial = self._chooser(second, "Trial", [])
        self.trial.currentIndexChanged.connect(lambda _i: self.draw())
        self.plot = small_plot(240, large)
        self.plot.setLabel("bottom", "Time from event (s)")
        self.plot.setLabel("left", "Frequency (Hz)")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.bar = pg.ColorBarItem(colorMap=erd_colormap(), values=(-1, 1), interactive=False, width=14)
        self.bar.setImageItem(self.image, insert_in=self.plot.getPlotItem())
        settings = result.settings
        for edge in settings.active:
            self.plot.addItem(pg.InfiniteLine(edge, angle=90, pen=pg.mkPen(**WINDOW_PEN)))
        for edge in settings.bands[0][1:] if settings.bands else ():
            self.plot.addItem(pg.InfiniteLine(edge, angle=0, pen=pg.mkPen(
                color=(255, 255, 255, 200), width=1, style=QtCore.Qt.PenStyle.DotLine)))
        self.grid = self.series = self.moment = None
        # head maps over time: which band, how long each map, and the moment shown large; above the maps
        self.time_row = QtWidgets.QHBoxLayout()
        self.column.addLayout(self.time_row)
        self.column.addWidget(self.plot)
        self.bands = [(name, low, high) for name, low, high in settings.bands]
        # the block's bands to pick, or any typed (6-20) and Enter: a look at the same maps, nothing computed again
        self.band = self._chooser(self.time_row, "Band",
                                  ["{} {:g}–{:g} Hz".format(name, low, high) for name, low, high in self.bands])
        self.band.setToolTip("One of the block's bands, or type any (6-20, alpha 9-12) and press Enter: the maps "
                             "are drawn at once from what the block computed")
        self.keep_band = QtWidgets.QToolButton()
        self.keep_band.setText("Add to block")
        self.keep_band.setToolTip("Add this band to the block's Bands: kept in the pipeline file, the report and the "
                                  "Python export (the block computes again at the next Run)")
        self.keep_band.clicked.connect(self._keep_band)
        self.time_row.insertWidget(self.time_row.indexOf(self.band) + 1, self.keep_band)
        self.controls.append(self.keep_band)
        self.step = self._chooser(self.time_row, "Each map", ["{:g} s".format(s) for s in MAP_STEPS])
        self.step.setCurrentIndex(MAP_STEPS.index(MAP_STEP))
        self.step.setToolTip("How long each head map is: pick, or type any length in seconds and press Enter")
        for box in (self.band, self.step):
            box.setEditable(True)
            box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            # picked from the list or set by the program: drawn at once; typed: when Enter is pressed or it is left
            box.currentTextChanged.connect(lambda _t, b=box: self.draw() if not b.lineEdit().hasFocus() else None)
            box.lineEdit().editingFinished.connect(self.draw)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(result.tf[0].times.size - 1, 0))
        self.slider.setToolTip("The moment shown in the large map")
        self.slider.valueChanged.connect(lambda _v: self._draw_moment())
        self.time_row.addWidget(self.slider, 2)
        self.controls.append(self.slider)
        self.time_widgets = [self.time_row.itemAt(k).widget() for k in range(self.time_row.count())
                             if self.time_row.itemAt(k).widget() is not self.keep_band]
        # what else the result shows, as the Analysis page did: a band over time with its SEM, the head map of the
        # active window, and the statistics of two conditions; "Compare with" adds a second condition and B − A
        third = QtWidgets.QHBoxLayout()            # a row of its own: the side panel is narrow
        self.column.insertLayout(self.column.indexOf(self.plot) - 1 if self.column.indexOf(self.plot) > 0 else
                                 self.column.count(), third)
        self.view_box = self._chooser(third, "View", TF_VIEWS)
        self.view_box.setToolTip("Map: time × frequency. Band course: the band's ERD over time, mean ± SEM of the "
                                 "trials. Head map: the band in the active window on every electrode computed. "
                                 "Statistics: per channel, the trials' active-window ERD of two conditions")
        self.compare = self._chooser(third, "Compare with", [NO_COMPARISON] + list(result.conditions))
        self.compare.setToolTip("A second condition, B: drawn beside the condition chosen (A), with B − A")
        for box in (self.view_box, self.compare):
            box.currentIndexChanged.connect(lambda _i: self.draw())
        self.course = self.head = self.table = None
        self.caption = self.add_note("")
        self._fill_trials()

    def with_tools(self):
        """The usual tools, and Save every trial…: one picture per trial, as it is drawn now."""
        super().with_tools()
        row = self.column.itemAt(self.column.count() - 1).layout()
        every = QtWidgets.QToolButton()
        every.setText("Save every trial…")
        every.setToolTip("One PNG per trial of every condition, and each condition's average, into a folder: "
                         "drawn as here (channel, view, map)")
        every.clicked.connect(lambda: self.save_every_trial())
        row.insertWidget(row.count() - 1, every)
        self.controls.append(every)
        return self

    def save_every_trial(self, folder=None):
        """Every trial's picture, and each condition's average, into `folder`: the files written."""
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the trial pictures")
            if not folder:
                return []
        from .report import prepare
        folder = Path(folder)
        total = sum(len(ids) + 1 for ids in self.value.epoch_ids.values())
        progress = QtWidgets.QProgressDialog("Saving trial pictures…", "Stop", 0, total, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(400)
        written = []
        with paper():
            # one copy drawn once on white; only the condition and the trial change between pictures
            copy = type(self)(self.value, self.large).choose(self.choices())
            prepare(copy, width=max(self.width(), 700))
            try:
                for c, condition in enumerate(self.value.conditions):
                    copy.condition.setCurrentIndex(c)
                    ids = self.value.epoch_ids[condition]
                    for k in range(len(ids) + 1):
                        if progress.wasCanceled():
                            return written
                        copy.trial.setCurrentIndex(k)
                        copy.say_choices()
                        QtWidgets.QApplication.processEvents()
                        # ids are numbers or text (from Select trials): padded when numbers, so files sort in order
                        trial_id = str(ids[k - 1]) if k else ""
                        name = "average" if k == 0 else "trial" + (trial_id.zfill(3) if trial_id.isdigit()
                                                                    else _safe_name(trial_id))
                        path = folder / "{}_{}.png".format(_safe_name(condition), name)
                        picture(copy).save(str(path), "PNG")
                        written.append(path)
                        progress.setValue(len(written))
            finally:
                progress.close()
                copy.close()
                copy.deleteLater()
        return written

    def said_choices(self):
        over_time = self.channel.currentText() == OVER_TIME
        view = self.view_box.currentText() if hasattr(self, "view_box") else MAP_VIEW
        banded = over_time or view != MAP_VIEW
        return [(name, box) for name, box in self.choosers
                if (banded or name != "Band") and (over_time or name != "Each map")
                and (name != "Trial" or view == MAP_VIEW) and (name != "Compare with" or view != MAP_VIEW)]

    def _show(self, view):
        """Only `view` among the plot, the grid, the maps over time, the band course, the head map and the table."""
        over_time = view == "over time"
        banded = view in ("over time", "course", "head", "stats")
        for widget in self.time_widgets:
            widget.setVisible(over_time or (banded and widget in (self.band, self.band.label)))
        for name, widget in (("one", self.plot), ("all", self.grid), ("course", self.course), ("head", self.head),
                             ("stats", self.table)):
            if widget is not None:
                widget.setVisible(view == name)
        for widget in (self.series, self.moment):
            if widget is not None:
                widget.setVisible(over_time)
        if hasattr(self, "view_box"):
            mapped = self.view_box.currentText() == MAP_VIEW
            for box in (self.trial, self.kind_box):
                box.setEnabled(mapped or box is self.kind_box and view == "head")
            self.compare.setEnabled(not mapped)
        if not over_time:
            self.keep_band.hide()

    def _chooser(self, row, name, items):
        box = short_list(NoScrollComboBox())
        box.addItems(list(items))
        label = QtWidgets.QLabel(name)
        row.addWidget(label)
        row.addWidget(box, 1)
        self.controls += [label, box]
        self.choosers.append((name, box))
        box.label = label
        return box

    def _pair(self):
        """[A] or [A, B]: the condition chosen and the one compared with it, when another is chosen."""
        a = self._condition()
        b = self.compare.currentText()
        return [a, b] if b in self.value.conditions and b != a else [a]

    def _block_band(self):
        """The block's own band chosen (name, low, high): what the head maps and statistics were computed for."""
        name, low, high = self.chosen_band()
        found = next((b for b in self.value.settings.bands if abs(b[1] - low) < 1e-9 and abs(b[2] - high) < 1e-9),
                     None)
        if found is None:
            raise ValueError("The head map and the statistics are computed for the block's bands ({}): add "
                             "{:g}–{:g} Hz to the block and Run".format(
                                 ", ".join(b[0] for b in self.value.settings.bands), low, high))
        return tuple(found)

    def _average_said(self):
        return ("pooled: the trials' mean power against their mean baseline"
                if self.value.settings.erd_mode == ep.POOLED else "the mean of each trial's % change")

    def _draw_course(self, channels):
        """The band's ERD over time, mean ± SEM of the trials (core.epochs.band_course), A and B on one plot."""
        result = self.value
        if self.course is None:
            self.course = self._add_view(small_plot(240, self.large), 260)
            self.course.setLabel("bottom", "Time from event (s)")
            self.course.setLabel("left", "ERD / ERS (%)")
            self.course_legend = legend(self.course)
        self._show("course")
        plot = self.course
        plot.clear()
        try:
            name, low, high = self.chosen_band()
        except ValueError as exc:
            self.caption.setText(str(exc))
            return
        if len(channels) != 1:
            self.caption.setText("The band course is drawn for one channel: choose it in Channel.")
            return
        channel = result.tf[channels[0]]
        for edge in result.settings.active:
            plot.addItem(pg.InfiniteLine(edge, angle=90, pen=pg.mkPen(**WINDOW_PEN)))
        plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(color("dim"))))
        said = []
        for k, cond in enumerate(self._pair()):
            mean, sem, n = ep.band_course(channel, cond, (low, high), result.settings.erd_mode)
            tint = pg.mkColor(CHANNEL_COLORS[k])
            finite = np.isfinite(mean) & np.isfinite(sem)
            if finite.any() and np.any(sem[finite] > 0):
                shade = pg.mkColor(tint)
                shade.setAlpha(50)
                fill = pg.FillBetweenItem(pg.PlotCurveItem(channel.times[finite], (mean + sem)[finite]),
                                          pg.PlotCurveItem(channel.times[finite], (mean - sem)[finite]), brush=shade)
                fill.setZValue(-5)
                plot.addItem(fill)
            plot.plot(channel.times, mean, pen=pg.mkPen(tint, width=2), name="{} n={}".format(cond, n))
            said.append("{} n={}".format(cond, n))
        self.caption.setText("{} {}{:g}–{:g} Hz over time: {}; shaded ± SEM across trials. Average: {}. Dashed: the "
                             "active window.".format(channel.name, name + " " if name else "", low, high,
                                                     ", ".join(said), self._average_said()))

    def head_panels(self, kind):
        """[(title, value per electrode name)] of the block's band in the active window: A, B and B − A.

        ERD is the pooled ERD of every electrode (core.epochs: one STFT per electrode, window means); power its
        mean in dB. B − A is in percentage points for ERD, in dB for power."""
        result = self.value
        name, _, _ = self._block_band()
        panels = []
        for cond in self._pair():
            values = np.full(len(result.names), np.nan)
            source = (result.topo_pooled[name][cond] if kind == ep.ERD_MAP else
                      10 * np.log10(np.maximum(result.topo_power[name][cond], 1e-12)))
            values[result.topo_channels] = source
            # an electrode not measured during the trials (switched off or flat) gets no colour
            values[[k for k, n in enumerate(result.names) if n in getattr(result, "unmeasured", ())]] = np.nan
            panels.append(("{} n={}".format(cond, len(result.epoch_ids[cond])), values))
        if len(panels) == 2:
            pair = self._pair()
            panels.append(("{} − {} ({})".format(pair[1], pair[0], "percentage points" if kind == ep.ERD_MAP
                                                  else "dB"), panels[1][1] - panels[0][1]))
        return panels

    def _draw_head(self):
        from ..topomap import TopoGrid
        result = self.value
        if self.head is None:
            self.head = self._add_view(TopoGrid(), 300)
        self._show("head")
        kind = ep.POWER_MAP if TF_KINDS[max(0, self.kind_box.currentIndex())][1] == ep.POWER_MAP else ep.ERD_MAP
        try:
            name, low, high = self._block_band()
            panels = self.head_panels(kind)
        except ValueError as exc:
            self.head.clear_maps()
            self.caption.setText(str(exc))
            return
        label = ("{} ERD / ERS (%)".format(name) if kind == ep.ERD_MAP else
                 "{} power (dB re 1 {})".format(name, result.tf[0].units))
        self.head.draw(result.names, panels, "{}, {:g}–{:g} s".format(label, *result.settings.active),
                       diverging=kind == ep.ERD_MAP or len(panels) == 3,
                       highlight={result.names[c] for c in result.channels})
        self.caption.setText("{}{} {:g}–{:g} Hz in the active window, {:g}–{:g} s, on every electrode computed "
                             "({}); {}. ERD here is pooled over trials (window means).".format(
                                 self._few_channels(), name, low, high, *result.settings.active,
                                 len(result.topo_channels),
                                 "B − A: the second condition minus the first" if len(panels) == 3 else
                                 "choose a condition in Compare with to add B and B − A")
                             + (" Not measured (switched off or flat), left out: {}.".format(
                                 ", ".join(result.unmeasured)) if getattr(result, "unmeasured", None) else ""))

    def statistics(self):
        """core.epochs.statistics of A and B (B = A when nothing is compared) for the block's band chosen."""
        name = self._block_band()[0]
        pair = self._pair()
        return [row for row in ep.statistics(self.value, pair[0], pair[-1]) if row["band"] == name]

    def _draw_stats(self):
        if self.table is None:
            self.table = self._add_view(QtWidgets.QTableWidget(), 220)
            self.table.setColumnCount(len(STAT_COLUMNS))
            self.table.setHorizontalHeaderLabels([c[0] for c in STAT_COLUMNS])
            self.table.horizontalHeader().setStretchLastSection(True)
        self._show("stats")
        try:
            rows = self.statistics()
        except ValueError as exc:
            self.table.setRowCount(0)
            self.caption.setText(str(exc))
            return
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (_, key) in enumerate(STAT_COLUMNS):
                value = row[key]
                text = value if isinstance(value, str) else str(value) if isinstance(value, int) else \
                    "" if not np.isfinite(value) else "{:.3g}".format(value)
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setFlags(cell.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, cell)
        pair = self._pair()
        self.caption.setText(
            "Per channel, each trial's ERD in the active window ({:g}–{:g} s) of {} (A){}. Difference: B − A in "
            "percentage points; Cliff's delta: B over A; Welch's t-test p, not corrected for the {} tests made: a "
            "screen, not a finding.".format(*self.value.settings.active, pair[0],
                                            " and {} (B)".format(pair[1]) if len(pair) == 2 else
                                            "; choose a condition in Compare with for B", len(rows)))

    def _condition(self):
        return self.value.conditions[max(0, self.condition.currentIndex())]

    def _fill_trials(self):
        ids = self.value.epoch_ids[self._condition()]
        self.trial.blockSignals(True)
        self.trial.clear()
        self.trial.addItems(["Average of {}".format(len(ids))] + ["Trial {}".format(i) for i in ids])
        self.trial.blockSignals(False)
        self.draw()

    def _kind_changed(self, index):
        TFPreview.kind = TF_KINDS[max(0, index)][1]
        self.draw()

    def draw(self):
        result, condition = self.value, self._condition()
        kind = TF_KINDS[max(0, self.kind_box.currentIndex())][1]
        trial = self.trial.currentIndex() - 1
        what, channels = self.channels_chosen()
        view = self.view_box.currentText() if hasattr(self, "view_box") else MAP_VIEW
        if view == COURSE_VIEW:
            return self._draw_course(channels if what in ("one", "mean") else [])
        if view == HEAD_VIEW:
            return self._draw_head()
        if view == STATS_VIEW:
            return self._draw_stats()
        if what == "all":
            self._draw_grid(channels, condition, trial, kind)
            return
        if what == "over time":
            self._draw_over_time(condition, trial, kind)
            return
        self._show("one")
        channel = result.tf[channels[0]]
        values = tf_map(result, [result.tf[c] for c in channels], condition, trial, kind)
        levels, colormap, label = tf_scale(kind, values, channel.units)
        self.image.setImage(np.nan_to_num(np.asarray(values, dtype=float).T), levels=levels)
        self.bar.setColorMap(colormap)
        self.bar.setLevels(levels)
        self.bar.axis.setLabel(label)
        self.image.setRect(_tf_rect(channel))
        text = self._caption(kind, trial, condition)
        if what == "mean":
            text = "Mean of {}: their power averaged, then {}. ".format(
                ", ".join(result.tf[c].name for c in channels),
                "ERD against their averaged baseline" if kind == ep.ERD_MAP else "drawn") + text
        self.caption.setText(text)

    def _draw_grid(self, channels, condition, trial, kind):
        """Every channel's map on one colour scale, a colour bar beside the last."""
        result = self.value
        if self.grid is None:
            self.grid = self._add_grid()
        self._show("all")
        shown = head_order([tf.name for tf in result.tf], channels)[:GRID_MOST]
        maps = [tf_map(result, [result.tf[c]], condition, trial, kind) for c in shown]
        levels, colormap, label = tf_scale(kind, np.concatenate([np.ravel(m) for m in maps]), result.tf[0].units)
        images = []
        plots = self.grid.fill([result.tf[c].name for c in shown], "Time from event (s)")
        for plot, c, values in zip(plots, shown, maps):
            image = pg.ImageItem(np.nan_to_num(np.asarray(values, dtype=float).T), levels=levels)
            image.setColorMap(colormap)
            rect = _tf_rect(result.tf[c])
            image.setRect(rect)
            plot.addItem(image)
            plot.setXRange(rect.left(), rect.right(), padding=0)
            plot.setYRange(rect.top(), rect.bottom(), padding=0)
            for edge in result.settings.active:
                plot.addItem(pg.InfiniteLine(edge, angle=90, pen=pg.mkPen(**WINDOW_PEN)))
            images.append(image)
        plots[0].setLabel("left", "Hz")
        if self.grid.bar is None:
            self.grid.bar = pg.ColorBarItem(colorMap=colormap, values=levels, interactive=False, width=12)
            self.grid.bar.setImageItem(images, insert_in=plots[-1])
        else:                                   # the plots were kept: so is their colour bar
            self.grid.bar.setColorMap(colormap)
            self.grid.bar.setLevels(levels)
            self.grid.bar.setImageItem(images)
        self.grid.bar.axis.setLabel(label)
        self.caption.setText("Every channel on one colour scale ({}). {}".format(
            "the first {} of {}".format(GRID_MOST, len(channels)) if len(channels) > GRID_MOST else
            "{} channels".format(len(channels)), self._caption(kind, trial, condition)))

    def _add_view(self, widget, height, after=None):
        """A view under the one-channel plot (or under `after`), made the first time it is asked for."""
        if self.large:
            widget.setMinimumHeight(int(height * 1.6))
        else:
            widget.setFixedHeight(height)
        self.column.insertWidget(self.column.indexOf(after or self.plot) + 1, widget)
        self.fill_height()
        return widget

    def _few_channels(self):
        n = len(self.value.tf)
        return ("Only {} channel{} computed: leave the block's Channels empty for every electrode. "
                .format(n, "" if n == 1 else "s") if n < 8 else "")

    def chosen_band(self):
        """(name, low, high) of the band picked or typed; ValueError when it does not read or lies outside what
        the block computed."""
        name, low, high = read_band(self.band.currentText())
        freqs = self.value.tf[0].freqs
        if low < freqs[0] - 1e-9 or high > freqs[-1] + 1e-9:
            raise ValueError("{:g}-{:g} Hz is outside what the block computed, {:g}-{:g} Hz: set its Lowest / "
                             "Highest and Run".format(low, high, freqs[0], freqs[-1]))
        return name, low, high

    def chosen_step(self):
        """Each map's length, s; ValueError when it does not read, is shorter than a frame or longer than a trial."""
        step = read_seconds(self.step.currentText())
        times = self.value.tf[0].times
        frame = float(np.median(np.diff(times))) if times.size > 1 else step
        if step < frame - 1e-9 or step > times[-1] - times[0] + frame:
            raise ValueError("Each map can last from {:.2f} s (one frame) to {:.2f} s (the whole trial)".format(
                frame, times[-1] - times[0] + frame))
        return step

    def _block_has(self, band):
        return any(abs(low - band[1]) < 1e-9 and abs(high - band[2]) < 1e-9 for _name, low, high in self.bands)

    def _keep_band(self):
        """The band typed goes into the block's Bands (the Inspector does it)."""
        try:
            name, low, high = self.chosen_band()
        except ValueError:
            return
        self.band_kept.emit([name, low, high])

    def _band_traces(self, condition, trial, kind):
        """name -> the chosen band over time, for every channel."""
        band = self.chosen_band()
        return {tf.name: ep.band_trace(self.value, tf, condition, band[1:], trial, kind) for tf in self.value.tf}

    def _draw_over_time(self, condition, trial, kind):
        """One band's head map for each step of time, on one scale; the slider's moment large above them."""
        from ..topomap import TopoGrid
        result = self.value
        if self.series is None:
            self.moment = self._add_view(TopoGrid(), 300)
            self.series = self._add_view(TopoGrid(), 360, after=self.moment)
        self._show("over time")
        names = [tf.name for tf in result.tf]
        times = result.tf[0].times
        try:
            band = self.chosen_band()
            step = self.chosen_step()
            traces = self._band_traces(condition, trial, kind)
        except ValueError as exc:                # typed so it does not read, or no frequency bin in the band
            self.series.clear_maps()
            self.moment.clear_maps()
            self.traces = None
            self.keep_band.hide()
            self.caption.setText(str(exc))
            return
        # typed and not among the block's bands: it can be kept there
        self.keep_band.setVisible(not self._block_has(band))
        # windows on whole multiples of the step (0–0.5, 0.5–1 s…), whatever the first frame's centre
        first = np.floor(times[0] / step + 1e-9) * step
        every_start = np.round(np.arange(first, times[-1] + 1e-9, step), 6)
        panels = []
        for start in every_start[:MAPS_MOST]:
            frames = (times >= start) & (times < start + step)
            if frames.any():
                with np.errstate(all="ignore"):
                    panels.append(("{:g}–{:g} s".format(round(start, 3), round(start + step, 3)),
                                   np.array([np.nanmean(traces[n][frames]) for n in names])))
        every = np.concatenate([np.ravel(t) for t in traces.values()])
        self.levels, self.colormap, self.series_label = tf_scale(kind, every, result.tf[0].units)
        rows = (len(panels) - 1) // MAP_COLUMNS + 1
        if self.large:
            self.series.setMinimumHeight(int(rows * 210 + 20))
        else:
            self.series.setFixedHeight(int(rows * 150 + 20))
        self.series.draw(names, panels, self.series_label, diverging=kind == ep.ERD_MAP, columns=MAP_COLUMNS,
                         levels=self.levels, colormap=self.colormap)
        self.traces, self.names = traces, names
        self._draw_moment()
        self.caption.setText("{}{}{:g}–{:g} Hz, {:g} s per map{}; above, the moment the slider is on. {}".format(
            self._few_channels(), band[0] + " " if band[0] else "", band[1], band[2], step,
            " (the first {})".format(MAPS_MOST) if every_start.size > MAPS_MOST else "",
            self._caption(kind, trial, condition)))

    def _draw_moment(self):
        if self.moment is None or self.moment.isHidden() or getattr(self, "traces", None) is None:
            return
        times = self.value.tf[0].times
        k = min(self.slider.value(), times.size - 1)
        values = np.array([self.traces[n][k] for n in self.names])
        self.moment.draw(self.names, [("{:.2f} s".format(times[k]), values)], self.series_label,
                         diverging=self.levels[0] < 0, levels=self.levels, colormap=self.colormap)

    def _caption(self, kind, trial, condition):
        settings = self.value.settings
        ids = self.value.epoch_ids[condition]
        what = ("Trial {} of {}".format(ids[trial], condition) if trial >= 0 else
                "Each of {} {} trials transformed, then averaged".format(len(ids), condition))
        method = {ep.STFT: "STFT, {:g} s window ({:g} Hz apart)".format(settings.window, 1 / settings.window),
                  ep.MORLET: "complex Morlet wavelets, {:g} cycles per Hz".format(settings.n_cycles),
                  ep.CWT: "complex Morlet wavelet {}".format(settings.wavelet)}[settings.method]
        text = "{}; {}. Dashed: the active window, {:g}–{:g} s.".format(what, method, *settings.active)
        if kind == ep.ERD_MAP:
            text += " Red is a decrease from the baseline (ERD), blue an increase (ERS)."
        elif kind == ep.AMPLITUDE_MAP and trial >= 0 and settings.smooth:
            text += (" From power smoothed over {:g} s: set the block's Smoothing to 0 for this trial's raw {}."
                     .format(settings.smooth, "|STFT|" if settings.method == ep.STFT else "wavelet amplitude"))
        return text


class ContrastPreview(_Preview):
    """Two windows compared: how much each band's power differs, channel by channel, and how sure that is."""

    def __init__(self, contrast, large=False):
        super().__init__(contrast, large)
        a, b = contrast.names
        self.add_note("{}. {} {:g}–{:g} s against {} {:g}–{:g} s; change: mean power of {} over that of {}, "
                      "in %. {} test of the log power across trials; p adjusted by Holm over all {} channel-band "
                      "pairs.".format(contrast.describe(), a, *contrast.windows[0], b, *contrast.windows[1], a, b,
                                      "Wilcoxon signed-rank" if contrast.test == "wilcoxon" else "Paired t",
                                      len(contrast.rows)))
        self.plot = small_plot(220, large)
        self.plot.setLabel("left", "Change of {} from {} (%)".format(a, b))
        self.column.addWidget(self.plot)
        self.legend = legend(self.plot)
        channels = list(dict.fromkeys(r["channel"] for r in contrast.rows))
        bands = [band[0] for band in contrast.bands]
        width = 0.8 / max(len(bands), 1)
        for k, band in enumerate(bands):
            rows = [r for r in contrast.rows if r["band"] == band]
            x = np.arange(len(rows)) + (k - (len(bands) - 1) / 2) * width
            heights = [r["change"] for r in rows]
            brushes = [pg.mkBrush(CHANNEL_COLORS[k % len(CHANNEL_COLORS)]) for _ in rows]
            self.plot.addItem(pg.BarGraphItem(x=x, height=heights, width=width * 0.9, brushes=brushes))
            self.legend.addItem(pg.BarGraphItem(x=[0], height=[0], width=0, brush=CHANNEL_COLORS[k % len(CHANNEL_COLORS)]),
                                band)
            for xi, row in zip(x, rows):
                if row["p_holm"] <= 0.05:
                    mark = pg.TextItem("*", color=color("text"), anchor=(0.5, 1.0 if row["change"] >= 0 else 0.0))
                    mark.setPos(xi, row["change"])
                    self.plot.addItem(mark)
        self.plot.getAxis("bottom").setTicks([list(enumerate(channels))])
        self.plot.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(color("dim"), width=1)))
        table = QtWidgets.QTableWidget(len(contrast.rows), 7)
        table.setHorizontalHeaderLabels(["Channel", "Band", "{} µV²".format(a), "{} µV²".format(b), "Change %",
                                         "p", "p (Holm)"])
        for i, row in enumerate(contrast.rows):
            cells = [row["channel"], row["band"], "{:.3g}".format(row["mean_a"]), "{:.3g}".format(row["mean_b"]),
                     "{:+.1f}".format(row["change"]), "{:.3g}".format(row["p"]), "{:.3g}".format(row["p_holm"])]
            for j, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if row["p_holm"] <= 0.05:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(i, j, item)
        table.horizontalHeader().setStretchLastSection(True)
        table.setFixedHeight(min(700 if large else 300, 40 + 26 * len(contrast.rows)))
        self.column.addWidget(table)
        self.add_note("* and bold: p (Holm) ≤ 0.05. One recording: a difference here is this person's, in this "
                      "session; whether it holds for a group needs the same test over people.")
        save = QtWidgets.QPushButton("Save the table as CSV…")
        save.clicked.connect(lambda: self.save())
        self.column.addWidget(save, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.controls.append(save)

    def save(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save comparison", "comparison.csv", "CSV (*.csv)")
            if not path:
                return None
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerows(self.value.table())
        return path


def _tf_rect(channel):
    t, f = channel.times, channel.freqs
    dt = t[1] - t[0] if t.size > 1 else 1.0
    df = f[1] - f[0] if f.size > 1 else 1.0
    return QtCore.QRectF(t[0] - dt / 2, f[0] - df / 2, dt * t.size, df * f.size)


def tf_map(result, channels, condition, trial, kind):
    """The map of one channel (as core.epochs draws it), or of several averaged: their power averaged first,
    then ERD against their averaged baseline, dB of the mean power, or their amplitude maps averaged.
    `trial` -1: the condition's trials averaged."""
    if len(channels) == 1:
        channel = channels[0]
        return (ep.trial_map(result, channel, condition, trial, kind) if trial >= 0
                else ep.condition_map(result, channel, condition, kind))
    if kind == ep.AMPLITUDE_MAP and trial < 0:
        return np.mean([ep.amplitude_map(c, condition) for c in channels], axis=0)
    if trial >= 0:
        power = np.mean([c.power[condition][trial] for c in channels], axis=0)
        base = np.mean([c.baseline[condition][trial] for c in channels], axis=0)
    else:
        power = np.mean([c.power[condition].mean(axis=0) for c in channels], axis=0)
        base = np.mean([c.baseline[condition].mean(axis=0) for c in channels], axis=0)
    if kind == ep.ERD_MAP:
        reference = np.where(base > 0, base, np.nan)[:, None]
        return 100.0 * (power - reference) / reference
    if kind == ep.POWER_MAP:
        return 10 * np.log10(np.maximum(power, 1e-12))
    return ep.amplitude_from_power(power, result.fs, result.settings, result.times.size)


class PreviewWindow(QtWidgets.QDialog):
    """A block's result in a window of its own, as large as the screen allows; several can stay open."""

    def __init__(self, title, outputs, parent=None, choices=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(body)
        #: the previews shown, opened with what was chosen beside the settings (`choices`, by output name)
        self.previews = {}
        grows = False
        for key, value in outputs.items():
            widget = preview(value, large=True)
            if widget is not None:
                fills = False
                if isinstance(widget, _Preview):
                    widget.choose((choices or {}).get(key))
                    self.previews[key] = widget
                    fills = widget.fill_height()
                column.addWidget(widget, 1 if fills else 0)
                grows = grows or fills
        if not grows:
            column.addStretch(1)                # tables and notes only: at the top, not spread out
        area.setWidget(body)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(area)
        self.resize(1000, 720)


class IcaPreview(_Preview):
    """Which components were removed, why, and what each looks like on the scalp."""

    #: scalp maps per row, and at most this many maps beside the settings (all in the larger window)
    PER_ROW = 5
    SHOWN = 10

    def __init__(self, ica, large=False):
        super().__init__(ica, large)
        removed = ", ".join(str(c) for c in ica.excluded) or "none"
        self.add_note("{} components ({}). Removed: {}{}.".format(
            ica.n_components, ica.method, removed,
            " - {} matched {}".format(", ".join(str(c) for c in ica.found), ", ".join(ica.eog_channels))
            if ica.found else ""), quiet=False)
        if ica.eog_scores.size:
            plot = small_plot(150, large)
            plot.setLabel("bottom", "Component")
            plot.setLabel("left", "|r| with EOG")
            x = np.arange(ica.eog_scores.size)
            brushes = ["#e0556a" if c in ica.excluded else color("dim") for c in x]
            plot.addItem(pg.BarGraphItem(x=x, height=ica.eog_scores, width=0.7, brushes=brushes))
            self.column.addWidget(plot)
            self.add_note("Red: removed. A blink component stands far above the rest." if ica.found else
                          "No component stood out. Blinks live below 4 Hz: if a filter before ICA removes "
                          "them, put ICA before it, after a 1-40 Hz filter.", quiet=bool(ica.found))
        from ..topomap import TopoGrid
        count = ica.n_components if large else min(ica.n_components, self.SHOWN)
        order = list(ica.excluded) + [c for c in range(ica.n_components) if c not in ica.excluded]
        for start in range(0, count, self.PER_ROW):
            grid = TopoGrid()
            grid.setFixedHeight(190 if large else 120)
            panels = [("{}{}".format(c, " ✕" if c in ica.excluded else ""), ica.patterns[:, c])
                      for c in order[start:start + self.PER_ROW]]
            if not grid.draw(ica.names, panels, "weight"):
                break
            self.column.addWidget(grid)
        if count < ica.n_components:
            self.add_note("First {} of {} components; Larger shows all.".format(count, ica.n_components))
