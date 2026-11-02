"""Plot widgets. Each one only knows how to draw what it is given."""
import math
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..core.config import BANDS
from ..core.processing import decimate_minmax
from .theme import color
from .icons import only

#: One trace on its own (a filtered signal, a filter's response): at least 3.4:1 on white and on the
#: dark plot background alike, so it reads in the window and in a report.
TRACE_COLOUR = "#4f8fd0"
TRACE_PEN = pg.mkPen(TRACE_COLOUR, width=1)
#: Event markers on the traces: unlike any channel colour, readable on white and on dark (3.5:1).
MARKER_COLOUR = "#b45309"


def marker_text(label):
    """A marker's name as written on the traces: 'Primary/P  1' -> 'P 1'.

    Recording software prefixes the kind of marker and pads the code; on a
    line across the traces only the code itself is worth the room.
    """
    text = " ".join(str(label).split("/")[-1].split())
    return text or " ".join(str(label).split())


PSD_PEN = pg.mkPen("#1a7f5a", width=2)
BAR_BRUSH = "#c0662b"

#: Eight hues, not sixteen. Sixteen colours cannot all be told apart - the old
#: sixteen-colour set had pairs operators mixed up - so each board uses the same
#: eight in the same order: Cyton pins 1-8 and Daisy pins 9-16. Identity comes
#: from the channel name on every row, grid cell and table line; where curves
#: share one plot (spectrum Overlay) the Daisy channels are dashed.
#: The order is a validated categorical order: every neighbouring pair (which
#: are neighbouring rows in the stacked view) passed colour-blind separation
#: (ΔE >= 8.4), normal-vision separation (ΔE >= 19.3) and >= 3:1 contrast on
#: white. O1/O2, the demo channels, are violet and red (ΔE 33.6).
_HUES = ("#2a78d6", "#d95926", "#199e70", "#c98500",   # blue orange aqua yellow
         "#d55181", "#008300", "#7f6fe0", "#e34948")   # magenta green violet red
CHANNEL_COLORS = _HUES + _HUES
#: Pins 9-16 sit on the Daisy board.
DAISY_FIRST = 8
#: Channel line width in pixels (Settings → Monitor → Trace width). Qt strokes
#: wide lines far more slowly: measured on the lab PC, 16 channels × 10 s at
#: 1600 px cost 3 ms a frame at 1 px and 17 ms at 2 px (1.5 px: 16 ms), the
#: difference between 30 fps and under 20. 2 px reads better on a projector.
TRACE_WIDTH = 1
TRACE_WIDTHS = (1, 2)


def set_trace_width(width):
    """Width for channel pens made from now on (existing curves: `restyle`)."""
    global TRACE_WIDTH
    TRACE_WIDTH = width if width in TRACE_WIDTHS else 1
    return TRACE_WIDTH


#: Fixed width for an image grid's colour bar, so the space it takes is
#: predictable and a neighbouring plot can reserve the same amount.
COLORBAR_WIDTH = 90


def erd_colormap():
    """Red for a decrease (ERD), white at zero, blue for an increase (ERS).

    The colours ERD/ERS maps are published in (Graimann et al. 2002, and MNE's
    ERDS maps after them); the other way round reads as its opposite to anyone
    used to them.
    """
    positions = np.linspace(0.0, 1.0, 256)
    return pg.ColorMap(positions, pg.colormap.get("CET-D1A").map(positions[::-1], mode="byte"))


def tf_scale(kind, values, units):
    """(levels, colour map, colour-bar label) for time-frequency maps of `kind` (core.epochs' *_MAP).

    ERD symmetric around zero; amplitude linear from zero, so a rhythm that stops
    shows as a gap in its band; power in dB between its 2nd and 98th percentile.
    """
    from ..core import epochs as ep
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if kind == ep.ERD_MAP:
        bound = float(np.percentile(np.abs(values), 98)) if values.size else 100.0
        return (-bound, bound if bound > 0 else 1.0), erd_colormap(), "ERD / ERS (%)"
    if kind == ep.AMPLITUDE_MAP:
        low = min(0.0, float(values.min())) if values.size else 0.0
        high = float(np.percentile(values, 99.5)) if values.size else 1.0
        return (low, high if high > low else low + 1.0), pg.colormap.get("viridis"), "Amplitude (µV)"
    low, high = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    return (float(low), float(high) if high > low else float(low) + 1.0), pg.colormap.get("viridis"), \
        "dB re 1 " + units


def channel_pen(index, width=None, alpha=255, shared_plot=False):
    """Pen for one channel. `alpha` fades it, for drawing many overlaid parts.

    `shared_plot`: the curve shares a plot with other channels, so a Daisy
    channel (same hue as its Cyton counterpart) is drawn dashed.
    """
    width = TRACE_WIDTH if width is None else width
    colour = pg.mkColor(CHANNEL_COLORS[index % len(CHANNEL_COLORS)])
    colour.setAlpha(int(alpha))
    pen = pg.mkPen(colour, width=width)
    if shared_plot and index % 16 >= DAISY_FIRST:
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
    return pen


def value_axis(plot, label, unit="", side="left"):
    """Label an axis with a fixed physical unit.

    pyqtgraph's `setLabel(units=...)` rescales ticks with an SI prefix, which
    is right for seconds but wrong for EEG units: a PSD of 5000 µV²/Hz became
    "5 kµV²/Hz", a unit that does not exist (and a kilo-microvolt squared is
    not a millivolt squared). The prefix is switched off and the unit written
    out in full.
    """
    axis = plot.getAxis(side)
    axis.enableAutoSIPrefix(False)
    plot.setLabel(side, "{} ({})".format(label, unit) if unit else label)


def plot_items(widget):
    """Every PlotItem drawn inside a widget (PlotWidget, layouts, nested scroll areas)."""
    found = []
    views = [widget] if isinstance(widget, pg.GraphicsView) else []
    views += widget.findChildren(pg.GraphicsView)
    for view in views:
        scene = view.scene()
        if scene is not None:
            found += [item for item in scene.items() if isinstance(item, pg.PlotItem)]
    return list(dict.fromkeys(found))


def set_zoom(widget, enabled):
    """Wheel/drag zoom on or off for every plot in `widget`. Off by default:
    a scroll over a graph should scroll the page, not silently rescale it."""
    for plot in plot_items(widget):
        plot.setMouseEnabled(x=enabled, y=enabled)
        plot.setMenuEnabled(enabled)


def reset_view(widget):
    for plot in plot_items(widget):
        plot.enableAutoRange()


class ZoomControls(QtWidgets.QWidget):
    """[Zoom] [Reset] for one area. Plots stay locked until Zoom is pressed;
    `apply()` re-locks plots created by a redraw."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.zoom = QtWidgets.QToolButton()
        only(self.zoom, "zoom-in", "Zoom", size=16)
        self.zoom.setCheckable(True)
        self.zoom.setToolTip("Allow mouse-wheel zoom and drag-pan on these graphs")
        self.zoom.toggled.connect(self.apply)
        reset = QtWidgets.QToolButton()
        only(reset, "rotate-ccw", "Reset", size=16)
        reset.setToolTip("Fit the graphs to their data")
        reset.clicked.connect(lambda: reset_view(self.target))
        row.addWidget(self.zoom)
        row.addWidget(reset)

    def apply(self, *args):
        set_zoom(self.target, self.zoom.isChecked())


class RowCurve(pg.PlotCurveItem):
    """One stacked trace, whose extent is known before it is drawn.

    A trace is clipped to its own row (offset ± 1) across the window, or, when
    large waves may cross rows, to the plot (`low` to `high`). Plain
    PlotCurveItem does not know that, and on every update rescans all points
    for their min and max - about a quarter of the monitor's per-frame Python
    time, measured, to rediscover a rectangle that never changes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rect = QtCore.QRectF()

    def set_row(self, x, y, span, offset, low=None, high=None, connect=None):
        low = offset - 1 if low is None else low
        high = offset + 1 if high is None else high
        # A little padding so the pen's width stays inside the repainted area.
        rect = QtCore.QRectF(-0.01 * span, low - 0.1, 1.02 * span, high - low + 0.2)
        if rect != self._rect:
            self.prepareGeometryChange()
            self._rect = rect
        self.setData(x, y, connect="all" if connect is None else connect)

    def boundingRect(self):
        return self._rect

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        r = self._rect
        return (r.left(), r.right()) if ax == 0 else (r.top(), r.bottom())


class TimeSeriesView(pg.PlotWidget):
    """Stacked, color-matched traces with persistent decimated channel curves."""

    def __init__(self):
        super().__init__(title="Time series")
        self.setLabel("bottom", "Time", units="s")
        # No grid. It was the most expensive thing on screen - its lines span
        # all sixteen rows, so every trace update repainted them - and each row
        # already carries its own scale label.
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().getViewBox().disableAutoRange()
        self._curve = pg.PlotCurveItem(pen=TRACE_PEN, antialias=False)
        self.addItem(self._curve)
        self._layout = None      # what the axis is currently configured for
        self._note = ""
        self._shades = []        # movement bands, reused between frames
        self._channel_curves = {}
        #: True: a wave larger than its row (a blink at ±150 µV) is drawn in
        #: full, over the neighbouring rows, as EEG viewers do. False: it is
        #: cut flat at the row edge. Settings → Monitor → Large waves.
        self.overflow = True
        #: False: the traces scroll, the newest sample entering on the right.
        #: True: they stand still and a cursor writes the new signal over the
        #: old from left to right, as on a clinical EEG screen: the eye rests
        #: on a picture instead of following it. Settings → Monitor → Traces.
        self.sweep = False
        self._sweep_origin = None    # (x of the oldest sample shown, window seconds) when sweeping
        self._cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color("dim"), width=1))
        self._cursor.setZValue(10)
        self._cursor.setVisible(False)
        self.addItem(self._cursor)
        # Ruled like EEG paper: a line every second, finer ones every 0.2 s when
        # the window is short enough to tell them apart, and each row's zero.
        # Three plain items behind the traces, touched only when the lines move:
        # the axis grid this replaces repainted sixteen rows of ticks every frame.
        self._grid = {}
        for name, colour, depth in (("minor", color("grid"), -32), ("zero", color("line"), -31),
                                    ("major", color("field"), -30)):
            item = pg.PlotCurveItem(pen=pg.mkPen(colour, width=1), connect="pairs", antialias=False)
            item.setZValue(depth)
            self.addItem(item)
            self._grid[name] = item
        self._grid_key = None
        # Event markers of the recording being played: a line and its label.
        self._marker_times = np.empty(0)
        self._marker_names = []
        self._marker_lines = pg.PlotCurveItem(pen=pg.mkPen(MARKER_COLOUR, width=1), connect="pairs",
                                              antialias=False)
        self._marker_lines.setZValue(5)
        self.addItem(self._marker_lines)
        self._marker_labels = []     # text items, reused between frames
        self._marker_key = None
        for side in ('left', 'bottom'):
            self.getAxis(side).setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)

    #: Finer 0.2 s lines are drawn up to this many seconds across the plot.
    MINOR_GRID_SECONDS = 6.0
    #: Marker labels drawn at once; more than this is a blur, not information.
    MAX_MARKER_LABELS = 40

    def set_markers(self, onsets=(), labels=()):
        """The events to mark on the traces: seconds from the first sample, and their names."""
        onsets = np.asarray(list(onsets), dtype=float)
        labels = [marker_text(label) for label in labels]
        keep = np.isfinite(onsets)
        order = np.argsort(onsets[keep], kind="stable")
        self._marker_times = onsets[keep][order]
        self._marker_names = [name for name, ok in zip(labels, keep) if ok]
        self._marker_names = [self._marker_names[i] for i in order]
        self._marker_key = None

    def _draw_grid(self, span, first, low, high, offsets):
        """`first`: where the first whole second falls on the x axis."""
        key = (round(span, 6), round(first, 4), len(offsets))
        if key == self._grid_key:
            return
        self._grid_key = key

        def verticals(positions):
            return np.repeat(positions, 2), np.tile([low, high], len(positions))

        seconds = np.arange(first, span + 1e-9, 1.0)
        self._grid["major"].setData(*verticals(seconds))
        if span <= self.MINOR_GRID_SECONDS:
            fifths = np.round(np.arange(first - 1.0, span + 1e-9, 0.2), 9)     # -0.0000…1 is second 0
            self._grid["minor"].setData(*verticals(fifths[(fifths >= 0) & (fifths <= span)]))
        else:
            self._grid["minor"].setData([], [])
        self._grid["zero"].setData(np.tile([0.0, span], len(offsets)), np.repeat(offsets, 2))

    def _draw_markers(self, places, high, low):
        """`places`: [(x, label)] of the markers inside the window."""
        key = (tuple(places), high, low)
        if key == self._marker_key:
            return
        self._marker_key = key
        xs = np.array([x for x, _ in places], dtype=float)
        self._marker_lines.setData(np.repeat(xs, 2), np.tile([low, high], len(xs)))
        places = places[:self.MAX_MARKER_LABELS]
        while len(self._marker_labels) < len(places):
            label = pg.TextItem(color=MARKER_COLOUR, anchor=(0, 0))
            label.setZValue(6)
            self.addItem(label)
            self._marker_labels.append(label)
        for item, (x, text) in zip(self._marker_labels, places):
            if item.toPlainText() != text:
                item.setText(text)
            item.setPos(x, high)
            item.setVisible(True)
        for item in self._marker_labels[len(places):]:
            item.setVisible(False)

    def _clear_rulings(self):
        for item in self._grid.values():
            item.setData([], [])
        self._grid_key = None
        self._draw_markers([], 0.0, 0.0)

    #: Share of the window left blank ahead of the sweep cursor, so the newest
    #: signal is not read as joined to the oldest.
    SWEEP_GAP = 0.03

    def _sweep_parts(self, block, position, window):
        """Where a sweep puts the last samples: [(samples, first slot)], left to right.

        Sample number k of the stream always sits at slot k mod window, so a
        wave stays where it was written until the cursor comes round again.
        Returns the parts, the cursor's slot and the oldest sample's slot.
        """
        position, window = int(position), int(window)
        origin = (position - block.shape[1]) % window
        # The oldest samples lie just ahead of the cursor: leave them out.
        trim = max(int(window * self.SWEEP_GAP) - (window - block.shape[1]), 0)
        block = block[:, min(trim, max(block.shape[1] - 1, 0)):]
        first = (position - block.shape[1]) % window
        kept = min(block.shape[1], window - first)      # samples before the slots wrap to 0
        parts = [(block[:, kept:], 0), (block[:, :kept], first)] if kept < block.shape[1] else [(block, first)]
        return parts, position % window, origin

    def restyle(self):
        """Give the existing channel curves the current line width."""
        for channel, curve in self._channel_curves.items():
            curve.setPen(channel_pen(channel))

    def shade_spans(self, spans):
        """Mark these (start, end) second ranges behind the traces.

        Used for the moments the motion sensor says the head moved, so an
        artefact can be told from the signal at a glance. The region items are
        reused: making and destroying them every frame was visible as a stall.
        """
        spans = [(float(a), float(b)) for a, b in spans if b > a]
        if self._sweep_origin is not None:
            # Spans count from the oldest sample shown; a sweep has that sample
            # somewhere in the middle, and a span may run off the right edge.
            start, width = self._sweep_origin
            moved = []
            for a, b in spans:
                a, length = (start + a) % width, min(b - a, width)
                moved += [(a, a + length)] if a + length <= width else [(a, width), (0.0, a + length - width)]
            spans = moved
        while len(self._shades) < len(spans):
            region = pg.LinearRegionItem(movable=False, brush=(200, 80, 80, 28),
                                         pen=pg.mkPen(None))
            region.setZValue(-20)
            region.setVisible(False)
            self.addItem(region)
            self._shades.append(region)
        for region, span in zip(self._shades, spans):
            region.setRegion(span)
            region.setVisible(True)
        for region in self._shades[len(spans):]:
            region.setVisible(False)

    def draw(self, data, channels, amplitude, fs, names, note="", position=None, window=None):
        """data: (n_ch, n) uV. channels: indices to show. amplitude: uV per row.

        `fs` and `names` come from the recording rather than module constants,
        so a file from another system gets a correct time axis and its own
        electrode labels. `note` is appended to the title, for what is left out
        of the drawing (switched-off channels). `position` (samples of the
        stream up to the end of `data`) and `window` (samples in a full
        window) are what a sweep needs to know where each sample belongs.
        """
        self._note = note
        if not channels or data.shape[1] == 0:
            self._curve.setData([], [])
            for curve in self._channel_curves.values():
                curve.setData([], [])
            self._cursor.setVisible(False)
            self._sweep_origin = None
            self._clear_rulings()
            # Nothing to draw is worth saying, not an empty box: every selected
            # channel may simply be switched off.
            self.setTitle("Time series" + (note or ": no channel to show"))
            self._layout = None
            return

        selected = data[channels]
        sweeping = bool(self.sweep and position is not None and window and window >= selected.shape[1])
        if sweeping:
            parts, cursor, origin = self._sweep_parts(selected, position, window)
            span = window / fs
            self._sweep_origin = (origin / fs, span)
            self._cursor.setPos(cursor / fs)
        else:
            parts, span = [(selected, 0)], data.shape[1] / fs
            self._sweep_origin = None
        self._cursor.setVisible(sweeping)
        # Decimate to about the pixel width first: the screen cannot show more,
        # and the points it cannot show are most of the cost.
        # Half the pixel width in bins, because each bin yields two points -
        # its min and its max - which lands at about one point per column.
        # Decimate before clipping: clipping is monotonic, so the two commute,
        # and doing it second means clipping a few thousand numbers instead of
        # a few hundred thousand.
        bins = max(self.width() // 2, 100)
        shown = max(span * fs, 1)                       # samples across the whole plot
        drawn = [(decimate_minmax(part, max(round(bins * part.shape[1] / shown), 1), return_indices=True), first)
                 for part, first in parts if part.shape[1]]
        block = np.hstack([values for (values, _), _ in drawn])
        sample_indices = np.hstack([indices + first for (_, indices), first in drawn])
        n_points = block.shape[1]
        # A sweep has two stretches per row, the new and the old: the line
        # lifts between them instead of joining the newest sample to the oldest.
        joined = None
        if len(drawn) > 1:
            joined = np.ones(n_points, dtype=np.uint8)
            joined[drawn[0][0][0].shape[1] - 1] = 0
        per_channel = np.ndim(amplitude) > 0
        scales = np.broadcast_to(np.asarray(amplitude, dtype=float), (len(channels),))
        # Normalize only display coordinates. Input samples and power stay in uV.
        block = block / scales[:, None]

        spacing = 2.0
        offsets = (len(channels) - 1 - np.arange(len(channels))) * spacing
        # The plot's own edges: a railed channel then lies along the top or the
        # bottom of the plot instead of leaving it.
        low, high = -spacing, (len(channels) - 1) * spacing + spacing
        x = (sample_indices / fs).ravel()
        if self.overflow:
            y = np.clip(block + offsets[:, np.newaxis], low, high).ravel()
        else:
            y = (np.clip(block, -1, 1) + offsets[:, np.newaxis]).ravel()

        # Persistent, color-matched channel curves for the operator overview.
        for channel, curve in self._channel_curves.items():
            if channel not in channels:
                curve.setData([], [])
        for slot, channel in enumerate(channels):
            if channel not in self._channel_curves:
                curve = RowCurve(pen=channel_pen(channel), antialias=False)
                self.addItem(curve)
                self._channel_curves[channel] = curve
            start = slot * n_points
            self._channel_curves[channel].set_row(
                x[start:start+n_points], y[start:start+n_points], span, offsets[slot],
                *((low, high) if self.overflow else (None, None)), connect=joined)

        # Rulings and markers belong to time. Sweeping, the plot is a fixed
        # ruler from its left edge; scrolling, a line stays on its second and
        # moves with the signal. Without a position the left edge is second 0.
        count = selected.shape[1]
        start = 0.0 if position is None else (float(position) - count) / fs      # stream time of the oldest sample
        self._draw_grid(span, 0.0 if sweeping else (np.ceil(start - 1e-9) - start) % 1.0, low, high, offsets)
        places = []
        if self._marker_times.size and position is not None:
            first, last = np.searchsorted(self._marker_times, [start, start + count / fs])
            for i in range(first, last):
                when = float(self._marker_times[i])
                x = (when * fs % window) / fs if sweeping else when - start
                places.append((round(x, 4), self._marker_names[i]))
        self._draw_markers(places, high, low)

        self._sync_axes(channels, tuple(scales), fs, names, span, spacing, offsets, per_channel)

    def _sync_axes(self, channels, amplitude, fs, names, span, spacing, offsets, per_channel):
        """Ticks, ranges and title - only when they have actually changed.

        These each force a layout pass, and at twenty frames a second that is
        twenty needless relayouts for a set of labels that has not moved.
        """
        # Keyed on the text shown, not the raw scales: the auto scaler moves by
        # fractions of a microvolt that print identically, and keying on those
        # rebuilt the axes on almost every frame.
        labels = tuple(f"{names[ch]}  ±{_fmt(scale)} uV"
                       for ch, scale in zip(channels, amplitude))
        title = ("Time series, each channel on its own scale" if per_channel else
                 "Time series, one ±{} uV scale".format(_fmt(amplitude[0])))
        title += self._note
        layout = (tuple(channels), labels, round(span, 1), title)
        if layout == self._layout:
            return
        self._layout = layout

        self.getAxis("left").setTicks(
            [[(float(offset), label) for offset, label in zip(offsets, labels)]])
        self.setYRange(-spacing, (len(channels) - 1) * spacing + spacing)
        self.setXRange(0, max(span, 1e-3))
        self.setTitle(title)


class SpectrumView(pg.PlotWidget):
    """Welch power spectrum of a single channel."""

    #: values below this are treated as zero; log10 of a true zero is -inf
    LOG_FLOOR = 1e-6

    #: decades to show below the peak. The stop band sits near LOG_FLOOR, so
    #: without a cap the axis spans ~7 decades and squashes the real signal.
    LOG_DECADES = 5.0

    def __init__(self):
        super().__init__(title="Power spectrum")
        self.setLabel("bottom", "Frequency", units="Hz")
        self.showGrid(x=True, y=True, alpha=0.2)
        # Locked like the other two monitor plots. A brushed trackpad otherwise
        # zooms the spectrum by accident and it never comes back on its own,
        # because every draw sets the range explicitly only on the y axis.
        self.setMouseEnabled(x=False, y=False)
        self._curve = self.plot(pen=PSD_PEN)
        self._log = None  # force the first draw to configure the axis

    def draw(self, freqs, power, log=True, channel="", y_range=None):
        """`y_range` is a scale decided elsewhere (the Spectrum panel's).

        Without it this view fitted its own axis to every 0.5 s update, so the
        curve jumped while the panel beside it stayed still.
        """
        if freqs.size == 0:
            self._curve.setData([], [])
            return

        if log != self._log:
            self.setLogMode(x=False, y=log)
            self.setLabel("left", "uV^2/Hz (log)" if log else "uV^2/Hz")
            self._log = log

        if y_range is not None:
            self._curve.setData(freqs, np.maximum(power, self.LOG_FLOOR) if log else power)
            self.setYRange(*y_range, padding=0)
            self.setTitle("Power spectrum{} ({})".format(
                " - " + channel if channel else "", "log" if log else "linear"))
            return

        # Both branches set the range explicitly. Autorange is not reliable
        # across a log/linear switch - it keeps the range from the old units.
        if log:
            floored = np.maximum(power, self.LOG_FLOOR)
            self._curve.setData(freqs, floored)
            high = math.log10(float(floored.max()))
            low = max(math.log10(float(floored.min())), high - self.LOG_DECADES)
            if high - low < 1.0:  # a flat channel would give a zero-height axis
                low = high - 1.0
            pad = 0.05 * (high - low)
            self.setYRange(low - pad, high + pad)
        else:
            self._curve.setData(freqs, power)
            # a dead channel returns ~1e-53; never let that define the axis
            top = float(power.max())
            self.setYRange(0, top * 1.1 if top > 1e-3 else 1.0)

        self.setTitle("Power spectrum{} ({})".format(
            " - " + channel if channel else "", "log" if log else "linear"))


class BandPowerView(pg.PlotWidget):
    """Display-smoothed band powers on a shared, held scale."""

    UPDATE_SECONDS = 0.5
    SMOOTH_SECONDS = 2.0
    #: time constant of the movement between two updates
    GLIDE_SECONDS = 0.3

    def __init__(self):
        super().__init__(title="Band power (smoothed over 2 s)")
        self.setLabel("left", "uV^2")
        self.showGrid(y=True, alpha=0.2)
        self.setMouseEnabled(x=False, y=False)
        self.getAxis("bottom").setTicks([[(i, b[0]) for i, b in enumerate(BANDS)]])
        self._bars = pg.BarGraphItem(x=list(range(len(BANDS))),
                                     height=[0.0] * len(BANDS),
                                     width=0.6, brush=BAR_BRUSH)
        self.addItem(self._bars)

        self.setXRange(-0.6, len(BANDS) - 0.4, padding=0)
        self.setToolTip("Updates twice per second with 2 s smoothing. The shared scale "
                        "only expands. Right-click → Fit band power scale to refit it.")
        self.getViewBox().menu.addAction("Fit band power scale", self.fit_scale)
        self.reset()

    def reset(self):
        """Do not blend powers across sources, channels, filters or seeks."""
        self._smoothed = None
        self._drawn = None
        self._last_draw = None
        self._last_glide = None
        self._scale_top = 1.0
        self._bars.setOpts(height=[0.0] * len(BANDS))
        self.setYRange(0, self._scale_top, padding=0)

    def fit_scale(self):
        heights = self._drawn if self._drawn is not None else self._smoothed
        top = float(np.max(heights)) if heights is not None else 0.0
        target = max(1.0, top * 1.25)
        decade = 10 ** math.floor(math.log10(target))
        self._scale_top = next(v * decade for v in (1, 2, 5, 10)
                               if v * decade >= target)
        self.setYRange(0, self._scale_top, padding=0)

    def draw(self, heights, now=None):
        """New powers arrive twice a second; the bars move every frame.

        Drawn as they arrive, the bars stepped twice a second beside traces
        moving twenty times a second, which reads as stutter. The target is
        updated at UPDATE_SECONDS and the drawn heights glide towards it.
        """
        now = time.perf_counter() if now is None else now
        values = np.asarray(heights, dtype=float)
        fresh = values.shape == (len(BANDS),) and np.all(np.isfinite(values))
        due = self._last_draw is None or now - self._last_draw >= self.UPDATE_SECONDS
        if fresh and due:
            values = np.maximum(values, 0.0)
            if self._smoothed is None:
                self._smoothed = values.copy()
            else:
                # A playback pause must not bypass smoothing on the next frame.
                dt = min(now - self._last_draw, self.UPDATE_SECONDS)
                weight = -math.expm1(-dt / self.SMOOTH_SECONDS)
                self._smoothed += weight * (values - self._smoothed)
            self._last_draw = now
        if self._smoothed is None:
            return
        self._glide(now)

    def _glide(self, now):
        first = self._drawn is None
        if first or self._last_glide is None or now < self._last_glide:
            self._drawn = self._smoothed.copy()
        else:
            step = -math.expm1(-min(now - self._last_glide, 1.0) / self.GLIDE_SECONDS)
            self._drawn = self._drawn + step * (self._smoothed - self._drawn)
        self._last_glide = now
        self._bars.setOpts(height=self._drawn.copy())
        if first or float(np.max(self._drawn)) > self._scale_top:
            self.fit_scale()


class MultiCurveView(pg.PlotWidget):
    """Several channels overlaid on shared axes, with a legend.

    Used where comparing channels matters more than isolating them: spectra
    and ERD traces. One optional shaded band shows a spread (SD or SEM).
    """

    def __init__(self):
        super().__init__()
        self.showGrid(x=True, y=True, alpha=0.2)
        self._legend = self.addLegend(offset=(-10, 10))
        self._curves = []
        self._shade = None
        self._zero = None

    def _clear_curves(self):
        for item in self._curves:
            self.removeItem(item)
        self._curves = []
        self._legend.clear()
        if self._shade is not None:
            self.removeItem(self._shade)
            self._shade = None

    def draw(self, series, log_y=False, shade=None, zero_line=False):
        """`series` is [(label, x, y, colour_index)].

        `shade` is an optional (x, low, high) drawn behind the first curve.
        """
        self._clear_curves()
        self.setLogMode(x=False, y=log_y)

        if shade is not None:
            x, low, high = shade
            upper = pg.PlotDataItem(x, high, pen=None)
            lower = pg.PlotDataItem(x, low, pen=None)
            self._shade = pg.FillBetweenItem(upper, lower, brush=(31, 78, 121, 55))
            self.addItem(self._shade)

        for label, x, y, index in series:
            self._curves.append(self.plot(x, y, pen=channel_pen(index), name=label))

        if zero_line and self._zero is None:
            self._zero = pg.InfiniteLine(
                angle=0, pen=pg.mkPen(color("dim"), style=QtCore.Qt.PenStyle.DashLine))
            self.addItem(self._zero)
        if self._zero is not None:
            self._zero.setVisible(zero_line)


def _fmt(value):
    return "{:g}".format(round(value, 2))
