"""Scalp maps: one value per electrode, interpolated over the head.

MNE draws these with matplotlib, which the packaged app leaves out, so this is
a small pyqtgraph version: the same azimuthal layout as the channel head map
(`montages.positions_2d`), Clough-Tocher interpolation inside the electrodes'
convex hull, nearest-electrode fill out to the head outline, and nothing drawn
outside the head. Values are never invented beyond what the electrodes carry;
the nearest-neighbour rim is only there so the map is not a jagged polygon.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

from ..core import montages as mo
from .plots import COLORBAR_WIDTH, erd_colormap
from .theme import color, plot_background

GRID = 128
OUTLINE_PEN = pg.mkPen(color("subtle"), width=2)


def interpolate(points, values, radius, size=GRID, reach=None):
    """(image (size, size) with NaN outside the head, extent).

    `reach` (same units as the points): colour only within this distance of a
    measured electrode. For a montage with unmeasured electrodes, the rim fill
    would otherwise spread four posterior values over the whole head.
    """
    finite = np.isfinite(values)
    points, values = points[finite], values[finite]
    axis = np.linspace(-radius, radius, size)
    xx, yy = np.meshgrid(axis, axis)
    image = np.full(xx.shape, np.nan)
    if len(values) == 0:
        return image
    nearest = NearestNDInterpolator(points, values)(xx, yy)
    if len(values) >= 3:
        try:
            # Anchor the rim with inverse-distance averages of the nearest electrodes, so
            # the map fades smoothly to the outline instead of in nearest-neighbour blocks.
            angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            rim = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            distance = np.linalg.norm(rim[:, None, :] - points[None, :, :], axis=2)
            k = min(3, len(values))
            closest = np.argsort(distance, axis=1)[:, :k]
            weights = 1.0 / np.maximum(np.take_along_axis(distance, closest, axis=1), 1e-6) ** 2
            rim_values = (values[closest] * weights).sum(axis=1) / weights.sum(axis=1)
            smooth = CloughTocher2DInterpolator(np.vstack([points, rim]), np.r_[values, rim_values])(xx, yy)
            image = np.where(np.isfinite(smooth), smooth, nearest)
        except Exception:  # degenerate layouts (collinear electrodes)
            image = nearest
    else:
        image = nearest
    image[np.hypot(xx, yy) > radius] = np.nan
    if reach is not None:
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        nearest_distance = np.min(np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=2), axis=1)
        image[(nearest_distance > reach).reshape(image.shape)] = np.nan
    return image


class TopoGrid(pg.GraphicsLayoutWidget):
    """Several head maps sharing one colour scale: in one row, or `columns` to a row."""

    channel_clicked = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setBackground(plot_background())
        self._items = []
        # Why no map can be drawn, in words that wrap to the room there is. A LabelItem keeps one long line,
        # centred: in a narrow view it ran off both edges and its start was lost on the left.
        self._note = QtWidgets.QLabel(self.viewport())
        self._note.setWordWrap(True)
        self._note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._note.hide()

    def clear_maps(self):
        self.clear()
        self._items = []
        self._note.hide()

    @property
    def note(self):
        """The words shown instead of a map, "" while maps are drawn."""
        return self._note.text() if not self._note.isHidden() else ""

    def _say(self, text):
        """Show `text` instead of a map, wrapped to the view's width; False, as `draw` returns then."""
        self._note.setStyleSheet("color: {}; background: transparent;".format(color("subtle")))
        self._note.setText(text)
        self._note.show()
        self._place_note()
        return False

    def _place_note(self):
        self._note.setGeometry(self.viewport().rect().adjusted(12, 8, -12, -8))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_note", None) is not None:      # pyqtgraph resizes once before __init__ is done
            self._place_note()

    def draw(self, names, panels, label, diverging=True, limit=None, highlight=(), columns=None, levels=None,
             colormap=None):
        """`panels` is [(title, values per name)]; names without a position are skipped. `columns`: maps per
        row (all in one row when None); `levels` and `colormap`: a scale shared with other maps."""
        self.clear_maps()
        positions = mo.positions_2d(names)
        placed = [i for i, n in enumerate(names) if n in positions]
        if len(placed) < 3 or not panels:
            return self._say("A scalp map needs at least three electrodes with known 10-20 positions. "
                             "Check channel names.")
        measured = [i for i in placed if any(np.isfinite(np.asarray(v, float)[i]) for _, v in panels)]
        if len(measured) < 3:
            # Two electrodes would be smeared over the whole head by the fill.
            return self._say("A scalp map needs at least three measured electrodes; {} here{}. "
                             "Switched-off or flat channels carry no data.".format(
                                 len(measured), " (" + ", ".join(names[i] for i in measured) + ")"
                                 if measured else ""))
        points = np.array([positions[names[i]] for i in placed])
        radius = max(1.0, float(np.linalg.norm(points, axis=1).max()) * 1.08)
        reach = None
        if len(measured) < len(placed):
            # Three quarters of the usual spacing between neighbouring
            # electrodes: colour stays next to what was measured.
            gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            np.fill_diagonal(gaps, np.inf)
            reach = 0.75 * float(np.median(gaps.min(axis=1)))
        stacked = np.concatenate([np.asarray(v, float)[placed] for _, v in panels])
        stacked = stacked[np.isfinite(stacked)]
        if levels is not None:
            colormap = colormap or (erd_colormap() if diverging else pg.colormap.get("viridis"))
        elif diverging:
            bound = limit or (float(np.nanpercentile(np.abs(stacked), 98)) if stacked.size else 1.0) or 1.0
            levels = (-bound, bound)
            colormap = erd_colormap()
        else:
            low, high = (np.nanpercentile(stacked, [2, 98]) if stacked.size else (0.0, 1.0))
            levels = (float(low), float(high) if high > low else float(low) + 1.0)
            colormap = pg.colormap.get("viridis")
        images = []
        columns = columns or len(panels)
        for k, (title, values) in enumerate(panels):
            plot = self.addPlot(row=k // columns, col=k % columns, title=title)
            plot.setAspectLocked(True)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            image = pg.ImageItem()
            data = interpolate(points, np.asarray(values, float)[placed], radius, reach=reach)
            image.setImage(data.T, autoLevels=False, levels=levels)
            image.setLookupTable(colormap.getLookupTable(nPts=256, alpha=True))
            image.setRect(QtCore.QRectF(-radius, -radius, 2 * radius, 2 * radius))
            plot.addItem(image)
            images.append(image)
            self._outline(plot, radius)
            here = set(measured)
            brushes = [pg.mkBrush("#111") if names[i] in highlight else pg.mkBrush("#ffffff") if i in here
                       else pg.mkBrush(0, 0, 0, 0) for i in placed]
            pens = [pg.mkPen("#222", width=0.8) if i in here else pg.mkPen(color("dim"), width=1.2)
                    for i in placed]
            scatter = pg.ScatterPlotItem(points[:, 0], points[:, 1], size=6, brush=brushes,
                                         pen=pens, hoverable=True,
                                         data=[names[i] if i in here else names[i] + " (not measured)"
                                               for i in placed],
                                         tip=lambda x, y, data: "{}".format(data))
            scatter.sigClicked.connect(lambda _s, pts, _e=None: self.channel_clicked.emit(
                pts[0].data().replace(" (not measured)", "")) if len(pts) else None)
            plot.addItem(scatter)
            pad = radius * 1.25
            plot.setXRange(-pad, pad, padding=0)
            plot.setYRange(-pad, pad, padding=0)
        bar = pg.ColorBarItem(colorMap=colormap, values=levels, label=label, interactive=False)
        bar.setImageItem(images)
        for setter in ("setMinimumWidth", "setPreferredWidth", "setMaximumWidth"):
            getattr(bar, setter)(COLORBAR_WIDTH)
        self.addItem(bar, row=0, col=min(columns, len(panels)), rowspan=(len(panels) - 1) // columns + 1)
        return True

    @staticmethod
    def _outline(plot, r):
        angle = np.linspace(0, 2 * np.pi, 240)
        plot.addItem(pg.PlotCurveItem(r * np.cos(angle), r * np.sin(angle), pen=OUTLINE_PEN))
        nose = 0.13 * r
        plot.addItem(pg.PlotCurveItem([-nose, 0.0, nose], [r * 0.99, r * 1.16, r * 0.99], pen=OUTLINE_PEN))
        for side in (-1.0, 1.0):
            t = np.linspace(-0.9, 0.9, 40)
            plot.addItem(pg.PlotCurveItem(side * (r + 0.055 * r * np.cos(t * 1.6)), 0.34 * r * np.sin(t),
                                          pen=OUTLINE_PEN))

