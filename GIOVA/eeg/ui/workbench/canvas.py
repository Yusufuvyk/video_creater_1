"""The Workbench sheet: blocks, their ports and the wires between them.

The canvas only draws a `pipeline.Graph` and edits it through the Graph's own
methods, so every rule (a wire's type must fit, no loops) is the engine's and
the sheet can never hold a pipeline the engine would refuse.

Mouse and touchpad: drag a block to move it; drag from a port to another to wire
them (the ports that would accept the wire light up); drag on empty sheet to move
around (also the middle button, or Space + drag); Shift + drag on empty sheet to
select. Two-finger scroll (or the wheel) moves around; pinch, Ctrl + wheel or the
+ / - keys zoom. Delete removes the selection.
"""
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ...pipeline import block, graph as g
from ..theme import color

#: One colour per kind of data, on the ports and the wires that carry it: mid tones that read on the
#: dark and the light sheet alike, and no red - on GIOVA's sheet red only means an error.
TYPE_COLOURS = {g.RAW: "#5B8FD0", g.EPOCHS: "#3FA89A", g.SPECTRUM: "#D2A04A", g.FEATURES: "#9A7FD0",
                g.MODEL: "#8B97A8", g.SCORES: "#C07AA6", g.TF: "#C98A4E", g.ICA: "#6FA8C8",
                g.CONTRAST: "#7FAE5A"}
TYPE_NAMES = {g.RAW: "recording", g.EPOCHS: "trials", g.SPECTRUM: "spectrum", g.FEATURES: "features",
              g.MODEL: "model", g.SCORES: "scores", g.TF: "time-frequency", g.ICA: "ICA components",
              g.CONTRAST: "windows compared"}
#: The thin band along the top of a block, by its category (kept on top, as in v3, at the user's choice on
#: 31 October 2026): muted, so the block body stays neutral and the accent (selection) and the state colours
#: stand out. Machine learning's steps share one mauve
#: family; no red.
CATEGORY_COLOURS = {"Input": "#6E8FB5", "Preprocessing": "#5E9EA6", "Trials": "#7FA27A",
                    "Spectral": "#C49A5A", "Feature extraction": "#9A86C4", "Statistics": "#94A05E",
                    "Machine learning": "#B07A9E",
                    "Data augmentation": "#A98BAF", "Spatial filters": "#B0809A", "Riemannian geometry": "#9F7FB5",
                    "Scaling & selection": "#A88A9C", "Classifiers": "#B07A9E", "Evaluation": "#8B97A8"}
#: Each category's icon (icons.py), beside its name in the Blocks panel.
CATEGORY_ICONS = {"Input": "recording", "Preprocessing": "filter-wave", "Trials": "trials", "Spectral": "spectrum",
                  "Feature extraction": "features", "Statistics": "statistics", "Machine learning": "classifier",
                  "Data augmentation": "augmentation", "Spatial filters": "spatial-filter",
                  "Riemannian geometry": "riemann", "Scaling & selection": "sliders-horizontal",
                  "Classifiers": "classifier", "Evaluation": "evaluation"}
#: The category band's height.
STRIPE = 4
BLOCK_MIME = "application/x-eegstudio-block"

WIDTH = 176
HEADER = 26
ROW = 20
RADIUS = 6
PORT = 6
#: Fit never zooms out further than this: smaller text cannot be read. 0.5, not 0.75: at 0.75 a seven-block
#: pipeline did not fit a laptop's sheet and Fit seemed to do nothing (31 October 2026); block names still read.
MIN_FIT = 0.5
#: Blocks dragged by hand land on this grid, so a pipeline lines up without effort.
SNAP = 10
#: The sheet's dots: one every GRID, a stronger one every GRID * MAJOR.
GRID = 20
MAJOR = 5


class PortItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, node_item, port, output):
        super().__init__(-PORT, -PORT, 2 * PORT, 2 * PORT, node_item)
        self.node_item, self.port, self.output = node_item, port, output
        self.type = port.types[0]
        if output or len(port.types) == 1:
            self.setBrush(QtGui.QColor(TYPE_COLOURS[self.type]))
        else:
            # an input that takes several kinds of data: their colours side by side
            gradient = QtGui.QLinearGradient(-PORT, 0, PORT, 0)
            for k, kind in enumerate(port.types):
                gradient.setColorAt(k / len(port.types), QtGui.QColor(TYPE_COLOURS[kind]))
                gradient.setColorAt((k + 1) / len(port.types) - 1e-3, QtGui.QColor(TYPE_COLOURS[kind]))
            self.setBrush(QtGui.QBrush(gradient))
        self.setPen(QtGui.QPen(QtGui.QColor(color("block")), 1.5))
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        kinds = " or ".join(TYPE_NAMES[t] for t in port.types)
        self.setToolTip("{} - {}{}".format(port.label or port.name, kinds, " (optional)" if port.optional else ""))
        self.wires = []

    @property
    def node_id(self):
        return self.node_item.node_id

    def accepts(self, other):
        """Could a wire join this port and `other`? Only the types are checked here."""
        if other.output == self.output or other.node_item is self.node_item:
            return False
        out, inp = (other, self) if other.output else (self, other)
        return out.type in inp.port.types

    def light(self, on):
        """Grow when a wire being drawn could end here."""
        self.setScale(1.6 if on else 1.0)

    def hoverEnterEvent(self, event):
        self.setScale(1.4)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)


class NodeItem(QtWidgets.QGraphicsObject):
    """One block: a card with its title, its main setting, ports and its latest result."""

    def __init__(self, canvas, node_id):
        super().__init__()
        self.canvas, self.node_id = canvas, node_id
        self.kind = block(self.node.type)
        self.state = ""          # "", "done", "error", "stale"; "running", "waiting" during a run
        self.result = ""         # one line under the card after a run
        self.message = ""        # the error, in the tooltip
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.inputs = [PortItem(self, p, False) for p in self.kind.inputs]
        self.outputs = [PortItem(self, p, True) for p in self.kind.outputs]
        rows = max(len(self.inputs), len(self.outputs), 1)
        self.height = HEADER + 22 + rows * ROW
        for i, port in enumerate(self.inputs):
            port.setPos(0, HEADER + 22 + ROW * i + ROW / 2)
        for i, port in enumerate(self.outputs):
            port.setPos(WIDTH, HEADER + 22 + ROW * i + ROW / 2)
        self.setPos(*self.node.position)
        self.setZValue(1)
        # drawn once and reused while it moves: dragging stays smooth with many blocks
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)

    @property
    def node(self):
        return self.canvas.graph.nodes[self.node_id]

    def port(self, name, output):
        for item in (self.outputs if output else self.inputs):
            if item.port.name == name:
                return item
        raise KeyError(name)

    def boundingRect(self):
        return QtCore.QRectF(-PORT - 2, -2, WIDTH + 2 * PORT + 4, self.height + 22)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        card = QtCore.QRectF(0, 0, WIDTH, self.height)
        border = {"error": color("bad"), "stale": color("stale"), "running": color("accent")}.get(self.state,
                                                                                              color("block_border"))
        thick = self.isSelected() or self.state == "running"
        pen = QtGui.QPen(QtGui.QColor(color("accent") if self.isSelected() else border), 2.4 if thick else 1.2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(color("block")))
        painter.drawRoundedRect(card, RADIUS, RADIUS)
        outline = QtGui.QPainterPath()
        outline.addRoundedRect(card, RADIUS, RADIUS)
        stripe = QtGui.QPainterPath()
        stripe.addRect(QtCore.QRectF(0, 0, WIDTH, STRIPE))
        painter.fillPath(outline.intersected(stripe), QtGui.QColor(CATEGORY_COLOURS.get(self.kind.category,
                                                                                      color("dim"))))

        title = QtGui.QFont(painter.font())
        title.setBold(True)
        painter.setFont(title)
        painter.setPen(QtGui.QColor(color("text")))
        metrics = QtGui.QFontMetrics(title)
        name = metrics.elidedText(self.node.label or self.kind.title, QtCore.Qt.TextElideMode.ElideRight, WIDTH - 30)
        painter.drawText(QtCore.QRectF(10, 7, WIDTH - 30, 18), QtCore.Qt.AlignmentFlag.AlignVCenter, name)
        self._paint_state(painter)

        small = QtGui.QFont(painter.font())
        small.setBold(False)
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1))
        painter.setFont(small)
        metrics = QtGui.QFontMetrics(small)
        painter.setPen(QtGui.QColor(color("subtle")))
        summary = self.kind.short(self.canvas.graph.params(self.node_id))
        painter.drawText(QtCore.QRectF(10, HEADER, WIDTH - 20, 18), QtCore.Qt.AlignmentFlag.AlignVCenter,
                         metrics.elidedText(summary, QtCore.Qt.TextElideMode.ElideRight, WIDTH - 20))
        painter.setPen(QtGui.QColor(color("header_text")))
        half = WIDTH / 2 - 14
        for item in self.inputs:
            painter.drawText(QtCore.QRectF(12, item.y() - 9, half, 18), QtCore.Qt.AlignmentFlag.AlignVCenter,
                             metrics.elidedText(item.port.label or item.port.name,
                                                QtCore.Qt.TextElideMode.ElideRight, int(half)))
        for item in self.outputs:
            painter.drawText(QtCore.QRectF(WIDTH / 2 + 2, item.y() - 9, half, 18),
                             QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight,
                             metrics.elidedText(item.port.label or item.port.name,
                                                QtCore.Qt.TextElideMode.ElideRight, int(half)))
        if self.result or self.message:
            painter.setPen(QtGui.QColor({"error": color("bad"), "stale": color("stale"),
                                         "running": color("accent")}.get(self.state, color("good"))))
            text = self.message if self.state == "error" else self.result
            painter.drawText(QtCore.QRectF(0, self.height + 2, WIDTH, 18), QtCore.Qt.AlignmentFlag.AlignVCenter,
                             metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, WIDTH))

    def _paint_state(self, painter):
        colour = {"done": color("good"), "error": color("bad"), "stale": color("stale"),
                  "running": color("accent")}.get(self.state)
        if colour:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(colour))
            painter.drawEllipse(QtCore.QPointF(WIDTH - 13, 16), 4.5, 4.5)

    def set_state(self, state, result="", message=""):
        """"done", "error", "stale"; during a run "running" (the block being computed: drawn in the accent
        colour) and "waiting" (still to come in this run: faint); "" nothing to say."""
        if (state, result, message) == (self.state, self.result, self.message):
            return
        self.state, self.result, self.message = state, result, message
        self.setToolTip(message or result)
        self.setOpacity(0.45 if state == "waiting" else 1.0)
        self.update()

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.canvas.snapping:
            return QtCore.QPointF(round(value.x() / SNAP) * SNAP, round(value.y() / SNAP) * SNAP)
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for port in self.inputs + self.outputs:
                for wire in port.wires:
                    wire.reroute()
            self.node.position = (round(self.pos().x(), 1), round(self.pos().y(), 1))
            self.canvas.dragged = True
        return super().itemChange(change, value)


class WireItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, source, target):
        super().__init__()
        self.source, self.target = source, target
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(0)
        self.colour = QtGui.QColor(TYPE_COLOURS[source.type])
        source.wires.append(self)
        target.wires.append(self)
        self.reroute()

    def reroute(self):
        self.setPath(curve(self.source.scenePos(), self.target.scenePos()))

    def paint(self, painter, option, widget=None):
        pen = QtGui.QPen(self.colour, 3.2 if self.isSelected() else 2.0)
        if self.isSelected():
            pen.setColor(QtGui.QColor(color("accent")))
        self.setPen(pen)
        option.state &= ~QtWidgets.QStyle.StateFlag.State_Selected      # no dotted box
        super().paint(painter, option, widget)

    def shape(self):
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(10)
        return stroker.createStroke(self.path())

    def detach(self):
        for port in (self.source, self.target):
            if self in port.wires:
                port.wires.remove(self)


def curve(a, b):
    path = QtGui.QPainterPath(a)
    bend = max(40.0, abs(b.x() - a.x()) / 2)
    path.cubicTo(QtCore.QPointF(a.x() + bend, a.y()), QtCore.QPointF(b.x() - bend, b.y()), b)
    return path


class Canvas(QtWidgets.QGraphicsView):
    """Shows one Graph and edits it. `changed` fires on every edit the pipeline would notice."""

    selection = QtCore.Signal(object)     # node id or None
    changed = QtCore.Signal()             # structure or parameters
    message = QtCore.Signal(str)
    run_to = QtCore.Signal(str)           # "run up to this block", from its menu
    help_for = QtCore.Signal(str)         # a block type whose Help page is asked for, from its menu
    moved = QtCore.Signal()               # blocks were dragged to new places
    released = QtCore.Signal()            # the mouse button is up again

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.scene().setSceneRect(-5000, -5000, 10000, 10000)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        # no background cache: Qt clears it on zoom but not on scroll, and shifts it by whole pixels while
        # the squares sit on fractional ones - stray, doubled lines and missing pieces when the sheet is
        # panned. A few hundred lines are drawn in no time.
        self.setCacheMode(QtWidgets.QGraphicsView.CacheModeFlag.CacheNone)
        self.setBackgroundBrush(QtGui.QColor(color("sheet")))
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.scene().selectionChanged.connect(self._selection_changed)
        self.graph = g.Graph()
        self.cards = {}
        self._drag = None          # (port item, temporary path item) while a wire is drawn
        self._pan = None
        self._panned = False
        self._space = False
        self.dragged = False
        #: templates and files keep their own places; only hand-dragged blocks snap
        self.snapping = False
        #: True while the left button is down on the sheet
        self.pressing = False

    # ---------------- building ----------------

    def set_graph(self, graph):
        self.graph = graph
        self.scene().clear()
        self.cards = {}
        self._drag = None
        for node_id in graph.nodes:
            self._add_item(node_id)
        for wire in graph.wires:
            self._add_wire(wire)
        self.dragged = False
        self.fit()

    def _add_item(self, node_id):
        self.snapping = False
        item = NodeItem(self, node_id)
        self.snapping = True
        self.scene().addItem(item)
        self.cards[node_id] = item
        return item

    def _add_wire(self, wire):
        source = self.cards[wire.source].port(wire.output, True)
        target = self.cards[wire.target].port(wire.input, False)
        item = WireItem(source, target)
        self.scene().addItem(item)
        return item

    def add_block(self, type_name, position=None, after=None):
        """Add a block. With `after` (a node id) it is placed to its right and wired to it when it fits."""
        kind = block(type_name)
        if position is None:
            if after is not None and after in self.cards:
                anchor = self.cards[after].pos()
                position = QtCore.QPointF(anchor.x() + WIDTH + 50, anchor.y())
                while any((item.pos() - position).manhattanLength() < 40 for item in self.cards.values()):
                    position += QtCore.QPointF(0, 120)
            else:
                position = self.mapToScene(self.viewport().rect().center()) - QtCore.QPointF(WIDTH / 2, 40)
        node_id = self.graph.add(type_name, position=(position.x(), position.y()))
        self._add_item(node_id)
        if after is not None and after in self.graph.nodes:
            out = block(self.graph.nodes[after].type).outputs
            for port in kind.inputs:
                if out and out[0].types[0] in port.types:
                    self.wire(after, out[0].name, node_id, port.name)
                    break
        self.select(node_id)
        self.changed.emit()
        return node_id

    def wire(self, source, output, target, input):
        """Wire two ports through the Graph; False and a message when it refuses."""
        try:
            self.graph.connect(source, output, target, input)
        except ValueError as exc:
            self.message.emit(str(exc))
            return False
        self._sync_wires()
        self.changed.emit()
        return True

    def _sync_wires(self):
        """Rebuild the wire items from the Graph: a new wire on an input replaces the old one."""
        for item in [i for i in self.scene().items() if isinstance(i, WireItem)]:
            item.detach()
            self.scene().removeItem(item)
        for wire in self.graph.wires:
            self._add_wire(wire)

    def delete_selected(self):
        nodes = [i.node_id for i in self.scene().selectedItems() if isinstance(i, NodeItem)]
        wires = [i for i in self.scene().selectedItems() if isinstance(i, WireItem)]
        if not nodes and not wires:
            return
        for wire in wires:
            self.graph.disconnect(wire.target.node_id, wire.target.port.name)
        for node_id in nodes:
            self.graph.remove(node_id)
            self.scene().removeItem(self.cards.pop(node_id))
        self._sync_wires()
        self.selection.emit(None)
        self.changed.emit()

    def refresh(self, node_id=None):
        """Redraw after a parameter or name changed."""
        for item in ([self.cards[node_id]] if node_id in self.cards else self.cards.values()):
            item.update()

    # ---------------- selection and view ----------------

    def selected(self):
        nodes = [i.node_id for i in self.scene().selectedItems() if isinstance(i, NodeItem)]
        return nodes[0] if len(nodes) == 1 else None

    def select(self, node_id):
        self.scene().clearSelection()
        if node_id in self.cards:
            self.cards[node_id].setSelected(True)

    def _selection_changed(self):
        self.selection.emit(self.selected())

    def fit(self):
        if not self.cards:
            self.resetTransform()
            self.centerOn(0, 0)
            return
        box = self.scene().itemsBoundingRect().adjusted(-40, -40, 40, 40)
        self.fitInView(box, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        scale = self.transform().m11()
        if scale > 1.0:                           # never blow a small pipeline up
            self.resetTransform()
            self.centerOn(box.center())
        elif scale < MIN_FIT:                     # nor shrink a long one past reading: scroll instead
            self.setTransform(QtGui.QTransform.fromScale(MIN_FIT, MIN_FIT))
            self.centerOn(QtCore.QPointF(box.left() + self.viewport().width() / (2 * MIN_FIT), box.center().y()))

    def zoom(self, factor):
        scale = self.transform().m11() * factor
        if 0.2 <= scale <= 3.0:
            self.scale(factor, factor)

    def drawBackground(self, painter, rect):
        """Dotted paper (the user's choice, 31 October 2026): a dot every GRID, a stronger one every
        GRID * MAJOR. Each dot is placed on a whole screen pixel and drawn at the same size at every
        zoom, so it stays a crisp round dot instead of a smudge; zoomed far out only the strong ones stay."""
        painter.fillRect(rect, QtGui.QColor(color("sheet")))
        scale = self.transform().m11()
        step = GRID if scale * GRID >= 9 else GRID * MAJOR      # the fine dots crowd when zoomed far out
        if scale * step < 9:
            return
        left = int(rect.left()) - int(rect.left()) % step
        top = int(rect.top()) - int(rect.top()) % step
        fine, major = [], []
        for x in range(left, int(rect.right()) + 1, step):
            for y in range(top, int(rect.bottom()) + 1, step):
                point = self.mapFromScene(QtCore.QPointF(x, y))       # view pixels: whole numbers
                strong = x % (GRID * MAJOR) == 0 and y % (GRID * MAJOR) == 0
                (major if strong else fine).append(QtCore.QPointF(point.x() + 0.5, point.y() + 0.5))
        painter.save()
        painter.resetTransform()                    # draw in the view's own pixels
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for points, colour, size in ((fine, color("sheet_line"), 1.8), (major, color("sheet_major"), 2.6)):
            pen = QtGui.QPen(QtGui.QColor(colour), size)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)     # a round dot, not a square speck
            painter.setPen(pen)
            painter.drawPoints(points)
        painter.restore()

    def wheelEvent(self, event):
        """Ctrl + wheel (a touchpad's pinch arrives as that on Windows) zooms; scrolling moves the sheet."""
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self.zoom(1.15 ** steps)
        else:
            # a touchpad gives pixels where it can, a wheel 120 per notch: a notch moves 60 pixels
            delta = event.pixelDelta() if not event.pixelDelta().isNull() else event.angleDelta() / 2
            dx, dy = delta.x(), delta.y()
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier and not dx:
                dx, dy = dy, 0
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(dx))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(dy))
        event.accept()

    def viewportEvent(self, event):
        # pinch on a touchpad that sends gestures (macOS, some Linux drivers)
        if event.type() == QtCore.QEvent.Type.NativeGesture and \
                event.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom(1.0 + event.value())
            return True
        return super().viewportEvent(event)

    # ---------------- mouse: wires and panning ----------------

    def _port_at(self, view_pos):
        for item in self.items_at(view_pos):
            if isinstance(item, PortItem):
                return item
        return None

    def items_at(self, view_pos):
        # a little slack around the point, so a port is easy to hit
        area = QtCore.QRect(view_pos - QtCore.QPoint(5, 5), QtCore.QSize(11, 11))
        return self.items(area)

    def mousePressEvent(self, event):
        left = event.button() == QtCore.Qt.MouseButton.LeftButton
        # empty sheet under a plain left press: move around, as a touchpad's click-and-drag would
        empty = left and not (event.modifiers() & (QtCore.Qt.KeyboardModifier.ShiftModifier |
                                                   QtCore.Qt.KeyboardModifier.ControlModifier)) and \
            not self.items_at(event.position().toPoint())
        if event.button() == QtCore.Qt.MouseButton.MiddleButton or (self._space and left) or empty:
            self._pan = event.position().toPoint()
            self._panned = False
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        port = self._port_at(event.position().toPoint()) if event.button() == QtCore.Qt.MouseButton.LeftButton else None
        if port is not None:
            self.start_wire(port)
            event.accept()
            return
        self.pressing = event.button() == QtCore.Qt.MouseButton.LeftButton
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            delta = event.position().toPoint() - self._pan
            self._pan = event.position().toPoint()
            self._panned = self._panned or not delta.isNull()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._drag is not None:
            self.drag_wire_to(self.mapToScene(event.position().toPoint()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
            if not self._panned and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.scene().clearSelection()          # a click on empty sheet still selects nothing
            if self._space:
                self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            return
        if self._drag is not None:
            self.finish_wire(self._port_at(event.position().toPoint()))
            return
        super().mouseReleaseEvent(event)
        self.pressing = False
        if self.dragged:
            self.dragged = False
            self.moved.emit()
        self.released.emit()

    def start_wire(self, port):
        temporary = QtWidgets.QGraphicsPathItem()
        temporary.setPen(QtGui.QPen(QtGui.QColor(TYPE_COLOURS[port.type]), 2, QtCore.Qt.PenStyle.DashLine))
        temporary.setZValue(2)
        self.scene().addItem(temporary)
        self._drag = (port, temporary)
        for item in self.cards.values():
            for other in item.inputs + item.outputs:
                other.light(other.accepts(port))
        self.drag_wire_to(port.scenePos())

    def drag_wire_to(self, point):
        port, temporary = self._drag
        a, b = (port.scenePos(), point) if port.output else (point, port.scenePos())
        temporary.setPath(curve(a, b))

    def finish_wire(self, other):
        port, temporary = self._drag
        self._drag = None
        self.scene().removeItem(temporary)
        for item in self.cards.values():
            for each in item.inputs + item.outputs:
                each.light(False)
        if other is None or other is port:
            return False
        if other.output == port.output:
            self.message.emit("Join an output (right side) to an input (left side)")
            return False
        out, inp = (port, other) if port.output else (other, port)
        return self.wire(out.node_id, out.port.name, inp.node_id, inp.port.name)

    # ---------------- keys, menu, dropping blocks ----------------

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.delete_selected()
        elif event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        elif event.key() == QtCore.Qt.Key.Key_F and not event.modifiers():
            self.fit()
        elif event.key() in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self.zoom(1.15)
        elif event.key() == QtCore.Qt.Key.Key_Minus:
            self.zoom(1 / 1.15)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.unsetCursor()
        else:
            super().keyReleaseEvent(event)

    def contextMenuEvent(self, event):
        node = next((i for i in self.items_at(event.pos()) if isinstance(i, NodeItem)), None)
        menu = QtWidgets.QMenu(self)
        if node is not None:
            self.select(node.node_id)
            menu.addAction("Run up to here", lambda: self.run_to.emit(node.node_id))
            menu.addAction("Duplicate", lambda: self.duplicate(node.node_id))
            menu.addAction("Help", lambda: self.help_for.emit(self.graph.nodes[node.node_id].type))
            menu.addSeparator()
            menu.addAction("Delete", self.delete_selected)
        else:
            menu.addAction("Fit to window   F", self.fit)
        menu.exec(event.globalPos())

    def duplicate(self, node_id):
        node = self.graph.nodes[node_id]
        copy = self.graph.add(node.type, dict(node.params),
                              position=(node.position[0] + 30, node.position[1] + 30), label=node.label)
        self._add_item(copy)
        self.select(copy)
        self.changed.emit()
        return copy

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(BLOCK_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(BLOCK_MIME):
            return super().dropEvent(event)
        type_name = bytes(event.mimeData().data(BLOCK_MIME)).decode("utf-8")
        point = self.mapToScene(event.position().toPoint()) - QtCore.QPointF(WIDTH / 2, 20)
        self.add_block(type_name, point)
        event.acceptProposedAction()

    # ---------------- run results ----------------

    def show_states(self, states):
        """{node id: (state, result line, message)}; nodes not named are cleared."""
        for node_id, item in self.cards.items():
            item.set_state(*states.get(node_id, ("", "", "")))


class BlockPreview(QtWidgets.QWidget):
    """A block as it will look on the sheet, drawn under the Blocks list while the mouse is on one of its
    names: its category band, title, main setting and ports. Nothing is added to the sheet; a drag or a double
    click does that."""

    def __init__(self):
        super().__init__()
        self.kind = None
        self.setFixedHeight(HEADER + 22 + 4 * ROW + 16)
        self.setAccessibleName("Block preview")

    def set_block(self, type_name):
        kind = block(type_name) if type_name else None
        if kind is not self.kind:
            self.kind = kind
            self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.kind is None:
            painter.setPen(QtGui.QColor(color("dim")))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Point at a block to see it")
            return
        kind = self.kind
        rows = max(len(kind.inputs), len(kind.outputs), 1)
        height = HEADER + 22 + rows * ROW
        scale = min(1.0, (self.width() - 2 * PORT - 8) / WIDTH, (self.height() - 8) / height)
        painter.translate((self.width() - WIDTH * scale) / 2, (self.height() - height * scale) / 2)
        painter.scale(scale, scale)
        card = QtCore.QRectF(0, 0, WIDTH, height)
        painter.setPen(QtGui.QPen(QtGui.QColor(color("block_border")), 1.2))
        painter.setBrush(QtGui.QColor(color("block")))
        painter.drawRoundedRect(card, RADIUS, RADIUS)
        outline = QtGui.QPainterPath()
        outline.addRoundedRect(card, RADIUS, RADIUS)
        stripe = QtGui.QPainterPath()
        stripe.addRect(QtCore.QRectF(0, 0, WIDTH, STRIPE))
        painter.fillPath(outline.intersected(stripe), QtGui.QColor(CATEGORY_COLOURS.get(kind.category, color("dim"))))
        title = QtGui.QFont(self.font())
        title.setBold(True)
        painter.setFont(title)
        painter.setPen(QtGui.QColor(color("text")))
        painter.drawText(QtCore.QRectF(10, 7, WIDTH - 20, 18), QtCore.Qt.AlignmentFlag.AlignVCenter,
                         QtGui.QFontMetrics(title).elidedText(kind.title, QtCore.Qt.TextElideMode.ElideRight,
                                                              WIDTH - 20))
        small = QtGui.QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1))
        painter.setFont(small)
        metrics = QtGui.QFontMetrics(small)
        painter.setPen(QtGui.QColor(color("subtle")))
        painter.drawText(QtCore.QRectF(10, HEADER, WIDTH - 20, 18), QtCore.Qt.AlignmentFlag.AlignVCenter,
                         metrics.elidedText(kind.short(kind.defaults()), QtCore.Qt.TextElideMode.ElideRight,
                                            WIDTH - 20))
        half = WIDTH / 2 - 14
        for output, ports in ((False, kind.inputs), (True, kind.outputs)):
            for i, port in enumerate(ports):
                y = HEADER + 22 + ROW * i + ROW / 2
                painter.setPen(QtGui.QPen(QtGui.QColor(color("block")), 1.5))
                painter.setBrush(QtGui.QColor(TYPE_COLOURS[port.types[0]]))
                painter.drawEllipse(QtCore.QPointF(WIDTH if output else 0, y), PORT, PORT)
                painter.setPen(QtGui.QColor(color("header_text")))
                where = (QtCore.QRectF(WIDTH / 2 + 2, y - 9, half, 18) if output else QtCore.QRectF(12, y - 9, half, 18))
                align = QtCore.Qt.AlignmentFlag.AlignVCenter | (QtCore.Qt.AlignmentFlag.AlignRight if output
                                                               else QtCore.Qt.AlignmentFlag.AlignLeft)
                painter.drawText(where, align, metrics.elidedText(port.label or port.name,
                                                                  QtCore.Qt.TextElideMode.ElideRight, int(half)))
