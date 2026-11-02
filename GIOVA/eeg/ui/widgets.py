"""Small Qt helpers shared by the control panels.

The scroll-proof variants exist because a control panel is a tall scrolling
column: rolling the wheel to reach a widget would otherwise change the value of
whatever it passed over, silently re-running the analysis with settings nobody
chose.
"""
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class NoScrollSpinBox(QtWidgets.QDoubleSpinBox):
    """A spin box that ignores the wheel. Type, or use the arrows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


class NoScrollComboBox(QtWidgets.QComboBox):
    """A combo box that ignores the wheel. Click to open it instead."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


class NoScrollIntBox(QtWidgets.QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


class NoScrollSlider(QtWidgets.QSlider):
    """A slider that ignores the wheel. Drag or click it instead."""

    def wheelEvent(self, event):
        event.ignore()


#: Controls whose value the wheel would change. Scroll bars are sliders too, and
#: are left alone: they are how a panel scrolls.
_WHEEL_BLOCKED = (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox,
                  QtWidgets.QAbstractSlider, QtWidgets.QTabBar)


class WheelGuard(QtCore.QObject):
    """App-wide: the wheel never changes a value box, list, slider or page tab.

    The same rule as the NoScroll widgets, for every control including ones
    built elsewhere (pyqtgraph's menus). The wheel event is ignored, so Qt hands
    it to the parent and the surrounding panel scrolls instead. A graph whose
    zoom is off (the default, plots.set_zoom) passes the wheel on the same way:
    it used to swallow it, so a settings column stopped scrolling wherever the
    mouse passed over a graph. A graph with zoom on still zooms.
    """

    def eventFilter(self, obj, event):
        if event.type() != QtCore.QEvent.Type.Wheel:
            return False
        if isinstance(obj, _WHEEL_BLOCKED) and not isinstance(obj, QtWidgets.QScrollBar):
            event.ignore()
            return True
        if _still_graph(obj):
            # handed straight to the panel around it: left to propagate, it sometimes died in pyqtgraph's view
            area = obj.parent()
            while area is not None and not (isinstance(area, QtWidgets.QAbstractScrollArea)
                                            and not isinstance(area, QtWidgets.QGraphicsView)):
                area = area.parent()
            if area is not None:
                QtWidgets.QApplication.sendEvent(area.verticalScrollBar(), event)
            return True
        return False


def _still_graph(widget):
    """A pyqtgraph view (or its drawing area) none of whose plots takes the mouse: nothing for the wheel there."""
    view = widget if isinstance(widget, QtWidgets.QGraphicsView) else widget.parent()
    if not isinstance(view, QtWidgets.QGraphicsView) or view.scene() is None:
        return False
    import pyqtgraph as pg
    if not isinstance(view, pg.GraphicsView):
        return False
    boxes = [item for item in view.scene().items() if isinstance(item, pg.ViewBox)]
    return bool(boxes) and not any(any(box.state["mouseEnabled"]) for box in boxes)


def install_wheel_guard(app):
    """Install one WheelGuard on `app` (kept on it so it is not collected)."""
    if getattr(app, "_wheel_guard", None) is None:
        app._wheel_guard = WheelGuard(app)
        app.installEventFilter(app._wheel_guard)
    return app._wheel_guard


class TooltipSwitch(QtCore.QObject):
    """Hover explanations on or off for the whole program.

    One filter on the application swallows the ToolTip event, so no widget's
    tooltip text has to be touched and nothing is lost when they come back.
    """

    def __init__(self, parent=None, enabled=True):
        super().__init__(parent)
        self.enabled = bool(enabled)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not self.enabled:
            QtWidgets.QToolTip.hideText()

    def eventFilter(self, obj, event):
        # an icon alone keeps its name on hover (icons.only): that is its label, not an explanation
        return (not self.enabled and event.type() == QtCore.QEvent.Type.ToolTip
                and not (isinstance(obj, QtWidgets.QWidget) and obj.property("icon_only")))


def install_tooltip_switch(app, enabled=True):
    """Install one TooltipSwitch on `app` (kept on it so it is not collected)."""
    if getattr(app, "_tooltip_switch", None) is None:
        app._tooltip_switch = TooltipSwitch(app, enabled)
        app.installEventFilter(app._tooltip_switch)
    else:
        app._tooltip_switch.set_enabled(enabled)
    return app._tooltip_switch


def fit_to_screen(widget, width, height, share=0.92):
    """Resize `widget` to width x height, or less where its screen is smaller.

    Sizes are logical pixels: a 1920x1080 screen at 150 % Windows scaling has
    room for 1280x720 of them, a 1366x768 laptop for less than 1366x768. A
    fixed size made for the lab monitor then ran off the edge. Returns True
    when the screen was too small for the size asked for.
    """
    screen = widget.screen() or QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        widget.resize(width, height)
        return False
    room = screen.availableGeometry()
    fitted = QtCore.QSize(min(width, int(room.width() * share)), min(height, int(room.height() * share)))
    widget.resize(fitted)
    if widget.isWindow():
        frame = widget.frameGeometry()
        frame.moveCenter(room.center())
        widget.move(frame.topLeft())
    return fitted != QtCore.QSize(width, height)


def spin_box(low, high, value, step=1.0, decimals=0, suffix=""):
    """A scroll-proof QDoubleSpinBox configured in one line."""
    box = NoScrollSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    if suffix:
        box.setSuffix(suffix)
    return box


def combo_box(items=(), current=None):
    """A scroll-proof QComboBox, optionally pre-filled."""
    box = NoScrollComboBox()
    if items:
        box.addItems(list(items))
    if current is not None:
        box.setCurrentText(current)
    return box


def check_box(label, checked=False, tooltip=""):
    """A checkbox in one line.

    Here so that a setting which means the same thing on both pages is also
    spelled the same way on both pages - the log scale used to be a dropdown on
    the monitor and a tick box on the analysis page.
    """
    box = QtWidgets.QCheckBox(label)
    box.setChecked(bool(checked))
    if tooltip:
        box.setToolTip(tooltip)
    return box


class ElidedLabel(QtWidgets.QLabel):
    """One line of plain text that ends in … when the room is short, the whole text in its tooltip.

    A plain QLabel asks for its whole text's width, so a long message (an error, a run's progress)
    makes the window wider than the screen; this one asks for a few characters and shows what fits.
    """

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text):
        self._full = str(text or "")
        self.setToolTip(self._full)
        self._fit()

    def text(self):
        return self._full

    def minimumSizeHint(self):
        return QtCore.QSize(self.fontMetrics().horizontalAdvance("M" * 4), super().minimumSizeHint().height())

    def sizeHint(self):
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        shown = self.fontMetrics().elidedText(self._full, QtCore.Qt.TextElideMode.ElideRight, max(self.width(), 1))
        QtWidgets.QLabel.setText(self, shown)


class WrapRow(QtWidgets.QLayout):
    """A row of widgets that goes on to a second line when the window is too narrow for one.

    Each widget keeps its own width; one widget may be `stretch`: it fills what is left of its line
    (and can shrink to its minimum), pushing the widgets after it to the line's end, as a stretch in a
    plain row would. Wide, it looks like a QHBoxLayout; narrow, nothing is pushed off the screen.
    """

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._stretch = None
        self.setSpacing(spacing)

    def add(self, widget, stretch=False):
        self.addWidget(widget)
        if stretch:
            self._stretch = widget

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def sizeHint(self):
        margins = self.contentsMargins()
        visible = [i for i in self._items if not i.isEmpty()]
        width = sum(i.sizeHint().width() for i in visible) + self.spacing() * max(len(visible) - 1, 0)
        height = max((i.sizeHint().height() for i in visible), default=0)
        return QtCore.QSize(width + margins.left() + margins.right(), height + margins.top() + margins.bottom())

    def minimumSize(self):
        margins = self.contentsMargins()
        visible = [i for i in self._items if not i.isEmpty()]
        width = max((self._width(i, minimum=True) for i in visible), default=0)
        height = max((i.minimumSize().height() for i in visible), default=0)
        return QtCore.QSize(width + margins.left() + margins.right(), height + margins.top() + margins.bottom())

    def _width(self, item, minimum=False):
        if item.widget() is self._stretch:
            return item.widget().minimumSizeHint().width()
        hint = item.sizeHint().width() if not minimum else max(item.minimumSize().width(),
                                                               item.widget().minimumSizeHint().width()
                                                               if item.widget() else 0)
        return hint

    def _place(self, rect, apply):
        """Lay the items out in lines inside `rect`; the height they take."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        lines, line, used = [], [], 0
        for item in (i for i in self._items if not i.isEmpty()):
            width = min(self._width(item), area.width())
            if line and used + self.spacing() + width > area.width():
                lines.append(line)
                line, used = [], 0
            used += (self.spacing() if line else 0) + width
            line.append((item, width))
        if line:
            lines.append(line)
        y = area.y()
        for line in lines:
            height = max(item.sizeHint().height() for item, _ in line)
            spare = area.width() - sum(w for _, w in line) - self.spacing() * (len(line) - 1)
            x = area.x()
            for item, width in line:
                if item.widget() is self._stretch:
                    width += max(spare, 0)
                if apply:
                    item.setGeometry(QtCore.QRect(x, y, width, height))
                x += width + self.spacing()
            y += height + self.spacing()
        return y - self.spacing() - area.y() + margins.top() + margins.bottom() if lines else 0


def group(*widgets, spacing=4):
    """Widgets side by side in one widget, so a WrapRow keeps them together on one line."""
    box = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        row.addWidget(widget)
    return box


class BusyTrace(QtWidgets.QWidget):
    """A short EEG-like trace that runs while the program computes, its colour slowly turning;
    hidden when nothing is being computed. `set_busy` follows `jobs.activity().changed`.

    The trace is a sum of alpha, theta and beta rhythms with a slow drift, scrolled one step per
    frame, so it reads as a live recording rather than a spinner.
    """

    FRAME_MS = 33
    #: (Hz on the trace's own scale, amplitude): what gives it an EEG look
    RHYTHMS = ((0.9, 0.55), (0.45, 0.3), (2.3, 0.18), (0.13, 0.25))

    def __init__(self, parent=None, width=132, height=22):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setToolTip("Computing…")
        self.phase = 0.0
        self.hue = 0.55
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.FRAME_MS)
        self.timer.timeout.connect(self._step)
        self.hide()

    def set_busy(self, busy):
        if busy:
            self.show()
            self.timer.start()
        else:
            self.timer.stop()
            self.hide()

    def _step(self):
        self.phase += 0.35
        self.hue = (self.hue + 0.004) % 1.0
        self.update()

    def points(self):
        """The trace now: one point per pixel, scaled to the widget's height."""
        import math
        w, h = self.width(), self.height()
        out = []
        for x in range(w):
            t = (x + self.phase * 12) / 12.0
            value = sum(a * math.sin(f * t * 2 * math.pi / 3 + k) for k, (f, a) in enumerate(self.RHYTHMS))
            out.append(QtCore.QPointF(x, h / 2 - value * h * 0.3))
        return out

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        colour = QtGui.QColor.fromHsvF(self.hue, 0.65, 0.8)
        # the newest samples on the right, older ones fading out to the left, as on the Monitor
        gradient = QtGui.QLinearGradient(0, 0, self.width(), 0)
        faded = QtGui.QColor(colour)
        faded.setAlpha(0)
        gradient.setColorAt(0.0, faded)
        gradient.setColorAt(0.35, colour)
        gradient.setColorAt(1.0, colour)
        painter.setPen(QtGui.QPen(QtGui.QBrush(gradient), 1.6))
        painter.drawPolyline(self.points())
        painter.end()
