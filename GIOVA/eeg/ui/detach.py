"""Views that can leave the main window, and side panels that can be widened.

A tab can be dragged out of the tab bar into its own window - onto a second
monitor, say - as in a web browser. Dragging that window's tab back onto the
tab bar, or closing the window, puts it back. Where each popped-out window
was, and whether it was out at all, is remembered, so the lab screen
arrangement survives a restart.
"""
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .theme import color, muted

TAB_HINT = "Drag out to open in its own window, sideways to reorder"


#: where a dragged view can be dropped
TAB_STRIP, LEFT, RIGHT = 'tabs', 'left', 'right'
#: how much of the view area, at each side, docks a dropped view beside the
#: other one. The middle stays free so that a view dragged out can simply be
#: let go over the window it came from, as a browser tab can.
EDGE = 0.25


def pop_out_button(parent):
    """A small 'open in its own window' button with the platform's window icon."""
    button = QtWidgets.QToolButton(parent)
    button.setIcon(parent.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton))
    button.setAutoRaise(True)
    button.setToolTip("Open this view in its own window (or drag its tab out). "
                      "Drag the window's tab back, or close the window, to put it back.")
    return button


class _WindowDrag(QtCore.QObject):
    """Moves a popped-out window with the mouse until the button is released.

    Mouse events are taken from the whole application: the widget where the
    drag began may be removed or hidden meanwhile (the tab itself, or the wide
    Monitor layout) and would stop receiving them.
    """

    def __init__(self, tabs, window, grip, armed=True):
        super().__init__(window)
        self.tabs, self.window, self.grip = tabs, window, grip
        self.active = True
        #: a view just torn off near a side must not snap straight back: it
        #: docks only after it has been out of the side zones once
        self.armed = armed
        #: the window's own geometry while it is snapped into a side, else None
        self.free_geometry = None
        QtWidgets.QApplication.instance().installEventFilter(self)

    def move(self, point):
        zone = self.tabs.drop_zone(point)
        if zone not in (LEFT, RIGHT):
            self.armed = True
        if zone == TAB_STRIP or (zone in (LEFT, RIGHT) and self.armed):
            # The view itself shows where it would go: the window takes the
            # place it would dock into. No coloured box or bar over the screen.
            self.snap(self.tabs.zone_rect(zone))
            return
        self.unsnap()
        # move() places the frame; keep the cursor on the window's own tab.
        frame = self.window.geometry().topLeft() - self.window.frameGeometry().topLeft()
        self.window.move(point - self.grip - frame)

    def snap(self, target):
        """Fit the window's frame into `target` (global), remembering its own size."""
        window = self.window
        if self.free_geometry is None:
            self.free_geometry = window.geometry()
        inner, outer = window.geometry(), window.frameGeometry()
        client = target.adjusted(inner.left() - outer.left(), inner.top() - outer.top(),
                                 inner.right() - outer.right(), inner.bottom() - outer.bottom())
        if window.geometry() != client:
            window.setGeometry(client)
        window.raise_()

    def unsnap(self):
        """Back to the window's own size, before it follows the cursor again."""
        if self.free_geometry is not None:
            self.window.resize(self.free_geometry.size())
            self.free_geometry = None

    def finish(self, point):
        if not self.active:
            return
        self.active = False
        QtWidgets.QApplication.instance().removeEventFilter(self)
        zone = self.tabs.drop_zone(point)
        if zone in (LEFT, RIGHT) and not self.armed:
            zone = None                     # it never showed where it would dock
        if self.free_geometry is not None:
            # Docking saves the window's geometry for its next pop-out: that
            # should be its own, not the half of the screen it was snapped to.
            self.window.setGeometry(self.free_geometry)
            self.free_geometry = None
        if zone == TAB_STRIP:
            self.tabs.dock(self.window.name, point)
        elif zone in (LEFT, RIGHT):
            self.tabs.dock_beside(self.window.name, zone)
        else:
            self.window.raise_()
            self.window.activateWindow()

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QtCore.QEvent.Type.MouseMove:
            point = event.globalPosition().toPoint()
            if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
                self.move(point)
            else:                           # the release was never delivered
                self.finish(point)
            return True
        if kind == QtCore.QEvent.Type.MouseButtonRelease:
            self.finish(event.globalPosition().toPoint())
            return True
        return False


class DragTab(QtWidgets.QLabel):
    """A tab-like label that starts a window drag once the mouse moves far enough."""

    def __init__(self, name, start_drag, parent=None):
        super().__init__(name, parent)
        self.start_drag = start_drag     # (global point, global press point) -> None
        self._press = None
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = event.globalPosition().toPoint()
        if (self._press is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton
                and (point - self._press).manhattanLength() >= QtWidgets.QApplication.startDragDistance()):
            press, self._press = self._press, None
            self.start_drag(point, press)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press = None
        super().mouseReleaseEvent(event)


class TearOffTabBar(QtWidgets.QTabBar):
    """Tabs that reorder when dragged sideways and tear off when dragged away."""

    torn = QtCore.Signal(int, QtCore.QPoint)     # tab index, global cursor position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)
        self._pressed = None

    def mousePressEvent(self, event):
        # By name, not index: a sideways drag moves the tab to another index
        # before it is pulled out of the bar.
        index = self.tabAt(event.position().toPoint())
        left = event.button() == QtCore.Qt.MouseButton.LeftButton
        self._pressed = self.tabText(index) if left and index >= 0 else None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        y = event.position().y()
        # One tab height of dead zone meant dragging without anything happening.
        # The bar still reorders while the cursor is on it; leaving it tears the
        # tab off, as a browser does. The last tab may leave too: an empty tab
        # area says so and takes it back.
        away = QtWidgets.QApplication.startDragDistance()
        outside = y < -away or y > self.height() + away
        if (self._pressed is not None and outside
                and event.buttons() & QtCore.Qt.MouseButton.LeftButton):
            name, self._pressed = self._pressed, None
            # End the bar's own sliding drag before the tab leaves it.
            super().mouseReleaseEvent(QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonRelease, event.position(), event.globalPosition(),
                QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton, event.modifiers()))
            index = next((i for i in range(self.count()) if self.tabText(i) == name), -1)
            if index >= 0:
                self.torn.emit(index, event.globalPosition().toPoint())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._pressed = None
        super().mouseReleaseEvent(event)


class ViewWindow(QtWidgets.QWidget):
    """A top-level window holding one popped-out tab page."""

    closing = QtCore.Signal(object)

    def __init__(self, title, page, footer=None):
        super().__init__(None, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("GIOVA · " + title)
        self.name = title
        #: set by DetachableTabs: (global point, press point) -> drag the window by its tab
        self.drag_handler = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.tab = DragTab(title, self._drag_self)
        self.tab.setObjectName("viewWindowTab")
        self.tab.setToolTip("Drag onto the main window's tabs to put this view back")
        self.tab.setStyleSheet("#viewWindowTab { border: 1px solid " + color("field") + "; border-bottom: none; "
                               "border-top-left-radius: 5px; border-top-right-radius: 5px; "
                               "padding: 3px 12px; background: palette(base); }")
        header.addWidget(self.tab)
        header.addStretch()
        layout.addLayout(header)
        layout.addWidget(page, 1)
        self.footer = QtWidgets.QLabel()
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(muted())
        self.footer.setVisible(False)
        layout.addWidget(self.footer)
        app = QtWidgets.QApplication.instance()
        if app is not None and not app.windowIcon().isNull():
            self.setWindowIcon(app.windowIcon())

    def _drag_self(self, point, press):
        if self.drag_handler is not None:
            self.drag_handler(point, press)

    def tab_grip(self):
        """Where the cursor holds the window while it is dragged by its tab."""
        return self.tab.mapTo(self, QtCore.QPoint(min(40, self.tab.width() // 2), self.tab.height() // 2))

    def set_footer(self, text):
        self.footer.setText(text)
        self.footer.setVisible(bool(text))

    def closeEvent(self, event):
        self.closing.emit(self)
        event.accept()


class DetachableTabs(QtWidgets.QTabWidget):
    """A tab widget whose pages can be popped out and put back.

    Drag a tab out of the bar (or use the ⧉ corner button, a double click or
    the context menu) to pop it out; drag sideways to reorder. Dropping the
    window's tab on the bar puts the page back where it lands; closing the
    window puts it back at its place in the tab order.
    """

    detached = QtCore.Signal(str, object)
    reattached = QtCore.Signal(str, object)
    #: a view was dropped: name, and one of TAB_STRIP / LEFT / RIGHT
    docked = QtCore.Signal(str, str)

    def __init__(self, key, preferences):
        super().__init__()
        self.setTabBar(TearOffTabBar(self))
        self.key, self.preferences = key, preferences
        self._order = []          # every page name, in its home order
        self._pages = {}          # name -> widget
        self._windows = {}        # name -> ViewWindow
        self._restoring = False
        self._drag = None
        self._area = None         # what replaces these tabs while they are hidden
        button = pop_out_button(self)
        button.clicked.connect(lambda: self.pop_out(self.tabText(self.currentIndex())))
        self.setCornerWidget(button, QtCore.Qt.Corner.TopRightCorner)
        bar = self.tabBar()
        bar.torn.connect(lambda index, point: self.begin_drag(self.tabText(index), point))
        bar.tabMoved.connect(lambda *_: self._sync_order())
        bar.tabBarDoubleClicked.connect(lambda index: self.pop_out(self.tabText(index)) if index >= 0 else None)
        bar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._menu)

    # ---------------- pages ----------------

    def addTab(self, widget, name):
        self._order.append(name)
        self._pages[name] = widget
        index = super().addTab(widget, name)
        self.setTabToolTip(index, TAB_HINT)
        return index

    def page(self, name):
        return self._pages[name]

    def is_detached(self, name):
        return name in self._windows

    def window_for(self, name):
        return self._windows.get(name)

    def visible_names(self):
        """The selected tab plus every popped-out view."""
        names = list(self._windows)
        if self.count():
            current = self.tabText(self.currentIndex())
            if current not in names:
                names.insert(0, current)
        return names

    def _menu(self, point):
        index = self.tabBar().tabAt(point)
        if index < 0:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Open in a separate window", lambda: self.pop_out(self.tabText(index)))
        menu.exec(self.tabBar().mapToGlobal(point))

    # ---------------- detach / attach ----------------

    def pop_out(self, name):
        if name not in self._pages or name in self._windows:
            return None
        page = self._pages[name]
        # The last view may leave too: the empty tab area says where its views
        # are and takes them back, which is less surprising than a tab that
        # refuses to move.
        if self.indexOf(page) >= 0:
            self.removeTab(self.indexOf(page))
        window = ViewWindow(name, page)
        window.closing.connect(lambda w: self.put_back(w.name))
        window.drag_handler = lambda point, press, n=name: self.begin_drag(n, point, press)
        self._windows[name] = window
        self._place(window, name)
        page.show()
        window.show()
        self._remember()
        self.detached.emit(name, page)
        return window

    def put_back(self, name):
        window = self._windows.pop(name, None)
        if window is None:
            return
        self.preferences.setValue(self._geometry_key(name), window.saveGeometry())
        page = self.insert_home(name)
        window.hide()
        window.deleteLater()
        self._remember()
        self.reattached.emit(name, page)

    def insert_home(self, name):
        """Put a page back into the tab bar at its original position."""
        page = self._pages[name]
        home = self._order.index(name)
        position = sum(1 for other in self._order[:home] if self.indexOf(self._pages[other]) >= 0)
        index = self.insertTab(position, page, name)
        self.setTabToolTip(index, TAB_HINT)
        page.show()
        return page

    def reattach_all(self, remember=True):
        """Put every page back (on application exit). `remember` keeps the list
        of popped-out views so they reopen next time."""
        names = list(self._windows)
        for name in names:
            window = self._windows[name]
            window.closing.disconnect()
            self.preferences.setValue(self._geometry_key(name), window.saveGeometry())
            window.close()
            self.put_back(name)
        if remember and names:
            self.preferences.setValue(self.key + "/detached", ";".join(names))

    def restore(self):
        """Reopen the views that were popped out when the app last closed."""
        saved = str(self.preferences.value(self.key + "/detached", "") or "")
        for name in [n for n in saved.split(";") if n]:
            self.pop_out(name)

    def set_footer(self, name, text):
        window = self._windows.get(name)
        if window is not None:
            window.set_footer(text)

    # ---------------- dragging ----------------

    def begin_drag(self, name, point, press=None):
        """Move `name`'s window with the mouse, popping the view out first if needed.

        A window already out keeps the spot where it was grabbed (`press`);
        a freshly torn-off one comes up held by its own tab.
        """
        if self._drag is not None and self._drag.active:
            return None
        window = self._windows.get(name)
        fresh = window is None
        if window is not None and press is not None:
            grip = window.mapFromGlobal(press)
        else:
            window = window or self.pop_out(name)
            if window is None:
                return None
            grip = window.tab_grip()
        self._drag = _WindowDrag(self, window, grip, armed=not fresh)
        self._drag.move(point)
        return self._drag

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count():
            return
        # Every view is in a window of its own: an empty grey box would look
        # like a fault, and this is also where they can be dropped back.
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(color("dim")))
        painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                         "Every view is in its own window."
                         " Close one, or drop it here, to bring it back.")
        painter.end()

    def set_drop_area(self, widget):
        """What stands in for these tabs while the page hides them.

        The Monitor swaps the tabs for a side-by-side layout on a wide window.
        Without this, a view dragged out there has nowhere to be dropped: the
        tab widget is hidden, so every drop landed on nothing.
        """
        self._area = widget

    def drop_area(self):
        if self._area is not None and self._area.isVisible():
            return self._area
        return self

    def drop_zone(self, point):
        """Where this global point would put a dropped view, or None."""
        area = self.drop_area()
        if not area.isVisible() or self.window().isMinimized():
            return None
        rect = QtCore.QRect(area.mapToGlobal(QtCore.QPoint(0, 0)), area.size())
        if not rect.contains(point):
            return None
        if area is self:
            strip = QtCore.QRect(rect.topLeft(), QtCore.QSize(rect.width(), self.tabBar().height() + 12))
            if strip.contains(point):
                return TAB_STRIP
        edge = max(40, int(rect.width() * EDGE))
        if point.x() <= rect.left() + edge:
            return LEFT
        if point.x() >= rect.right() - edge:
            return RIGHT
        return None

    def zone_rect(self, zone):
        """The area a drop in `zone` would fill, in global coordinates.

        Back on the tabs, a view fills the page under the bar (the bar stays in
        sight: that is where the cursor is); beside, it takes half.
        """
        area = self.drop_area()
        rect = QtCore.QRect(area.mapToGlobal(QtCore.QPoint(0, 0)), area.size())
        if zone == TAB_STRIP:
            return rect.adjusted(0, self.tabBar().height(), 0, 0)
        half = QtCore.QSize(max(1, rect.width() // 2), rect.height())
        return QtCore.QRect(rect.topLeft() if zone == LEFT else QtCore.QPoint(rect.center().x() + 1, rect.top()), half)

    def dock_beside(self, name, side):
        """Put a popped-out view back, next to the other one rather than as a tab."""
        if name not in self._windows:
            return
        self.put_back(name)
        page = self._pages[name]
        if self.indexOf(page) >= 0:
            self.setCurrentWidget(page)
        self.docked.emit(name, side)

    def dock(self, name, point):
        """Put a popped-out view back into the bar where it was dropped."""
        if name not in self._windows:
            return
        bar = self.tabBar()
        x = bar.mapFromGlobal(point).x()
        index = next((i for i in range(self.count()) if x < bar.tabRect(i).center().x()), self.count())
        attached = [self.tabText(i) for i in range(self.count())]
        self._order.remove(name)
        if index < len(attached):
            self._order.insert(self._order.index(attached[index]), name)
        elif attached:
            self._order.insert(self._order.index(attached[-1]) + 1, name)
        else:
            self._order.append(name)
        self.put_back(name)
        page = self._pages[name]
        if self.indexOf(page) >= 0:
            self.setCurrentWidget(page)
        self.docked.emit(name, TAB_STRIP)

    def _sync_order(self):
        """After a sideways drag the bar's order becomes the home order;
        popped-out views keep their slots."""
        attached = [self.tabText(i) for i in range(self.count())]
        slots = [i for i, name in enumerate(self._order) if name in attached]
        if len(slots) == len(attached):
            for slot, name in zip(slots, attached):
                self._order[slot] = name

    def _geometry_key(self, name):
        return "{}/window/{}".format(self.key, name)

    def _remember(self):
        self.preferences.setValue(self.key + "/detached", ";".join(self._windows))

    def _place(self, window, name):
        geometry = self.preferences.value(self._geometry_key(name), None)
        if geometry is not None and window.restoreGeometry(geometry):
            screens = [screen.availableGeometry() for screen in QtGui.QGuiApplication.screens()]
            if any(area.intersects(window.frameGeometry()) for area in screens):
                return
        # First time, or the monitor it was on is gone: next to the main window.
        size = self.size()
        window.resize(max(640, size.width()), max(420, size.height()))
        anchor = self.mapToGlobal(QtCore.QPoint(40, 40))
        window.move(anchor)


class ResizableSide(QtWidgets.QSplitter):
    """A horizontal splitter: side panels you can drag wider or narrower, and a
    main area that takes the rest. Each side panel's width is remembered."""

    def __init__(self, key, preferences):
        super().__init__(QtCore.Qt.Orientation.Horizontal)
        self.key, self.preferences = key, preferences
        self._sides = {}          # widget -> (name, width)
        self.setChildrenCollapsible(False)
        self.setHandleWidth(6)
        self.splitterMoved.connect(lambda *_: self.save())

    def add_side(self, widget, name, width, minimum=260, maximum=700):
        widget.setMinimumWidth(minimum)
        widget.setMaximumWidth(maximum)
        saved = self.preferences.value("{}/{}".format(self.key, name), None)
        try:
            width = int(saved) if saved is not None else width
        except (TypeError, ValueError):
            pass
        self._sides[widget] = [name, max(minimum, min(maximum, width))]
        self.addWidget(widget)
        self.setStretchFactor(self.count() - 1, 0)
        self._lock_fixed_handles()

    def add_main(self, widget):
        self.addWidget(widget)
        self.setStretchFactor(self.count() - 1, 1)
        self._lock_fixed_handles()

    def _lock_fixed_handles(self):
        """A side whose width cannot change gets a handle that does not offer to.

        The handle would still show the resize cursor, promising a drag that
        does nothing. It is the one between the side and the main area.
        """
        main = next((i for i in range(self.count()) if self.widget(i) not in self._sides), None)
        if main is None:
            return
        for i in range(self.count()):
            widget = self.widget(i)
            if widget in self._sides and widget.minimumWidth() == widget.maximumWidth():
                handle = self.handle(i if i > main else i + 1)
                handle.setEnabled(False)
                handle.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def _width(self, widget):
        """A side's width: what was dragged, or its strip while it is folded away."""
        folded = getattr(widget, "collapsed", False)
        return widget.maximumWidth() if folded else self._sides[widget][1]

    def apply_sizes(self):
        total = max(self.width(), 1)
        fixed = sum(self._width(widget) for widget in self._sides if widget.isVisibleTo(self))
        self.setSizes([self._width(self.widget(i)) if self.widget(i) in self._sides else max(total - fixed, 100)
                       for i in range(self.count())])

    def save(self):
        for widget, entry in self._sides.items():
            # A folded side is a strip: its open width is what is remembered.
            if widget.isVisible() and widget.width() > 0 and not getattr(widget, "collapsed", False):
                entry[1] = widget.width()
                self.preferences.setValue("{}/{}".format(self.key, entry[0]), int(widget.width()))

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_sized", False):
            self._sized = True
            self.apply_sizes()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not getattr(self, "_sized", False) and self.width() > 0:
            self.apply_sizes()
