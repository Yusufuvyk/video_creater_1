"""The icon rail down the window's left edge, and the one side panel beside it.

GIOVA's layout: the modes at the top of the rail (Home,
Live monitor, Workbench, Games when it is on), the side panels below a line (Projects in every mode,
Blocks and Library on the Workbench), Help and Settings at the bottom. Icons alone: the name and the
shortcut are in the tooltip, the one in use is drawn in the accent colour.

One panel is open at a time, beside the rail. A click on the open panel's icon folds it away; which
panel each mode had open is remembered.
"""
from pyqtgraph.Qt import QtCore, QtWidgets

from .icons import icon
from .theme import color

#: The rail's width, and its buttons' side.
WIDTH = 48
BUTTON = 40
ICON = 20


class RailButton(QtWidgets.QToolButton):
    """An icon-only, checkable button: grey, the text colour under the mouse, the accent when in use."""

    def __init__(self, key, shape, title, shortcut=""):
        super().__init__()
        self.key, self.shape = key, shape
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setFixedSize(BUTTON, BUTTON)
        self.setIconSize(QtCore.QSize(ICON, ICON))
        self.setToolTip("{} ({})".format(title, shortcut) if shortcut else title)
        self.setAccessibleName(title)
        self.setProperty("icon_only", True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QToolButton {{ border: none; border-radius: 6px; padding: 0; }}"
            "QToolButton:hover {{ background: {hover}; }}"
            "QToolButton:checked {{ background: {checked}; }}".format(
                hover=color("hover"), checked=color("checked")))
        self.toggled.connect(lambda _on: self._paint_icon())
        self._paint_icon()

    def _paint_icon(self, hovered=False):
        role = "accent" if self.isChecked() else "text" if hovered else "subtle"
        self.setIcon(icon(self.shape, role=role))

    def enterEvent(self, event):
        self._paint_icon(hovered=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._paint_icon()
        super().leaveEvent(event)


class IconRail(QtWidgets.QWidget):
    """Modes on top, panels under a line, Help and Settings at the foot."""

    mode_chosen = QtCore.Signal(str)
    panel_chosen = QtCore.Signal(str)        # the panel's name: open it, or fold it if it is the open one
    help_clicked = QtCore.Signal()
    settings_clicked = QtCore.Signal()

    def __init__(self, modes, panels, mark=None):
        """`modes` and `panels`: (key, icon, title, shortcut) each; `mark` a widget at the very top."""
        super().__init__()
        self.setFixedWidth(WIDTH)
        self.setObjectName("rail")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#rail {{ background: {}; border-right: 1px solid {}; }}".format(
            color("header"), color("line")))
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(4, 8, 4, 8)
        column.setSpacing(4)
        if mark is not None:
            column.addWidget(mark, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
            column.addSpacing(8)
        self.modes, self.panels = {}, {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for key, shape, title, shortcut in modes:
            button = RailButton(key, shape, title, shortcut)
            button.clicked.connect(lambda _checked=False, k=key: self.mode_chosen.emit(k))
            group.addButton(button)
            self.modes[key] = button
            column.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.line = QtWidgets.QFrame()
        self.line.setFixedSize(24, 1)
        self.line.setStyleSheet("background: {};".format(color("field")))
        column.addSpacing(6)
        column.addWidget(self.line, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addSpacing(6)
        for key, shape, title, shortcut in panels:
            button = RailButton(key, shape, title, shortcut)
            button.clicked.connect(lambda _checked=False, k=key: self.panel_chosen.emit(k))
            self.panels[key] = button
            column.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addStretch(1)
        self.help_button = RailButton("Help", "circle-help", "Help", "F1")
        self.help_button.setCheckable(False)
        self.help_button.clicked.connect(self.help_clicked.emit)
        self.settings_button = RailButton("Settings", "settings", "Settings")
        self.settings_button.setCheckable(False)
        self.settings_button.clicked.connect(self.settings_clicked.emit)
        column.addWidget(self.help_button, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(self.settings_button, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

    def set_mode(self, key):
        self.modes[key].setChecked(True)

    def set_mode_shown(self, key, shown):
        self.modes[key].setVisible(shown)

    def offer_panels(self, keys, open_key=""):
        """The panel buttons this mode has, and which of them is open ("" none)."""
        for key, button in self.panels.items():
            button.setVisible(key in keys)
            button.setChecked(key == open_key)
        self.line.setVisible(bool(keys))


class SidePanel(QtWidgets.QWidget):
    """The panel beside the rail: one of its pages at a time, under the page's name."""

    shown = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidepanel")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#sidepanel {{ background: {}; }}".format(color("card")))
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(8, 8, 4, 4)
        column.setSpacing(6)
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("color: {}; font-weight: 600; letter-spacing: 0.5px;".format(color("subtle")))
        column.addWidget(self.title)
        self.stack = QtWidgets.QStackedWidget()
        column.addWidget(self.stack, 1)
        self.pages = {}
        self.current = ""

    def add(self, key, widget):
        if key not in self.pages:
            self.pages[key] = widget
            self.stack.addWidget(widget)

    def show_page(self, key):
        """Show the page `key`, or fold the panel away when `key` is ""."""
        self.current = key if key in self.pages else ""
        if self.current:
            self.stack.setCurrentWidget(self.pages[self.current])
            self.title.setText(self.current.upper())
        self.setVisible(bool(self.current))
        if self.current:
            self.shown.emit(self.current)


def mark(size=36):
    """The program's mark for the top of the rail: the same tile as the taskbar icon."""
    from .splash import icon_pixmap
    from .theme import window_is_dark
    label = QtWidgets.QLabel()
    ratio = 2.0
    pixmap = icon_pixmap(int(size * ratio), light=not window_is_dark())
    pixmap.setDevicePixelRatio(ratio)
    label.setPixmap(pixmap)
    label.setFixedSize(size, size)
    return label
