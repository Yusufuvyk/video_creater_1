"""Choosing a profile applies it; saving and sharing live in one small menu."""
import os

from pyqtgraph.Qt import QtCore, QtWidgets


class MemorySettings:
    """QSettings stand-in for offscreen tests: nothing leaks between runs."""

    def __init__(self):
        self._values = {}

    def value(self, key, default=None, type=None):
        value = self._values.get(key, default)
        return type(value) if type is not None and value is not None else value

    def setValue(self, key, value):
        self._values[key] = value


def page_settings(name):
    """Desktop preferences for a page; in-memory under offscreen validation."""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return MemorySettings()
    return QtCore.QSettings("GIOVA", name)


def menu_button(tooltip, actions, parent=None):
    """A compact '⋯' button whose menu holds secondary actions: [(text, callback) | None]."""
    button = QtWidgets.QToolButton(parent)
    button.setText("…")
    button.setStyleSheet("QToolButton { padding: 2px 6px; font-weight: 600; } "
                         "QToolButton::menu-indicator { image: none; width: 0px; }")
    button.setToolTip(tooltip)
    button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setAutoRaise(True)
    menu = QtWidgets.QMenu(button)
    for entry in actions:
        if entry is None:
            menu.addSeparator()
        else:
            menu.addAction(entry[0], entry[1])
    button.setMenu(menu)
    return button
