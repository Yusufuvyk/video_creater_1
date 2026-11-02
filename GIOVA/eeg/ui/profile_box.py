"""Choosing a profile applies it; saving and sharing live in one small menu."""
import os
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ..core import analysis_profiles as ap
from .widgets import NoScrollComboBox


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


class ProfileBox(QtWidgets.QWidget):
    """[profile ▾] [⋯]. Picking an entry applies it at once.

    `capture(name)` returns an AnalysisProfile of the page's current state.
    """

    apply_requested = QtCore.Signal(object)
    message = QtCore.Signal(str)

    def __init__(self, capture, preferences, root=None):
        super().__init__()
        self.capture, self.preferences, self.root = capture, preferences, root
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.combo = NoScrollComboBox()
        self.combo.activated.connect(lambda _index: self.apply_current())
        row.addWidget(self.combo, 1)
        row.addWidget(menu_button("Save, share or delete profiles", [
            ("Save current settings", self.save),
            ("Save as new profile…", self.save_as),
            None,
            ("Import profile…", self.import_file),
            ("Export profile…", self.export_file),
            None,
            ("Delete profile", self.delete)], self))
        self.reload(self.preferences.value("profile", ""))

    def current(self):
        index = self.combo.currentIndex()
        return self._profiles[index] if 0 <= index < len(self._profiles) else None

    def reload(self, select=None):
        self._profiles = ap.available(self.root)
        wanted = select if select is not None else self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        for index, profile in enumerate(self._profiles):
            self.combo.addItem(profile.name, profile.name)
            self.combo.setItemData(index, profile.description, QtCore.Qt.ItemDataRole.ToolTipRole)
        found = self.combo.findData(wanted)
        self.combo.setCurrentIndex(max(found, 0))
        self.combo.blockSignals(False)
        profile = self.current()
        self.combo.setToolTip(profile.description if profile else "")

    def apply_current(self):
        profile = self.current()
        if profile is not None:
            self.preferences.setValue("profile", profile.name)
            self.combo.setToolTip(profile.description)
            self.apply_requested.emit(profile)

    def select(self, name):
        """Choose and apply a profile by name (used by tests and sessions)."""
        index = self.combo.findData(name)
        if index >= 0:
            self.combo.setCurrentIndex(index)
            self.apply_current()

    # ---------------- menu actions ----------------

    def _store(self, name):
        try:
            profile = self.capture(name)
            ap.save_user_profile(profile, self.root)
        except (ValueError, OSError) as exc:
            self.message.emit("Cannot save profile: " + str(exc))
            return
        self.preferences.setValue("profile", profile.name)
        self.reload(profile.name)
        self.message.emit("Saved profile '{}'".format(profile.name))

    def save(self):
        profile = self.current()
        if profile is None or profile.name in ap.BUILTIN_NAMES:
            self.save_as()
            return
        self._store(profile.name)

    def save_as(self):
        current = self.current()
        suggestion = "My " + current.name if current is not None else "My profile"
        name, ok = QtWidgets.QInputDialog.getText(self, "Save analysis profile", "Profile name", text=suggestion)
        name = name.strip()
        if not ok or not name:
            return
        if any(p.name == name for p in self._profiles) and QtWidgets.QMessageBox.question(
                self, "Replace profile", "Replace the profile '{}'?".format(name)) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._store(name)

    def delete(self):
        profile = self.current()
        if profile is None:
            return
        if profile.name in ap.BUILTIN_NAMES:
            self.message.emit("Built-in profiles cannot be deleted")
            return
        if QtWidgets.QMessageBox.question(self, "Delete profile", "Delete '{}'?".format(profile.name)) != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        ap.delete_user_profile(profile.name, self.root)
        self.reload()
        self.message.emit("Deleted profile '{}'".format(profile.name))

    def import_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import analysis profile", "",
                                                        "Analysis profile (*.json)")
        if not path:
            return
        try:
            profile = ap.load_profile(path)
            ap.save_user_profile(profile, self.root)
        except (ValueError, OSError) as exc:
            self.message.emit("Cannot import profile: " + str(exc))
            return
        self.reload(profile.name)
        self.apply_current()

    def export_file(self):
        profile = self.current()
        if profile is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export analysis profile", ap.slug(profile.name) + ap.SUFFIX, "Analysis profile (*.json)")
        if path:
            try:
                ap.save_profile(Path(path), profile)
                self.message.emit("Exported profile: " + path)
            except (ValueError, OSError) as exc:
                self.message.emit("Cannot export profile: " + str(exc))
