"""The program's Settings: one place for what is set once and rarely changed.

What belongs here: thresholds, limits, folders, defaults. What does not: the
choices made while working (board, filters, channel selection, window length).
Those stay in the Monitor's Controls panel, where they are needed. Anything shown
here is stored by `preferences.py`, so every page reads the same value.

Pages are given by their owners (the Monitor hands over the widgets it already
has), so no setting is defined in two places.
"""
import os
import subprocess
import sys
from pathlib import Path

from pyqtgraph.Qt import QtWidgets

from ..core import sources
from ..core.spectral import AMPLITUDE, AMPLITUDE_DENSITY, POWER_DENSITY
from .preferences import (DEFAULTS, data_dir, downsample_hz, psd_unit, setting, set_setting,
                          user_settings)
from .theme import muted
from .widgets import NoScrollComboBox, fit_to_screen, install_tooltip_switch, spin_box


def reveal(path):
    """Show a file or folder in the system's file manager."""
    path = Path(path)
    try:
        if sys.platform.startswith("win"):
            if path.is_dir():
                os.startfile(str(path))                                  # noqa: S606
            else:
                subprocess.Popen(["explorer", "/select,", str(path)])     # noqa: S607
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R" if path.is_file() else "", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])
        return True
    except OSError:
        return False


class RecordingSettings(QtWidgets.QWidget):
    """Where live recordings are written, and what they are called."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QFormLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(str(data_dir()))
        self.folder.setReadOnly(True)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._choose)
        show = QtWidgets.QPushButton("Open")
        show.clicked.connect(lambda: reveal(self.folder.text()))
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        row.addWidget(show)
        layout.addRow("Folder", row)

        self.prefix = QtWidgets.QLineEdit(str(setting('recording/prefix')))
        self.prefix.setToolTip("The start of every live recording's name; the date and time follow")
        self.prefix.textChanged.connect(self._prefix_changed)
        layout.addRow("File name", self.prefix)
        self.preview = QtWidgets.QLabel()
        self.preview.setStyleSheet(muted())
        self.preview.setWordWrap(True)
        layout.addRow("", self.preview)
        note = QtWidgets.QLabel("A new name is used by the next recording; one being written keeps its own. "
                                "Rename a finished recording in the Projects pane: right-click it → "
                                "Rename recording.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addRow(note)
        self._show_preview()

    def reload(self):
        """Show the folder and name in use (the Projects pane, or Restore defaults, changed them)."""
        self.folder.setText(str(data_dir()))
        self.prefix.setText(str(setting('recording/prefix')))
        self._show_preview()

    def _choose(self):
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for recordings", self.folder.text())
        if not chosen:
            return
        path = Path(chosen)
        try:                                        # a folder that cannot be written is no folder
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".eegstudio_write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Folder", "Cannot write there:\n{}".format(exc))
            return
        set_setting('recording/folder', str(path))
        self.folder.setText(str(path))
        self._show_preview()

    def _prefix_changed(self, text):
        cleaned = sources.safe_stem(text)
        set_setting('recording/prefix', cleaned or 'live')
        self._show_preview()

    def _show_preview(self):
        name = sources.live_record_path(Path(self.folder.text()), self.prefix.text()).name
        self.preview.setText("Next recording: " + name)


class InterfaceSettings(QtWidgets.QWidget):
    """How the program explains itself, and which of its pages are shown."""

    def __init__(self, on_games=None):
        super().__init__()
        self._on_games = on_games
        layout = QtWidgets.QFormLayout(self)
        self.games = QtWidgets.QCheckBox("Show the Games tab")
        self.games.setChecked(setting('ui/show_games'))
        self.games.setToolTip("Games played on the Monitor's live stream (festival demos). "
                              "Off hides the tab; nothing else changes.")
        self.games.toggled.connect(self._games_changed)
        layout.addRow("Pages", self.games)
        self.tooltips = QtWidgets.QCheckBox("Show an explanation when the mouse rests on a button or field")
        self.tooltips.setChecked(setting('ui/tooltips'))
        self.tooltips.toggled.connect(self._tooltips_changed)
        layout.addRow("Hover tips", self.tooltips)
        note = QtWidgets.QLabel("Off hides every hover explanation at once; Help (F1) still has "
                                "all of them. Remembered for the next start.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addRow(note)
        self.theme = NoScrollComboBox()
        self.theme.addItem("Dark", "dark")
        self.theme.addItem("Light", "light")
        self.theme.setCurrentIndex(max(0, self.theme.findData(setting('ui/theme'))))
        self.theme.currentIndexChanged.connect(lambda _i: set_setting('ui/theme', self.theme.currentData()))
        layout.addRow("Theme", self.theme)
        note = QtWidgets.QLabel("The window's colours, from the next start. Reports and saved figures are "
                                "always on white, ready for a page.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addRow(note)
        self.drag = NoScrollComboBox()
        self.drag.addItem("Moves the sheet (Shift + drag selects)", "pan")
        self.drag.addItem("Selects blocks (Space or middle button + drag moves the sheet)", "select")
        self.drag.setCurrentIndex(max(0, self.drag.findData(setting('workbench/drag'))))
        self.drag.currentIndexChanged.connect(lambda _i: set_setting('workbench/drag', self.drag.currentData()))
        layout.addRow("Dragging on the Workbench's sheet", self.drag)
        note = QtWidgets.QLabel("Either way two fingers or the wheel move the sheet and Ctrl + wheel or a pinch "
                                "zooms. At once, no restart.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addRow(note)

    def _tooltips_changed(self, on):
        set_setting('ui/tooltips', bool(on))
        install_tooltip_switch(QtWidgets.QApplication.instance(), on)

    def _games_changed(self, on):
        set_setting('ui/show_games', bool(on))
        if self._on_games is not None:
            self._on_games()

    def reload(self):
        """Show the stored value again (after Restore defaults)."""
        self.games.setChecked(setting('ui/show_games'))
        self.tooltips.setChecked(setting('ui/tooltips'))
        install_tooltip_switch(QtWidgets.QApplication.instance(), self.tooltips.isChecked())
        self.theme.setCurrentIndex(max(0, self.theme.findData(setting('ui/theme'))))
        self.drag.setCurrentIndex(max(0, self.drag.findData(setting('workbench/drag'))))


class GraphSettings(QtWidgets.QWidget):
    """The unit PSD graphs are drawn in, on the Monitor and in the Workbench's spectra."""

    def __init__(self, on_change=None):
        super().__init__()
        self._on_change = on_change
        layout = QtWidgets.QFormLayout(self)
        self.unit = NoScrollComboBox()
        for label, key in (("Power density · µV²/Hz", POWER_DENSITY),
                           ("Amplitude density · µV/√Hz", AMPLITUDE_DENSITY),
                           ("Amplitude (voltage) · µV", AMPLITUDE)):
            self.unit.addItem(label, key)
        self.unit.setToolTip("µV²/Hz: the Welch density itself. µV/√Hz: its square root. "
                             "µV: √(2 · density · 1.5 Δf), so a sine of amplitude A reads A.")
        self.reload()
        self.unit.currentIndexChanged.connect(self._changed)
        layout.addRow("PSD graphs in", self.unit)
        note = QtWidgets.QLabel("Applies at once to the PSD views of the Monitor and to the Workbench's "
                                "spectra (linear, log and dB scales; the Workbench's Scale is chosen beside "
                                "each spectrum). Display only: band powers, features, SNR, "
                                "time-frequency maps and exported values stay in µV²/Hz and µV². "
                                "The FFT view is always an amplitude in µV.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addRow(note)

    def _changed(self, _index=None):
        set_setting('graphs/psd_unit', self.unit.currentData())
        if self._on_change is not None:
            self._on_change()

    def reload(self):
        """Show the stored value again (after Restore defaults)."""
        self.unit.blockSignals(True)
        self.unit.setCurrentIndex(max(0, self.unit.findData(psd_unit())))
        self.unit.blockSignals(False)


class FileSettings(QtWidgets.QWidget):
    """The rate files are opened at (Monitor playback)."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QFormLayout(self)
        self.downsample = spin_box(0, 10000, downsample_hz(), 25, 0, " Hz")
        self.downsample.setSpecialValueText("off (keep the file's rate)")
        self.downsample.setToolTip("Files sampled faster than this are reduced when opened: low-pass at "
                                   "the new Nyquist, then every n-th sample. The file is not changed.")
        self.downsample.valueChanged.connect(
            lambda value: set_setting('files/downsample_hz', float(value)))
        layout.addRow("Downsample files to", self.downsample)
        note = QtWidgets.QLabel("Used the next time a file is played on the Monitor: open the recording again "
                                "to apply. The rate is divided by a whole number and stays at or above this "
                                "value: a 500 Hz file with 250 Hz becomes 250 Hz, a 512 Hz file 256 Hz. Events "
                                "keep their times. Keep the new rate above twice the highest frequency you "
                                "filter or notch (125 Hz or more for a 50 Hz notch). Live streams are not "
                                "reduced; the Workbench reads a file at its own rate (its Downsample block "
                                "reduces it).")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addRow(note)

    def reload(self):
        """Show the stored values again (after Restore defaults)."""
        self.downsample.setValue(downsample_hz())


class AdvancedSettings(QtWidgets.QWidget):
    """The settings file itself, and the way back to the defaults."""

    def __init__(self, on_reset=None):
        super().__init__()
        self._on_reset = on_reset
        layout = QtWidgets.QFormLayout(self)
        store = user_settings()
        where = QtWidgets.QLineEdit(store.fileName())
        where.setReadOnly(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(where, 1)
        show = QtWidgets.QPushButton("Open")
        show.clicked.connect(lambda: reveal(store.fileName()))
        row.addWidget(show)
        layout.addRow("Settings file", row)
        reset = QtWidgets.QPushButton("Restore defaults")
        reset.clicked.connect(self._reset)
        layout.addRow(reset)
        note = QtWidgets.QLabel("Restoring clears the settings on these pages. Recordings are never touched.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addRow(note)

    def _reset(self):
        answer = QtWidgets.QMessageBox.question(
            self, "Restore defaults", "Put every setting back to its default?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for key in DEFAULTS:
            user_settings().remove(key)
        if self._on_reset is not None:
            self._on_reset()


#: The pages that apply where Settings is opened; from Home (or a mode not named here) every page.
#: Interface and Advanced concern the whole program and are everywhere.
MODE_PAGES = {
    "Monitor": ("Recording", "Interface", "Graphs", "Monitor", "Files", "OpenBCI boards", "Advanced"),
    # the Workbench sets everything in its blocks; the PSD unit (Graphs) is set from Home or the Monitor
    "Workbench": ("Interface", "Advanced"),
    "Games": ("Recording", "Interface", "Monitor", "OpenBCI boards", "Advanced"),
}


class SettingsDialog(QtWidgets.QDialog):
    """Categories on the left, one page at a time on the right: those that apply where it is opened (every
    one from Home), and Show all settings for the rest."""

    def __init__(self, parent=None, pages=()):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        fit_to_screen(self, 720, 460)
        root = QtWidgets.QVBoxLayout(self)
        body = QtWidgets.QHBoxLayout()
        self.categories = QtWidgets.QListWidget()
        self.categories.setFixedWidth(160)
        self.pages = QtWidgets.QStackedWidget()
        body.addWidget(self.categories)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        self.show_all = QtWidgets.QPushButton("Show all settings")
        self.show_all.setFlat(True)
        self.show_all.setToolTip("Every page, whatever mode Settings was opened from")
        self.show_all.clicked.connect(lambda: self.show_for(None))
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.show_all)
        bottom.addStretch(1)
        bottom.addWidget(buttons)
        root.addLayout(bottom)
        #: the mode the pages shown are for; None: every page
        self.mode = None
        self.categories.currentRowChanged.connect(self.pages.setCurrentIndex)
        for name, widget in pages:
            self.add_page(name, widget)
        if self.categories.count():
            self.categories.setCurrentRow(0)

    def add_page(self, name, widget):
        """One category. The widget may belong to another page: it is borrowed,
        and given back when this dialog is destroyed with the window."""
        holder = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(holder)
        column.setContentsMargins(4, 4, 4, 4)
        column.addWidget(widget)
        column.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(holder)
        self.categories.addItem(name)
        self.pages.addWidget(scroll)

    def names(self, shown_only=False):
        return [self.categories.item(r).text() for r in range(self.categories.count())
                if not (shown_only and self.categories.isRowHidden(r))]

    def show_for(self, mode):
        """Only the pages that apply in `mode` (MODE_PAGES); None, Home or another: every page."""
        wanted = MODE_PAGES.get(mode)
        self.mode = mode if wanted else None
        for row in range(self.categories.count()):
            self.categories.setRowHidden(row, wanted is not None and self.categories.item(row).text() not in wanted)
        self.setWindowTitle("Settings · {}".format(mode) if wanted else "Settings")
        self.show_all.setVisible(wanted is not None)
        current = self.categories.currentRow()
        if current < 0 or self.categories.isRowHidden(current):
            first = next((r for r in range(self.categories.count()) if not self.categories.isRowHidden(r)), -1)
            self.categories.setCurrentRow(first)

    def show_page(self, name):
        for row in range(self.categories.count()):
            if self.categories.item(row).text() == name:
                if self.categories.isRowHidden(row):
                    self.show_for(None)          # asked for by name: never out of reach
                self.categories.setCurrentRow(row)
                return

    def open_at(self, name=None, mode=None):
        """Show the dialog for `mode` (its pages), at page `name` if given."""
        self.show_for(mode)
        if name:
            self.show_page(name)
        self.show()
        self.raise_()
        self.activateWindow()
