"""Giving a recording's channels their real electrode names.

A BrainFlow CSV has sixteen unlabelled columns. Which electrode sits on which
column is decided when the headset goes on the head, and the only place that
knowledge exists is in the person who put it there. This asks them, once, and
`formats.channels` remembers it beside the recording.

Naming them properly is not cosmetic: it is what lets the head map place them,
what makes a `Motor (C3, Cz, C4)` preset mean anything, and what stops a plot
labelled EEG7 from being unreadable to anyone but its author.
"""
from pyqtgraph.Qt import QtWidgets

from ..core import montages as mo
from .theme import muted
from .widgets import NoScrollComboBox

GENERIC = "(generic CH1..CHn)"
KEEP = "(leave as they are)"


class ChannelNamesDialog(QtWidgets.QDialog):
    """Pick a montage, then correct individual electrodes by hand."""

    def __init__(self, parent, names):
        super().__init__(parent)
        self.setWindowTitle("Channel names")
        self.resize(420, 560)
        self.names = list(names)

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "{} channels. Apply a montage, then edit any electrode that is "
            "not where the montage assumes. Saved beside the recording, so it "
            "loads with it from now on.".format(len(self.names)))
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        layout.addWidget(note)

        row = QtWidgets.QHBoxLayout()
        self.montage = NoScrollComboBox()
        self.montage.addItem(KEEP)
        self.montage.addItem(GENERIC)
        # Only montages that actually fit: offering a 64-electrode cap for a
        # 16-channel recording is an invitation to a silent mismatch.
        for label, montage in mo.MONTAGES:
            if len(montage) == len(self.names):
                self.montage.addItem(label)
        row.addWidget(self.montage, 1)
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self._apply_montage)
        row.addWidget(apply_button)
        layout.addLayout(row)

        self.table = QtWidgets.QTableWidget(len(self.names), 2)
        self.table.setHorizontalHeaderLabels(["Channel", "Electrode"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self._fill(self.names)
        layout.addWidget(self.table, 1)

        self.warning = QtWidgets.QLabel("")
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.itemChanged.connect(self._check)
        self._check()

    def _fill(self, names):
        self.table.blockSignals(True)
        for row, name in enumerate(names):
            index = QtWidgets.QTableWidgetItem(str(row + 1))
            index.setFlags(index.flags() & ~QtWidgets.QTableWidgetItem().flags().ItemIsEditable)
            self.table.setItem(row, 0, index)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(name)))
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()

    def _apply_montage(self):
        choice = self.montage.currentText()
        if choice == KEEP:
            return
        names = (mo.generic_names(len(self.names)) if choice == GENERIC
                 else mo.montage_named(choice))
        if names:
            self._fill(names)
        self._check()

    def _check(self):
        """Say how many of the current names the head map will be able to place."""
        names = self.chosen
        placed = len(mo.positions_2d(names))
        duplicates = len(names) - len(set(n.strip().lower() for n in names))
        text = "{} of {} will appear on the head map.".format(placed, len(names))
        if duplicates:
            text += "  {} duplicate name(s) - they will be indistinguishable.".format(
                duplicates)
        self.warning.setStyleSheet(muted())
        self.warning.setText(text)

    @property
    def chosen(self):
        out = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            text = item.text().strip() if item else ""
            out.append(text or "CH{}".format(row + 1))
        return out

    @classmethod
    def ask(cls, parent, names):
        dialog = cls(parent, names)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return dialog.chosen
        return None
