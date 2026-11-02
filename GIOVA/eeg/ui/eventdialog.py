"""Naming event codes once: '769' is 'left', '770' is 'right', '1023' is ignored.

Files store what their recording software wrote - numeric codes, or annotation
strings like 'T1'. Which code is which condition is written in the dataset's
documentation, not in the file, so the user says it here and it is saved beside
the recording (`<name>.eventmap.json`) and never asked again.
"""
from pyqtgraph.Qt import QtCore, QtWidgets


class EventMapDialog(QtWidgets.QDialog):
    def __init__(self, parent, events, mapping=None, note=""):
        super().__init__(parent)
        self.setWindowTitle("Name the events")
        self.resize(520, 420)
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(
            (note + "\n\n" if note else "") +
            "Give each event code a condition name. Leave a name empty to ignore that code. "
            "Codes with the same name are pooled into one condition.")
        text.setWordWrap(True)
        layout.addWidget(text)
        counts = events.counts()
        mapping = mapping or {}
        self.table = QtWidgets.QTableWidget(len(counts), 3)
        self.table.setHorizontalHeaderLabels(["Code in file", "Count", "Condition name"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        for row, (code, count) in enumerate(counts.items()):
            for column, value in enumerate((code, str(count))):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(mapping.get(code, code)))
        layout.addWidget(self.table, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mapping(self):
        return {self.table.item(row, 0).text(): self.table.item(row, 2).text().strip()
                for row in range(self.table.rowCount())}

    @classmethod
    def ask(cls, parent, events, mapping=None, note=""):
        dialog = cls(parent, events, mapping, note)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return dialog.mapping()
        return None
