"""Editing Cyton+Daisy channel profiles, with or without a board connected.

A profile is only text until it is used: 16 pin names and 16 On/BIAS pairs.
So it can be written the evening before, at a desk, with no headset - and
picked in the morning. Storage and applying stay in the Monitor
(`channel_profiles` + `MonitorPage._profile_chosen`); this dialog only edits.
"""
from pyqtgraph.Qt import QtCore, QtWidgets

from ..core import channel_profiles as cp
from ..core import hardware as hw
from ..core import montages as mo
from .theme import muted, warning
from .widgets import fit_to_screen

#: Offered while typing an electrode name. Plain lists, no MNE import: the
#: dialog must open instantly.
SUGGESTIONS = sorted(set(mo.CHO2017_EEG + mo.TEN_TWENTY_19 + mo.UNICORN_8 + mo.OCCIPITAL
                         + mo.FRONTAL + mo.MIDLINE))


class ChannelProfileDialog(QtWidgets.QDialog):
    """Profiles on the left; the chosen one's 16 pins on the right.

    `store` is an object with `profiles()`, `save(profile)`, `delete(name)`
    and `use(name)` - the Monitor's own profile code.
    """

    def __init__(self, parent, store, current=None):
        super().__init__(parent)
        self.setWindowTitle("Channel profiles · Cyton+Daisy")
        fit_to_screen(self, 760, 640)
        self.store = store
        self._loading = False

        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.currentTextChanged.connect(self._show)
        left.addWidget(self.list, 1)
        for text, callback in (("New from this…", self._new), ("Delete", self._delete)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            left.addWidget(button)
            if text == "Delete":
                self.delete_button = button
        layout.addLayout(left)

        right = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Name"))
        self.name = QtWidgets.QLineEdit()
        self.name.textEdited.connect(self._check)
        row.addWidget(self.name, 1)
        right.addLayout(row)
        self.builtin_note = QtWidgets.QLabel("Built-in profile: read only. Use “New from this…” "
                                             "to make an editable copy.")
        self.builtin_note.setStyleSheet(muted())
        self.builtin_note.setWordWrap(True)
        right.addWidget(self.builtin_note)

        self.table = QtWidgets.QTableWidget(cp.N_PINS, 4)
        self.table.setHorizontalHeaderLabels(["Pin", "Electrode", "On", "BIAS"])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in (0, 2, 3):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.electrodes, self.on_boxes, self.bias_boxes = [], [], []
        completer = QtWidgets.QCompleter(SUGGESTIONS, self)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        for pin in range(cp.N_PINS):
            label = QtWidgets.QLabel("{}{}".format(pin + 1, "  (Daisy)" if pin >= 8 else ""))
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            electrode = QtWidgets.QLineEdit()
            electrode.setCompleter(completer)
            electrode.textEdited.connect(self._check)
            on, bias = QtWidgets.QCheckBox(), QtWidgets.QCheckBox()
            on.toggled.connect(self._check)
            bias.toggled.connect(self._check)
            for column, widget in enumerate((label, electrode, self._centred(on), self._centred(bias))):
                self.table.setCellWidget(pin, column, widget)
            self.electrodes.append(electrode); self.on_boxes.append(on); self.bias_boxes.append(bias)
        right.addWidget(self.table, 1)

        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)
        right.addWidget(self.message)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.clicked.connect(self._save)
        self.use_button = QtWidgets.QPushButton("Save && use")
        self.use_button.setProperty("primary", True)
        self.use_button.setToolTip("Save, then apply: names at once, On/BIAS to the board if "
                                   "streaming, otherwise at the next Start.")
        self.use_button.clicked.connect(self._use)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        for button in (self.save_button, self.use_button, close):
            buttons.addWidget(button)
        right.addLayout(buttons)
        layout.addLayout(right, 1)

        self._reload(current)

    @staticmethod
    def _centred(box):
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        row.addWidget(box)
        return holder

    # ---------------- list ----------------

    def _reload(self, select=None):
        self._profiles = {p.name: p for p in self.store.profiles()}
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems(list(self._profiles))
        self.list.blockSignals(False)
        names = list(self._profiles)
        self.list.setCurrentRow(names.index(select) if select in names else 0)
        self._show(self.list.currentItem().text() if self.list.currentItem() else None)

    def _show(self, name):
        profile = self._profiles.get(name)
        if profile is None:
            return
        self._loading = True
        builtin = profile.name in cp.BUILTIN_NAMES
        self.name.setText(profile.name)
        for i in range(cp.N_PINS):
            self.electrodes[i].setText(profile.names[i])
            self.on_boxes[i].setChecked(profile.settings[i].on)
            self.bias_boxes[i].setChecked(profile.settings[i].bias)
        for widget in [self.name] + self.electrodes + self.on_boxes + self.bias_boxes:
            widget.setEnabled(not builtin)
        self.builtin_note.setVisible(builtin)
        self.delete_button.setEnabled(not builtin)
        self._loading = False
        self._check()

    # ---------------- editing ----------------

    def edited(self):
        """The profile as the form shows it (not validated)."""
        return cp.ChannelProfile(
            self.name.text().strip(),
            [box.text().strip() for box in self.electrodes],
            [hw.ChannelSetting(on.isChecked(), on.isChecked() and bias.isChecked())
             for on, bias in zip(self.on_boxes, self.bias_boxes)])

    def _problem(self):
        profile = self.edited()
        current = self.list.currentItem().text() if self.list.currentItem() else None
        if current in cp.BUILTIN_NAMES:
            return None
        if profile.name in cp.BUILTIN_NAMES:
            return "“{}” is a built-in name; choose another.".format(profile.name)
        try:
            profile.validate()
        except ValueError as exc:
            return str(exc)
        return None

    def _check(self, *_):
        if self._loading:
            return
        for on, bias in zip(self.on_boxes, self.bias_boxes):
            bias.setEnabled(on.isEnabled() and on.isChecked())
        profile = self.edited()
        problem = self._problem()
        builtin = (self.list.currentItem() is not None
                   and self.list.currentItem().text() in cp.BUILTIN_NAMES)
        self.save_button.setEnabled(problem is None and not builtin)
        self.use_button.setEnabled(problem is None)
        on = [n for n, s in zip(profile.names, profile.settings) if s.on]
        if problem:
            self.message.setStyleSheet(warning())
            self.message.setText(problem)
        else:
            self.message.setStyleSheet(muted())
            self.message.setText("{} pins on: {}".format(len(on), ", ".join(on) or "none"))

    def _new(self):
        base = self.edited()
        name, ok = QtWidgets.QInputDialog.getText(self, "New channel profile", "Name:",
                                                  text="{} (copy)".format(base.name))
        name = name.strip()
        if not ok or not name:
            return
        copy = cp.ChannelProfile(name, list(base.names), list(base.settings))
        try:
            if name in cp.BUILTIN_NAMES:
                raise ValueError("“{}” is a built-in name; choose another.".format(name))
            self.store.save(copy)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "New channel profile", str(exc))
            return
        self._reload(name)

    def _save(self):
        if self._problem() is not None:
            return False
        old = self.list.currentItem().text()
        profile = self.edited()
        self.store.save(profile)
        if old != profile.name and old not in cp.BUILTIN_NAMES:
            self.store.delete(old)            # a rename, not a second copy
        self._reload(profile.name)
        return True

    def _use(self):
        current = self.list.currentItem().text()
        if current in cp.BUILTIN_NAMES:
            self.store.use(current)
        elif self._save():
            self.store.use(self.name.text().strip())
        else:
            return
        self.accept()

    def _delete(self):
        name = self.list.currentItem().text()
        if name in cp.BUILTIN_NAMES:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete channel profile",
                                                "Delete “{}”?".format(name))
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.store.delete(name)
            self._reload()
