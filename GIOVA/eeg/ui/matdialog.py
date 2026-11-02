"""Asking how to read a .mat file, and remembering the answer.

`matfile` can often guess a layout, but when it cannot, the choice is genuinely
the user's: which of six arrays is the condition to analyse, where the EEG rows
stop and the EMG begins, what the stored integers mean in microvolts. This is
where that conversation happens.

Two things this fixes about the old one-shot dialog. It can be re-opened - the
sidecar it writes is an answer, not a life sentence, and switching from
`imagery_left` to `imagery_right` is an ordinary thing to want. And it shows,
live, what the current choice would actually produce, so a scale that is out by
a factor of 32 is visible before it silently poisons every number on the page.

Both pages call `open_recording`; neither one repeats any of this.
"""
import html
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from ..formats import layouts as lo
from ..formats import matfile as mat
from ..core import montages as mo
from ..core import paradigm as pdm
from ..formats import presets as ps
from .theme import color, note_box
from .widgets import combo_box, spin_box

#: Unit scales offered, and what they multiply by. "A/D counts" is here because
#: it is the one that actually bit us: Cho 2017 stores Biosemi counts at
#: 31.25 nV each, and reading them as microvolts makes everything 32x too big.
UNITS = (
    ("Auto-detect", None),
    ("Already microvolts", 1.0),
    ("Volts (x 1e6)", 1e6),
    ("Millivolts (x 1e3)", 1e3),
    ("Biosemi A/D counts (/32)", ps.CHO2017_COUNTS_TO_UV),
)

GENERIC_NAMES = "(generic CH1..CHn)"
FILE_NAMES = "from variable: {}"
#: An optional variable left unchosen.
NONE = "(none)"

#: What the dialog asks, in the order to answer it: said at its top for someone who has not read a layout before.
STEPS = ("<b>How to read this file.</b> The list shows every variable in it. "
         "1. <b>Signal</b>: the large 2-D array; numbered ones (data{1}.X, data{2}.X ...) are runs, the {*} entry "
         "reads them all. 2. <b>Sampling rate</b>. 3. <b>Events</b>, if a variable holds their times: the trials "
         "are cut around them. 4. <b>Channel names</b>. The list at the bottom says what is still missing; "
         "nothing is read until it says Ready.")

#: Colours of a checklist line's state.
STATE_WORDS = {mat.READY: "Ready", mat.MISSING: "Missing", mat.OPTIONAL: "Optional"}


class MatMappingDialog(QtWidgets.QDialog):
    """How should this file be read? Pre-filled, live-previewed, re-openable."""

    def __init__(self, parent, name, reason, variables, mapping=None,
                 preset=None, layouts_root=None):
        super().__init__(parent)
        self.setWindowTitle("How should {} be read?".format(name))
        self.setMinimumWidth(560)
        self.variables = variables
        self.preset = preset
        self.file_name = name
        self.layouts_root = layouts_root
        #: optional fields the user changed by hand: a suggestion never overwrites them
        self._touched = set()

        layout = QtWidgets.QVBoxLayout(self)
        if reason:
            message = QtWidgets.QLabel(reason)
            message.setWordWrap(True)
            layout.addWidget(message)
        steps = QtWidgets.QLabel(STEPS)
        steps.setWordWrap(True)
        steps.setStyleSheet("color: {};".format(color("subtle")))
        layout.addWidget(steps)

        if preset is not None:
            layout.addWidget(self._preset_box(preset))

        layout.addWidget(self._variable_table())
        layout.addLayout(self._form())

        self.checks = QtWidgets.QLabel()
        self.checks.setWordWrap(True)
        self.checks.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.checks)
        self.preview = QtWidgets.QLabel("-")
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet(note_box())
        layout.addWidget(self.preview)

        keep = QtWidgets.QHBoxLayout()
        self.keep = QtWidgets.QCheckBox("Read every file laid out like this one the same way, as")
        self.keep.setToolTip("Kept by name in Data/layouts: a file with the same variables and number of channels "
                             "is then read without asking. Leave it off for a file that is one of a kind.")
        self.keep.setChecked(preset is None)
        self.keep_name = QtWidgets.QLineEdit("Layout of {}".format(Path(name).stem))
        keep.addWidget(self.keep)
        keep.addWidget(self.keep_name, 1)
        layout.addLayout(keep)

        buttons = self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Read the file")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._parts = None
        self._bad_window = None
        start = mapping or (preset.conditions[0][1] if preset else None)
        self._fill_from(start)
        if start is None:
            self._data_changed()        # a first look: what lies beside the signal is offered
        else:
            self._refresh()

    # ---------------- pieces ----------------

    def _preset_box(self, preset):
        group = QtWidgets.QGroupBox("Recognised dataset: {}".format(preset.name))
        box = QtWidgets.QVBoxLayout(group)
        note = QtWidgets.QLabel(preset.note)
        note.setWordWrap(True)
        box.addWidget(note)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Condition"))
        # The one thing a preset must never decide for you. Left hand and right
        # hand are different experiments, not different spellings.
        self.condition = combo_box(preset.labels)
        self.condition.currentIndexChanged.connect(self._condition_changed)
        row.addWidget(self.condition, 1)
        apply_button = QtWidgets.QPushButton("Use preset")
        apply_button.clicked.connect(self._condition_changed)
        row.addWidget(apply_button)
        box.addLayout(row)
        return group

    def _variable_table(self):
        rows = mat.describe_variables(self.variables)
        # every run at once first: data{*}.X stands for data{1}.X ... data{9}.X
        self._patterns = mat.patterns(self.variables)
        rows = [(key, "{} runs".format(len(mat.runs(self.variables, key))), "every run")
                for key in self._patterns] + rows
        table = QtWidgets.QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Variable", "Shape", "Kind"])
        table.verticalHeader().setVisible(False)
        for row, (key, shape, kind) in enumerate(rows):
            for column, text in enumerate((key, str(shape), kind)):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        table.resizeColumnsToContents()
        table.setMaximumHeight(170)
        self._keys = [key for key, _, _ in rows]
        return table

    def _form(self):
        form = self.form = QtWidgets.QFormLayout()

        self.data_key = combo_box(self._keys)
        self.data_key.setToolTip("The variable holding the EEG: a 2-D array, channels x samples or samples x "
                                 "channels. A {*} entry reads every numbered run one after another.")
        self.data_key.currentIndexChanged.connect(self._data_changed)
        form.addRow("Signal", self.data_key)

        self.fs_value = spin_box(1, 100000, 512, 1, 2, " Hz")
        self.fs_value.valueChanged.connect(self._refresh)
        form.addRow("Sampling rate", self.fs_value)

        self.transpose = QtWidgets.QCheckBox(
            "Rows are samples, columns are channels")
        self.transpose.toggled.connect(self._refresh)
        form.addRow("", self.transpose)

        # a signal already cut into trials (3-D): which axis is which, and where each trial starts
        self.axes = QtWidgets.QComboBox()
        self.axes.setToolTip("The signal is cut into trials. Which of its three sizes is the trials, the channels "
                             "and the samples? The list below says when the result looks the wrong way round.")
        self.axes.currentIndexChanged.connect(self._refresh)
        form.addRow("Axes", self.axes)
        self.trial_start = spin_box(-60, 60, 0, 0.1, 3, " s")
        self.trial_start.setToolTip("When each stored trial starts, from its event (its cue): -0.5 if it begins "
                                    "half a second before the cue, 0 if at the cue")
        self.trial_start.valueChanged.connect(self._refresh)
        form.addRow("Trial starts at", self.trial_start)

        self.rows = QtWidgets.QLineEdit()
        self.rows.setPlaceholderText("all rows - or e.g. 1-64 to drop EMG")
        self.rows.textChanged.connect(self._refresh)
        form.addRow("EEG rows", self.rows)

        self.names_key = combo_box(self._name_options())
        self.names_key.currentIndexChanged.connect(self._refresh)
        form.addRow("Channel names", self.names_key)

        self.units = combo_box([label for label, _ in UNITS])
        self.units.currentIndexChanged.connect(self._refresh)
        form.addRow("Units", self.units)

        optional = [NONE] + self._keys
        self.onsets_key = self._optional(optional, "When each event happens: one number per event (trial, pos, "
                                                   "onset, latency ...). For runs, the {*} entry beside the signal's.")
        form.addRow("Event times", self.onsets_key)
        self.onset_unit = combo_box(list(mat.ONSET_UNITS))
        self.onset_unit.setToolTip("MATLAB counts samples from 1 (the first sample is 1); Python from 0. "
                                   "The list below says so when the times run past the signal.")
        self.onset_unit.currentIndexChanged.connect(self._refresh)
        form.addRow("  counted in", self.onset_unit)
        self.codes_key = self._optional(optional, "Which kind each event is: one code or name per event (y, "
                                                  "type, label ...). None: every event is called 'event'.")
        form.addRow("Event codes", self.codes_key)
        self.labels_key = self._optional(optional, "Names of the codes, when the codes are numbers: code 1 is "
                                                   "the first name (classes ...).")
        form.addRow("Code names", self.labels_key)
        self.bad_key = self._optional(optional, "One value per event, true (1) for a trial the dataset marks as "
                                                "bad (artifacts ...): trials over it are left out.")
        form.addRow("Trials marked bad", self.bad_key)
        return form

    def _optional(self, items, tip):
        box = combo_box(items)
        box.setToolTip(tip)
        box.currentIndexChanged.connect(lambda _i, b=box: (self._touched.add(id(b)), self._refresh()))
        return box

    def _signal_array(self, key=None):
        found = mat.runs(self.variables, key or self.data_key.currentText())
        return np.asarray(self.variables[found[0][1]]) if found else None

    def _show_trial_fields(self, axes=None):
        """Axes and Trial starts at for a signal cut into trials (3-D); the event time fields otherwise."""
        array = self._signal_array()
        cut = array is not None and array.ndim == 3
        if cut:
            self.axes.blockSignals(True)
            self.axes.clear()
            words = {"t": "trials", "c": "channels", "s": "samples"}
            for order in ("tcs", "tsc", "cts", "cst", "sct", "stc"):
                self.axes.addItem(" x ".join("{} ({})".format(words[r], n) for r, n in zip(order, array.shape)),
                                  order)
            wanted = axes or mat.guess_axes(array.shape)
            self.axes.setCurrentIndex(max(self.axes.findData(wanted), 0))
            self.axes.blockSignals(False)
        for widget, shown in ((self.axes, cut), (self.trial_start, cut), (self.transpose, not cut),
                              (self.onsets_key, not cut), (self.onset_unit, not cut)):
            self.form.setRowVisible(widget, shown)
        return cut

    def _data_changed(self):
        """A new signal: the fields beside it that look like its events and rate are offered, never forced."""
        self._show_trial_fields()
        suggested = mat.suggest(self.variables, self.data_key.currentText())
        for box, slot in ((self.onsets_key, "onsets_key"), (self.codes_key, "codes_key"),
                          (self.labels_key, "labels_key"), (self.bad_key, "bad_key")):
            if id(box) in self._touched or slot not in suggested:
                continue
            box.blockSignals(True)
            box.setCurrentText(suggested[slot])
            box.blockSignals(False)
        rate_key = suggested.get("fs_key")
        if rate_key:
            found = mat.runs(self.variables, rate_key)
            try:
                rate = float(np.asarray(self.variables[found[0][1]]).reshape(-1)[0]) if found else 0.0
            except (TypeError, ValueError):
                rate = 0.0
            if rate > 0:
                self.fs_value.blockSignals(True)
                self.fs_value.setValue(rate)
                self.fs_value.blockSignals(False)
        array = None
        found = mat.runs(self.variables, self.data_key.currentText())
        if found:
            array = np.asarray(self.variables[found[0][1]])
        if array is not None and array.ndim == 3:
            self.onsets_key.blockSignals(True)
            self.onsets_key.setCurrentText(NONE)         # each trial is its own event
            self.onsets_key.blockSignals(False)
        if array is not None and array.ndim == 2:
            self.transpose.blockSignals(True)
            self.transpose.setChecked(array.shape[0] > array.shape[1])
            self.transpose.blockSignals(False)
        self._refresh()

    def _name_options(self):
        """Generic, then every montage, then any variable holding labels."""
        options = [GENERIC_NAMES]
        options.extend(label for label, _ in mo.MONTAGES)
        options.extend(FILE_NAMES.format(key) for key in self._keys)
        return options

    # ---------------- state ----------------

    def _fill_from(self, mapping):
        """Show an existing mapping, so re-opening starts where you left off."""
        if mapping is None:
            return
        self._parts = getattr(mapping, "parts", None)
        for widget in (self.data_key, self.fs_value, self.transpose, self.rows,
                       self.names_key, self.units):
            widget.blockSignals(True)

        if mapping.data_key in self._keys:
            self.data_key.setCurrentText(mapping.data_key)
        self._show_condition(mapping.data_key, mapping)
        if mapping.fs:
            self.fs_value.setValue(float(mapping.fs))
        self.transpose.setChecked(bool(mapping.transpose))
        self.rows.setText(mapping.rows or "")

        if mapping.names:
            label = mo.guess_montage(mapping.names, len(mapping.names))
            self.names_key.setCurrentText(label or GENERIC_NAMES)
        elif mapping.names_key:
            self.names_key.setCurrentText(FILE_NAMES.format(mapping.names_key))
        else:
            self.names_key.setCurrentText(GENERIC_NAMES)

        match = next((label for label, scale in UNITS
                      if scale is not None
                      and abs(scale - mapping.scale) < 1e-12), None)
        self.units.setCurrentText(match or "Auto-detect")
        for box, key in ((self.onsets_key, mapping.onsets_key), (self.codes_key, mapping.codes_key),
                         (self.labels_key, mapping.labels_key), (self.bad_key, mapping.bad_key)):
            box.blockSignals(True)
            box.setCurrentText(key if key in self._keys else NONE)
            box.blockSignals(False)
            if key:
                self._touched.add(id(box))
        self.onset_unit.setCurrentText(mapping.onset_unit or mat.SAMPLES_1)
        self._bad_window = getattr(mapping, "bad_window", None)
        self._show_trial_fields(getattr(mapping, "axes", None))
        self.trial_start.blockSignals(True)
        self.trial_start.setValue(float(getattr(mapping, "trial_start", None) or 0.0))
        self.trial_start.blockSignals(False)

        for widget in (self.data_key, self.fs_value, self.transpose, self.rows,
                       self.names_key, self.units):
            widget.blockSignals(False)

    def _show_condition(self, data_key, mapping=None):
        """Point the condition combo at whatever data variable is chosen.

        Otherwise re-opening on a saved `imagery_right` shows `imagery_left` in
        the combo beside it, and the dialog contradicts itself on sight.
        """
        if self.preset is None:
            return
        field = str(data_key).split(".")[-1]
        if mapping is not None and getattr(mapping, "parts", None):
            field = next((label for label, m in self.preset.conditions
                          if m.parts == mapping.parts), field)
        if field not in self.preset.labels:
            return
        self.condition.blockSignals(True)
        self.condition.setCurrentText(field)
        self.condition.blockSignals(False)

    def _condition_changed(self):
        if self.preset is None:
            return
        self._fill_from(self.preset.mapping_for(self.condition.currentText()))
        self._refresh()

    def _refresh(self):
        """Recompute the preview line from whatever is currently chosen."""
        mapping = self.mapping()
        valid = True
        try:
            text = mat.preview(self.variables, mapping)
        except Exception as exc:  # a half-typed field is not a crash
            text = "{}: {}".format(type(exc).__name__, exc)
            valid = False

        try:
            warning = self._names_warning(mapping)
        except mat.MatError as exc:
            warning = str(exc)
        try:
            lines = mat.checklist(self.variables, mapping)
        except Exception as exc:  # a half-chosen layout is not a crash
            lines = [(mat.MISSING, "{}: {}".format(type(exc).__name__, exc))]
        if warning:
            lines.append((mat.MISSING, "Channel names: " + warning))
        elif self.names_key.currentText() == GENERIC_NAMES and not any(s == mat.MISSING for s, _ in lines):
            lines.append((mat.OPTIONAL, "Channel names: CH1, CH2 ... The file names none: choose the dataset's "
                                        "montage under Channel names (its paper lists them), or a variable "
                                        "holding them. Head maps and channel picks need real names."))
        missing = any(state == mat.MISSING for state, _ in lines)
        self.checks.setText("<br>".join(
            "<span style='color: {}; font-weight: 600;'>{}</span> {}".format(
                color({mat.READY: "good", mat.MISSING: "bad"}.get(state, "subtle")), STATE_WORDS[state],
                html.escape(text)) for state, text in lines))
        self.preview.setText(text)
        bad = bool(warning) or "too large" in text or "too small" in text
        self.preview.setStyleSheet(note_box(bad))
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(valid and not missing)

    def _names_warning(self, mapping):
        """Say so when the chosen montage does not fit the chosen rows.

        `apply_mapping` falls back to generic names in that case rather than
        mislabelling electrodes, but silently swapping the labels the user
        asked for is exactly the confusion this dialog exists to end.
        """
        if not mapping.names:
            return ""
        array = self._signal_array(mapping.data_key)
        if array is None or array.ndim not in (2, 3) or (array.ndim == 3 and not mapping.axes):
            return ""
        n_rows = mat.channels_of(array, mapping)
        rows = mat.parse_rows(mapping.rows, n_rows)
        n_kept = len(rows) if rows is not None else n_rows
        if len(mapping.names) == n_kept:
            return ""
        return ("This montage has {} electrodes but {} rows are selected - the "
                "names will be ignored. Fix the EEG rows, or pick another "
                "montage.".format(len(mapping.names), n_kept))

    def mapping(self):
        """The Mapping the current widget state describes."""
        data_key = self.data_key.currentText()
        transpose = self.transpose.isChecked()
        # A joined preset stays joined only while its first array is the data variable.
        parts = self._parts if self._parts and self._parts[0][0] == data_key else None

        scale = dict(UNITS)[self.units.currentText()]
        if scale is None:  # auto: judge from the oriented data
            found = mat.runs(self.variables, data_key)
            array = np.asarray(self.variables[found[0][1]] if found else [[0.0]], dtype=float)
            scale = mat.guess_scale(array.T if transpose else array)

        chosen = self.names_key.currentText()
        names, names_key = None, None
        if chosen != GENERIC_NAMES:
            montage = mo.montage_named(chosen)
            if montage is not None:
                names = montage
            else:
                names_key = chosen[len(FILE_NAMES.format("")):] or None

        chosen_key = lambda box: None if box.currentText() == NONE else box.currentText()
        signal = self._signal_array(data_key)
        cut = signal is not None and signal.ndim == 3
        onsets_key = None if cut else chosen_key(self.onsets_key)
        labelled = bool(onsets_key) or cut
        return mat.Mapping(
            data_key, fs=self.fs_value.value(), transpose=transpose,
            names_key=names_key, scale=scale, rows=self.rows.text().strip(),
            names=names, preset=self.preset.name if self.preset else None, parts=parts,
            onsets_key=onsets_key, onset_unit=self.onset_unit.currentText() if onsets_key else None,
            codes_key=chosen_key(self.codes_key) if labelled else None,
            labels_key=chosen_key(self.labels_key) if labelled else None,
            bad_key=chosen_key(self.bad_key) if labelled else None,
            bad_window=getattr(self, "_bad_window", None) if onsets_key else None,
            axes=self.axes.currentData() if cut else None,
            trial_start=self.trial_start.value() if cut else None)

    def keep_layout(self, mapping):
        """Keep `mapping` by name for files laid out alike, when asked; the Layout kept, or None."""
        if not self.keep.isChecked():
            return None
        array = self._signal_array(mapping.data_key)
        channels = mat.channels_of(array, mapping) if array is not None and array.ndim in (2, 3) else None
        layout = lo.Layout(self.keep_name.text().strip() or "Layout of {}".format(Path(self.file_name).stem),
                           "Chosen for {}".format(self.file_name), mapping, channels=channels)
        lo.save(layout, self.layouts_root)
        return layout

    @classmethod
    def ask(cls, parent, name, reason, variables, mapping=None, preset=None, layouts_root=None):
        dialog = cls(parent, name, reason, variables, mapping, preset, layouts_root)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            mapping = dialog.mapping()
            dialog.keep_layout(mapping)
            return mapping
        return None


# ----------------------------------------------------------------------
# opening a recording
# ----------------------------------------------------------------------


def is_mat(path):
    return Path(path).suffix.lower() == ".mat"


def _reduced(recording, note):
    """Downsample what was just read, when Settings → Analysis asks for it."""
    if recording is None:
        return recording, note
    from ..core.sources import downsample
    from .preferences import downsample_hz
    return downsample(recording, downsample_hz()), note


def open_recording(parent, path):
    """Load any recording, asking about the layout only when guessing fails.

    Returns (recording, note). `recording` is None when the user cancelled or
    the file could not be read; `note` is a line for the status bar, or None.
    """
    return _reduced(*_open_recording(parent, path))


def _open_recording(parent, path):
    from ..formats.recordings import load_recording

    try:
        return load_recording(path), None
    except mat.MatLayoutUnknown as exc:
        variables = mat.read_variables(path)
        preset = ps.detect(variables)
        reason = str(exc)
        if preset is not None:
            # The rival arrays that defeated auto-detection are the conditions;
            # say so, rather than presenting it as a failure.
            reason = ("This file is a recognised dataset. Choose which "
                      "condition to analyse.")
        mapping = MatMappingDialog.ask(parent, Path(path).name, reason,
                                       variables, preset=preset)
        if mapping is None:
            return None, "cancelled: {}".format(exc)
        return _accept(parent, path, variables, mapping, preset)
    except mat.MatError as exc:
        QtWidgets.QMessageBox.warning(parent, "Could not read", str(exc))
        return None, str(exc)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(
            parent, "Could not read", "{}: {}".format(type(exc).__name__, exc))
        return None, "could not read {}".format(Path(path).name)


def remap(parent, path):
    """Re-open the layout dialog for a .mat, pre-filled with its current answer.

    The saved sidecar is an answer, not a life sentence: switching condition, or
    correcting a unit scale that turned out to be wrong, has to be possible
    without deleting files by hand.
    """
    if not is_mat(path):
        return None, "layout only applies to .mat files"

    try:
        variables = mat.read_variables(path)
    except mat.MatError as exc:
        QtWidgets.QMessageBox.warning(parent, "Could not read", str(exc))
        return None, str(exc)

    preset = ps.detect(variables)
    try:
        current = mat.mapping_for(path, variables)    # its own, a kept layout's, or a guess
    except mat.MatLayoutUnknown:
        current = None

    mapping = MatMappingDialog.ask(
        parent, Path(path).name,
        "Change how this file is read. The preview below updates as you edit.",
        variables, mapping=current, preset=preset)
    if mapping is None:
        return None, None
    return _reduced(*_accept(parent, path, variables, mapping, preset))


def _accept(parent, path, variables, mapping, preset):
    """Apply a chosen mapping, remember it, and note any trial structure."""
    try:
        recording = mat.apply_mapping(variables, mapping, path)
    except mat.MatError as exc:
        QtWidgets.QMessageBox.warning(parent, "Could not read", str(exc))
        return None, str(exc)

    mat.save_mapping(path, mapping)  # do not ask again for this file
    note = "{}: {}".format(Path(path).name, recording.describe())
    if recording.events is not None and len(recording.events):
        note += " | events: " + recording.events.describe() + " - use the Epoch / ERD tab"
    if preset is not None and _write_paradigm(path, variables, mapping, preset):
        note += " | trial structure saved - trial averaging is available"
    return recording, note


def _write_paradigm(path, variables, mapping, preset):
    """Save the preset's trial structure beside the recording. True if written.

    Not written over a paradigm someone edited by hand: only over one this
    preset wrote earlier, which is the case that matters when the condition
    changes and the trial count changes with it.
    """
    paradigm = ps.paradigm_for(variables, mapping)
    if paradigm is None:
        return False

    existing = pdm.load_for(path)
    if existing is not None and not existing.name.startswith(preset.name):
        return False
    pdm.save(pdm.path_for(path), paradigm)
    return True
