"""The right-hand side of the Workbench: the selected block's settings and its result.

A block's parameters become a short form. Only the ones that usually matter are
shown; the rest wait behind Options. A setting that means nothing with the others
as they are (a Chebyshev's ripple under a Butterworth design) is hidden, its value
kept. Lists are typed as text ("C3, Cz, C4", "mu 8-13, beta 13-30"); where the
recording is known, Pick… offers its channel names or event labels so nobody has to
remember how a dataset spells them.
"""
import json
import re

import numpy as np

from pyqtgraph.Qt import QtCore, QtWidgets

from ...pipeline import block
from ..theme import color, muted, plot_background
from ..widgets import NoScrollComboBox, NoScrollIntBox, NoScrollSpinBox
from .previews import EpochsPreview, PreviewWindow, _Preview, preview, short_list
from .problems import ProblemBox
from ..icons import only

#: Files the Recording block opens, for its Browse button.
RECORDING_FILTER = ("Recordings (*.csv *.txt *.edf *.bdf *.gdf *.vhdr *.fif *.set *.mat);;All files (*)")
#: Parameters whose values a recording can offer to pick from, and those that are electrodes (picked on a head).
PICKABLE = ("channels", "conditions", "selected", "eog_channels")
ELECTRODES = ("channels", "eog_channels")


#: List parameters that hold numbers even when empty by default, and the Epochs block's trial table.
NUMBER_LISTS = ("notches",)
TRIAL_TABLES = ("trials",)


#: A citation: an author and a year, as "Müller-Putz et al., 2008", "Cohen, 1960" or "Ojala & Garriga, 2010".
_CITED = re.compile(r"[A-ZÀ-Ž][^,;()]*?,\s*(19|20)\d\d[a-z]?")


def uncited(text):
    """The text without its citations: "(alpha 0.05; Müller-Putz et al., 2008)" becomes "(alpha 0.05)", a
    bracket holding only citations goes, and "Cohen's kappa (Cohen, 1960)" becomes "Cohen's kappa"."""
    def bracket(match):
        kept = [part.strip() for part in match.group(1).split(";") if not _CITED.fullmatch(part.strip())]
        return " ({})".format("; ".join(kept)) if kept else ""
    return re.sub(r"\s*\(([^()]*)\)", bracket, text)


def first_sentence(text):
    """Up to the first full stop that ends a sentence (followed by a space and a capital, or the end)."""
    match = re.search(r"\.(?=\s+[A-Z]|$)", text)
    return text[:match.end()] if match else text


def field_kind(param):
    default = param.default
    if param.choices:
        return "choice"
    if param.name in NUMBER_LISTS:
        return "numbers"
    if param.name in TRIAL_TABLES:
        return "trials"
    if param.name == "ratios":
        return "ratios"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, list):
        if default and isinstance(default[0], list):
            # [name, low, high] bands, or unnamed [low, high] ranges (a filter bank's)
            return "bands" if isinstance(default[0][0], str) else "ranges"
        if default and all(isinstance(v, (int, float)) for v in default):
            return "numbers"
        return "names"
    return "path" if param.name == "path" else "text"


def number_text(value):
    """30 as 30 and 7.5 as 7.5, but 3600.125 in full: 12 significant digits, so what a field shows reads back as
    the number it holds. Six (the usual {:g}) wrote a trial at 3600.125 s as 3600.12, 5 ms off once edited."""
    return "{:.12g}".format(float(value))


def to_text(kind, value):
    if kind == "names":
        return ", ".join(str(v) for v in (value or []))
    if kind == "numbers":
        return ", ".join(number_text(v) for v in value)
    if kind == "bands":
        return ", ".join("{} {}-{}".format(name, number_text(low), number_text(high)) for name, low, high in value)
    if kind == "ranges":
        return ", ".join("{}-{}".format(number_text(low), number_text(high)) for low, high in value)
    if kind == "ratios":
        return ", ".join("{}={}/{}".format(name, a, b) for name, a, b in value)
    if kind == "trials":
        return "; ".join("{} | {} | {}-{}{}".format(row[0], row[1], number_text(row[2]), number_text(row[3]),
                                                    "" if len(row) < 5 or row[4] else " | out")
                         for row in value or [])
    return str(value)


#: a number as a field shows or a person types it: 30, -0.5, .5, 1e-05
_UNSIGNED = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER = "-?" + _UNSIGNED
_TRIAL = re.compile(r"^\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(" + _NUMBER + r")\s*[-–]\s*(" + _NUMBER + r")\s*"
                    r"(?:\|\s*(out|in)\s*)?$")


def trials_from_text(text):
    """Rows [id, condition, start, end, included] from "1 | Eyes open | 0-30; 2 | Eyes closed | 30-60 | out"."""
    rows = []
    for part in (p for p in text.split(";") if p.strip()):
        match = _TRIAL.match(part)
        if not match:
            raise ValueError("Write each trial as id | condition | start-end, adding | out when it is left out "
                             "('{}' is not)".format(part.strip()))
        rows.append([match.group(1), match.group(2), float(match.group(3)), float(match.group(4)),
                     match.group(5) != "out"])
    return rows


_BAND = re.compile(r"^\s*(\S+)\s+(" + _NUMBER + r")\s*[-–]\s*(" + _NUMBER + r")\s*$")
_RANGE = re.compile(r"^\s*(" + _UNSIGNED + r")\s*[-–]\s*(" + _UNSIGNED + r")\s*$")


def from_text(kind, text, length=None):
    """The value typed in a text field; ValueError with a reason when it does not read."""
    if kind == "trials":
        return trials_from_text(text)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if kind == "ratios":
        ratios = []
        for part in parts:
            name, equals, fraction = part.rpartition("=") if "=" in part else (part, "", part)
            top, slash, bottom = fraction.partition("/")
            if not slash or not top.strip() or not bottom.strip():
                raise ValueError("Write each ratio as name=band/band, e.g. alpha/beta=alpha/beta")
            ratios.append([name.strip(), top.strip(), bottom.strip()])
        return ratios
    if kind == "names":
        return parts
    if kind == "numbers":
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise ValueError("Numbers separated by commas, e.g. -1, 0") from None
        if length and len(values) != length:
            raise ValueError("Give {} numbers".format(length))
        return values
    if kind == "bands":
        bands = []
        for part in parts:
            match = _BAND.match(part)
            if not match:
                raise ValueError("Write each band as name low-high, e.g. mu 8-13, beta 13-30")
            name, low, high = match.group(1), float(match.group(2)), float(match.group(3))
            if not 0 <= low < high:
                raise ValueError("In {} the low edge must be below the high edge".format(name))
            bands.append([name, low, high])
        if not bands:
            raise ValueError("Give at least one band")
        return bands
    if kind == "ranges":
        ranges = []
        for part in parts:
            match = _RANGE.match(part)
            if not match:
                raise ValueError("Write each range as low-high in Hz, e.g. 4-8, 8-12")
            low, high = float(match.group(1)), float(match.group(2))
            if not low < high:
                raise ValueError("In {} the low edge must be below the high edge".format(part))
            ranges.append([low, high])
        if not ranges:
            raise ValueError("Give at least one range")
        return ranges
    return text


class NumberBox(NoScrollSpinBox):
    """Shows 30 and 7.5, not 30.000 and 7.500: three trailing zeros read as thousands."""

    def textFromValue(self, value):
        return self.locale().toString(float(value), "g", 8)


#: "Low cut (Hz)": a setting's name and, in brackets at its end, a short unit.
UNIT = re.compile(r"^(.+?) \(([^()]{1,6})\)$")


class ParamField(QtWidgets.QWidget):
    """One parameter's editor. `edited` carries a valid value; a bad text turns the field red.

    A number's unit goes inside its field ("40 Hz"), out of its name: `label` is the name to show beside it."""

    edited = QtCore.Signal(str, object)

    def __init__(self, param, value, choices_for=None):
        super().__init__()
        self.param, self.kind = param, field_kind(param)
        self.label = param.label or param.name
        unit = UNIT.match(self.label) if self.kind in ("int", "float") else None
        if unit:
            self.label = unit.group(1)
        self.choices_for = choices_for
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        if self.kind == "choice":
            self.widget = short_list(NoScrollComboBox())
            self.widget.addItems([str(c) for c in param.choices])
            self.widget.setCurrentText(str(value))
            self.widget.currentTextChanged.connect(lambda text: self.edited.emit(param.name, text))
        elif self.kind == "bool":
            self.widget = QtWidgets.QCheckBox()
            self.widget.setChecked(bool(value))
            self.widget.toggled.connect(lambda on: self.edited.emit(param.name, bool(on)))
        elif self.kind == "int":
            self.widget = NoScrollIntBox()
            self.widget.setRange(-1000000, 1000000)
            self.widget.setValue(int(value))
            self.widget.valueChanged.connect(lambda v: self.edited.emit(param.name, int(v)))
        elif self.kind == "float":
            self.widget = NumberBox()
            self.widget.setRange(-100000.0, 100000.0)
            self.widget.setDecimals(6)          # a device's µV per count, 0.02235, is not 0.022
            self.widget.setValue(float(value))
            self.widget.setKeyboardTracking(False)
            self.widget.valueChanged.connect(lambda v: self.edited.emit(param.name, float(v)))
        else:
            self.widget = QtWidgets.QLineEdit(to_text(self.kind, value))
            self.widget.editingFinished.connect(self._text_done)
            if self.kind == "names":
                self.widget.setPlaceholderText("all" if param.name in ("channels", "conditions", "keep") else "")
        if unit:
            self.widget.setSuffix(" " + unit.group(2))
        row.addWidget(self.widget, 1)
        if self.kind == "path":
            browse = QtWidgets.QPushButton("Browse…")
            browse.clicked.connect(self._browse)
            row.addWidget(browse)
        if self.kind == "names" and param.name in PICKABLE and choices_for is not None:
            electrodes = param.name in ELECTRODES
            pick = QtWidgets.QPushButton("Head…" if electrodes else "Pick…")
            pick.setToolTip("Choose the electrodes on a head map (after a run)" if electrodes else
                            "Choose from the recording's own names (after a run)")
            pick.clicked.connect(self._pick)
            row.addWidget(pick)
        if param.help:
            self.setToolTip(param.help)
        self._length = len(param.default) if self.kind == "numbers" else None

    def _text_done(self):
        try:
            value = from_text(self.kind, self.widget.text(), self._length)
        except ValueError as exc:
            self.widget.setStyleSheet("border: 1px solid {};".format(color("bad")))
            self.widget.setToolTip(str(exc))
            return
        self.widget.setStyleSheet("")
        self.widget.setToolTip(self.param.help)
        self.edited.emit(self.param.name, value)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Recording", self.widget.text(), RECORDING_FILTER)
        if path:
            self.widget.setText(path)
            self._text_done()

    def _pick(self):
        options = self.choices_for(self.param.name)
        if not options:
            QtWidgets.QToolTip.showText(self.mapToGlobal(QtCore.QPoint(0, self.height())),
                                        "Run the pipeline once; then the recording's names are offered here.")
            return
        current = from_text("names", self.widget.text())
        if self.param.name in ELECTRODES:
            chosen = self.pick_on_head(options, current)
            if chosen is not None:
                self.widget.setText(", ".join(chosen))
                self._text_done()
            return
        dialog = PickDialog(self.param.label or self.param.name, options, set(current), self)
        if dialog.exec():
            self.widget.setText(", ".join(dialog.chosen()))
            self._text_done()

    def pick_on_head(self, names, current):
        """Electrodes chosen on the head map, the ones already listed first in their order; None if cancelled."""
        from ..headmap import ChannelSelectorDialog
        indices = ChannelSelectorDialog.ask(self, list(names), [names.index(n) for n in current if n in names])
        if indices is None:
            return None
        picked = [names[i] for i in indices]
        return [n for n in current if n in picked] + [n for n in picked if n not in current]


class PickDialog(QtWidgets.QDialog):
    def __init__(self, title, options, chosen, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        column = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        for option in options:
            item = QtWidgets.QListWidgetItem(option)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if option in chosen else QtCore.Qt.CheckState.Unchecked)
            self.list.addItem(item)
        column.addWidget(self.list)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)
        self.resize(280, 420)

    def chosen(self):
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.CheckState.Checked]


class Inspector(QtWidgets.QWidget):
    """Settings and result of one node. `edited(node id)` after every accepted change."""

    edited = QtCore.Signal(str)
    renamed = QtCore.Signal(str)
    #: node id of a Recording block whose MATLAB file's layout is to be chosen
    layout_requested = QtCore.Signal(str)

    def __init__(self, choices_for=None, rate_for=None, input_for=None):
        super().__init__()
        self.graph, self.node_id = None, None
        #: (node id, parameter name) -> names to pick from, or None
        self.choices_for = choices_for
        #: node id -> the sampling rate reaching it, or None before a run
        self.rate_for = rate_for
        #: (node id, input name) -> what reaches that input, from the last run, or None
        self.input_for = input_for
        self.response = self.picker = None
        #: parameter name -> its field, for the node shown
        self.fields = {}
        self.column = QtWidgets.QVBoxLayout(self)
        self.column.setContentsMargins(8, 8, 8, 8)
        self.column.setSpacing(8)
        self.body = self.result = self.error = self.result_box = self.methods = self.options_toggle = None
        #: the forms the settings are in: the main one, and the one behind Options
        self.basic_form = self.advanced_form = None
        self.error_title = ""
        #: result windows still open
        self.windows = []
        #: (graph, node id) -> output name -> what its preview had chosen: a block shown again, run again or
        #: opened larger keeps the channel, span and page the user picked
        self.chosen = {}
        #: output name -> the preview beside the settings now
        self.previews = {}
        self.show_node(None, None)

    def _keep_choices(self):
        """Remember what the previews shown now have chosen, before they are replaced."""
        if self.previews and self.graph is not None:
            chosen = {k: w.choices() for k, w in self.previews.items()}
            self.chosen[(id(self.graph), self.node_id)] = chosen
            if self.node_id in self.graph.nodes:
                # kept in the pipeline too: saved with it, back after a restart, said by its report
                self.graph.views[self.node_id] = json.loads(json.dumps(chosen))

    def _choices(self):
        """What this node's previews had chosen: this session's, else what the pipeline file kept."""
        found = self.chosen.get((id(self.graph), self.node_id))
        if found is None and self.graph is not None:
            found = self.graph.views.get(self.node_id)
        return found or {}

    def show_node(self, graph, node_id, outputs=None):
        self._keep_choices()
        self.previews = {}
        self.graph, self.node_id = graph, node_id
        if self.body is not None:
            self.body.hide()
            self.column.removeWidget(self.body)
            self.body.deleteLater()
        self.body = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(self.body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        self.column.addWidget(self.body)
        self.result = self.error = self.result_box = self.methods = self.response = self.picker = None
        self.options_toggle = self.basic_form = self.advanced_form = None
        self.fields = {}
        if graph is None or node_id not in graph.nodes:
            hint = QtWidgets.QLabel(
                "Select a block to see its settings and its result.\n\n"
                "Add blocks from the list on the left: drag one onto the sheet, or double-click it to add it "
                "after the selected block, already wired.\n\nWire a block by dragging from a port on its right "
                "to one on the left of the next. Ports of the same colour fit.")
            hint.setWordWrap(True)
            hint.setStyleSheet(muted())
            column.addWidget(hint)
            column.addStretch(1)
            return
        node = graph.nodes[node_id]
        kind = block(node.type)
        title = QtWidgets.QLabel("<b>{}</b> <span style='{}'>· {}</span>".format(kind.title, muted(), kind.category))
        title.setWordWrap(True)
        column.addWidget(title)
        # what went wrong comes first, where it is seen, with what can be done about it
        self.error = ProblemBox()
        self.error_title = node.label or kind.title
        column.addWidget(self.error)
        name = QtWidgets.QLineEdit(node.label)
        name.setPlaceholderText("Name on the sheet (optional)")
        name.editingFinished.connect(lambda: self._rename(name.text().strip()))
        column.addWidget(name)
        self.methods = QtWidgets.QLabel()
        self.methods.setWordWrap(True)
        self.methods.setStyleSheet(muted())
        column.addWidget(self.methods)

        params = graph.params(node_id)
        basic = QtWidgets.QFormLayout()
        advanced_box = QtWidgets.QWidget()
        advanced = QtWidgets.QFormLayout(advanced_box)
        advanced.setContentsMargins(0, 0, 0, 0)
        for form in (basic, advanced):
            # a narrow panel puts a setting's name above its field instead of pushing the field off the edge
            form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        for param in kind.params:
            field = ParamField(param, params[param.name],
                               (lambda name: self.choices_for(self.node_id, name)) if self.choices_for else None)
            field.edited.connect(self._edited)
            self.fields[param.name] = field
            (advanced if param.advanced else basic).addRow(field.label, field)
        column.addLayout(basic)
        if advanced.rowCount():
            toggle = QtWidgets.QToolButton()
            toggle.setText("Options ▸")
            toggle.setCheckable(True)
            advanced_box.setVisible(False)
            toggle.toggled.connect(lambda on: (advanced_box.setVisible(on),
                                               toggle.setText("Options ▾" if on else "Options ▸")))
            self.options_toggle = toggle
            column.addWidget(toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            column.addWidget(advanced_box)
        self.basic_form, self.advanced_form = basic, advanced
        self._show_applicable()
        if node.type == "load" and str(params.get("path") or "").lower().endswith(".mat"):
            layout = QtWidgets.QPushButton("How this file is read…")
            layout.setToolTip("Which variable is the signal, its sampling rate, channel names and where the events "
                              "are; kept for this file, and for files laid out the same way if you like")
            layout.clicked.connect(lambda _c=False, node=node_id: self.layout_requested.emit(node))
            column.addWidget(layout, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        if node.type in ("filter", "trial_filter"):
            self.response = FilterResponse()
            column.addWidget(self.response)
        if node.type == "select_trials":
            column.addWidget(self._trial_picker())
        self.trial_editor = None
        if node.type == "epochs":
            column.addWidget(self._trial_table(params, outputs))
        self.result_box = QtWidgets.QVBoxLayout()
        column.addLayout(self.result_box)
        column.addStretch(1)
        self._update_methods()
        self._update_response()
        self.show_outputs(outputs)

    def _trial_picker(self):
        """The trials reaching Select trials, to click the good ones; a hint before the first run."""
        trials = self.input_for(self.node_id, "epochs") if self.input_for else None
        if trials is None:
            hint = QtWidgets.QLabel("Run the pipeline once: the trials arriving here are then drawn one above the "
                                    "other, to click the ones to keep.")
            hint.setWordWrap(True)
            hint.setStyleSheet(muted())
            return hint
        self.picker = EpochsPreview(trials, kept=self._kept(trials.ids), toggled=self._toggle_trial)
        return self.picker.with_tools()

    def _trial_table(self, params, outputs):
        """The Epochs block's trial table with a timeline (trials by hand), or the way to make one from the trials
        the events or the sequence give."""
        raw = self.input_for(self.node_id, "raw") if self.input_for else None
        if params["source"] == "manual":
            from .trial_editor import TrialEditor
            self.trial_editor = TrialEditor(params["trials"], raw, (outputs or {}).get("epochs"))
            self.trial_editor.edited.connect(self._trials_edited)
            return self.trial_editor
        button = QtWidgets.QPushButton("Trials as a table…")
        button.setToolTip("Write the trials the events or the sequence give into the trial table, to add, change or "
                          "leave out trials by hand; the block then reads the table")
        button.setEnabled(raw is not None)
        if raw is None:
            button.setToolTip("Run the pipeline once: the trials are made from the recording")
        button.clicked.connect(lambda: self.make_table(raw))
        return button

    def make_table(self, raw):
        """The trials the block makes now become its trial table (source: manual), each start and end as cut."""
        from ...core.trials import Trial
        from ...pipeline import compute
        params = self.graph.params(self.node_id)
        try:
            if params["source"] == "events" and not params["event_length"]:
                epochs = compute.cut_epochs(raw, **params)
                trials = [Trial(m["id"], m["condition"], m["start"], m["end"]) for m in epochs.meta]
                if epochs.rejected:
                    self.show_error("Trials already left out ({}) are not in the table".format(
                        ", ".join(str(i) for i, _, _ in epochs.rejected[:8])))
            else:
                trials = compute.defined_trials(raw, params["source"], params["event_length"], params["trials"],
                                                params["sequence"], params["sequence_start"], params["repeats"])
        except ValueError as exc:
            self.show_error(str(exc))
            return False
        rows = [[t.id, t.condition, t.start, t.end, t.included] for t in trials]
        self.graph.set(self.node_id, source="manual", trials=rows)
        self.edited.emit(self.node_id)
        self.show_node(self.graph, self.node_id)
        return True

    def _trials_edited(self, rows):
        if "trials" in self.fields:
            self.fields["trials"].widget.setText(to_text("trials", rows))
        self._edited("trials", rows)

    def _kept(self, ids):
        """The trial ids Select trials keeps now, of `ids`; unknown numbers are ignored here."""
        from ...pipeline.blocks import trial_numbers
        params = self.graph.params(self.node_id)
        try:
            keep, leave = trial_numbers(params["keep"], ids), trial_numbers(params["leave_out"], ids)
        except ValueError:
            return set(ids)
        return {i for i in ids if (not keep or i in keep) and i not in leave}

    def _toggle_trial(self, trial):
        """A trial clicked in the picker: kept now if it was left out, and the other way round."""
        from ...pipeline.blocks import trial_text
        ids = self.picker.value.ids
        kept = self._kept(ids) ^ {trial}
        if not kept:
            self.show_error("Keep at least one trial")
            return
        self.show_error("")
        keep = [] if kept == set(ids) else trial_text(kept)
        self.graph.set(self.node_id, keep=keep, leave_out=[])
        for name, value in (("keep", keep), ("leave_out", [])):
            if name in self.fields:
                self.fields[name].widget.setText(to_text("names", value))
        self.picker.set_kept(kept)
        self._update_methods()
        self.edited.emit(self.node_id)

    def _show_applicable(self):
        """Only the settings that mean something with the others as they are: an STFT's window while the method is
        STFT, a Chebyshev's ripple for that design. A hidden one keeps its value (and the file keeps it), so
        switching back finds it as it was. Options itself goes when nothing behind it applies."""
        if self.graph is None or self.node_id not in self.graph.nodes or self.basic_form is None:
            return
        kind = block(self.graph.nodes[self.node_id].type)
        params = self.graph.params(self.node_id)
        for param in kind.params:
            field = self.fields.get(param.name)
            if field is not None:
                form = self.advanced_form if param.advanced else self.basic_form
                form.setRowVisible(field, param.applies(params))
        if self.options_toggle is not None:
            self.options_toggle.setVisible(any(p.advanced and p.applies(params) for p in kind.params))

    def _update_methods(self):
        """What the block does, in its first sentence and without citations; all of it on hover. The papers are
        the report's and Help's to cite (the report's Methods are this same text, references kept)."""
        if self.methods is not None and self.node_id in self.graph.nodes:
            kind = block(self.graph.nodes[self.node_id].type)
            text = uncited(kind.describe_in(self.graph, self.node_id))
            self.methods.setText(first_sentence(text))
            self.methods.setToolTip(text if text != first_sentence(text) else "")

    def _edited(self, name, value):
        try:
            self.graph.set(self.node_id, **{name: value})
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self._show_applicable()
        self._update_methods()
        self._update_response()
        self.edited.emit(self.node_id)
        if name == "source" and self.graph.nodes[self.node_id].type == "epochs":
            # the trial table or the way to make one follows the source: shown again after this edit returns
            QtCore.QTimer.singleShot(0, lambda node=self.node_id, graph=self.graph: self.show_node(graph, node)
                                     if self.node_id == node and self.graph is graph else None)

    def _update_response(self):
        if self.response is not None:
            fs = self.rate_for(self.node_id) if self.rate_for else None
            self.response.draw(self.graph.params(self.node_id), fs)

    def _rename(self, text):
        if self.graph.nodes[self.node_id].label != text:
            self.graph.nodes[self.node_id].label = text
            self.renamed.emit(self.node_id)

    def show_error(self, text, actions=()):
        """The block's error at the top of its settings (empty: none), with a button per (label, tooltip, action)."""
        if self.error is not None:
            self.error.show_problem(text, self.error_title, actions)

    def focus_field(self, node_id, name):
        """Bring a setting of the block shown into view, its Options opened when it is one of them."""
        field = self.fields.get(name) if node_id == self.node_id else None
        if field is None:
            return False
        if not field.isVisibleTo(self) and self.options_toggle is not None:
            self.options_toggle.setChecked(True)
        field.widget.setFocus()
        if isinstance(field.widget, QtWidgets.QLineEdit):
            field.widget.selectAll()
        scroll = self.parentWidget().parentWidget() if self.parentWidget() is not None else None
        if isinstance(scroll, QtWidgets.QScrollArea):
            scroll.ensureWidgetVisible(field)
        return True

    def set_value(self, node_id, name, value):
        """Change a setting of the block shown as if typed, then show its settings again."""
        if node_id != self.node_id or name not in self.fields:
            return False
        self._edited(name, value)
        return True

    def browse(self, node_id, name):
        """The file dialog of a path setting of the block shown."""
        field = self.fields.get(name) if node_id == self.node_id else None
        if field is not None and field.kind == "path":
            field._browse()

    def show_outputs(self, outputs):
        """The node's outputs, when they match its settings as they are now."""
        if self.result_box is None:
            return
        self._keep_choices()
        self.previews = {}
        while self.result_box.count():
            item = self.result_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not outputs:
            return
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QtWidgets.QLabel("<b>Result</b>"))
        header.addStretch(1)
        larger = only(QtWidgets.QToolButton(), "maximize-2", "Larger", size=16)
        larger.setToolTip("Open this result in a window of its own, to look at it closely")
        larger.clicked.connect(lambda: self.open_larger(outputs))
        header.addWidget(larger)
        holder = QtWidgets.QWidget()
        holder.setLayout(header)
        self.result_box.addWidget(holder)
        chosen = self._choices()
        for key, value in outputs.items():
            widget = preview(value)
            if widget is not None:
                if isinstance(widget, _Preview):
                    widget.choose(chosen.get(key))
                    self.previews[key] = widget
                if hasattr(widget, "band_kept"):
                    widget.band_kept.connect(self._add_band)
                self.result_box.addWidget(widget)
                self.result = widget

    def open_larger(self, outputs):
        node = self.graph.nodes[self.node_id]
        self._keep_choices()
        window = PreviewWindow("{} - result".format(node.label or block(node.type).title), outputs, self.window(),
                               choices=self._choices())
        window.finished.connect(lambda _code: self._larger_closed(window))
        self.windows.append(window)          # kept, so it is not collected while open
        window.show()
        return window

    def _add_band(self, band):
        """A band typed for the head maps joins the block's Bands (named apart from those already there)."""
        if self.graph is None or "bands" not in self.graph.params(self.node_id):
            return
        bands = [list(b) for b in self.graph.params(self.node_id)["bands"]]
        names = {b[0] for b in bands}
        # a band typed without a name is named by its edges: band_6_20
        base = band[0] or "band_{:g}_{:g}".format(band[1], band[2]).replace(".", "p")
        name, k = base, 2
        while name in names:
            name, k = "{}_{}".format(base, k), k + 1
        bands.append([name, float(band[1]), float(band[2])])
        if "bands" in self.fields:
            self.fields["bands"].widget.setText(to_text("bands", bands))
        self._edited("bands", bands)

    def _larger_closed(self, window):
        """What was chosen in the larger window is chosen beside the settings too, if that block is still shown."""
        if window in self.windows:
            self.windows.remove(window)
        for key, widget in window.previews.items():
            if key in self.previews and self.previews[key].value is widget.value:
                self.previews[key].choose(widget.choices())


class FilterResponse(QtWidgets.QWidget):
    """The Filter block's gain against frequency, as it will be applied, updated with every setting."""

    #: rate drawn for before a recording has been read
    DEFAULT_RATE = 250.0

    def __init__(self):
        super().__init__()
        import pyqtgraph as pg
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(muted())
        self.plot = pg.PlotWidget()
        self.plot.setBackground(plot_background())
        self.plot.setFixedHeight(170)
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left", "Gain (dB)")
        self.curve = self.plot.plot(pen=pg.mkPen(color("ink"), width=1))
        column.addWidget(self.plot)
        column.addWidget(self.note)

    def draw(self, params, fs=None):
        from ...pipeline import filters as fl
        rate = fs or self.DEFAULT_RATE
        try:
            freqs, gain = fl.response(rate, params)
        except ValueError as exc:
            self.curve.setData([], [])
            self.note.setText(str(exc))
            return
        self.curve.setData(freqs, np.maximum(gain, -100))
        self.plot.setYRange(-100, 5)
        self.plot.setXRange(0, rate / 2)
        self.note.setText("Response at {:g} Hz{}, as applied ({}).".format(
            rate, "" if fs else " (no recording read yet)",
            "zero-phase: the gain counts twice" if params.get("phase") == "zero" and params.get("design") != "fir"
            else "once"))
