"""Small feature selection, editable profiles, and linked trial distributions."""
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..core.feature_profiles import (FeatureProfile, band_power_profile, load_profile,
                                     save_profile, summary, cliffs_delta, condition_values, time_features)
from .widgets import combo_box, fit_to_screen, spin_box
from .jobs import Jobs
from .plots import value_axis
from .theme import CONDITIONS, plot_background


def fmt(value):
    """A value as the table shows it; a dash when it could not be computed (as tables in papers, APA)."""
    return "—" if not np.isfinite(value) else f"{value:.4g}"


class ProfileDialog(QtWidgets.QDialog):
    def __init__(self, parent, profile):
        super().__init__(parent)
        self.setWindowTitle("Feature profile: edit a copy")
        fit_to_screen(self, 650, 570)
        self.profile = deepcopy(profile)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit()
        self.low = spin_box(0, 1e5, 1, .5, 2)
        self.high = spin_box(.01, 1e5, 45, .5, 2)
        self.window = spin_box(.01, 600, 2, .5, 2)
        self.step = spin_box(.001, 600, 1, .25, 3)
        form.addRow("Profile name", self.name)
        row = QtWidgets.QHBoxLayout()
        for label, box in (("Total from (Hz)", self.low), ("to", self.high)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(box)
        form.addRow(row)
        row = QtWidgets.QHBoxLayout()
        for label, box in (("Time window (s)", self.window), ("Step (s)", self.step)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(box)
        form.addRow(row)
        layout.addLayout(form)
        tabs = QtWidgets.QTabWidget()
        self.bands = QtWidgets.QTableWidget(0, 3)
        self.bands.setHorizontalHeaderLabels(["Name", "Low Hz", "High Hz"])
        self.ratios = QtWidgets.QTableWidget(0, 3)
        self.ratios.setHorizontalHeaderLabels(["Name", "Numerator band", "Denominator band"])
        for title, table in (("Bands", self.bands), ("Ratios", self.ratios)):
            page = QtWidgets.QWidget()
            box = QtWidgets.QVBoxLayout(page)
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            box.addWidget(table)
            row = QtWidgets.QHBoxLayout()
            add = QtWidgets.QPushButton("Add row")
            add.clicked.connect(lambda checked=False, t=table: t.insertRow(t.rowCount()))
            remove = QtWidgets.QPushButton("Remove selected row")
            remove.clicked.connect(lambda checked=False, t=table: t.removeRow(t.currentRow()))
            row.addWidget(add)
            row.addWidget(remove)
            box.addLayout(row)
            tabs.addTab(page, title)
        self.choices = QtWidgets.QListWidget()
        tabs.addTab(self.choices, "Visible features")
        layout.addWidget(tabs, 1)
        row = QtWidgets.QHBoxLayout()
        for text, action in (("Refresh feature choices", self.refresh_choices),
                             ("Load profile…", self.load), ("Save profile copy…", self.save)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(action)
            row.addWidget(button)
        layout.addLayout(row)
        self.note = QtWidgets.QLabel("Ratios reference band names. Refresh choices after changing band names.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.fill(profile)

    def fill(self, profile):
        self.name.setText(profile.name)
        self.low.setValue(profile.total_range[0])
        self.high.setValue(profile.total_range[1])
        self.window.setValue(profile.time_window)
        self.step.setValue(profile.time_step)
        for table, rows in ((self.bands, profile.bands), (self.ratios, profile.ratios)):
            table.setRowCount(len(rows))
            for r, values in enumerate(rows):
                for c, value in enumerate(values):
                    table.setItem(r, c, QtWidgets.QTableWidgetItem(str(value)))
        self._fill_choices(profile)

    def _fill_choices(self, profile):
        self.choices.clear()
        for spec in profile.catalog():
            item = QtWidgets.QListWidgetItem(f"{spec.name} ({spec.unit})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, spec.id)
            item.setCheckState(QtCore.Qt.CheckState.Checked if spec.id in profile.selected
                               else QtCore.Qt.CheckState.Unchecked)
            item.setToolTip(spec.description)
            self.choices.addItem(item)

    def read(self, validate=True):
        def rows(table):
            return [[table.item(r, c).text().strip() if table.item(r, c) else ""
                     for c in range(3)] for r in range(table.rowCount())]
        bands = [[name, float(lo.replace(",", ".")), float(hi.replace(",", "."))] for name, lo, hi in rows(self.bands)]
        chosen = [self.choices.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.choices.count())
                  if self.choices.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        profile = FeatureProfile(self.name.text().strip(), bands, rows(self.ratios), chosen,
                                 [self.low.value(), self.high.value()], self.window.value(), self.step.value())
        if validate:
            profile.validate()
        return profile

    def refresh_choices(self):
        try:
            profile = self.read(False)
            valid = {s.id for s in profile.catalog()}
            profile.selected = [key for key in profile.selected if key in valid] or ["rms"]
            profile.validate()
            self._fill_choices(profile)
            self.note.setText("Choose the measurements to display; unavailable frequency ranges will be explained.")
        except (ValueError, TypeError) as exc:
            self.note.setText(str(exc))

    def accept(self):
        try:
            self.profile = self.read()
            super().accept()
        except (ValueError, TypeError) as exc:
            self.note.setText(str(exc))

    def load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Feature profile", "", "JSON (*.json)")
        if path:
            try:
                self.fill(load_profile(path))
            except (ValueError, OSError) as exc:
                self.note.setText(str(exc))

    def save(self):
        try:
            profile = self.read()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save profile copy", "feature-profile.json",
                                                            "JSON (*.json)")
            if path:
                save_profile(path, profile)
                self.note.setText("Profile copy saved")
        except (ValueError, OSError) as exc:
            self.note.setText(str(exc))


class FeatureView(QtWidgets.QWidget):
    profile_changed = QtCore.Signal(object)
    trial_requested = QtCore.Signal(str)
    detail_opened = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.profile = FeatureProfile()
        self.result = None
        self.time_rows = None
        self.key = None
        self.jobs = Jobs(self, busy=True)
        self.jobs.finished.connect(self._time_done)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.presets = combo_box(["Eyes open / closed", "Band powers", "Custom"])
        self.presets.currentIndexChanged.connect(self._preset)
        row.addWidget(self.presets)
        edit = QtWidgets.QPushButton("Choose features…")
        edit.clicked.connect(self.edit_profile)
        row.addWidget(edit)
        self.source = combo_box(["Trial comparison", "Mean signal values"])
        self.source.currentIndexChanged.connect(self.render)
        row.addWidget(self.source)
        self.a = combo_box()
        self.b = combo_box()
        for label, combo in (("A", self.a), ("B", self.b)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(combo)
            combo.currentIndexChanged.connect(self.render)
        row.addStretch()
        layout.addLayout(row)
        splitter = self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Feature", "Unit", "A mean ± SD", "B mean ± SD", "B − A", "n A / B",
                                              "Cliff’s δ", "Availability"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self.select_row)
        splitter.addWidget(self.table)
        self.detail = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(self.detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.time_toggle = QtWidgets.QCheckBox("Time detail")
        self.time_toggle.setToolTip("Change window and step in Choose features → profile settings.")
        self.time_toggle.toggled.connect(self.draw_detail)
        row.addWidget(self.time_toggle)
        close = QtWidgets.QPushButton("Hide detail")
        close.clicked.connect(self.detail.hide)
        row.addStretch()
        row.addWidget(close)
        detail_layout.addLayout(row)
        self.plot = pg.PlotWidget()
        self.plot.addLegend(colCount=4, offset=(10, 5))
        self.plot.setMinimumHeight(160)
        detail_layout.addWidget(self.plot, 1)
        self.detail_note = QtWidgets.QLabel()
        self.detail_note.setWordWrap(True)
        detail_layout.addWidget(self.detail_note)
        splitter.addWidget(self.detail)
        splitter.setSizes([220, 250])
        self.detail.hide()
        self.note = QtWidgets.QLabel("Select a feature row to inspect its trial distribution.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

    def set_profile(self, profile):
        """Adopt a feature profile chosen elsewhere (an analysis profile)."""
        self.profile = profile
        index = 0 if profile == FeatureProfile() else 1 if profile == band_power_profile() else 2
        self.presets.blockSignals(True)
        self.presets.setCurrentIndex(index)
        self.presets.blockSignals(False)
        self.profile_changed.emit(self.profile)

    def _preset(self, index):
        if index < 2:
            self.profile = FeatureProfile() if index == 0 else band_power_profile()
            self.profile_changed.emit(self.profile)

    def edit_profile(self):
        dialog = ProfileDialog(self, self.profile)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.profile = dialog.profile
            self.presets.blockSignals(True)
            self.presets.setCurrentIndex(2)
            self.presets.blockSignals(False)
            self.profile_changed.emit(self.profile)

    def clear(self):
        self.jobs.invalidate()
        self.result = None
        self.time_rows = None
        self.table.setRowCount(0)
        self.plot.clear()
        self.detail.hide()
        self.note.setText("Analyze the current selection to compare features.")

    def set_result(self, result):
        self.clear()
        self.result = result
        self.profile = result.features.profile
        index = 0 if self.profile == FeatureProfile() else 1 if self.profile == band_power_profile() else 2
        self.presets.blockSignals(True)
        self.presets.setCurrentIndex(index)
        self.presets.blockSignals(False)
        names = [c.name for c in result.conditions]
        for combo, index in ((self.a, 0), (self.b, min(1, len(names) - 1))):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self.render()

    def render(self):
        if self.result is None:
            return
        specs = self.profile.visible_specs()
        self.table.setRowCount(len(specs))
        a, b = self.a.currentText(), self.b.currentText()
        means_only = self.source.currentIndex() == 1
        for row, spec in enumerate(specs):
            va, vb = condition_values(self.result, a, spec.id), condition_values(self.result, b, spec.id)
            sa, sb = summary(va), summary(vb)
            if means_only:
                ma = self.result.features.mean_signals[a].values[spec.id]
                mb = self.result.features.mean_signals[b].values[spec.id]
                cells = [spec.name, spec.unit, fmt(ma), fmt(mb), fmt(mb - ma), "1 mean / 1 mean", "",
                         "Mean signal only"]
            else:
                cells = [spec.name, spec.unit, f"{fmt(sa['mean'])} ± {fmt(sa['sd'])}",
                         f"{fmt(sb['mean'])} ± {fmt(sb['sd'])}", fmt(sb['mean'] - sa['mean']),
                         f"{sa['n']} / {sb['n']}", fmt(cliffs_delta(va, vb)) if a != b else "",
                         "" if min(sa['n'], sb['n']) >= 3 else "Need 3 trials/condition for δ"]
            reasons = sorted({v.reasons[spec.id] for v in self.result.features.trials.values() if spec.id in v.reasons})
            if reasons:
                cells[-1] = "; ".join(reasons)
            for col, value in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(spec.description + ("\n" + "; ".join(reasons) if reasons else ""))
                self.table.setItem(row, col, item)
        self.time_toggle.setEnabled(not means_only)
        self.note.setText(f"Profile: {self.profile.name} · {self.result.channel} · " +
                          ("Mean signal values have no trial distribution or separation score." if means_only else
                           "Each point is one trial. Cliff’s δ describes B relative to A; it is not accuracy or a "
                           "significance test."))
        if self.detail.isVisible():
            self.draw_detail()

    def select_row(self, row, column=0):
        self.key = self.profile.selected[row]
        self.detail.show()
        self.detail_opened.emit()
        self.splitter.setSizes([125, 260])
        self.draw_detail()

    def draw_detail(self):
        if self.result is None or self.key not in self.profile.selected:
            return
        self.plot.clear()
        self.plot.enableAutoRange()
        spec = next(s for s in self.profile.catalog() if s.id == self.key)
        value_axis(self.plot, spec.name, spec.unit)
        self.plot.setTitle(spec.name)
        if self.source.currentIndex() == 1:
            self.detail_note.setText("Switch to Trial comparison to inspect individual observations.")
            return
        if self.time_toggle.isChecked():
            if self.time_rows is None:
                self.detail_note.setText("Computing time detail…")
                result, profile = self.result, deepcopy(self.profile)
                self.jobs.start(lambda: time_features(result, profile))
                return
            self.plot.getAxis("bottom").setTicks(None)
            self.plot.setLabel("bottom", "Time from trial start", units="s")
            for t in self.result.trials:
                if t.trial.condition not in (self.a.currentText(), self.b.currentText()):
                    continue
                times, rows = self.time_rows[t.trial.id]
                color = CONDITIONS[0] if t.trial.condition == self.a.currentText() else CONDITIONS[1]
                self.plot.plot(times, [r.values[self.key] for r in rows], pen=pg.mkPen(color, width=1),
                               name=f"{t.trial.condition}/{t.trial.id}")
            self.detail_note.setText(f"{self.profile.time_window:g} s windows / {self.profile.time_step:g} s step. "
                                     "Window centers are shown. Incomplete tails are omitted; trial sample counts "
                                     "stay unchanged.")
            return
        self.plot.setLabel("bottom", "Condition (click a point to inspect its trial)")
        names = list(dict.fromkeys([self.a.currentText(), self.b.currentText()]))
        self.plot.getAxis("bottom").setTicks([list(enumerate(names))])
        notes = []
        for index, name in enumerate(names):
            entries = [t for t in self.result.trials if t.trial.condition == name]
            values = [self.result.features.trials[t.trial.id].values[self.key] for t in entries]
            stats = summary(values)
            color = CONDITIONS[index % 2]
            points = [dict(pos=(index + (j - (len(entries)-1)/2)*.035, value), data=t.trial.id)
                      for j, (t, value) in enumerate(zip(entries, values)) if np.isfinite(value)]
            scatter = pg.ScatterPlotItem(points, size=10, brush=color, pen=plot_background())
            scatter.sigClicked.connect(lambda obj, pts, event: self.trial_requested.emit(pts[0].data()))
            self.plot.addItem(scatter)
            if stats["n"]:
                self.plot.plot([index-.16, index+.16], [stats["median"]]*2, pen=pg.mkPen(color, width=3))
                self.plot.plot([index, index], [stats["q1"], stats["q3"]], pen=pg.mkPen(color, width=6))
            notes.append(f"{name}: n={stats['n']}, median {fmt(stats['median'])}, "
                         f"IQR {fmt(stats['q1'])}–{fmt(stats['q3'])}")
        self.plot.setXRange(-.4, max(len(names)-.6, .4))
        self.detail_note.setText(" | ".join(notes) + ". Bars show median and IQR; points are individual trials.")

    def _time_done(self, result, error):
        if error:
            self.detail_note.setText(error)
            return
        self.time_rows = result
        self.draw_detail()
