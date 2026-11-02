"""Compare: several pipelines over many recordings, side by side.

Left, the pipelines: the sheet as it is, saved pipelines and templates; right,
the recordings. Each pipeline runs on each recording with the same folds, so
the recordings pair up: the table shows every recording's accuracy under every
pipeline, and the Wilcoxon test asks whether one pipeline beats another across
recordings, which two separate means cannot tell.

Runs in the background; Stop ends it after the recording in hand. Results save
as CSV or as a report.
"""
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from ...pipeline import Graph, batch, templates
from ..preferences import data_dir, setting, user_settings
from ..theme import color, muted
from ..icons import only, put
from ..widgets import NoScrollComboBox
from ...pipeline import literature
from .. import recent
from ..jobs import activity, busy
from .page import library_menu, ready_made
from .report import ask_path, comparison_plot, comparison_report, saved_text

#: How a pipeline is judged on the recordings, as offered in the window
EVALUATIONS = (("Within each recording", batch.WITHIN),
               ("Across recordings (train on the others)", batch.ACROSS))

#: Files a folder is searched for when it is added.
RECORDING_PATTERNS = ("*.mat", "*.edf", "*.bdf", "*.gdf", "*.vhdr", "*.fif", "*.set", "*.csv")


class CompareJob(QtCore.QRunnable):
    class Signals(QtCore.QObject):
        progress = QtCore.Signal(str)
        done = QtCore.Signal(object, str)

    def __init__(self, graphs, paths, evaluation=batch.WITHIN):
        super().__init__()
        self.graphs, self.paths, self.evaluation = graphs, paths, evaluation
        self.stopping = False
        self.signals = CompareJob.Signals()

    def run(self):
        try:
            run = batch.across if self.evaluation == batch.ACROSS else batch.compare
            result = run(self.graphs, self.paths, progress=self.signals.progress.emit, stop=lambda: self.stopping)
            self.signals.done.emit(result, "")
        except Exception as exc:              # reported in the window, never lost
            self.signals.done.emit(None, "{}: {}".format(type(exc).__name__, exc))


class CompareWindow(QtWidgets.QDialog):
    finished_run = QtCore.Signal()

    def __init__(self, sheet=None, parent=None, tabs=()):
        """`sheet`: the pipeline in front, added at once; `tabs`: the Workbench's other tabs, to add with one
        button."""
        super().__init__(parent)
        self.tabs = [g for g in tabs]
        self.setWindowTitle("Compare pipelines")
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)
        self.graphs = []              # (Graph, where it came from)
        self.result = None
        self.job = None

        # Three steps side by side, read left to right (GIOVA, 31 October 2026: the old window's two framed lists
        # and a row of buttons were hard to follow): 1 the pipelines, 2 the recordings, 3 how and go. Adding and
        # removing are a + menu and a − beside each step's name; Compare turns into Stop while it runs.
        self.pipelines = QtWidgets.QListWidget()
        self.recordings = QtWidgets.QListWidget()
        self.recordings.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        add_pipeline = only(QtWidgets.QToolButton(), "plus", "Add a pipeline",
                            more="This sheet's other tabs, a saved pipeline, a template or a method from the "
                                 "literature")
        add_pipeline.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        add_pipeline.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")
        menu = QtWidgets.QMenu(add_pipeline)
        self.open_tabs_action = menu.addAction("The other open tabs", lambda: self.add_open_tabs())
        self.open_tabs_action.setEnabled(bool(self.tabs))
        menu.addAction("A saved pipeline…", lambda: self.add_saved())
        menu.addSeparator()
        ready = menu.addMenu("A template")
        for title in templates.TEMPLATES:
            ready.addAction(title, lambda t=title: self.add_pipeline(templates.TEMPLATES[t](), "template"))
        library_menu(menu, lambda t: self.add_pipeline(ready_made(t), "literature"),
                     lambda: [self.add_pipeline(build(), "literature") for build in literature.LIBRARY.values()])
        add_pipeline.setMenu(menu)
        drop = only(QtWidgets.QToolButton(), "minus", "Remove the pipeline selected")
        drop.clicked.connect(self._remove_pipeline)
        add_recordings = only(QtWidgets.QToolButton(), "plus", "Add recordings", more="Files, or every recording "
                              "in a folder")
        add_recordings.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        add_recordings.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")
        files = QtWidgets.QMenu(add_recordings)
        files.addAction("Files…", lambda: self.add_files())
        files.addAction("Every recording in a folder…", lambda: self.add_folder())
        add_recordings.setMenu(files)
        clear = only(QtWidgets.QToolButton(), "minus", "Remove the recordings selected")
        clear.clicked.connect(self._remove_recordings)

        self.evaluation = NoScrollComboBox()
        for label, key in EVALUATIONS:
            self.evaluation.addItem(label, key)
        self.evaluation.setToolTip("Within: each recording cross-validated on its own trials. Across: each "
                                   "recording tested with a model trained on all the others - does it carry "
                                   "over to a recording it never saw? Usually much harder. Each file counts as "
                                   "one recording: two files of one person are not kept apart.")
        self.run_button = put(QtWidgets.QPushButton(), "play", "Compare", primary=True)
        self.run_button.setProperty("primary", True)
        self.run_button.clicked.connect(lambda: self.stop() if self.job is not None else self.run())
        self.stop_button = QtWidgets.QPushButton("Stop")          # the same as Compare while it runs; not shown
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.hide()
        self.state = QtWidgets.QLabel("Pick the pipelines and the recordings, then Compare: every pipeline runs on "
                                      "every recording with the same folds.")
        self.state.setWordWrap(True)
        self.state.setStyleSheet(muted())
        self.save_csv = only(QtWidgets.QToolButton(), "download", "Save the table as CSV")
        self.save_csv.clicked.connect(lambda: self.save_table())
        self.save_report = only(QtWidgets.QToolButton(), "file-text", "Report",
                                more="The comparison as PDF, Word, web page or figures")
        self.save_report.clicked.connect(lambda: self.save_report_as())
        for b in (self.save_csv, self.save_report):
            b.setEnabled(False)

        def step(number, title, tools, body, note=""):
            box = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(box)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(4)
            head = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel("<span style='color:{}'>{}</span>&nbsp;&nbsp;{}".format(
                color("accent"), number, title))
            name.setStyleSheet("font-weight: 600;")
            head.addWidget(name, 1)
            for tool in tools:
                head.addWidget(tool)
            column.addLayout(head)
            if isinstance(body, QtWidgets.QLayout):
                column.addLayout(body, 1)
            else:
                column.addWidget(body, 1)
            if note:
                hint = QtWidgets.QLabel(note)
                hint.setWordWrap(True)
                hint.setStyleSheet(muted())
                column.addWidget(hint)
            return box

        go = QtWidgets.QVBoxLayout()
        go.setSpacing(8)
        go.addWidget(self.evaluation)
        go.addWidget(self.run_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        go.addWidget(self.state)
        go.addStretch(1)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(16)
        top.addWidget(step("1", "Pipelines", (add_pipeline, drop), self.pipelines,
                           "Each needs a Recording block (its file is swapped for each recording) and a "
                           "Cross-validate block."), 3)
        top.addWidget(step("2", "Recordings", (add_recordings, clear), self.recordings,
                           "The files each pipeline runs on: + adds files, or every recording in a folder."), 3)
        top.addWidget(step("3", "Compare", (self.save_csv, self.save_report), go), 2)

        self.results = QtWidgets.QWidget()
        self.results_column = QtWidgets.QVBoxLayout(self.results)
        self.results_column.setContentsMargins(0, 0, 0, 0)
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.results)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top, 2)
        layout.addWidget(area, 3)
        self.resize(1100, 820)
        if sheet is not None:
            self.add_pipeline(sheet.copy(), "this sheet")
        for path in str(setting("compare/recordings", "") or "").split("\n"):
            if path and Path(path).is_file():
                self._add_recording(path)

    # ---------------- pipelines ----------------

    def add_pipeline(self, graph, origin):
        names = {g.name for g, _ in self.graphs}
        base, n = graph.name, 2
        while graph.name in names:
            graph.name = "{} ({})".format(base, n)
            n += 1
        try:
            batch.source_node(graph)
            if not batch.score_nodes(graph):
                raise ValueError("'{}' has no Cross-validate block".format(graph.name))
        except ValueError as exc:
            self.state.setText(str(exc))
            return False
        self.graphs.append((graph, origin))
        self.pipelines.addItem("{}   ·   {}".format(graph.name, origin))
        self._list_changed()
        return True

    def add_open_tabs(self):
        """Every other tab's pipeline that can be compared; those that cannot are left out quietly."""
        added = [g.name for g in self.tabs if self._comparable(g) and self.add_pipeline(g.copy(), "tab")]
        self.state.setText("Added {} tab{}".format(len(added), "s" if len(added) != 1 else "") if added else
                           "No other tab has a Recording and a Cross-validate block")
        return added

    @staticmethod
    def _comparable(graph):
        try:
            batch.source_node(graph)
        except ValueError:
            return False
        return bool(batch.score_nodes(graph))

    def add_saved(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Add pipeline", str(data_dir()),
                                                            "Pipelines (*.pipeline.json)")
            if not path:
                return False
        try:
            return self.add_pipeline(Graph.load(path), Path(path).name)
        except ValueError as exc:
            self.state.setText(str(exc))
            return False

    def _remove_pipeline(self):
        row = self.pipelines.currentRow()
        if row >= 0:
            self.pipelines.takeItem(row)
            del self.graphs[row]
            self._list_changed()

    def _list_changed(self):
        """After a run, the table and its report stay those of that run: say so, so nobody takes them for the
        pipelines listed now."""
        if self.result is not None:
            self.state.setText("The list has changed: the results, CSV and report are still those of the last run "
                               "({}). Run again to compare the pipelines listed now.".format(
                                   ", ".join(g.name for g in self.result.graphs)))

    # ---------------- recordings ----------------

    def _add_recording(self, path):
        path = str(Path(path))
        if path not in self.paths():
            self.recordings.addItem(path)

    def paths(self):
        return [self.recordings.item(i).text() for i in range(self.recordings.count())]

    def add_files(self, paths=None):
        if paths is None:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add recordings", str(data_dir()),
                "Recordings ({});;All files (*)".format(" ".join(RECORDING_PATTERNS)))
        for path in paths:
            self._add_recording(path)
        self._remember()

    def add_folder(self, folder=None):
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add every recording in a folder",
                                                                str(data_dir()))
            if not folder:
                return
        found = sorted({p for pattern in RECORDING_PATTERNS for p in Path(folder).glob(pattern)
                        if not p.name.endswith((".events.csv", ".hardware.tsv"))})
        for path in found:
            self._add_recording(path)
        self.state.setText("{} recordings added from {}".format(len(found), Path(folder).name))
        self._remember()

    def _remove_recordings(self):
        for item in self.recordings.selectedItems():
            self.recordings.takeItem(self.recordings.row(item))
        self._remember()

    def _remember(self):
        user_settings().setValue("compare/recordings", "\n".join(self.paths()))

    # ---------------- running ----------------

    def run(self):
        if not self.graphs or not self.paths():
            self.state.setText("Add at least one pipeline and one recording")
            return False
        self.job = CompareJob([g.copy() for g, _ in self.graphs], self.paths(), self.evaluation.currentData())
        self.job.signals.progress.connect(self.state.setText)
        self.job.signals.done.connect(self._done)
        put(self.run_button, "square", "Stop", primary=True)      # the same button stops it
        self.stop_button.setEnabled(True)
        activity().begin()
        QtCore.QThreadPool.globalInstance().start(self.job)
        return True

    def stop(self):
        if self.job is not None:
            self.job.stopping = True
            self.state.setText("Stopping after this recording…")

    def _done(self, result, error):
        activity().end()
        put(self.run_button, "play", "Compare", primary=True)
        self.stop_button.setEnabled(False)
        self.job = None
        if error:
            self.state.setText(error)
        else:
            self.result = result
            failed = len(result.errors)
            self.state.setText("{} recordings × {} pipelines{}".format(
                len(result.subjects), len(result.columns), " · {} failed".format(failed) if failed else ""))
            self.show_result(result)
        self.finished_run.emit()

    def show_result(self, result):
        while self.results_column.count():
            item = self.results_column.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        better = result.better_counts()
        self.results_column.addWidget(_table(("Pipeline", "Mean", "SD", "Median", "Above chance", "Recordings"),
                                             [(n, "{:.1%}".format(m), "{:.1%}".format(s) if np.isfinite(s) else "-",
                                               "{:.1%}".format(md), "{} of {}".format(better[j], k), k)
                                              for j, (n, m, s, md, k) in enumerate(result.summary())]))
        # one measure at a time for every recording, chosen here: the plot and the table below show it
        measures = [(title, key) for title, key in batch.MEASURES
                    if key == batch.ACCURACY or np.isfinite(result.values(key)).any()]
        self.measure = NoScrollComboBox()
        for title, key in measures:
            self.measure.addItem(title, key)
        self.measure.setToolTip("The measure the plot and the table of recordings show; each is the mean of the "
                                "folds' own values. The * stays accuracy's verdict against chance.")
        chooser = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(chooser)
        row.setContentsMargins(0, 6, 0, 0)
        row.addWidget(QtWidgets.QLabel("Measure"))
        row.addWidget(self.measure)
        row.addStretch(1)
        self.results_column.addWidget(chooser)
        self.plot = comparison_plot(result) if result.subjects else QtWidgets.QWidget()
        self.results_column.addWidget(self.plot)
        # each recording's value right under the plot that draws it; the paired tests after
        note = QtWidgets.QLabel("* above chance: accuracy above the limit a random classifier reaches with that "
                                "recording's tested trials (alpha 0.05). Double-click a recording for its full "
                                "result: folds, confusion matrix, ROC curve")
        note.setToolTip("The binomial limit of a random classifier: Müller-Putz et al. (2008); Combrisson & Jerbi "
                        "(2015). It takes the tested trials as independent, which cross-validated ones are not, "
                        "so it is approximate (Noirhomme et al., 2014)")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        self.results_column.addWidget(note)
        self.per_recording = self._recordings(result, batch.ACCURACY)
        self.results_column.addWidget(self.per_recording)
        self.measure.currentIndexChanged.connect(lambda _i: self.show_measure(self.measure.currentData()))
        paired = result.paired_holm()
        if paired:
            label = QtWidgets.QLabel("Paired, per recording, on accuracy: mean difference and Wilcoxon signed-rank "
                                     "p; Holm: adjusted for testing every pair")
            label.setStyleSheet(muted())
            self.results_column.addWidget(label)
            shown = lambda v: "{:.3g}".format(v) if np.isfinite(v) else "-"
            self.results_column.addWidget(_table(("A", "B", "A - B", "p", "p (Holm)", "Recordings"),
                                                 [(a, b, "{:+.1%}".format(d), shown(p), shown(q), n)
                                                  for a, b, d, p, q, n in paired]))
        for (subject, pipeline), message in result.errors.items():
            label = QtWidgets.QLabel("{} · {}: {}".format(subject, pipeline, message))
            label.setWordWrap(True)
            self.results_column.addWidget(label)
        self.results_column.addStretch(1)
        for b in (self.save_csv, self.save_report):
            b.setEnabled(True)

    def _recordings(self, result, measure):
        """Every recording's value of `measure` under every pipeline, then their mean and SD."""
        names = [result.column_name(c) for c in result.columns]
        values = result.values(measure)
        rows = [[s] + [result.cell(i, j, measure) for j in range(len(names))] for i, s in enumerate(result.subjects)]
        for title, summary in (("Mean", np.nanmean), ("SD", lambda v: np.nanstd(v, ddof=1))):
            line = [title]
            for j in range(len(names)):
                finite = values[:, j][np.isfinite(values[:, j])]
                enough = finite.size > (1 if title == "SD" else 0)
                line.append(batch.shown(float(summary(finite)), measure) if enough else "-")
            rows.append(line)
        table = _table(["Recording"] + names, rows)
        for k in (len(rows) - 2, len(rows) - 1):                 # the summary rows set apart, in bold
            for j in range(table.columnCount()):
                item = table.item(k, j)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
        table.cellDoubleClicked.connect(self.open_recording)
        return table

    def show_measure(self, measure):
        """The plot and the recordings' table, again, for another measure."""
        if self.result is None:
            return
        for old, new in ((self.plot, comparison_plot(self.result, measure) if self.result.subjects else None),
                         (self.per_recording, self._recordings(self.result, measure))):
            if new is None:
                continue
            self.results_column.replaceWidget(old, new)
            old.deleteLater()
            if old is self.plot:
                self.plot = new
            else:
                self.per_recording = new

    def open_recording(self, row, column=0):
        """One recording's full result in a window: every pipeline's (the recording's name double-clicked) or
        one pipeline's (its cell) - the same view as the Cross-validate block's result on the sheet."""
        from .previews import PreviewWindow
        result = self.result
        if result is None or not 0 <= row < len(result.subjects):
            return None
        subject = result.subjects[row]
        columns = range(len(result.columns)) if column == 0 else [column - 1]
        outputs = {result.column_name(result.columns[j]): result.scores[(subject, result.columns[j])]
                   for j in columns if (subject, result.columns[j]) in result.scores}
        if not outputs:
            self.state.setText("{}: no result (it failed)".format(subject))
            return None
        window = PreviewWindow("{} - {}".format(subject, ", ".join(outputs)), outputs, self)
        titles = list(outputs)
        if len(titles) > 1:
            # several pipelines in one window: each result under its pipeline's name
            body = window.findChild(QtWidgets.QScrollArea).widget().layout()
            for k, title in enumerate(titles):
                label = QtWidgets.QLabel("<b>{}</b>".format(title))
                body.insertWidget(body.indexOf(window.previews[title]), label)
        self._windows = [w for w in getattr(self, "_windows", []) if w.isVisible()] + [window]
        window.show()
        return window

    def save_table(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save table", str(Path(data_dir()) / "compare.csv"),
                                                            "CSV (*.csv)")
        if path:
            self.result.save_csv(path)
        return path

    def save_report_as(self, path=None):
        if path is None:
            path = ask_path(self, Path(data_dir()) / "compare.pdf")
        if path:
            with busy():
                saved = comparison_report(self.result).save(path)
            recent.remember(recent.REPORTS, saved[0] if isinstance(saved, list) else saved, name="Comparison")
            self.state.setText(saved_text(saved))
        return path

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


def _table(header, rows):
    table = QtWidgets.QTableWidget(len(rows), len(header))
    table.setHorizontalHeaderLabels([str(h) for h in header])
    table.verticalHeader().setVisible(False)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            item = QtWidgets.QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, j, item)
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    # every row in view: the height of the rows as this screen's font draws them, not a guess (a fixed
    # 26 px a row showed the first pipeline alone with Windows fonts, the others behind a scroll bar)
    height = table.horizontalHeader().sizeHint().height() + table.verticalHeader().length() + 2 * table.frameWidth()
    table.setFixedHeight(min(420, height + 2))
    return table
