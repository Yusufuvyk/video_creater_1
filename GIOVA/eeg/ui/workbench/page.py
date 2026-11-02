"""The Workbench page: build an analysis from blocks, run it, look at each step.

Left: the blocks, by category, and beside them the Library of pipelines to start
from (library.py). Middle: the sheet. Right: the selected block's
settings and result. The sheet is kept between sessions as it was left; Save
writes it to a `.pipeline.json` that anyone with GIOVA can open.

Running happens in the background on a copy of the pipeline, so the sheet can
be edited meanwhile. Outputs are kept by the engine's keys: a block shows its
result for exactly the settings that produced it, and turns amber as soon as
something before it changes.

Several sheets can be open, as tabs (sheets.py): each its own pipeline, file and
run, one cache for all of them. The settings, the bar and the Run button are
for the tab in front (or the sheet whose own window was clicked last); the
page's `graph`, `canvas`, `path` and `modified` speak for that one. At most
MAX_RUNNING sheets compute at once, the others wait in line; Run all queues
every tab.
"""
import json
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ...pipeline import Graph, PipelineError, REGISTRY, Runner, block
from ...pipeline.graph import Stopped
from ...pipeline import blocks as pb
from ...pipeline import templates
from .. import recent
from ..jobs import activity
from ..preferences import data_dir, setting, user_settings
from ..theme import color, muted, soften
from ..widgets import ElidedLabel, WrapRow, group
from ..icons import only
from .canvas import BLOCK_MIME, CATEGORY_COLOURS, CATEGORY_ICONS, BlockPreview, Canvas
from .inspector import Inspector
from .library import Library
from .previews import result_line
from . import problems
from .sheets import (IDLE, MAX_RUNNING, QUEUED, RUNNING, STOPPED, STOPPING, ERROR, DONE, RunStrip, Sheet,
                     SheetTabBar, SheetWindow, clock, run_or_stop, show_run_button)

#: The template a first start opens with.
FIRST_TEMPLATE = "Band power per condition"
SUFFIX = ".pipeline.json"


def ready_made(title):
    """A template or a literature profile by its menu title, freshly built."""
    from ...pipeline import literature
    return (templates.TEMPLATES.get(title) or literature.LIBRARY[title])()


def library_menu(menu, chosen, everything=None):
    """'From the literature ▸' in `menu`: each profile, its paper as the hover text; `everything`, if
    given, adds an entry that takes all of them (for Compare)."""
    from ...pipeline import literature
    menu.addSeparator()
    sub = menu.addMenu("From the literature")
    sub.setToolTipsVisible(True)
    for title, build in literature.LIBRARY.items():
        action = sub.addAction(title, lambda t=title: chosen(t))
        action.setToolTip(build().source["citation"])
    if everything is not None:
        sub.addSeparator()
        sub.addAction("All of them", everything)
    return sub


class Palette(QtWidgets.QTreeWidget):
    """Every block, by category. Drag one onto the sheet, or double-click to add it; pointing at one shows it
    in the preview below the list (`hovered`: its type, "" when the mouse leaves)."""

    chosen = QtCore.Signal(str)
    hovered = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setIndentation(12)
        groups = {}
        for category in pb.CATEGORIES:
            parent = self._heading(category)
            group = pb.GROUPS.get(category)
            if group is None:
                self.addTopLevelItem(parent)
            else:
                # Machine learning: augmentation, feature steps, classifiers, evaluation beneath it
                if group not in groups:
                    groups[group] = self._heading(group)
                    self.addTopLevelItem(groups[group])
                    groups[group].setExpanded(True)
                groups[group].addChild(parent)
            for type_name, kind in REGISTRY.items():
                if kind.category == category:
                    item = QtWidgets.QTreeWidgetItem([kind.title])
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, type_name)
                    item.setToolTip(0, kind.describe(kind.defaults()))
                    parent.addChild(item)
            parent.setExpanded(True)
        self.itemDoubleClicked.connect(self._double_clicked)
        self.setMouseTracking(True)
        self.itemEntered.connect(
            lambda item, _column: self.hovered.emit(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or ""))

    def leaveEvent(self, event):
        self.hovered.emit("")
        super().leaveEvent(event)

    @staticmethod
    def _heading(text):
        item = QtWidgets.QTreeWidgetItem([text])
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QtGui.QBrush(QtGui.QColor(soften(CATEGORY_COLOURS[text]) if text in CATEGORY_COLOURS
                                                        else color("text"))))
        if text in CATEGORY_ICONS:
            from ..icons import tinted
            item.setIcon(0, tinted(CATEGORY_ICONS[text], soften(CATEGORY_COLOURS.get(text, color("subtle")))))
        return item

    def _double_clicked(self, item, _column):
        type_name = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if type_name:
            self.chosen.emit(type_name)

    def mimeData(self, items):
        data = QtCore.QMimeData()
        type_name = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole) if items else None
        if type_name:
            data.setData(BLOCK_MIME, type_name.encode("utf-8"))
        return data

    def mimeTypes(self):
        return [BLOCK_MIME]


class RunJob(QtCore.QRunnable):
    """Runs a copy of a sheet's pipeline off the window's thread."""

    class Signals(QtCore.QObject):
        progress = QtCore.Signal(object, int, str, str)          # sheet, revision, node, text
        done = QtCore.Signal(object, int, object, object, str)   # sheet, revision, outputs, failed node, message

    def __init__(self, sheet, revision, graph, cache, targets):
        super().__init__()
        self.sheet, self.revision, self.graph, self.cache, self.targets = sheet, revision, graph, cache, targets
        self.stop = sheet.stop
        self.signals = RunJob.Signals()

    def run(self):
        runner = Runner(self.graph, self.cache)
        try:
            outputs = runner.run(self.targets, progress=lambda node, text: self.signals.progress.emit(
                self.sheet, self.revision, node, text), stop=self.stop.is_set)
            self.signals.done.emit(self.sheet, self.revision, outputs, None, "")
        except Stopped:
            self.signals.done.emit(self.sheet, self.revision, None, None, STOPPED)
        except PipelineError as exc:
            self.signals.done.emit(self.sheet, self.revision, None, exc.node, str(exc))
        except Exception as exc:          # never let a worker die silently
            self.signals.done.emit(self.sheet, self.revision, None, None, "{}: {}".format(type(exc).__name__, exc))


def _sheet_property(name):
    """An attribute of the sheet being shown, as the page's own: the page speaks for its current tab."""
    return property(lambda self: getattr(self.sheet, name), lambda self, value: setattr(self.sheet, name, value))


class WorkbenchPage(QtWidgets.QWidget):
    status_changed = QtCore.Signal(str)
    #: emitted when a run has finished, failed or not
    ran = QtCore.Signal()

    path = _sheet_property("path")              # the .pipeline.json last saved or opened
    modified = _sheet_property("modified")
    errors = _sheet_property("errors")          # node id -> message from the last run
    revision = _sheet_property("revision")

    def __init__(self):
        super().__init__()
        #: outputs by the engine's keys, shared by every sheet: two that read the same file read it once
        self.cache = {}
        self.sheets = []
        self.sheet = None
        self.queue = []                         # sheets waiting for a free place to run
        self.jobs = set()                       # runs going on now
        self._restoring = False

        self.palette = Palette()
        from ..project_panel import stored_folders
        self.library = Library(lambda: list(dict.fromkeys([str(data_dir())] + stored_folders())))
        self.library.open_file.connect(lambda path: self.open_in_tab(path))
        self.library.open_ready_made.connect(lambda title: self.new_tab(title))
        self.library.open_profile.connect(self.quick_with_profile)
        # the blocks, and under them the one the mouse is on as it will look on the sheet
        self.preview = BlockPreview()
        self.palette.hovered.connect(self.preview.set_block)
        blocks = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(blocks)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self.palette, 1)
        column.addWidget(self.preview)
        #: Blocks and Library live in the main window's side panel, beside the rail (rail.py), with Projects
        self.panels = {"Blocks": blocks, "Library": self.library}
        self.inspector = Inspector(choices_for=self.choices_for, rate_for=self.rate_for, input_for=self.input_for)
        side = self.side_scroll = QtWidgets.QScrollArea()
        side.setWidgetResizable(True)
        # settings wrap to the panel's width; a scroll bar only if something still cannot, never cut off
        side.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        side.setWidget(self.inspector)
        side.setMinimumWidth(320)
        self.tabs = QtWidgets.QTabWidget()
        self.tab_bar = SheetTabBar()
        self.tabs.setTabBar(self.tab_bar)
        self.tabs.setDocumentMode(True)
        self.tab_bar.tabCloseRequested.connect(lambda index: self.close_sheet(self._sheet_at(index)))
        self.tab_bar.detach.connect(lambda index: self.detach(self._sheet_at(index)))
        self.tab_bar.tabMoved.connect(lambda _a, _b: self._remember())
        self.tab_bar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tab_bar.customContextMenuRequested.connect(self._tab_menu)
        self.tabs.currentChanged.connect(self._tab_changed)
        plus = QtWidgets.QToolButton()
        plus.setText("+")
        plus.setToolTip("A new tab: an empty sheet or one of the two pipelines saved last; templates and the "
                        "literature are in the Library")
        plus.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        plus.setMenu(self._new_menu(plus, lambda: self.new_tab(None), self.open_in_tab))
        self.tabs.setCornerWidget(plus, QtCore.Qt.Corner.TopRightCorner)
        self.split = QtWidgets.QSplitter()
        self.split.addWidget(self.tabs)
        self.split.addWidget(side)
        self.split.setStretchFactor(0, 1)
        self.split.setSizes([1090, 380])
        # the sheet never folds away when a side is dragged wide: it keeps room to be seen and found again
        self.tabs.setMinimumWidth(260)
        self.split.setCollapsible(0, False)

        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("Pipeline name")
        self.name.textChanged.connect(self._fit_name)
        self.name.editingFinished.connect(self._renamed_pipeline)
        # icons alone: the name and shortcut on hover; Run keeps its word
        new = QtWidgets.QToolButton()
        only(new, "plus", "New", more="Replace this tab's sheet: empty, or one of the two pipelines saved last; "
             "+ beside the tabs opens a new tab instead")
        new.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        new.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")
        new.setMenu(self._new_menu(new, lambda: self.new_pipeline(None), self.open_pipeline))
        open_button = QtWidgets.QToolButton()
        only(open_button, "folder-open", "Open", "Ctrl+O")
        open_button.clicked.connect(lambda: self.open_pipeline())
        save = QtWidgets.QToolButton()
        only(save, "save", "Save", "Ctrl+S")
        save.clicked.connect(lambda: self.save_pipeline(self.path))
        save_as = QtWidgets.QToolButton()
        only(save_as, "copy", "Save as")
        save_as.clicked.connect(lambda: self.save_pipeline(None))
        # one button: Run, Stop while the sheet in front runs (sheets.show_run_button)
        self.run_button = QtWidgets.QPushButton()
        self.run_button.setProperty("primary", True)
        show_run_button(self.run_button, None)
        self.run_button.clicked.connect(lambda: run_or_stop(self, self.sheet))
        self.run_all_button = QtWidgets.QToolButton()
        only(self.run_all_button, "fast-forward", "Run all tabs",
             more="Compute every tab's pipeline, {} at a time; the others wait in line".format(MAX_RUNNING))
        self.run_all_button.clicked.connect(lambda: self.run_all())
        self.state = ElidedLabel()      # a long message ends in …, whole in its tooltip: the window stays on screen
        self.state.setStyleSheet(muted())
        compare = QtWidgets.QToolButton()
        only(compare, "columns-2", "Compare", more="Run this pipeline and others over many recordings, side by side")
        compare.clicked.connect(lambda: self.open_compare())
        report = QtWidgets.QToolButton()
        only(report, "file-text", "Report", more="The pipeline, its methods and every result: PDF, Word, web page, "
             "or only the figures as PNG")
        report.clicked.connect(lambda: self.save_report())
        python = QtWidgets.QToolButton()
        only(python, "code-xml", "Python", more="The pipeline as Python that runs without GIOVA and computes "
                                                "the same numbers: a zip with a short run.py, or one script")
        python.clicked.connect(lambda: self.export_python())
        # one line on a wide screen; on a narrow one (a laptop at high scaling) the groups go on to a second
        # line instead of pushing the window past the screen's edge
        bar = WrapRow(spacing=12)
        bar.setContentsMargins(6, 4, 6, 0)
        bar.add(group(new, open_button, save, save_as))
        bar.add(self.name)
        bar.add(group(self.run_button, self.run_all_button))
        bar.add(self.state, stretch=True)
        bar.add(group(compare, report, python))
        self.strip = RunStrip()
        self.strip.clicked.connect(lambda: self.show_bottom(not self.bottom.isVisible()))

        # Under the sheet, folded until the strip is clicked: Results (Compare, compare.py) and the Log of
        # every run. It never opens by itself: a run only names its errors on the strip.
        self.bottom = QtWidgets.QTabWidget()
        self.bottom.setDocumentMode(True)
        self.results = QtWidgets.QWidget()
        results = QtWidgets.QVBoxLayout(self.results)
        results.setContentsMargins(6, 6, 6, 6)
        self.results_hint = QtWidgets.QLabel("Compare pipelines over many recordings: the Compare button above "
                                             "puts this sheet here with the recordings used last.")
        self.results_hint.setWordWrap(True)
        self.results_hint.setStyleSheet(muted())
        results.addWidget(self.results_hint)
        results.addStretch(1)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        from ..fonts import mono
        self.log.setFont(mono())
        self.bottom.addTab(self.results, "Results")
        self.bottom.addTab(self.log, "Log")
        self.bottom.currentChanged.connect(self._bottom_tab_changed)
        self.errors_unseen = 0
        self.vsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.vsplit.addWidget(self.split)
        self.vsplit.addWidget(self.bottom)
        self.vsplit.setCollapsible(0, False)
        self.vsplit.setCollapsible(1, False)
        self.vsplit.setStretchFactor(0, 1)

        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addLayout(bar)
        column.addWidget(self.vsplit, 1)
        column.addWidget(self.strip)
        self.bottom.setCurrentIndex(1 if user_settings().value("workbench/bottom_tab", "") == "Log" else 0)
        self.show_bottom(user_settings().value("workbench/bottom_open", False, type=bool), remember=False)

        # once a second while anything runs: the time spent and left move on
        self.ticker = QtCore.QTimer(self)
        self.ticker.setInterval(1000)
        self.ticker.timeout.connect(self._tick)

        self.palette.chosen.connect(lambda t: self.canvas.add_block(t, after=self.canvas.selected()))
        self._pending = False
        self.inspector.edited.connect(self._param_edited)
        self.inspector.layout_requested.connect(self.choose_layout)
        self.inspector.renamed.connect(lambda node: (self.canvas.refresh(node), self._changed()))
        for key, action in (("F5", lambda: run_or_stop(self, self.sheet)),
                            ("Ctrl+S", lambda: self.save_pipeline(self.path)),
                            ("Ctrl+O", lambda: self.open_pipeline())):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)
        self._restore()

    #: Pipelines saved last that + and New offer; the rest, and the templates, are in the Library.
    LAST_SAVED = 2

    def last_saved(self, count=LAST_SAVED):
        """The pipeline files saved last, newest first: those the Library lists (project folders and the ones
        used lately elsewhere), by when the file was written."""
        paths = [path for _project, found in self.library.mine() for path in found]
        dated = []
        for path in paths:
            try:
                dated.append((path.stat().st_mtime, path))
            except OSError:
                continue
        return [path for _when, path in sorted(dated, key=lambda item: -item[0])[:count]]

    def _new_menu(self, parent, empty, open_file):
        """Empty sheet, then the pipelines saved last; filled each time it opens, so a file just saved is there."""
        menu = QtWidgets.QMenu(parent)
        menu.setToolTipsVisible(True)

        def fill():
            menu.clear()
            menu.addAction("Empty sheet", empty)
            saved = self.last_saved()
            if saved:
                menu.addSeparator()
                from .library import describe_file
                for path in saved:
                    info = describe_file(path)
                    action = menu.addAction(info[0] if info else path.name, lambda p=str(path): open_file(p))
                    action.setToolTip(str(path))
        menu.aboutToShow.connect(fill)          # not filled before: the project folders are read when it opens
        return menu

    # ---------------- Results and Log, under the sheet ----------------

    #: The panel's height when it opens, pixels; dragged taller or shorter afterwards.
    BOTTOM_HEIGHT = 300

    def show_bottom(self, shown, tab=None, remember=True):
        """Open (or fold) the panel under the sheet, at `tab` ("Results" or "Log") if named."""
        if tab is not None:
            self.bottom.setCurrentIndex(["Results", "Log"].index(tab))
        if shown and not self.bottom.isVisible():
            self.bottom.show()
            total = max(sum(self.vsplit.sizes()), self.BOTTOM_HEIGHT * 2)
            self.vsplit.setSizes([total - self.BOTTOM_HEIGHT, self.BOTTOM_HEIGHT])
        elif not shown:
            self.bottom.hide()
        self.strip.set_open(shown)
        if shown and self.bottom.currentWidget() is self.log:
            self._errors_seen()
        if remember:
            user_settings().setValue("workbench/bottom_open", bool(shown))

    def _bottom_tab_changed(self, index):
        user_settings().setValue("workbench/bottom_tab", self.bottom.tabText(index))
        if self.bottom.widget(index) is self.log and self.bottom.isVisible():
            self._errors_seen()

    def _errors_seen(self):
        self.errors_unseen = 0
        self.strip.set_errors(0)

    def write_log(self, sheet, text, error=False):
        """One line in the Log: the time, the pipeline, what happened. An error is also counted on the strip
        until the Log is looked at."""
        import time
        name = (sheet.graph.name or "Pipeline") if sheet is not None else ""
        self.log.appendPlainText("{}  {}{}{}".format(time.strftime("%H:%M:%S"), name, "  ·  " if name else "",
                                                     ("ERROR  " if error else "") + text))
        if error and not (self.bottom.isVisible() and self.bottom.currentWidget() is self.log):
            self.errors_unseen += 1
            self.strip.set_errors(self.errors_unseen)

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self.sheet, "fitted", False):
            # built while hidden, the sheet had no size to fit to: its first view once it has one
            self.sheet.fitted = True
            QtCore.QTimer.singleShot(0, self.canvas, self.canvas.first_view)  # bound: never after the canvas is gone

    # ---------------- the sheets ----------------

    @property
    def canvas(self):
        return self.sheet.canvas

    @property
    def graph(self):
        return self.sheet.canvas.graph

    @property
    def running(self):
        return self.sheet.busy

    def _make_sheet(self, graph, path=None, modified=False, index=None):
        """A new sheet in a tab of its own (at `index`, else last), not yet shown."""
        canvas = Canvas()
        sheet = Sheet(canvas)
        canvas.selection.connect(lambda node, s=sheet: self._from(s, self._selected, node))
        canvas.changed.connect(lambda s=sheet: self._from(s, self._changed))
        canvas.message.connect(lambda text, s=sheet: self._from(s, self._say, text))
        canvas.run_to.connect(lambda node, s=sheet: self._from(s, self.run, [node]))
        canvas.help_for.connect(self.help_for)
        canvas.moved.connect(lambda s=sheet: self._from(s, self._moved))
        canvas.released.connect(lambda s=sheet: self._from(s, self._show_pending))
        canvas.set_graph(graph)
        sheet.path = Path(path) if path else None
        sheet.modified = modified
        if index is None:
            self.sheets.append(sheet)
            self.tabs.addTab(canvas, "")
        else:
            self.sheets.insert(index, sheet)
            self.tabs.insertTab(index, canvas, "")
        self._label(sheet)
        return sheet

    def choose_layout(self, node_id):
        """How the MATLAB file a Recording block reads is laid out, asked in the layout dialog; kept beside the file
        (and for files laid out the same way when asked), then the pipeline runs again."""
        from ..matdialog import remap
        path = str(self.graph.params(node_id).get("path") or "")
        if not path:
            return False
        recording, note = remap(self, path)
        if recording is None:
            return False
        self._say(note or "")                    # the layout's .mapping.json is in the block's key: it reads anew
        self.run()
        return True

    def help_for(self, type_name):
        """The block library's page for `type_name`, in the main window's Help (or one of the page's own)."""
        key = "block:" + type_name
        main = self.window()
        if hasattr(main, "open_help"):
            return main.open_help(key)
        if getattr(self, "_help", None) is None:
            from ..help_dialog import HelpDialog
            self._help = HelpDialog(self)
        return self._help.open_at(key)

    def _from(self, sheet, handler, *args):
        """A canvas spoke: its sheet becomes the one the settings and the bar are for, then `handler`."""
        if sheet is not self.sheet:
            self.activate(sheet)
        return handler(*args)

    def _sheet_at(self, index):
        widget = self.tabs.widget(index)
        return next((s for s in self.sheets if s.canvas is widget), None)

    def _tab_changed(self, index):
        sheet = self._sheet_at(index)
        if sheet is not None and sheet is not self.sheet:
            self.activate(sheet)

    def activate(self, sheet):
        """Show `sheet`'s name, run and selected block; its tab comes to the front unless it has a window."""
        if sheet not in self.sheets:
            return
        self.sheet = sheet
        if sheet.window is None and self.tabs.currentWidget() is not sheet.canvas:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentWidget(sheet.canvas)
            self.tabs.blockSignals(False)
        self.name.blockSignals(True)
        self.name.setText(sheet.graph.name)
        self.name.blockSignals(False)
        self._fit_name(sheet.graph.name)
        self._refresh_states(sheet)
        self._show_run(sheet)
        self.inspector.show_node(sheet.graph, None)
        if sheet.canvas.selected() is not None:
            self._selected(sheet.canvas.selected())
        if not getattr(sheet, "fitted", False):
            self._fit_later(sheet)
        self._remember()

    def _fit_later(self, sheet):
        """The sheet's first view (as left last session, else the whole pipeline) once its canvas has a size: a
        tab made while hidden has none yet."""
        sheet.fitted = sheet.canvas.isVisible()
        if sheet.fitted:
            QtCore.QTimer.singleShot(0, sheet.canvas, sheet.canvas.first_view)  # bound: never after it is gone

    def new_tab(self, template=None, graph=None):
        """A new tab, with `graph`, a template or literature profile by title, or an empty sheet; shown."""
        if graph is None:
            graph = ready_made(template) if template else Graph("New pipeline")
            if template and self.sheet is not None:
                self._carry_recording(graph)
        sheet = self._make_sheet(graph)
        self.activate(sheet)
        self._update_run_all()
        self._say("New tab: {}".format(graph.name))
        return sheet

    def _carry_recording(self, graph):
        # one recording for the whole pipeline: carry the one in use over
        loads = self.graph.sources("load")
        path = self.graph.params(loads[0])["path"] if len(loads) == 1 else ""
        for node in graph.sources("load"):
            if path:
                graph.set(node, path=path)

    def close_sheet(self, sheet):
        """Close a tab, asking about unsaved changes; a running pipeline is stopped. The last tab is
        replaced by an empty sheet rather than closed."""
        if sheet is None:
            return False
        if not self._may_discard(sheet):
            return False
        self.stop(sheet)
        sheet.revision += 1                    # whatever it still delivers belongs to no tab
        if len(self.sheets) == 1:
            self.set_graph(Graph("New pipeline"))
            return True
        if sheet.window is not None:
            window, sheet.window = sheet.window, None
            window.hide()
            window.deleteLater()
        index = self.tabs.indexOf(sheet.canvas)
        if index >= 0:
            self.tabs.removeTab(index)
        self.sheets.remove(sheet)
        sheet.canvas.deleteLater()
        if self.sheet is sheet:
            self.activate(self._sheet_at(self.tabs.currentIndex()) or self.sheets[0])
        self._prune()
        self._update_run_all()
        self._remember()
        return True

    def detach(self, sheet):
        """`sheet` in a window of its own, for a second screen; closing the window brings it back."""
        if sheet is None or sheet.window is not None:
            return None
        if len([s for s in self.sheets if s.window is None]) == 1:
            self._say("The last tab stays: open another with + to move this one out")
            return None
        index = self.tabs.indexOf(sheet.canvas)
        self.tabs.removeTab(index)
        window = SheetWindow(self, sheet)
        sheet.window = window
        # where the tab was let go, but whole on that screen: a window half off it looks like a lost sheet
        cursor = QtGui.QCursor.pos()
        screen = QtGui.QGuiApplication.screenAt(cursor) or self.screen()
        room = screen.availableGeometry()
        size = window.size().boundedTo(room.size())
        window.resize(size)
        x = min(max(cursor.x() - 80, room.left()), room.right() - size.width())
        y = min(max(cursor.y() - 20, room.top()), room.bottom() - size.height())
        window.move(x, y)
        window.show()
        window.raise_()
        window.activateWindow()
        if self.sheet is sheet:
            self.activate(self._sheet_at(self.tabs.currentIndex()))
        self._remember()
        return window

    def attach(self, sheet):
        """A sheet's window closed: the sheet back among the tabs, in front."""
        if sheet is None or sheet.window is None or sheet not in self.sheets:
            return
        sheet.window = None
        self.tabs.addTab(sheet.canvas, "")
        sheet.canvas.show()
        sheet.fitted = False                      # the tab is not the window's size
        self._label(sheet)
        self.activate(sheet)

    def attach_all(self):
        for sheet in [s for s in self.sheets if s.window is not None]:
            window = sheet.window
            self.attach(sheet)
            window.hide()
            window.deleteLater()

    def shutdown(self):
        """The program closes: every run asked to stop, every window back as a tab, the tabs remembered."""
        for sheet in self.sheets:
            sheet.stop.set()
            sheet.revision += 1
        self.queue.clear()
        self.attach_all()
        self._remember()

    def _tab_menu(self, point):
        sheet = self._sheet_at(self.tab_bar.tabAt(point))
        if sheet is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Run", lambda: self.run(sheet=sheet)).setEnabled(not sheet.busy)
        if sheet.busy:
            menu.addAction("Stop", lambda: self.stop(sheet))
        menu.addAction("Run all tabs", self.run_all)
        menu.addSeparator()
        menu.addAction("Move to its own window", lambda: self.detach(sheet))
        menu.addAction("Close", lambda: self.close_sheet(sheet))
        menu.exec(self.tab_bar.mapToGlobal(point))

    def _label(self, sheet):
        """The tab's title and colour, or its window's, as the sheet stands."""
        if sheet.window is not None:
            sheet.window.refresh()
            return
        index = self.tabs.indexOf(sheet.canvas)
        if index < 0:
            return
        self.tabs.setTabText(index, sheet.tab_text())
        self.tabs.setTabToolTip(index, str(sheet.path) if sheet.path else "Not saved yet")
        colour = sheet.tab_colour()
        self.tab_bar.setTabTextColor(index, QtGui.QColor(colour) if colour else QtGui.QColor())

    def _update_run_all(self):
        self.run_all_button.setVisible(len(self.sheets) > 1)

    # ---------------- the pipeline on the sheet ----------------

    def set_graph(self, graph, path=None):
        sheet = self.sheet
        self.stop(sheet)
        sheet.revision += 1          # a run still going belongs to the old sheet
        sheet.status = IDLE
        sheet.accuracy = None
        sheet.canvas.set_graph(graph)
        sheet.errors.clear()
        sheet.path = Path(path) if path else None
        sheet.modified = False
        self.name.setText(graph.name)
        self._fit_later(sheet)
        self._prune()
        self._refresh_states(sheet)
        self._show_run(sheet)
        self.inspector.show_node(graph, None)
        self._remember()

    def new_pipeline(self, template=None):
        if not self._may_discard():
            return
        graph = ready_made(template) if template else Graph("New pipeline")
        if template:
            self._carry_recording(graph)
        self.set_graph(graph)
        self._say("New pipeline: {}".format(graph.name))

    def panel_shown(self, key):
        """The main window's side panel shows one of this page's panels."""
        if key == "Library":
            self.library.refresh()              # a file saved or copied in meanwhile is there

    def open_in_tab(self, path):
        """A saved pipeline in a tab of its own (the one it is already in, if open); nothing is replaced."""
        shown = next((s for s in self.sheets if s.path and _same_file(s.path, path)), None)
        if shown is not None:
            self.activate(shown)
            if shown.window is not None:
                shown.window.raise_()
                shown.window.activateWindow()
            return shown
        try:
            graph = Graph.load(path)
        except ValueError as exc:
            self._say("Cannot open {}: {}".format(Path(path).name, exc))
            return None
        sheet = self._make_sheet(graph, path)
        self.activate(sheet)
        self._update_run_all()
        recent.remember(recent.PIPELINES, path, name=graph.name)
        self._say("Opened {}".format(Path(path).name))
        return sheet

    def open_pipeline(self, path=None):
        # every way in (Open…, Home's Continue, a file named by the program) asks before the sheet is replaced
        if path is not None:
            shown = next((s for s in self.sheets if s.path and _same_file(s.path, path)), None)
            if shown is not None and not shown.modified:
                self.activate(shown)                # already open in a tab: to it, nothing replaced
                if shown.window is not None:
                    shown.window.raise_()
                    shown.window.activateWindow()
                return True
        if not self._may_discard():
            return False
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open pipeline", str(self._folder()),
                                                            "Pipelines (*{})".format(SUFFIX))
            if not path:
                return False
        try:
            graph = Graph.load(path)
        except ValueError as exc:
            self._say("Cannot open {}: {}".format(Path(path).name, exc))
            return False
        self.set_graph(graph, path)
        recent.remember(recent.PIPELINES, path, name=graph.name)
        self._say("Opened {}".format(Path(path).name))
        return True

    def save_pipeline(self, path=None, sheet=None):
        sheet = sheet or self.sheet
        if path is None:
            folder = sheet.path.parent if sheet.path else Path(data_dir())
            suggested = folder / (_safe(sheet.graph.name) + SUFFIX)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save pipeline", str(suggested),
                                                            "Pipelines (*{})".format(SUFFIX))
            if not path:
                return False
            if not path.endswith(SUFFIX):
                path = str(Path(path).with_suffix("")).removesuffix(".pipeline") + SUFFIX
        sheet.graph.save(path)
        sheet.path, sheet.modified = Path(path), False
        self._label(sheet)
        self._remember()
        recent.remember(recent.PIPELINES, path, name=sheet.graph.name)
        self.library.refresh()
        self._say("Saved {}".format(Path(path).name))
        return True

    def open_recording(self, path):
        """A recording chosen in the Projects pane goes into the pipeline's Recording block, and runs."""
        loads = self.graph.sources("load")
        if not loads:
            node = self.canvas.add_block("load")
            loads = [node]
        selected = self.canvas.selected()
        target = selected if selected in loads else loads[0]
        self.graph.set(target, path=str(path))
        self._param_edited(target)
        self.canvas.select(target)
        self.run()
        return True

    def select_output(self, kind):
        """Select the first block of the tab in view whose result is of `kind` (spectrum, tf, features, epochs):
        its result is then shown beside the settings. Its id, or None when the tab has none."""
        for node_id in self.graph.order():
            if block(self.graph.nodes[node_id].type).outputs[0].types[0] == kind:
                self.canvas.select(node_id)
                return node_id
        return None

    def quick_with_profile(self, name):
        """An analysis profile chosen in the Library: a quick analysis with it, on the recording of the tab in view
        when it reads one, else on a recording asked for."""
        from ...core import analysis_profiles as ap
        profile = next((p for p in ap.available(self.library.profile_root) if p.name == name), None)
        if profile is None:
            self._say("No analysis profile named {}".format(name))
            return None
        context = self.recording_context()
        if context is None:
            from .quick import LAST
            LAST["profile"] = name
            return self.quick_analysis()
        return self.quick_analysis(context[2], profile)

    def recording_context(self):
        """(sheet, Recording node, file) of the tab in view when it reads exactly one recording, else None: what
        a profile opened from the Library runs on. Two Recording blocks are ambiguous, so neither is taken."""
        if self.sheet is None:
            return None
        loads = self.graph.sources("load")
        if len(loads) != 1:
            return None
        path = self.graph.params(loads[0])["path"]
        return (id(self.sheet), loads[0], str(path)) if path else None

    def quick_analysis(self, path=None, profile=None):
        """Quick analysis: a recording and a ready profile (asked for when not given) as the profile's pipeline in
        a new tab, running; the tab in view is left as it is. The new sheet, or None when cancelled."""
        from ...pipeline.convert import from_profile
        if profile is None:
            from .quick import QuickDialog
            dialog = QuickDialog(path or "", self, root=self.library.profile_root)
            if not dialog.exec():
                return None
            path, profile = dialog.chosen()
        graph, notes = from_profile(profile, path)
        graph.name = "Quick analysis: {}".format(profile.name)
        sheet = self.new_tab(graph=graph)
        sheet.modified = True
        self._label(sheet)
        self._remember()
        if path:
            recent.remember(recent.RECORDINGS, path)
            self.run()
        self._say("Quick analysis of {} with '{}'.{}".format(Path(path).name if path else "no recording",
                                                             profile.name, " " + " ".join(notes) if notes else ""))
        return sheet

    def import_session(self, path, recording=None):
        """A saved Analysis session (.analysis.json) as a pipeline in a new tab, not saved yet; the tab in view and
        the session file are left as they are. What could not be carried over is said. The new sheet, or None."""
        from ...pipeline.convert import from_session
        try:
            graph, notes = from_session(path, source_path=recording)
        except (ValueError, OSError) as exc:
            self._say("Cannot open the analysis session {}: {}".format(Path(path).name, exc))
            return None
        sheet = self.new_tab(graph=graph)
        sheet.modified = True
        self._label(sheet)
        self._remember()
        self._say("Opened the analysis session {}. Save it as a pipeline to keep it.{}".format(
            Path(path).name, " " + " ".join(notes) if notes else ""))
        if notes:
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Information, "Analysis session",
                                        "\n\n".join(notes), parent=self)
            box.setModal(False)
            box.show()
            self._notice = box
        return sheet

    def adopt(self, graph, notes=()):
        """A pipeline made elsewhere (the Analysis page) onto the sheet; what differs is said."""
        if not self._may_discard():
            return False
        self.set_graph(graph)
        self.modified = True
        self._label(self.sheet)
        self._remember()
        text = "Opened from the Analysis page." + (" " + " ".join(notes) if notes else "")
        self._say(text)
        if notes:
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Information, "Opened on the Workbench",
                                        "\n\n".join(notes), parent=self)
            box.setModal(False)
            box.show()
            self._notice = box
        return True

    # ---------------- reports, scripts, comparisons ----------------

    def open_compare(self, with_sheet=True):
        """Compare in the Results panel under the sheet (it was a window of its own before GIOVA): this sheet's
        pipeline added, unless `with_sheet` is False. A comparison still running is kept, not replaced."""
        from .compare import CompareWindow
        old = getattr(self, "_compare", None)
        if old is not None and old.job is not None:
            self.show_bottom(True, "Results")
            return old
        others = [s.graph for s in self.sheets if s is not self.sheet]
        window = CompareWindow(self.graph if with_sheet and self._comparable() else None, None, tabs=others)
        window.setWindowFlags(QtCore.Qt.WindowType.Widget)
        layout = self.results.layout()
        if old is not None:
            layout.removeWidget(old)
            old.deleteLater()
        self.results_hint.hide()
        for i in reversed(range(layout.count())):     # the spacer under the hint would take half the panel
            if layout.itemAt(i).spacerItem() is not None:
                layout.takeAt(i)
        layout.insertWidget(0, window, 1)
        window.show()
        self._compare = window
        self.show_bottom(True, "Results")
        return window

    def _comparable(self):
        from ...pipeline import batch
        try:
            batch.source_node(self.graph)
        except ValueError:
            return False
        return bool(batch.score_nodes(self.graph))

    def save_report(self, path=None):
        from ..jobs import busy
        from .report import ask_path, pipeline_report, saved_text
        outputs = self.outputs()
        if not outputs:
            self._say("Run the pipeline first: the report shows its results")
            return None
        if path is None:
            path = ask_path(self, self._folder() / (_safe(self.graph.name) + ".pdf"))
            if not path:
                return None
        self._say("Writing the report…")
        self.inspector._keep_choices()          # what the block shown has chosen now is what the report draws
        with busy():
            saved = pipeline_report(self.graph, outputs, self.canvas).save(path)
        recent.remember(recent.REPORTS, saved[0] if isinstance(saved, list) else saved, name=self.graph.name)
        self._say(saved_text(saved))
        return path

    def export_python(self, path=None):
        """The pipeline as Python: a zip (run.py and GIOVA's code beside it), or one .py script."""
        from ...pipeline.export import save_package, save_python
        if path is None:
            suggested = self._folder() / (_safe(self.graph.name).replace(" ", "_").lower() + ".zip")
            path, chosen = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save as Python", str(suggested),
                "Zip: run.py and the code it uses (*.zip);;One Python script (*.py)")
            if not path:
                return None
            if Path(path).suffix.lower() not in (".zip", ".py"):
                path = path + (".py" if "*.py" in chosen else ".zip")
        if Path(path).suffix.lower() == ".py":
            save_python(self.graph, path)
            self._say("Saved {}: it runs on its own with Python; its first lines say what to install".format(
                Path(path).name))
        else:
            save_package(self.graph, path)
            self._say("Saved {}: unzip it, then pip install -r requirements.txt and python run.py".format(
                Path(path).name))
        return path

    def _folder(self):
        return self.path.parent if self.path else Path(data_dir())

    def _may_discard(self, sheet=None):
        """May the sheet be replaced or closed? Unsaved changes, in a file or on a sheet never saved, are asked
        about: Save (then go on if it was saved), Discard, or Cancel."""
        sheet = sheet or self.sheet
        if not sheet.modified:
            return True
        if sheet is not self.sheet:
            self.activate(sheet)                  # the question is about what is in front
        buttons = QtWidgets.QMessageBox.StandardButton
        text = ("{} has changes that are not saved.".format(sheet.path.name) if sheet.path else
                "'{}' has never been saved.".format(sheet.graph.name))
        answer = QtWidgets.QMessageBox.question(self, "Unsaved changes", text + " Save it first?",
                                                buttons.Save | buttons.Discard | buttons.Cancel, buttons.Save)
        if answer == buttons.Save:
            return self.save_pipeline(sheet.path, sheet)
        return answer == buttons.Discard

    def _fit_name(self, text):
        """The name box as wide as its name: "Band power per condition" must not be cut."""
        width = self.name.fontMetrics().horizontalAdvance(text or self.name.placeholderText()) + 28
        # as wide as its name, up to a limit: the toolbar row wraps on a small screen rather than cutting it
        # (a minimum alone left the box at its default width, its name cut at the front)
        self.name.setFixedWidth(max(120, min(width, 420)))
        if not self.name.hasFocus():
            self.name.setCursorPosition(0)          # a name longer than the box shows its beginning

    def _renamed_pipeline(self):
        name = self.name.text().strip() or "Pipeline"
        if name != self.graph.name:
            self.graph.name = name
            self._changed()

    # ---------------- editing ----------------

    def _selected(self, node_id):
        # Building the settings and result takes a moment; not while a block is being dragged.
        if self.canvas.pressing:
            self._pending = True
            return
        self._pending = False
        if node_id != getattr(self, "_shown_node", None):
            self.side_scroll.verticalScrollBar().setValue(0)      # another block: its settings from the top
            self._shown_node = node_id
        self.inspector.show_node(self.graph, node_id, self._outputs(node_id))
        if node_id in self.errors:
            message = self.errors[node_id]
            self.inspector.show_error(message, problems.fixes(self, node_id, message))
            return
        keys = Runner(self.graph, self.cache).current_keys()
        if node_id is not None and node_id not in keys:
            message = ("; ".join(m for n, m in self.graph.problems() if n == node_id) or
                       "Waits for the blocks before it to get their inputs.")
            self.inspector.show_error(message, problems.fixes(self, node_id, message))

    def _param_edited(self, node_id):
        self.canvas.refresh(node_id)
        self._changed()
        if node_id == self.inspector.node_id:
            self.inspector.show_outputs(self._outputs(node_id))

    def _show_pending(self):
        if self._pending:
            self._selected(self.canvas.selected())

    def _moved(self):
        self.modified = True
        self._label(self.sheet)
        self._remember()

    def _changed(self):
        self.modified = True
        self._refresh_states(self.sheet)
        self._label(self.sheet)
        self._remember()

    # ---------------- running ----------------

    def run(self, targets=None, sheet=None):
        """Compute `targets` (default: every block that has its inputs) of `sheet` (default: the one shown) in
        the background; when MAX_RUNNING sheets compute already, it waits in line."""
        sheet = sheet or self.sheet
        if sheet.busy:
            return False
        runner = Runner(sheet.graph, self.cache)
        runnable = runner.runnable()
        targets = [t for t in (targets or runnable) if t in runnable]
        if not targets:
            if sheet is self.sheet:
                problems = sheet.graph.problems()
                self._say(problems[0][1] if problems else "Add a Recording block to start")
            return False
        sheet.targets = targets
        sheet.errors.clear()
        sheet.status = QUEUED
        self.queue.append(sheet)
        self._start_waiting()
        self._show_run(sheet)
        return True

    def run_all(self):
        """Every tab's pipeline that can run, MAX_RUNNING at a time, in the tabs' order."""
        order = [self._sheet_at(i) for i in range(self.tabs.count())] + [s for s in self.sheets if s.window]
        started = [s for s in order if s is not None and self.run(sheet=s)]
        self._say("Running {} pipeline{}".format(len(started), "s" if len(started) != 1 else "") if started else
                  "Nothing to run: every tab is up to date, running, or misses a Recording")
        return len(started)

    def stop(self, sheet):
        """End `sheet`'s run at its next step (the blocks computed keep their results), or take it out of line."""
        if sheet is None:
            return
        if sheet.status == QUEUED:
            self.queue.remove(sheet)
            sheet.status = STOPPED
            self._refresh_states(sheet)
            self._show_run(sheet)
        elif sheet.status == RUNNING:
            sheet.stop.set()
            sheet.status = STOPPING
            self._show_run(sheet)

    def _start_waiting(self):
        while self.queue and len(self.jobs) < MAX_RUNNING:
            self._start(self.queue.pop(0))

    def _start(self, sheet):
        keys = Runner(sheet.graph, self.cache).current_keys()
        nodes = [n for n in sheet.graph.order(sheet.targets) if keys.get(n) not in self.cache]
        sheet.revision += 1
        sheet.begin(nodes)
        job = RunJob(sheet, sheet.revision, sheet.graph.copy(), self.cache, sheet.targets)
        job.signals.progress.connect(self._progress)
        job.signals.done.connect(self._done)
        self.jobs.add(job)
        sheet._job = job
        self.write_log(sheet, "Run started: {} block{} to compute".format(len(nodes), "s" if len(nodes) != 1 else ""))
        activity().begin()
        self.ticker.start()
        self._refresh_states(sheet)
        self._show_run(sheet)
        QtCore.QThreadPool.globalInstance().start(job)

    def _progress(self, sheet, revision, node_id, text):
        if revision != sheet.revision or sheet.status not in (RUNNING, STOPPING):
            return
        node = sheet.graph.nodes.get(node_id)
        title = (node.label or block(node.type).title) if node is not None else ""
        moved_on = node_id != sheet.current
        sheet.advance(node_id, text, title)
        if moved_on:
            self.write_log(sheet, title or node_id)
            self._refresh_states(sheet)
        self._show_run(sheet)

    def _done(self, sheet, revision, outputs, failed, message):
        activity().end()                # every run ends once, even one a newer run replaced
        self.jobs = {j for j in self.jobs if not (j.sheet is sheet and j.revision == revision)}
        self._start_waiting()
        if not self.jobs and not self.queue:
            self.ticker.stop()
        if revision != sheet.revision or sheet not in self.sheets:
            return                      # a run replaced or closed: nothing to show of it
        sheet.took = sheet.elapsed
        if message == STOPPED:
            sheet.status = STOPPED
            text = "Stopped after {}: the blocks computed keep their results".format(clock(sheet.elapsed))
        elif failed is not None or message:
            sheet.status = ERROR
            if failed:
                sheet.errors[failed] = message
                node = sheet.graph.nodes.get(failed)
                title = (node.label or block(node.type).title) if node is not None else ""
                text = "Stopped at {}: {} (its settings, right, say what to do)".format(
                    title, problems.split(message, title)[0])
            else:
                text = message
        else:
            sheet.status = DONE
            text = "Done in {}".format(clock(sheet.elapsed))
            self._remember_result(sheet, outputs)
        self._prune()
        self._refresh_states(sheet)
        self._show_run(sheet)
        self.write_log(sheet, text if sheet.status != DONE or sheet.accuracy is None else
                       "{}  ·  accuracy {:.1%}".format(text, sheet.accuracy), error=sheet.status == ERROR)
        if sheet is self.sheet:
            self._say(text)
            node_id = self.canvas.selected()
            if failed is not None and failed != node_id:
                self.canvas.select(failed)          # the block that stopped, its problem and what to do in view
            elif node_id is not None:
                self._selected(node_id)
        window = sheet.window or self.window()
        if not window.isActiveWindow():
            QtWidgets.QApplication.alert(window)      # the taskbar button flashes; no sound
        self.ran.emit()

    def _tick(self):
        for sheet in self.sheets:
            if sheet.status in (RUNNING, STOPPING):
                self._show_run(sheet)

    def _show_run(self, sheet):
        """How `sheet`'s run stands: its tab or window, and the strip and Run button when it is shown."""
        self._label(sheet)
        if sheet is self.sheet:
            self.strip.show_sheet(sheet)
            show_run_button(self.run_button, sheet)
            if sheet.busy:
                self.state.setText("")

    def _remember_result(self, sheet, outputs):
        """The latest accuracy: on the tab, and for Home's list of recent pipelines when the sheet is saved."""
        from ...pipeline import batch
        for node in batch.score_nodes(sheet.graph):
            scores = (outputs or {}).get(node, {}).get("scores")
            if scores is not None:
                sheet.accuracy = scores.accuracy
                if sheet.path is not None:
                    recent.remember(recent.PIPELINES, sheet.path, name=sheet.graph.name, accuracy=scores.accuracy)
                return

    def _prune(self):
        """Keep only outputs some sheet can still show: a 200 MB recording per old setting adds up."""
        keep = set()
        for sheet in self.sheets:
            keep |= set(Runner(sheet.graph, self.cache).current_keys().values())
        for key in [k for k in list(self.cache) if k not in keep]:
            self.cache.pop(key, None)

    def outputs(self, sheet=None):
        """{node id: outputs} that match the sheet as it is now."""
        sheet = sheet or self.sheet
        keys = Runner(sheet.graph, self.cache).current_keys()
        return {n: self.cache[k] for n, k in keys.items() if k in self.cache}

    def _outputs(self, node_id):
        if node_id is None:
            return None
        keys = Runner(self.graph, self.cache).current_keys()
        key = keys.get(node_id)
        return self.cache.get(key) if key else None

    def _refresh_states(self, sheet=None):
        sheet = sheet or self.sheet
        keys = Runner(sheet.graph, self.cache).current_keys()
        states = {}
        for node_id in sheet.graph.nodes:
            key = keys.get(node_id)
            if node_id in sheet.errors:
                states[node_id] = ("error", "", sheet.errors[node_id].split(": ", 1)[-1])
            elif key is None:
                states[node_id] = ("", "", "")
            elif key in self.cache:
                lines = [result_line(v) for v in self.cache[key].values()]
                states[node_id] = ("done", " · ".join(line for line in lines if line), "")
            elif self.cache:
                states[node_id] = ("stale", "changed: Run to update", "")
        if sheet.status in (RUNNING, STOPPING):
            # this run's blocks: the one computed now, and those still to come
            for node_id in sheet.nodes:
                if node_id == sheet.current:
                    states[node_id] = ("running", "computing…", "")
                elif node_id not in sheet.finished and states.get(node_id, ("",))[0] != "done":
                    states[node_id] = ("waiting", "", "")
        sheet.canvas.show_states(states)
        if sheet is self.sheet and not sheet.busy:
            done = sum(1 for s in states.values() if s[0] == "done")
            self.state.setText("{} of {} blocks up to date".format(done, len(sheet.graph.nodes))
                               if sheet.graph.nodes else "Empty sheet: start from New ▾ or drag blocks in")

    # ---------------- names to pick from ----------------

    def choices_for(self, node_id, name):
        """The recording's channel names or event labels upstream of `node_id`, from the last run."""
        if name == "selected":
            from ...core.feature_profiles import FeatureProfile
            return [spec.id for spec in FeatureProfile().catalog()]
        outputs = self.outputs()
        before = self.graph.upstream(node_id)
        for other in reversed(self.graph.order()):
            if other not in before or other not in outputs:
                continue
            for value in outputs[other].values():
                if name in ("channels", "eog_channels"):
                    names = getattr(value, "channel_names", None) or getattr(value, "names", None)
                    if names:
                        return list(names)
                if name == "conditions":
                    if getattr(value, "events", None) is not None:
                        return value.events.unique_labels()
                    if getattr(value, "labels", None):
                        return list(dict.fromkeys(value.labels))
        return None

    def input_for(self, node_id, name):
        """What reaches input `name` of `node_id`, from the last run if still up to date; else None."""
        wire = self.graph.feeds(node_id).get(name)
        if wire is None:
            return None
        return self.outputs().get(wire.source, {}).get(wire.output)

    def rate_for(self, node_id):
        """The sampling rate of the recording feeding `node_id`, from the last run; None before one."""
        outputs = self.outputs()
        for other in reversed(self.graph.order()):
            if other in self.graph.upstream(node_id) and other in outputs:
                for value in outputs[other].values():
                    if getattr(value, "fs", None):
                        return float(value.fs)
        return None

    # ---------------- keeping the sheets ----------------

    def _remember(self):
        """Every tab, in order (a sheet in its own window counts as a tab), and which one is shown. The shown
        one also under the single-sheet keys earlier versions read."""
        if self._restoring or self.sheet is None:
            return
        store = user_settings()
        order = [self._sheet_at(i) for i in range(self.tabs.count())]
        order = [s for s in order if s is not None] + [s for s in self.sheets if s.window is not None]
        store.setValue("workbench/sheets", json.dumps([
            dict(sheet=s.graph.to_dict(), file=str(s.path) if s.path else "", modified=bool(s.modified),
                 view=s.canvas.view_state() if getattr(s, "fitted", False) else s.canvas.saved_view)
            for s in order]))
        store.setValue("workbench/current", order.index(self.sheet) if self.sheet in order else 0)
        store.setValue("workbench/sheet", json.dumps(self.graph.to_dict()))
        store.setValue("workbench/file", str(self.path) if self.path else "")
        store.setValue("workbench/modified", bool(self.modified))

    def _restore(self):
        # everything read before anything is written
        entries = []
        try:
            text = setting("workbench/sheets", "")
            entries = [(Graph.from_dict(e["sheet"]), e.get("file") or None, bool(e.get("modified")), e.get("view"))
                       for e in (json.loads(text) if text else [])]
        except (ValueError, TypeError, KeyError):
            entries = []
        current = int(setting("workbench/current", 0) or 0) if entries else 0
        if not entries:
            # one sheet, as earlier versions kept it
            graph, path = None, setting("workbench/file", "") or None
            modified = str(setting("workbench/modified", False)).lower() == "true"
            try:
                text = setting("workbench/sheet", "")
                graph = Graph.from_dict(json.loads(text)) if text else None
            except (ValueError, TypeError):
                graph = None
            if graph is None:
                graph, path, modified = templates.TEMPLATES[FIRST_TEMPLATE](), None, False
            entries = [(graph, path, modified, None)]
        self._restoring = True
        try:
            for graph, path, modified, view in entries:
                if path:
                    try:
                        # a sheet that is not what its file holds has changes, whatever the flag says
                        modified = modified or Graph.load(path).to_dict() != graph.to_dict()
                    except ValueError:
                        modified = True          # its file is gone or unreadable: the sheet is the only copy
                self._make_sheet(graph, path, modified).canvas.saved_view = view    # shown once it has a size
        finally:
            self._restoring = False
        self.activate(self.sheets[min(max(current, 0), len(self.sheets) - 1)])
        self._update_run_all()
        for sheet in self.sheets:
            self._refresh_states(sheet)
        self._remember()

    def _say(self, text):
        self.state.setText(text)
        self.status_changed.emit(text)


def _same_file(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def _safe(name):
    return "".join(c if c.isalnum() or c in " -_+" else "_" for c in name).strip() or "pipeline"
