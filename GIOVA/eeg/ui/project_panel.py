"""The projects pane: folders added as projects, each a tree of its recordings.

A project is a folder (see core.projects). The *active* project is the one new
live recordings are written to; it is the same folder as Settings → Recording →
Folder, so there is one place a recording goes and two ways to choose it.
Removing a project only takes it off this list: no file is ever deleted here.
"""
import html
import json
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..core import projects as pr
from .preferences import data_dir, set_setting, user_settings
from .theme import color, muted
from .icons import icon, only, put, tinted

FOLDERS_KEY = "projects/folders"
PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole
KIND_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
#: True on the active project's line (its folder icon is drawn in the accent)
ACTIVE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2
PROJECT = "project"


def project_names(folders):
    """{folder: (name, where)}: each project's name, and for names two projects share, the folder above
    it ("Data", "GIOVA"), going further up until the two differ; `where` is "" for a name used once."""
    names = {str(f): pr.display_name(f) for f in folders}
    found = {}
    for folder, name in names.items():
        twins = [f for f, n in names.items() if n == name and f != folder]
        where = ""
        if twins:
            mine = Path(folder).parts[:-1]
            depth = 1
            while depth < len(mine) and any(Path(t).parts[:-1][-depth:] == mine[-depth:] for t in twins):
                depth += 1
            where = "\\".join(mine[-depth:]).strip("\\") or str(Path(folder).parent)
        found[folder] = (name, where)
    return found


def stored_folders():
    """The folders added as projects, in the order they were added."""
    try:
        folders = json.loads(str(user_settings().value(FOLDERS_KEY, "") or "[]"))
    except ValueError:
        folders = []
    return [str(f) for f in folders if isinstance(f, str) and f]


def store_folders(folders):
    user_settings().setValue(FOLDERS_KEY, json.dumps(list(folders), ensure_ascii=False))


def writable(folder):
    """None when recordings can be written to `folder`, else why not."""
    probe = Path(folder) / ".eegstudio_write_test"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return str(exc)
    return None


class ProjectInfoDialog(QtWidgets.QDialog):
    """A few optional words about a project, kept in the project's own folder."""

    def __init__(self, parent, folder):
        super().__init__(parent)
        self.folder = Path(folder)
        self.setWindowTitle("Project info")
        self.resize(460, 320)
        info = pr.load_info(folder)
        form = QtWidgets.QFormLayout(self)
        where = QtWidgets.QLabel(str(self.folder))
        where.setWordWrap(True)
        where.setStyleSheet(muted())
        form.addRow("Folder", where)
        self.name = QtWidgets.QLineEdit(info.get("name", ""))
        self.name.setPlaceholderText(self.folder.name)
        form.addRow("Name", self.name)
        self.participant = QtWidgets.QLineEdit(info.get("participant", ""))
        form.addRow("Participant", self.participant)
        self.notes = QtWidgets.QPlainTextEdit(info.get("notes", ""))
        self.notes.setPlaceholderText("Protocol, electrodes, anything worth knowing later")
        form.addRow("Notes", self.notes)
        note = QtWidgets.QLabel("All optional. Saved as {} in the project folder; leave everything "
                                "empty and no file is kept.".format(pr.INFO_FILE))
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        form.addRow(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return dict(name=self.name.text(), participant=self.participant.text(), notes=self.notes.toPlainText())


class ProjectPanel(QtWidgets.QWidget):
    #: a recording was chosen to be opened: its path
    recording_activated = QtCore.Signal(str)
    #: a saved analysis was chosen: (recording path, analysis path)
    analysis_activated = QtCore.Signal(str, str)
    #: the folder new recordings go to has changed
    active_changed = QtCore.Signal(str)
    #: explicit requests from the right-click menu
    open_in_analysis = QtCore.Signal(str)
    play_in_monitor = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(4, 0, 4, 4)
        column.setSpacing(4)
        self.active_label = QtWidgets.QLabel()
        self.active_label.setWordWrap(True)
        self.active_label.setStyleSheet("color: {};".format(color("panel_text")))
        self.active_label.setToolTip("New live recordings are written to the active project's folder")
        column.addWidget(self.active_label)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        # the second column tells two projects of one name apart, quietly ("Data   GIOVA")
        self.tree.setColumnCount(2)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setIndentation(14)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        column.addWidget(self.tree, 1)
        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(muted())
        column.addWidget(self.detail)
        row = QtWidgets.QHBoxLayout()
        # icons alone, like every toolbar in GIOVA: the side panel is narrow and the words did not fit
        self.add_button = only(QtWidgets.QToolButton(), "plus", "Add project",
                               more="Choose a folder: it is listed here with the recordings inside it. "
                                    "Nothing is copied or moved.")
        self.add_button.clicked.connect(self.add_project)
        self.use_button = only(QtWidgets.QToolButton(), "circle-check", "Make active",
                               more="Write new live recordings to the selected project's folder")
        self.use_button.clicked.connect(lambda: self.activate(self.selected_project()))
        self.use_button.setEnabled(False)
        refresh = QtWidgets.QToolButton()
        put(refresh, "refresh-cw", "", size=16)          # a familiar compact action: the icon alone
        refresh.setAccessibleName("Refresh")
        refresh.setToolTip("Read the project folders again")
        refresh.clicked.connect(self.refresh)
        for widget in (self.add_button, self.use_button, refresh):
            row.addWidget(widget)
        row.addStretch(1)
        column.addLayout(row)
        self.refresh()

    # ---------------- the list of projects ----------------

    def folders(self):
        """Stored projects, with the folder recordings go to now always among them."""
        folders = stored_folders()
        active = str(data_dir())
        if not any(pr.same_folder(active, f) for f in folders):
            folders.insert(0, active)
        return folders

    def add_project(self, folder=None):
        if not folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of the project", str(data_dir()))
        if not folder:
            return None
        folder = str(Path(folder))
        folders = self.folders()
        if not any(pr.same_folder(folder, f) for f in folders):
            folders.append(folder)
        store_folders(folders)
        self.refresh()
        self.select_project(folder)
        self.status.emit("Project added: {}".format(pr.display_name(folder)))
        return folder

    def remove_project(self, folder):
        """Take a project off the list. Its folder and files are not touched."""
        if folder is None:
            return
        if pr.same_folder(folder, data_dir()):
            self.status.emit("The active project stays on the list: make another one active first")
            return
        store_folders([f for f in self.folders() if not pr.same_folder(f, folder)])
        self.refresh()
        self.status.emit("Removed from the list (files untouched): {}".format(Path(folder).name))

    def activate(self, folder):
        """Make `folder` the place new live recordings are written to."""
        if folder is None:
            return False
        folder = Path(folder)
        problem = None if folder.is_dir() else "the folder is not there"
        problem = problem or writable(folder)
        if problem:
            QtWidgets.QMessageBox.warning(self, "Project", "Cannot record into\n{}\n\n{}".format(folder, problem))
            return False
        store_folders(self.folders())            # the project left behind stays on the list
        set_setting("recording/folder", str(folder))
        if not any(pr.same_folder(folder, f) for f in stored_folders()):
            store_folders(stored_folders() + [str(folder)])
        self.refresh()
        self.select_project(folder)
        self.active_changed.emit(str(folder))
        self.status.emit("Active project: {} · new recordings go to {}".format(pr.display_name(folder), folder))
        return True

    def edit_info(self, folder):
        if folder is None:
            return
        dialog = ProjectInfoDialog(self, folder)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            try:
                pr.save_info(folder, dialog.values())
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, "Project info", "Cannot save:\n{}".format(exc))
                return
            self.refresh()
            self.select_project(folder)

    # ---------------- the tree ----------------

    def refresh(self):
        """Read every project folder again, keeping what was open and selected."""
        opened = {item.data(0, PATH_ROLE) for item in self._items() if item.isExpanded()}
        chosen = self.tree.currentItem().data(0, PATH_ROLE) if self.tree.currentItem() else None
        first = self.tree.topLevelItemCount() == 0
        self.tree.clear()
        active = data_dir()
        bold = QtGui.QFont(self.tree.font())
        bold.setBold(True)
        folders = self.folders()
        names = project_names(folders)
        for folder in folders:
            info = pr.load_info(folder)
            is_active = pr.same_folder(folder, active)
            name, where = names[str(folder)]
            item = QtWidgets.QTreeWidgetItem([name, where])
            item.setData(0, PATH_ROLE, str(folder))
            item.setData(0, KIND_ROLE, PROJECT)
            item.setData(0, ACTIVE_ROLE, is_active)
            item.setIcon(0, tinted("folder", color("accent")) if is_active else icon("folder", role="subtle"))
            item.setFont(0, bold)
            item.setForeground(1, QtGui.QBrush(QtGui.QColor(color("subtle"))))
            nodes = pr.tree(folder)
            tip = [str(folder), "{} recording(s)".format(pr.count_recordings(nodes))]
            if not Path(folder).is_dir():
                tip.append("This folder is not there now (moved, renamed, or a drive that is not connected).")
                item.setForeground(0, QtGui.QBrush(QtGui.QColor(color("dim"))))
            if is_active:
                tip.append("Active: new live recordings are written here.")
            tip += ["{}: {}".format(key.capitalize(), info[key]) for key in ("participant", "notes") if key in info]
            item.setToolTip(0, "\n".join(tip))
            item.setToolTip(1, item.toolTip(0))
            self._fill(item, nodes)
            self.tree.addTopLevelItem(item)
            item.setExpanded(str(folder) in opened or (first and is_active))
        for item in self._items():
            if item.parent() is not None and item.data(0, PATH_ROLE) in opened:
                item.setExpanded(True)
            if chosen is not None and item.data(0, PATH_ROLE) == chosen:
                self.tree.setCurrentItem(item)
        name, where = next((names[f] for f in names if pr.same_folder(f, active)), (pr.display_name(active), ""))
        self.active_label.setText("Active project: <b>{}</b>{}".format(
            html.escape(name), " <span style='color: {};'>{}</span>".format(color("subtle"), html.escape(where)) if where else ""))
        self._selection_changed()

    def _fill(self, parent, nodes):
        for node in nodes:
            item = QtWidgets.QTreeWidgetItem([node.name if node.kind != pr.ANALYSIS else "saved analysis"])
            item.setData(0, PATH_ROLE, str(node.path))
            item.setData(0, KIND_ROLE, node.kind)
            item.setToolTip(0, str(node.path) + ("\nDouble-click to open" if node.kind != pr.FOLDER else ""))
            if node.kind == pr.FOLDER:
                item.setForeground(0, QtGui.QBrush(QtGui.QColor(color("header_text"))))
            elif node.kind == pr.ANALYSIS:
                item.setForeground(0, QtGui.QBrush(QtGui.QColor(color("link"))))
            parent.addChild(item)
            self._fill(item, node.children)

    def _items(self):
        found, stack = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            found.append(item)
            stack.extend(item.child(i) for i in range(item.childCount()))
        return found

    def find(self, path):
        """The tree line for a file or folder, or None."""
        path = str(path)
        return next((item for item in self._items() if item.data(0, PATH_ROLE) == path), None)

    def select_project(self, folder):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if pr.same_folder(item.data(0, PATH_ROLE), folder):
                self.tree.setCurrentItem(item)
                item.setExpanded(True)
                return item
        return None

    def selected_project(self):
        """The project the selected line belongs to, or None."""
        item = self.tree.currentItem()
        while item is not None and item.parent() is not None:
            item = item.parent()
        return item.data(0, PATH_ROLE) if item is not None else None

    def _selection_changed(self):
        project = self.selected_project()
        self.use_button.setEnabled(project is not None and not pr.same_folder(project, data_dir()))
        if project is None:
            self.detail.setText("")
            return
        info = pr.load_info(project)
        self.detail.setText(" · ".join(
            [str(project)] + ["{}: {}".format(key, info[key]) for key in ("participant", "notes") if key in info]))

    # ---------------- opening ----------------

    def _double_clicked(self, item, _column=0):
        kind, path = item.data(0, KIND_ROLE), item.data(0, PATH_ROLE)
        if kind == pr.RECORDING:
            self.recording_activated.emit(path)
        elif kind == pr.ANALYSIS:
            self.analysis_activated.emit(item.parent().data(0, PATH_ROLE), path)
        else:
            item.setExpanded(not item.isExpanded())

    def _menu(self, point):
        item = self.tree.itemAt(point)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        kind, path = item.data(0, KIND_ROLE), item.data(0, PATH_ROLE)
        menu = QtWidgets.QMenu(self)
        if kind == pr.RECORDING:
            menu.addAction("Quick analysis…", lambda: self.open_in_analysis.emit(path))
            menu.addAction("Play in Monitor", lambda: self.play_in_monitor.emit(path))
            menu.addSeparator()
            menu.addAction("Rename recording…", lambda: self.rename_recording(path))
            menu.addAction("Channel names…", lambda: self.channel_names(path))
            menu.addAction("Event names…", lambda: self.event_names(path))
            if Path(path).suffix.lower() == ".mat":
                menu.addAction("MAT layout / dataset preset…", lambda: self.mat_layout(path))
        elif kind == pr.ANALYSIS:
            menu.addAction("Open on the Workbench",
                           lambda: self.analysis_activated.emit(item.parent().data(0, PATH_ROLE), path))
        elif kind == PROJECT:
            use = menu.addAction("Make active (record here)", lambda: self.activate(path))
            use.setEnabled(not pr.same_folder(path, data_dir()))
            menu.addAction("Project info…", lambda: self.edit_info(path))
        menu.addAction("Show in folder", lambda: self._reveal(path))
        if kind == PROJECT:
            menu.addSeparator()
            remove = menu.addAction("Remove from this list (files stay)", lambda: self.remove_project(path))
            remove.setEnabled(not pr.same_folder(path, data_dir()))
        menu.exec(self.tree.viewport().mapToGlobal(point))

    def _reveal(self, path):
        from .settings_dialog import reveal
        reveal(path)

    # ---------------- a recording's own files ----------------
    # Each writes beside the recording (its name, a sidecar): a pipeline reading the file sees the change at its
    # next run, as its Recording block watches those files.

    def rename_recording(self, path, new=None):
        """The recording renamed, its sidecars (channel names, events, hardware log) with it."""
        from ..core import sources
        if new is None:
            new, ok = QtWidgets.QInputDialog.getText(self, "Rename recording", "New name (no extension):",
                                                     text=Path(path).stem)
            if not ok or not new.strip():
                return None
        companions = len(sources.sidecar_paths(Path(path)))
        try:
            moved = sources.rename_recording(Path(path), new.strip())
        except ValueError as exc:
            self.status.emit(str(exc))
            return None
        self.refresh()
        self.status.emit("Renamed to {}{}".format(moved.name, " (with {} companion file{})".format(
            companions, "" if companions == 1 else "s") if companions else ""))
        return moved

    def channel_names(self, path, names=None):
        """The electrodes' names, saved beside the recording (10-20 names place them on head maps)."""
        from ..formats.channels import save_names
        from ..formats.recordings import load_recording
        if names is None:
            from .channelnames import ChannelNamesDialog
            try:
                recording = load_recording(path)
            except Exception as exc:                     # any reader's failure: said, never a crash of the pane
                self.status.emit("Cannot read {}: {}".format(Path(path).name, exc))
                return None
            dialog = ChannelNamesDialog(self, recording.channel_names)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return None
            names = dialog.chosen
        save_names(path, names)
        self.status.emit("Channel names saved beside {}".format(Path(path).name))
        return names

    def event_names(self, path, mapping=None):
        """What each event code of the recording means, saved beside it."""
        from ..core import epochs as ep
        if mapping is None:
            from ..formats.recordings import load_recording
            from .eventdialog import EventMapDialog
            try:
                raw, source = ep.raw_events_for(load_recording(path), path)
            except Exception as exc:
                self.status.emit("Cannot read {}: {}".format(Path(path).name, exc))
                return None
            if raw is None:
                self.status.emit(source)
                return None
            mapping = EventMapDialog.ask(self, raw, ep.load_label_map(path) or {})
            if mapping is None:
                return None
        ep.save_label_map(path, mapping)
        self.status.emit("Event names saved beside {}".format(Path(path).name))
        return mapping

    def mat_layout(self, path):
        """A MATLAB file's layout (or dataset preset) chosen again."""
        from .matdialog import remap
        recording, note = remap(self, path)
        if note:
            self.status.emit(note)
        return recording
