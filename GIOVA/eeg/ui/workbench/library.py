"""The Library tab beside the blocks: every pipeline there is to start from, in one list.

Mine: the `.pipeline.json` files saved in the project folders, by project, and those opened or
saved elsewhere lately. Templates: the ready-made pipelines. Literature: the published methods.
Analysis profiles: the ready set-ups of a quick analysis (built in, read-only, and the user's own).
Each with what helps to choose one - when it was saved, how many blocks, the last accuracy.
Double-click opens it in a tab of its own; nothing open is replaced.

No pipeline is computed here; a file is only read for its name and its blocks.
"""
import json
import os
from pathlib import Path
import time

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ...core import projects as pr
from ...pipeline import templates
from ...pipeline.graph import SUFFIX
from .. import recent
from ..preferences import data_dir
from ..theme import color, muted

MINE, TEMPLATES, LITERATURE, PROFILES = "Mine", "Templates", "Literature", "Analysis profiles"
#: How deep into a project folder, and how many entries at most, pipelines are looked for:
#: a folder of thousands of recordings must not hold the tab up.
DEPTH, MOST_ENTRIES = 4, 4000
#: Item roles: what an entry opens (a file path, or a template / literature title) and its kind.
TARGET_ROLE = QtCore.Qt.ItemDataRole.UserRole
KIND_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


def find_pipelines(folder, depth=DEPTH, most=MOST_ENTRIES):
    """The pipeline files under `folder`, at most `depth` folders down, looking at `most` entries at most."""
    found, seen = [], 0
    root = Path(folder)
    if not root.is_dir():
        return found
    for here, folders, files in os.walk(root):
        level = len(Path(here).relative_to(root).parts)
        if level >= depth:
            folders[:] = []
        folders[:] = [f for f in folders if not f.startswith(".")]
        seen += len(files) + len(folders)
        found += [Path(here) / f for f in files if f.endswith(SUFFIX)]
        if seen > most:
            break
    return found


def describe_file(path):
    """(name, number of blocks, seconds since it was saved) of a pipeline file; None if it cannot be read."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return raw.get("name") or Path(path).name, len(raw.get("nodes", [])), time.time() - Path(path).stat().st_mtime
    except (OSError, ValueError, AttributeError):
        return None


class Library(QtWidgets.QWidget):
    """Search, then the three groups. `open_file(path)` and `open_ready_made(title)` say what was chosen."""

    open_file = QtCore.Signal(str)
    open_ready_made = QtCore.Signal(str)
    #: an analysis profile's name: a quick analysis with it
    open_profile = QtCore.Signal(str)

    def __init__(self, project_folders=None):
        super().__init__()
        #: () -> the project folders to look in; the active one when not given
        self.project_folders = project_folders or (lambda: [str(data_dir())])
        #: where the user's analysis profiles are kept; None: the program's profiles folder
        self.profile_root = None
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search pipelines")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.setIndentation(12)
        self.tree.setRootIsDecorated(True)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemActivated.connect(self._activated)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.hint = QtWidgets.QLabel("Double-click to open in a new tab")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(muted())
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.search)
        column.addWidget(self.tree, 1)
        column.addWidget(self.hint)
        self.refresh()

    # ---------------- filling ----------------

    def mine(self):
        """[(project name, [path, ...])]: pipelines in each project folder, then those saved or opened
        elsewhere lately; each file once."""
        groups, listed = [], set()
        for folder in self.project_folders():
            paths = [p for p in sorted(find_pipelines(folder), key=lambda p: -p.stat().st_mtime)
                     if str(p.resolve()) not in listed]
            listed |= {str(p.resolve()) for p in paths}
            if paths:
                groups.append((pr.display_name(folder), paths))
        elsewhere = [Path(e["path"]) for e in recent.entries(recent.PIPELINES) if e["path"] not in listed]
        if elsewhere:
            groups.append(("Elsewhere, used lately", elsewhere))
        return groups

    def refresh(self):
        """Read the folders again, keeping what was opened and the search."""
        closed = {self.tree.topLevelItem(i).text(0) for i in range(self.tree.topLevelItemCount())
                  if not self.tree.topLevelItem(i).isExpanded()}
        self.tree.clear()
        accuracy = {e["path"]: e.get("accuracy") for e in recent.entries(recent.PIPELINES)}
        mine = self._heading(MINE)
        groups = self.mine()
        for project, paths in groups:
            parent = mine
            if len(groups) > 1:
                parent = self._heading(project, sub=True)
                mine.addChild(parent)
            for path in paths:
                info = describe_file(path)
                if info is None:
                    continue
                name, blocks, age = info
                score = accuracy.get(str(path.resolve()))
                detail = recent.ago(time.time() - age) + ("  ·  {:.1%}".format(score) if score is not None else "")
                item = self._entry(name, detail, str(path), MINE)
                item.setToolTip(0, "{}\n{} block{}{}".format(path, blocks, "" if blocks == 1 else "s",
                                                             "\nLast accuracy {:.1%}".format(score)
                                                             if score is not None else ""))
                parent.addChild(item)
        if not groups:
            empty = QtWidgets.QTreeWidgetItem(["Nothing saved yet: Save puts a pipeline here"])
            empty.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            mine.addChild(empty)
        ready = self._heading(TEMPLATES)
        for title, build in templates.TEMPLATES.items():
            graph = build()
            item = self._entry(title, "{} blocks".format(len(graph.nodes)), title, TEMPLATES)
            item.setToolTip(0, graph.description or title)
            ready.addChild(item)
        from ...pipeline import literature
        papers = self._heading(LITERATURE)
        for title, build in literature.LIBRARY.items():
            graph = build()
            item = self._entry(title, "{} blocks".format(len(graph.nodes)), title, LITERATURE)
            item.setToolTip(0, graph.source.get("citation", title))
            papers.addChild(item)
        from ...core import analysis_profiles as ap
        profiles = self._heading(PROFILES)
        for profile in ap.available(self.profile_root):
            built_in = profile.name in ap.BUILTIN_NAMES
            item = self._entry(profile.name, "built in" if built_in else "yours", profile.name, PROFILES)
            item.setToolTip(0, "{}\nDouble-click: a quick analysis with this profile, in a new tab".format(
                profile.description or profile.name))
            profiles.addChild(item)
        for heading in (mine, ready, papers, profiles):
            self.tree.addTopLevelItem(heading)
            heading.setExpanded(heading.text(0) not in closed)
            for k in range(heading.childCount()):
                heading.child(k).setExpanded(True)
        self._filter(self.search.text())

    def _heading(self, text, sub=False):
        item = QtWidgets.QTreeWidgetItem([text])
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QtGui.QBrush(QtGui.QColor(color("header_text") if sub else color("text"))))
        return item

    @staticmethod
    def _entry(name, detail, target, kind):
        item = QtWidgets.QTreeWidgetItem([name, detail])
        item.setData(0, TARGET_ROLE, target)
        item.setData(0, KIND_ROLE, kind)
        item.setForeground(1, QtGui.QBrush(QtGui.QColor(color("subtle"))))
        return item

    def entries(self):
        """Every entry that opens something, wherever it sits in the tree."""
        found, stack = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            if item.data(0, TARGET_ROLE):
                found.append(item)
            stack += [item.child(k) for k in range(item.childCount())]
        return found

    def _filter(self, text):
        """Only entries whose name has every word typed; a heading with nothing left is hidden."""
        words = text.lower().split()

        def show(item):
            if item.data(0, TARGET_ROLE):
                visible = all(w in item.text(0).lower() for w in words)
            else:
                children = [show(item.child(k)) for k in range(item.childCount())]
                visible = any(children) or (not words and item.flags() == QtCore.Qt.ItemFlag.NoItemFlags)
                if words and visible:
                    item.setExpanded(True)
            item.setHidden(not visible)
            return visible
        for i in range(self.tree.topLevelItemCount()):
            show(self.tree.topLevelItem(i))

    # ---------------- choosing ----------------

    def choose(self, item):
        target, kind = item.data(0, TARGET_ROLE), item.data(0, KIND_ROLE)
        if not target:
            return
        if kind == MINE:
            self.open_file.emit(target)
        elif kind == PROFILES:
            self.open_profile.emit(target)
        else:
            self.open_ready_made.emit(target)

    def _activated(self, item, _column):
        self.choose(item)

    # ---------------- analysis profiles ----------------

    def import_profile(self, path=None):
        """A profile file (.profile.json) becomes one of the user's profiles; a built-in name is refused."""
        from ...core import analysis_profiles as ap
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import analysis profile", "",
                                                            "Analysis profiles (*.json)")
            if not path:
                return None
        try:
            profile = ap.load_profile(path)
            ap.save_user_profile(profile, self.profile_root)
        except (ValueError, OSError) as exc:
            self.hint.setText("Cannot import: {}".format(exc))
            return None
        self.refresh()
        self.hint.setText("Imported {}".format(profile.name))
        return profile

    def export_profile(self, name, path=None):
        """A profile, built in or the user's, written to a file to share."""
        from ...core import analysis_profiles as ap
        profile = next((p for p in ap.available(self.profile_root) if p.name == name), None)
        if profile is None:
            return None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export analysis profile",
                                                            ap.slug(name) + ap.SUFFIX, "Analysis profiles (*.json)")
            if not path:
                return None
        ap.save_profile(path, profile)
        self.hint.setText("Exported {}".format(Path(path).name))
        return path

    def copy_profile(self, name, new_name=None):
        """A profile saved again under a new name, as the user's own (a built-in one stays as it is)."""
        from ...core import analysis_profiles as ap
        from dataclasses import replace
        profile = next((p for p in ap.available(self.profile_root) if p.name == name), None)
        if profile is None:
            return None
        if new_name is None:
            new_name, ok = QtWidgets.QInputDialog.getText(self, "Save a copy as", "Name of your profile:",
                                                          text=name + " (mine)")
            if not ok or not new_name.strip():
                return None
        copy = replace(profile, name=new_name.strip())
        try:
            ap.save_user_profile(copy, self.profile_root)
        except ValueError as exc:
            self.hint.setText(str(exc))
            return None
        self.refresh()
        return copy

    def delete_profile(self, name):
        """One of the user's profiles removed; built-in ones cannot be."""
        from ...core import analysis_profiles as ap
        try:
            gone = ap.delete_user_profile(name, self.profile_root)
        except ValueError as exc:
            self.hint.setText(str(exc))
            return False
        self.refresh()
        return gone

    def _menu(self, point):
        item = self.tree.itemAt(point)
        if item is not None and item.text(0) == PROFILES and not item.data(0, TARGET_ROLE):
            menu = QtWidgets.QMenu(self)
            menu.addAction("Import a profile…", self.import_profile)
            menu.exec(self.tree.viewport().mapToGlobal(point))
            return
        if item is None or not item.data(0, TARGET_ROLE):
            return
        menu = QtWidgets.QMenu(self)
        if item.data(0, KIND_ROLE) == PROFILES:
            from ...core import analysis_profiles as ap
            name = item.data(0, TARGET_ROLE)
            menu.addAction("Quick analysis with it", lambda: self.choose(item))
            menu.addAction("Save a copy as mine…", lambda: self.copy_profile(name))
            menu.addAction("Export…", lambda: self.export_profile(name))
            menu.addAction("Import a profile…", self.import_profile)
            delete = menu.addAction("Delete", lambda: self.delete_profile(name))
            delete.setEnabled(name not in ap.BUILTIN_NAMES)          # built-in profiles are read-only
            menu.exec(self.tree.viewport().mapToGlobal(point))
            return
        menu.addAction("Open in a new tab", lambda: self.choose(item))
        if item.data(0, KIND_ROLE) == MINE:
            folder = str(Path(item.data(0, TARGET_ROLE)).parent)
            menu.addAction("Show in its folder",
                           lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder)))
        menu.exec(self.tree.viewport().mapToGlobal(point))
