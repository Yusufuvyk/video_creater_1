"""The main window: Home, and the modes - live Monitor, Workbench (quick analysis included), Games.

A mode's page is built the first time it is opened, and its module imported
then too: someone who only analyses never loads the live monitor, and the
window opens sooner. Code that needs a page asks for it by name
(`self.monitor`, `self.workbench`, ...), which builds it if it is not there yet;
code that only tidies up after a page (saving, closing) skips pages never built.
"""
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import recent
from .home import HomePage, QUICK, TITLES, start_in_last_mode

#: The pages, in their tab order. Home is built at once; the others when first opened.
PAGES = ("Home", "Monitor", "Workbench", "Games")
#: Modes earlier versions had, and the page they open now: a quick analysis is made on the Workbench.
RETIRED = {"Analysis": "Workbench"}
#: Shortcut of each mode; Quick analysis is a task on the Workbench.
SHORTCUTS = {"Home": "Ctrl+0", "Monitor": "Ctrl+1", "Workbench": "Ctrl+2", QUICK: "Ctrl+3", "Games": "Ctrl+4"}
#: The rail's modes and side panels, top to bottom, with their icons.
MODE_ICONS = (("Home", "house"), ("Monitor", "activity"), ("Workbench", "workflow"), ("Games", "gamepad-2"))
PANEL_ICONS = (("Projects", "folder"), ("Blocks", "blocks"), ("Library", "library"))
#: The side panels each mode offers, and the one open the first time; Home is the choice alone.
PANELS = {"Home": (), "Monitor": ("Projects",), "Workbench": ("Projects", "Blocks", "Library"),
          "Games": ("Projects",)}
FIRST_PANEL = {"Monitor": "Projects", "Workbench": "Blocks", "Games": ""}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        from .. import NAME
        self.setWindowTitle(NAME)
        from .widgets import fit_to_screen
        #: True when the screen had no room for the full size: open maximised
        self.screen_is_small = fit_to_screen(self, 1600, 950)

        from .theme import plot_background, plot_foreground
        pg.setConfigOptions(antialias=True, background=plot_background(), foreground=plot_foreground())

        self._pages = {}
        self._views_restored = False
        self.home = HomePage()
        self.home.chosen.connect(self.open_mode)
        self.home.task.connect(self.run_task)
        self.home.reopen.connect(self.reopen_recent)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.home, "Home")
        for name in PAGES[1:]:
            self.tabs.addTab(QtWidgets.QWidget(), name)     # stands in until the page is opened
        self.tabs.tabBar().hide()
        # The rail down the left edge: the modes, the side panels, Help and Settings, icons alone.
        # Every mode keeps its shortcut, also while its icon is out of sight.
        from .rail import IconRail, SidePanel, mark
        self.brand = mark()
        self.rail = IconRail([(name, shape, TITLES.get(name, name), SHORTCUTS.get(name, "Ctrl+0"))
                              for name, shape in MODE_ICONS],
                             [(name, shape, name, "") for name, shape in PANEL_ICONS], mark=self.brand)
        self.rail.mode_chosen.connect(self.open_mode)
        self.rail.panel_chosen.connect(self.toggle_panel)
        self.rail.help_clicked.connect(lambda: self.open_help())
        self.rail.settings_clicked.connect(lambda: self.open_settings())
        self.help_button, self.settings_button = self.rail.help_button, self.rail.settings_button
        self.mode_actions = {}
        for name in ("Monitor", "Workbench", QUICK, "Games"):
            action = QtGui.QAction(TITLES[name], self)
            action.triggered.connect((lambda: self.run_task("analyse")) if name == QUICK
                                     else (lambda _checked=False, n=name: self.open_mode(n)))
            action.setShortcut(QtGui.QKeySequence(SHORTCUTS[name]))
            action.setShortcutContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            self.addAction(action)
            self.mode_actions[name] = action
        home_key = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self)
        home_key.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        home_key.activated.connect(lambda: self.open_mode("Home"))
        self.tabs.currentChanged.connect(self._page_shown)
        # Hover explanations: on or off in Settings → Interface.
        from .preferences import setting
        from .widgets import install_tooltip_switch
        install_tooltip_switch(QtWidgets.QApplication.instance(), setting('ui/tooltips'))
        # One side panel beside the rail: Projects in every mode (the project chosen and the recording
        # opened stay the same from Monitor to Workbench), the Workbench's Blocks and Library there.
        from .detach import ResizableSide
        from .preferences import user_settings
        from .project_panel import ProjectPanel
        self.projects = ProjectPanel()
        self.side = SidePanel()
        self.side.add("Projects", self.projects)
        self.side.shown.connect(self._panel_shown)
        self.shell = ResizableSide("window/side", user_settings())
        self.shell.add_side(self.side, "panel", 250, minimum=190, maximum=520)
        self.shell.add_main(self.tabs)
        central = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.rail)
        row.addWidget(self.shell, 1)
        self.setCentralWidget(central)
        self.projects.recording_activated.connect(self.open_recording)
        self.projects.open_in_analysis.connect(lambda path: self.quick_analysis(path))
        self.projects.play_in_monitor.connect(lambda path: self.open_recording(path, self.monitor))
        self.projects.analysis_activated.connect(self.open_saved_analysis)
        self.projects.active_changed.connect(self._active_project_changed)
        self.settings_dialog = None
        self.help_dialog = None
        help_key = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_F1), self)
        help_key.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        help_key.activated.connect(lambda: self.open_help())

        self.status = self.statusBar()
        self.projects.status.connect(lambda text: self.status.showMessage(text, 8000))
        # Always in view: a live recording goes to this project's folder.
        self.project_label = QtWidgets.QLabel()
        self.project_label.setToolTip("New live recordings are written to this project's folder "
                                      "(Projects pane, or Settings → Recording)")
        self.status.addPermanentWidget(self.project_label)
        self._show_active_project()
        self._show_optional_pages()
        self._page_shown(0)
        last = str(setting("ui/last_mode", "") or "")
        last = RETIRED.get(last, last)            # a mode of an earlier version: where it is now
        if start_in_last_mode() and last in PAGES[1:] and self._mode_available(last):
            self.open_mode(last)

    # ---------------- modes and pages ----------------

    def _build(self, name):
        """The page called `name`, made and joined to the rest of the window."""
        if name == "Monitor":
            from .monitor import MonitorPage
            page = MonitorPage()
            page.file_loaded.connect(lambda path: recent.remember(recent.RECORDINGS, path))
        elif name == "Workbench":
            from .workbench import WorkbenchPage
            page = WorkbenchPage()
        elif name == "Games":
            from .games_page import GamesPage
            page = GamesPage(self.monitor)
        else:
            raise KeyError(name)
        page.status_changed.connect(self.status.showMessage)
        return page

    def page(self, name):
        """The page called `name`, built now if it has not been opened yet."""
        if name == "Home":
            return self.home
        if name not in self._pages:
            page = self._build(name)
            self._pages[name] = page
            index = PAGES.index(name)
            current = self.tabs.currentIndex()
            stand_in = self.tabs.widget(index)
            self.tabs.blockSignals(True)
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, page, name)
            self.tabs.setCurrentIndex(current)
            self.tabs.blockSignals(False)
            stand_in.deleteLater()
            for key, widget in getattr(page, "panels", {}).items():
                self.side.add(key, widget)          # the Workbench's Blocks and Library
            self._show_optional_pages()
            if self._views_restored:
                self._restore_views_of(name)
        return self._pages[name]

    def built(self, name):
        """The page if it has been opened, else None: for tidying up, which needs no new page."""
        return self._pages.get(name)

    monitor = property(lambda self: self.page("Monitor"))
    workbench = property(lambda self: self.page("Workbench"))
    games = property(lambda self: self.page("Games"))

    def current_mode(self):
        return PAGES[self.tabs.currentIndex()]

    def open_mode(self, name):
        """Show a mode (or Home), building its page the first time."""
        page = self.page(name)
        self.tabs.setCurrentWidget(page)
        return page

    def run_task(self, key):
        """A task on Home: its mode, already set up for it."""
        if key == "record":
            return self.open_mode("Monitor")
        if key == "analyse":
            return self.quick_analysis()
        workbench = self.open_mode("Workbench")
        if key == "classify":
            workbench.new_pipeline("Motor imagery: CSP + LDA")
        elif key == "build":
            workbench.new_pipeline(None)
        elif key == "literature":
            from ..pipeline import literature, templates
            window = workbench.open_compare(with_sheet=False)          # not the sheet's pipeline as well
            window.add_pipeline(templates.csp_lda(), "template")
            for build in literature.LIBRARY.values():
                window.add_pipeline(build(), "literature")
            return window
        return workbench

    def reopen_recent(self, kind, path):
        """An entry under Home's "Continue": a recording in Quick analysis, a pipeline on the Workbench,
        a report in the program that shows its kind of file."""
        if kind == recent.RECORDINGS:
            return self.quick_analysis(path) is not None
        if kind == recent.PIPELINES:
            workbench = self.open_mode("Workbench")
            return workbench.open_pipeline(path)
        return QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _page_shown(self, index):
        name = PAGES[index]
        if name != "Home":
            self.page(name)
            from .preferences import user_settings
            user_settings().setValue("ui/last_mode", name)
        self.rail.set_mode(name)
        self._show_panel(self.open_panel(name))

    # ---------------- the side panel ----------------

    def open_panel(self, mode=None):
        """The side panel `mode` (the one in view) has open, "" when it is folded away."""
        mode = mode or self.current_mode()
        from .preferences import user_settings
        key = str(user_settings().value("window/panel/" + mode, FIRST_PANEL.get(mode, "")) or "")
        return key if key in PANELS.get(mode, ()) and key in self.side.pages else ""

    def toggle_panel(self, key):
        """A panel icon on the rail: open that panel, or fold it away if it is the one open."""
        mode = self.current_mode()
        key = "" if key == self.side.current or key not in PANELS.get(mode, ()) else key
        from .preferences import user_settings
        user_settings().setValue("window/panel/" + mode, key)
        self._show_panel(key)

    def _show_panel(self, key):
        mode = self.current_mode()
        self.rail.offer_panels([k for k in PANELS.get(mode, ()) if k in self.side.pages], key)
        was_hidden = self.side.isHidden()
        self.side.show_page(key)
        if was_hidden and key:
            # bound to the shell: a window closed before the timer fires takes the timer with it
            QtCore.QTimer.singleShot(0, self.shell, self.shell.apply_sizes)

    def _panel_shown(self, key):
        workbench = self.built("Workbench")
        if workbench is not None:
            workbench.panel_shown(key)

    def _mode_available(self, name):
        from .preferences import setting
        return name != "Games" or bool(setting('ui/show_games'))

    # ---------------- projects ----------------

    def open_recording(self, path, page=None):
        """Open a recording from the Projects pane on `page`, or on the page in view.

        Monitor plays it; the Workbench runs its pipeline on it (also from Home and Games, which have no use
        for a file).
        """
        if page is None:
            page = self.monitor if self.current_mode() == "Monitor" else self.workbench
        self.tabs.setCurrentWidget(page)
        if page is self.built("Monitor"):
            return page.open_file(path)
        recent.remember(recent.RECORDINGS, path)
        return page.open_recording(path)

    def quick_analysis(self, path=None, profile=None):
        """Quick analysis on the Workbench: a recording and a ready profile, in a new tab (asked for when not
        given). The new sheet, or None when cancelled."""
        return self.open_mode("Workbench").quick_analysis(path, profile)

    def open_saved_analysis(self, recording, saved):
        """A saved analysis session (.analysis.json) as a pipeline in a new Workbench tab, from its own trials and
        settings; the recording it names is checked, the file only read."""
        sheet = self.open_mode("Workbench").import_session(saved, recording)
        if sheet is None:
            self.status.showMessage("Cannot open the saved analysis {}: see the Workbench's status line".format(
                saved), 8000)
        return sheet is not None

    def _show_active_project(self):
        from ..core.projects import display_name
        from .preferences import data_dir
        self.project_label.setText("Project: {}  ".format(display_name(data_dir())))

    def _active_project_changed(self, _folder=None):
        """New recordings go elsewhere now: every place that names the folder follows."""
        self._show_active_project()
        monitor = self.built("Monitor")
        if monitor is not None and monitor.source is None:      # never under a running session
            monitor.panel.refresh_files()
        if getattr(self, "recording_settings", None) is not None:
            self.recording_settings.reload()

    def open_settings(self, page=None):
        """Show the Settings dialog, building it the first time.

        It is kept afterwards because it borrows widgets the Monitor owns:
        rebuilding it every time would move them back and forth.
        """
        if self.settings_dialog is None:
            from .settings_dialog import (AdvancedSettings, FileSettings, GraphSettings,
                                          InterfaceSettings, RecordingSettings, SettingsDialog)
            self.interface_settings = InterfaceSettings(on_games=self._show_optional_pages)
            self.graph_settings = GraphSettings(on_change=self._graph_units_changed)
            self.file_settings = FileSettings()
            self.recording_settings = RecordingSettings()
            # The folder chosen there is the active project: the pane follows.
            self.recording_settings.folder.textChanged.connect(lambda _text: self._folder_chosen_in_settings())
            self.settings_dialog = SettingsDialog(self, pages=[
                ("Recording", self.recording_settings),
                ("Interface", self.interface_settings),
                ("Graphs", self.graph_settings),
                ("Monitor", self.monitor.settings_page()),
                ("Files", self.file_settings),
                ("OpenBCI boards", self.monitor.board_settings_page()),
                ("Advanced", AdvancedSettings(on_reset=self._settings_reset)),
            ])
        # the pages that apply where it is opened; a page asked for by name is shown whatever the mode
        self.settings_dialog.open_at(page if isinstance(page, str) else None, self.current_mode())
        return self.settings_dialog

    def open_help(self, topic=None):
        """Show Help at `topic`, or at the topic for the page being looked at."""
        if self.help_dialog is None:
            from .help_dialog import HelpDialog
            self.help_dialog = HelpDialog(self)
        return self.help_dialog.open_at(topic or self.current_mode())

    def _show_optional_pages(self):
        """Settings → Interface: Games is there for those who use it."""
        shown = self._mode_available("Games")
        index = PAGES.index("Games")
        if not shown and self.tabs.currentIndex() == index:
            self.open_mode("Home")
        self.tabs.setTabVisible(index, shown)
        self.mode_actions["Games"].setVisible(shown)
        self.rail.set_mode_shown("Games", shown)
        self.home.show_games(shown)

    def _folder_chosen_in_settings(self):
        self.projects.refresh()
        self._show_active_project()

    def _graph_units_changed(self):
        """The PSD unit is display only: the Monitor and the Workbench's results redraw what they already have."""
        if self.built("Monitor") is not None:
            self.monitor.apply_graph_units()
        if self.built("Workbench") is not None:
            for widget in self.workbench.inspector.previews.values():
                if hasattr(widget, "draw"):
                    widget.draw()

    def _settings_reset(self):
        self.interface_settings.reload()
        self._show_optional_pages()
        self.graph_settings.reload()
        self.file_settings.reload()
        self.recording_settings.reload()
        self._folder_chosen_in_settings()
        self._graph_units_changed()
        if self.built("Monitor") is not None:
            self.monitor.apply_settings()
        self.status.showMessage("Settings restored to their defaults", 4000)

    def restore_views(self):
        """Reopen views that were in their own windows when the app last closed (pages opened later: then)."""
        self._views_restored = True
        if self.built("Monitor") is not None:
            self._restore_views_of("Monitor")

    def _restore_views_of(self, name):
        if name == "Monitor":
            self.monitor.view_tabs.restore()

    def closeEvent(self, event):
        # Popped-out views come back first, so nothing outlives the main window;
        # which ones were out is remembered for next time. Pages never opened have nothing to save.
        monitor = self.built("Monitor")
        if self.built("Games") is not None:
            self.games.shutdown()
        if monitor is not None:
            monitor.view_tabs.reattach_all()
        self.shell.save()
        if monitor is not None:
            monitor.save_preferences()
            monitor.detail.close()
            monitor.stop()
        if self.built("Workbench") is not None:
            self.workbench.shutdown()            # runs stopped, sheets in windows back as tabs, tabs remembered
        event.accept()


def use_light_theme(app):
    """The light theme for this run, whatever the setting (scripts that picture the window)."""
    from .theme import set_theme, use_theme
    set_theme("light")
    use_theme(app)


def main(splash=None):
    """Run the app. Entry points put the start-up screen up before importing
    this module (its imports are most of the start-up time) and pass it in."""
    from .splash import app_icon
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from .theme import use_theme
    from .fonts import load_fonts
    load_fonts(app)                 # Inter, carried with the program
    use_theme(app)                  # dark unless Settings → Interface says light
    from .widgets import install_wheel_guard
    install_wheel_guard(app)
    from .theme import window_is_dark
    app.setWindowIcon(app_icon(light=not window_is_dark()))   # the taskbar's tile follows the theme, as the splash does
    window = MainWindow()
    if window.screen_is_small:      # a laptop, or a screen at high Windows scaling
        window.showMaximized()
    else:
        window.show()
    window.restore_views()
    if splash is not None:
        splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
