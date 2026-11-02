"""The application look: the stylesheet parses, and it is only a look."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

from pyqtgraph.Qt import QtCore, QtWidgets

from eeg.ui import theme

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class ThemeTests(unittest.TestCase):
    def test_stylesheet_parses(self):
        """Qt drops a stylesheet it cannot parse with only a console warning."""
        messages = []
        previous = QtCore.qInstallMessageHandler(lambda kind, context, text: messages.append(text))
        try:
            old = app.styleSheet()
            app.setStyleSheet(theme.app_stylesheet())
            button = QtWidgets.QPushButton("x")
            button.setProperty("primary", True)
            button.show()
            app.processEvents()
            button.close()
            app.setStyleSheet(old)
        finally:
            QtCore.qInstallMessageHandler(previous)
        self.assertEqual([m for m in messages if "style" in m.lower() or "parse" in m.lower()], [])

    def test_both_themes_parse_and_paper_is_always_light(self):
        import pyqtgraph as pg
        before = theme._dark
        messages = []
        previous = QtCore.qInstallMessageHandler(lambda kind, context, text: messages.append(text))
        try:
            for name in theme.THEMES:
                theme.set_theme(name)
                self.assertEqual(theme.is_dark(), name == "dark")
                widget = QtWidgets.QWidget()
                widget.setStyleSheet(theme.app_stylesheet())
                widget.show()
                app.processEvents()
                widget.close()
                foreground = pg.getConfigOption("foreground")
                with theme.paper():
                    self.assertFalse(theme.is_dark())
                    self.assertEqual(theme.plot_background(), theme.PAPER["plot"])
                    self.assertEqual(pg.getConfigOption("foreground"), theme.PAPER["plot_text"])
                self.assertEqual(pg.getConfigOption("foreground"), foreground)
                self.assertEqual(theme.is_dark(), name == "dark")
            theme.set_theme("dark")
            self.assertNotEqual(theme.color("text"), theme.LIGHT["text"])
            self.assertEqual(set(theme.DARK), set(theme.LIGHT))          # every role in both
            self.assertEqual(set(theme.PAPER), set(theme.LIGHT))         # and on paper
            with theme.paper():
                self.assertEqual(theme.color("plot"), "#ffffff")         # a page is white, whatever the window
        finally:
            QtCore.qInstallMessageHandler(previous)
            theme._dark = before
        self.assertEqual([m for m in messages if "style" in m.lower() or "parse" in m.lower()], [])

    def test_a_sheet_tab_closes_with_a_quiet_mark_shown_only_where_it_is_wanted(self):
        from eeg.ui.workbench.sheets import SheetTabBar
        bar = SheetTabBar()
        for name in ("one", "two", "three"):
            bar.addTab(name)
        bar.setCurrentIndex(0)
        side = QtWidgets.QTabBar.ButtonPosition.RightSide
        marks = [bar.tabButton(i, side) for i in range(3)]
        self.assertEqual([m.graphicsEffect().opacity() for m in marks], [1.0, 0.0, 0.0])
        asked = []
        bar.tabCloseRequested.connect(asked.append)
        marks[2].click()
        self.assertEqual(asked, [2])

    def test_the_window_is_dark_unless_set_light(self):
        from eeg.ui.preferences import DEFAULTS
        self.assertEqual(DEFAULTS["ui/theme"][0], "dark")

    def test_fields_are_giovas_number_fields_without_arrows_lists_with_a_chevron(self):
        from pathlib import Path
        sheet = theme.app_stylesheet()
        self.assertIn("QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0;".replace("{{", "{"),
                      sheet)
        arrow = sheet.split("QComboBox::down-arrow { image: url(")[1].split(")")[0]
        self.assertTrue(Path(arrow).is_file())                     # the chevron the list opens from
        from eeg.ui.workbench.inspector import ParamField
        from eeg.pipeline import block
        field = ParamField(block("filter").params[0], 1.0)
        self.assertEqual((field.label, field.widget.suffix()), ("Low cut", " Hz"))   # the unit in the field

    def test_the_program_carries_inter_and_jetbrains_mono(self):
        from eeg.ui.fonts import load_fonts, mono
        found = load_fonts(app)
        self.assertTrue({"Inter", "JetBrains Mono"} <= found, found)
        self.assertEqual(app.font().family(), "Inter")
        self.assertEqual(mono().family(), "JetBrains Mono")

    def test_brand_label(self):
        label = theme.brand_label()
        texts = [w.text() for w in label.findChildren(QtWidgets.QLabel)]
        self.assertTrue(any("GIOVA" in t for t in texts))

    def test_main_window_has_brand_and_primary_buttons(self):
        from eeg.ui.app import MainWindow
        window = MainWindow()
        self.assertFalse(window.brand.pixmap().isNull())          # the mark at the top of the rail
        self.assertTrue(window.monitor.panel.start_button.property("primary"))
        self.assertTrue(window.monitor.panel.hardware.apply_button.property("primary"))
        window.close()

    def test_the_rail_switches_modes_and_opens_one_side_panel_at_a_time(self):
        from eeg.ui.app import MainWindow
        from eeg.ui import preferences as prefs
        store = prefs.user_settings()
        keys = ("window/panel/Monitor", "window/panel/Workbench")
        before = {key: store.value(key) for key in keys}
        for key in keys:
            store.remove(key)
        window = MainWindow()
        try:
            window.resize(1600, 900)
            window.show()
            QtWidgets.QApplication.processEvents()
            rail = window.rail
            self.assertLess(window.brand.mapTo(window, QtCore.QPoint(0, 0)).x(), 20)   # at the left edge
            # icons alone: the name is in the tooltip and the accessible name
            for button in list(rail.modes.values()) + list(rail.panels.values()):
                self.assertEqual(button.text(), "")
                self.assertTrue(button.toolTip() and button.accessibleName())
            self.assertTrue(window.tabs.tabBar().isHidden())
            rail.modes["Monitor"].click()
            self.assertIs(window.tabs.currentWidget(), window.monitor)
            self.assertTrue(rail.modes["Monitor"].isChecked())
            self.assertEqual(window.side.current, "Projects")         # Monitor offers Projects alone
            self.assertEqual([k for k, b in rail.panels.items() if b.isVisibleTo(rail)], ["Projects"])
            rail.panels["Projects"].click()                           # the open one: folds away
            self.assertTrue(window.side.isHidden())
            rail.panels["Projects"].click()
            self.assertEqual(window.side.current, "Projects")
            window.mode_actions["Workbench"].trigger()
            self.assertEqual(window.side.current, "Blocks")           # the Workbench opens on its blocks
            name = window.workbench.name                              # the pipeline's name fits its box
            self.assertGreaterEqual(name.width(), name.fontMetrics().horizontalAdvance(name.text()))
            self.assertTrue(window.side.stack.currentWidget().isAncestorOf(window.workbench.palette))
            # pointing at a block shows it below the list; nothing is added to the sheet
            blocks = len(window.workbench.graph.nodes)
            window.workbench.palette.hovered.emit("filter")
            self.assertEqual(window.workbench.preview.kind.title, "Filter")
            self.assertEqual(len(window.workbench.graph.nodes), blocks)
            window.workbench.palette.hovered.emit("")
            self.assertIsNone(window.workbench.preview.kind)
            rail.panels["Library"].click()
            self.assertIs(window.side.stack.currentWidget(), window.workbench.library)
            self.assertTrue(rail.panels["Library"].isChecked() and not rail.panels["Blocks"].isChecked())
            rail.modes["Home"].click()
            self.assertTrue(window.side.isHidden())
            rail.modes["Workbench"].click()
            self.assertEqual(window.side.current, "Library")         # each mode keeps the panel it had open
        finally:
            for key, value in before.items():
                if value is None:
                    store.remove(key)
                else:
                    store.setValue(key, value)
            window.close()


if __name__ == "__main__":
    unittest.main()
