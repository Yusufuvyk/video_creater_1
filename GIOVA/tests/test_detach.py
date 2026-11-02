"""Popped-out views and resizable side panels.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from eeg.ui.detach import DetachableTabs, ResizableSide  # noqa: E402
from eeg.ui.profile_box import MemorySettings  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def labels(tabs):
    return [tabs.tabText(i) for i in range(tabs.count())]


class DetachableTabsTests(unittest.TestCase):
    def setUp(self):
        self.prefs = MemorySettings()
        self.tabs = DetachableTabs("test/views", self.prefs)
        self.pages = {name: QtWidgets.QLabel(name) for name in ("A", "B", "C")}
        for name, page in self.pages.items():
            self.tabs.addTab(page, name)
        self.tabs.show()

    def tearDown(self):
        self.tabs.reattach_all(remember=False)
        self.tabs.close()
        self.tabs.deleteLater()
        app.processEvents()

    def test_pop_out_and_close_returns_page_to_its_place(self):
        window = self.tabs.pop_out("B")
        self.assertEqual(labels(self.tabs), ["A", "C"])
        self.assertTrue(window.isVisible())
        self.assertEqual(window.windowTitle(), "GIOVA · B")
        self.assertIs(self.pages["B"].window(), window)
        self.tabs.setCurrentIndex(1)
        self.assertEqual(self.tabs.visible_names(), ["C", "B"])
        self.tabs.pop_out("A")
        self.assertEqual(labels(self.tabs), ["C"])
        last = self.tabs.pop_out("C")                      # even the last one may leave
        self.assertEqual(labels(self.tabs), [])
        last.close()
        app.processEvents()
        self.assertEqual(labels(self.tabs), ["C"])
        window.close()
        app.processEvents()
        self.assertEqual(labels(self.tabs), ["B", "C"])
        self.tabs.reattach_all()
        self.assertEqual(labels(self.tabs), ["A", "B", "C"])
        self.assertEqual(self.prefs.value("test/views/detached"), "A")   # remembered for next start

    def test_restore_reopens_views_left_out(self):
        self.prefs.setValue("test/views/detached", "C")
        self.tabs.restore()
        self.assertTrue(self.tabs.is_detached("C"))
        self.assertEqual(labels(self.tabs), ["A", "B"])


def mouse(widget, kind, local, held=True):
    """Deliver one left-button mouse event to `widget` at a local point."""
    local = QtCore.QPointF(local)
    kinds = {"press": QtCore.QEvent.Type.MouseButtonPress, "move": QtCore.QEvent.Type.MouseMove,
             "release": QtCore.QEvent.Type.MouseButtonRelease}
    left, none = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton
    button = none if kind == "move" else left
    buttons = left if held and kind != "release" else none
    event = QtGui.QMouseEvent(kinds[kind], local, QtCore.QPointF(widget.mapToGlobal(local.toPoint())),
                              button, buttons, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, event)
    app.processEvents()


class DragTests(unittest.TestCase):
    """Chrome-style: drag a tab out into a window, drag the window's tab back."""

    def setUp(self):
        self.prefs = MemorySettings()
        self.tabs = DetachableTabs("test/drag", self.prefs)
        self.pages = {name: QtWidgets.QLabel(name) for name in ("A", "B", "C")}
        for name, page in self.pages.items():
            self.tabs.addTab(page, name)
        self.tabs.resize(700, 400)
        self.tabs.move(100, 100)
        self.tabs.show()
        app.processEvents()
        self.bar = self.tabs.tabBar()

    def tearDown(self):
        self.tabs.reattach_all(remember=False)
        self.tabs.close()
        self.tabs.deleteLater()
        app.processEvents()

    def tear(self, name, to=QtCore.QPoint(0, 300)):
        """Press on a tab, drag it well below the bar, and let go in the middle.

        The middle keeps the view in its own window; the sides dock it beside
        the other one (test_dropping_beside_docks_the_view)."""
        centre = self.bar.tabRect(labels(self.tabs).index(name)).center()
        mouse(self.bar, "press", centre)
        mouse(self.bar, "move", centre + QtCore.QPoint(0, 10))       # still sliding in the bar
        self.assertFalse(self.tabs.is_detached(name))
        mouse(self.bar, "move", centre + to)
        window = self.tabs.window_for(name)
        self.assertIsNotNone(window)
        grab = self.bar.mapToGlobal(centre + to)
        # The window has come off under the cursor, held by its own tab.
        self.assertTrue(window.tab.rect().contains(window.tab.mapFromGlobal(grab)))
        middle = QtCore.QPoint(self.tabs.width() // 2 - centre.x(), 40)
        mouse(self.bar, "move", centre + to + middle)
        self.assertTrue(window.tab.rect().contains(window.tab.mapFromGlobal(grab + middle)))
        mouse(self.bar, "release", centre + to + middle, held=False)
        return window

    def test_dragging_a_tab_out_opens_a_window_that_follows_the_mouse(self):
        window = self.tear("B")
        self.assertTrue(window.isVisible())
        self.assertEqual(labels(self.tabs), ["A", "C"])
        self.assertIs(self.pages["B"].window(), window)
        self.assertFalse(self.tabs._drag.active)          # the drag has ended ...
        mouse(self.bar, "move", QtCore.QPoint(5, 5), held=False)
        self.assertEqual(labels(self.tabs), ["A", "C"])   # ... and later mouse moves are the bar's again
        window.close()
        app.processEvents()
        self.assertEqual(labels(self.tabs), ["A", "B", "C"])

    def test_dropping_the_window_tab_on_the_bar_puts_it_where_it_lands(self):
        window = self.tear("C")
        chip = window.tab
        mouse(chip, "press", chip.rect().center())
        target = self.bar.mapToGlobal(self.bar.tabRect(0).topLeft() + QtCore.QPoint(3, 5))
        mouse(chip, "move", chip.rect().center() + QtCore.QPoint(30, 0))   # starts dragging the window
        mouse(chip, "move", chip.mapFromGlobal(target))
        # The window itself shows where it lands: the page under the bar.
        self.assertEqual(window.frameGeometry().topLeft(), self.tabs.zone_rect("tabs").topLeft())
        mouse(chip, "release", chip.mapFromGlobal(target), held=False)
        app.processEvents()
        self.assertFalse(self.tabs.is_detached("C"))
        self.assertEqual(labels(self.tabs), ["C", "A", "B"])
        self.assertEqual(self.tabs.tabText(self.tabs.currentIndex()), "C")
        # The new place is its home now: popping out and closing returns it there.
        self.tabs.pop_out("C").close()
        app.processEvents()
        self.assertEqual(labels(self.tabs), ["C", "A", "B"])

    def test_dropping_elsewhere_just_moves_the_window(self):
        window = self.tear("A")
        chip = window.tab
        before = window.pos()
        mouse(chip, "press", chip.rect().center())
        mouse(chip, "move", chip.rect().center() + QtCore.QPoint(60, 80))
        mouse(chip, "release", chip.rect().center(), held=False)
        self.assertTrue(self.tabs.is_detached("A"))
        self.assertEqual(window.pos(), before + QtCore.QPoint(60, 80))

    def test_sideways_drag_reorders_and_the_order_is_kept(self):
        self.bar.moveTab(0, 2)                      # what the bar does for a sideways drag
        self.assertEqual(labels(self.tabs), ["B", "C", "A"])
        self.tabs.pop_out("C").close()
        app.processEvents()
        self.assertEqual(labels(self.tabs), ["B", "C", "A"])

    def test_tab_dragged_sideways_then_out_is_the_one_torn_off(self):
        start = self.bar.tabRect(0).center()
        # one tab's step: A's far edge passes B's middle, not C's (past B's right edge, as before, slid A past C
        # too once the tabs were set in Inter, whose short names make narrow tabs)
        beyond_b = self.bar.tabRect(1).center().x() - start.x() + 2
        mouse(self.bar, "press", start)
        for dx in range(0, beyond_b, 10):
            mouse(self.bar, "move", start + QtCore.QPoint(dx, 0))
        self.assertEqual(labels(self.tabs), ["B", "A", "C"])   # A slid past B
        out = QtCore.QPoint(self.tabs.width() // 2 - start.x(), 300)      # the middle: no docking
        mouse(self.bar, "move", start + QtCore.QPoint(beyond_b, 300))
        mouse(self.bar, "move", start + out)
        mouse(self.bar, "release", start + out, held=False)
        self.assertTrue(self.tabs.is_detached("A"))
        self.assertEqual(labels(self.tabs), ["B", "C"])

    def test_a_tab_torn_off_at_the_side_does_not_snap_straight_back(self):
        # Torn off inside the left dock zone: it comes out under the cursor and
        # stays out when let go there, since it never showed a place to dock.
        self.tabs.pop_out("A"); self.tabs.pop_out("B")
        centre = self.bar.tabRect(0).center()
        mouse(self.bar, "press", centre)
        mouse(self.bar, "move", centre + QtCore.QPoint(0, 300))
        window = self.tabs.window_for("C")
        self.assertIsNone(self.tabs._drag.free_geometry)             # not snapped
        grab = self.bar.mapToGlobal(centre + QtCore.QPoint(0, 300))
        self.assertTrue(window.tab.rect().contains(window.tab.mapFromGlobal(grab)))
        mouse(self.bar, "release", centre + QtCore.QPoint(0, 300), held=False)
        self.assertEqual(labels(self.tabs), [])
        self.assertTrue(self.tabs.is_detached("C"))


class ResizableSideTests(unittest.TestCase):
    def test_side_width_is_remembered(self):
        prefs = MemorySettings()
        for expected in (300, 450):
            split = ResizableSide("test/side", prefs)
            side, main = QtWidgets.QWidget(), QtWidgets.QWidget()
            split.add_side(side, "left", 300, minimum=260, maximum=640)
            split.add_main(main)
            split.resize(1200, 600)
            split.show()
            app.processEvents()
            self.assertAlmostEqual(side.width(), expected, delta=8)
            split.setSizes([450, 750])
            split.save()
            split.close()
            split.deleteLater()

    def test_a_fixed_side_offers_no_resize_handle(self):
        split = ResizableSide("test/fixed", MemorySettings())
        main, side, free = QtWidgets.QWidget(), QtWidgets.QWidget(), QtWidgets.QWidget()
        split.add_side(free, "free", 300, minimum=260, maximum=640)
        split.add_main(main)
        split.add_side(side, "fixed", 320, minimum=320, maximum=320)
        fixed_handle, free_handle = split.handle(2), split.handle(1)
        self.assertFalse(fixed_handle.isEnabled())
        self.assertEqual(fixed_handle.cursor().shape(), QtCore.Qt.CursorShape.ArrowCursor)
        self.assertTrue(free_handle.isEnabled())                       # a resizable side keeps its own
        split.deleteLater()


class MonitorDetachTests(unittest.TestCase):
    def setUp(self):
        from eeg.ui.monitor import MonitorPage
        self.page = MonitorPage()
        # A layout the operator chose earlier is kept between sessions; these
        # tests are about the automatic one.
        self.page.preferences.setValue("monitor/views/layout", "auto")
        self.page.resize(1366, 730)
        self.page.show()
        app.processEvents()

    def tearDown(self):
        self.page.view_tabs.reattach_all(remember=False)
        self.page.stop()
        self.page.detail.close()
        self.page.close()
        QtCore.QThreadPool.globalInstance().waitForDone(5000)
        app.processEvents()

    def test_popped_out_spectrum_keeps_computing_when_page_is_hidden(self):
        from unittest.mock import patch
        from eeg.core.sources import FileSource, Recording
        from eeg.ui.playback import Player
        from test_live_monitor import sine
        page = self.page
        page.source = FileSource(Recording(np.tile(sine(seconds=12), (16, 1)), 125, page.panel.channel_names))
        page.player = Player(page.source, 125)
        page.panel.window_seconds.setValue(4)            # spectra from the 4 s just fed
        page._ingest(np.tile(sine(), (16, 1)))
        window = page.view_tabs.pop_out("Spectrum")
        app.processEvents()
        page.hide()
        self.assertTrue(page.multi_spectrum.isVisible())
        with patch.object(page.spectral_jobs, "start") as start:
            page.sample_position += 6
            with patch("eeg.ui.monitor.time.perf_counter", return_value=10.0):
                page._update_live_views()
            self.assertEqual(start.call_count, 1)
        window.close()
        app.processEvents()
        with patch.object(page.spectral_jobs, "start") as start:
            page.sample_position += 6
            with patch("eeg.ui.monitor.time.perf_counter", return_value=20.0):
                page._update_live_views()
            self.assertEqual(start.call_count, 0)       # nothing on screen, nothing computed

    def test_wide_layout_title_drags_out_and_back_like_a_tab(self):
        from eeg.ui.detach import DragTab
        page = self.page
        page.resize(1920, 1080)
        app.processEvents()
        self.assertTrue(page._wide)
        title = page.wide_slots["Spectrum"].findChild(DragTab)
        start = title.rect().center()
        mouse(title, "press", start)
        mouse(title, "move", start + QtCore.QPoint(0, 200))
        window = page.view_tabs.window_for("Spectrum")
        self.assertIsNotNone(window)
        self.assertFalse(page._wide)                     # the slot (and the title) are gone now ...
        self.assertFalse(title.isVisible())
        grab = title.mapToGlobal(start + QtCore.QPoint(300, 250))
        mouse(title, "move", start + QtCore.QPoint(300, 250))
        # ... yet the window keeps following the mouse
        self.assertTrue(window.tab.rect().contains(window.tab.mapFromGlobal(grab)))
        mouse(title, "release", start + QtCore.QPoint(300, 250), held=False)
        self.assertTrue(page.view_tabs.is_detached("Spectrum"))
        # Back onto the tab bar: both views side by side again.
        bar = page.view_tabs.tabBar()
        chip = window.tab
        target = bar.mapToGlobal(bar.tabRect(0).topRight() + QtCore.QPoint(-2, 5))
        mouse(chip, "press", chip.rect().center())
        mouse(chip, "move", chip.rect().center() + QtCore.QPoint(30, 0))
        mouse(chip, "move", chip.mapFromGlobal(target))
        mouse(chip, "release", chip.mapFromGlobal(target), held=False)
        app.processEvents()
        self.assertFalse(page.view_tabs.is_detached("Spectrum"))
        self.assertTrue(page._wide)
        self.assertIs(page.multi_spectrum.window(), page.window())

    def test_dropping_beside_the_other_view_docks_it_there(self):
        page = self.page
        page.resize(1200, 700)                       # too small for the automatic side-by-side
        app.processEvents()
        self.assertFalse(page._wide)
        window = page.view_tabs.pop_out("Spectrum")
        app.processEvents()
        chip = window.tab
        area = page.view_tabs.drop_area()
        left_edge = area.mapToGlobal(QtCore.QPoint(10, area.height() // 2))
        self.assertEqual(page.view_tabs.drop_zone(left_edge), "left")
        mouse(chip, "press", chip.rect().center())
        mouse(chip, "move", chip.rect().center() + QtCore.QPoint(30, 0))
        mouse(chip, "move", chip.mapFromGlobal(left_edge))
        own_size = page.view_tabs._drag.free_geometry.size()
        # No coloured box: the view itself takes the half it would dock into.
        target = page.view_tabs.zone_rect("left")
        self.assertEqual(window.frameGeometry().topLeft(), target.topLeft())
        middle = area.mapToGlobal(QtCore.QPoint(area.width() // 2, area.height() // 2))
        mouse(chip, "move", chip.mapFromGlobal(middle))              # away from the edge ...
        self.assertEqual(window.size(), own_size)                   # ... it is its own size again
        mouse(chip, "move", chip.mapFromGlobal(left_edge))
        self.assertIsNotNone(page.view_tabs._drag.free_geometry)
        mouse(chip, "release", chip.mapFromGlobal(left_edge), held=False)
        app.processEvents()
        self.assertFalse(page.view_tabs.is_detached("Spectrum"))
        self.assertTrue(page._wide)                                  # side by side, as dropped
        order = [page.wide_container.widget(i) for i in range(page.wide_container.count())]
        self.assertEqual(order.index(page.wide_slots["Spectrum"]), 0)
        self.assertEqual(page.preferences.value("monitor/views/layout"), "split")
        page.resize(1000, 600)                       # the choice survives a resize
        app.processEvents()
        self.assertTrue(page._wide)
        page.preferences.setValue("monitor/views/layout", "auto")

    def test_the_last_view_can_be_pulled_out_and_the_empty_area_says_so(self):
        page = self.page
        page.view_tabs.pop_out("Spectrum")
        window = page.view_tabs.pop_out("Signals")
        app.processEvents()
        self.assertIsNotNone(window)
        self.assertEqual(labels(page.view_tabs), [])
        self.assertTrue(page.view_tabs.isVisible())
        window.close()
        app.processEvents()
        self.assertEqual(labels(page.view_tabs), ["Signals"])

    def test_wide_layout_leaves_popped_out_view_alone(self):
        page = self.page
        page.resize(1920, 1080)
        app.processEvents()
        self.assertTrue(page._wide)
        window = page.view_tabs.pop_out("Spectrum")
        app.processEvents()
        self.assertFalse(page._wide)
        self.assertEqual(labels(page.view_tabs), ["Signals"])
        self.assertIs(page.multi_spectrum.window(), window)
        window.close()
        app.processEvents()
        self.assertTrue(page._wide)
        page.resize(1366, 768)
        app.processEvents()
        self.assertEqual(labels(page.view_tabs), ["Signals", "Spectrum"])


if __name__ == "__main__":
    unittest.main()
