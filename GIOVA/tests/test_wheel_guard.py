"""The mouse wheel scrolls panels and zooms plots; it never changes a value."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from eeg.ui.widgets import WheelGuard, install_wheel_guard

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def roll(widget, steps=3):
    """A wheel event through QApplication, so event filters apply; returns it.

    Qt passes only *spontaneous* (real mouse) wheel events on to the parent, so
    a test cannot watch the panel scroll. It checks the precondition instead:
    the event comes back ignored, which is what makes Qt hand it on.
    """
    centre = QtCore.QPointF(widget.width() / 2, widget.height() / 2)
    event = QtGui.QWheelEvent(centre, widget.mapToGlobal(centre), QtCore.QPoint(0, 0),
                              QtCore.QPoint(0, -120 * steps), QtCore.Qt.MouseButton.NoButton,
                              QtCore.Qt.KeyboardModifier.NoModifier,
                              QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(widget, event)
    return event


class WheelGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_wheel_guard(app)

    def _panel(self):
        area = QtWidgets.QScrollArea()
        inner = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(inner)
        spin = QtWidgets.QDoubleSpinBox(); spin.setRange(0, 100); spin.setValue(50)
        combo = QtWidgets.QComboBox(); combo.addItems(["a", "b", "c"]); combo.setCurrentIndex(1)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); slider.setRange(0, 100)
        slider.setValue(50)
        tabs = QtWidgets.QTabBar(); [tabs.addTab(t) for t in "xyz"]; tabs.setCurrentIndex(1)
        for widget in (spin, combo, slider, tabs):
            column.addWidget(widget)
        column.addSpacing(3000)             # tall enough to scroll
        area.setWidget(inner); area.resize(300, 300); area.show()
        app.processEvents()
        return area, spin, combo, slider, tabs

    def test_values_do_not_change_and_the_wheel_is_passed_on(self):
        area, spin, combo, slider, tabs = self._panel()
        for widget in (spin, combo, slider, tabs):
            self.assertFalse(roll(widget).isAccepted(), type(widget).__name__)
        self.assertFalse(roll(spin.findChild(QtWidgets.QLineEdit)).isAccepted())
        self.assertEqual(spin.value(), 50)
        self.assertEqual(combo.currentIndex(), 1)
        self.assertEqual(slider.value(), 50)
        self.assertEqual(tabs.currentIndex(), 1)
        area.close()

    def test_scroll_bars_still_scroll(self):
        bar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        bar.setRange(0, 100); bar.resize(20, 200); bar.show(); app.processEvents()
        roll(bar)
        self.assertGreater(bar.value(), 0)
        bar.close()

    def test_plots_still_zoom(self):
        import pyqtgraph as pg
        plot = pg.PlotWidget(); plot.setXRange(0, 10, padding=0); plot.resize(400, 300); plot.show()
        app.processEvents()
        before = plot.getViewBox().viewRange()[0]
        roll(plot.viewport())
        app.processEvents()
        self.assertNotEqual(plot.getViewBox().viewRange()[0], before)
        plot.close()

    def test_installed_once(self):
        self.assertIs(install_wheel_guard(app), install_wheel_guard(app))
        self.assertIsInstance(install_wheel_guard(app), WheelGuard)

    def test_every_value_control_in_the_window_ignores_the_wheel_on_its_own(self):
        """Without the app-wide guard too: pages are also built outside main()."""
        from eeg.ui.app import MainWindow
        app.removeEventFilter(app._wheel_guard)
        try:
            window = MainWindow()
            for name in ("Monitor", "Workbench", "Games"):
                window.page(name)            # built when first opened: open them all to check them all
            window.open_settings()
            kinds = (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QSlider)
            loose = []
            for widget in window.findChildren(QtWidgets.QWidget) + \
                    window.settings_dialog.findChildren(QtWidgets.QWidget):
                if isinstance(widget, kinds) and type(widget).wheelEvent in (
                        QtWidgets.QAbstractSpinBox.wheelEvent, QtWidgets.QDoubleSpinBox.wheelEvent,
                        QtWidgets.QSpinBox.wheelEvent, QtWidgets.QComboBox.wheelEvent,
                        QtWidgets.QAbstractSlider.wheelEvent, QtWidgets.QSlider.wheelEvent):
                    if type(widget).__module__.startswith("eeg"):
                        loose.append(type(widget).__name__)
                    elif type(widget).__module__.startswith(("PySide", "PyQt")):
                        loose.append("{} in {}".format(type(widget).__name__,
                                                       type(widget.parent()).__module__))
            eeg_owned = [entry for entry in loose if "pyqtgraph" not in entry]
            self.assertEqual(eeg_owned, [])
            window.close()
        finally:
            app.installEventFilter(app._wheel_guard)



class TooltipSwitchTests(unittest.TestCase):
    def test_switch_swallows_tooltips_only_when_off(self):
        from eeg.ui.widgets import install_tooltip_switch
        switch = install_tooltip_switch(app, True)
        button = QtWidgets.QPushButton("x"); button.setToolTip("help")
        event = QtGui.QHelpEvent(QtCore.QEvent.Type.ToolTip, QtCore.QPoint(1, 1), QtCore.QPoint(1, 1))
        self.assertFalse(switch.eventFilter(button, event))
        self.assertIs(install_tooltip_switch(app, False), switch)      # one filter, re-used
        self.assertTrue(switch.eventFilter(button, event))
        other = QtCore.QEvent(QtCore.QEvent.Type.Enter)
        self.assertFalse(switch.eventFilter(button, other))            # nothing else is touched
        install_tooltip_switch(app, True)
        self.assertEqual(button.toolTip(), "help")                     # the text itself is kept


if __name__ == "__main__":
    unittest.main()
