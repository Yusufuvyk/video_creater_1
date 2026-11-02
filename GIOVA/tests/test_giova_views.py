"""GIOVA 3.1's views: band power drawn, not only listed; the run trace as agreed; the logo's size.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtGui, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def band_features():
    from eeg.pipeline.data import Features
    rng = np.random.default_rng(0)
    channels, bands = ["C3", "Cz", "C4"], ["theta", "alpha", "beta"]
    labels = ["odor"] * 10 + ["air"] * 10
    values = rng.normal(size=(20, 9))
    values[:10, 3:6] += 2.0                        # alpha higher under odor, on every channel
    names = ["{} {}".format(c, b) for b in bands for c in channels]
    return Features(values, names, labels, list(range(20)), ["log10 µV²"] * 9)


class BandPowerTests(unittest.TestCase):
    def test_band_power_is_drawn_as_bars_per_band_and_condition(self):
        from eeg.ui.workbench.previews import FeaturesPreview, band_layout
        features = band_features()
        self.assertEqual(band_layout(features.names), (["C3", "Cz", "C4"], ["theta", "alpha", "beta"]))
        preview = FeaturesPreview(features)
        bars = [i for i in preview.plot.getPlotItem().items if isinstance(i, pg.BarGraphItem)]
        self.assertEqual(len(bars), 2)                                   # odor and air
        self.assertEqual(len(preview.legend.items), 2)                   # each condition named once
        groups = preview.band_values()
        self.assertGreater(groups["odor"][:, 1].mean(), groups["air"][:, 1].mean() + 1)   # the alpha difference
        preview.band_channel.setCurrentText("C4")                        # one channel
        self.assertEqual(preview.band_values()["odor"].shape, (10, 3))

    def test_other_features_stay_a_table(self):
        from eeg.pipeline.data import Features
        from eeg.ui.workbench.previews import FeaturesPreview, band_layout
        features = Features(np.zeros((4, 2)), ["CSP 1", "CSP 2 log var"], ["a", "a", "b", "b"], [0, 1, 2, 3])
        self.assertIsNone(band_layout(features.names))
        self.assertFalse(hasattr(FeaturesPreview(features), "plot"))



class ScrollTests(unittest.TestCase):
    def test_the_wheel_over_a_graph_scrolls_the_panel_unless_zoom_is_on(self):
        from pyqtgraph.Qt import QtCore, QtGui
        from eeg.ui.widgets import install_wheel_guard
        from eeg.ui.workbench.previews import small_plot
        install_wheel_guard(app)
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(inner)
        column.addWidget(QtWidgets.QLabel("settings"))
        plot = small_plot(240)
        column.addWidget(plot)
        column.addSpacing(2000)                          # a long column: room to scroll
        area.setWidget(inner)
        area.resize(300, 400)
        area.show()
        app.processEvents()
        bar = area.verticalScrollBar()

        def wheel():
            spot = QtCore.QPointF(plot.viewport().rect().center())
            event = QtGui.QWheelEvent(spot, plot.viewport().mapToGlobal(spot), QtCore.QPoint(0, 0),
                                      QtCore.QPoint(0, -120), QtCore.Qt.MouseButton.NoButton,
                                      QtCore.Qt.KeyboardModifier.NoModifier, QtCore.Qt.ScrollPhase.NoScrollPhase,
                                      False)
            QtWidgets.QApplication.sendEvent(plot.viewport(), event)
            app.processEvents()

        wheel()
        self.assertGreater(bar.value(), 0)               # the panel scrolled past the graph
        moved = bar.value()
        plot.setMouseEnabled(x=True, y=True)             # Zoom on: the graph takes the wheel
        wheel()
        self.assertEqual(bar.value(), moved)
        area.close()


class CompareTests(unittest.TestCase):
    def test_compare_reads_as_three_steps_and_one_button_runs_and_stops(self):
        from eeg.ui.workbench.compare import CompareWindow
        from eeg.ui.widgets import BusyTrace
        window = CompareWindow(None)
        try:
            texts = [label.text() for label in window.findChildren(QtWidgets.QLabel)]
            for title in ("Pipelines", "Recordings", "Compare"):
                self.assertTrue(any(title in t for t in texts), title)
            self.assertEqual(window.findChildren(BusyTrace), [])          # no second moving trace
            self.assertTrue(window.stop_button.isHidden())                # Compare itself turns into Stop
            self.assertEqual(window.run_button.text(), "Compare")
            self.assertFalse(window.run())                                  # nothing listed: it says so
            self.assertIn("Add at least one pipeline", window.state.text())
        finally:
            window.close()

    def test_result_tables_show_every_row_without_scrolling(self):
        from eeg.ui.workbench.compare import _table
        big = QtGui.QFont(QtWidgets.QApplication.font())
        big.setPointSizeF(big.pointSizeF() * 1.6)                     # rows taller than any fixed guess
        before = QtWidgets.QApplication.font()
        QtWidgets.QApplication.setFont(big)
        try:
            table = _table(("Pipeline", "Mean"), [("CSP + LDA", "78%"), ("Riemann", "74%"), ("Band power", "70%")])
            table.resize(420, table.height())
            table.show()
            QtWidgets.QApplication.processEvents()
            self.assertGreaterEqual(table.viewport().height(), table.verticalHeader().length())
            self.assertFalse(table.verticalScrollBar().isVisible())
            table.close()
        finally:
            QtWidgets.QApplication.setFont(before)

    def test_compare_takes_the_whole_results_panel(self):
        from eeg.ui.workbench.page import WorkbenchPage
        page = WorkbenchPage()
        try:
            page.resize(1200, 900)
            page.show()
            window = page.open_compare()
            page.vsplit.setSizes([150, 700])
            QtWidgets.QApplication.processEvents()
            layout = page.results.layout()
            self.assertFalse(any(layout.itemAt(i).spacerItem() for i in range(layout.count())))
            self.assertGreater(window.height(), 0.9 * page.results.height() - 20)   # not half of it
        finally:
            page.shutdown()
            page.close()

    def test_comparison_plot_joins_each_recordings_own_points(self):
        from eeg.ui.workbench.report import comparison_plot, mean_interval
        result = made_up_comparison()
        plot = comparison_plot(result)
        items = plot.getPlotItem().items
        lines = [i for i in items if isinstance(i, pg.PlotDataItem) and i.xData is not None and len(i.xData) == 3]
        points = [i for i in items if isinstance(i, pg.ScatterPlotItem)]
        self.assertEqual(len(lines), 5)                                  # one per recording, through all three
        for j, scatter in enumerate(points):                             # every line meets its own points
            x, y = scatter.getData()
            self.assertTrue(np.all(np.abs(x - j) < 0.2))                 # in its column, side by side when tied
            marks = {(round(float(a), 6), round(float(b), 6)) for a, b in zip(x, y)}
            for line in lines:
                self.assertIn((round(float(line.xData[j]), 6), round(float(line.yData[j]), 6)), marks)
        x, y = points[1].getData()                                       # s02 and s04 tie at 72%: not on top
        tied = [float(a) for a, b in zip(x, y) if abs(b - 0.72) < 1e-9]
        self.assertEqual(len(tied), 2)
        self.assertNotAlmostEqual(tied[0], tied[1])
        tips = [p.data() for p in points[0].points()]
        self.assertIn("s01 · CSP + LDA · Accuracy 78.0% (above chance)", tips)
        low, high = plot.getPlotItem().viewRange()[1]
        self.assertLess(low, 0.5 + 1e-6)                                # differences shown at their real size
        self.assertGreater(high, 1.0 - 1e-6)
        mean, half = mean_interval([0.78, 0.76, 0.80, 0.77, 0.60])
        self.assertAlmostEqual(mean, 0.742)
        self.assertAlmostEqual(half, 2.7764451 * np.std([0.78, 0.76, 0.80, 0.77, 0.60], ddof=1) / np.sqrt(5), 5)
        self.assertTrue(np.isnan(mean_interval([0.7])[1]))

    def test_each_recording_is_listed_right_under_the_plot(self):
        from eeg.ui.workbench.compare import CompareWindow
        from eeg.ui.workbench.report import pg as report_pg
        window = CompareWindow(None)
        try:
            result = made_up_comparison()
            window.result = result
            window.show_result(result)
            order = [window.results_column.itemAt(i).widget() for i in range(window.results_column.count())]
            order = [w for w in order if w is not None]
            plot = next(k for k, w in enumerate(order) if isinstance(w, report_pg.PlotWidget))
            self.assertIs(order[plot + 2], window.per_recording)          # the plot, its note, then the table
            table = window.per_recording
            self.assertEqual(table.rowCount(), 5 + 2)                       # the recordings, then Mean and SD
            self.assertEqual(table.item(5, 0).text(), "Mean")
            self.assertEqual(table.item(5, 1).text(), "74.2%")              # (78 + 76 + 80 + 77 + 60) / 5
            self.assertEqual(table.item(6, 1).text(), "{:.1%}".format(np.std([.78, .76, .80, .77, .60], ddof=1)))
            self.assertEqual(table.item(0, 1).text(), "78.0% *")
            kappa = window.measure.findData("kappa")
            self.assertGreaterEqual(kappa, 0)
            window.measure.setCurrentIndex(kappa)                          # another measure: plot and table follow
            self.assertEqual(window.per_recording.item(0, 1).text(), "0.560 *")   # the star stays accuracy's
            self.assertEqual(window.per_recording.item(5, 1).text(), "{:.3f}".format(2 * 0.742 - 1))
            self.assertEqual(window.plot.getPlotItem().getAxis("left").labelText, "Cohen's κ")
        finally:
            window.close()


class MethodsTextTests(unittest.TestCase):
    def test_the_inspector_says_what_a_block_does_without_citations(self):
        from eeg.ui.workbench.inspector import first_sentence, uncited
        text = ("Stratified 5-fold cross-validation (seed 42); every step fitted on the training folds only. "
                "Accuracy judged against the upper limit (alpha 0.05; Müller-Putz et al., 2008); Cohen's kappa "
                "(Cohen, 1960) and macro F1 too. Permutation test (Ojala & Garriga, 2010).")
        plain = uncited(text)
        self.assertNotIn("Müller-Putz", plain)
        self.assertNotIn("1960", plain)
        self.assertIn("(alpha 0.05)", plain)                     # what is not a citation stays
        self.assertIn("(seed 42)", plain)
        self.assertIn("Cohen's kappa and macro F1", plain)
        self.assertTrue(plain.endswith("Permutation test."))
        self.assertEqual(first_sentence(plain), "Stratified 5-fold cross-validation (seed 42); every step fitted "
                                                "on the training folds only.")
        self.assertEqual(first_sentence("Filtered: band-pass 8–30 Hz (order 4; zero-phase)."),
                         "Filtered: band-pass 8–30 Hz (order 4; zero-phase).")

def made_up_comparison():
    """Five recordings, three pipelines (a real Comparison, its Scores stand-ins carrying what it reads); a random
    limit of 56.9% for every recording."""
    from types import SimpleNamespace
    from eeg.pipeline.batch import Comparison
    names = ["CSP + LDA", "Riemann + logistic regression", "Band power + LDA"]
    columns = [(n, "cross_validate1") for n in names]
    subjects = ["Data/Cho2017/s0{}.mat".format(i) for i in range(1, 6)]
    accuracy = np.array([[0.78, 0.83, 0.77], [0.76, 0.72, 0.68], [0.80, 0.82, 0.55], [0.77, 0.72, 0.60],
                         [0.60, 0.53, 0.53]])
    scores = {(s, c): SimpleNamespace(random_limit=0.569, better_than_random=bool(accuracy[i, j] > 0.569),
                                      balanced_accuracy=accuracy[i, j], kappa=2 * accuracy[i, j] - 1,
                                      f1=accuracy[i, j] - 0.01, auc=accuracy[i, j] + 0.05)
              for i, s in enumerate(subjects) for j, c in enumerate(columns)}
    return Comparison(columns, subjects, accuracy, accuracy + 0.05, scores)


if __name__ == "__main__":
    unittest.main()
