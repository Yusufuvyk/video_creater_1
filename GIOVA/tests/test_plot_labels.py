"""Every result view is readable: no two tick labels on any axis overlap, and a spectrum shows mains clear of its
edge.

Filming tutorial 3 found a log axis printing "7 8 9 10" over each other, and the 60 Hz mains peak on the very
border of a spectrum. These tests draw each kind of result GIOVA shows - a recording, trials, FFT and Welch
spectra in every scale, band power, time-frequency maps in every view, two windows compared, scores - from real
pipeline runs at three signal sizes, and look at every axis of every plot in them.

Run: .venv/Scripts/python -m unittest discover -s eegstudio/tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from test_epochs import lateralised  # noqa: E402
from test_settings import release  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.pipeline import api as ee  # noqa: E402
from eeg.pipeline.data import AMPLITUDE, Spectrum  # noqa: E402
from eeg.ui import axes, theme  # noqa: E402,F401  (theme installs the readable axes, as the program does)
from eeg.ui.workbench.previews import SPECTRUM_HZ, SpectrumPreview, TFPreview, preview  # noqa: E402


def axis_labels(widget):
    """[(where, [(rect, text)])] for every visible axis of every plot inside `widget`, as it would be drawn."""
    widget.resize(760, 1400)
    widget.show()
    for _ in range(5):
        app.processEvents()
    found = []
    for view in widget.findChildren(pg.GraphicsView):
        if not view.isVisibleTo(widget):
            continue
        for item in view.scene().items():
            if isinstance(item, pg.AxisItem) and item.isVisible() and item.style.get("showValues", True):
                image = QtGui.QImage(800, 800, QtGui.QImage.Format.Format_ARGB32)
                painter = QtGui.QPainter(image)
                specs = item.generateDrawSpecs(painter)
                painter.end()
                if specs:
                    found.append(("{} {}".format(type(widget).__name__, item.orientation),
                                  [(QtCore.QRectF(rect), text) for rect, _flags, text in specs[2]]))
    return found


def overlaps(widget):
    """Pairs of labels drawn over each other, on any axis in `widget`."""
    bad = []
    for where, labels in axis_labels(widget):
        for i, (a, ta) in enumerate(labels):
            for b, tb in labels[i + 1:]:
                if a.intersects(b):
                    bad.append((where, ta, tb))
    return bad


def choose(widget, wanted):
    """Every combo box of `widget` that offers `wanted`, set to it; True when one did."""
    done = False
    for box in widget.findChildren(QtWidgets.QComboBox):
        items = [box.itemText(i) for i in range(box.count())]
        if wanted in items and box.isEnabled():
            box.setCurrentIndex(items.index(wanted))
            app.processEvents()
            done = True
    return done


def results(scale):
    """Each kind of result, from a lateralised 3-channel recording `scale` times the usual size."""
    base = lateralised(trials=20)
    raw = Recording(base.data * scale, base.fs, base.channel_names, events=base.events)
    filtered = ee.filter(raw, low=1.0, high=40.0, notch=0.0)
    trials = ee.epochs(filtered, tmin=2.0, tmax=5.0)
    whole = ee.epochs(filtered, tmin=0.0, tmax=7.0)
    welch = ee.psd(trials, window=1.0)
    return {"recording": raw, "trials": trials, "fft": ee.fft(trials), "welch": welch,
            "band power": ee.band_power(welch),
            "time-frequency": ee.erd(filtered, channels=[], tmin=0.5, tmax=6.5, baseline=[0.5, 1.5],
                                     active=[2.5, 4.5], fmin=6.0, topography=True),
            "windows": ee.compare_windows(whole, window_a=[2.5, 4.5], window_b=[0.5, 1.5]),
            "scores": ee.cross_validate(trials, ee.lda(ee.csp(n_components=2)), folds=4),
            # spectra falling one to two decades, the span where a log axis has the most labels to place
            # (tutorial 3's Fp1 FFT: about 10 µV at 1 Hz down to 0.4 µV)
            **{"spectrum {:g}-{:g}".format(low, high): sloped(low * scale, high * scale)
               for low, high in ((0.4, 12.0), (0.1, 2.5), (0.05, 30.0), (0.5, 3.0))}}


def sloped(low, high):
    """An FFT amplitude falling smoothly from `high` at 0 Hz to `low` at 128 Hz, in two conditions."""
    freqs = np.arange(0, 129.0)
    shape = np.geomspace(high, low, freqs.size)
    power = np.stack([shape * (1 + 0.05 * np.random.default_rng(i).standard_normal(freqs.size))
                      for i in range(6)])[:, None, :]
    return Spectrum(freqs, power, ["Fp1"], ["left"] * 3 + ["right"] * 3, [str(i) for i in range(6)],
                    kind=AMPLITUDE)


class NoOverlappingLabels(unittest.TestCase):
    def test_the_log_axis_keeps_decades_and_drops_what_would_overlap(self):
        self.assertEqual(axes.label_value("10³"), 1000.0)
        self.assertAlmostEqual(axes.label_value("2×10⁻¹"), 0.2)
        self.assertEqual(axes.label_value("−3"), -3.0)
        self.assertEqual([axes.roundness(v) for v in (1, 10, 0.1, 5, 0.2, 7)], [0, 0, 0, 1, 2, 3])
        rects = [QtCore.QRectF(x, 0, 12, 10) for x in (0, 8, 30)]
        self.assertEqual(axes.apart(rects), [0, 2])

    def setUp(self):
        # the views below change what a preview remembers for the session (the last scale, the last map):
        # put it back, or the tests run after these would see it
        self.remembered = SpectrumPreview.scale, TFPreview.kind
        self.addCleanup(self.restore)

    def restore(self):
        SpectrumPreview.scale, TFPreview.kind = self.remembered

    def test_every_result_view_at_every_size(self):
        views = {"fft": ["Log", "Linear", "dB"], "welch": ["Log", "Linear", "dB", "SNR", "Contrast (d)"],
                 "time-frequency": ["ERD / ERS %", "Power (dB)", "Amplitude (µV)", "Band course", "Head map",
                                    "Statistics", "Map"]}
        for scale in (0.01, 1.0, 100.0):
            for name, value in results(scale).items():
                widget = preview(value)
                if widget is None:
                    continue
                try:
                    for option in [None] + views.get(name, []):
                        if option is not None and not choose(widget, option):
                            continue
                        with self.subTest(scale=scale, result=name, view=option):
                            self.assertEqual(overlaps(widget), [])
                finally:
                    release(widget)


class SpectrumRange(unittest.TestCase):
    def test_mains_at_50_and_60_hz_stand_clear_of_the_edge(self):
        freqs = np.arange(0, 256.5, 0.5)
        amplitude = np.ones_like(freqs)
        amplitude[np.isin(freqs, (50.0, 60.0))] = 20.0
        widget = preview(Spectrum(freqs, amplitude[None, None, :], ["Cz"], ["A"], ["1"], kind=AMPLITUDE))
        try:
            x, y = widget.plot.getPlotItem().listDataItems()[0].getData()
            self.assertGreaterEqual(x.max(), 65.0)            # 60 Hz is drawn with room beyond it
            self.assertEqual(x.max(), SPECTRUM_HZ[1])
            self.assertEqual(sorted(x[np.argsort(y)[-2:]]), [50.0, 60.0])     # the two peaks, wherever drawn
        finally:
            release(widget)

    def test_a_low_rate_spectrum_stops_at_its_nyquist_frequency(self):
        freqs = np.arange(0, 62.75, 0.25)                  # 125.5 Hz sampling: nothing above 62.75 Hz exists
        widget = preview(Spectrum(freqs, np.ones((1, 1, freqs.size)), ["Cz"], ["A"], ["1"], kind=AMPLITUDE))
        try:
            x, _ = widget.plot.getPlotItem().listDataItems()[0].getData()
            self.assertEqual(x.max(), freqs.max())
        finally:
            release(widget)


class BandPowerBars(unittest.TestCase):
    """Band power's bars stand inside the plot: log power (negative) on the lowest value shown, power on 0.
    Found filming tutorial 3: log bars were drawn from 0, off the top of the plot, cut, under the legend."""

    def test_every_bar_is_whole_inside_the_plot(self):
        welch = results(1.0)["welch"]
        for log in (True, False):
            widget = preview(ee.band_power(welch, log=log))
            try:
                plot = widget.plot.getPlotItem()
                (_, _), (low, high) = plot.vb.viewRange()
                bars = [item for item in plot.items if isinstance(item, pg.BarGraphItem)]
                self.assertTrue(bars)
                for item in bars:
                    base = np.broadcast_to(np.asarray(item.opts["y0"], dtype=float), np.shape(item.opts["height"]))
                    tops = base + np.asarray(item.opts["height"], dtype=float)
                    with self.subTest(log=log):
                        self.assertTrue((base >= low - 1e-9).all() and (tops <= high + 1e-9).all())
                        self.assertTrue((tops >= base).all())               # bars grow upwards from their base
                        if not log:
                            self.assertTrue(np.allclose(base, 0.0))          # power: a bar's height is the power
            finally:
                release(widget)


if __name__ == "__main__":
    unittest.main()
