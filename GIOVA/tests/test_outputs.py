"""Graphs that do not zoom by accident, reports in four formats, FFT and spectrum scales, units,
the categories of the block list and the busy trace (24 October 2026).

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest
import zipfile
from xml.etree import ElementTree

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtGui, QtWidgets  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_pipeline import save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from test_workbench import wait_for_run  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.pipeline import PipelineError, templates  # noqa: E402
from eeg.pipeline import api as ee  # noqa: E402
from eeg.pipeline.data import Epochs  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def trials(fs=250.0, n=4):
    """Trials of a 10 Hz sine of amplitude 10 µV on C3, nothing on C4."""
    t = np.arange(int(fs * 2)) / fs
    data = np.zeros((n, 2, t.size))
    data[:, 0] = 10 * np.sin(2 * np.pi * 10 * t)
    return Epochs(data, fs, ["C3", "C4"], ["a", "b"] * (n // 2), 0.0, [str(i + 1) for i in range(n)])


class GraphTests(unittest.TestCase):
    def test_graphs_do_not_zoom_until_asked(self):
        from eeg.ui.workbench.previews import preview
        widget = preview(trials())
        plot = widget.findChildren(pg.PlotWidget)[0].getPlotItem()
        self.assertEqual(plot.getViewBox().state["mouseEnabled"], [False, False])
        zoom = next(w for w in widget.controls if hasattr(w, "zoom"))
        zoom.zoom.setChecked(True)
        self.assertEqual(plot.getViewBox().state["mouseEnabled"], [True, True])

    def test_a_preview_saves_a_sharp_png_without_its_buttons(self):
        from eeg.ui.workbench.previews import PICTURE_SCALE, preview
        widget = preview(trials())
        widget.resize(500, 320)
        widget.show()
        app.processEvents()
        with scratch() as folder:
            path = widget.save_image(folder / "f.png")
            image = QtGui.QImage(str(path))
        self.assertEqual(image.width(), 500 * PICTURE_SCALE)
        self.assertTrue(all(w.isVisible() for w in widget.controls if w.isVisibleTo(widget)))
        widget.close()

    def test_spectrum_scales(self):
        from eeg.ui.workbench.previews import SpectrumPreview
        spectrum = ee.psd(trials())
        widget = SpectrumPreview(spectrum)
        for name, log in (("Linear", False), ("dB", False), ("Log", True)):
            widget.scale_box.setCurrentText(name)
            self.assertEqual(widget.plot.getPlotItem().ctrl.logYCheck.isChecked(), log, name)
        widget.scale_box.setCurrentText("dB")
        self.assertIn("dB", widget.plot.getPlotItem().getAxis("left").labelText)


class RecordingPreviewTests(unittest.TestCase):
    def test_one_channel_on_a_microvolt_axis_or_all_stacked(self):
        from eeg.ui.workbench.previews import STACKED, preview
        fs = 250.0
        t = np.arange(int(fs * 60)) / fs
        data = np.vstack([1000 + 40 * np.sin(2 * np.pi * 10 * t), 5 * np.sin(2 * np.pi * 3 * t)])
        widget = preview(Recording(data, fs, ["C3", "Pz"]))
        self.assertEqual(widget.channel.currentText(), "C3")             # opens on one channel, readable
        axis = widget.plot.getPlotItem().getAxis("left")
        self.assertIn("µV", axis.labelText)
        curve = widget.plot.getPlotItem().listDataItems()[0]
        x, y = curve.getData()
        self.assertAlmostEqual(y.max(), 40, delta=0.5)                 # its own amplitude, offset taken off
        self.assertAlmostEqual(x[-1] - x[0], 10, delta=0.01)
        self.assertIn("79.8", widget.caption.text())                     # peak to peak
        widget.span.setCurrentText("2 s")
        widget.start.setValue(30.0)
        x, _ = widget.plot.getPlotItem().listDataItems()[0].getData()
        self.assertAlmostEqual(x[0], 30.0, delta=0.01)
        self.assertAlmostEqual(x[-1] - x[0], 2, delta=0.01)
        widget.span.lineEdit().setText("7")                            # any length, typed
        widget.span.lineEdit().editingFinished.emit()
        x, _ = widget.plot.getPlotItem().listDataItems()[0].getData()
        self.assertAlmostEqual(x[-1] - x[0], 7, delta=0.01)
        self.assertEqual(widget.span.currentText(), "7 s")
        widget.channel.setCurrentText(STACKED)
        self.assertEqual(len(widget.plot.getPlotItem().listDataItems()), 2)
        self.assertIn("apart", widget.caption.text())
        widget.for_report()
        captions = [label.text() for label in widget.findChildren(QtWidgets.QLabel)]
        self.assertTrue(any("from 30.0 s" in c or "from 30,0 s" in c for c in captions), captions)   # locale


class TrialPreviewTests(unittest.TestCase):
    def test_trials_with_an_offset_are_drawn_around_their_mean_and_the_means_get_their_own_range(self):
        from eeg.ui.workbench.previews import EACH, MEANS, preview
        fs = 250.0
        t = np.arange(int(7 * fs)) / fs
        data = 20000 + 10 * np.sin(2 * np.pi * 10 * t)[None, None, :] + np.zeros((30, 1, 1))   # a DC level of 20 mV
        widget = preview(Epochs(data, fs, ["C3"], ["right"] * 30, 0.0, [str(k + 1) for k in range(30)]))
        widget.show_box.setCurrentText(EACH)
        self.assertLess(float(widget.caption.text().split(", ")[1].split(" ")[0]), 30)   # µV apart: the sine's
        _, y = widget.plot.getPlotItem().listDataItems()[0].getData()
        self.assertAlmostEqual(np.ptp(y), 20, delta=0.5)
        widget.show_box.setCurrentText(MEANS)
        low, high = widget.plot.getPlotItem().viewRange()[1]
        self.assertLess(low, 20000)
        self.assertGreater(high, 20000)                         # the mean is on screen, not off it

    def test_all_trials_as_an_image_and_pages_of_any_size(self):
        from eeg.ui.workbench.previews import EACH, IMAGE, preview
        fs, n = 100.0, 100
        t = np.arange(int(7 * fs)) / fs
        data = np.sin(2 * np.pi * 10 * t)[None, None, :] * np.ones((n, 1, 1))
        data[41, 0] *= 50                                           # trial 42: an artefact
        ids = [str(k + 1) for k in range(n)]
        widget = preview(Epochs(data, fs, ["C3"], ["left", "right"] * 50, 0.0, ids))
        widget.show_box.setCurrentText(IMAGE)
        self.assertTrue(widget.image_plot.isVisibleTo(widget) and not widget.plot.isVisibleTo(widget))
        image = widget.image.image
        self.assertEqual(image.shape, (t.size, n))                  # every trial, a row each
        row = n - 1 - 41                                            # the first trial is drawn on top
        self.assertGreater(np.abs(image[:, row]).max(), 10 * np.abs(np.delete(image, row, axis=1)).max())
        self.assertEqual(len(widget.mean_plot.getPlotItem().listDataItems()), 2)    # the condition means under it
        self.assertIn("All 100 trials", widget.caption.text())
        widget.show_box.setCurrentText(EACH)
        widget.page_box.lineEdit().setText("40")                  # any number per page
        widget.page_box.lineEdit().editingFinished.emit()
        self.assertEqual(len(widget.plot.getPlotItem().listDataItems()), 40)
        widget.forward.click()
        self.assertEqual(int(widget.start.value()), 41)
        self.assertIn("Trials 41–80 of 100", widget.caption.text())
        widget.forward.click()
        self.assertFalse(widget.forward.isEnabled())               # 81-100: the last page
        widget.back.click()
        self.assertEqual(int(widget.start.value()), 41)
        picked = []
        picker = type(widget)(widget.value, kept=set(ids) - {"42"}, toggled=picked.append)
        picker.show_box.setCurrentText(IMAGE)
        self.assertIn("Kept 99 of 100", picker.caption.text())
        self.assertIn("42 ✕", [label for _, label in picker.image_plot.getAxis("left")._tickLevels[0]])
        picker.resize(600, 700)
        picker.show()
        QtWidgets.QApplication.processEvents()
        view = picker.image_plot.getPlotItem().getViewBox()
        from pyqtgraph.Qt import QtCore

        class Click:
            def scenePos(self):
                return view.mapViewToScene(QtCore.QPointF(3.0, -41.2))     # on trial 42's row
        picker._image_clicked(Click())
        picker.close()
        self.assertEqual(picked, ["42"])


class FftTests(unittest.TestCase):
    def test_a_sine_reads_its_amplitude(self):
        spectrum = ee.fft(trials())
        self.assertEqual(spectrum.kind, "amplitude")
        self.assertEqual(spectrum.unit, "µV")
        peak = np.argmax(spectrum.power[0, 0])
        self.assertAlmostEqual(spectrum.freqs[peak], 10.0)
        self.assertAlmostEqual(spectrum.power[0, 0, peak], 10.0, places=6)
        self.assertLess(spectrum.power[0, 1].max(), 1e-9)

    def test_band_power_needs_a_density(self):
        with self.assertRaisesRegex(PipelineError, "Welch PSD"):
            ee.band_power(ee.fft(trials()))


class UnitsTests(unittest.TestCase):
    def test_volts_and_gain_become_microvolts(self):
        data = np.full((2, 100), 2e-5)                    # 20 µV stored in volts
        raw = Recording(data, 100.0, ["C3", "C4"])
        self.assertTrue(np.allclose(ee.units(raw, stored="V").data, 20.0))
        amplified = Recording(np.full((2, 100), 20000.0), 100.0, ["C3", "C4"])
        self.assertTrue(np.allclose(ee.units(amplified, gain=1000.0).data, 20.0))
        self.assertIs(ee.units(raw), raw)                  # nothing to change, nothing copied
        with self.assertRaisesRegex(PipelineError, "gain"):
            ee.units(raw, gain=0.0)

    def test_the_recording_says_when_its_units_look_wrong(self):
        from eeg.ui.workbench.previews import unit_advice
        rng = np.random.default_rng(1)
        eeg = rng.normal(0, 10, (4, 5000))
        self.assertEqual(unit_advice(eeg), "")
        self.assertEqual(unit_advice(eeg + 40000 + np.linspace(0, 3000, 5000)), "")   # a raw DC offset and drift
        self.assertIn("in V", unit_advice(eeg * 1e-6))
        self.assertIn("in mV", unit_advice(eeg * 1e-3))
        self.assertIn("gain", unit_advice(eeg * 1000))


class ReportFormatTests(unittest.TestCase):
    def test_pdf_word_html_and_figures(self):
        from eeg.ui.workbench import WorkbenchPage
        from eeg.ui.workbench.report import FIGURE_WIDTH, pipeline_report
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            page = WorkbenchPage()
            page.set_graph(templates.csp_lda(str(path)))
            page.run()
            wait_for_run(page)
            report = pipeline_report(page.graph, page.outputs(), page.canvas)
            figures = [image for image in report.images if image.devicePixelRatio() == 2]
            self.assertTrue(figures)
            self.assertTrue(all(image.width() == 2 * FIGURE_WIDTH for image in figures))
            pdf = Path(page.save_report(str(directory / "r.pdf"))).read_bytes()
            docx = page.save_report(str(directory / "r.docx"))
            with zipfile.ZipFile(docx) as archive:
                names = archive.namelist()
                document = ElementTree.fromstring(archive.read("word/document.xml"))
                ElementTree.fromstring(archive.read("word/styles.xml"))
            page.save_report(str(directory / "fig.png"))
            pngs = sorted(p.name for p in directory.glob("fig_*.png"))
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(len([n for n in names if n.startswith("word/media/")]), len(report.images))
        text = "".join(document.itertext())
        self.assertIn("Methods", text)
        self.assertIn("Common spatial patterns", text)
        self.assertEqual(pngs, ["fig_{:02d}.png".format(k) for k in range(1, len(report.images) + 1)])

    def test_a_typed_name_gets_the_chosen_format(self):
        from unittest.mock import patch
        from eeg.ui.workbench.report import ask_path
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("C:/x/report", "Word (*.docx)")):
            self.assertEqual(ask_path(None, "C:/x/a.pdf").name, "report.docx")


class BusyTests(unittest.TestCase):
    def test_background_work_shows_the_trace_until_it_ends(self):
        from pyqtgraph.Qt import QtCore
        from eeg.ui.jobs import Jobs, activity, busy
        from eeg.ui.widgets import BusyTrace
        trace = BusyTrace()
        activity().changed.connect(trace.set_busy)
        jobs = Jobs(busy=True)
        done = []
        jobs.finished.connect(lambda result, error: done.append(result))
        jobs.start(lambda: sum(range(1000)))
        self.assertTrue(activity().busy)
        self.assertTrue(trace.isVisibleTo(trace.parentWidget()) if trace.parentWidget() else not trace.isHidden())
        deadline = QtCore.QDeadlineTimer(5000)
        while activity().busy and not deadline.hasExpired():
            app.processEvents()
        self.assertFalse(activity().busy)
        self.assertTrue(trace.isHidden())
        self.assertEqual(done, [499500])
        with busy():
            self.assertTrue(activity().busy)
        self.assertFalse(activity().busy)
        self.assertEqual(len(trace.points()), trace.width())


if __name__ == "__main__":
    unittest.main()
