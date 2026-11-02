"""Offscreen screenshots of a quick analysis on the Workbench, every result view (not a unit test).

Run: .venv/Scripts/python eegstudio/tests/validate_views.py
Eyes open / closed on a real recording (GIOVA_ALPHA_RECORDING), as Home's Quick analysis opens it: the Epochs block with its trial
table, the spectrum (mean of trials, mean signal, contrast), the features, and the time-frequency views (map,
band course, head map A / B / B - A, statistics), at 1366 x 768 and 1600 x 950. Set QT_SCALE_FACTOR=1.5 to see
them at 150 %. Writes eegstudio/validation-output/quick-<view>-<size>.png.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: E402

from eeg.core import analysis_profiles as ap  # noqa: E402
from eeg.ui.app import MainWindow  # noqa: E402
from eeg.ui.workbench import previews as pv  # noqa: E402

out = ROOT / "validation-output"
out.mkdir(exist_ok=True)
scale = os.environ.get("QT_SCALE_FACTOR", "1")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
QtGui.QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
app.setFont(QtGui.QFont("Segoe UI", 9))
window = MainWindow()
window.show()
sizes = {}


def settle(seconds=0.0):
    QtCore.QThreadPool.globalInstance().waitForDone(60000)
    end = time.perf_counter() + seconds
    while True:
        app.processEvents()
        if time.perf_counter() >= end:
            break


def capture(name):
    settle(0.2)
    key = "quick-{}-{}x{}{}".format(name, window.width(), window.height(), "" if scale == "1" else "-x" + scale)
    sizes[key] = [window.width(), window.height()]
    assert window.grab().save(str(out / (key + ".png"))), key


eyes = next(p for p in ap.builtin_profiles() if p.name == "Eyes open / closed")
eyes = ap.AnalysisProfile.from_dict({**eyes.to_dict(), "tab": "band"})   # with its time-frequency block
workbench = window.open_mode("Workbench")
started = time.perf_counter()
sheet = workbench.quick_analysis(os.environ["GIOVA_ALPHA_RECORDING"], eyes)
workbench.graph.set("psd1", signal_mean=True)
from eeg.ui.workbench.sheets import QUEUED, RUNNING, STOPPING  # noqa: E402
wanted = {"epochs1", "psd1", "features1", "erd1"}
while not wanted <= set(workbench.outputs()) and time.perf_counter() - started < 600:
    if sheet.status not in (QUEUED, RUNNING, STOPPING):
        workbench.run()                       # the quick analysis' own run ended before the change above
    settle(0.5)
assert wanted <= set(workbench.outputs()), workbench.errors
print("ran in {:.1f} s".format(time.perf_counter() - started))

for width, height in ((1366, 768), (1600, 950)):
    window.resize(width, height)
    settle(0.3)
    inspector = workbench.inspector
    workbench.canvas.select("epochs1")
    capture("epochs-trial-table")
    workbench.canvas.select("psd1")
    spectrum = inspector.previews["spectrum"]
    capture("spectrum-mean-of-trials")
    spectrum.average.setCurrentText(pv.MEAN_SIGNAL)
    capture("spectrum-mean-signal")
    spectrum.show_box.setCurrentText(pv.CONTRAST)
    capture("spectrum-contrast")
    workbench.canvas.select("features1")
    capture("features")
    workbench.canvas.select("erd1")
    tf = inspector.previews["tf"]
    capture("tf-map")
    tf.compare.setCurrentText("close")
    for view in (pv.COURSE_VIEW, pv.HEAD_VIEW, pv.STATS_VIEW):
        tf.view_box.setCurrentText(view)
        capture("tf-" + view.lower().replace(" ", "-"))
    tf.view_box.setCurrentText(pv.MAP_VIEW)

for s in workbench.sheets:
    s.modified = False
window.close()
settle()
(out / "view-sizes.json").write_text(json.dumps(sizes, indent=2))
print("wrote", len(sizes), "pictures to", out)
