"""Offscreen workload comparison; reports frame cost, not real hardware FPS."""
import os
import sys
import time
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui
from eeg.ui.monitor import MonitorPage
from eeg.core.processing import AutoScaler, spectrum, band_powers
from eeg.core.sources import FileSource, Recording
from eeg.ui.playback import Player

app = QtWidgets.QApplication([])
QtGui.QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
app.setFont(QtGui.QFont("Segoe UI", 9))
pg.setConfigOptions(antialias=True, background="w", foreground="k")
results = {}
for mode in ("unthrottled_shared_scale", "optimized_per_channel"):
    page = MonitorPage()
    page.resize(1366, 730)
    page.show()
    page.open_features()
    app.processEvents()
    times = np.arange(125*30)/125
    data = np.array([(2+i*10)*np.sin(2*np.pi*(8+i/3)*times)+np.sin(2*np.pi*3*times) for i in range(16)])
    page.source = FileSource(Recording(data, 125, page.panel.channel_names))
    page.player = Player(page.source, 125)
    page.transport.configure(data.shape[1], 125)
    page.transport.set_playing(True)
    page.panel.channel_button.set_selected(range(16))
    page._tick(10)
    app.processEvents()
    old_scale = AutoScaler()
    if mode.startswith("unthrottled"):
        def old_render():
            if page.filled < 10:
                return
            block = page.buffer[:, -page.filled:]
            channels, names = page.panel.selected_channels, page.panel.channel_names
            page.time_view.draw(block, channels, old_scale.update(block[channels]), page.fs, names)
            index = page.panel.psd_channel.currentIndex()
            freqs, power = spectrum(block[index], page.fs)
            page.spectrum_view.draw(freqs, power, page.panel.log_psd, names[index])
            page.band_view.draw(band_powers(freqs, power))
            if page._new_samples:
                page.features_ready.emit(page.feature_buffer, float(page.fs), names, page.sample_position/page.fs)
                page._new_samples = False
        page._render = old_render
        page.time_view._curve.opts["antialias"] = True
    clock = [time.perf_counter()]
    origin = clock[0]
    durations = []
    with patch("eeg.ui.monitor.time.perf_counter", side_effect=lambda: clock[0]):
        for frame in range(160):
            clock[0] = origin+frame*.05
            started = time.perf_counter_ns()
            page._tick(.05)
            app.processEvents()
            durations.append((time.perf_counter_ns()-started)/1e6)
    results[mode] = dict(mean_frame_ms=float(np.mean(durations[10:])),
                         p95_frame_ms=float(np.percentile(durations[10:], 95)))
    if mode == "optimized_per_channel":
        page.grab().save(str(ROOT/"validation-output"/"monitor-independent-scales.png"))
    page.stop()
    page.close()
    app.processEvents()
results["context"] = ("Offscreen Qt, 16 synthetic channels, 125 Hz, 10 s display, live features open, 160 simulated "
                      "ticks. CPU/render-loop cost, not hardware FPS. Baseline recreates unthrottled shared-scale "
                      "work on the same layout.")
(ROOT/"validation-output"/"monitor-performance.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
