"""Offscreen screenshots and accelerated 30-minute stream soak (not hardware FPS)."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
import time
import json
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from eeg.ui.app import MainWindow
from eeg.core.sources import Recording, FileSource
from eeg.ui.playback import Player

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=int, default=1800)
parser.add_argument('--performance-seconds', type=int, default=60)
args = parser.parse_args()
out = ROOT / 'validation-output'
out.mkdir(exist_ok=True)
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QtGui.QFont('Segoe UI', 9))
window = MainWindow(); page = window.monitor
window.show(); app.processEvents()
t = np.arange(125 * 12) / 125
data = np.array([(10 + i) * np.sin(2*np.pi*(8+i/2)*t) for i in range(16)])
data[3] += 60 * np.sin(2*np.pi*50*t)
data[7] = 187500
data[11] = 0
page.source = FileSource(Recording(data, 125, page.panel.channel_names))
page.player = Player(page.source, 125)
page.panel.file_combo.addItem('Synthetic validation · 16 channels', None)
page.panel.file_combo.setCurrentIndex(page.panel.file_combo.count() - 1)
page.panel.set_running(True); page.panel.start_button.setText('Stop')
page.transport.configure(data.shape[1], 125); page.transport.set_playing(True)
page._rail_uv = 187500
page.multi_spectrum.mode.setCurrentText('FFT')
page.multi_spectrum.arrangement.setCurrentText('Grid')
page._tick(5)
page._last_quality = -float('inf'); page._tick(1)
QtCore.QThreadPool.globalInstance().waitForDone(5000); app.processEvents()
sizes = {}
for width, height in [(1366, 768), (1920, 1080)]:
    window.resize(width, height); app.processEvents()
    for mode in ['Signals', 'Grid', 'Overlay']:
        page.view_tabs.setCurrentIndex(0 if mode == 'Signals' else 1)
        if mode != 'Signals':
            page.multi_spectrum.arrangement.setCurrentText(mode)
        app.processEvents()
        name = f'festival-{mode.lower()}-{width}x{height}'
        window.grab().save(str(out / (name + '.png')))
        sizes[name] = [window.width(), window.height(), window.minimumSizeHint().width(), window.minimumSizeHint().height()]
page._show_channel(3); app.processEvents()
page.detail.grab().save(str(out / 'festival-channel-detail.png')); page.detail.hide()
window.resize(1920, 1080); app.processEvents()
page.multi_spectrum.arrangement.setCurrentText('Grid')
page.player = None
origin = time.perf_counter()
clock = [origin]
durations = []; queue_sizes = []; samples = []
try:
    import psutil
    process = psutil.Process()
except ImportError:
    process = None

def resident_bytes():
    if process:
        return process.memory_info().rss
    # Read this process only, without an extra optional dependency on Windows.
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak', 'working', 'quota_peak_paged', 'quota_paged',
                                                    'quota_peak_nonpaged', 'quota_nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if get_info(handle(), ctypes.byref(counters), counters.cb):
            return counters.working
    return None
started = time.perf_counter()
with patch('eeg.ui.monitor.time.perf_counter', new=lambda: clock[0]):
    for frame in range(args.seconds * 20):
        clock[0] = origin + frame * .05
        begin = time.perf_counter_ns()
        # Alternating 6/7 samples keeps the 125 Hz acquisition clock exact.
        n = 6 if frame % 2 == 0 else 7
        times = (page.sample_position + np.arange(n)) / 125
        block = np.array([(10+i)*np.sin(2*np.pi*(8+i/2)*times) for i in range(16)])
        page._ingest(block); page._render()
        queue_sizes.append(len(page.spectral_jobs.active))
        # Full painting for the performance segment; thereafter coalesce GUI
        # paints to 2 Hz while still ingesting/rendering all 20 ticks/second.
        if frame < args.performance_seconds * 20 or frame % 10 == 0:
            app.processEvents()
        if page.spectral_jobs.active:
            QtCore.QThreadPool.globalInstance().waitForDone(5000)
            app.processEvents()
        durations.append((time.perf_counter_ns() - begin) / 1e6)
        if frame % 1200 == 0:
            samples.append(dict(seconds=frame/20, rss_bytes=resident_bytes(),
                                raw=page.raw_buffer.shape[1], filtered=page.spectral_buffer.shape[1],
                                curves=len(page.multi_spectrum._curves)))
            print(f'Soak: {frame/20:.0f}/{args.seconds}s simulated', flush=True)
elapsed = time.perf_counter() - started
measured = durations[min(200, len(durations)//2):min(len(durations), args.performance_seconds * 20)]
report = dict(context='Accelerated synthetic stream, 16 channels, 125 Hz, Qt offscreen; not real hardware FPS. '
                      'All frames painted during performance segment; subsequent soak coalesces paints at 2 Hz.',
              simulated_seconds=args.seconds, performance_seconds=min(args.seconds, args.performance_seconds),
              wall_seconds=elapsed, mean_frame_ms=float(np.mean(measured)),
              p95_frame_ms=float(np.percentile(measured,95)), max_pending=max(queue_sizes), memory_samples=samples,
              window_sizes=sizes)
(out / 'festival-monitor-validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
page.stop(); window.hide()
# Validation must not overwrite the operator's saved preferences.
QtCore.QThreadPool.globalInstance().waitForDone(5000)
