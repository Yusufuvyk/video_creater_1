"""The Monitor's spectra are computed from the display window (Controls → Window)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from eeg.core.monitoring import spectra_snapshot
from eeg.core.sources import FileSource, Recording
from eeg.ui.playback import Player

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
FS = 125


def snapshot_after(page, seconds):
    """Play `seconds` of file, then wait for the background spectrum."""
    page._latest_snapshot = None
    for _ in range(int(seconds / .05)):
        page._tick(.05)
    for _ in range(60):
        if getattr(page, "_latest_snapshot", None) is not None:
            break
        page.sample_position += 1                       # a new position lets a job start
        page._update_live_views()
        QtCore.QThreadPool.globalInstance().waitForDone(500)
        app.processEvents()
    return page._latest_snapshot


class SpectralWindowTests(unittest.TestCase):
    def setUp(self):
        from eeg.ui.monitor import MonitorPage
        self.page = page = MonitorPage()
        page.resize(1400, 900); page.show()
        t = np.arange(FS * 40) / FS
        data = np.array([10 * np.sin(2 * np.pi * 10 * t) + i for i in range(4)])
        page.source = FileSource(Recording(data, FS, ["O1", "O2", "PO3", "PO4"]))
        page.fs, page.n_channels = float(FS), 4
        page.panel.set_channels(["O1", "O2", "PO3", "PO4"])
        page.player = Player(page.source, FS)

    def tearDown(self):
        self.page.stop()
        self.page.close()
        QtCore.QThreadPool.globalInstance().waitForDone(3000)

    def start(self, window):
        self.page.panel.window_seconds.setValue(window)
        self.page._begin_streaming(); self.page.timer.stop()
        self.page.transport.set_playing(True)

    def test_fft_resolution_follows_the_window(self):
        for window in (6, 3):
            self.setUp() if window == 3 else None
            self.start(window)
            snapshot = snapshot_after(self.page, window + 2)
            self.assertIsNotNone(snapshot, window)
            spacing = np.diff(snapshot["fft_freqs"][:3])
            np.testing.assert_allclose(spacing, 1.0 / window, rtol=1e-9)
            # An FFT of exactly `window` seconds: n // 2 + 1 bins, as the core
            # function gives for a block of that length.
            n = window * FS
            self.assertEqual(self.page.spectral_buffer.shape[1], n)
            self.assertEqual(snapshot["fft"].shape[1], n // 2 + 1)
            reference = spectra_snapshot(self.page.spectral_buffer, FS)
            np.testing.assert_array_equal(snapshot["fft_freqs"], reference["fft_freqs"])
            np.testing.assert_array_equal(snapshot["psd_freqs"], reference["psd_freqs"])
            self.assertIn("last {} s".format(window), self.page.multi_spectrum.caption.text())
            # The quality table keeps its fixed 4 s of raw data.
            self.assertEqual(self.page.raw_buffer.shape[1], 4 * FS)

    def test_changing_the_window_restarts_the_spectra(self):
        self.start(6)
        self.assertIsNotNone(snapshot_after(self.page, 8))
        self.page.panel.window_seconds.setValue(2)
        self.assertIsNone(self.page._latest_snapshot)          # no mix of two grids
        self.assertLessEqual(self.page.spectral_buffer.shape[1], 2 * FS)
        snapshot = snapshot_after(self.page, 3)
        np.testing.assert_allclose(np.diff(snapshot["fft_freqs"][:3]), 0.5)

    def test_settings_change_the_spectra_while_streaming(self):
        """Settings → Monitor → Spectra: window and Welch segment, applied live."""
        from eeg.ui.preferences import user_settings
        store = user_settings()
        for key in ("monitor/spectrum_seconds", "monitor/welch_seconds", "monitor/welch_overlap"):
            store.remove(key)
        try:
            page = self.page
            self.settings = page.settings_page()                  # builds (and keeps) the controls
            self.assertEqual(page.spectrum_seconds.text(), "Same as display window")
            self.start(10)
            first = snapshot_after(page, 11)
            np.testing.assert_allclose(np.diff(first["psd_freqs"][:3]), 0.5)   # 2 s segments
            # Live: 3 s window, 1 s segments without overlap.
            page.spectrum_seconds.setValue(3)
            page.welch_seconds.setValue(1)
            page.welch_overlap.setValue(0)
            self.assertIsNone(page._latest_snapshot)              # restarted, not mixed
            self.assertEqual(page.spectral_buffer.shape[1], 3 * FS)
            self.assertIn("3 segment(s) of 1 s", page.spectral_summary.text())
            snapshot = snapshot_after(page, 1)
            np.testing.assert_allclose(np.diff(snapshot["fft_freqs"][:3]), 1 / 3)
            np.testing.assert_allclose(np.diff(snapshot["psd_freqs"][:3]), 1.0)
            reference = spectra_snapshot(page.spectral_buffer, FS, 1.0, 0.0)
            np.testing.assert_array_equal(snapshot["psd_freqs"], reference["psd_freqs"])
            self.assertIn("Welch 1 s, 0 % overlap", page.multi_spectrum.caption.text())
            # The quality table keeps its 4 s.
            self.assertEqual(page.raw_buffer.shape[1], 4 * FS)
            # Back to "same as display window".
            page.spectrum_seconds.setValue(0)
            self.assertIn("last 10 s", page.multi_spectrum.caption.text())
        finally:
            for key in ("monitor/spectrum_seconds", "monitor/welch_seconds", "monitor/welch_overlap"):
                store.remove(key)

    def test_defaults_give_the_old_psd(self):
        """Welch 2 s / 50 % stays the default: same numbers as before the setting existed."""
        t = np.arange(FS * 8) / FS
        block = np.vstack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 7 * t)])
        from eeg.core.spectral import welch_psd
        f, p = welch_psd(block, FS)
        snapshot = spectra_snapshot(block, FS)
        np.testing.assert_array_equal(snapshot["psd_freqs"], f)
        np.testing.assert_array_equal(snapshot["psd"], p)

    def test_raw_input_uses_the_same_window(self):
        self.page.multi_spectrum.input.setCurrentText("Raw")
        self.start(5)
        snapshot = snapshot_after(self.page, 7)
        np.testing.assert_allclose(np.diff(snapshot["fft_freqs"][:3]), 0.2)
        self.assertEqual(self.page.raw_spectral_buffer.shape[1], 5 * FS)


if __name__ == "__main__":
    unittest.main()
