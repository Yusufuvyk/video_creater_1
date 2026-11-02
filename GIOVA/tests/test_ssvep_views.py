"""SSVEP helpers: neighbour-bin SNR and the condition contrast view.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from test_epochs import app  # noqa: E402
from eeg.core import epochs as ep  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.core.spectral import (condition_contrast, neighbour_snr, strongest_frequencies,  # noqa: E402
                               welch_psd)


def flicker_recording(fs=250, blocks=10, seconds=8, seed=0):
    """O1/O2: 'left' blocks carry 10 Hz (and its 20 Hz harmonic), 'right' blocks 15 Hz."""
    rng = np.random.default_rng(seed)
    n = blocks * seconds * fs
    t = np.arange(n) / fs
    data = rng.normal(0, 4, (2, n))
    onsets, labels = [], []
    for k in range(blocks):
        label = "left" if k % 2 == 0 else "right"
        s, e = k * seconds * fs, (k + 1) * seconds * fs
        if label == "left":
            data[:, s:e] += 2 * np.sin(2 * np.pi * 10 * t[s:e]) + 1.5 * np.sin(2 * np.pi * 20 * t[s:e])
        else:
            data[:, s:e] += 2 * np.sin(2 * np.pi * 15 * t[s:e])
        onsets.append(k * seconds)
        labels.append(label)
    return Recording(data, fs, ["O1", "O2"], events=ep.EventTable(onsets, [seconds] * blocks, labels))


class CoreTests(unittest.TestCase):
    def test_neighbour_snr_is_one_on_noise_and_high_on_a_line(self):
        rng = np.random.default_rng(1)
        fs = 250
        x = rng.normal(0, 5, (2, 30 * fs))
        x[0] += 2 * np.sin(2 * np.pi * 15 * np.arange(30 * fs) / fs)
        f, p = welch_psd(x, fs)
        snr = neighbour_snr(f, p)
        self.assertEqual(snr.shape, p.shape)
        k = np.argmin(abs(f - 15))
        # 2 µV line: 2 µV² over the 0.75 Hz Hann bandwidth on 25 / 125 µV²/Hz of noise -> about 14
        self.assertGreater(snr[0, k], 10)
        self.assertAlmostEqual(float(np.median(snr[1, (f > 5) & (f < 40)])), 1, delta=.25)
        self.assertEqual(neighbour_snr(f, p[0]).shape, p[0].shape)       # 1-D in, 1-D out

    def test_condition_contrast_is_cohens_d_of_log_power(self):
        a = np.exp(np.array([[1.0, 5.0], [3.0, 5.0]]))
        b = np.exp(np.array([[0.0, 5.0], [2.0, 5.0]]))
        d = condition_contrast(a, b)
        # means 2 vs 1, SDs sqrt(2) each -> d = 1 / sqrt(2); equal columns -> NaN (0 / 0)
        self.assertAlmostEqual(d[0], 1 / np.sqrt(2))
        self.assertTrue(np.isnan(d[1]))
        with self.assertRaises(ValueError):
            condition_contrast(a[:1], b)
        # With frequencies, the SD comes from the neighbourhood: one bin whose
        # few trials happen to scatter more cannot hide its own peak.
        f = np.arange(10, 20, .5)
        rng = np.random.default_rng(0)
        base = np.exp(rng.normal(0, .2, (5, f.size)))
        line = base.copy()
        line[:, f == 15] *= 20
        line[:, (f == 14.5) | (f == 15.5)] *= 5                  # Hann leakage beside the line
        noisy = base.copy()
        noisy[:, f == 15] *= np.exp([[-1.5], [1.5], [-1.5], [1.5], [0]])
        self.assertNotEqual(f[np.nanargmax(condition_contrast(line, noisy))], 15)   # per-bin SD misses it
        self.assertEqual(f[np.nanargmax(condition_contrast(line, noisy, f))], 15)

    def test_strongest_frequencies_keeps_peaks_apart(self):
        f = np.arange(0, 30, .5)
        v = np.zeros_like(f)
        v[f == 10] = 5
        v[f == 10.5] = 4.9
        v[f == 20] = 3
        v[f == 25] = 1
        self.assertEqual(strongest_frequencies(f, v, 1, 30, count=3), [(10.0, 5.0), (20.0, 3.0), (25.0, 1.0)])
        self.assertEqual(strongest_frequencies(f, v, 1, 15, count=2)[0], (10.0, 5.0))


class SpectrumViewTests(unittest.TestCase):
    """SNR and the condition contrast on the Workbench's spectrum, as the Analysis page showed them."""

    def setUp(self):
        app()
        from eeg.pipeline import compute
        from eeg.ui.workbench.previews import SpectrumPreview
        epochs = compute.cut_epochs(flicker_recording(), tmin=0.0, tmax=8.0)
        self.preview = SpectrumPreview(compute.welch_spectrum(epochs, window=2.0))

    def test_contrast_names_each_lights_frequencies(self):
        from eeg.ui.workbench.previews import CONTRAST, SPECTRUM
        preview = self.preview
        preview.show_box.setCurrentText(CONTRAST)
        self.assertFalse(preview.scale_box.isEnabled())
        preview.a_box.setCurrentText("left")
        preview.b_box.setCurrentText("right")
        text = preview.caption.text()
        more_left, more_right = text.split(" · ")[:2]
        self.assertIn("10 Hz", more_left)
        self.assertIn("20 Hz", more_left)
        self.assertIn("15 Hz", more_right)
        self.assertIn("trials 5 / 5", text)
        preview.b_box.setCurrentText("left")                          # same condition twice
        self.assertIn("two different conditions", preview.caption.text())
        preview.show_box.setCurrentText(SPECTRUM)                     # back: the contrast's controls hidden
        self.assertTrue(preview.scale_box.isEnabled())
        self.assertTrue(preview.a_box.isHidden())

    def test_snr_scale_puts_the_lines_on_top(self):
        from eeg.ui.workbench.previews import SNR
        self.preview.scale_box.setCurrentText(SNR)
        peaks = {}
        for item in self.preview.plot.listDataItems():
            x, y = item.getData()
            keep = (x >= 5) & (x <= 40)
            peaks[(item.name() or "").split(" ")[0]] = x[keep][np.argmax(y[keep])]
        self.assertEqual(peaks.get("left"), 10)
        self.assertEqual(peaks.get("right"), 15)


if __name__ == "__main__":
    unittest.main()
