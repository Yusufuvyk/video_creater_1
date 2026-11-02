"""Time-frequency methods: STFT, and complex Morlet wavelets whose width is given as cycles per Hz or as one
wavelet cmorB-C. Morlet and CWT were offered as two methods; they are one transform, and pipelines saved with
those names open as the wavelet method and give the same numbers. And what the head map says about itself: which
transform its window means come from, how many electrodes it was drawn from, and whose positions they are.

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

from eeg.core import epochs as ep  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.core.wavelets import cmor_cycles, cwt_amplitude  # noqa: E402
from eeg.pipeline import Graph, block  # noqa: E402
from eeg.pipeline import api as ee  # noqa: E402
from eeg.pipeline import compute  # noqa: E402


def recording(fs=128.0, trials=6, seconds=4.0):
    """Trials of 10 Hz that drops after 1 s, with events, as a continuous recording."""
    rng = np.random.default_rng(3)
    n = int(trials * seconds * fs)
    t = np.arange(n) / fs
    within = t % seconds
    data = np.vstack([np.where(within < 1.0, 10.0, 3.0) * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 1, n)
                      for _ in range(2)])
    events = ep.EventTable(np.arange(trials) * seconds + 0.5, None, ["A"] * trials)
    return Recording(data, fs, ["C3", "C4"], events=events)


COMMON = dict(channels=["C3"], tmin=0.0, tmax=3.0, baseline=[0.0, 0.5], active=[1.5, 2.5], fmin=4.0, fmax=20.0)


class NamesTests(unittest.TestCase):
    def test_older_pipelines_open_as_the_wavelet_method(self):
        old = {"version": 1, "name": "old", "wires": [],
               "nodes": [{"id": "a", "type": "erd", "params": {"method": "morlet", "n_cycles": 0.4}},
                         {"id": "b", "type": "erd", "params": {"method": "cwt", "wavelet": "cmor2.0-1.5"}}]}
        graph = Graph.from_dict(old)
        self.assertEqual((graph.params("a")["method"], graph.params("a")["width"], graph.params("a")["n_cycles"]),
                         (ep.WAVELET, ep.PER_HZ, 0.4))
        self.assertEqual((graph.params("b")["method"], graph.params("b")["width"], graph.params("b")["wavelet"]),
                         (ep.WAVELET, ep.FIXED, "cmor2.0-1.5"))
        saved = graph.to_dict()
        self.assertEqual({n["params"]["method"] for n in saved["nodes"]}, {ep.WAVELET})   # saved as now named

    def test_the_two_methods_are_offered_and_the_wavelet_is_free(self):
        kind = block("erd")
        params = {p.name: p for p in kind.params}
        self.assertEqual(params["method"].choices, (ep.STFT, ep.WAVELET))
        self.assertEqual(params["width"].choices, ep.WAVELET_WIDTHS)
        self.assertIsNone(params["wavelet"].choices)                      # any cmorB-C can be typed
        sentence = kind.describe({**kind.defaults(), "method": ep.WAVELET, "width": ep.FIXED,
                                  "wavelet": "cmor1.5-1.0"})
        self.assertIn("cmor1.5-1.0", sentence)
        self.assertIn("5.44 cycles", sentence)


class SameNumbersTests(unittest.TestCase):
    def test_old_and_new_names_give_the_same_maps(self):
        raw = recording()
        for old, width in ((ep.MORLET, ep.PER_HZ), (ep.CWT, ep.FIXED)):
            before = compute.time_frequency(raw, method=old, **COMMON)
            now = compute.time_frequency(raw, method=ep.WAVELET, width=width, **COMMON)
            self.assertEqual(now.settings.method, old)
            np.testing.assert_array_equal(before.tf[0].freqs, now.tf[0].freqs)
            np.testing.assert_array_equal(before.tf[0].power["A"], now.tf[0].power["A"])
        # the block run on its own, with the older name, reads it the same way
        self.assertEqual(ee.erd(raw, method="cwt", **COMMON).settings.method, ep.CWT)

    def test_a_cmor_is_a_morlet_of_so_many_cycles(self):
        """What Help says: cmorB-C is a Morlet wavelet of 2 pi C sqrt(B / 2) cycles, the same numbers."""
        from mne.time_frequency import morlet, tfr_array_morlet
        self.assertAlmostEqual(cmor_cycles("cmor1.5-1.0"), 5.441, places=3)
        fs, rng = 250.0, np.random.default_rng(0)
        t = np.arange(int(8 * fs)) / fs
        x = 10 * np.sin(2 * np.pi * 10 * t) * (t > 3) + 5 * np.sin(2 * np.pi * 22 * t) + rng.normal(0, 3, t.size)
        freqs = np.array([6.0, 10.0, 22.0, 30.0])
        for wavelet in ("cmor1.5-1.0", "cmor2.0-1.5"):
            cycles = cmor_cycles(wavelet)
            cwt = cwt_amplitude(x, fs, freqs, wavelet) ** 2 / 2
            power = tfr_array_morlet(x[None, None], fs, freqs, n_cycles=cycles, output="power", verbose="error")
            gain = np.array([np.abs(w).sum() for w in morlet(fs, freqs, n_cycles=cycles)])
            mne_power = power[0, 0] * (2.0 / gain ** 2)[:, None]
            inner = slice(int(1.5 * fs), int(6.5 * fs))                     # away from the ends
            np.testing.assert_allclose(cwt[:, inner], mne_power[:, inner], rtol=1e-3, atol=1e-3)


class HeadMapTextTests(unittest.TestCase):
    """The head map and the statistics are window means of a separate STFT, from every electrode with Scalp map
    on, at template positions: said where they are shown and in the methods."""

    def test_the_methods_name_the_transform_of_the_window_means(self):
        kind = block("erd")
        sentence = kind.describe({**kind.defaults(), "method": ep.WAVELET, "window": 0.5, "overlap": 0.9,
                                  "topography": True})
        self.assertIn("separate STFT of every electrode (0.5 s Hann window, 50% overlap, not smoothed)", sentence)
        same = kind.describe({**kind.defaults(), "method": ep.STFT, "overlap": 0.5, "smooth": 0.0})
        self.assertIn("window means of the same STFT, on each analysed channel", same)

    def test_the_head_map_counts_the_electrodes_it_is_drawn_from(self):
        from pyqtgraph.Qt import QtWidgets
        from eeg.core.montages import TEN_TWENTY_19
        from eeg.ui.workbench.previews import HEAD_VIEW, STATS_VIEW, TFPreview
        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        rng = np.random.default_rng(5)
        fs, trials, seconds = 128.0, 6, 4.0
        data = rng.normal(0, 5, (len(TEN_TWENTY_19), int(trials * seconds * fs)))
        events = ep.EventTable(np.arange(trials) * seconds + 0.5, None, ["A"] * trials)
        raw = Recording(data, fs, list(TEN_TWENTY_19), events=events)
        result = compute.time_frequency(raw, **{**COMMON, "channels": ["C3", "Cz", "C4"]}, method=ep.WAVELET,
                                        topography=True)
        preview = TFPreview(result)
        self.addCleanup(preview.close)
        preview.view_box.setCurrentText(HEAD_VIEW)
        caption = preview.caption.text()
        self.assertNotIn("Only 3 channels", caption)                 # the map has all 19, not the 3 in detail
        self.assertIn("every electrode computed (19)", caption)
        self.assertIn("separate STFT of every electrode", caption)
        self.assertIn(ep.TEMPLATE_POSITIONS, caption)
        self.assertEqual(preview.head.toolTip(), ep.TEMPLATE_POSITIONS)
        preview.view_box.setCurrentText(STATS_VIEW)
        self.assertIn("separate STFT", preview.caption.text())


class EmptyCellTests(unittest.TestCase):
    """As tables in papers (APA): a cell whose value does not apply is blank; one that could not be computed has a
    dash, and the caption says what it means."""

    def test_the_statistics_table_tells_not_applicable_from_not_computed(self):
        from pyqtgraph.Qt import QtWidgets
        from eeg.ui.workbench.previews import COMPARISON_KEYS, NO_COMPARISON, STAT_COLUMNS, STATS_VIEW, TFPreview
        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        rng = np.random.default_rng(2)
        fs, seconds = 128.0, 4.0
        labels = ["A", "A", "A", "B"]                       # one B trial: its SD and p cannot be computed
        data = rng.normal(0, 5, (2, int(len(labels) * seconds * fs)))
        events = ep.EventTable(np.arange(len(labels)) * seconds + 0.5, None, labels)
        result = compute.time_frequency(Recording(data, fs, ["C3", "C4"], events=events), **COMMON)
        preview = TFPreview(result)
        self.addCleanup(preview.close)
        preview.view_box.setCurrentText(STATS_VIEW)
        columns = {key: j for j, (_, key) in enumerate(STAT_COLUMNS)}
        preview.compare.setCurrentText(NO_COMPARISON)          # A alone: nothing to compare it with
        self.assertEqual(preview._pair(), ["A"])
        for key in COMPARISON_KEYS:
            self.assertEqual(preview.table.item(0, columns[key]).text(), "", key)
        preview.compare.setCurrentText("B")
        self.assertEqual(preview._pair(), ["A", "B"])
        self.assertEqual(preview.table.item(0, columns["sd_b"]).text(), "—")         # one trial: no SD
        self.assertEqual(preview.table.item(0, columns["p_welch"]).text(), "—")
        self.assertIn("—: could not be computed", preview.caption.text())


if __name__ == "__main__":
    unittest.main()
