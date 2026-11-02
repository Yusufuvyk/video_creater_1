"""Regression checks for readable live band-power comparisons."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from pyqtgraph.Qt import QtWidgets
from eeg.core.config import BANDS
from eeg.ui.plots import BandPowerView


class BandPowerDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.view = BandPowerView()
        self.addCleanup(self.view.close)

    def powers(self, value):
        return np.full(len(BANDS), value, dtype=float)

    def test_updates_are_throttled_smoothed_and_glided(self):
        self.view.draw(self.powers(10), now=0)
        values = self.powers(100)
        self.view.draw(values, now=.25)                       # too early for a new target
        np.testing.assert_allclose(self.view._smoothed, 10)
        np.testing.assert_allclose(self.view._bars.opts['height'], 10)
        self.view.draw(values, now=.5)
        target = 10 + (1 - np.exp(-.25)) * 90
        np.testing.assert_allclose(self.view._smoothed, target)
        drawn = np.asarray(self.view._bars.opts['height'])    # gliding, not stepping
        self.assertTrue(np.all(drawn > 10) and np.all(drawn < target))
        for i in range(1, 10):                                # frames between two updates
            self.view.draw(values, now=.5 + i * .05)
        np.testing.assert_allclose(self.view._bars.opts["height"], self.view._smoothed, rtol=.1)
        np.testing.assert_array_equal(values, self.powers(100))

    def test_scale_holds_on_drop_and_can_be_refitted(self):
        self.view.draw(self.powers(100), now=0)
        initial = self.view.viewRange()[1]
        for i in range(1, 41):
            self.view.draw(self.powers(1), now=i * .5)
        self.assertEqual(self.view.viewRange()[1], initial)
        self.view.fit_scale()
        self.assertLess(self.view.viewRange()[1][1], initial[1])
        self.assertEqual(self.view.viewRange()[1][0], 0)

    def test_growth_fits_all_bands_and_preserves_ratios(self):
        self.view.draw(self.powers(0), now=0)
        values = np.arange(1, len(BANDS) + 1, dtype=float) * 100
        self.view.draw(values, now=.5)
        shown = self.view._bars.opts['height']
        np.testing.assert_allclose(shown / shown[0], values / values[0])
        self.assertGreater(self.view.viewRange()[1][1], max(shown))

    def test_reset_does_not_mix_previous_channel_or_recording(self):
        self.view.draw(self.powers(10000), now=0)
        self.view.reset()
        self.view.draw(self.powers(2), now=.1)
        np.testing.assert_allclose(self.view._bars.opts['height'], 2)
        self.assertLess(self.view.viewRange()[1][1], 10)

    def test_invalid_frame_does_not_poison_future_updates(self):
        self.view.draw(self.powers(10), now=0)
        self.view.draw(self.powers(np.nan), now=.5)
        self.view.draw(self.powers(20), now=1)
        self.assertTrue(np.all(np.isfinite(self.view._bars.opts['height'])))


if __name__ == '__main__':
    unittest.main()
