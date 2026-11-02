"""A time-frequency map must not invent changes at the edges of a trial.

A signal whose power never changes must give an ERD map near zero everywhere, for every method, wherever the
trials sit in the recording: in the middle, or at its very start and end. Wavelets (Morlet, CWT) convolve each
trial as a whole; without samples beyond the trial they see zeros there, and the edges swell. Tutorial 4 found it
on Cho2017 (trials 0.5-6.5 s): the baseline took in the swollen edge and the whole map read as a drop.
"""
import unittest

import numpy as np

from eeg.core import epochs as ep
from eeg.core.sources import Recording

FS = 256.0
SETTINGS = dict(tmin=0.5, tmax=6.5, baseline=[0.5, 1.5], active=[2.5, 4.5], fmin=6.0, fmax=40.0,
                filters=dict(low=1.0, high=40.0, order=4), smooth=0.0, erd_mode="pooled")


def stationary(seconds, starts, seed=0):
    """A 10 Hz rhythm of constant amplitude in white noise, and trials starting at `starts` (s)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    data = (10 * np.sin(2 * np.pi * 10 * t) + 4 * rng.standard_normal(t.size))[None]
    events = ep.EventTable(list(starts), [7.0] * len(starts), ["a"] * len(starts))
    return Recording(data, FS, ["C3"], events=events)


def pooled_erd(recording, method, **changes):
    settings = ep.EpochSettings(method=method, **{**SETTINGS, **changes})
    result = ep.compute_epochs(recording.data, recording.fs, recording.channel_names, [0], recording.events,
                               settings)
    tf = result.tf[0]
    power = tf.power["a"].mean(axis=0)
    base = tf.baseline["a"].mean(axis=0)[:, None]
    return tf.freqs, tf.times, 100.0 * (power - base) / base


class EdgesOfTrials(unittest.TestCase):
    def check(self, recording, label, flat=15.0, **changes):
        for method in (ep.STFT, ep.MORLET, ep.CWT):
            with self.subTest(label=label, method=method):
                freqs, times, erd = pooled_erd(recording, method, **changes)
                self.assertGreaterEqual(times.min(), SETTINGS["tmin"] - 1e-9)
                self.assertLessEqual(times.max(), SETTINGS["tmax"] + 1e-9)
                # the 10 Hz row carries most of the power: it must stay flat to within a few percent
                row = np.argmin(np.abs(freqs - 10.0))
                self.assertLess(np.max(np.abs(erd[row])), flat, "{} {}: 10 Hz row".format(label, method))
                # and no frequency may swell at the edges: there the map may wander no more than in the middle,
                # where only the noise of a few trials moves it
                edge = (times < SETTINGS["tmin"] + 0.3) | (times > SETTINGS["tmax"] - 0.3)
                middle = (times > 3.0) & (times < 4.0)
                self.assertLess(np.median(np.abs(erd[:, edge])), 1.5 * np.median(np.abs(erd[:, middle])) + 5.0,
                                "{} {}: edges".format(label, method))

    def test_trials_inside_a_long_recording(self):
        self.check(stationary(60.0, [5.0, 15.0, 25.0, 35.0, 45.0]), "inside")

    def test_trials_touching_the_start_and_end_of_the_recording(self):
        # the first trial's 0.5 s and the last trial's 6.5 s fall right at the recording's edges, where a wavelet
        # has nothing beyond the trial to see and the trial is mirrored instead. A mirror keeps the signal
        # continuous but turns its slope, so a little power leaks there (about 30 %, against 75 % with nothing):
        # allowed up to 40 %. Unfiltered: a filter has its own transient at a recording's first and last second,
        # whatever the time-frequency method, and that is not what is tested here.
        starts = [-0.5] + [7.0 * k for k in range(1, 7)] + [48.0]
        self.check(stationary(54.5, starts), "at the edges", flat=40.0, filters={})


if __name__ == "__main__":
    unittest.main()
