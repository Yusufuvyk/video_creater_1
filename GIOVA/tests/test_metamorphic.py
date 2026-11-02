"""Metamorphic relations: how results must change, or not, when the input changes in a known way.

An EEG analysis has no oracle for its exact output: nobody knows the "right" ERD of a recording. What is known is
how the output must follow a change of input (Kanewala & Bieman, 2014, Information and Software Technology 56(10),
1219-1232; Segura et al., 2016, IEEE TSE 42(9), 805-824), and a relation that fails is a bug whatever the true
values are:

    amplitude × c        spectra × c², band powers × c²; ERD %, the model's features up to a constant, and every
                         prediction unchanged
    channels reordered   every channel's spectrum and ERD unchanged (looked up by name), every prediction unchanged
    time shifted         a recording that starts earlier, its events shifted with it: the same trials, the same
                         ERD and the same predictions
    conditions renamed   the same accuracy and predictions, under the new names
    stored in volts      read through the Units block, the same numbers as the same recording stored in µV

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

from test_epochs import lateralised  # noqa: E402
from eeg.core import epochs as ep  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402
from eeg.pipeline import api as ee  # noqa: E402

ERD = dict(channels=["C3", "Cz", "C4"], tmin=0.0, tmax=7.0, baseline=[0.5, 1.5], active=[2.5, 4.5], fmin=4.0,
           fmax=30.0, window=0.5, overlap=0.8)


def changed(recording, data=None, names=None, labels=None, shift=0.0):
    """The recording with other samples, channel names, condition names or event times."""
    events = recording.events
    if labels is not None or shift:
        events = ep.EventTable(np.asarray(events.onsets) + shift, events.durations,
                               [labels.get(label, label) for label in events.labels] if labels else events.labels)
    return Recording(recording.data if data is None else data, recording.fs,
                     recording.channel_names if names is None else names, events=events)


def results(raw):
    """What the relations are checked on: per trial spectra, band powers, ERD and cross-validated predictions."""
    filtered = ee.filter(raw, low=6.0, high=32.0, notch=0.0)
    trials = ee.epochs(filtered, tmin=0.0, tmax=7.0)
    spectrum = ee.psd(trials, window=1.0)
    scores = ee.cross_validate(trials, ee.lda(ee.csp(n_components=2)), folds=5)
    return dict(trials=trials, spectrum=spectrum, bands=ee.band_power(spectrum, log=False),
                erd=ee.erd(raw, **ERD), scores=scores)


def channel(result, name):
    """A channel's spectra and ERD maps, by name."""
    k = result["spectrum"].names.index(name)
    tf = next(t for t in result["erd"].tf if t.name == name)
    return result["spectrum"].power[:, k], {c: ep.erd_map(tf, c) for c in tf.power}


def predicted(scores, names=None):
    return [(trial, (names or {}).get(guess, guess)) for trial, _, guess, _ in scores.predictions]


class MetamorphicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = lateralised(trials=20, seed=4)
        cls.base = results(cls.raw)

    def test_amplitude_scales_power_and_nothing_else(self):
        c = 37.5
        scaled = results(changed(self.raw, data=self.raw.data * c))
        np.testing.assert_allclose(scaled["spectrum"].power, self.base["spectrum"].power * c ** 2, rtol=1e-7)
        np.testing.assert_allclose(scaled["bands"].values, self.base["bands"].values * c ** 2, rtol=1e-7)
        # time-frequency power is kept in single precision (memory), so ERD % agrees to its 7 digits
        for name in ERD["channels"]:
            for condition, erd in channel(self.base, name)[1].items():
                np.testing.assert_allclose(channel(scaled, name)[1][condition], erd, rtol=1e-5, atol=1e-4)
        self.assertEqual(predicted(scaled["scores"]), predicted(self.base["scores"]))
        self.assertEqual(scaled["scores"].accuracy, self.base["scores"].accuracy)

    def test_channel_order_changes_nothing_by_name(self):
        order = [2, 0, 1]
        names = [self.raw.channel_names[k] for k in order]
        moved = results(changed(self.raw, data=self.raw.data[order], names=names))
        for name in self.raw.channel_names:
            spectra, erd = channel(moved, name)
            np.testing.assert_allclose(spectra, channel(self.base, name)[0], rtol=1e-9)
            for condition, values in channel(self.base, name)[1].items():
                np.testing.assert_allclose(erd[condition], values, rtol=1e-9, atol=1e-9)
        self.assertEqual(predicted(moved["scores"]), predicted(self.base["scores"]))

    def test_an_earlier_start_with_its_events_moved_changes_nothing(self):
        lead = 3.0                                             # whole samples: 750 at 250 Hz
        before = np.random.default_rng(9).normal(0, 1, (self.raw.n_channels, int(lead * self.raw.fs)))
        earlier = changed(self.raw, data=np.hstack([before, self.raw.data]), shift=lead)
        np.testing.assert_array_equal(ee.epochs(earlier, tmin=0.0, tmax=7.0).data,
                                      ee.epochs(self.raw, tmin=0.0, tmax=7.0).data)
        erd = ee.erd(earlier, **ERD)
        for tf, base in zip(erd.tf, self.base["erd"].tf):
            for condition in base.power:
                np.testing.assert_allclose(ep.erd_map(tf, condition), ep.erd_map(base, condition), rtol=1e-9)
        # the filter runs over the whole recording, so its start reaches the trials only as rounding
        scores = results(earlier)["scores"]
        self.assertEqual(predicted(scores), predicted(self.base["scores"]))

    def test_renamed_conditions_are_the_same_conditions(self):
        # names in the same sorted order: the folds are drawn condition by condition in that order
        names = {"left": "imagery L", "right": "imagery R"}
        renamed = results(changed(self.raw, labels=names))
        self.assertEqual(renamed["scores"].accuracy, self.base["scores"].accuracy)
        self.assertEqual(predicted(renamed["scores"]), predicted(self.base["scores"], names))
        self.assertEqual(sorted(renamed["erd"].conditions), sorted(names.values()))
        for tf, base in zip(renamed["erd"].tf, self.base["erd"].tf):
            for old, new in names.items():
                np.testing.assert_array_equal(ep.erd_map(tf, new), ep.erd_map(base, old))

    def test_volts_through_the_units_block_are_microvolts(self):
        volts = changed(self.raw, data=self.raw.data * 1e-6)
        read = ee.units(volts, stored="V")
        np.testing.assert_allclose(read.data, self.raw.data, rtol=1e-12)
        np.testing.assert_allclose(ee.psd(ee.epochs(read, tmin=0.0, tmax=7.0)).power,
                                   ee.psd(ee.epochs(self.raw, tmin=0.0, tmax=7.0)).power, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
