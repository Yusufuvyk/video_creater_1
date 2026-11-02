"""Trials picked by hand and their time-frequency maps: Select trials, Filter trials, the ideal filter,
Time-frequency fed by trials, one-trial maps, and the Cho2017 trial template.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.core import epochs as ep  # noqa: E402
from eeg.pipeline import Graph, Runner, api as ee, filters as fl, templates  # noqa: E402
from eeg.pipeline.blocks import LEFT_OUT, trial_numbers, trial_text  # noqa: E402
from eeg.pipeline.export import to_python  # noqa: E402

FS = 512


def cho_file(directory, trials=10, seed=0):
    """A Cho2017-shaped .mat: 64 EEG + 4 EMG rows of A/D counts, 7 s trials end to end, 10 Hz on C3 that
    drops to a quarter 2-5 s into every right-hand trial."""
    from scipy.io import savemat
    from eeg.core.montages import CHO2017_EEG
    rng = np.random.default_rng(seed)
    n = 7 * FS
    t = np.arange(n) / FS
    sides = {}
    for side in ("left", "right"):
        data = rng.normal(0, 32 * 2, (68, trials * n))
        for k in range(trials):
            envelope = np.where((t >= 2) & (t < 5), 0.25, 1.0) if side == "right" else 1.0
            data[CHO2017_EEG.index("C3"), k * n:(k + 1) * n] += 32 * 10 * envelope * np.sin(2 * np.pi * 10 * t)
        sides["imagery_" + side] = data
    path = directory / "s99.mat"
    savemat(str(path), {"eeg": dict(srate=float(FS), n_imagery_trials=float(trials), **sides)})
    return path


class TrialNumberTests(unittest.TestCase):
    def test_numbers_and_ranges_read_and_write_back(self):
        self.assertEqual(trial_numbers("1-3, 5, 7"), ["1", "2", "3", "5", "7"])
        self.assertEqual(trial_numbers(["8", "2-3", "3"]), ["8", "2", "3"])
        self.assertEqual(trial_text(["5", "1", "2", "3", "7", "8"]), ["1-3", "5", "7-8"])
        self.assertEqual(trial_text([]), [])
        for bad in ("a", "3-1", "1-x"):
            with self.assertRaises(ValueError, msg=bad):
                trial_numbers(bad)


class SelectTrialsTests(unittest.TestCase):
    def setUp(self):
        raw = lateralised(trials=10)
        self.trials = ee.epochs(raw, tmin=0.0, tmax=7.0)

    def test_keeps_the_named_trials_and_lists_the_rest_as_left_out(self):
        kept = ee.select_trials(self.trials, keep=["1-3", "5"], leave_out=["2"])
        self.assertEqual(kept.ids, ["1", "3", "5"])
        self.assertEqual(kept.labels, ["left", "left", "left"])
        np.testing.assert_array_equal(kept.data[1], self.trials.data[2])
        self.assertEqual([r[0] for r in kept.rejected], ["2", "4", "6", "7", "8", "9", "10"])
        self.assertTrue(all(r[2] == LEFT_OUT for r in kept.rejected))
        self.assertEqual(ee.select_trials(self.trials).ids, self.trials.ids)

    def test_unknown_numbers_and_nothing_left_are_refused(self):
        with self.assertRaisesRegex(Exception, "No trial 11 .* numbered 1-10"):
            ee.select_trials(self.trials, keep=["11"])
        with self.assertRaisesRegex(Exception, "Every trial is left out"):
            ee.select_trials(self.trials, keep=["1"], leave_out=["1"])


class IdealFilterTests(unittest.TestCase):
    def test_matches_an_fft_mask_and_is_zero_phase_only(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 5, 3584) + 40
        steps = fl.steps(FS, dict(low=8.0, high=40.0, notch=0.0, design="ideal"))
        out = fl.apply(x, FS, steps)
        # the lab's zero_phase_filter: bins from 8 to 40 Hz kept, edges included
        freqs = np.fft.rfftfreq(x.size, 1 / FS)
        expected = np.fft.irfft(np.fft.rfft(x) * ((freqs >= 8) & (freqs <= 40)), n=x.size)
        np.testing.assert_allclose(out, expected, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "zero-phase only"):
            fl.steps(FS, dict(low=8.0, high=40.0, design="ideal", phase="causal"))
        freqs, gain = fl.response(FS, dict(low=8.0, high=40.0, notch=0.0, design="ideal"))
        self.assertEqual(gain[(freqs > 9) & (freqs < 39)].max(), 0.0)
        self.assertLess(gain[freqs > 41].max(), -200)
        self.assertIn("ideal FFT band-pass 8–40 Hz", fl.describe(dict(low=8.0, high=40.0, design="ideal")))

    def test_filter_trials_never_crosses_from_one_trial_to_the_next(self):
        trials = ee.epochs(lateralised(trials=4), tmin=0.0, tmax=7.0)
        out = ee.trial_filter(trials, low=8.0, high=30.0, notch=0.0)
        chain = fl.steps(trials.fs, dict(low=8.0, high=30.0, notch=0.0))
        np.testing.assert_allclose(out.data[2, 1], fl.apply(trials.data[2, 1], trials.fs, chain))
        self.assertEqual(out.ids, trials.ids)


class TrialTimeFrequencyTests(unittest.TestCase):
    def test_trials_in_give_the_maps_the_recording_gives(self):
        raw = lateralised(trials=8)
        trials = ee.epochs(raw, tmin=0.0, tmax=7.0)
        settings = dict(channels=["C3", "C4"], tmin=0.0, tmax=7.0, baseline=[0.5, 1.5], active=[2.5, 4.5],
                        window=0.5, overlap=0.5)
        from_raw = ee.erd(raw, **settings)
        from_trials = ee.erd(trials, **settings)
        for a, b in zip(from_raw.tf, from_trials.tf):
            for condition in ("left", "right"):
                np.testing.assert_allclose(b.power[condition], a.power[condition], rtol=1e-5)
                np.testing.assert_allclose(b.amplitude[condition], a.amplitude[condition], rtol=1e-5)
        self.assertEqual(from_trials.epoch_ids, from_raw.epoch_ids)

    def test_what_was_done_to_the_trials_reaches_the_map(self):
        trials = ee.epochs(lateralised(trials=8), tmin=0.0, tmax=7.0)
        kept = ee.select_trials(trials, keep=["2", "4", "5"])
        result = ee.erd(kept, channels=[], baseline=[0.5, 1.5], active=[2.5, 4.5], smooth=0.0)
        self.assertEqual(result.epoch_ids, {"right": ["2", "4"], "left": ["5"]})
        self.assertEqual([c.name for c in result.tf], ["C3", "Cz", "C4"])
        self.assertEqual(sorted(r[0] for r in result.rejected), ["1", "3", "6", "7", "8"])
        self.assertAlmostEqual(result.settings.tmax, 7.0)

    def test_one_trial_maps(self):
        raw = lateralised(trials=6)
        result = ee.erd(ee.epochs(raw, tmin=0.0, tmax=7.0), channels=["C3"], baseline=[0.5, 1.5],
                        active=[2.5, 4.5], window=1.0, overlap=0.5, smooth=0.0)
        channel = result.tf[0]
        right = [ep.trial_map(result, channel, "right", k, ep.AMPLITUDE_MAP) for k in range(3)]
        # without smoothing, the trials' own amplitudes average to the condition's map
        np.testing.assert_allclose(np.mean(right, axis=0), ep.amplitude_map(channel, "right"), rtol=1e-5)
        row = int(np.argmin(np.abs(channel.freqs - 10)))
        before = channel.times < 2
        during = (channel.times > 2.6) & (channel.times < 4.4)
        # a 10 µV sine reads 10 µV, and half of it on C3 during right-hand imagery
        self.assertAlmostEqual(right[0][row, before].mean(), 10, delta=1.0)
        self.assertAlmostEqual(right[0][row, during].mean(), 5, delta=1.0)
        erd = ep.trial_map(result, channel, "right", 0, ep.ERD_MAP)
        self.assertAlmostEqual(np.nanmean(erd[row, during]), -75, delta=8)
        power = ep.trial_map(result, channel, "right", 0, ep.POWER_MAP)
        np.testing.assert_allclose(power, 10 * np.log10(channel.power["right"][0]), rtol=1e-6)
        self.assertEqual(ep.condition_map(result, channel, "right", ep.POWER_MAP).shape, power.shape)


class WaveletTests(unittest.TestCase):
    def setUp(self):
        self.trials = ee.epochs(lateralised(fs=FS, trials=4), tmin=0.0, tmax=7.0)

    def test_denoising_thresholds_the_details_of_each_trial(self):
        import pywt
        out = ee.wavelet_denoise(self.trials, level=4, mode="soft")
        x = self.trials.data[1, 2]
        coefficients = pywt.wavedec(x, "db4", level=4)
        sigma = np.median(np.abs(coefficients[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(coefficients[-1])))
        kept = [coefficients[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coefficients[1:]]
        np.testing.assert_allclose(out.data[1, 2], pywt.waverec(kept, "db4")[:x.size], atol=1e-10)
        fixed = ee.wavelet_denoise(self.trials, threshold=1000.0)
        # every detail gone: what is left is the signal rebuilt from its approximation alone
        x = self.trials.data[0, 0]
        coefficients = pywt.wavedec(x, "db4", level=4)
        alone = pywt.waverec([coefficients[0]] + [np.zeros_like(c) for c in coefficients[1:]], "db4")[:x.size]
        np.testing.assert_allclose(fixed.data[0, 0], alone, atol=1e-10)
        with self.assertRaisesRegex(Exception, "negative"):
            ee.wavelet_denoise(self.trials, threshold=-1.0)

    def test_dwt_bands_add_up_to_the_signal_and_its_energy(self):
        from eeg.core.wavelets import bands
        x = self.trials.data[0, 0]
        split = bands(x, FS, "db4", 6)
        self.assertEqual([(n, lo, hi) for n, lo, hi, _ in split][:4],
                         [("cA6", 0.0, 4.0), ("cD6", 4.0, 8.0), ("cD5", 8.0, 16.0), ("cD4", 16.0, 32.0)])
        np.testing.assert_allclose(sum(b[3] for b in split), x, atol=1e-9)
        envelopes, shares = ee.dwt_bands(self.trials)
        self.assertEqual(len(envelopes.names), 3 * 7)
        self.assertEqual(envelopes.names[2], "C3 · cD5 8–16 Hz")
        np.testing.assert_allclose(shares.values.reshape(4, 3, 7).sum(axis=2), 100.0)
        self.assertEqual(int(np.argmax(shares.values[0, :7])), 2)          # 10 Hz lives in 8-16 Hz
        self.assertTrue((envelopes.data >= 0).all())
        waves, _ = ee.dwt_bands(self.trials, envelope=False)
        np.testing.assert_allclose(waves.data[0, :7].sum(axis=0), x, atol=1e-9)

    def test_cwt_reads_a_sine_and_takes_only_complex_morlets(self):
        from eeg.core.wavelets import cwt_amplitude, cmor_parameters
        t = np.arange(4 * FS) / FS
        amplitude = cwt_amplitude(7 * np.sin(2 * np.pi * 12 * t), FS, [12.0, 30.0])
        self.assertAlmostEqual(amplitude[0, FS:3 * FS].mean(), 7.0, places=3)
        self.assertLess(amplitude[1, FS:3 * FS].mean(), 0.1)
        self.assertEqual(cmor_parameters("cmor1.5-1.0"), (1.5, 1.0))
        with self.assertRaisesRegex(ValueError, "cmorB-C"):
            cmor_parameters("morl")
        result = ee.erd(self.trials, channels=["C3"], method="cwt", baseline=[0.5, 1.5], active=[2.5, 4.5],
                        smooth=0.0)
        channel = result.tf[0]
        self.assertEqual(channel.freqs.size, ep.CWT_FREQS)
        self.assertAlmostEqual(channel.times[1] - channel.times[0], 10 / FS)
        row = int(np.argmin(np.abs(channel.freqs - 10)))
        self.assertAlmostEqual(channel.amplitude["left"][row, (channel.times > 0.8) & (channel.times < 1.8)].mean(),
                               10, delta=0.6)
        np.testing.assert_allclose(ep.trial_map(result, channel, "left", 0, ep.AMPLITUDE_MAP).shape,
                                   channel.amplitude["left"].shape)


class ChoTemplateTests(unittest.TestCase):
    def test_one_side_of_a_cho_file_has_one_event_per_trial(self):
        with scratch() as directory:
            raw = ee.load(path=str(cho_file(directory, trials=4)), condition="imagery_right")
            self.assertEqual(raw.events.labels, ["right"] * 4)
            np.testing.assert_allclose(raw.events.onsets, [0, 7, 14, 21])

    def test_the_trial_template_runs_and_exports(self):
        with scratch() as directory:
            graph = templates.trial_lab(str(cho_file(directory)))
            self.assertEqual(graph.problems(), [])
            out = Runner(graph).run()
            kept = out["select_trials1"]["epochs"]
            self.assertEqual(kept.ids, ["1", "2", "3", "5", "7", "8"])
            tf = out["erd1"]["tf"]
            channel = tf.tf[0]
            self.assertEqual(channel.name, "C3")
            self.assertAlmostEqual(channel.freqs[1] - channel.freqs[0], 1.0)
            row = int(np.argmin(np.abs(channel.freqs - 10)))
            amplitude = ep.amplitude_map(channel, "right")[row]
            self.assertGreater(amplitude[channel.times < 2].mean(), 2.5 * amplitude[
                (channel.times > 2.5) & (channel.times < 4.5)].mean())
            cwt = out["erd2"]["tf"]
            self.assertEqual(cwt.settings.method, ep.CWT)
            self.assertEqual(out["dwt_bands1"]["features"].values.shape, (6, 7))
            self.assertEqual(out["wavelet_denoise1"]["epochs"].ids, kept.ids)
            script = to_python(graph)
            for line in ("select_trials(", "filter_trials(", "design='ideal'", "time_frequency(",
                         "show_tf(out['erd1'])", "method='wavelet'", "width='one wavelet'", "dwt_bands(",
                         "wavelet_denoise(", "PyWavelets"):
                self.assertIn(line, script)

    def test_show_draws_a_time_frequency_map(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        result = ee.erd(ee.epochs(lateralised(trials=4), tmin=0.0, tmax=7.0), channels=["C3"],
                        baseline=[0.5, 1.5], active=[2.5, 4.5])
        with mock.patch.object(plt, "show"):
            figure = ee.show(result, kind="amplitude")
        self.assertIn("C3", figure.axes[0].get_title())
        plt.close(figure)


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyqtgraph.Qt import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_map_kind_and_trial_are_chosen_in_the_preview(self):
        from eeg.ui.workbench.previews import TFPreview, preview
        result = ee.erd(ee.epochs(lateralised(trials=6), tmin=0.0, tmax=7.0), channels=["C3"],
                        baseline=[0.5, 1.5], active=[2.5, 4.5], smooth=0.0)
        widget = preview(result)
        self.assertIsInstance(widget, TFPreview)
        self.assertEqual([widget.trial.itemText(i) for i in range(widget.trial.count())],
                         ["Average of 3", "Trial 1", "Trial 3", "Trial 5"])
        widget.kind_box.setCurrentText("Amplitude (µV)")
        widget.trial.setCurrentIndex(2)
        shown = widget.image.image.T
        np.testing.assert_allclose(shown, ep.trial_map(result, result.tf[0], "left", 1, ep.AMPLITUDE_MAP), rtol=1e-5)
        self.assertIn("Trial 3 of left", widget.caption.text())
        self.assertEqual(widget.bar.axis.labelText, "Amplitude (µV)")
        widget.condition.setCurrentIndex(1)
        self.assertEqual(widget.trial.itemText(1), "Trial 2")
        TFPreview.kind = ep.ERD_MAP

    def test_head_maps_over_time_the_grid_in_head_order_and_every_trial_saved(self):
        import pyqtgraph as pg
        from eeg.ui.workbench import previews as pv
        with scratch() as directory:
            raw = ee.load(path=str(cho_file(directory, trials=3)))
            result = ee.erd(ee.epochs(raw, tmin=0.0, tmax=7.0), channels=[], baseline=[0.5, 1.5],
                            active=[2.5, 4.5])
            self.assertGreater(len(result.tf), 60)                       # empty Channels: every electrode
            c3 = next(tf for tf in result.tf if tf.name == "C3")
            # one band over time, as the maps draw it: the pooled ERD of the band course already used
            trace = ep.band_trace(result, c3, "right", (8.0, 13.0))
            np.testing.assert_allclose(trace, ep.band_course(c3, "right", (8.0, 13.0))[0])
            self.assertLess(np.nanmean(trace[(c3.times > 2.5) & (c3.times < 4.5)]), -50)
            widget = pv.preview(result)
            self.assertEqual(widget.channel.findText("On the head"), -1)   # dropped: All channels and the maps say it
            # All channels: front to back, left to right in a row, as the electrodes sit on the head
            widget.channel.setCurrentText(pv.ALL_CHANNELS)
            titles = [item.titleLabel.text for item in widget.grid.items]
            self.assertEqual(titles[:3], ["Fp1", "Fpz", "Fp2"])
            row = [t for t in titles if t in ("T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8")]
            self.assertEqual(row, ["T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8"])
            self.assertLess(titles.index("Fz"), titles.index("Cz"))
            self.assertLess(titles.index("Cz"), titles.index("Pz"))
            # another condition: the same plots, emptied and drawn again, and still one colour bar
            plots, bar = list(widget.grid.items), widget.grid.bar
            widget.condition.setCurrentIndex(1)
            self.assertEqual(widget.grid.items, plots)
            self.assertIs(widget.grid.bar, bar)
            self.assertEqual(sum(isinstance(i, pg.ImageItem) for i in plots[0].items), 1)
            self.assertFalse(widget.band.isVisibleTo(widget))            # the band belongs to maps over time
            widget.channel.setCurrentText(pv.OVER_TIME)
            widget.step.setCurrentText("1 s")
            self.assertTrue(widget.series.isVisibleTo(widget) and widget.band.isVisibleTo(widget))
            maps = [item.titleLabel.text for item in widget.series.ci.items
                    if isinstance(item, pg.PlotItem) and not isinstance(item, pg.ColorBarItem)]
            self.assertEqual(maps, ["{}–{} s".format(k, k + 1) for k in range(7)])     # 0-7 s, 1 s each
            self.assertIn("Band mu", " · ".join("{} {}".format(n, pv._choice(b)) for n, b in widget.said_choices()))
            self.assertFalse(widget.keep_band.isVisibleTo(widget))      # the block's own band: nothing to add
            # any band typed: drawn at once from what was computed, and offered to the block
            widget.band.setCurrentText("6-20")
            self.assertIn("6–20 Hz", widget.caption.text())
            self.assertTrue(widget.keep_band.isVisibleTo(widget))
            trace = ep.band_trace(result, c3, "right", (6.0, 20.0))
            np.testing.assert_allclose(widget.traces["C3"], trace)
            widget.band.setCurrentText("alpha 9,5-12.5 Hz")              # a name, a decimal comma, Hz
            self.assertIn("alpha 9.5–12.5 Hz", widget.caption.text())
            for typed, said in (("2-60", "outside what the block computed"), ("9-9.5", "No frequency bin"),
                                ("twelve", "low-high in Hz"), ("13-8", "low edge")):
                widget.band.setCurrentText(typed)
                self.assertIn(said, widget.caption.text(), typed)
                self.assertFalse(widget.keep_band.isVisibleTo(widget), typed)
            widget.band.setCurrentText("6-20")
            widget.step.setCurrentText("0,75 s")                          # any length typed
            maps = [item.titleLabel.text for item in widget.series.ci.items
                    if isinstance(item, pg.PlotItem) and not isinstance(item, pg.ColorBarItem)]
            self.assertEqual(maps[:2], ["0–0.75 s", "0.75–1.5 s"])
            widget.step.setCurrentText("0.01")
            self.assertIn("one frame", widget.caption.text())
            widget.step.setCurrentText("1 s")
            # the larger window opens on the same view, band and step, the typed band too
            large = pv.preview(result, large=True).choose(widget.choices())
            self.assertEqual((large.channel.currentText(), large.band.currentText(), large.step.currentText()),
                             (pv.OVER_TIME, "6-20", "1 s"))
            self.assertIn("6–20 Hz", large.caption.text())
            # every trial of every condition, and each condition's average, drawn as chosen
            widget.channel.setCurrentText("C3")
            (directory / "trials").mkdir()
            written = widget.save_every_trial(directory / "trials")
            names = sorted(p.name for p in written)
            self.assertEqual(len(written), sum(len(ids) + 1 for ids in result.epoch_ids.values()))
            self.assertIn("left_average.png", names)
            # named by the trial numbers the Trial menu shows
            self.assertEqual([n for n in names if n.startswith("right_trial")],
                             ["right_trial{}.png".format(str(i).zfill(3)) for i in result.epoch_ids["right"]])
            self.assertTrue(all(p.stat().st_size > 1000 for p in written))

    def test_a_typed_band_is_added_to_the_block(self):
        from eeg.ui.workbench import WorkbenchPage
        from eeg.ui.workbench import previews as pv
        from test_pipeline import save_fif
        from test_workbench import fresh_sheet, wait_for_run
        fresh_sheet(self)
        with scratch() as directory:
            path = save_fif(lateralised(trials=8), directory)
            page = WorkbenchPage()
            graph = Graph("bands")
            graph.chain(("load", {"path": str(path)}), ("epochs", {"tmin": 0.0, "tmax": 7.0}),
                        ("erd", {"channels": [], "baseline": [0.5, 1.5], "active": [2.5, 4.5]}))
            page.set_graph(graph)
            page.run()
            wait_for_run(page)
            page.canvas.select("erd1")
            widget = page.inspector.result
            widget.channel.setCurrentText(pv.OVER_TIME)
            widget.band.setCurrentText("6-20")
            self.assertTrue(widget.keep_band.isVisibleTo(widget))
            widget.keep_band.click()
            bands = page.graph.params("erd1")["bands"]
            self.assertEqual(bands[-1], ["band_6_20", 6.0, 20.0])
            self.assertEqual([b[0] for b in bands[:2]], ["mu", "beta"])      # the block's own bands stay
            self.assertIn("band_6_20 6-20", page.inspector.fields["bands"].widget.text())
            self.assertTrue(page.modified)
            self.assertEqual(page.canvas.cards["erd1"].state, "stale")         # computed again at the next Run
            page.run()
            wait_for_run(page)
            page.canvas.select("erd1")
            widget = page.inspector.result
            # the view kept: maps over time on the band typed, now one of the block's own
            self.assertEqual((widget.channel.currentText(), widget.band.currentText()), (pv.OVER_TIME, "6-20"))
            self.assertFalse(widget.keep_band.isVisibleTo(widget))
            self.assertIn("band_6_20 6–20 Hz", [widget.band.itemText(i) for i in range(widget.band.count())])
            page.sheet.modified = False

    def test_trials_are_picked_by_clicking_them(self):
        from eeg.ui.workbench import WorkbenchPage
        from test_pipeline import save_fif
        from test_workbench import wait_for_run
        with scratch() as directory:
            path = save_fif(lateralised(trials=12), directory)
            page = WorkbenchPage()
            graph = Graph("pick")
            graph.chain(("load", {"path": str(path)}), ("epochs", {"tmin": 0.0, "tmax": 7.0}),
                        ("trial_filter", {"low": 1.0, "high": 40.0, "notch": 0.0}), "select_trials",
                        ("erd", {"channels": ["C3"], "baseline": [0.5, 1.5], "active": [2.5, 4.5]}))
            page.set_graph(graph)
            page.canvas.select("select_trials1")
            self.assertIsNone(page.inspector.picker)          # nothing to pick from before a run
            page.run()
            wait_for_run(page)
            page.canvas.select("select_trials1")
            picker = page.inspector.picker
            self.assertEqual(picker.value.ids, [str(k) for k in range(1, 13)])
            self.assertEqual(picker.show_box.currentText(), "Each trial")
            self.assertTrue(picker.start.isVisibleTo(picker))      # 12 trials, 10 at a time
            page.inspector._toggle_trial("4")
            page.inspector._toggle_trial("12")
            self.assertEqual(page.graph.params("select_trials1")["keep"], ["1-3", "5-11"])
            self.assertEqual(page.inspector.fields["keep"].widget.text(), "1-3, 5-11")
            self.assertIn("Kept 10 of 12", picker.caption.text())
            self.assertEqual(page.canvas.cards["erd1"].state, "stale")
            page.run()
            wait_for_run(page)
            tf = page.outputs()["erd1"]["tf"]
            self.assertEqual(sum(len(v) for v in tf.epoch_ids.values()), 10)
            page.canvas.select("select_trials1")
            page.inspector._toggle_trial("4")
            page.inspector._toggle_trial("12")
            self.assertEqual(page.graph.params("select_trials1")["keep"], [])


if __name__ == "__main__":
    unittest.main()
