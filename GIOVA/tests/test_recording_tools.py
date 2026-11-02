"""What a recording says about its units, events renamed and moved, motion, bad channels, and the
previews' every-channel grid and channel means.

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
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtWidgets  # noqa: E402

from test_trial_analysis import scratch  # noqa: E402
from eeg.pipeline import api as ee  # noqa: E402
from eeg.pipeline import PipelineError  # noqa: E402
from eeg.pipeline.data import derive  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

NAMES = ["Fp1", "Fz", "Cz", "Pz", "T7", "T8", "P7", "P8"]
RESOLUTION = 0.0406901


def brainvision(directory, seconds=60, fs=500, motion=True, seed=0, stem="rec"):
    """A LiveAmp-like BrainVision file: 8 EEG channels stored with a resolution, a 3-axis accelerometer in mg,
    a 'P  1' marker every 5 s; P8 carries only noise. Returns (path, EEG in µV as recorded)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    eeg = 10 * np.sin(2 * np.pi * 10 * t)[None, :] * np.linspace(1, 0.6, 8)[:, None] + rng.normal(0, 2, (8, n))
    eeg[7] = rng.normal(0, 60, n)                            # P8: a loose electrode
    acc = rng.normal(0, 2, (3, n))
    acc[0, int(20.5 * fs):int(21 * fs)] += 300               # the head moved during the fifth trial
    channels = list(NAMES) + (["x_dir", "y_dir", "z_dir"] if motion else [])
    stored = np.vstack([eeg / RESOLUTION] + ([acc] if motion else [])).astype("<f4")
    base = directory / stem
    stored.T.tofile(str(base) + ".eeg")
    lines = ["Brain Vision Data Exchange Header File Version 1.0", "[Common Infos]", "Codepage=UTF-8",
             "DataFile={}.eeg".format(stem), "MarkerFile={}.vmrk".format(stem), "DataFormat=BINARY",
             "DataOrientation=MULTIPLEXED", "NumberOfChannels={}".format(len(channels)),
             "SamplingInterval={}".format(int(1e6 / fs)), "[Binary Infos]", "BinaryFormat=IEEE_FLOAT_32",
             "[Channel Infos]"]
    lines += ["Ch{}={},,{},µV".format(k + 1, name, RESOLUTION) for k, name in enumerate(NAMES)]
    if motion:
        lines += ["Ch{}={},,1,mg".format(9 + k, name) for k, name in enumerate(["x_dir", "y_dir", "z_dir"])]
    Path(str(base) + ".vhdr").write_text("\n".join(lines) + "\n", encoding="utf-8")
    markers = ["Brain Vision Data Exchange Marker File Version 1.0", "[Common Infos]", "Codepage=UTF-8",
               "DataFile={}.eeg".format(stem), "[Marker Infos]", "Mk1=New Segment,,1,1,0"]
    markers += ["Mk{}=Primary,P  1,{},1,0".format(k + 2, int((k * 5 + 0.5) * fs) + 1) for k in range(11)]
    Path(str(base) + ".vmrk").write_text("\n".join(markers) + "\n", encoding="utf-8")
    return Path(str(base) + ".vhdr"), eeg


class RecordingTests(unittest.TestCase):
    def test_the_header_resolution_is_applied_and_said(self):
        with scratch() as directory:
            path, eeg = brainvision(directory)
            raw = ee.load(path=str(path))
        np.testing.assert_allclose(raw.data, eeg, rtol=1e-5, atol=1e-3)     # µV as recorded, not ×24.6
        self.assertIn("× 0.0406901 µV", raw.unit_note)
        self.assertIn("BrainVision", raw.unit_note)
        self.assertEqual(raw.channel_names, NAMES)                           # the accelerometer is not EEG
        self.assertEqual(raw.motion_names, ["x_dir", "y_dir", "z_dir"])
        self.assertEqual(raw.motion_units, ["mg"] * 3)
        self.assertIn("× 0.5 (Units / gain)", ee.units(raw, gain=2.0).unit_note)

    def test_events_renamed_and_moved(self):
        with scratch() as directory:
            path, _ = brainvision(directory)
            plain = ee.load(path=str(path))
            raw = ee.load(path=str(path), rename=["primary/p 1=odour"], shift_ms=30.0)
            self.assertEqual(raw.events.unique_labels(), ["odour"])          # case and spaces aside
            np.testing.assert_allclose(raw.events.onsets, plain.events.onsets + 0.030)
            with self.assertRaisesRegex(PipelineError, "No event named P2"):
                ee.load(path=str(path), rename=["P2=x"])
            with self.assertRaisesRegex(PipelineError, "old=new"):
                ee.load(path=str(path), rename=["odour"])

    def test_a_trial_that_moved_is_left_out(self):
        with scratch() as directory:
            path, _ = brainvision(directory)
            trials = ee.epochs(ee.load(path=str(path)), tmin=0.0, tmax=4.0, motion_limit=100.0)
            self.assertEqual(len(trials.labels), 10)
            self.assertEqual(trials.rejected, [("5", "Primary/P  1", "Motion > 100 mg (x_dir)")])
            still, _ = brainvision(directory, motion=False, stem="still")
            with self.assertRaisesRegex(PipelineError, "no motion channels"):
                ee.epochs(ee.load(path=str(still)), motion_limit=100.0)

    def test_a_bad_channel_is_rebuilt_from_its_neighbours(self):
        with scratch() as directory:
            path, _ = brainvision(directory)
            raw = ee.filter(ee.load(path=str(path)), low=1.0, high=30.0, notch=0.0)
        fixed = ee.bad_channels(raw, channels=["P8"])
        dropped = ee.bad_channels(raw, channels=["P8"], action="drop")
        self.assertEqual(fixed.channel_names, NAMES)
        np.testing.assert_array_equal(fixed.data[:7], raw.data[:7])         # the good ones untouched
        self.assertLess(fixed.data[7].std(), 0.5 * raw.data[7].std())       # the noise gone, the 10 Hz back
        self.assertGreater(np.corrcoef(fixed.data[7], fixed.data[6])[0, 1], 0.9)
        self.assertEqual(dropped.channel_names, NAMES[:7])
        # dropping every channel is an error, not the recording given back whole
        with self.assertRaisesRegex(PipelineError, "leaves no channel"):
            ee.bad_channels(raw, channels=NAMES, action="drop")
        with self.assertRaisesRegex(PipelineError, "not a 10-20"):
            ee.bad_channels(derive(raw, names=NAMES[:7] + ["E8"]), channels=["Fz"])


class CompareWindowsTests(unittest.TestCase):
    def test_a_band_that_grows_in_one_window_is_found_and_the_rest_are_not(self):
        from eeg.core.epochs import EventTable
        from eeg.core.sources import Recording
        rng = np.random.default_rng(4)
        fs, n_trials = 250.0, 30
        t = np.arange(int(10 * fs)) / fs
        trials = []
        for _ in range(n_trials):
            x = rng.normal(0, 3, (3, t.size))
            x[1] += np.where(t < 2, 8.0, 2.0) * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6.3))   # theta on Cz, 0-2 s
            trials.append(x)
        raw = Recording(np.hstack(trials), fs, ["Fz", "Cz", "Pz"],
                        events=EventTable(np.arange(n_trials) * 10.0, [10.0] * n_trials, ["odour"] * n_trials))
        result = ee.compare_windows(ee.epochs(raw, tmin=0.0, tmax=10.0), window_a=[0.0, 2.0],
                                    window_b=[4.0, 10.0], name_a="odour", name_b="air")
        found = {(r["channel"], r["band"]) for r in result.significant()}
        self.assertEqual(found, {("Cz", "theta")})
        row = next(r for r in result.rows if (r["channel"], r["band"]) == ("Cz", "theta"))
        self.assertGreater(row["change"], 300)                      # 8² over 2²: about 16 times the power
        self.assertEqual(row["trials_up"], 1.0)
        self.assertLessEqual(row["p"], row["p_holm"])
        self.assertEqual(result.table()[0][4], "mean_odour_uV2")
        with self.assertRaisesRegex(PipelineError, "inside the trials"):
            ee.compare_windows(ee.epochs(raw, tmin=0.0, tmax=10.0), window_b=[8.0, 12.0])


class PlayerTests(unittest.TestCase):
    def test_the_recording_plays_with_the_spectrum_of_each_moment(self):
        from eeg.ui.workbench.previews import preview
        from eeg.ui.workbench.player import MEAN_OF_ALL
        with scratch() as directory:
            path, _ = brainvision(directory)
            raw = ee.load(path=str(path))
        player = preview(raw).play()
        try:
            player.seek(30.0)
            freqs, amplitude = player.now_curve.getData()
            self.assertAlmostEqual(freqs[np.argmax(amplitude)], 10.0, delta=0.6)   # the 10 Hz of Fz... is there
            self.assertIn("after Primary/P  1", player.clock.text())
            self.assertAlmostEqual(player.cursor.value(), 30.0)
            player.speed.setCurrentText("10×")
            player.toggle()
            for _ in range(5):
                player._step()
            self.assertAlmostEqual(player.position, 32.0, places=6)               # 5 frames of 40 ms at 10×
            player.toggle()
            player.channel.setCurrentText(MEAN_OF_ALL)
            self.assertIn(tuple(range(8)), player.cache)
            player.seek(1e9)
            self.assertEqual(player.position, raw.duration)
        finally:
            player.close()


class GridTests(unittest.TestCase):
    def test_every_channel_side_by_side_and_a_mean_of_some(self):
        from eeg.ui.workbench.previews import ALL_CHANNELS, mean_text, preview
        with scratch() as directory:
            path, _ = brainvision(directory)
            trials = ee.epochs(ee.load(path=str(path)), tmin=0.0, tmax=4.0)
        spectrum = ee.psd(trials, window=1.0)
        widget = preview(spectrum)
        widget.scale_box.setCurrentText("Linear")
        widget.channel.setCurrentText(ALL_CHANNELS)
        self.assertEqual(len(widget.grid.items), 8)
        self.assertFalse(widget.plot.isVisibleTo(widget))
        names = widget.channel_names
        widget.mean_of = [names.index("Fz"), names.index("Cz")]
        widget.channel.insertItem(widget.channel.count() - 1, mean_text(names, widget.mean_of))
        widget.channel.setCurrentIndex(widget.channel.count() - 2)
        _, y = widget.plot.getPlotItem().listDataItems()[0].getData()
        from eeg.ui.workbench.previews import SPECTRUM_HZ
        keep = (spectrum.freqs >= SPECTRUM_HZ[0]) & (spectrum.freqs <= SPECTRUM_HZ[1])
        np.testing.assert_allclose(y, spectrum.power[:, [1, 2]][:, :, keep].mean(axis=(0, 1)), rtol=1e-6)
        self.assertIn("Fz, Cz", widget.caption.text())
        maps = preview(ee.erd(trials, channels=[], baseline=[0.0, 0.5], active=[1.0, 3.0]))
        maps.channel.setCurrentText(ALL_CHANNELS)
        self.assertEqual(len(maps.grid.items), 8)


class MarkedBadTests(unittest.TestCase):
    """Spans a file marks as bad (MNE's BAD_ annotations, a boundary) leave the trials over them out, as MNE's
    epochs do, with the mark as the reason; they are never trials themselves."""

    def test_a_trial_over_a_bad_span_or_a_boundary_is_left_out(self):
        import mne
        from eeg.pipeline import compute
        fs = 250.0
        info = mne.create_info(["C3", "Cz", "C4"], fs, "eeg")
        raw = mne.io.RawArray(np.random.default_rng(0).normal(0, 5e-6, (3, int(50 * fs))), info, verbose="error")
        raw.set_annotations(mne.Annotations([0.0, 12.0, 24.0, 36.0, 5.0, 30.0], [0.0] * 4 + [3.0, 0.0],
                                            ["A", "A", "A", "A", "BAD_motion", "boundary"]))
        with scratch() as directory:
            path = directory / "marked_raw.fif"
            raw.save(str(path), verbose="error")
            recording = compute.read_recording(path)
        self.assertEqual(recording.events.labels, ["A"] * 4)
        self.assertEqual(recording.bad_spans, [(5.0, 8.0, "BAD_motion"), (30.0, 30.0, "boundary")])
        trials = compute.cut_epochs(recording, tmin=0.0, tmax=10.0)
        self.assertEqual(trials.ids, ["2", "4"])
        self.assertEqual(trials.rejected, [("1", "A", "Marked BAD_motion in the file"),
                                           ("3", "A", "Marked boundary in the file")])
        # trials defined by hand are judged the same way, and a filtered copy keeps the marks
        filtered = compute.filter_recording(recording, low=1.0, high=40.0)
        self.assertEqual(filtered.bad_spans, recording.bad_spans)
        manual = compute.cut_epochs(filtered, tmin=0.0, tmax=4.0, source="manual",
                                    trials=[["a", "A", 4.0, 8.0], ["b", "A", 8.0, 12.0], ["c", "A", 40.0, 44.0]])
        self.assertEqual(manual.ids, ["b", "c"])
        self.assertEqual(manual.rejected, [("a", "A", "Marked BAD_motion in the file")])


class ChannelLogTests(unittest.TestCase):
    """The amplifier's on/off log is by channel position: picking or reordering channels takes it along."""

    def test_a_switched_off_channel_stays_the_one_named(self):
        from eeg.core.sources import Recording
        from eeg.pipeline import compute
        rng = np.random.default_rng(1)
        raw = Recording(rng.normal(0, 5, (3, 2500)), 250.0, ["C3", "Cz", "C4"])
        raw.channel_log = [(0.0, [True, False, True])]                  # Cz off from the start
        trials = [["a", "A", 0.0, 4.0], ["b", "A", 4.0, 8.0]]

        def unmeasured(picked):
            return compute.cut_epochs(picked, tmin=0.0, tmax=4.0, source="manual", trials=trials).unmeasured

        self.assertEqual(unmeasured(raw), ["Cz"])
        self.assertEqual(unmeasured(compute.pick_channels(raw, ["C4", "C3"])), [])
        self.assertEqual(unmeasured(compute.pick_channels(raw, ["C4", "C3", "Cz"])), ["Cz"])
        self.assertEqual(unmeasured(compute.repair_channels(raw, ["C3"], action="drop")), ["Cz"])


class BrainVisionCodepageTests(unittest.TestCase):
    """A UTF-8 header naming veri/ölçüm.eeg: the file watched for changes is the one MNE reads."""

    def test_a_changed_data_file_with_a_turkish_name_is_read_again(self):
        from eeg.formats.recordings import dependencies
        from eeg.pipeline import Graph, Runner
        with scratch() as directory:
            header = directory / "valid.vhdr"
            (directory / "veri").mkdir()
            data = directory / "veri" / "ölçüm.eeg"
            header.write_bytes("\n".join([
                "Brain Vision Data Exchange Header File Version 1.0", "[Common Infos]", "Codepage=UTF-8",
                "DataFile=veri/ölçüm.eeg", "MarkerFile=veri/ölçüm.vmrk", "DataFormat=BINARY",
                "DataOrientation=MULTIPLEXED", "NumberOfChannels=1", "SamplingInterval=10000",
                "[Binary Infos]", "BinaryFormat=IEEE_FLOAT_32", "[Channel Infos]", "Ch1=C3,,1,µV", ""]).encode("utf-8"))
            (directory / "veri" / "ölçüm.vmrk").write_bytes("\n".join([
                "Brain Vision Data Exchange Marker File, Version 1.0", "[Common Infos]", "Codepage=UTF-8",
                "DataFile=ölçüm.eeg", "[Marker Infos]", "Mk1=New Segment,,1,1,0", ""]).encode("utf-8"))
            self.assertIn(data, dependencies(header))
            data.write_bytes(np.full(500, 10.0, dtype="<f4").tobytes())
            graph = Graph("t")
            load = graph.add("load", {"path": str(header)})
            runner = Runner(graph)
            self.assertAlmostEqual(float(runner.run([load])[load]["raw"].data.mean()), 10.0, places=4)
            data.write_bytes(np.full(500, 20.0, dtype="<f4").tobytes())       # same size, other numbers
            self.assertAlmostEqual(float(runner.run([load])[load]["raw"].data.mean()), 20.0, places=4)


if __name__ == "__main__":
    unittest.main()
