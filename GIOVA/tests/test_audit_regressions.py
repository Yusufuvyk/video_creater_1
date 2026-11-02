"""Reproductions of the September 2026 numerical and lifecycle audit findings."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import time
import unittest
from unittest.mock import patch

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import sosfreqz

from test_trial_analysis import scratch
from eeg.core import boards, epochs
from eeg.core.sources import (Recording, FileSource, load_brainflow_csv,
                              write_board_metadata, board_metadata_path)
from eeg.core.processing import design_filter, decimate_minmax
from eeg.core.impedance import estimate
from eeg.core.recording_clock import sample_positions
from eeg.core.spectral import welch_psd
from eeg.core.trials import Trial, Settings, analyze


def csv_rows(board, seconds=4):
    t = np.arange(round(board.fs * seconds)) / board.fs
    rows = np.zeros((len(t), board.n_rows))
    rows[:, list(board.eeg_rows)] = (3 * np.sin(2 * np.pi * 10 * t))[:, None]
    rows[:, board.timestamp_row] = 1700000000 + t
    return rows


class RecordingRegressions(unittest.TestCase):
    def test_device_rate_duration_channels_and_frequency(self):
        with scratch() as directory:
            for board in boards.CSV_BOARDS:
                with self.subTest(board=board.key):
                    path = directory / (board.key + ".csv")
                    np.savetxt(path, csv_rows(board), delimiter="\t")
                    rec = load_brainflow_csv(path)
                    self.assertEqual((rec.fs, rec.n_channels, rec.duration), (board.fs, board.n_channels, 4))
                    freqs, power = welch_psd(rec.data[0], rec.fs)
                    self.assertEqual(freqs[power.argmax()], 10)

    def test_unknown_and_ambiguous_layouts_are_rejected(self):
        with scratch() as directory:
            path = directory / "unknown.csv"
            for columns in (3, 23, 32):
                np.savetxt(path, np.zeros((100, columns)), delimiter="\t")
                with self.assertRaises(ValueError):
                    load_brainflow_csv(path)
            np.savetxt(path, np.zeros((100, 32)), delimiter="\t")
            write_board_metadata(path, boards.SYNTHETIC)
            self.assertEqual(load_brainflow_csv(path).fs, 250)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_brainflow_csv(path, board_id=2)
            board_metadata_path(path).write_text("[]")
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_brainflow_csv(path)

    def test_long_pause_is_missing_data_and_markers_remain_aligned(self):
        with scratch() as directory:
            path = directory / "gap.csv"
            rows = csv_rows(boards.CYTON_DAISY, 10)
            rows[625:, 30] += 2
            rows[750, 31] = 7
            np.savetxt(path, rows, delimiter="\t")
            rec = load_brainflow_csv(path)
            self.assertEqual((rec.duration, rec.gap_samples), (12, 250))
            self.assertTrue(np.isnan(rec.data[:, 625:875]).all())
            np.testing.assert_allclose(rec.data[:, 1000], rows[750, 1:17])
            self.assertEqual(rec.events.labels, ["7"])
            np.testing.assert_allclose(rec.events.onsets, [8])
            self.assertAlmostEqual(rec.marker_seconds(1700000008), 8, delta=1e-5)
            # The missing interval excludes a trial; no silent interpolation.
            result = analyze(rec.data, rec.fs, 0, "C3", [Trial("bad", "A", 4, 8),
                             Trial("good", "A", 8, 12)], Settings(4))
            self.assertIn("bad", result.rejected)
            self.assertEqual([t.trial.id for t in result.trials], ["good"])
            # Unix sidecars use the same repaired clock when no embedded events exist.
            rec.events = None
            path.with_name("gap_events.csv").write_text("1700000008,task\n1700000011,end\n")
            table, _ = epochs.raw_events_for(rec, path)
            self.assertAlmostEqual(table.onsets[0], 8, delta=1e-5)

    def test_short_packet_loss_does_not_require_a_long_clock_pause(self):
        t = 1700000000 + np.arange(500) / 125
        counter = (np.arange(500) * 2) % 256
        keep = np.ones(500, bool)
        keep[200:203] = False
        indices = sample_positions(t[keep], 125, counter[keep], 2)
        self.assertEqual(indices[-1], 499)
        self.assertEqual(indices[200], 203)

    def test_reordered_or_repeated_packets_are_not_lost_samples(self):
        # Counters seen in real Daisy files: 172 172 168 176 ... a step back
        # (mod 256: +252) must not become 126 missing samples.
        t = 1700000000 + np.arange(500) / 125
        counter = ((np.arange(500) * 2) % 256).astype(float)
        counter[[100, 300]] = counter[[99, 299]]          # repeated packet
        counter[[200, 201]] = counter[[201, 200]]         # swapped packets
        np.testing.assert_array_equal(sample_positions(t, 125, counter, 2), np.arange(500))
        counter[400:] = (counter[400:] + 6) % 256          # three packets really lost
        self.assertEqual(sample_positions(t, 125, counter, 2)[-1], 502)

    @staticmethod
    def packet_stamps(n, fs=125, packet=60):
        """Host stamps as Cyton writes them: one arrival time per ~0.5 s packet."""
        idx = np.arange(n)
        return 1700000000 + np.minimum((idx // packet) * packet + packet - 1, n - 1) / fs + .03

    def test_loss_longer_than_half_a_counter_cycle_is_read_from_the_clock(self):
        # 90 lost Daisy samples: the counter steps 2 * 91 = 182, which alone
        # reads as a step back. The clock around it shows ~0.7 s missing.
        n = 125 * 300
        keep = np.ones(n, bool)
        keep[7500:7590] = False
        t, c = self.packet_stamps(n)[keep], ((np.arange(n) * 2) % 256)[keep].astype(float)
        self.assertAlmostEqual(sample_positions(t, 125, c, 2)[-1] + 1 - keep.sum(), 90, delta=3)

    def test_paused_stream_restarting_its_counter_is_measured_by_the_clock(self):
        n = 125 * 120
        t = self.packet_stamps(n)
        c = ((np.arange(n) * 2) % 256).astype(float)
        t[6000:] += 3
        c[6000:] = (np.arange(n - 6000) * 2) % 256          # counter starts again at 0
        self.assertAlmostEqual(sample_positions(t, 125, c, 2)[-1] + 1 - n, 375, delta=3)

    def test_uniform_clock_slightly_off_rate_invents_no_samples(self):
        for true in (249.95, 250.05):
            t = 1700000000 + np.arange(int(600 * true)) / true
            np.testing.assert_array_equal(sample_positions(t, 250), np.arange(t.size))
        t = 1700000000 + np.arange(int(600 * 250.05)) / 250.05
        t[25000:] += 50 / 250
        self.assertEqual(sample_positions(t, 250)[-1] + 1 - t.size, 50)

    def test_blank_row_is_not_counted_twice_by_packet_clock(self):
        t = 1700000000 + np.arange(500) / 125
        counter = ((np.arange(500) * 2) % 256).astype(float)
        counter[200] = np.nan
        t[200] = np.nan
        np.testing.assert_array_equal(sample_positions(t, 125, counter, 2), np.arange(500))

    def test_nearby_pauses_and_short_uniform_clock_loss_are_exact(self):
        t = 1700000000 + np.arange(1250) / 125
        t[200:] += 2
        t[240:] += 3
        indices = sample_positions(t, 125)
        self.assertEqual(indices[-1], 1249 + 625)
        self.assertEqual(indices[200], 450)
        self.assertEqual(indices[240], 865)
        t = 1700000000 + np.arange(1250) / 250
        t[200:] += 3 / 250
        self.assertEqual(sample_positions(t, 250)[-1], 1252)

    def test_usb_bursts_and_temporary_delivery_delay_are_not_missing_samples(self):
        t = 1700000000 + np.floor(np.arange(1250) / 60) * 60 / 125 + np.arange(1250) * 1e-6
        np.testing.assert_array_equal(sample_positions(t, 125), np.arange(1250))
        t = 1700000000 + np.arange(1250) / 125
        t[400:500] = t[500] + np.arange(100) * 1e-7
        t[500] = t[499] + 1e-7
        np.testing.assert_array_equal(sample_positions(t, 125), np.arange(1250))

    def test_bad_clock_cannot_create_an_unbounded_allocation(self):
        t = 1700000000 + np.arange(500) / 125
        t[250:] += 1e8
        with self.assertRaisesRegex(ValueError, "too much missing"):
            sample_positions(t, 125)
        t[250:] -= 2e8
        with self.assertRaisesRegex(ValueError, "backwards"):
            sample_positions(t, 125)

    def test_embedded_pulses_preserve_codes_and_repeated_events(self):
        with scratch() as directory:
            for board in boards.CSV_BOARDS:
                path = directory / (board.key + ".csv")
                rows = csv_rows(board)
                rows[board.fs, board.marker_row] = 1
                rows[board.fs + 1, board.marker_row] = 1
                rows[2 * board.fs, board.marker_row] = 2
                rows[-1, board.marker_row] = np.nan
                np.savetxt(path, rows, delimiter="\t")
                rec = load_brainflow_csv(path)
                table, _ = epochs.raw_events_for(rec, path)
                self.assertEqual(table.labels, ["1", "1", "2"])
                np.testing.assert_allclose(table.onsets, [1, 1 + 1 / board.fs, 2])
                np.testing.assert_array_equal(table.durations, 0)

    def test_packet_arrival_stamps_do_not_move_unix_markers(self):
        # Device really at 125.1 Hz; each 60-sample packet is stamped when it
        # arrives, 50 ms after its last sample. Markers every 30 s of wall time.
        n, true_fs = 125 * 200, 125.1
        acquired = 1700000000 + np.arange(n) / true_fs
        last = np.minimum((np.arange(n) // 60) * 60 + 59, n - 1)
        rec = Recording(np.zeros((1, n)), 125, timestamps=acquired[last] + .05)
        when = 1700000000 + np.arange(10, 190, 30.0)
        truth = (when - 1700000000) * true_fs / 125
        placed = np.array([rec.marker_seconds(w) for w in when])
        # A constant delivery delay is invisible in arrival stamps; packet
        # jitter (up to 0.48 s here) must not reach the markers.
        np.testing.assert_allclose(placed, truth, atol=.05 + 1 / 125)
        np.testing.assert_allclose(np.diff(placed), np.diff(truth), atol=1 / 125)

    def test_labelled_sidecar_names_embedded_marker_codes(self):
        with scratch() as directory:
            path = directory / "rec.csv"
            rows = csv_rows(boards.CYTON_DAISY, 10)
            rows[250, 31], rows[750, 31] = 1, 2
            np.savetxt(path, rows, delimiter="\t")
            rec = load_brainflow_csv(path)
            table, source = epochs.raw_events_for(rec, path)
            self.assertEqual((table.labels, source), (["1", "2"], "marker channel in the file"))
            path.with_name("rec_events.csv").write_text(
                "unix_time,label\n1700000002,open\n1700000006,close\n")
            table, source = epochs.raw_events_for(rec, path)
            self.assertEqual((table.labels, source), (["open", "close"], "rec_events.csv"))
            np.testing.assert_allclose(table.onsets, [2, 6])

    def test_late_delivered_block_does_not_become_a_gap(self):
        # Samples 400-499 arrive 0.8 s late, then catch up: nothing is missing.
        t = 1700000000 + np.arange(1250) / 125
        t[400:500] = t[500] + np.arange(100) * 1e-7
        t[500] = t[499] + 1e-7
        np.testing.assert_array_equal(sample_positions(t, 125), np.arange(1250))
        # The same jump that never catches up is a real 100-sample gap.
        t = 1700000000 + np.arange(1250) / 125
        t[400:] += 100 / 125
        self.assertEqual(sample_positions(t, 125)[-1], 1349)


class NumericRegressions(unittest.TestCase):
    def test_invalid_filters_never_silently_become_another_filter(self):
        for values in ({"low": 40, "high": 10}, {"low": 10, "high": 10},
                       {"low": 63}, {"high": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                design_filter(125, **values)
        sos = design_filter(125, low=1, high=45, notches=(50, 100))
        _, h = sosfreqz(sos, worN=[10], fs=125)
        self.assertGreater(abs(h[0]), .99)

    def test_impedance_missing_values_cannot_return_a_number(self):
        t = np.arange(500) / 125
        signal = 502200 * 6e-9 * 1e6 * np.sin(2 * np.pi * 31.25 * t)
        self.assertAlmostEqual(estimate(0, signal, 125).ohms, 500000, delta=1000)
        for index, value in ((250, np.nan), (0, np.inf), (-1, -np.inf)):
            bad = signal.copy()
            bad[index] = value
            result = estimate(0, bad, 125)
            self.assertEqual(result.status, "Missing samples")
            self.assertTrue(np.isnan(result.ohms))
            self.assertEqual(result.level(), "grey")
        self.assertEqual(estimate(0, signal, 50).status, "No test signal")

    def test_trace_tail_and_true_extrema_times_are_preserved(self):
        data = np.zeros((2, 2500))
        data[0, -1] = 999
        data[1, -2] = -888
        out, indices = decimate_minmax(data, 600, return_indices=True)
        self.assertEqual(out[0].max(), 999)
        self.assertEqual(out[1].min(), -888)
        self.assertEqual(indices[0, out[0].argmax()], 2499)
        self.assertEqual(indices[1, out[1].argmin()], 2498)
        self.assertTrue((np.diff(indices, axis=1) >= 0).all())
        np.testing.assert_array_equal(out, np.take_along_axis(data, indices, axis=1))
        self.assertLessEqual(out.shape[1], 1202)
        # A leftover must not be drawn as one wide bin (the straight "V" at
        # the end of each trace): points stay evenly spaced to the last sample.
        for n, bins in ((2500, 540), (5000, 960), (1251, 625)):
            _, spread = decimate_minmax(np.random.default_rng(0).normal(size=(2, n)), bins, return_indices=True)
            self.assertLessEqual(np.diff(spread, axis=1).max(), 2 * (n // bins + 1))
            self.assertGreaterEqual(spread[:, -1].min(), n - (n // bins + 1))
        for n in (1, 2, 99, 1200):
            small = data[:, :n]
            out, indices = decimate_minmax(small, 600, return_indices=True)
            np.testing.assert_array_equal(out, small)
            np.testing.assert_array_equal(indices[0], np.arange(n))


class UIRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_monitor_invalid_filter_pauses_results_and_recovers(self):
        from eeg.ui.monitor import MonitorPage
        from eeg.ui.playback import Player
        page = MonitorPage()
        page.show()
        try:
            data = np.tile(np.sin(2 * np.pi * 10 * np.arange(2500) / 125), (16, 1))
            page.source = FileSource(Recording(data, 125, page.panel.channel_names))
            page.player = Player(page.source, 125)
            page._tick(5)
            page.panel.filters.low_cut.setValue(40)
            page.panel.filters.high_cut.setValue(10)
            self.app.processEvents()
            self.assertTrue(page.panel.filters.warning.isVisibleTo(page.panel.filters))
            page._tick(1)
            self.assertIsNone(page.stream)
            self.assertEqual(page.filled, 0)
            self.assertEqual(page.feature_buffer.shape[1], 0)
            self.assertEqual(page.raw_buffer.shape[1], 500)
            page.panel.filters.low_cut.setValue(1)
            page._tick(1)
            self.assertIsNotNone(page.stream)
            self.assertGreater(page.filled, 0)
            self.assertTrue(np.isfinite(page.buffer[:, -page.filled:]).all())
        finally:
            page.stop()
            QtCore.QThreadPool.globalInstance().waitForDone(5000)
            self.app.processEvents()
            page.close()

    def test_game_failed_start_cleans_up_and_allows_retry(self):
        from test_games import fake_game
        from eeg.ui.monitor import MonitorPage
        from eeg.ui.games_page import GamesPage
        from eeg.ui.playback import Player
        with scratch() as directory:
            # scratch owns flat files; put the fixture directly in it.
            fake_game(directory)
            monitor = MonitorPage()
            monitor.source = FileSource(Recording(np.zeros((16, 250)), 125, monitor.panel.channel_names))
            monitor.player = Player(monitor.source, 125)
            game = GamesPage(monitor)
            try:
                game.preferences.setValue("folders", [str(directory)])
                game.reload(select=str(directory))
                self.assertIsNotNone(game.current_game())
                with patch("eeg.ui.games_page.game_command", return_value=(str(directory / "missing.exe"), [])):
                    for _ in range(2):
                        game.start_game()
                        deadline = time.monotonic() + 3
                        while game.process is not None and time.monotonic() < deadline:
                            self.app.processEvents()
                            time.sleep(.005)
                        self.assertIsNone(game.process)
                        self.assertIsNone(game.bridge)
                        self.assertTrue(game.start_button.isEnabled())
                        self.assertFalse(game.stop_button.isEnabled())
            finally:
                game.shutdown()
                game.close()
                monitor.stop()
                monitor.close()


if __name__ == "__main__":
    unittest.main()
