"""Lead-off impedance: estimate, sweep order, board commands and monitor flow."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from eeg.core import hardware as hw
from eeg.core import impedance as imp

FS = 125


def lead_off(ohms, seconds=4.0, hz=imp.LEAD_OFF_HZ, noise=10.0, offset=-120_000.0, seed=0):
    """What a channel sees during the test: offset + drift + tone + noise, µV."""
    t = np.arange(round(seconds * FS)) / FS
    peak = imp.DRIVE_AMPS * (ohms + imp.SERIES_OHMS) * 1e6
    rng = np.random.default_rng(seed)
    return offset + 300 * t + peak * np.sin(2 * np.pi * hz * t) + noise * rng.standard_normal(t.size)


class Estimate(unittest.TestCase):
    def test_recovers_impedance_on_and_off_bin(self):
        for ohms, hz in [(300e3, 31.25), (1.2e6, 31.25), (300e3, 31.1)]:
            r = imp.estimate(0, lead_off(ohms, hz=hz), FS)
            self.assertAlmostEqual(r.ohms / ohms, 1, delta=0.03, msg=(ohms, hz))
        # Colour follows the OpenBCI GUI limits; no verbal grade in the text.
        cases = [(300e3, 'green', 'kΩ'), (1.2e6, 'yellow', 'MΩ'), (3e6, 'red', 'MΩ')]
        for ohms, colour, unit in cases:
            r = imp.estimate(0, lead_off(ohms, offset=0), FS)
            self.assertEqual((r.status, r.level()), ('Measured', colour))
            self.assertTrue(r.text.endswith(unit)); self.assertNotIn('(', r.text)
        open_lead = imp.estimate(0, lead_off(5e6, offset=0), FS)
        self.assertEqual((open_lead.status, open_lead.level()), ('Open?', 'red'))
        self.assertTrue(open_lead.text.startswith('Open? (5.0 MΩ'))
        # Limits are the operator's: a stricter green turns 300 kΩ yellow.
        self.assertEqual(imp.estimate(0, lead_off(300e3), FS).level(green=200e3), 'yellow')

    def test_noise_without_test_current_is_not_a_reading(self):
        # The OpenBCI GUI turns channel SD into ohms, so this reads "normal" there.
        rng = np.random.default_rng(1)
        r = imp.estimate(0, -50_000 + 400 * rng.standard_normal(500), FS)
        self.assertEqual(r.status, 'No test signal')
        self.assertTrue(np.isnan(r.ohms))
        self.assertEqual(imp.estimate(0, np.full(500, -4_500_000.0), FS).status, 'Railed')
        # Gain 1 during the test: 300 mV is a live reading, not the gain-24 rail.
        self.assertNotEqual(imp.estimate(0, lead_off(300e3, offset=300_000), FS).status, 'Railed')
        # A powered-down channel sends exact zeros; that is not "0 kΩ, Good".
        self.assertEqual(imp.estimate(0, np.zeros(500), FS).status, 'No test signal')
        dead = [r, imp.estimate(1, np.full(500, 4_500_000.0), FS)]
        self.assertIn('BIAS electrode', imp.summary(dead, ['O1', 'O2']))
        self.assertEqual([x.level() for x in dead], ['grey', 'grey'])
        html, ready = imp.summary_html(dead, ['O1', 'O2'])
        self.assertFalse(ready); self.assertIn('BIAS electrode', html)

    def test_all_green_only_when_every_channel_is_green(self):
        good = [imp.estimate(6, lead_off(400e3, seed=1), FS), imp.estimate(7, lead_off(600e3, seed=2), FS)]
        html, ready = imp.summary_html(good, ['x'] * 6 + ['O1', 'O2'])
        self.assertTrue(ready); self.assertIn('All green', html)
        mixed = good[:1] + [imp.estimate(7, lead_off(1.5e6), FS)]
        html, ready = imp.summary_html(mixed, ['x'] * 6 + ['O1', 'O2'])
        self.assertFalse(ready); self.assertIn(imp.COLOURS['yellow'], html)

    def test_commands_match_the_gui_log(self):
        # From Console_2026-09-14_15-57-11: "x7000100Xz701Z" ... "z700Z"
        self.assertEqual(imp.test_command(6), 'x7000100Xz701Z')
        self.assertEqual(imp.test_command(15), 'xI000100XzI01Z')
        self.assertEqual(imp.stop_command(6), 'z700Z')

    def test_sweep_drops_settling_samples_and_walks_channels(self):
        sweep = imp.Sweep([6, 7], FS)
        good = np.zeros((16, round(5.5 * FS)))
        good[6, :round(1.5 * FS)] = 1e5                        # settling junk, ignored
        good[6, round(1.5 * FS):] = lead_off(300e3)
        actions = [sweep.feed(good[:, i:i + 6]) for i in range(0, good.shape[1], 6)]
        self.assertEqual([a for a in actions if a], [('switch', 6, 7)])
        second = np.zeros((16, round(5.5 * FS))); second[7, round(1.5 * FS):] = lead_off(2e6)
        actions = [sweep.feed(second[:, i:i + 6]) for i in range(0, second.shape[1], 6)]
        self.assertEqual([a for a in actions if a], [('done', 7, None)])
        self.assertTrue(sweep.finished)
        self.assertEqual([r.level() for r in sweep.results], ['green', 'yellow'])


class Board(unittest.TestCase):
    def test_step_restores_operator_setting_not_board_default(self):
        from eeg.core.sources import LiveSource
        board = Mock(); board.config_board.return_value = 'Success$$$'
        only_o = [hw.ChannelSetting(i in (6, 7), False) for i in range(16)]
        with patch('brainflow.board_shim.BoardShim', return_value=board):
            source = LiveSource('COM7', only_o)
        board.reset_mock()
        source.impedance_step(6, 7, only_o)
        calls = board.method_calls
        self.assertEqual(calls[0][0], 'stop_stream'); self.assertEqual(calls[-1][0], 'start_stream')
        sent = [c[1][0] for c in calls if c[0] == 'config_board']
        self.assertEqual(sent, ['z700Z', 'x7060010X', 'x8000100Xz801Z'])
        self.assertNotIn('d', sent)
        board.reset_mock(); board.config_board.return_value = 'Failure: nope$$$'
        with self.assertRaises(RuntimeError):
            source.impedance_step(None, 6, only_o)
        self.assertEqual(board.method_calls[-1][0], 'start_stream')

    def test_internal_short_restores_every_operator_setting(self):
        from eeg.core.sources import LiveSource
        from eeg.core import shorted_test
        board = Mock(); board.config_board.return_value = 'Success$$$'
        settings = [hw.ChannelSetting(i in (6, 7), False) for i in range(16)]
        with patch('brainflow.board_shim.BoardShim', return_value=board):
            source = LiveSource('COM7', settings)
        board.reset_mock()
        source.internal_short(True, settings)
        sent = [c[1][0] for c in board.method_calls if c[0] == 'config_board']
        self.assertEqual(sent, [shorted_test.channel_command(i) for i in range(16)])
        board.reset_mock()
        source.internal_short(False, settings)
        sent = [c[1][0] for c in board.method_calls if c[0] == 'config_board']
        self.assertEqual(sent, [hw.channel_command(i, setting) for i, setting in enumerate(settings)])


class InternalShort(unittest.TestCase):
    def test_capture_reports_small_noise_without_a_large_50_hz_peak(self):
        from eeg.core import shorted_test
        rng = np.random.default_rng(7)
        capture = shorted_test.Capture([0, 1], FS, settle=0, measure=20)
        t = np.arange(20 * FS) / FS
        data = np.vstack((0.25 * rng.standard_normal(t.size),
                          2.0 + 0.25 * rng.standard_normal(t.size)))
        self.assertTrue(capture.feed(data))
        self.assertEqual([r.status for r in capture.results], ['Clean', 'Clean'])
        self.assertTrue(all(r.noise_uv < 1 for r in capture.results))

    def test_large_internal_noise_or_mains_is_flagged(self):
        from eeg.core import shorted_test
        t = np.arange(20 * FS) / FS
        data = np.vstack((10 * np.sin(2 * np.pi * 50 * t), np.full(t.size, 100.0)))
        results = shorted_test.analyse(data, FS)
        self.assertEqual([r.status for r in results], ['Check', 'Check'])


class MonitorFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_check_runs_on_on_channels_keeps_display_clean_and_reports(self):
        from eeg.ui.monitor import MonitorPage
        page = MonitorPage(); page.panel.mode.setCurrentIndex(1)
        self.addCleanup(page.close)
        page.panel.hardware.only(['O1', 'O2']); page.panel.hardware.no_bias()
        tones = {6: lead_off(400e3, seconds=6), 7: lead_off(3e6, seconds=6, offset=0)}
        state = dict(channel=None, pos=0)

        def step(previous, following, hardware):
            state['channel'], state['pos'] = following, 0

        def read(n):
            chunk = np.zeros((16, 6))
            c = state['channel']
            if c is not None:
                seg = np.roll(tones[c], 0)[round(state['pos']):round(state['pos']) + 6]
                chunk[c, :seg.size] = seg
                state['pos'] += 6
            return chunk

        import threading
        gui = threading.get_ident()
        step_threads = []

        def step(previous, following, hardware):
            step_threads.append(threading.get_ident())
            state['channel'], state['pos'] = following, 0

        def flush():
            page.board_pool.waitForDone(5000)             # board commands run in the Monitor's own pool
            QtCore.QThreadPool.globalInstance().waitForDone(5000)
            self.app.processEvents()

        source = Mock(fs=125); source.read.side_effect = read
        source.impedance_step.side_effect = step
        with patch('eeg.core.sources.LiveSource', return_value=source):
            page.start(); flush()
            page.start_impedance()
            self.assertFalse(page.panel.hardware.impedance_button.isEnabled())
            for _ in range(400):
                page._tick(); flush()
                if page.impedance_sweep is None and not page._configuring:
                    break
                self.assertEqual(page.raw_buffer.shape[1], 0)   # test current never displayed
            self.assertNotIn(gui, step_threads)                 # board commands never on the GUI thread
            steps = [c[0][:2] for c in source.impedance_step.call_args_list]
            self.assertEqual(steps, [(None, 6), (6, 7), (7, None)])
            text = page.panel.hardware.impedance.text()
            self.assertIn('O1', text); self.assertIn('kΩ', text); self.assertIn('O2', text)
            self.assertIn(imp.COLOURS['green'], text)     # O1 400 kΩ
            self.assertIn(imp.COLOURS['red'], text)       # O2 3 MΩ
            self.assertFalse(page.impedance_ready)
            page.panel.hardware.yellow_limit.setValue(4000)   # operator relaxes the limit
            self.assertIn(imp.COLOURS['yellow'], page.panel.hardware.impedance.text())
            self.assertTrue(page.panel.hardware.impedance_button.isEnabled())
            source.log_note.assert_called_once()
            page.stop()
        page.start_impedance()   # not running: refuses without touching a board
        self.assertIsNone(page.impedance_sweep)


if __name__ == '__main__':
    unittest.main()
