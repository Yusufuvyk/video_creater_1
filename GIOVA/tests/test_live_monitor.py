"""Festival operator monitor: measurements, failure isolation and lifecycle."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import patch, Mock
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from eeg.core.monitoring import QualityLimits, QualityTracker, quality_snapshot, spectra_snapshot
from eeg.core.sources import Recording, FileSource
from eeg.ui.playback import Player
from eeg.ui.monitor import MonitorPage
from eeg.ui.live_views import SpectrumPanel


def sine(hz=10, amplitude=10, seconds=4, fs=125):
    return amplitude * np.sin(2 * np.pi * hz * np.arange(round(seconds * fs)) / fs)


class Measurements(unittest.TestCase):
    def test_quality_clean_flat_saturated_missing_and_mains(self):
        # Row 4's EEG differs from row 0's: two identical rows are one node.
        data = np.stack([sine(), np.ones(500), np.full(500, 187500),
                         np.full(500, np.nan), sine(50, 30) + sine(11)])
        result = quality_snapshot(data, 125, 4, 187500)
        self.assertEqual([r.status for r in result],
                         ['No issues detected', 'Flat', 'Saturated', 'Missing data', 'Line noise'])
        self.assertAlmostEqual(result[0].rms, 10 / np.sqrt(2), places=2)
        self.assertGreater(result[4].mains, .8)
        # Mains and drift are outside the EEG band: they do not inflate the RMS.
        self.assertAlmostEqual(result[4].rms, 10 / np.sqrt(2), delta=.3)
        drift = quality_snapshot((sine() + 5000 * np.linspace(0, 1, 500))[None], 125, 4, 187500)
        self.assertLess(drift[0].rms, 100)
        big = quality_snapshot(sine(10, 300)[None], 125, 4, 187500)
        self.assertEqual(big[0].status, 'Check signal')
        self.assertTrue(np.isnan(quality_snapshot(data[:1], 125, 4)[0].saturation))
        self.assertEqual(quality_snapshot(data[:, :100], 125, .8)[0].status, 'Collecting data')

    def test_identical_channels_are_one_node_not_eeg(self):
        rng = np.random.default_rng(0)
        eeg = lambda: rng.normal(0, 8, 500)
        floating = 400000 + rng.normal(0, 10, 500)              # inputs off the head, below the rail
        mains = sine(50, 400)                                    # large, shared pick-up
        data = np.stack([floating, floating, floating + 1e-3, eeg(), eeg() + mains, eeg() + mains,
                         floating - 5000])                     # same waveform, 5 mV apart
        result = quality_snapshot(data, 125, 4, 750000)
        self.assertEqual([r.status for r in result[:3]], ['Identical signal'] * 3)
        self.assertEqual({r.same_as for r in result[:3]} <= {0, 1, 2}, True)
        self.assertEqual(result[3].status, 'No issues detected')
        # Shared mains is outside the EEG band: those two are still different EEG.
        self.assertNotEqual(result[4].status, 'Identical signal')
        self.assertNotEqual(result[5].status, 'Identical signal')
        self.assertNotEqual(result[6].status, 'Identical signal')
        self.assertEqual(result[3].same_as, -1)
        # A channel already railed or too large keeps that verdict.
        railed = quality_snapshot(np.stack([np.full(500, 750000.0)] * 2), 125, 4, 750000)
        self.assertEqual([r.status for r in railed], ['Saturated'] * 2)
        # The 18 Sept Unicorn: loose inputs float at ~714 mV, inside 5% of the
        # 750 mV rail, so they already read as Saturated.
        unicorn = quality_snapshot(np.stack([714000 + rng.normal(0, 10, 1000)] * 4), 250, 4, 750000)
        self.assertEqual({r.status for r in unicorn}, {'Saturated'})

    def test_debounce_immediate_faults_and_three_step_recovery(self):
        tracker = QualityTracker()
        clean = quality_snapshot(sine()[None], 125, 4, 187500)
        flat = quality_snapshot(np.zeros((1, 500)), 125, 4, 187500)
        rail = quality_snapshot(np.full((1, 500), 187500), 125, 4, 187500)
        self.assertEqual(tracker.update(clean)[0].status, 'No issues detected')
        self.assertEqual(tracker.update(flat)[0].status, 'No issues detected')
        self.assertEqual(tracker.update(flat)[0].status, 'Flat')
        for _ in range(2):
            self.assertEqual(tracker.update(clean)[0].status, 'Flat')
        self.assertEqual(tracker.update(clean)[0].status, 'No issues detected')
        self.assertEqual(tracker.update(rail)[0].status, 'Saturated')

    def test_fft_amplitude_psd_integral_and_bad_channel_isolation(self):
        data = np.stack([sine(amplitude=8), sine(20, 4), np.full(500, np.nan)])
        result = spectra_snapshot(data, 125)
        for row, hz, amplitude in [(0, 10, 8), (1, 20, 4)]:
            peak = np.argmax(result['fft'][row])
            self.assertEqual(result['fft_freqs'][peak], hz)
            self.assertAlmostEqual(result['fft'][row, peak], amplitude, places=3)
            self.assertAlmostEqual(np.trapezoid(result['psd'][row], result['psd_freqs']), amplitude**2 / 2, places=3)
        self.assertTrue(np.isnan(result['fft'][2]).all())
        self.assertTrue(np.isfinite(result['fft'][:2]).all())

    def test_line_noise_is_measured_at_the_mains_chosen(self):
        # 60 Hz hum (the Americas, Korea, Japan) is line noise when the mains is 60 Hz, and only then.
        hum = (sine(60, 30) + sine(11))[None]
        self.assertEqual(quality_snapshot(hum, 125, 4, 187500, mains=60.0)[0].status, 'Line noise')
        self.assertGreater(quality_snapshot(hum, 125, 4, 187500, mains=60.0)[0].mains, .8)
        self.assertEqual(quality_snapshot(hum, 125, 4, 187500)[0].status, 'No issues detected')
        from eeg.core import shorted_test
        shorted = np.vstack([10 * sine(60, 1, 20), np.zeros(2500)]) + np.random.default_rng(1).normal(0, .1, 2500)
        results = shorted_test.analyse(shorted, 125, mains=60.0)
        self.assertEqual([r.status for r in results], ['Check', 'Clean'])
        self.assertIn('60 Hz', shorted_test.summary(results, ['O1', 'O2']))

    def test_low_rate_mains_unavailable_and_adjustable_thresholds(self):
        result = quality_snapshot(sine(fs=80)[None], 80, 4, limits=QualityLimits(high_uv=1))
        self.assertTrue(np.isnan(result[0].mains))
        self.assertEqual(result[0].status, 'Check signal')


class LiveUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.page = MonitorPage()
        self.page.resize(1366, 730); self.page.show(); self.app.processEvents()
        self.page.multi_spectrum.mode.setCurrentText('FFT')
        self.page.multi_spectrum.input.setCurrentText('Filtered')
        self.page.multi_spectrum.arrangement.setCurrentText('Grid')
        self.page.view_tabs.setCurrentIndex(0)
        data = np.tile(sine(seconds=12), (16, 1))
        self.page.source = FileSource(Recording(data, 125, self.page.panel.channel_names))
        self.page.player = Player(self.page.source, 125)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.page.stop(); self.page.detail.close(); self.page.close()
        QtCore.QThreadPool.globalInstance().waitForDone(5000)
        self.app.processEvents()

    def flush(self):
        self.page.board_pool.waitForDone(5000)            # connecting and board commands
        QtCore.QThreadPool.globalInstance().waitForDone(5000)
        self.app.processEvents()

    def test_live_cyton_mode_uses_only_eight_physical_channels(self):
        from eeg.core.boards import CYTON
        page = self.page
        index = page.panel.mode.findText("Live (Cyton only - 8 channels)")
        self.assertGreater(index, 0)
        page.panel.mode.setCurrentIndex(index)
        self.assertIs(page.panel.board, CYTON)
        self.assertEqual(page.panel.hardware.active_count, 8)
        self.assertEqual(len(page.panel.hardware.settings()), 8)

    def test_raw_quality_survives_filter_changes_and_hidden_channels(self):
        page = self.page
        # Independent noise per channel: sixteen identical rows are one node.
        raw = np.tile(sine(50, 30) + sine(), (16, 1)) + np.random.default_rng(0).normal(0, 2, (16, 500))
        page._ingest(raw)
        before = page.raw_buffer.copy()
        page._rebuild_stream()
        np.testing.assert_array_equal(page.raw_buffer, before)
        self.assertEqual(page.spectral_buffer.shape[1], 0)
        page.panel.channel_button.set_selected([])
        for now in (100, 101):
            page.sample_position += 1
            with patch('eeg.ui.monitor.time.perf_counter', return_value=now):
                page._render()
        self.assertEqual(page.quality_table.rowCount(), 16)
        self.assertEqual(page.quality_table.item(15, 1).text(), 'Line noise')
        self.assertEqual(page.panel.selected_channels, [])

    def test_seek_pause_and_revision_discard(self):
        page = self.page
        page._tick(5)
        self.assertLessEqual(len(page.spectral_jobs.active), 1)
        page.player.pause()
        page._seek(0)
        self.flush()
        self.assertIsNone(page._latest_snapshot)
        self.assertEqual(page.raw_buffer.shape[1], 0)
        self.assertEqual(page.quality_table.item(0, 1).text(), 'Collecting data')
        page._render()
        self.assertIn('Paused', page.connection_label.text())

    def test_stale_live_data_and_recovery_collects_new_window(self):
        page = self.page; page.player = None
        with patch('eeg.ui.monitor.time.perf_counter', return_value=0):
            page._ingest(np.tile(sine(), (16, 1)))
        with patch('eeg.ui.monitor.time.perf_counter', return_value=3):
            page._render()
            self.assertIn('Stale', page.connection_label.text())
            self.assertEqual(page.quality_table.item(0, 1).text(), 'Missing data')
            page._ingest(np.tile(sine(seconds=.1), (16, 1)))
            page._render()
        self.assertLess(page.raw_buffer.shape[1], 500)
        self.assertEqual(page.quality_table.item(0, 1).text(), 'Collecting data')

    def test_grid_overlay_scale_smoothing_and_invalid_row(self):
        view = self.page.multi_spectrum
        view.configure(['A', 'B'], [0, 1], 125)
        first = spectra_snapshot(np.stack([sine(), sine(20)]), 125)
        view.update_snapshot(first, 0)
        bounds = view._bounds
        lowered = spectra_snapshot(np.stack([sine(amplitude=1), sine(20, 1)]), 125)
        view.update_snapshot(lowered, .5)
        self.assertEqual(view._bounds[1], bounds[1])
        self.assertGreater(np.max(view._values[0]), np.max(lowered['fft'][0]))
        shown = view._values.copy()
        view.arrangement.setCurrentText('Overlay')
        np.testing.assert_array_equal(view._values, shown)
        self.assertIs(view._plots[0], view._plots[1])
        broken = spectra_snapshot(np.stack([sine(), np.full(500, np.nan)]), 125)
        view.update_snapshot(broken, 1)
        self.assertIsNone(view._curves[1].getData()[0])
        self.assertIsNotNone(view._curves[0].getData()[0])

    def test_scale_recovers_after_an_artefact_without_pressing_fit(self):
        view = self.page.multi_spectrum
        view.configure(['A', 'B'], [0, 1], 125)
        view.log.setChecked(False)
        loud = spectra_snapshot(np.stack([sine(amplitude=3000), sine(20, 3000)]), 125)
        quiet = spectra_snapshot(np.stack([sine(amplitude=3), sine(20, 3)]), 125)
        view.update_snapshot(loud, 0)
        tall = view._bounds[1]
        # Quiet for a moment, then loud again: the scale stays put.
        view.update_snapshot(quiet, 20)
        view.update_snapshot(quiet, 21)
        view.update_snapshot(loud, 22)
        self.assertEqual(view._bounds[1], tall)
        # Quiet for longer than SCALE_SETTLE_S: the scale fits the data again.
        for now in np.arange(40, 52, .5):          # values are smoothed: ~5-6 s in all
            view.update_snapshot(quiet, now)
        self.assertLess(view._bounds[1], tall / 100)
        shown = view._values[0][(view._freqs >= 1) & (view._freqs <= view.maximum.value())]
        self.assertGreaterEqual(view._bounds[1], shown.max())     # nothing is clipped

    def test_mains_lines_do_not_set_the_automatic_scale(self):
        # 18 Sept Fz: 166 µV²/Hz left at 50 Hz by the notch, a 6.7 µV²/Hz SSVEP
        # line at 15 Hz, EEG up to ~9 µV²/Hz.
        freqs = np.arange(0, 125.5, .5)
        psd = np.ones((1, freqs.size))
        psd[0, freqs == 15] = 6.7; psd[0, freqs == 9.5] = 9; psd[0, freqs == 50] = 166
        snapshot = dict(psd=psd, psd_freqs=freqs, fft=psd, fft_freqs=freqs)
        view = self.page.multi_spectrum
        view.configure(['Fz'], [0], 250)
        view.mode.setCurrentText('PSD'); view.log.setChecked(False)
        view.input.setCurrentText('Filtered')
        view.set_scale_exclusions([50.0, 100.0])
        view.update_snapshot(snapshot, 0)
        self.assertLess(view._bounds[1], 20)                  # 15 Hz is a third of the plot
        view.input.setCurrentText('Raw')                      # still the EEG that is measured
        view.update_snapshot(snapshot, 1)
        self.assertLess(view._bounds[1], 20)
        view.scale.setCurrentIndex(view.scale.findData('peak'))
        view.update_snapshot(snapshot, 2)
        self.assertGreaterEqual(view._bounds[1], 166)         # Peak fits the tallest line
        view.scale.setCurrentIndex(view.scale.findData('manual'))
        view.scale_top.setValue(42)
        view.update_snapshot(snapshot, 3)
        self.assertEqual(view._bounds, (0, 42))
        view.scale.setCurrentIndex(view.scale.findData('auto'))
        # The monitor passes its notch settings on.
        self.page.panel.filters.changed.emit()
        self.assertEqual(view._scale_exclusions,
                         tuple(float(f) for f in self.page.panel.filter_kwargs['notches']))

    def test_the_alias_of_the_mains_is_ignored_too(self):
        from eeg.core.spectral import line_frequencies
        self.assertEqual(line_frequencies(125), (25.0, 50.0))    # 100 and 150 Hz fold onto 25 Hz
        self.assertEqual(line_frequencies(250), (50.0, 100.0))
        freqs = np.arange(0, 62.75, .25)
        psd = np.ones((1, freqs.size))
        psd[0, freqs == 15] = 5; psd[0, freqs == 25] = 300       # the alias at 125 Hz
        snapshot = dict(psd=psd, psd_freqs=freqs, fft=psd, fft_freqs=freqs)
        view = self.page.multi_spectrum
        view.configure(['O1'], [0], 125)
        view.mode.setCurrentText('PSD'); view.log.setChecked(False)
        view.update_snapshot(snapshot, 0)
        self.assertLess(view._bounds[1], 20)

    def test_snr_mode_shows_a_flicker_line_as_a_ratio(self):
        from eeg.ui.live_views import neighbour_snr
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 5, (2, 1000))
        flicker = noise + np.stack([sine(15, 3, 8), np.zeros(1000)])
        snapshot = spectra_snapshot(flicker, 125)
        f, p = snapshot['psd_freqs'], snapshot['psd']
        snr = neighbour_snr(f, p)
        k15 = np.argmin(abs(f - 15))
        self.assertGreater(snr[0, k15], 10)                      # the flickering channel
        self.assertLess(snr[1, k15], 3)                          # noise only
        self.assertAlmostEqual(float(np.nanmedian(snr[1, (f > 5) & (f < 40)])), 1, delta=.35)
        view = self.page.multi_spectrum
        view.configure(['A', 'B'], [0, 1], 125)
        view.mode.setCurrentText('SNR'); view.log.setChecked(False)
        view.update_snapshot(snapshot, 0)
        self.assertEqual(view._axes_key[1], 'SNR (×)')
        x, y = view._curves[0].getData()
        self.assertAlmostEqual(x[np.argmax(y)], 15)

    def test_only_the_chosen_band_is_drawn_and_kept(self):
        view = self.page.multi_spectrum
        view.configure(['A', 'B'], [0, 1], 125)
        view.arrangement.setCurrentText('Grid')
        view.minimum.setValue(12); view.maximum.setValue(18)
        view.update_snapshot(spectra_snapshot(np.stack([sine(), sine()]), 125), 0)
        self.assertEqual(view.frequency_range(), (12.0, 18.0))
        x, _ = view._curves[0].getData()
        self.assertGreaterEqual(x.min(), 12); self.assertLessEqual(x.max(), 18)
        self.assertEqual(tuple(round(v, 1) for v in view._plots[0].vb.viewRange()[0]), (12.0, 18.0))
        view.maximum.setValue(200)                            # never beyond what fs can show
        self.assertEqual(view.frequency_range()[1], 62.5)
        # Kept with the Monitor's other spectrum settings. The settings file is
        # shared by every test: put it back as it was.
        prefs = self.page.preferences
        before = {key: prefs.value(key) for key in prefs.allKeys()}
        try:
            view.minimum.setValue(8); view.maximum.setValue(30)
            view.scale.setCurrentIndex(view.scale.findData('peak'))
            self.page.save_preferences()
            self.assertEqual(prefs.value('spectrum_min', type=float), 8)
            self.assertEqual(prefs.value('spectrum_max', type=float), 30)
            self.assertEqual(prefs.value('spectrum_scale'), 'peak')
        finally:
            prefs.clear()
            for key, value in before.items():
                prefs.setValue(key, value)
            prefs.sync()

    def test_curves_glide_to_new_spectra_instead_of_jumping(self):
        view = self.page.multi_spectrum
        view.configure(['A'], [0], 125)
        view.mode.setCurrentText('PSD'); view.log.setChecked(False)
        low = spectra_snapshot(sine(10, 2)[None], 125)
        high = spectra_snapshot(sine(10, 20)[None], 125)
        view.update_snapshot(low, 0)
        k = np.argmin(abs(view._freqs - 10))
        start = view._curves[0].getData()[1][np.argmin(abs(view._curves[0].getData()[0] - 10))]
        view._values = high['psd'].copy()                         # a new target arrives
        view.redraw()
        drawn = lambda: view._curves[0].getData()[1][np.argmin(abs(view._curves[0].getData()[0] - 10))]
        self.assertAlmostEqual(drawn(), start)                    # not drawn at once
        view.animate(1.0); first = drawn()
        view.animate(1.2); second = drawn()
        self.assertTrue(start < first < second < high['psd'][0, k])
        for now in np.arange(1.4, 4, .2):
            view.animate(now)
        self.assertAlmostEqual(drawn(), high['psd'][0, k], delta=.01 * high['psd'][0, k])

    def test_layout_and_channel_detail(self):
        page = self.page
        self.assertFalse(page._wide)
        page.resize(1920, 1080); self.app.processEvents()
        self.assertTrue(page._wide)
        page._tick(5); self.flush()
        page._show_channel(7)
        self.assertTrue(page.detail.isVisible())
        self.assertEqual(page.panel.psd_channel.currentIndex(), 7)
        page.resize(1366, 768); self.app.processEvents()
        self.assertFalse(page._wide)
        self.assertEqual(page.view_tabs.count(), 2)

    def test_failed_connection_unlocks_controls(self):
        page = self.page; page.stop(); page.panel.mode.setCurrentIndex(1)
        with patch('eeg.core.sources.LiveSource', side_effect=RuntimeError('port busy')), \
             patch('eeg.ui.monitor.QtWidgets.QMessageBox.critical'), \
             patch('eeg.ui.monitor.traceback.print_exc'):
            page.start()
            self.assertTrue(page._starting)   # Start returned while the board opens
            self.flush()
        self.assertIsNone(page.source)
        self.assertTrue(page.panel.start_button.isEnabled())
        self.assertTrue(page.panel.mode.isEnabled())
        self.assertFalse(page.timer.isActive())

    def test_spectrum_two_hz_pause_and_bounded_pending_work(self):
        page = self.page
        page.panel.window_seconds.setValue(4)            # spectra from the 4 s just fed
        page._ingest(np.tile(sine(), (16, 1)))
        with patch.object(page.spectral_jobs, 'start') as start:
            for i in range(20):
                page.sample_position += 6
                with patch('eeg.ui.monitor.time.perf_counter', return_value=i * .05):
                    page._render()
            self.assertEqual(start.call_count, 2)
            with patch('eeg.ui.monitor.time.perf_counter', return_value=2):
                page._render()
                count = start.call_count
                for _ in range(10):
                    page._render()
                self.assertEqual(start.call_count, count)
            page.spectral_jobs.active[999] = object()
            page.sample_position += 6
            with patch('eeg.ui.monitor.time.perf_counter', return_value=3):
                page._render()
            self.assertEqual(start.call_count, count)
            page.spectral_jobs.active.clear()

    def test_successful_start_stop_and_reconnect_locks_source(self):
        page = self.page; page.stop(); page.panel.mode.setCurrentIndex(1)
        source = Mock(fs=125)
        with patch('eeg.core.sources.LiveSource', return_value=source) as live:
            page.start(); self.flush()
            from eeg.ui.preferences import data_dir
            self.assertEqual(live.call_args.kwargs['record_dir'], data_dir())  # every live start is recorded
            self.assertEqual(live.call_args.kwargs['record_prefix'], 'live')
            self.assertTrue(page.timer.isActive())
            self.assertFalse(page.panel.port.isEnabled())
            page._ingest(np.tile(sine(), (16, 1)))
            page.stop(); page.start(); self.flush()
            self.assertEqual(page.raw_buffer.shape[1], 0)
            self.assertTrue(page.timer.isActive())
        page.stop()
        self.assertTrue(page.panel.port.isEnabled())

    def test_more_than_sixteen_channels_scroll_without_losing_quality(self):
        page = self.page
        page.stop(); page.n_channels = 64
        names = [f'EEG{i+1}' for i in range(64)]
        page.panel.set_channels(names)
        page.panel.channel_button.set_selected(range(64))
        page._reset_live_data(); page._resize_buffer()
        page.multi_spectrum.configure(names, list(range(64)), 125)
        self.assertEqual(page.quality_table.rowCount(), 64)
        self.assertEqual(len(page.multi_spectrum._curves), 64)
        self.assertGreater(page.multi_spectrum.graphics.minimumHeight(), 1000)

    def test_live_source_releases_session_if_start_fails(self):
        from eeg.core.sources import LiveSource
        board = Mock(); board.start_stream.side_effect = RuntimeError('failed')
        board.stop_stream.side_effect = RuntimeError('not running')
        with patch('brainflow.board_shim.BoardShim', return_value=board):
            with self.assertRaises(RuntimeError):
                LiveSource('COM7')
        board.release_session.assert_called_once()

    def test_spectrum_channel_filter_is_display_only(self):
        panel = SpectrumPanel()
        names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        panel.configure(names, range(8), 125)
        panel.arrangement.setCurrentText('Overlay')
        panel.show_only(['O1', 'O2'])
        self.assertEqual(panel._channels, [6, 7])
        self.assertEqual(len(panel._curves), 2)
        self.assertEqual(panel.channel_menu_button.text(), 'Channels 2/8')
        values = np.ones((8, 20)); values[0] = 1e6   # a loud hidden channel
        panel.update_snapshot(dict(fft=values, fft_freqs=np.arange(20.0), psd=values,
                                   psd_freqs=np.arange(20.0), valid=np.ones(8, bool)), 0)
        self.assertLess(panel._bounds[1], 3)          # scale ignores hidden channels
        panel._fill_channel_menu()
        checks = panel._channel_menu.actions()[0].defaultWidget().findChildren(QtWidgets.QCheckBox)
        self.assertEqual([c.isChecked() for c in checks], [False] * 6 + [True, True])
        checks[0].setChecked(True)
        self.assertEqual(panel._channels, [0, 6, 7])
        panel.configure(names, [0, 1], 125)            # Graphs selection still applies
        self.assertEqual(panel._channels, [0])
        panel.set_hidden([])
        self.assertEqual(panel._channels, [0, 1])

    def test_channel_commands_follow_cyton_sdk(self):
        from eeg.core import hardware as hw
        from eeg.core import shorted_test
        self.assertEqual(hw.channel_command(0, hw.ChannelSetting()), 'x1060110X')   # board default
        self.assertEqual(hw.channel_command(6, hw.ChannelSetting(True, False)), 'x7060010X')
        self.assertEqual(hw.channel_command(15, hw.ChannelSetting(False, True)), 'xI161000X')
        self.assertEqual(shorted_test.channel_command(0), 'x1061000X')
        self.assertEqual(shorted_test.channel_command(15), 'xI061000X')
        settings = [hw.ChannelSetting(i in (6, 7), False) for i in range(16)]
        self.assertEqual(hw.decode(hw.encode(settings)), settings)
        self.assertEqual(hw.decode('garbage'), hw.default_settings())

    def test_live_source_applies_hardware_after_reset_before_stream(self):
        from eeg.core import hardware as hw
        from eeg.core.sources import LiveSource
        board = Mock(); board.config_board.return_value = 'Success: Channel set$$$'
        settings = [hw.ChannelSetting(i < 8, False) for i in range(16)]
        with patch('brainflow.board_shim.BoardShim', return_value=board):
            source = LiveSource('COM7', settings)
        calls = [c[0] for c in board.method_calls]
        self.assertEqual(calls[0], 'prepare_session')
        self.assertEqual(calls[-1], 'start_stream')
        self.assertEqual(calls.count('config_board'), 16)
        # Leftover stream bytes (0xA0 packet header) as the first reply: resend.
        board.reset_mock()
        stale = UnicodeDecodeError('utf-8', b'\xa0', 0, 1, 'invalid start byte')
        board.config_board.side_effect = [stale] + ['Success$$$'] * 16
        source.configure(settings)
        self.assertEqual(board.config_board.call_count, 17)
        self.assertEqual(board.config_board.call_args_list[0], board.config_board.call_args_list[1])
        board.reset_mock(); board.config_board.side_effect = None
        board.config_board.return_value = 'Failure: bad$$$'
        with self.assertRaises(RuntimeError):
            source.configure(settings)
        self.assertEqual([c[0] for c in board.method_calls][-1], 'start_stream')  # stream resumes

    def test_cyton_only_resets_and_standbys_only_the_daisy_before_channels(self):
        from eeg.core import hardware as hw
        from eeg.core.boards import CYTON
        from eeg.core.sources import LiveSource
        board = Mock(); board.config_board.return_value = 'daisy removed$$$'
        settings = [hw.ChannelSetting(True, False) for _ in range(8)]
        with patch('brainflow.board_shim.BoardShim', return_value=board):
            LiveSource('COM7', settings, board=CYTON)
        commands = [call.args[0] for call in board.config_board.call_args_list]
        self.assertEqual(commands[0], 'c')
        self.assertEqual(commands[1:], [hw.channel_command(i, settings[i]) for i in range(8)])

    def test_live_source_records_raw_stream_and_hardware_log(self):
        import tempfile
        from pathlib import Path
        from eeg.core import hardware as hw
        from eeg.core.sources import LiveSource, hardware_log_path, load_brainflow_csv
        board = Mock(); board.config_board.return_value = 'Success$$$'
        only_o = [hw.ChannelSetting(i in (6, 7), False) for i in range(16)]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('brainflow.board_shim.BoardShim', return_value=board):
            source = LiveSource('COM7', only_o, record_dir=tmp)
            calls = [c[0] for c in board.method_calls]
            # Streamer attached after the settings and before the stream starts.
            self.assertLess(calls.index('config_board'), calls.index('add_streamer'))
            self.assertLess(calls.index('add_streamer'), calls.index('start_stream'))
            target = board.add_streamer.call_args[0][0]
            self.assertEqual(target, 'file://{}:w'.format(source.record_path))
            self.assertEqual(source.record_path.parent, Path(tmp))
            source.configure(hw.default_settings())
            log = hardware_log_path(source.record_path).read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(log), 3)
            self.assertIn(hw.encode(only_o), log[1])
            self.assertIn('BIAS from: none', log[1])
            self.assertIn('applied', log[2])
            # A second session in the same second gets its own files.
            self.assertNotEqual(LiveSource('COM7', only_o, record_dir=tmp).record_path, source.record_path)
            # The streamed layout is what the playback loader reads.
            row = np.zeros(32); row[1:17] = np.arange(16); row[30] = 1.7e9
            np.savetxt(source.record_path, np.tile(row, (250, 1)), delimiter='\t')
            rec = load_brainflow_csv(source.record_path)
            self.assertEqual(rec.data.shape, (16, 250))
            self.assertEqual(rec.data[7, 0], 7)
        board.reset_mock()
        with patch('brainflow.board_shim.BoardShim', return_value=board):
            self.assertIsNone(LiveSource('COM7').record_path)
        self.assertNotIn('add_streamer', [c[0] for c in board.method_calls])

    def test_apply_button_reconfigures_live_board_and_selects_on_channels(self):
        page = self.page; page.stop(); page.panel.mode.setCurrentIndex(1)
        source = Mock(fs=125)
        import threading
        gui_thread = threading.get_ident()
        release = threading.Event()
        calls = []

        def slow_configure(settings, previous=None):
            calls.append((threading.get_ident(), settings, previous))
            release.wait(5)   # the firmware taking its time to answer

        source.configure.side_effect = slow_configure
        with patch('eeg.core.sources.LiveSource', return_value=source):
            page.start(); self.flush()
            page.panel.hardware.only(['O1', 'O2']); page.panel.hardware.no_bias()
            page.panel.hardware.apply_button.click()
            # The click returns at once: the window keeps running meanwhile.
            self.assertTrue(page._configuring)
            self.assertFalse(page.panel.hardware.apply_button.isEnabled())
            reads = source.read.call_count
            page._tick()
            self.assertEqual(source.read.call_count, reads)   # no read blocks on the busy board
            release.set(); self.flush()
            self.assertFalse(page._configuring)
            thread, sent, previous = calls[0]
            self.assertNotEqual(thread, gui_thread)
            self.assertEqual([i for i, s in enumerate(sent) if s.on], [6, 7])
            self.assertFalse(any(s.bias for s in sent))
            self.assertEqual(list(page.panel.selected_channels), [6, 7])
            # Switching one more channel off passes what the board already has.
            page.panel.hardware.on_boxes[7].setChecked(False)
            page.panel.hardware.apply_button.click(); self.flush()
            self.assertEqual(calls[1][2], sent)
            page.panel.hardware.resend_button.click(); self.flush()
            self.assertIsNone(calls[2][2])                    # Resend all: no diff
            page.stop()
        page.panel.hardware.all_on()

    def test_unicorn_live_source_uses_serial_number_and_its_own_rows(self):
        import tempfile
        from eeg.core.boards import UNICORN
        from eeg.core import hardware as hw
        from eeg.core.sources import LiveSource, hardware_log_path, load_brainflow_csv
        self.assertEqual((UNICORN.fs, UNICORN.n_channels, UNICORN.n_rows), (250, 8, 19))
        self.assertEqual(UNICORN.names, ('Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'))
        board = Mock()
        packet = np.arange(19 * 5, dtype=float).reshape(19, 5)
        board.get_board_data.return_value = packet
        params = Mock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch('brainflow.board_shim.BoardShim', return_value=board) as shim, \
                patch('brainflow.board_shim.BrainFlowInputParams', return_value=params):
            source = LiveSource('UN-2021.05.36', hw.default_settings(), record_dir=tmp, board=UNICORN)
            self.assertEqual(shim.call_args[0][0], UNICORN.board_id)
            self.assertEqual(params.serial_number, 'UN-2021.05.36')
            self.assertEqual(source.fs, 250)
            # No Cyton SDK command ever reaches the Unicorn.
            self.assertNotIn('config_board', [c[0] for c in board.method_calls])
            np.testing.assert_array_equal(source.read(5), packet[:8])
            with self.assertRaises(RuntimeError):
                source.configure(hw.default_settings())
            with self.assertRaises(RuntimeError):
                source.impedance_step(None, 0, None)
            log = hardware_log_path(source.record_path).read_text(encoding='utf-8')
            self.assertIn('Unicorn Hybrid Black: no channel settings', log)
            # A Unicorn CSV (19 columns) plays back at 250 Hz with its own names and rail.
            row = np.zeros(19); row[:8] = np.arange(8); row[17] = 1.7e9
            np.savetxt(source.record_path, np.tile(row, (500, 1)), delimiter='\t')
            rec = load_brainflow_csv(source.record_path)
            self.assertEqual((rec.data.shape, rec.fs, rec.rail_uv), ((8, 500), 250.0, 750_000.0))
            self.assertEqual(rec.channel_names[6], 'Oz')
            self.assertEqual(rec.data[5, 0], 5)
            self.assertEqual(rec.timestamps[0], 1.7e9)

    def test_unicorn_mode_starts_eight_channels_without_hardware_controls(self):
        page = self.page; page.stop()
        page.panel.port.setText('COM9')
        page.panel.mode.setCurrentIndex(2)
        self.assertEqual(page.panel.board.key, 'unicorn')
        self.assertFalse(page.panel.hardware.isEnabled())
        # No channel commands on the Unicorn: the software choice replaces the Cyton's box.
        self.assertFalse(page.panel.hardware.isVisibleTo(page.panel))
        self.assertTrue(page.panel.unicorn_channels.isVisibleTo(page.panel))
        page.panel.unicorn_channels.all_on()
        page.panel.port.setText('UN-2021.05.36')
        source = Mock(fs=250)
        with patch('eeg.core.sources.LiveSource', return_value=source) as live:
            page.start(); self.flush()
            self.assertEqual(live.call_args[0][0], 'UN-2021.05.36')
            # The software choice travels to the recording's log; nothing is sent to the headset.
            chosen = live.call_args[0][1]
            self.assertEqual(len(chosen), 8)
            self.assertTrue(all(s.on and not s.bias for s in chosen))
            self.assertEqual(live.call_args.kwargs['board'].key, 'unicorn')
            self.assertEqual((page.fs, page.n_channels), (250, 8))
            self.assertEqual(page.panel.channel_names[-1], 'PO8')
            self.assertEqual(list(page.panel.selected_channels), list(range(8)))
            self.assertEqual(page._rail_uv, 750_000.0)
            page._ingest(np.tile(sine(fs=250), (8, 1)))
            page.start_impedance(); page._apply_hardware()
            source.impedance_step.assert_not_called(); source.configure.assert_not_called()
            page.stop()
        # The Cyton's port is kept apart from the Unicorn's serial.
        page.panel.mode.setCurrentIndex(1)
        self.assertEqual(page.panel.port.text(), 'COM9')
        self.assertTrue(page.panel.hardware.isEnabled())
        from eeg.core.boards import UNICORN
        self.assertEqual(page.panel.address(UNICORN), 'UN-2021.05.36')

    def test_apply_sends_only_changed_channels(self):
        from eeg.core import hardware as hw
        board = Mock(); board.config_board.return_value = 'Success$$$'
        before = [hw.ChannelSetting(i in (6, 7), False) for i in range(16)]
        after = [hw.ChannelSetting(i == 6, False) for i in range(16)]
        sent = hw.apply_settings(board, after, before)
        self.assertEqual([c for c, _ in sent], ['x8161000X'])
        self.assertEqual(len(hw.apply_settings(board, after)), 16)
        self.assertEqual(hw.apply_settings(board, after, after), [])


if __name__ == '__main__':
    unittest.main()
