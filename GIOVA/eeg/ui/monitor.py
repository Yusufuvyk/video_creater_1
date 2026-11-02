"""The live monitor page: rolling traces, spectrum, band power, transport."""
import threading
import time
import traceback
import os
from pathlib import Path

from dataclasses import replace

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from .collapsible import STRIP, Collapsible, Strip, fold_button
from .detach import DetachableTabs, DragTab, LEFT, ResizableSide, TAB_STRIP, pop_out_button
from .preferences import data_dir, recording_prefix, user_settings
from ..core.game_bridge import SignalTap
from ..core import hardware as hw
from ..core import channel_profiles as cp
from ..core.boards import CYTON, CYTON_DAISY, UNICORN
from ..core.config import CH_NAMES, DATA_DIR, FS, N_CH
from .matdialog import open_recording, remap
from .panel import ControlPanel
from .playback import Player, TransportBar
from .plots import BandPowerView, SpectrumView, TimeSeriesView, channel_pen
from .live_views import MotionView, QualityTable, SpectrumPanel
from .jobs import Jobs
from .widgets import NoScrollComboBox, NoScrollSpinBox, fit_to_screen
from ..core.monitoring import (QualityLimits, QualityTracker, movement_spans, quality_snapshot,
                               spectra_snapshot)
from ..core.processing import (ChannelAutoScaler, StreamFilter,
                               band_powers, design_filter)
from .theme import color, muted

FRAME_MS = 50  # 20 fps: the standard rate, and the fallback
#: The smooth rate. Measured offscreen on the lab PC: 16 channels cost p95 21 ms
#: a frame, 4 channels 8 ms, so 33 ms leaves room; a slower computer drops back
#: to FRAME_MS on its own (see `_keep_up`). Only repainting changes: samples are
#: taken by elapsed time, so data, features and recordings are the same.
SMOOTH_FRAME_MS = 33
#: 60 fps. Measured the same way, 16 channels at 1 px traces: 55-58 fps reached.
FAST_FRAME_MS = 16
#: Settings → Monitor → Display rate: fps -> frame period.
DISPLAY_RATES = {60: FAST_FRAME_MS, 30: SMOOTH_FRAME_MS}


class MonitorPage(QtWidgets.QWidget):
    """Everything that updates continuously."""

    status_changed = QtCore.Signal(str)
    file_loaded = QtCore.Signal(str)
    #: (filtered microvolts, fs, channel names) once per drawn frame. The live
    #: feature window listens to this rather than reading the board itself, so
    #: its numbers describe exactly the signal on screen.
    frame_ready = QtCore.Signal(object, float, object)
    features_ready = QtCore.Signal(object, float, object, float)
    #: (fs, channel names) whenever a new recording or the board is opened.
    source_changed = QtCore.Signal(float, object)

    def __init__(self):
        super().__init__()
        self.source = None
        self.player = None
        self.features_window = None
        #: The raw stream as games see it (eeg.core.game_bridge).
        self.tap = SignalTap()
        self._game_markers = None
        # Board rate and channel count until a file says otherwise. Both belong
        # to whatever is being played, never to the board that is not connected.
        self.fs = FS
        self.n_channels = N_CH
        self.scaler = ChannelAutoScaler()
        self._trace_signature = None
        self._spectrum_signature = None
        self._band_signature = None
        self._last_spectrum = 0.0
        self._spectrum_position = None
        # The buffer holds FILTERED samples: each one passes through the filter
        # exactly once, on arrival, and never changes afterwards.
        self.buffer = np.zeros((self.n_channels, FS * 10))
        self.filled = 0
        self.stream = None
        self.feature_buffer = np.empty((self.n_channels, 0))
        self.feature_capacity = int(self.fs * 3)
        self.sample_position = 0
        self._last_tick = None
        self._new_samples = False
        self.raw_buffer = np.empty((self.n_channels, 0))
        self.raw_spectral_buffer = np.empty((self.n_channels, 0))
        self.spectral_buffer = np.empty((self.n_channels, 0))
        self.motion_buffer = None
        self._new_motion = False
        self.quality_tracker = QualityTracker()
        self._last_quality = -float('inf')
        self._quality_position = None
        self._last_arrival = None
        self._waiting_since = time.perf_counter()
        self._rail_uv = None
        self._spectral_revision = 0
        self._job_position = None
        self._last_job = -float('inf')
        self._starting = False
        self._frame_costs = []
        self._frame_ms = FRAME_MS
        self._frame_gaps = []
        self.impedance_sweep = None
        self.impedance_results = []
        self.short_capture = None
        self.short_results = []
        self._configuring = False       # a board command is running in the background
        self._pending_hardware = None
        self.hardware_jobs = Jobs(self)
        self.hardware_jobs.finished.connect(self._hardware_applied)
        self.connect_jobs = Jobs(self)
        self.connect_jobs.finished.connect(self._connected)
        self._connect_cancel = None
        self._connect_request = None
        self.board_jobs = Jobs(self)
        self.board_jobs.finished.connect(self._board_job_finished)
        self._board_job_done = None
        self._finished_sweep = None
        self._applied_hardware = None   # what the board has, not what the boxes say
        self._off_channels = frozenset()
        self._last_off_check = -float('inf')

        self.panel = ControlPanel(FS)
        self.time_view = TimeSeriesView()
        self.spectrum_view = SpectrumView()
        self.band_view = BandPowerView()
        self.transport = TransportBar()
        self.quality_table = QualityTable()
        self.quality_table.set_channels(self.panel.channel_names)
        self.quality_table.show_state('Stopped')
        self.multi_spectrum = SpectrumPanel()
        self.spectral_jobs = Jobs(self)
        self.spectral_jobs.finished.connect(self._spectra_ready)
        self.multi_spectrum.computation_changed.connect(self._invalidate_spectra)
        self._lay_out()
        self.quality_table.channel_clicked.connect(self._show_channel)
        self.multi_spectrum.channel_clicked.connect(self._show_channel)
        self.panel.psd_channel.currentIndexChanged.connect(lambda _: self._draw_detail())
        self.multi_spectrum.log.toggled.connect(lambda _: self._draw_detail())
        self.multi_spectrum.minimum.valueChanged.connect(lambda _: self._draw_detail())
        self.multi_spectrum.maximum.valueChanged.connect(lambda _: self._draw_detail())
        self.multi_spectrum.scale.currentIndexChanged.connect(lambda _: self._draw_detail())
        self.source_changed.connect(self._source_labels)
        # The traces' time span or width changes -> the motion plots follow.
        self.time_view.plotItem.vb.sigXRangeChanged.connect(lambda *_: self._align_motion())
        self.time_view.plotItem.vb.sigResized.connect(lambda *_: self._align_motion())

        self.panel.start_clicked.connect(self.toggle)
        self.panel.remap_clicked.connect(self.remap)
        self.panel.features_clicked.connect(self.open_features)
        self.panel.names_changed.connect(self._names_changed)
        self.panel.hardware.apply_clicked.connect(self._apply_hardware)
        self.panel.hardware.resend_clicked.connect(lambda: self._apply_hardware(resend_all=True))
        self.panel.hardware.impedance_clicked.connect(self.start_impedance)
        self.panel.hardware.short_clicked.connect(self.start_internal_short)
        self.panel.hardware.green_limit.valueChanged.connect(lambda _: self._show_impedance())
        self.panel.hardware.yellow_limit.valueChanged.connect(lambda _: self._show_impedance())
        self.panel.filters.changed.connect(self._rebuild_stream)
        self.panel.window_seconds.valueChanged.connect(self._resize_buffer)
        self.transport.play_toggled.connect(self._toggle_play)
        self.transport.seek_requested.connect(self._seek)
        self.transport.speed_changed.connect(self._set_speed)

        self.timer = QtCore.QTimer(self)
        # Measured on the lab PC (Windows 11) asking for 33 ms: the default
        # coarse timer fired every 49.7 ms median (p5 34, p95 55), so 30 fps
        # never happened; the precise timer: 33.0 ms, SD 0.5 ms.
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.panel.set_address(CYTON_DAISY, self.preferences.value('port', 'COM7'))
        self.panel.set_address(CYTON, self.preferences.value(
            'cyton_port', self.preferences.value('port', 'COM7')))
        self.panel.set_address(UNICORN, self.preferences.value('unicorn_serial', ''))
        self.panel.hardware.set_settings(hw.decode(self.preferences.value('hardware', '')))
        self.panel.unicorn_channels.set_settings(
            hw.decode(self.preferences.value('unicorn_channels', ''), n=len(UNICORN.names)))
        self.panel.unicorn_channels.apply_clicked.connect(self._apply_hardware)
        self.panel.profile_chosen.connect(self._profile_chosen)
        self.panel.profile_save_clicked.connect(self._save_profile)
        self.panel.profile_edit_clicked.connect(self.edit_profiles)
        self.panel.profile_delete_clicked.connect(self._delete_profile)
        self._restore_channel_profile()
        self.panel.hardware.green_limit.setValue(self.preferences.value('impedance_green_k', 750, type=float))
        self.panel.hardware.yellow_limit.setValue(self.preferences.value('impedance_yellow_k', 2500, type=float))
        for key, combo in [('spectrum_mode', self.multi_spectrum.mode),
                           ('spectrum_layout', self.multi_spectrum.arrangement),
                           ('spectrum_input', self.multi_spectrum.input)]:
            combo.setCurrentText(self.preferences.value(key, combo.currentText()))
        self.multi_spectrum.log.setChecked(self.preferences.value('spectrum_log', True, type=bool))
        hidden = str(self.preferences.value('spectrum_hidden', '') or '')
        self.multi_spectrum.set_hidden([n for n in hidden.split(',') if n])
        self.multi_spectrum.minimum.setValue(self.preferences.value('spectrum_min', 1, type=float))
        self.multi_spectrum.maximum.setValue(self.preferences.value('spectrum_max', 60, type=float))
        self.multi_spectrum.scale.setCurrentIndex(
            max(0, self.multi_spectrum.scale.findData(self.preferences.value('spectrum_scale', 'auto'))))
        self.view_tabs.setCurrentIndex(self.preferences.value('tab', 0, type=int))
        self._apply_trace_width()
        from .panel import DEFAULT_WINDOW_S
        self.panel.window_seconds.setValue(self.preferences.value('trace_window', DEFAULT_WINDOW_S, type=float))
        self._apply_spectral_settings()
        self.panel.amplitude.setValue(self.preferences.value('trace_amplitude', 100, type=float))
        self.panel.auto_amplitude.setChecked(self.preferences.value('trace_auto', True, type=bool))
        for i, control in enumerate(self.quality_controls):
            control.setValue(self.preferences.value(f'quality_{i}', control.value(), type=float))
            control.valueChanged.connect(self._quality_limits_changed)

    def _lay_out(self):
        self.preferences = user_settings()
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.panel.source_bar, 1)
        settings = QtWidgets.QPushButton('Controls'); settings.setCheckable(True)
        settings.setToolTip('Show or hide the controls: source and board, channel profile, filters, '
                            'channels and the trace window. Thresholds and folders are in Settings.')
        top.addWidget(settings)
        root.addLayout(top)
        self.connection_label = QtWidgets.QLabel('Stopped · no data')
        self.connection_label.setStyleSheet('padding: 5px; background: {}; color: {};'.format(color('panel'), color('panel_text')))
        root.addWidget(self.connection_label)
        # Side panels are dragged wider or narrower; their widths are remembered.
        body = ResizableSide('monitor/side', self.preferences); root.addWidget(body, 1)
        left = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(left); side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.quality_table, 1)
        choose = QtWidgets.QHBoxLayout()
        choose.addWidget(QtWidgets.QLabel('Graphs:'))
        for label, callback in [('All', lambda: self.panel.channel_button.set_selected(range(self.n_channels))),
                                ('Clear', lambda: self.panel.channel_button.set_selected([]))]:
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); choose.addWidget(button)
        side.addLayout(choose)
        note = QtWidgets.QLabel('Click a channel for details. Signal checks are not impedance measurements.')
        note.setWordWrap(True); side.addWidget(note)
        # Folds to a strip at the left edge when the traces need the room.
        self.quality_pane = Collapsible('ALL CHANNELS · RAW SIGNAL QUALITY', left, 'monitor/quality_pane',
                                        self.preferences, side='left')
        body.add_side(self.quality_pane, 'quality', 414, minimum=280, maximum=800)
        self.quality_pane.restore()

        self.chart_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.chart_splitter.setChildrenCollapsible(False)
        charts = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(charts)
        column.setContentsMargins(0,0,0,0)
        self.time_scroll = QtWidgets.QScrollArea()
        self.time_scroll.setWidgetResizable(True)
        self.time_scroll.setWidget(self.time_view)
        self.time_scroll.setMinimumHeight(180)
        self.time_scroll.setToolTip("Scroll inside the signal panel to inspect more channels. Each row shows its microvolt scale.")
        # Either view can be popped out into its own window (e.g. the spectrum on
        # a second monitor); closing that window brings it back.
        # The motion sensor sits under the traces, on the same time axis, and
        # travels with them into the wide layout or a window of their own.
        self.motion_view = MotionView()
        self.motion_view.toggle.setChecked(self.preferences.value('motion_shown', True, type=bool))
        self.signals_page = QtWidgets.QWidget()
        signals_box = QtWidgets.QVBoxLayout(self.signals_page)
        signals_box.setContentsMargins(0, 0, 0, 0); signals_box.setSpacing(0)
        signals_box.addWidget(self.time_scroll, 1)
        signals_box.addWidget(self.motion_view)
        self.view_tabs = DetachableTabs('monitor/views', self.preferences)
        self.view_tabs.addTab(self.signals_page, 'Signals')
        self.view_tabs.addTab(self.multi_spectrum, 'Spectrum')
        self.view_tabs.detached.connect(lambda *_: self._arrange())
        self.view_tabs.reattached.connect(lambda *_: self._arrange())
        self.view_tabs.docked.connect(self._view_dropped)
        # Wide screens show both views side by side, each with its own pop-out button.
        self.wide_container = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.wide_container.setChildrenCollapsible(False)
        self.wide_slots = {}
        for name in ('Signals', 'Spectrum'):
            slot = QtWidgets.QWidget()
            box = QtWidgets.QVBoxLayout(slot); box.setContentsMargins(0, 0, 0, 0); box.setSpacing(0)
            head = QtWidgets.QWidget()
            header = QtWidgets.QHBoxLayout(head); header.setContentsMargins(0, 0, 0, 0)
            # Drag the title out, as a tab, to give the view its own window.
            title = DragTab(name, lambda point, _press, n=name: self.view_tabs.begin_drag(n, point))
            title.setToolTip("Drag out to open in its own window")
            header.addWidget(title); header.addStretch()
            pop = pop_out_button(slot)
            pop.clicked.connect(lambda _=False, n=name: self.view_tabs.pop_out(n))
            header.addWidget(pop)
            box.addWidget(head)
            if name == 'Spectrum':
                # Beside the signals, the spectrum folds to a strip at the right edge.
                fold = fold_button('Spectrum', closes_towards_left=False)
                fold.clicked.connect(lambda: self.set_spectrum_collapsed(True))
                header.addWidget(fold)
                self.spectrum_head = head
                self.spectrum_strip = Strip('Spectrum', opens_towards_right=False)
                self.spectrum_strip.clicked.connect(lambda: self.set_spectrum_collapsed(False))
                self.spectrum_strip.hide()
                box.addWidget(self.spectrum_strip, 1)
            self.wide_container.addWidget(slot)
            self.wide_slots[name] = slot
        self._spectrum_collapsed = self.preferences.value('monitor/spectrum_collapsed', False, type=bool)
        self._wide_sizes = None
        self.wide_container.hide(); self._wide = False
        # Where a view dragged out of the hidden tabs can be dropped back.
        self.view_tabs.set_drop_area(self.wide_container)
        column.addWidget(self.view_tabs, 1)
        column.addWidget(self.wide_container, 1)
        self.detail = QtWidgets.QDialog(self)
        self.detail.setWindowTitle('Channel detail'); fit_to_screen(self.detail, 900, 420)
        detail_layout = QtWidgets.QVBoxLayout(self.detail)
        detail_layout.addWidget(self.panel.psd_channel)
        row = QtWidgets.QHBoxLayout(); detail_layout.addLayout(row, 1)
        row.addWidget(self.spectrum_view, 2); row.addWidget(self.band_view, 1)
        self.spectrum_view.setMinimumHeight(130)
        self.band_view.setMinimumHeight(130)
        column.addWidget(self.transport, 0)
        self.feature_area = QtWidgets.QWidget()
        self.feature_layout = QtWidgets.QVBoxLayout(self.feature_area)
        self.feature_layout.setContentsMargins(0,0,0,0)
        self.feature_area.setVisible(False)
        self.chart_splitter.addWidget(charts)
        self.chart_splitter.addWidget(self.feature_area)
        body.add_main(self.chart_splitter)
        self.settings_scroll = QtWidgets.QScrollArea()
        self.settings_scroll.setWidgetResizable(True); self.settings_scroll.setWidget(self.panel)
        self.settings_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The panel is 300 px wide plus the scroll bar, and stays so: narrower
        # cut it off, wider only left an empty strip beside it.
        body.add_side(self.settings_scroll, 'settings', 320, minimum=320, maximum=320)
        self.settings_scroll.hide()
        self.body = body
        settings.toggled.connect(self.settings_scroll.setVisible)
        self.quality_settings = QtWidgets.QGroupBox('Quality warning thresholds')
        form = QtWidgets.QFormLayout(self.quality_settings)
        self.quality_controls = []
        for label, value, maximum, suffix in [('Flat (EEG µV <)', .1, 100, ' µV'), ('Check signal (EEG µV >)', 100, 100000, ' µV'),
                                               ('Line noise (50 Hz >)', 20, 100, ' %'), ('Saturated (rail >)', 15, 100, ' %')]:
            spin = NoScrollSpinBox(); spin.setRange(0, maximum); spin.setDecimals(2)
            spin.setValue(value); spin.setSuffix(suffix); form.addRow(label, spin)
            self.quality_controls.append(spin)
        note = QtWidgets.QLabel('Display warnings only, from the last 4 s of raw data. EEG µV = RMS of the '
                                '1–45 Hz band. Rail: 187,500 µV at Cyton gain 24×, 750,000 µV on the Unicorn; '
                                'unknown files: unavailable.')
        note.setWordWrap(True); form.addRow(note)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange()

    def settings_page(self):
        """The Monitor's page in the program's Settings.

        The widgets are the ones the Monitor already owns, so a threshold has
        one home and one value, wherever it is shown.
        """
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        views = QtWidgets.QGroupBox('Signals and Spectrum')
        form = QtWidgets.QFormLayout(views)
        self.layout_choice = NoScrollComboBox()
        for label, key in (('Automatic (by window size)', 'auto'), ('Tabs', 'tabs'), ('Side by side', 'split')):
            self.layout_choice.addItem(label, key)
        stored = self.preferences.value('monitor/views/layout', 'auto') or 'auto'
        self.layout_choice.setCurrentIndex(max(0, self.layout_choice.findData(stored)))
        self.layout_choice.currentIndexChanged.connect(self._layout_choice_changed)
        form.addRow('Arrangement', self.layout_choice)
        hint = QtWidgets.QLabel('Dragging a view onto the left or right edge of the other one also '
                                'sets "Side by side".')
        hint.setWordWrap(True); hint.setStyleSheet(muted())
        form.addRow(hint)
        self.display_rate = NoScrollComboBox()
        self.display_rate.addItem('30 fps (drops to 20 if the computer is slow)', 30)
        self.display_rate.addItem('60 fps (smoothest; drops to 30, then 20, if the computer is slow)', 60)
        from .preferences import setting
        self.display_rate.setCurrentIndex(max(0, self.display_rate.findData(setting('monitor/display_fps', 30))))
        self.display_rate.currentIndexChanged.connect(self._display_rate_changed)
        form.addRow('Display rate', self.display_rate)
        self.trace_width = NoScrollComboBox()
        self.trace_width.addItem('Thin · 1 px (fastest)', 1)
        self.trace_width.addItem('Bold · 2 px (projector; about 5× slower to draw)', 2)
        self.trace_width.setCurrentIndex(max(0, self.trace_width.findData(setting('monitor/trace_width', 1))))
        self.trace_width.setToolTip('Line width of the traces and spectra. Wide lines are much slower '
                                    'for Qt to draw: 1 px keeps 30 fps with 16 channels.')
        self.trace_width.currentIndexChanged.connect(self._trace_width_changed)
        form.addRow('Trace width', self.trace_width)
        self.trace_sweep = NoScrollComboBox()
        self.trace_sweep.addItem('Scroll: the newest signal enters on the right', False)
        self.trace_sweep.addItem('Sweep: traces stand still, a cursor writes over them', True)
        self.trace_sweep.setCurrentIndex(max(0, self.trace_sweep.findData(bool(setting('monitor/trace_sweep')))))
        self.trace_sweep.setToolTip('Scroll moves every wave across the plot once per window, which tires '
                                    'the eye with a short window. Sweep keeps each wave where it was written '
                                    'until the cursor comes round again, as on a clinical EEG screen.')
        self.trace_sweep.currentIndexChanged.connect(self._trace_sweep_changed)
        form.addRow('Traces', self.trace_sweep)
        column.addWidget(views)
        column.addWidget(self._spectral_settings_group())
        column.addWidget(self.quality_settings)
        return page

    def board_settings_page(self):
        """What only the OpenBCI boards have, on a Settings page of its own.

        Kept apart from the Monitor's page: with a file or another amplifier
        none of it applies, and among the general settings it read as if it did.
        """
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        impedance = QtWidgets.QGroupBox('Impedance colours (OpenBCI Cyton and Cyton+Daisy only)')
        box = QtWidgets.QVBoxLayout(impedance)
        box.addWidget(self.panel.hardware.limits)
        note = QtWidgets.QLabel('For the impedance check of the OpenBCI boards only: 750 and 2500 kΩ are the '
                                'OpenBCI GUI defaults for its 6 nA lead-off test with passive electrodes. '
                                'They are not general EEG limits and are not used for the Unicorn or for '
                                'files (a BrainVision recording keeps its own impedances in the .vhdr; '
                                'Brain Products systems aim for tens of kΩ).')
        note.setWordWrap(True); note.setStyleSheet(muted())
        box.addWidget(note)
        column.addWidget(impedance)
        diagnostics = QtWidgets.QGroupBox('Diagnostics (Cyton+Daisy, while streaming)')
        box = QtWidgets.QVBoxLayout(diagnostics)
        box.addWidget(self.panel.hardware.diagnostics)
        column.addWidget(diagnostics)
        return page

    def _spectral_settings_group(self):
        """FFT/PSD window and Welch segment, applied live."""
        group = QtWidgets.QGroupBox('Spectra (Monitor)')
        form = QtWidgets.QFormLayout(group)
        self.spectrum_seconds = NoScrollSpinBox()
        self.spectrum_seconds.setRange(0, 120); self.spectrum_seconds.setDecimals(1)
        self.spectrum_seconds.setSingleStep(1); self.spectrum_seconds.setSuffix(' s')
        self.spectrum_seconds.setSpecialValueText('Same as display window')
        self.spectrum_seconds.setToolTip('Seconds of signal the FFT and PSD use. FFT resolution '
                                         'Δf = 1 / window; longer is finer but reacts more slowly.')
        self.welch_seconds = NoScrollSpinBox()
        self.welch_seconds.setRange(0.25, 30); self.welch_seconds.setDecimals(2)
        self.welch_seconds.setSingleStep(0.5); self.welch_seconds.setSuffix(' s')
        self.welch_seconds.setToolTip('Length of each Welch segment: PSD resolution Δf = 1 / segment. '
                                      'Shorter segments: more of them to average, smoother but coarser.')
        self.welch_overlap = NoScrollSpinBox()
        self.welch_overlap.setRange(0, 90); self.welch_overlap.setDecimals(0)
        self.welch_overlap.setSingleStep(5); self.welch_overlap.setSuffix(' %')
        self.welch_overlap.setToolTip('How much consecutive Welch segments overlap. 50 % is the usual '
                                      'choice with a Hann window (Welch, 1967).')
        self._load_spectral_settings()
        for spin in (self.spectrum_seconds, self.welch_seconds, self.welch_overlap):
            spin.valueChanged.connect(self._spectral_settings_changed)
        form.addRow('FFT/PSD window', self.spectrum_seconds)
        form.addRow('Welch segment', self.welch_seconds)
        form.addRow('Welch overlap', self.welch_overlap)
        self.spectral_summary = QtWidgets.QLabel(self._spectral_summary())
        self.spectral_summary.setWordWrap(True)
        self.spectral_summary.setStyleSheet(muted())
        form.addRow(self.spectral_summary)
        note = QtWidgets.QLabel('Applied at once, also while streaming: the spectrum restarts on the '
                                'new frequency grid. The quality table keeps 4 s and 2 s segments.')
        note.setWordWrap(True); note.setStyleSheet(muted())
        form.addRow(note)
        return group

    def _load_spectral_settings(self):
        from .preferences import setting
        for spin, key, default in ((self.spectrum_seconds, 'monitor/spectrum_seconds', 0.0),
                                   (self.welch_seconds, 'monitor/welch_seconds', 2.0),
                                   (self.welch_overlap, 'monitor/welch_overlap', 50.0)):
            spin.blockSignals(True)
            spin.setValue(float(setting(key, default)))
            spin.blockSignals(False)

    def _spectral_settings_changed(self, _value=None):
        self.preferences.setValue('monitor/spectrum_seconds', float(self.spectrum_seconds.value()))
        self.preferences.setValue('monitor/welch_seconds', float(self.welch_seconds.value()))
        self.preferences.setValue('monitor/welch_overlap', float(self.welch_overlap.value()))
        self._apply_spectral_settings()

    def _trace_width_changed(self, _index=None):
        self.preferences.setValue('monitor/trace_width', int(self.trace_width.currentData()))
        self._apply_trace_width()

    def _trace_sweep_changed(self, _index=None):
        self.preferences.setValue('monitor/trace_sweep', bool(self.trace_sweep.currentData()))
        self._apply_trace_width()

    def _apply_trace_width(self):
        """Line width, scroll or sweep: restyle the traces and rebuild the spectrum curves."""
        from . import plots
        from .preferences import setting
        plots.set_trace_width(int(setting('monitor/trace_width', 1)))
        self.time_view.sweep = bool(setting('monitor/trace_sweep'))
        self.time_view.restyle()
        self._trace_signature = None
        self.multi_spectrum._layout_key = None
        self.multi_spectrum.redraw()

    def _display_rate_changed(self, _index=None):
        """Repaint rate only; applied at once while streaming."""
        self.preferences.setValue('monitor/display_fps', int(self.display_rate.currentData()))
        if self.timer.isActive():
            self._frame_ms = self._target_frame_ms()
            self._frame_costs, self._frame_gaps = [], []
            self.timer.setInterval(self._frame_ms)

    def _layout_choice_changed(self, _index=None):
        self.preferences.setValue('monitor/views/layout', self.layout_choice.currentData())
        self._arrange()

    def apply_graph_units(self):
        """Settings → Graphs: the PSD unit, applied to what is already on screen."""
        from .preferences import psd_unit
        self.multi_spectrum.set_psd_unit(psd_unit())
        self._draw_detail()

    def apply_settings(self):
        """Read back what the Settings dialog may have changed."""
        if hasattr(self, 'layout_choice'):
            stored = self.preferences.value('monitor/views/layout', 'auto') or 'auto'
            index = self.layout_choice.findData(stored)
            if index >= 0 and index != self.layout_choice.currentIndex():
                self.layout_choice.blockSignals(True)
                self.layout_choice.setCurrentIndex(index)
                self.layout_choice.blockSignals(False)
        if hasattr(self, 'display_rate'):
            from .preferences import setting
            self.display_rate.setCurrentIndex(max(0, self.display_rate.findData(setting('monitor/display_fps', 30))))
        if hasattr(self, 'spectrum_seconds'):
            self._load_spectral_settings()
        if hasattr(self, 'trace_width'):
            from .preferences import setting
            self.trace_width.setCurrentIndex(max(0, self.trace_width.findData(setting('monitor/trace_width', 1))))
        if hasattr(self, 'trace_sweep'):
            from .preferences import setting
            self.trace_sweep.blockSignals(True)
            self.trace_sweep.setCurrentIndex(max(0, self.trace_sweep.findData(bool(setting('monitor/trace_sweep')))))
            self.trace_sweep.blockSignals(False)
        self._apply_trace_width()
        self._apply_spectral_settings()
        self._arrange()

    def _view_dropped(self, name, side):
        """A view was dropped beside the other one, or back on the tabs.

        Dropping beside is a choice and is kept; dropping on the tabs is only
        "put it back", so it leaves the layout as it was.
        """
        if side == TAB_STRIP:
            return
        self.preferences.setValue('monitor/views/layout', 'split')
        if hasattr(self, 'layout_choice'):
            self.layout_choice.blockSignals(True)
            self.layout_choice.setCurrentIndex(max(0, self.layout_choice.findData('split')))
            self.layout_choice.blockSignals(False)
        self._arrange()
        if self._wide:
            self.wide_container.insertWidget(0 if side == LEFT else 1, self.wide_slots[name])

    def _arrange(self):
        """Tabs, or side by side. The window's size decides until the user drops
        a view somewhere, and from then on their choice does. Popped-out views
        stay in their own windows and are left out of both."""
        if not hasattr(self, 'wide_container'):
            return
        mode = self.preferences.value('monitor/views/layout', 'auto') or 'auto'
        wide = (mode == 'split' if mode in ('split', 'tabs') else
                self.window().width() >= 1700 and self.window().height() >= 900)
        attached = [(w, n) for w, n in ((self.signals_page, 'Signals'), (self.multi_spectrum, 'Spectrum'))
                    if not self.view_tabs.is_detached(n)]
        side_by_side = wide and len(attached) == 2
        in_tabs = all(self.view_tabs.indexOf(w) >= 0 for w, _ in attached)
        if side_by_side == self._wide and (side_by_side or in_tabs):
            return
        if side_by_side:
            self._compact_tab = self.view_tabs.currentIndex()
            for widget, name in attached:
                self.view_tabs.removeTab(self.view_tabs.indexOf(widget))
                self.wide_slots[name].layout().addWidget(widget, 1)
                widget.show()
        else:
            for widget, name in attached:
                if self.view_tabs.indexOf(widget) < 0:
                    self.view_tabs.insert_home(name)
            if self.view_tabs.count():
                self.view_tabs.setCurrentIndex(min(getattr(self, '_compact_tab', 0), self.view_tabs.count() - 1))
        self._wide = side_by_side
        self.view_tabs.setVisible(not side_by_side); self.wide_container.setVisible(side_by_side)
        self._apply_spectrum_fold()

    def set_spectrum_collapsed(self, collapsed, remember=True):
        """Fold the spectrum beside the signals to a strip, or bring it back."""
        self._spectrum_collapsed = bool(collapsed)
        if remember:
            self.preferences.setValue('monitor/spectrum_collapsed', self._spectrum_collapsed)
        self._apply_spectrum_fold()

    def _apply_spectrum_fold(self):
        """Only side by side is there a right-hand pane to fold: as a tab, or in
        its own window, the spectrum is shown or not by the tab or the window."""
        if not hasattr(self, 'spectrum_strip'):
            return
        slot = self.wide_slots['Spectrum']
        inside = self.multi_spectrum.parentWidget() is slot
        folded = self._spectrum_collapsed and self._wide and inside
        was_folded = self.spectrum_strip.isVisibleTo(slot)
        self.spectrum_head.setVisible(not folded)
        self.spectrum_strip.setVisible(folded)
        if inside:
            self.multi_spectrum.setVisible(not folded)
        if folded:
            if not was_folded:
                self._wide_sizes = self.wide_container.sizes()
            slot.setFixedWidth(STRIP)
        else:
            slot.setMinimumWidth(0); slot.setMaximumWidth(16777215)
            if was_folded and self._wide_sizes and min(self._wide_sizes) > STRIP:
                self.wide_container.setSizes(self._wide_sizes)
            elif was_folded:
                total = sum(self.wide_container.sizes())
                self.wide_container.setSizes([total // 2, total - total // 2])
            self._wide_sizes = None

    def _switched_off_now(self):
        """Channels powered down on the board right now: live settings, or the recording's log."""
        board = self.panel.board
        if self.is_live and self._applied_hardware and len(self._applied_hardware) == self.n_channels:
            return {i for i, s in enumerate(self._applied_hardware) if not s.on}
        recording = getattr(self.source, 'recording', None)
        if recording is not None and getattr(recording, 'channel_log', None):
            now = self.sample_position / self.fs
            return recording.switched_off(max(0.0, now - 4), now)
        return set()

    def _as_the_board_gives(self, chunk):
        """The raw chunk a game should see.

        A game reads the board's own signal, so the Monitor's filters and
        display choices are none of its business. Switched-off channels are:
        the Cyton stops sending them, the Unicorn cannot be told to, so those
        rows are zeroed here and "off" means the same in the game, the views
        and the analysis. The recording keeps whatever the board sent.
        """
        board = self.panel.board
        if not (self.is_live and board is not None and not board.channel_settings):
            return chunk
        off = [i for i, setting in enumerate(self._applied_hardware or ())
               if not setting.on and i < chunk.shape[0]]
        if not off:
            return chunk
        chunk = chunk.copy()
        chunk[off] = 0.0
        return chunk

    def _refresh_off_channels(self, force=False):
        """Keep the views' idea of the powered-down channels current.

        A switched-off Cyton channel reads exactly 0 uV: a flat trace and an
        empty spectrum that take the room of a real channel and are read as
        measurements. They are dropped from both views instead, and said so.
        """
        now = time.perf_counter()
        if not force and now - self._last_off_check < .5:
            return
        self._last_off_check = now
        off = frozenset(self._switched_off_now())
        if off == self._off_channels:
            return
        self._off_channels = off
        self._trace_signature = None
        names = self.panel.channel_names
        self.multi_spectrum.set_off([names[i] for i in sorted(off) if i < len(names)])

    def _drawn_channels(self):
        """Selected channels, without the ones the board has switched off."""
        return [c for c in self.panel.selected_channels if c not in self._off_channels]

    def _source_labels(self, fs, names):
        self.quality_table.set_channels(names)

    def _show_channel(self, index):
        self.panel.psd_channel.setCurrentIndex(index)
        self.detail.show()
        self._draw_detail()

    def save_preferences(self):
        self.preferences.setValue('port', self.panel.address(CYTON_DAISY))
        self.preferences.setValue('cyton_port', self.panel.address(CYTON))
        self.preferences.setValue('unicorn_serial', self.panel.address(UNICORN))
        # Preserve channels 9-16 while the Cyton-only mode hides them.
        self.preferences.setValue('hardware', hw.encode(self.panel.hardware.all_settings()))
        for key, combo in [('spectrum_mode', self.multi_spectrum.mode),
                           ('spectrum_layout', self.multi_spectrum.arrangement),
                           ('spectrum_input', self.multi_spectrum.input)]:
            self.preferences.setValue(key, combo.currentText())
        self.preferences.setValue('spectrum_log', self.multi_spectrum.log.isChecked())
        self.preferences.setValue('spectrum_hidden', ','.join(sorted(self.multi_spectrum.hidden)))
        self.preferences.setValue('spectrum_min', self.multi_spectrum.minimum.value())
        self.preferences.setValue('spectrum_max', self.multi_spectrum.maximum.value())
        self.preferences.setValue('spectrum_scale', self.multi_spectrum.scale.currentData())
        self.preferences.setValue('motion_shown', self.motion_view.toggle.isChecked())
        self.preferences.setValue('unicorn_channels', hw.encode(self.panel.unicorn_channels.settings()))
        self.preferences.setValue('tab', getattr(self, '_compact_tab', 0) if self._wide else self.view_tabs.currentIndex())
        self.body.save()
        self.preferences.setValue('trace_window', self.panel.window_seconds.value())
        self.preferences.setValue('trace_amplitude', self.panel.amplitude.value())
        self.preferences.setValue('trace_auto', self.panel.auto_amplitude.isChecked())
        for i, control in enumerate(self.quality_controls):
            self.preferences.setValue(f'quality_{i}', control.value())
        self.preferences.setValue('impedance_green_k', self.panel.hardware.green_limit.value())
        self.preferences.setValue('impedance_yellow_k', self.panel.hardware.yellow_limit.value())

    def _quality_limits_changed(self):
        self.quality_tracker.reset()
        self._last_quality = -float('inf'); self._quality_position = None

    # ---------------- lifecycle ----------------

    def toggle(self):
        self.stop() if self.timer.isActive() else self.start()

    def open_file(self, path):
        """Play a recording chosen elsewhere (the Projects pane). False when refused.

        A live session is never ended from there: stopping the board, and the
        recording it is writing, stays a deliberate press of Stop.
        """
        if self.source is not None or self._starting:
            if self.panel.is_live:
                self.status_changed.emit("Stop the live stream before playing a file")
                return False
            self.stop()
        self.panel.mode.setCurrentIndex(0)
        combo = self.panel.file_combo
        index = combo.findData(str(path))
        if index < 0:
            combo.addItem(Path(path).name, str(path))
            index = combo.count() - 1
        combo.setCurrentIndex(index)
        self.start()
        return self.source is not None

    def start(self):
        from ..core.sources import FileSource, LiveSource
        if self._starting or self.timer.isActive():
            return
        if self.panel.filters.problem():
            self.status_changed.emit("Cannot start: " + self.panel.filters.problem())
            return
        self._starting = True
        self.panel.start_button.setEnabled(False)
        self.panel.set_running(True)
        self.connection_label.setText('Connecting…')
        connecting = False
        try:
            if self.panel.is_live:
                board = self.panel.board
                address = self.panel.address(board)
                hardware = (self.panel.hardware.settings() if board.channel_settings
                            else self.panel.unicorn_channels.settings())
                names = list(self.panel.hardware.names) if board is CYTON_DAISY else None
                # Opening the board (reset, channel commands, each waiting for a
                # reply) takes seconds; off the GUI thread the window stays alive.
                cancel = threading.Event()
                self._connect_cancel = cancel

                def connect():
                    source = LiveSource(address, hardware, record_dir=data_dir(), board=board,
                                        record_prefix=recording_prefix(), names=names)
                    if cancel.is_set():          # Stop or window closed meanwhile
                        source.close()
                        return None
                    return source

                self._connect_request = (board, address, hardware)
                self.connection_label.setText('Connecting to {}…'.format(board.label))
                self.connect_jobs.start(connect)
                connecting = True
                return
            else:
                path = self.panel.selected_file
                if not path:
                    self.status_changed.emit("no file selected")
                    return
                # Shared with the analysis page, so a .mat asks its layout
                # question once and both pages get the same answer.
                recording, note = open_recording(self, path)
                if recording is None:
                    self.status_changed.emit(note or "cancelled")
                    return

                self.source = FileSource(recording)
                # The recording's events, as Analysis finds them, are marked on the traces.
                from ..core.epochs import raw_events_for
                events = raw_events_for(recording, path)[0]
                self.time_view.set_markers(*((events.onsets, events.labels) if events is not None else ((), ())))
                self.fs = self.source.fs
                self.n_channels = recording.n_channels
                self.panel.set_sampling_rate(self.fs)
                self.panel.set_channels(recording.channel_names)
                self.source_changed.emit(self.fs, list(recording.channel_names))
                self.player = Player(self.source, self.fs)
                self.transport.configure(self.source.n_total, self.fs)
                self._configure_motion()
                self.transport.set_playing(True)

                self.status_changed.emit("{} | {}".format(Path(path).name, recording.describe()))
                self.file_loaded.emit(path)
        except Exception as exc:
            self._start_failed(exc)
            traceback.print_exc()
            return
        finally:
            if not connecting:
                self._starting = False
                self.panel.start_button.setEnabled(True)
                if self.source is None:
                    self.panel.set_running(False)
                    if self.connection_label.text() == 'Connecting…':
                        self.connection_label.setText('Stopped · no source selected')
        self._begin_streaming()

    def _start_failed(self, exc):
        if self.source is not None:
            self.source.close()
        self.source = None; self.player = None
        self.connection_label.setText('Connection failed · check device and port')
        QtWidgets.QMessageBox.critical(
            self, "Could not start",
            "{}\n\nFor live mode, check the dongle is connected and "
            "the port is correct.".format(exc))

    def _connected(self, source, error):
        """The background connect finished: carry on with Start on the GUI thread."""
        self._starting = False
        self.panel.start_button.setEnabled(True)
        board, address, hardware = self._connect_request
        if error or source is None:
            if error:
                self._start_failed(error)
            self.panel.set_running(False)
            if source is None and not error:
                self.connection_label.setText('Stopped · no source selected')
            return
        self.source = source
        self.time_view.set_markers()          # a live stream has no recorded events to show
        self._applied_hardware = hardware
        if board.channel_settings:
            self.panel.hardware.show_applied(hardware)
        else:
            self.panel.hardware.show_pending("{}: no channel settings".format(board.label))
            self.panel.unicorn_channels.show_applied(hardware)
        self.player = None
        self.fs = self.source.fs
        # The channel profile's pin names, or the board's fixed ones.
        names = list(self.panel.hardware.names) if board is CYTON_DAISY else list(board.names)
        self.n_channels = len(names)
        self.panel.set_sampling_rate(self.fs)
        self.panel.set_channels(names)
        self.source_changed.emit(self.fs, names)
        self.transport.configure(0, self.fs)
        record = getattr(self.source, 'record_path', None)
        if board is CYTON_DAISY and isinstance(record, (str, Path)):
            # The CSV has no labels: keep the profile's names beside it for playback.
            from ..formats import channels as ch
            try:
                ch.save_names(record, names)
            except OSError:
                pass
        self.status_changed.emit(
            "streaming from {} {}".format(board.label, address or "(first paired)")
            + (" · recording {}".format(Path(record).name) if isinstance(record, (str, Path)) else ""))
        self._begin_streaming()

    def _begin_streaming(self):
        self.stream = None
        self._off_channels = frozenset()             # the new source decides afresh
        self._last_off_check = -float('inf')
        self._resize_buffer()
        self.sample_position = 0
        live_board = self.panel.board if self.player is None else None
        self._rail_uv = live_board.rail_uv if live_board is not None else None
        self._reset_live_data()
        if live_board is not None and live_board.channel_settings:
            shown = self.panel.hardware.on_channels()
        elif live_board is not None and self._applied_hardware:
            shown = [i for i, s in enumerate(self._applied_hardware) if s.on]
        else:
            shown = range(self.n_channels)
        self.panel.channel_button.set_selected(shown)
        self._configure_motion()
        self.feature_buffer = np.empty((self.n_channels,0))
        self.feature_capacity = int(self.fs * ((self.features_window.window_seconds.value() if self.features_window else 2) + 1))
        self._last_tick = time.perf_counter()
        self.panel.show_start(True)
        self._frame_ms = self._target_frame_ms()
        self._frame_costs, self._frame_gaps = [], []
        self.timer.start(self._frame_ms)

    def stop(self):
        self.timer.stop()
        if self._starting and self._connect_cancel is not None:
            # A connect is still opening the board: it closes what it opened.
            self._connect_cancel.set()
            QtCore.QThreadPool.globalInstance().waitForDone(20000)
            self._starting = False
            self.panel.start_button.setEnabled(True)
        if self._configuring:
            # Let the background command finish before the board is released.
            self.hardware_jobs.invalidate()
            self.board_jobs.invalidate()
            self._board_job_done = None
            QtCore.QThreadPool.globalInstance().waitForDone(10000)
            self._configuring = False
            self._set_hardware_busy(False)
        if self.impedance_sweep is not None:
            # The board is released below; the next connect resets it.
            self.impedance_sweep = None
            self._impedance_buttons(True)
            self.panel.hardware.impedance.setText("Impedance: check cancelled (stopped)")
        if self.short_capture is not None:
            self.short_capture = None
            self._impedance_buttons(True)
            self.panel.hardware.short_result.setText("Internal short: check cancelled (stopped)")
        if self.source:
            self.source.close()
            stranded = getattr(self.source, "stranded_path", None)
            if isinstance(stranded, (str, Path)):
                QtWidgets.QMessageBox.warning(
                    self, "Recording not moved",
                    "The recording could not be moved to\n{}\n({}).\n\nIt is safe, still at\n{}\n\n"
                    "Move it there by hand; a note is kept in {} beside it.".format(
                        self.source.record_path, self.source.move_error, stranded,
                        Path(stranded).with_name("NOT_MOVED.txt").name))
        self.source = None
        self.player = None
        self.tap.reset()
        self.panel.set_running(False)
        self._invalidate_spectra()
        self._latest_snapshot = None
        self.quality_table.show_state('Stopped')
        self.connection_label.setText('Stopped · graphs stale')
        self.detail.setWindowTitle('Channel detail · Stale (stopped)')
        self.panel.show_start(False)
        self.transport.setEnabled(False)
        self.status_changed.emit("stopped")
        if self.features_window is not None:
            self.features_window.record.setChecked(False)

    # ---------------- requests from a running game ----------------

    def source_label(self):
        if self.source is None:
            return "stopped"
        if self.player is not None:
            name = getattr(getattr(self.source, "recording", None), "path", None)
            return "file playback {}".format(Path(name).name if name else "")
        board = self.panel.board
        return "live {}".format(board.label if board is not None else "board")

    @property
    def is_live(self):
        return self.source is not None and self.player is None and hasattr(self.source, "board")

    def game_configure(self, disabled, use_bias, game):
        """Apply a game's DISABLED_CHANNELS / USE_BIAS, as the game did on its own board."""
        names = list(self.panel.channel_names)
        unknown = [name for name in disabled if name not in names]
        if unknown:
            raise ValueError("{} not in the Monitor's channels ({})".format(unknown, ", ".join(names)))
        board = self.panel.board
        if self.is_live and board is not None and not board.channel_settings:
            self.panel.unicorn_channels.set_settings(
                [hw.ChannelSetting(name not in disabled, False) for name in names])
            self._apply_software_channels()
            return "applied in software ({} has no channel commands): {} in use".format(
                board.label, len(names) - len(disabled))
        if not self.is_live or board is None or not board.channel_settings:
            return "channel settings not applicable ({})".format(self.source_label())
        settings = [hw.ChannelSetting(name not in disabled, bool(use_bias) and name not in disabled)
                    for name in names]
        self.panel.hardware.set_settings(settings)
        self._apply_hardware()
        try:
            self.source.log_note("game: {} requested channel settings".format(game), settings)
        except Exception:
            pass
        return "applied: {} on, BIAS {}".format(len(names) - len(disabled), "on" if use_bias else "off")

    def game_mark(self, value, when):
        if self.is_live:
            self.source.board.insert_marker(float(value))
        if self._game_markers is not None:
            with open(self._game_markers, "a", encoding="utf-8") as handle:
                handle.write("{:.6f},{:g}\n".format(when, value))
        return "marker {:g}".format(value)

    def game_start_recording(self, path):
        """A live board writes the game's CSV itself; file playback gets a marker file."""
        self.game_stop_recording()
        path = Path(path)
        if self.is_live:
            from ..core.sources import write_board_metadata
            write_board_metadata(path, self.source.info)
            # BrainFlow's C++ file writer cannot open a Windows path with
            # non-ASCII letters (Masaüstü, LAST_VERSİON): write to an ASCII
            # folder and move the CSV into place when the recording stops.
            from ..core.sources import ascii_stand_in
            target = ascii_stand_in(path)
            streamer = "file://{}:w".format(target or path)
            self.source.board.add_streamer(streamer)
            self._game_streamer = streamer
            self._game_move = (target, path) if target else None
            self._game_markers = None
            return "recording {}".format(path.name)
        self._game_markers = path.with_name(path.stem + "_events.csv")
        with open(self._game_markers, "w", encoding="utf-8") as handle:
            handle.write("# unix_time,label  (samples: {})\n".format(self.source_label()))
        return "file playback: markers to {}".format(self._game_markers.name)

    def game_stop_recording(self):
        streamer = getattr(self, "_game_streamer", None)
        if streamer and self.is_live:
            try:
                self.source.board.delete_streamer(streamer)
            except Exception:
                pass
        move = getattr(self, "_game_move", None)
        if move and Path(move[0]).exists():
            import shutil
            try:
                shutil.move(str(move[0]), str(move[1]))
            except OSError as exc:
                self.status_changed.emit("game recording left at {} ({})".format(move[0], exc))
        self._game_move = None
        self._game_streamer = None
        self._game_markers = None
        return "stopped"

    def _apply_hardware(self, resend_all=False):
        """Send channel power/BIAS to a connected board, or keep it for Start.

        Runs off the GUI thread: every command waits for the firmware's reply,
        and doing that here froze the window for seconds whenever a channel was
        switched off. Only changed channels are sent unless `resend_all`.
        """
        if self.panel.board is not None and not self.panel.board.channel_settings:
            self._apply_software_channels()
            return
        settings = self.panel.hardware.settings()
        if not (self.panel.is_live and self.source is not None and self.player is None):
            self.panel.hardware.show_pending("Not connected · applied at next Start")
            return
        if self.impedance_sweep is not None or self.short_capture is not None or self._configuring:
            self.status_changed.emit("board busy · try again in a moment")
            return
        previous = None if resend_all else self._applied_hardware
        count = len(hw.changed_channels(settings, previous))
        if count == 0:
            self.panel.hardware.show_applied(settings)
            return
        self._configuring = True
        self._set_hardware_busy(True)
        self.panel.hardware.show_pending("Applying {} channel{}…".format(count, "" if count == 1 else "s"))
        source = self.source
        self._pending_hardware = settings
        self.hardware_jobs.start(lambda: source.configure(settings, previous))

    def _apply_software_channels(self):
        """A board without channel commands: log the choice and treat those channels as off."""
        settings = self.panel.unicorn_channels.settings()
        self.panel.unicorn_channels.show_applied(settings)
        if not (self.is_live and len(settings) == self.n_channels):
            self.panel.unicorn_channels.state.setText(
                self.panel.unicorn_channels.state.text() + " · applied at next Start")
            return
        self._applied_hardware = settings
        try:
            self.source.log_note("channels in use (software)", settings)
        except Exception:
            pass
        self.panel.channel_button.set_selected([i for i, s in enumerate(settings) if s.on])
        self._last_quality = -float('inf')           # show the change at the next frame
        self._refresh_off_channels(force=True)

    def _set_hardware_busy(self, busy):
        for button in (self.panel.hardware.apply_button, self.panel.hardware.resend_button,
                       self.panel.hardware.impedance_button, self.panel.hardware.short_button):
            button.setEnabled(not busy)

    def _hardware_applied(self, _result, error):
        self._configuring = False
        self._set_hardware_busy(False)
        if self.source is None:
            return
        if error:
            self.panel.hardware.show_pending("Apply failed · {}".format(error))
            self.status_changed.emit("hardware settings failed: {}".format(error))
            # What landed is unknown: the next Apply sends everything.
            self._applied_hardware = None
            return
        settings = self._pending_hardware
        self._applied_hardware = settings
        self._refresh_off_channels(force=True)
        # The amplifier state changed under the filters and quality history.
        self.stream = None
        self._reset_live_data()
        self.panel.channel_button.set_selected(self.panel.hardware.on_channels())
        self.panel.hardware.show_applied(settings)
        self.status_changed.emit(hw.describe(settings, self.panel.hardware.names))

    def _names_changed(self, names):
        """Remember a rename beside the file, and tell the feature window."""
        path = self.panel.selected_file
        if path and not self.panel.is_live:
            from ..formats import channels as ch
            ch.save_names(path, names)
            self.status_changed.emit("saved {}".format(ch.path_for(path).name))
        elif self.panel.board is CYTON_DAISY and self.player is None:
            # A live rename is the pin wiring: "Save current as profile" keeps it.
            self.panel.hardware.set_names(names)
        self.source_changed.emit(float(self.fs), list(names))

    # ---------------- channel profiles ----------------

    def _channel_profiles(self):
        return cp.all_profiles(self.preferences.value('channel_profiles/user', '') or '')

    def _profile(self, name):
        return next((p for p in self._channel_profiles() if p.name == name), None)

    def _restore_channel_profile(self):
        """Open with the last profile's pin names; the boxes keep what was last left."""
        profile = (self._profile(str(self.preferences.value('channel_profiles/last', '') or ''))
                   or cp.builtin_profiles()[0])
        self.panel.set_profiles([p.name for p in self._channel_profiles()], profile.name)
        self._use_profile_names(profile.names)

    def _use_profile_names(self, names):
        """Relabel the pins; the traces follow unless something else is streaming."""
        self.panel.hardware.set_names(names)
        streaming_other = self.player is not None or (
            self.source is not None and self.panel.board is not CYTON_DAISY)
        if streaming_other or len(names) != len(self.panel.channel_names):
            return
        self.panel.set_channels(names)
        self._trace_signature = None
        self.source_changed.emit(float(self.fs), list(names))

    def _profile_chosen(self, name):
        profile = self._profile(name)
        if profile is None:
            return
        self.preferences.setValue('channel_profiles/last', profile.name)
        self._use_profile_names(profile.names)
        self.panel.hardware.set_settings(profile.settings)
        board = self.panel.board
        if board is not None and board.channel_settings:
            self._apply_hardware()      # live: sent now; stopped: at the next Start
        self.status_changed.emit("channel profile: {} · {}".format(
            profile.name, hw.describe(profile.settings, profile.names)))

    def _save_profile(self):
        current = self.panel.profile_combo.currentText()
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save channel profile",
            "Pin names and On/BIAS boxes as they are now.\nProfile name:",
            text="" if current in cp.BUILTIN_NAMES else current)
        name = name.strip()
        if not ok or not name:
            return
        if name in cp.BUILTIN_NAMES:
            QtWidgets.QMessageBox.warning(self, "Save channel profile",
                                          "'{}' is built in; choose another name.".format(name))
            return
        profile = cp.ChannelProfile(name, list(self.panel.hardware.names),
                                    self.panel.hardware.all_settings())
        try:
            profile.validate()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Save channel profile", str(exc))
            return
        self._store_profile(profile)
        self.preferences.setValue('channel_profiles/last', name)
        self.panel.set_profiles([p.name for p in self._channel_profiles()], name)
        self.status_changed.emit("saved channel profile {}".format(name))

    def _store_profile(self, profile):
        """Add or replace one user profile (validated; built-in names refused)."""
        profile.validate()
        if profile.name in cp.BUILTIN_NAMES:
            raise ValueError("'{}' is built in; choose another name".format(profile.name))
        user = [p for p in cp.decode_user(self.preferences.value('channel_profiles/user', '') or '')
                if p.name != profile.name] + [profile]
        self.preferences.setValue('channel_profiles/user', cp.encode_user(user))

    def _forget_profile(self, name):
        user = cp.decode_user(self.preferences.value('channel_profiles/user', '') or '')
        self.preferences.setValue('channel_profiles/user',
                                  cp.encode_user([p for p in user if p.name != name]))

    def edit_profiles(self):
        """The profile editor: works with or without a board, in any mode."""
        from .profile_editor import ChannelProfileDialog
        monitor = self

        class Store:
            profiles = staticmethod(self._channel_profiles)
            save = staticmethod(self._store_profile)
            delete = staticmethod(self._forget_profile)

            @staticmethod
            def use(name):
                monitor.panel.set_profiles([p.name for p in monitor._channel_profiles()], name)
                monitor._profile_chosen(name)

        current = self.panel.profile_combo.currentText()
        dialog = ChannelProfileDialog(self, Store(), current)
        dialog.exec()
        # Names in the box may have changed (new, renamed, deleted).
        shown = self.panel.profile_combo.currentText()
        names = [p.name for p in self._channel_profiles()]
        if shown not in names:
            shown = cp.builtin_profiles()[0].name
            self._profile_chosen(shown)
        self.panel.set_profiles(names, shown)
        return dialog

    def _delete_profile(self):
        name = self.panel.profile_combo.currentText()
        if name in cp.BUILTIN_NAMES:
            self.status_changed.emit("'{}' is built in and cannot be deleted".format(name))
            return
        self._forget_profile(name)
        fallback = cp.builtin_profiles()[0].name
        self.panel.set_profiles([p.name for p in self._channel_profiles()], fallback)
        self._profile_chosen(fallback)

    def open_features(self):
        """Show the live feature window, creating it the first time."""
        if self.features_window is None:
            from .features_window import FeaturesWindow
            self.features_window = FeaturesWindow(self)
            self.features_window.attach(self)
            self.feature_layout.addWidget(self.features_window)
        self.feature_area.setVisible(not self.feature_area.isVisible())
        if self.feature_area.isVisible():
            self.chart_splitter.setSizes([430,270])

    def remap(self):
        """Re-answer a .mat's layout question, then pick the new answer up."""
        path = self.panel.selected_file
        if not path:
            self.status_changed.emit("no file selected")
            return

        running = self.timer.isActive()
        if running:
            self.stop()

        recording, note = remap(self, path)
        if note:
            self.status_changed.emit(note)
        if recording is None:
            return

        self.panel.set_sampling_rate(recording.fs)
        self.panel.set_channels(recording.channel_names)
        if running:
            self.start()

    # ---------------- transport ----------------

    def _toggle_play(self):
        if not self.player:
            return
        if self.player.at_end and not self.player.playing:
            self._seek(0)
        self.player.toggle()
        self._last_tick = time.perf_counter()
        self.transport.set_playing(self.player.playing)

    def _seek(self, sample):
        if not self.player:
            return
        self.player.seek_samples(sample)
        self._last_tick = time.perf_counter()
        self._new_samples = False
        self.sample_position = self.player.position
        self.feature_buffer = np.empty((self.n_channels,0))
        if self.features_window is not None:
            self.features_window.reset()
        self.filled = 0
        self.scaler.reset()
        self._reset_band_power()
        self._reset_live_data()
        # Time jumped; the filter state describes samples that no longer
        # precede what comes next, so carrying it over would ring at the seam.
        self.stream = None
        self.transport.update_position(self.player.position)
        if not self.player.playing:
            self._refill_after_seek()

    def _set_speed(self, speed):
        if self.player:
            self.player.speed = speed

    def _refill_after_seek(self):
        """While paused, show the window ending at the playhead."""
        length = max(self.buffer.shape[1], self.feature_capacity, round(self.fs * 4),
                     self._spectral_samples())
        start = max(self.player.position - length, 0)
        chunk = self.source.data[:, start:self.player.position]
        if chunk.shape[1]:
            self.sample_position = start
            self._ingest(chunk)
            self._render()

    # ---------------- buffer ----------------

    def _spectral_seconds(self):
        """Seconds the spectra are computed from (Settings → Monitor → Spectra).

        0 means "the display window" (Controls → Window): FFT and Welch PSD then
        cover the same seconds as the traces. A longer window gives a finer FFT
        grid (Δf = 1/T) and a slower reaction. The quality table keeps its own
        fixed 4 s of raw data.
        """
        from .preferences import setting
        chosen = float(setting('monitor/spectrum_seconds', 0.0) or 0.0)
        return chosen if chosen > 0 else float(self.panel.window_seconds.value())

    def _spectral_samples(self):
        return max(int(round(self._spectral_seconds() * self.fs)), 2)

    def _welch(self):
        """(segment seconds, overlap fraction) for the Monitor's PSD."""
        from .preferences import setting
        seconds = float(setting('monitor/welch_seconds', 2.0) or 2.0)
        overlap = min(max(float(setting('monitor/welch_overlap', 50.0)), 0.0), 95.0) / 100.0
        return max(seconds, 0.05), overlap

    def _apply_spectral_settings(self):
        """Use changed spectral settings now, streaming or not.

        Buffers keep what still fits and collect the rest; the smoothing
        restarts on the new frequency grid, so two grids are never mixed.
        """
        keep = self._spectral_samples()
        self.spectral_buffer = self.spectral_buffer[:, -keep:].copy()
        self.raw_spectral_buffer = self.raw_spectral_buffer[:, -keep:].copy()
        welch_seconds, overlap = self._welch()
        self.multi_spectrum.set_window_seconds(keep / self.fs, welch_seconds, overlap)
        self._invalidate_spectra()
        if hasattr(self, 'spectral_summary'):
            try:
                self.spectral_summary.setText(self._spectral_summary())
            except RuntimeError:        # the Settings page was closed and deleted
                del self.spectral_summary

    def _spectral_summary(self):
        seconds = self._spectral_samples() / self.fs
        welch_seconds, overlap = self._welch()
        segment = min(welch_seconds, seconds)
        step = segment * (1 - overlap)
        segments = 1 + int((seconds - segment) / step + 1e-9) if step > 0 else 1
        return ('Now: FFT of the last {:g} s → Δf {:.4g} Hz · PSD from {} segment(s) of {:g} s '
                '→ Δf {:.4g} Hz · at {:g} Hz sampling.'.format(
                    seconds, 1 / seconds, segments, segment, 1 / segment, self.fs))

    def _resize_buffer(self):
        length = max(int(self.panel.window_seconds.value() * self.fs), int(self.fs))
        self.buffer = np.zeros((self.n_channels, length))
        self.filled = 0
        self.scaler.reset()
        self._reset_band_power()
        # The spectra may follow the window (Settings → Monitor → Spectra).
        self._apply_spectral_settings()
        # Display length does not govern the independent feature buffer.
        if self.stream is not None and self.stream.n_channels != self.n_channels:
            self.stream = None

    def resize_feature_buffer(self, window):
        self.feature_capacity = int(round((window + 1) * self.fs))
        self.feature_buffer = np.empty((self.n_channels, 0))

    def _rebuild_stream(self):
        """Design the filter once, not twenty times a second.

        Rebuilding also restarts the filter state, which is correct: the old
        state belongs to the old filter and carrying it over would ring.
        """
        problem = self.panel.filters.problem()
        if problem:
            # Do not label results with controls that were never applied. Keep
            # collecting raw samples, but invalidate filtered data until fixed.
            self.stream = None
            self.filled = 0
            self.feature_buffer = np.empty((self.n_channels, 0))
            self.spectral_buffer = np.empty((self.n_channels, 0))
            self._invalidate_spectra()
            self.time_view.draw(self.buffer, [], 1, self.fs, self.panel.channel_names)
            if self.features_window is not None:
                self.features_window.reset()
                self.features_window.record.setChecked(False)
            self.status_changed.emit("Filtered views paused: " + problem)
            return
        sos = design_filter(self.fs, **self.panel.filter_kwargs)
        self.multi_spectrum.set_scale_exclusions(self.panel.filter_kwargs.get('notches') or ())
        self.stream = StreamFilter(sos, self.n_channels)
        self.filled = 0
        self.scaler.reset()
        self._trace_signature = None
        self._spectrum_signature = None
        self.feature_buffer = np.empty((self.n_channels,0))
        self._reset_band_power()
        self.spectral_buffer = np.empty((self.n_channels, 0))
        if self.multi_spectrum.input.currentText() == 'Filtered':
            self._invalidate_spectra()
        if self.features_window is not None:
            self.features_window.reset()
        # The state is deliberately left unset, to be seeded from the next raw
        # chunk. It describes the filter's INPUT history, and the buffer holds
        # its OUTPUT: seeding a state that expects a 37 mV electrode offset with
        # a 30 uV filtered value tells the filter the input just jumped by the
        # whole offset, and it answers with a 37 mV transient.

    def _reset_band_power(self):
        self.band_view.reset()
        self._band_signature = None
        self._spectrum_signature = None

    def _ingest(self, chunk):
        """Filter the arriving samples once, then store them."""
        if not chunk.shape[1]:
            return
        if self.player is None and self._last_arrival is not None and time.perf_counter() - self._last_arrival >= 2:
            self._reset_live_data()
            self.stream = None
        if self.stream is None and not self.panel.filters.problem():
            self._rebuild_stream()
        capacity4 = round(self.fs * 4)
        if self.raw_buffer.shape[0] != self.n_channels:
            self._reset_live_data()
        self.raw_buffer = np.concatenate((self.raw_buffer, chunk), axis=1)[:, -capacity4:].copy()
        spectral = self._spectral_samples()
        self.raw_spectral_buffer = np.concatenate((self.raw_spectral_buffer, chunk), axis=1)[:, -spectral:].copy()
        self.tap.push(self._as_the_board_gives(chunk), self.fs, self.panel.channel_names, self.source_label())
        self._last_arrival = time.perf_counter()
        if self.stream is None:
            self.sample_position += chunk.shape[1]
            self._new_samples = False
            return
        chunk = self.stream.process(chunk)   # seeds itself from the first chunk
        self.spectral_buffer = np.concatenate((self.spectral_buffer, chunk), axis=1)[:, -spectral:].copy()
        self.feature_buffer = np.concatenate((self.feature_buffer, chunk),axis=1)[:, -self.feature_capacity:]
        self.sample_position += chunk.shape[1]
        self._new_samples = bool(chunk.shape[1])

        n = chunk.shape[1]
        capacity = self.buffer.shape[1]
        if n >= capacity:
            self.buffer = chunk[:, -capacity:].copy()
            self.filled = capacity
        else:
            self.buffer = np.roll(self.buffer, -n, axis=1)
            self.buffer[:, -n:] = chunk
            self.filled = min(self.filled + n, capacity)

    # ---------------- frame ----------------

    def tick(self):
        # Qt coalesces overdue timer events; do not throw away an extra frame
        # after every slow frame. The elapsed clock already accounts for delays.
        started = time.perf_counter()
        dt = started - self._last_tick if self._last_tick is not None else self._frame_ms / 1000
        self._last_tick = started
        try:
            self._tick(dt)
        except Exception as exc:
            self.stop()
            self.connection_label.setText(f'Stream error · {exc} · reconnect with Start')
            self.status_changed.emit(str(exc))
            return
        # Python-side cost of the frame plus the gap since the previous one.
        # Shown on the status line so stutter on the operator's machine can be
        # read off as numbers instead of described.
        self._frame_costs.append((time.perf_counter() - started) * 1000)
        del self._frame_costs[:-20]
        self._frame_gaps.append(dt * 1000)
        del self._frame_gaps[:-20]
        self._keep_up()

    def _target_frame_ms(self):
        """Frame period for the chosen display rate (60 or 30 fps; anything else: 30)."""
        from .preferences import setting
        return DISPLAY_RATES.get(int(setting('monitor/display_fps', 30)), SMOOTH_FRAME_MS)

    def _keep_up(self):
        """Step down 60 -> 30 -> 20 fps when this computer cannot keep the rate.

        Judged on the last 20 frames, by medians so one slow frame (a spectrum
        arriving, the quality table) does not count: the real spacing between
        frames, which includes painting, 30 % late; or the Python work alone
        taking 80 % of the budget. It stays down until the next Start.
        """
        if self._frame_ms >= FRAME_MS or len(self._frame_gaps) < 20:
            return
        slow_work = np.median(self._frame_costs) > 0.8 * self._frame_ms
        late = np.median(self._frame_gaps) > 1.3 * self._frame_ms
        if slow_work or late:
            named = {FAST_FRAME_MS: 60, SMOOTH_FRAME_MS: 30, FRAME_MS: 20}
            before = named.get(self._frame_ms, round(1000 / self._frame_ms))
            self._frame_ms = SMOOTH_FRAME_MS if self._frame_ms < SMOOTH_FRAME_MS else FRAME_MS
            self.timer.setInterval(self._frame_ms)
            self._frame_costs, self._frame_gaps = [], []
            self.status_changed.emit("display at {} fps: this computer did not keep up with {} fps".format(
                named[self._frame_ms], before))

    def _tick(self, dt=None):
        if self.source is None:
            return
        if self.player:
            chunk = self.player.advance(FRAME_MS / 1000.0 if dt is None else dt)
            if chunk.shape[1]:
                self._ingest(chunk)
                self._ingest_motion(getattr(self.source, 'last_motion', None))
                self.transport.update_position(self.player.position)
            elif self.player.playing and self.player.at_end:
                self.player.pause()
                self.transport.set_playing(False)
                self.status_changed.emit("end of file")
        else:
            if self._configuring:
                # The board is stopped and BrainFlow serialises calls: a read here
                # would wait for the background Apply and freeze the window.
                return
            chunk = self.source.read(max(int(self.fs * self._frame_ms / 1000), 1))
            if self.impedance_sweep is not None:
                # Test-current samples are not EEG: keep them off every display.
                self._impedance_tick(chunk)
                return
            if self.short_capture is not None:
                # Internal-short samples are diagnostics, never EEG display data.
                self._internal_short_tick(chunk)
                return
            if chunk.shape[1]:
                self._ingest(chunk)
                self._ingest_motion(getattr(self.source, 'last_motion', None))

        self._render()

    # ---------------- impedance ----------------

    def _impedance_buttons(self, enabled):
        self.panel.hardware.impedance_button.setEnabled(enabled)
        self.panel.hardware.short_button.setEnabled(enabled)
        self.panel.hardware.apply_button.setEnabled(enabled)

    def start_impedance(self, channels=None):
        """Lead-off impedance of the ON channels, one at a time, on the live board."""
        from ..core import impedance as imp
        board = self.panel.board
        if board is not None and not board.channel_settings:
            self.status_changed.emit("{} has no impedance check in BrainFlow".format(board.label))
            return
        if not (self.panel.is_live and self.source is not None and self.player is None
                and self.timer.isActive()):
            self.status_changed.emit("impedance check needs a running live board (Start first)")
            return
        if self.impedance_sweep is not None or self.short_capture is not None or self._configuring:
            return
        applied = self._applied_hardware or hw.default_settings(self.n_channels)
        channels = list(channels) if channels is not None else [i for i, s in enumerate(applied) if s.on]
        if not channels:
            self.status_changed.emit("no channel is ON on the board")
            return
        self.impedance_sweep = imp.Sweep(channels, self.fs)
        self._impedance_buttons(False)
        self.quality_table.show_state('Impedance check')
        self._impedance_progress()
        source, first = self.source, channels[0]
        self._board_job(lambda: source.impedance_step(None, first, applied),
                        lambda _r, error: self._impedance_after_step("start", error))

    def _board_job(self, function, done):
        """Run a board command off the GUI thread; `done(result, error)` after.

        While it runs the tick does not read the board (BrainFlow serialises
        calls, so a read would wait for the command and freeze the window).
        """
        self._configuring = True
        self._board_job_done = done
        self.board_jobs.start(function)

    def _board_job_finished(self, result, error):
        self._configuring = False
        done, self._board_job_done = self._board_job_done, None
        if done is not None and self.source is not None:
            done(result, error)

    def _impedance_after_step(self, kind, error):
        from ..core import impedance as imp
        if self.impedance_sweep is None and kind != "done":
            return   # stopped meanwhile
        if error:
            self._impedance_failed(error)
            return
        self.source.read(0)   # drop samples from before the channel switched
        if kind in ("start", "switch"):
            self._impedance_progress()
            return
        sweep, self._finished_sweep = self._finished_sweep, None
        applied = self._applied_hardware or hw.default_settings(self.n_channels)
        names = self.panel.channel_names
        line = imp.summary(sweep.results, names)
        self._impedance_buttons(True)
        self.impedance_results = list(sweep.results)
        self._impedance_time = time.strftime("%H:%M:%S")
        self._show_impedance()
        self.source.log_note("impedance result: " + line, applied)
        self.status_changed.emit("impedance: " + line)
        self.stream = None
        self._reset_live_data()

    def _impedance_progress(self):
        sweep = self.impedance_sweep
        names = self.panel.channel_names
        text = "Impedance check · {} ({}/{}) · keep still".format(
            names[sweep.current], sweep.position + 1, len(sweep.channels))
        self.connection_label.setText(text)
        self.panel.hardware.impedance.setText(text)

    def _impedance_tick(self, chunk):
        sweep = self.impedance_sweep
        applied = self._applied_hardware or hw.default_settings(self.n_channels)
        try:
            action = sweep.feed(chunk)
        except Exception as exc:
            self._impedance_failed(exc)
            return
        if action is None:
            return
        kind, previous, following = action
        if kind == "done":
            # Leave impedance mode now; the result is shown once the channel
            # has its setting back.
            self._finished_sweep, self.impedance_sweep = sweep, None
        source = self.source
        self._board_job(lambda: source.impedance_step(previous, following, applied),
                        lambda _r, error: self._impedance_after_step(kind, error))

    def impedance_limits(self):
        """Green and yellow limits in ohms, as set in Settings."""
        h = self.panel.hardware
        return h.green_limit.value() * 1e3, h.yellow_limit.value() * 1e3

    def _show_impedance(self):
        from ..core import impedance as imp
        if not self.impedance_results:
            return
        green, yellow = self.impedance_limits()
        html, ready = imp.summary_html(self.impedance_results, self.panel.channel_names, green, yellow)
        stamp = getattr(self, '_impedance_time', '')
        self.panel.hardware.impedance.setText("Impedance {}<br>{}".format(stamp, html))
        self.impedance_ready = ready

    def _impedance_failed(self, exc):
        self.impedance_sweep = None
        self._impedance_buttons(True)
        try:   # put the operator's settings back whatever state the test left
            self.source.configure(self._applied_hardware or hw.default_settings(self.n_channels))
        except Exception:
            pass
        self.panel.hardware.impedance.setText("Impedance check failed · {}".format(exc))
        self.status_changed.emit("impedance check failed: {}".format(exc))
        self.stream = None
        self._reset_live_data()

    # ---------------- ADS1299 internal short ----------------

    def start_internal_short(self):
        """Record 20 s with every ADS1299 channel internally shorted."""
        from ..core import shorted_test
        board = self.panel.board
        if board is not None and not board.channel_settings:
            self.status_changed.emit("{} has no ADS1299 internal-short command".format(board.label))
            return
        if not (self.panel.is_live and self.source is not None and self.player is None
                and self.timer.isActive()):
            self.status_changed.emit("internal short needs a running Cyton+Daisy (Start first)")
            return
        if self.impedance_sweep is not None or self.short_capture is not None or self._configuring:
            return
        applied = self._applied_hardware or hw.default_settings(self.n_channels)
        channels = list(range(self.n_channels))
        self.short_capture = shorted_test.Capture(channels, self.fs)
        self._impedance_buttons(False)
        self.quality_table.show_state("ADS1299 internal short")
        self.panel.hardware.short_result.setText("Internal short · configuring board…")
        source = self.source
        self._board_job(lambda: source.internal_short(True, applied), self._internal_short_started)

    def _internal_short_started(self, _result, error):
        if self.short_capture is None:
            return
        if error:
            self._internal_short_failed(error)
            return
        self.source.read(0)
        self._internal_short_progress()

    def _internal_short_progress(self):
        capture = self.short_capture
        if capture is None:
            return
        seconds = min(capture.progress * 20.0, 20.0)
        text = "ADS1299 internal short · {:.1f}/20.0 s · raw, filters bypassed".format(seconds)
        self.connection_label.setText(text)
        self.panel.hardware.short_result.setText(text)

    def _internal_short_tick(self, chunk):
        capture = self.short_capture
        try:
            finished = capture.feed(chunk)
        except Exception as exc:
            self._internal_short_failed(exc)
            return
        self._internal_short_progress()
        if not finished:
            return
        self.short_capture = None
        self._finished_short_capture = capture
        source = self.source
        applied = self._applied_hardware or hw.default_settings(self.n_channels)
        self._board_job(lambda: source.internal_short(False, applied), self._internal_short_finished)

    def _internal_short_finished(self, _result, error):
        capture = getattr(self, "_finished_short_capture", None)
        self._finished_short_capture = None
        if error:
            self._internal_short_failed(error)
            return
        from ..core import shorted_test
        self._impedance_buttons(True)
        self.short_results = list(capture.results)
        html = shorted_test.summary_html(self.short_results, self.panel.channel_names)
        line = shorted_test.summary(self.short_results, self.panel.channel_names)
        stamp = time.strftime("%H:%M:%S")
        self.panel.hardware.short_result.setText("Internal short {}<br>{}".format(stamp, html))
        self.panel.hardware.short_result.setToolTip(line)
        applied = self._applied_hardware or hw.default_settings(self.n_channels)
        self.source.log_note("internal short result: " + line, applied)
        clean = sum(result.status == "Clean" for result in self.short_results)
        self.status_changed.emit("internal short: {}/{} channels clean".format(clean, len(self.short_results)))
        self.stream = None
        self._reset_live_data()

    def _internal_short_failed(self, exc):
        self.short_capture = None
        self._finished_short_capture = None
        self._impedance_buttons(True)
        try:
            self.source.configure(self._applied_hardware or hw.default_settings(self.n_channels))
        except Exception:
            pass
        self.panel.hardware.short_result.setText("Internal short failed · {}".format(exc))
        self.status_changed.emit("internal short failed: {}".format(exc))
        self.stream = None
        self._reset_live_data()

    def _render(self):
        self._update_live_views()
        if self.multi_spectrum.isVisible():
            self.multi_spectrum.animate()
        if self.detail.isVisible():
            self._draw_detail()
        if (getattr(self, '_new_motion', False) and self.motion_buffer is not None
                and self.motion_view.isVisible() and self.motion_view.toggle.isChecked()):
            self.motion_view.draw(self.motion_buffer, self.fs)
            self._new_motion = False
            self._align_motion()
            self._shade_movement()
        if self.filled < 10:
            return

        # Already filtered on the way in. Re-filtering the window here is what
        # made the trace shimmer: filtfilt's edge transients moved with the
        # window. The mean subtraction went with it - the high-pass removes DC,
        # and re-estimating the mean every frame shifted the whole trace.
        filtered = self.buffer[:, -self.filled:]

        names = self.panel.channel_names
        channels = self._drawn_channels()
        hidden = len(self.panel.selected_channels) - len(channels)
        note = ' · {} switched off (hidden)'.format(hidden) if hidden else ''
        signature = (self.sample_position,self.filled,tuple(channels),tuple(names),self.fs,
                     self.panel.auto_amplitude.isChecked(),self.panel.amplitude.value(),note)
        # The widget's own visibility, not the page's: a popped-out view keeps
        # updating while the main window shows another tab.
        if self.time_scroll.isVisible() and signature != self._trace_signature:
            self.time_view.setMinimumHeight(max(160,80+20*len(channels)))
            self.time_view.draw(filtered, channels,
                                self._amplitude(filtered, channels), self.fs, names, note,
                                position=self.sample_position, window=self.buffer.shape[1])
            self._trace_signature = signature

        if self._new_samples:
            self.frame_ready.emit(filtered, float(self.fs), names)
            self.features_ready.emit(self.feature_buffer, float(self.fs), names, self.sample_position / self.fs)
            self._new_samples = False

    def _shade_movement(self):
        """Mark the moments the headset moved on the traces themselves."""
        movement = self.motion_view.movement_of(self.motion_buffer)
        if not movement.size:
            self.time_view.shade_spans([])
            return
        # The traces show the last `filled` samples; the motion buffer may hold
        # a different number, so spans are measured from the end of both.
        offset = (movement.size - self.filled) / self.fs
        spans = [(a - offset, b - offset) for a, b in movement_spans(movement, self.fs)]
        self.time_view.shade_spans([(a, b) for a, b in spans if b > 0])

    def _align_motion(self):
        """Line the motion plots up under the traces: same left edge, right edge and span."""
        if not self.motion_view.isVisible():
            return
        view = self.time_view
        box = view.plotItem.vb.sceneBoundingRect()
        left = view.mapToGlobal(view.mapFromScene(box.topLeft())).x()
        right = view.mapToGlobal(view.mapFromScene(box.topRight())).x()
        self.motion_view.align_to(left, right, view.plotItem.vb.viewRange()[0])

    def _configure_motion(self):
        """Show the source's motion axes; anything else counts as no sensor.

        A display extra: whatever a source reports here must never stop the stream.
        """
        names = getattr(self.source, 'motion_names', None)
        units = getattr(self.source, 'motion_units', None)
        ok = (isinstance(names, (list, tuple)) and isinstance(units, (list, tuple))
              and len(names) == len(units) and all(isinstance(n, str) for n in names))
        self.motion_view.configure(names if ok else [], units if ok else [])

    def _ingest_motion(self, block):
        """Keep the motion sensor's newest samples, as long as the trace window."""
        if (not isinstance(block, np.ndarray) or block.ndim != 2 or not block.size
                or block.shape[0] != len(self.motion_view.names)):
            return
        block = block.astype(float, copy=False)
        capacity = self.buffer.shape[1]
        if self.motion_buffer is None or self.motion_buffer.shape[0] != block.shape[0]:
            self.motion_buffer = np.empty((block.shape[0], 0))
        self.motion_buffer = np.concatenate((self.motion_buffer, block), axis=1)[:, -capacity:]
        self._new_motion = True

    def _reset_live_data(self):
        self.motion_buffer = None
        self._new_motion = False
        self.raw_buffer = np.empty((self.n_channels, 0))
        self.raw_spectral_buffer = np.empty((self.n_channels, 0))
        self.spectral_buffer = np.empty((self.n_channels, 0))
        self.quality_tracker.reset()
        self._last_quality = -float('inf'); self._quality_position = None
        self._last_arrival = None
        self._waiting_since = time.perf_counter()
        self.quality_table.set_channels(self.panel.channel_names)
        self.quality_table.show_state('Collecting data')
        self._invalidate_spectra()
        self._reset_band_power()

    def _invalidate_spectra(self):
        self.spectral_jobs.invalidate()
        self._spectral_revision += 1
        self._job_position = None; self._last_job = -float('inf')
        self._latest_snapshot = None
        self.multi_spectrum.reset()
        self.band_view.reset()
        self._band_signature = None
        self.spectrum_view._curve.setData([], [])

    def _update_live_views(self):
        now = time.perf_counter()
        self._refresh_off_channels()
        names = self.panel.channel_names
        self.quality_table.set_channels(names)
        config = (tuple(names), tuple(self.panel.selected_channels), self.fs)
        if config != getattr(self, '_multi_config', None):
            self.multi_spectrum.configure(names, self.panel.selected_channels, self.fs)
            self._multi_config = config
        age = None if self._last_arrival is None else max(0, now - self._last_arrival)
        stale = self.source is not None and self.player is None and (
            now - self._waiting_since >= 2 if age is None else age >= 2)
        if self.source is None:
            state = 'Stopped · graphs stale'
        elif self.player:
            state = 'End of file' if self.player.at_end else 'Playback' if self.player.playing else 'Paused'
        else:
            state = 'No data · Stale' if stale else 'Collecting data' if age is None else 'Live'
        age_text = '—' if age is None else f'{age:.1f} s ago'
        timing = (' · frame {:.0f} ms, max gap {:.0f} ms (target {})'.format(
            max(self._frame_costs), max(self._frame_gaps), self._frame_ms)
            if self._frame_costs else '')
        self.connection_label.setText(
            f'{state} · {self.fs:g} Hz · {self.n_channels} channels · last data {age_text}{timing}')
        self.detail.setWindowTitle(f'Channel detail · {state}')
        if stale:
            self.quality_table.show_state('Missing data')
            if not getattr(self, '_was_stale', False):
                self.quality_tracker.reset()
                self._invalidate_spectra()
            self._was_stale = True
            return
        if getattr(self, '_was_stale', False):
            self._last_quality = -float('inf')
        self._was_stale = False
        if self.source is None:
            return
        if now - self._last_quality >= 1 and self.sample_position != self._quality_position:
            flat, high, mains, saturated = [control.value() for control in self.quality_controls]
            limits = QualityLimits(flat, high, mains / 100, saturated / 100)
            quality = quality_snapshot(self.raw_buffer, self.fs, self.sample_position / self.fs, self._rail_uv, limits)
            values = self.quality_tracker.update(quality)
            off = self._off_channels
            if off:
                nan = float('nan')
                values = [replace(v, status='Switched off', rms=nan, saturation=nan, mains=nan, same_as=-1)
                          if v.channel in off else v for v in values]
            self.quality_table.update_values(values)
            self._last_quality = now; self._quality_position = self.sample_position
        # At most one running task, no growing queue. The next timer tick uses
        # the newest available window; obsolete revisions never reach widgets.
        if self.spectral_jobs.active or now - self._last_job < .5:
            return
        if self._job_position == self.sample_position:
            return
        data = self.raw_spectral_buffer if self.multi_spectrum.input.currentText() == 'Raw' else self.spectral_buffer
        if data.shape[1] < self._spectral_samples():
            return
        # The page (spectra stay ready for a tab switch), a popped-out spectrum
        # window, or the channel detail: skip only when none is on screen.
        if not (self.isVisible() or self.multi_spectrum.isVisible() or self.detail.isVisible()):
            return
        block, fs, position = data.copy(), self.fs, self.sample_position
        welch_seconds, overlap = self._welch()
        self._job_position = position; self._last_job = now
        self.spectral_jobs.start(lambda: (spectra_snapshot(block, fs, welch_seconds, overlap), position, now))

    def _spectra_ready(self, result, error):
        if error:
            self.status_changed.emit(f'Spectrum unavailable: {error}')
            return
        snapshot, position, now = result
        if snapshot is None or self.source is None:
            return
        self._latest_snapshot = snapshot
        self.multi_spectrum.update_snapshot(snapshot, now)
        self._draw_detail(now)

    def _draw_detail(self, now=None):
        snapshot = getattr(self, '_latest_snapshot', None)
        if snapshot is None or not self.detail.isVisible():
            return
        index = self.panel.psd_channel.currentIndex()
        if not 0 <= index < len(snapshot['valid']):
            return
        if index in self._off_channels:
            self.spectrum_view._curve.setData([], [])
            self.band_view.reset()
            self.spectrum_view.setTitle('{} · switched off'.format(self.panel.channel_names[index]))
            return
        if not snapshot['valid'][index]:
            self.spectrum_view._curve.setData([], [])
            self.band_view.reset()
            return
        if self._band_signature != index:
            self.band_view.reset(); self._band_signature = index
        panel = self.multi_spectrum
        kind = panel.mode.currentText()
        freqs = panel._freqs
        # The same numbers the panel draws: gliding between updates, and in the
        # panel's mode. Before, the detail redrew in 0.5 s jumps from the raw
        # PSD and refitted its axis every time.
        source = panel._shown if panel._shown is not None else panel._values
        if freqs is None or source is None or index >= source.shape[0]:
            return
        values = panel._display(source[index:index + 1])[0]
        self.spectrum_view._curve.setPen(channel_pen(index))
        self.spectrum_view.draw(freqs, values, panel.log.isChecked(), self.panel.channel_names[index],
                                y_range=panel._bounds)
        self.spectrum_view.setLabel('left', panel.unit_label())
        self.spectrum_view.setTitle(f'{self.panel.channel_names[index]} · {kind} · {panel.input.currentText()}')
        self.spectrum_view.setXRange(*panel.frequency_range(), padding=0)
        self.band_view.draw(band_powers(snapshot['psd_freqs'], snapshot['psd'][index]), now=now)

    def _amplitude(self, filtered, channels):
        """Display-only scales: each visible channel keeps its own history."""
        if not self.panel.auto_amplitude.isChecked():
            return self.panel.amplitude.value()
        return self.scaler.update(filtered, channels)
