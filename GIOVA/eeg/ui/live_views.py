"""Operator quality table and persistent multi-channel spectrum plots."""
import math
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .plots import CHANNEL_COLORS, channel_pen, value_axis
from .widgets import NoScrollComboBox, NoScrollSpinBox, WrapRow
from ..core.monitoring import MAINS_HZ, MOVEMENT_MG, mains_bands, movement_mg
from ..core.spectral import PSD_UNITS, line_frequencies, neighbour_snr, psd_in_unit
from .preferences import psd_unit, user_settings
from .scaling import HeldScale
from .theme import color, is_dark


class ChannelViewBox(pg.ViewBox):
    clicked = QtCore.Signal()

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mouseClickEvent(event)


#: (background, text) per status. Colour sits in the cell background with dark
#: text on it; coloured text alone was unreadable on dark and dim rows.
STATUS_STYLE = {
    'No issues detected': ('#d9f2e1', '#14532d'),
    'Check signal': ('#ffedb8', '#6b4400'),
    'Line noise': ('#dde8fb', '#1e3a8a'),
    'Flat': ('#f9d3ce', '#8a1c12'),
    'Identical signal': ('#f9d3ce', '#8a1c12'),
    'Switched off': ('#e4e7eb', '#616e7c'),
    'Saturated': ('#f9d3ce', '#8a1c12'),
    'Missing data': ('#f9d3ce', '#8a1c12'),
}
NEUTRAL_STYLE = ('#eef1f4', '#3c4650')
#: The same under the dark theme: a dark tint of the colour, the text a light one.
_GOOD, _CHECK, _HUM, _BAD = ('#16351f', '#8fe0a8'), ('#3d3112', '#f3d27a'), ('#1c2a47', '#a9c4ff'), ('#43201c', '#f6a99f')
STATUS_STYLE_DARK = {
    'No issues detected': _GOOD, 'Check signal': _CHECK, 'Line noise': _HUM, 'Flat': _BAD,
    'Identical signal': _BAD, 'Switched off': ('#262b33', '#98a2ae'), 'Saturated': _BAD, 'Missing data': _BAD,
}
NEUTRAL_STYLE_DARK = ('#20252c', '#c3cad3')
#: How the spectra's y axis is decided.
AUTO, PEAK, MANUAL = 'auto', 'peak', 'manual'
#: New spectra arrive twice a second. Drawn as they come, every curve jumped
#: twice a second beside traces moving twenty times a second, which read as
#: stutter. The drawn curves glide towards the newest values instead, with this
#: time constant, redrawn at most every ANIMATE_S: five steps a second. With
#: the full Monitor at 16 channels (offscreen) frame p95 went 21 -> 14 ms, no
#: frame over 50 ms; ten steps a second pushed p95 to 39 ms.
GLIDE_S = 0.3
ANIMATE_S = 0.2
#: Half-width of the band left out of the automatic scale around a mains line.
LINE_WIDTH_HZ = 1.5


def _swatch(color, size=12):
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtGui.QColor(color))
    return QtGui.QIcon(pixmap)


class QualityTable(QtWidgets.QTableWidget):
    channel_clicked = QtCore.Signal(int)

    def __init__(self):
        super().__init__(0, 5)
        self.setHorizontalHeaderLabels(['Channel', 'Status', 'EEG µV', 'Rail %', ''])
        for col, tip in enumerate([
                'Click a channel for its spectrum.',
                'Saturated: >15% of samples at the rail · Flat: EEG µV < 0.1 · '
                'Check signal: EEG µV > 100 · Identical signal: same waveform and offset as another '
                'channel (electrodes not touching the head, or bridged) · Line noise: mains % > 20. '
                'Thresholds in Settings.',
                'RMS of the 1–45 Hz band (last 4 s, raw data, linear drift removed). '
                'Awake scalp EEG is typically tens of µV; above ~100 µV is usually artefact.',
                'Share of samples within 5% of the amplifier rail (Cyton ±187,500 µV, '
                'Unicorn ±750,000 µV). '
                'A railed channel has no usable signal: the electrode is not on skin.']):
            self.horizontalHeaderItem(col).setToolTip(tip)
        self._mains = None
        self.set_mains(MAINS_HZ)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(27)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setSortingEnabled(False)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        for i, width in enumerate([65, 137, 65, 58, 65]):
            self.setColumnWidth(i, width)
        self.horizontalHeader().setStretchLastSection(True)
        self.cellClicked.connect(lambda row, col: self.channel_clicked.emit(row))
        self._names = None
        self.setToolTip('Quality uses the last 4 s of raw data, before display filters. '
                        'RMS is mean-removed. These are signal checks, not impedance measurements.')

    def set_mains(self, hz):
        """The last column measures the line noise at `hz` (the filters' Mains: 50 or 60 Hz)."""
        if hz == self._mains:
            return
        self._mains = hz
        (low, high), (total_low, total_high) = mains_bands(hz)
        item = self.horizontalHeaderItem(4)
        item.setText('{:g} Hz %'.format(hz))
        item.setToolTip('{:g}–{:g} Hz power as a share of {:g}–{:g} Hz power. Mains pick-up from cables '
                        'or poor contact; the notch filter removes it from the display. 50 or 60 Hz follows '
                        'the Mains chosen in the filters.'.format(low, high, total_low, total_high))

    def set_channels(self, names):
        if tuple(names) == self._names:
            return
        self._names = tuple(names)
        self.setRowCount(len(names))
        for row, name in enumerate(names):
            for col, text in enumerate([name, 'Collecting data', '—', '—', '—']):
                self.setItem(row, col, QtWidgets.QTableWidgetItem(text))
            channel = self.item(row, 0)
            channel.setIcon(_swatch(CHANNEL_COLORS[row % len(CHANNEL_COLORS)]))
            font = channel.font(); font.setBold(True); channel.setFont(font)
            self._style_status(row, None)

    def _style_status(self, row, status):
        styles, neutral = (STATUS_STYLE_DARK, NEUTRAL_STYLE_DARK) if is_dark() else (STATUS_STYLE, NEUTRAL_STYLE)
        background, text = styles.get(status, neutral)
        item = self.item(row, 1)
        key = status if status in STATUS_STYLE else 'neutral'
        if item.data(QtCore.Qt.ItemDataRole.UserRole) == key:
            return  # restyling unchanged cells every second only costs repaints
        item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
        item.setBackground(QtGui.QColor(background))
        item.setForeground(QtGui.QColor(text))

    def update_values(self, values):
        for value in values:
            row = value.channel
            if row >= self.rowCount():
                continue
            fields = [value.status] + [f'{v * scale:.1f}' if np.isfinite(v) else '—'
                                       for v, scale in [(value.rms, 1), (value.saturation, 100), (value.mains, 100)]]
            note = value.status
            if value.status == 'Switched off':
                note = 'Powered down on the board (hardware settings or the game): not measuring'
            if value.status == 'Identical signal':
                partner = (self._names[value.same_as] if 0 <= value.same_as < len(self._names)
                           else 'another channel')
                note = (f'Same waveform and offset as {partner}: the two inputs are one electrical '
                        f'node - electrodes not touching the head, or bridged by gel or sweat')
            for col, text in enumerate(fields, 1):
                self.item(row, col).setText(text)
                self.item(row, col).setToolTip(f'{note} · data time {value.data_time:.2f} s')
            self._style_status(row, value.status)

    def show_state(self, state):
        for row in range(self.rowCount()):
            self.item(row, 1).setText(state)
            self._style_status(row, None)
            for col in range(2, 5):
                self.item(row, col).setText('—')


class SpectrumPanel(QtWidgets.QWidget):
    channel_clicked = QtCore.Signal(int)
    computation_changed = QtCore.Signal()
    visibility_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        self.mode = NoScrollComboBox(); self.mode.addItems(['FFT', 'PSD', 'SNR'])
        self.mode.setToolTip('FFT: amplitude, µV · PSD: power density, µV²/Hz (or the unit chosen in '
                             'Settings → Graphs) · SNR: power over its '
                             'neighbours 0.75-2.5 Hz away (1 = no peak) - for flicker (SSVEP) responses')
        #: what the PSD mode draws: Settings → Graphs
        self.psd_unit = psd_unit()
        self.arrangement = NoScrollComboBox(); self.arrangement.addItems(['Grid', 'Overlay'])
        self.input = NoScrollComboBox(); self.input.addItems(['Filtered', 'Raw'])
        self.log = QtWidgets.QCheckBox('Log'); self.log.setChecked(True)
        # The band drawn. Zooming in on 12-18 Hz is how a flicker response is
        # read; before, only the upper edge could be moved.
        self.minimum = NoScrollSpinBox(); self.minimum.setRange(0, 998); self.minimum.setValue(1)
        self.minimum.setSuffix(' Hz'); self.minimum.setDecimals(0)
        self.minimum.setToolTip('Lowest frequency drawn')
        self.maximum = NoScrollSpinBox(); self.maximum.setRange(2, 1000); self.maximum.setValue(60)
        self.maximum.setSuffix(' Hz'); self.maximum.setDecimals(0)
        self.maximum.setToolTip('Highest frequency drawn')
        self.scale = NoScrollComboBox()
        for label, key in (('Auto', AUTO), ('Peak', PEAK), ('Manual', MANUAL)):
            self.scale.addItem(label, key)
        self.scale.setToolTip('Auto: fits the EEG, letting mains lines (50 Hz, and 25 Hz where they '
                              'fold back at 125 Hz) run off the top · Peak: fits the tallest line · '
                              'Manual: the top you type')
        self.scale_top = NoScrollSpinBox()
        self.scale_top.setRange(1e-6, 1e9); self.scale_top.setDecimals(2); self.scale_top.setValue(100)
        self.scale_top.setToolTip('Top of the y axis')
        self.scale_top.setVisible(False)
        self._held = HeldScale(log=True)
        # Display-only channel filter for the spectra. Nothing here reaches the
        # board; hardware power and BIAS live in Settings.
        self.hidden = set()      # channel names hidden from the spectra
        self.off = ()            # channel names the board has switched off
        self.channel_menu_button = QtWidgets.QToolButton()
        self.channel_menu_button.setText('Channels')
        self.channel_menu_button.setToolTip('Show or hide channels in the spectrum plots only (display, not hardware).')
        self.channel_menu_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._channel_menu = QtWidgets.QMenu(self)
        self._channel_menu.aboutToShow.connect(self._fill_channel_menu)
        self.channel_menu_button.setMenu(self._channel_menu)
        # What is changed while watching stays in view: the kind of spectrum and
        # the channels. The rest is set once and sits behind Options, which
        # names what it holds so nothing hidden is a surprise.
        self.options_button = QtWidgets.QToolButton()
        self.options_button.setCheckable(True)
        self.options_button.setMinimumWidth(80)     # its long name may be cut on a narrow window; tooltip below
        self.options_button.setToolTip('Show or hide the spectrum options: grid or overlay, filtered or raw '
                                       'signal, log scale, the frequencies drawn and the y axis')
        for widget in [self.mode, self.channel_menu_button]:
            controls.addWidget(widget)
        controls.addStretch(1)
        controls.addWidget(self.options_button)
        layout.addLayout(controls)
        self.options_row = QtWidgets.QWidget()
        # on a narrow window the options go on to a second line rather than widening the window
        options = WrapRow(self.options_row)
        options.setContentsMargins(0, 0, 0, 0)
        for widget in [self.arrangement, self.input, self.log, self.minimum, self.maximum,
                       self.scale, self.scale_top]:
            options.add(widget)
        layout.addWidget(self.options_row)
        shown = user_settings().value('spectrum_options_shown', False, type=bool)
        self.options_button.setChecked(shown)
        self.options_row.setVisible(shown)
        self.options_button.toggled.connect(self._options_toggled)
        for signal in (self.arrangement.currentIndexChanged, self.input.currentIndexChanged, self.log.toggled,
                       self.minimum.valueChanged, self.maximum.valueChanged, self.scale.currentIndexChanged):
            signal.connect(self._name_options)
        self._name_options()
        self.base_caption = ''
        self.set_window_seconds(5.0, refresh=False)       # replaced by the Monitor at once
        self.caption = QtWidgets.QLabel(self.base_caption)
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.scroll = QtWidgets.QScrollArea(); self.scroll.setWidgetResizable(True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.scroll.setWidget(self.graphics)
        layout.addWidget(self.scroll, 1)
        self._plots = []; self._curves = []; self._layout_key = None; self._axes_key = None
        self._values = None; self._freqs = None; self._names = []; self._channels = []
        self._bounds = None; self._last_time = None; self._fs = 125
        self._shown = None; self._last_animation = None
        #: notch frequencies: filtered spectra keep a residue there that must
        #: not set the scale (after a 50 Hz notch, 166 µV²/Hz of mains beside
        #: a 6 µV²/Hz SSVEP line on 18 Sept - the EEG lay flat on the axis)
        self._scale_exclusions = ()
        self._available = []
        self.mode.currentIndexChanged.connect(self._reset_computation)
        self.input.currentIndexChanged.connect(self._reset_computation)
        self.arrangement.currentIndexChanged.connect(self.redraw)
        self.log.toggled.connect(self.fit_scale)
        self.minimum.valueChanged.connect(self._range_changed)
        self.maximum.valueChanged.connect(self._range_changed)
        self.scale.currentIndexChanged.connect(self._scale_mode_changed)
        self.scale_top.valueChanged.connect(self.fit_scale)

    def _options_toggled(self, shown):
        self.options_row.setVisible(shown)
        user_settings().setValue('spectrum_options_shown', bool(shown))
        self._name_options()

    def _name_options(self, *_):
        """The Options button says what is set behind it."""
        self.options_button.setText('Options · {} · {} · {:g}–{:g} Hz{} {}'.format(
            self.arrangement.currentText(), self.input.currentText(), self.minimum.value(),
            self.maximum.value(), ' · log' if self.log.isChecked() else '',
            '▴' if self.options_button.isChecked() else '▾'))

    def _reset_computation(self):
        self.reset()
        self.computation_changed.emit()

    def reset(self):
        self._values = None; self._freqs = None; self._last_time = None; self._bounds = None
        self._shown = None
        for curve in self._curves:
            curve.setData([], [])

    def configure(self, names, channels, fs):
        self._names, self._available, self._fs = list(names), list(channels), fs
        self.maximum.setMaximum(max(2, fs / 2))
        self._apply_filter()

    # ---- display-only channel filter ----

    def _apply_filter(self):
        skip = self.hidden | set(self.off)
        self._channels = [c for c in self._available
                          if c < len(self._names) and self._names[c] not in skip]
        shown, total = len(self._channels), len(self._available)
        self.channel_menu_button.setText('Channels' if shown == total else f'Channels {shown}/{total}')
        drawn = [self._names[c] for c in self._available if c < len(self._names)]
        off = [name for name in self.off if name in drawn]
        self.caption.setText(self.base_caption + (
            ' · {} switched off (hidden): {}'.format(len(off), ', '.join(off)) if off else ''))
        self._bounds = None   # refit to what is left, not to a hidden loud channel
        self.redraw()

    def set_off(self, names):
        """Channels the board has switched off: a flat line says nothing here."""
        names = tuple(names)
        if names != self.off:
            self.off = names
            self._apply_filter()

    def set_hidden(self, names):
        self.hidden = set(names)
        self._apply_filter()
        self.visibility_changed.emit()

    def show_only(self, names):
        self.set_hidden(n for n in self._names if n not in names)

    def _fill_channel_menu(self):
        """Rebuilt on every opening, so it always lists the current recording's
        channels. Checkboxes sit in a widget action: ticking one does not close
        the menu, so several can be changed in one go."""
        menu = self._channel_menu
        menu.clear()
        box = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 6, 8, 6)
        presets = QtWidgets.QHBoxLayout()
        for text, callback in [('All', lambda: self.set_hidden([])),
                               ('None', lambda: self.set_hidden(self._names)),
                               ('Head map…', self._choose_on_head)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(menu.close)
            button.clicked.connect(callback)
            presets.addWidget(button)
        grid.addLayout(presets, 0, 0, 1, 4)
        available = set(self._available)
        for i, name in enumerate(self._names):
            switched_off = name in self.off
            check = QtWidgets.QCheckBox(name + ' (off)' if switched_off else name)
            check.setChecked(name not in self.hidden and not switched_off)
            check.setIcon(_swatch(CHANNEL_COLORS[i % len(CHANNEL_COLORS)]))
            # Channels off in the Graphs selection, or powered down on the
            # board, cannot be shown here either.
            check.setEnabled(i in available and not switched_off)
            check.toggled.connect(lambda on, n=name: self.set_hidden(
                self.hidden - {n} if on else self.hidden | {n}))
            grid.addWidget(check, 1 + i // 4, i % 4)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(box)
        menu.addAction(action)

    def _choose_on_head(self):
        """Choose the shown spectra on the head map instead of a fixed pair."""
        from .headmap import ChannelSelectorDialog
        shown = [i for i, name in enumerate(self._names) if name not in self.hidden]
        chosen = ChannelSelectorDialog.ask(self, self._names, shown)
        if chosen is not None:
            self.show_only([self._names[i] for i in chosen])

    def set_window_seconds(self, seconds, welch_seconds=2.0, overlap=0.5, refresh=True):
        """Say which stretch of signal the spectra are computed from."""
        self.base_caption = ('FFT/PSD of the last {:g} s (FFT Δf {:g} Hz) · Welch {:g} s, {:g} % overlap · '
                             '2 s smoothing · click a channel for details'.format(
                                 round(seconds, 3), round(1.0 / seconds, 4), welch_seconds, overlap * 100))
        if refresh:
            self.caption.setText(self.base_caption)

    def update_snapshot(self, snapshot, now):
        # SNR is worked out from the smoothed PSD when drawn: a ratio of
        # averaged powers is steadier than an average of noisy ratios.
        kind = 'psd' if self.mode.currentText() == 'SNR' else self.mode.currentText().lower()
        values, freqs = snapshot[kind], snapshot[kind + '_freqs']
        if self._values is None or self._values.shape != values.shape or not np.array_equal(freqs, self._freqs):
            self._values = values.copy()
            self._shown = None
        else:
            weight = -math.expm1(-min(max(now - self._last_time, 0), .5) / 2)
            valid = np.isfinite(values)
            previous = np.isfinite(self._values)
            self._values = np.where(valid & previous, self._values + weight * (values - self._values), values)
        self._freqs, self._last_time = freqs, now
        self.redraw()

    def _display(self, values):
        """What a row of the smoothed spectrum is drawn as, in the current mode."""
        if self.mode.currentText() == 'SNR':
            return neighbour_snr(self._freqs, values)
        if self.mode.currentText() == 'PSD':
            return psd_in_unit(self._freqs, values, self.psd_unit)
        return values

    def unit_label(self):
        """The y axis unit of the current mode."""
        return {'FFT': 'µV', 'PSD': PSD_UNITS[self.psd_unit], 'SNR': 'SNR (×)'}[self.mode.currentText()]

    def set_psd_unit(self, unit):
        """Draw the PSD in another unit; the smoothed densities are kept."""
        if unit != self.psd_unit and unit in PSD_UNITS:
            self.psd_unit = unit
            self.fit_scale()

    def animate(self, now=None):
        """Glide the drawn curves towards the newest spectra. Call every frame."""
        if self._values is None or self._shown is None or not self._curves:
            return
        now = time.perf_counter() if now is None else now
        last = self._last_animation
        # 10 % slack: frames 50 ms apart land at 0.19999 s as often as at 0.2.
        if last is not None and 0 <= now - last < 0.9 * ANIMATE_S:
            return
        # A clock that went backwards starts afresh rather than never moving.
        dt = ANIMATE_S if last is None or now < last else min(now - last, 1.0)
        self._last_animation = now
        valid = np.isfinite(self._values) & np.isfinite(self._shown)
        step = -math.expm1(-dt / GLIDE_S)
        moved = np.where(valid, self._shown + step * (self._values - self._shown), self._values)
        scale = np.nanmax(np.abs(self._values)) if np.isfinite(self._values).any() else 0.0
        if scale > 0 and np.nanmax(np.abs(moved - self._shown)) < 1e-4 * scale:
            return                      # settled: nothing visible would change
        self._shown = moved
        self._draw_curves()

    def frequency_range(self):
        """The band drawn, inside what this sampling rate can show."""
        top = min(self.maximum.value(), self._fs / 2)
        low = min(self.minimum.value(), max(0.0, top - 1))
        return low, max(top, low + 1)

    def _range_changed(self):
        self.fit_scale()

    def _scale_mode_changed(self):
        self.scale_top.setVisible(self.scale.currentData() == MANUAL)
        self.fit_scale()

    def set_scale_exclusions(self, freqs):
        """Extra frequencies (the notches in use) left out of the automatic scale."""
        freqs = tuple(float(f) for f in freqs)
        if freqs != self._scale_exclusions:
            self._scale_exclusions = freqs
            self.fit_scale()

    def ignored_by_auto(self):
        """Frequencies the automatic scale does not measure itself against.

        Mains and the harmonics that fold back into the band (25 Hz at 125 Hz)
        are tens of times taller than any EEG: fitting the axis to them flattens
        the signal onto the baseline. They are still drawn - they simply run off
        the top of the plot.
        """
        return tuple(dict.fromkeys(line_frequencies(self._fs) + tuple(self._scale_exclusions)))

    def fit_scale(self):
        self._bounds = None
        self._held.reset()
        self.redraw()

    def redraw(self):
        grid = self.arrangement.currentText() == 'Grid'
        key = (grid, tuple(self._channels), tuple(self._names))
        if key != self._layout_key:
            self.graphics.clear(); self._plots = []; self._curves = []
            self._layout_key = key
            self._axes_key = None
            overlay = legend = None
            if not grid and self._channels:
                overlay = self.graphics.addPlot(row=0, col=0)
                # Below the plot, not on it: a legend over the data area was
                # repainted with every curve update, and hid the low end.
                legend = pg.LegendItem(colCount=8, labelTextSize='9pt')
                self.graphics.addItem(legend, row=1, col=0)
                self.graphics.ci.layout.setRowStretchFactor(0, 10)
            for slot, channel in enumerate(self._channels):
                plot = self.graphics.addPlot(row=slot // 4, col=slot % 4, viewBox=ChannelViewBox()) if grid else overlay
                if grid:
                    plot.setTitle(self._names[channel], color=CHANNEL_COLORS[channel % len(CHANNEL_COLORS)], size='10pt')
                    plot.getViewBox().clicked.connect(lambda c=channel: self.channel_clicked.emit(c))
                curve = plot.plot(pen=channel_pen(channel, shared_plot=not grid), antialias=False)
                if legend is not None:
                    legend.addItem(curve, self._names[channel])
                curve.setCurveClickable(True, width=8)
                curve.sigClicked.connect(lambda *args, c=channel: self.channel_clicked.emit(c))
                plot.setMouseEnabled(x=False, y=False)
                if not grid:
                    plot.showGrid(x=True, y=True, alpha=.15)
                # Grid cells: no grid lines, and tick values only on the outer
                # edge. Grid lines belong to the axis item and span the whole
                # cell, so every spectrum update invalidated all sixteen axes'
                # caches and redrew their text: on the 14 Sept recording that
                # stalled the window 10-15 times per 20 s (up to 744 ms);
                # without them, no stall over 50 ms.
                if grid:
                    if slot % 4:
                        plot.getAxis('left').setStyle(showValues=False)
                    if slot < len(self._channels) - 4:
                        plot.getAxis('bottom').setStyle(showValues=False)
                plot.getViewBox().disableAutoRange()
                for side in ('left', 'bottom'):
                    axis = plot.getAxis(side)
                    axis.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
                    # Major and first minor level only. A log axis otherwise
                    # draws every sub-decade line, dense and costly to repaint.
                    axis.setStyle(maxTickLevel=1)
                self._plots.append(plot); self._curves.append(curve)
            self.graphics.setMinimumHeight(max(280, math.ceil(len(self._channels) / 4) * 135) if grid else 280)
        if self._values is None:
            return
        lower, upper = self.frequency_range()
        mask = (self._freqs >= lower) & (self._freqs <= upper)
        self._bounds = self._axis_bounds(mask)
        units = self.unit_label()
        log = self.log.isChecked()
        axes_key = (log, units, lower, upper, self._bounds)
        if axes_key != self._axes_key:
            grid = self.arrangement.currentText() == 'Grid'
            n = len(self._plots)
            for slot, plot in enumerate(dict.fromkeys(self._plots)):
                plot.setLogMode(x=False, y=log)
                # Grid: axis titles on the outer edge only, like the tick values.
                plot.setLabel('left', units if not grid or slot % 4 == 0 else '')
                plot.setLabel('bottom', 'Hz' if not grid or slot >= n - 4 else '')
                plot.setXRange(lower, upper, padding=0)
                plot.setYRange(*self._bounds, padding=0)
            self._axes_key = axes_key
        if self._shown is None or self._shown.shape != self._values.shape:
            self._shown = self._values.copy()
        else:
            # A channel that turned invalid, or came back, does so at once:
            # only the size of valid values glides.
            changed = np.isfinite(self._values) != np.isfinite(self._shown)
            self._shown = np.where(changed, self._values, self._shown)
        self._draw_curves()

    def _axis_bounds(self, mask):
        """(bottom, top) for the y axis, in the chosen way.

        Auto measures the EEG and lets the mains lines run off the top; Peak
        fits the tallest thing drawn; Manual uses the typed top. Auto and Peak
        both grow at once and shrink only after the values stay small.
        """
        log = self.log.isChecked()
        mode = self.scale.currentData()
        if self._held.log != log:
            self._held = HeldScale(log=log)
        if mode == MANUAL:
            top = max(float(self.scale_top.value()), 1e-6)
            return (math.log10(top) - 5, math.log10(top)) if log else (0, top)
        scaled = mask.copy()
        if mode == AUTO:
            for line in self.ignored_by_auto():
                scaled &= np.abs(self._freqs - line) > LINE_WIDTH_HZ
        values = self._display(self._values[self._channels])[:, scaled if scaled.any() else mask]
        finite = values[np.isfinite(values)]
        high = max(float(finite.max()) if finite.size else 1.0, 1e-6)
        if log:
            fit = math.ceil(math.log10(high) * 2) / 2 + .5
            top = self._held.update(fit, self._last_time)
            return top - 5, top
        top = self._held.update(max(1.0, high * 1.25), self._last_time)
        return 0, top

    def _draw_curves(self):
        if self._shown is None or self._freqs is None:
            return
        lower, upper = self.frequency_range()
        mask = (self._freqs >= lower) & (self._freqs <= upper)
        log = self.log.isChecked()
        shown = self._display(self._shown[self._channels]) if self._channels else self._shown[:0]
        for row, curve in zip(shown, self._curves):
            row = row[mask]
            if np.isfinite(row).all():
                curve.setData(self._freqs[mask], np.maximum(row, 1e-12) if log else row, skipFiniteCheck=True)
            else:
                curve.setData([], [])


#: Axis colours for the motion sensor: X red, Y green, Z blue, the usual convention.
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


class MotionView(QtWidgets.QWidget):
    """The board's motion sensor under the traces, on the same time axis.

    Accelerometer in g (at rest |a| is 1 g, gravity) and, on the Unicorn, the
    gyroscope in °/s. A burst here at the moment of an EEG artefact says the
    head, cable or headset moved.
    """

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(0)
        header = QtWidgets.QHBoxLayout()
        self.toggle = QtWidgets.QCheckBox("Motion sensor")
        self.toggle.setChecked(True)
        self.toggle.setToolTip("Movement (mg), the accelerometer's three axes (g) and, on the Unicorn, "
                               "the gyroscope (°/s), over the trace window")
        self.readout = QtWidgets.QLabel("")
        header.addWidget(self.toggle)
        header.addWidget(self.readout, 1)
        layout.addLayout(header)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setFixedHeight(230)
        self.graphics.ci.layout.setContentsMargins(0, 2, 0, 2)   # edges set by align_to
        layout.addWidget(self.graphics)
        self.toggle.toggled.connect(self.graphics.setVisible)
        self._key = None
        self._curves = []
        self._plots = []
        self._target = None
        self._following = False
        self.names, self.units = [], []
        self.setVisible(False)

    def configure(self, names, units):
        """Axes to show; none hides the view (the source has no motion sensor)."""
        names, units = list(names), list(units)
        if (tuple(names), tuple(units)) == self._key:
            return
        self._key = (tuple(names), tuple(units))
        self.names, self.units = names, units
        self.graphics.clear()
        self._curves = []
        self._plots = []
        self.movement = None
        groups = {}
        for index, unit in enumerate(units):
            groups.setdefault(unit, []).append(index)
        rows = 0
        if "g" in groups:
            # First, because it is the one that answers "did the head move?".
            # The three axes below say which way; gravity keeps them near 1 g
            # whether the head moves or not.
            plot = self._add_plot("Move", "mg")
            self.movement = plot.plot(pen=pg.mkPen(color("bad"), width=1.5))
            plot.addLine(y=MOVEMENT_MG, pen=pg.mkPen(color("dim"), style=QtCore.Qt.PenStyle.DashLine))
            plot.addLine(y=-MOVEMENT_MG, pen=pg.mkPen(color("dim"), style=QtCore.Qt.PenStyle.DashLine))
            rows = 1
        for unit, indices in groups.items():
            plot = self._add_plot("Accel" if unit == "g" else "Gyro", unit)
            rows += 1
            legend = plot.addLegend(offset=(-5, 2), colCount=3, labelTextSize="8pt")
            for k, index in enumerate(indices):
                curve = plot.plot(pen=pg.mkPen(AXIS_COLORS[k % 3], width=1.5), name=names[index][-1])
                self._curves.append((index, curve))
            if legend is not None:
                legend.setBrush(pg.mkBrush(0, 0, 0, 0))
        self.graphics.setFixedHeight(90 * max(1, rows))
        self.setVisible(bool(names))
        self.readout.setText("")

    def _add_plot(self, label, unit):
        plot = self.graphics.addPlot(row=len(self._plots), col=0)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.showGrid(x=True, y=True, alpha=.15)
        value_axis(plot, label, unit)
        self._plots.append(plot)
        plot.vb.sigResized.connect(self._follow)
        return plot

    def align_to(self, left, right, x_range):
        """Put every plot's data area between these global x pixels, over x_range.

        The traces above set their own left margin (channel labels) and time
        span; matching both keeps a moment at the same x in the two views.
        """
        self._target = (left, right, tuple(x_range))
        self._follow()

    def _follow(self, *_):
        """Apply the stored target now; runs again whenever these plots are resized."""
        if self._target is None or self._following or not self._plots:
            return
        left, right, x_range = self._target
        view = self.graphics
        self._following = True
        try:
            # The plots share one column: give them identical widths, measured on the first.
            for plot in self._plots:
                plot.setXRange(*x_range, padding=0)
            first = self._plots[0]
            for _ in range(3):                        # settles in one pass; the rest is a safety net
                view.ci.layout.activate()             # measure the geometry as it is now
                for plot in self._plots:
                    plot.layout.activate()
                box = first.vb.sceneBoundingRect()
                now_left = view.mapToGlobal(view.mapFromScene(box.topLeft())).x()
                now_right = view.mapToGlobal(view.mapFromScene(box.topRight())).x()
                if abs(left - now_left) < 1 and abs(now_right - right) < 1:
                    break
                width = int(max(20, first.getAxis("left").width() + (left - now_left)))
                m_left, m_top, m_right, m_bottom = first.layout.getContentsMargins()
                m_right = max(0.0, m_right + (now_right - right))
                for plot in self._plots:
                    plot.getAxis("left").setWidth(width)
                    _, top, _, bottom = plot.layout.getContentsMargins()
                    plot.layout.setContentsMargins(m_left, top, m_right, bottom)
        finally:
            self._following = False

    def draw(self, data, fs):
        """data: (n_axes, n) of the newest samples, oldest first."""
        data = np.asarray(data, dtype=float)
        if not self._curves or data.ndim != 2 or data.shape[1] < 2:
            return
        from ..core.processing import decimate_minmax
        # Movement is decimated together with the axes so both keep the same
        # moments: taking the extremes of each separately would not line up.
        rows = np.nan_to_num(data, nan=0.0)
        movement = self.movement_of(data)
        stacked = np.vstack([np.nan_to_num(movement, nan=0.0)[None], rows])
        values, indices = decimate_minmax(stacked, 400, return_indices=True)
        if self.movement is not None:
            self.movement.setData(indices[0] / fs, values[0], skipFiniteCheck=True)
        for index, curve in self._curves:
            if index + 1 < len(values):
                curve.setData(indices[index + 1] / fs, values[index + 1], skipFiniteCheck=True)
        self.readout.setText(motion_summary(data, fs, self.units))

    def movement_of(self, data):
        """Movement in mg for these rows, or zeros when there is no accelerometer."""
        data = np.asarray(data, dtype=float)
        accel = [i for i, unit in enumerate(self.units) if unit == "g"]
        if len(accel) < 3 or data.ndim != 2 or data.shape[1] == 0:
            return np.zeros(data.shape[1] if data.ndim == 2 else 0)
        return movement_mg(data[accel[:3]])


def motion_summary(data, fs, units):
    """Movement over the last second, in words that need no threshold."""
    recent = np.asarray(data, dtype=float)[:, -max(2, int(fs)):]
    parts = []
    accel = [i for i, unit in enumerate(units) if unit == "g"]
    gyro = [i for i, unit in enumerate(units) if unit != "g"]
    if len(accel) == 3:
        magnitude = np.linalg.norm(recent[accel], axis=0)
        magnitude = magnitude[np.isfinite(magnitude)]
        if magnitude.size:
            moved = np.abs(movement_mg(recent[accel]))
            peak = float(np.max(moved)) if moved.size else 0.0
            parts.append("{} ±{:.0f} mg".format("moving" if peak > MOVEMENT_MG else "still", peak))
            parts.append("|a| {:.2f} g".format(float(np.median(magnitude))))
    if len(gyro) == 3:
        rate = np.linalg.norm(recent[gyro], axis=0)
        rate = rate[np.isfinite(rate)]
        if rate.size:
            parts.append("rotation {:.1f} °/s".format(float(np.mean(rate))))
    return "last 1 s: " + " · ".join(parts) if parts else ""
