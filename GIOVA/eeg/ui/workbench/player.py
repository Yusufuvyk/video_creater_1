"""A recording played like a video: its signal scrolling, the spectrum of the moment, and the spectrogram of
the whole recording with a cursor moving across it.

Opened from a Recording's preview (Play…), on whatever that block gives: the file as read, or after a
filter, re-reference or bad-channel repair. Nothing is computed beyond what is drawn: the spectrum of the
last few seconds at each step, and the spectrogram once per channel choice.

- Signal: the last `SIGNAL_SECONDS` up to the cursor, around its mean, events marked.
- Spectrum now: the FFT amplitude of the last `window` seconds (Hann window, a sine of amplitude A reads
  about A), over the whole recording's average in grey: what is different about this moment.
- Spectrogram: power in dB of 1 s Hann segments every 0.25 s, 1-45 Hz, the whole recording; the cursor
  shows where the signal and spectrum are. Clicking it moves there.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..plots import CHANNEL_COLORS, MARKER_COLOUR
from ..theme import color, muted
from ..widgets import NoScrollComboBox
from .previews import ALL_CHANNELS, small_plot

#: Seconds of signal shown up to the cursor.
SIGNAL_SECONDS = 10.0
#: Playing speeds offered: seconds of recording per second of watching.
SPEEDS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
FRAME_MS = 40
#: The spectrogram: segment and step in s, highest frequency in Hz.
STFT_SEGMENT, STFT_STEP, STFT_FMAX = 1.0, 0.25, 45.0
MEAN_OF_ALL = "Mean of all channels"
#: How much of the spectrogram is shown, s (None: all of it).
SPANS = {"1 min": 60.0, "5 min": 300.0, "Whole": None}


def spectrogram_db(x, fs):
    """(freqs, frame times, power dB) of one channel; missing samples count as zero, as the gaps they are."""
    from scipy.signal import spectrogram
    nperseg = max(8, int(round(STFT_SEGMENT * fs)))
    step = max(1, int(round(STFT_STEP * fs)))
    freqs, times, power = spectrogram(np.nan_to_num(x), fs=fs, window="hann", nperseg=nperseg,
                                      noverlap=nperseg - step, detrend="constant", scaling="density")
    keep = (freqs >= 1.0) & (freqs <= min(STFT_FMAX, fs / 2))
    return freqs[keep], times, 10 * np.log10(np.maximum(power[keep], 1e-12))


def amplitude_now(x, fs):
    """(freqs, amplitude µV) of one stretch: Hann window, scaled so a sine of amplitude A reads A."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    x = x - x.mean()
    window = np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(x * window)) * 2 / max(window.sum(), 1e-12)
    return np.fft.rfftfreq(x.size, 1 / fs), spectrum


class PlayerWindow(QtWidgets.QWidget):
    def __init__(self, recording, title="Recording", parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("{} - play".format(title))
        self.recording = recording
        self.position = min(SIGNAL_SECONDS, recording.duration)
        self.cache = {}

        self.channel = NoScrollComboBox()
        self.channel.addItems(list(recording.channel_names) + [MEAN_OF_ALL])
        first = next((n for n in ("Cz", "C3", "Fz", "Pz", "Oz") if n in recording.channel_names), None)
        if first:
            self.channel.setCurrentText(first)
        self.channel.currentIndexChanged.connect(lambda _i: self._channel_changed())
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setProperty("primary", True)
        self.play_button.setToolTip("Play or pause (Space)")
        self.play_button.clicked.connect(self.toggle)
        self.speed = NoScrollComboBox()
        self.speed.addItems(["{:g}×".format(s) for s in SPEEDS])
        self.speed.setCurrentText("1×")
        self.speed.setToolTip("Seconds of recording per second of watching")
        self.window_box = NoScrollComboBox()
        self.window_box.addItems(["1 s", "2 s", "4 s", "8 s"])
        self.window_box.setCurrentText("2 s")
        self.window_box.setToolTip("How much of the signal before the cursor the spectrum is computed over: longer "
                                   "is finer in frequency, shorter follows faster changes")
        self.window_box.currentIndexChanged.connect(lambda _i: self.draw_moment())
        self.span = NoScrollComboBox()
        self.span.addItems(list(SPANS))
        self.span.setCurrentText("5 min")
        self.span.setToolTip("How much of the spectrogram is shown: the whole recording, or minutes around the "
                             "cursor, following it")
        self.span.currentIndexChanged.connect(lambda _i: self.draw_moment())
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(recording.duration * 10))
        self.slider.sliderMoved.connect(lambda v: self.seek(v / 10.0))
        self.clock = QtWidgets.QLabel()
        self.clock.setMinimumWidth(130)
        bar = QtWidgets.QHBoxLayout()
        for widget in (QtWidgets.QLabel("Channel"), self.channel, self.play_button, QtWidgets.QLabel("Speed"),
                       self.speed, QtWidgets.QLabel("Spectrum of the last"), self.window_box,
                       QtWidgets.QLabel("Spectrogram"), self.span):
            bar.addWidget(widget)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.clock)

        self.signal = small_plot(170, True)
        self.signal.setLabel("left", "µV")
        self.signal.setLabel("bottom", "Time (s)")
        self.signal_curve = self.signal.plot(pen=pg.mkPen(CHANNEL_COLORS[0], width=1))
        self.spectrum = small_plot(200, True)
        self.spectrum.setLogMode(x=False, y=True)
        self.spectrum.setLabel("left", "Amplitude (µV)")
        self.spectrum.setLabel("bottom", "Frequency (Hz)")
        self.spectrum.addLegend(offset=(-5, 5))
        self.average_curve = self.spectrum.plot(pen=pg.mkPen(color("dim"), width=1), name="whole recording")
        self.now_curve = self.spectrum.plot(pen=pg.mkPen(CHANNEL_COLORS[1], width=1), name="now")
        self.spectrogram = small_plot(230, True)
        self.spectrogram.setLabel("left", "Frequency (Hz)")
        self.spectrogram.setLabel("bottom", "Time (s)")
        self.image = pg.ImageItem()
        self.spectrogram.addItem(self.image)
        self.bar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"), values=(0, 1), interactive=False, width=12)
        self.bar.setImageItem(self.image, insert_in=self.spectrogram.getPlotItem())
        self.bar.axis.setLabel("dB re 1 µV²/Hz")
        self.cursor = pg.InfiniteLine(self.position, angle=90, pen=pg.mkPen(color("accent"), width=2))
        self.spectrogram.addItem(self.cursor)
        self.spectrogram.scene().sigMouseClicked.connect(self._spectrogram_clicked)
        events = recording.events
        if events is not None:
            for onset in events.onsets:
                self.spectrogram.addItem(pg.InfiniteLine(onset, angle=90, pen=pg.mkPen(MARKER_COLOUR, width=1,
                                                         style=QtCore.Qt.PenStyle.DotLine)))
        note = QtWidgets.QLabel("Spectrum: FFT of the last seconds before the cursor (Hann), over the whole "
                                "recording's average in grey. Spectrogram: 1 s segments every 0.25 s; click it to "
                                "go there. Space plays and pauses; Left / Right step 5 s.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        column = QtWidgets.QVBoxLayout(self)
        column.addLayout(bar)
        column.addWidget(self.signal)
        column.addWidget(self.spectrum)
        column.addWidget(self.spectrogram)
        column.addWidget(note)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self._step)
        for key, action in (("Space", self.toggle), ("Left", lambda: self.seek(self.position - 5)),
                            ("Right", lambda: self.seek(self.position + 5))):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(action)
        self.resize(1200, 860)
        self._channel_changed()

    # ---------------- what is shown ----------------

    def _rows(self):
        names = list(self.recording.channel_names)
        text = self.channel.currentText()
        return list(range(len(names))) if text in (MEAN_OF_ALL, ALL_CHANNELS) else [names.index(text)]

    def _signal(self, a, b):
        rows = self._rows()
        block = self.recording.data[rows, a:b]
        return block.mean(axis=0) if len(rows) > 1 else block[0]

    def _channel_changed(self):
        """The spectrogram of the whole recording and its average spectrum, for the channel (or mean) chosen."""
        key = tuple(self._rows())
        if key not in self.cache:
            fs = self.recording.fs
            maps = [spectrogram_db(self.recording.data[r], fs) for r in key]
            freqs, times = maps[0][0], maps[0][1]
            # several channels: their power averaged, then in dB
            power = np.mean([10 ** (m[2] / 10) for m in maps], axis=0)
            amplitude = np.sqrt(2 * power * (freqs[1] - freqs[0]) * 1.5) if freqs.size > 1 else np.sqrt(power)
            self.cache[key] = (freqs, times, 10 * np.log10(np.maximum(power, 1e-12)), amplitude.mean(axis=1))
        freqs, times, db, average = self.cache[key]
        finite = db[np.isfinite(db)]
        levels = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else (0, 1)
        self.image.setImage(db.T, levels=levels)
        self.bar.setLevels(levels)
        df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
        dt = times[1] - times[0] if times.size > 1 else STFT_STEP
        self.image.setRect(QtCore.QRectF(times[0] - dt / 2, freqs[0] - df / 2, dt * times.size, df * freqs.size))
        self.spectrogram.setYRange(freqs[0], freqs[-1], padding=0)
        self.average_curve.setData(freqs, np.maximum(average, 1e-6))
        self.draw_moment()

    def draw_moment(self):
        """The signal up to the cursor, the spectrum of its last seconds, the cursor on the spectrogram."""
        fs, rec = self.recording.fs, self.recording
        end = int(round(self.position * fs))
        start = max(0, end - int(SIGNAL_SECONDS * fs))
        if end - start > 1:
            x = self._signal(start, end)
            centre = np.nanmean(x) if np.isfinite(x).any() else 0.0
            self.signal_curve.setData(np.arange(start, end) / fs, x - centre, connect="finite")
            self.signal.setXRange(max(0.0, self.position - SIGNAL_SECONDS), max(self.position, SIGNAL_SECONDS),
                                  padding=0)
        seconds = float(self.window_box.currentText().split()[0])
        a = max(0, end - int(seconds * fs))
        if end - a > 8:
            freqs, amplitude = amplitude_now(self._signal(a, end), fs)
            keep = (freqs >= 1.0) & (freqs <= STFT_FMAX)
            self.now_curve.setData(freqs[keep], np.maximum(amplitude[keep], 1e-6))
        self.cursor.setValue(self.position)
        width = SPANS.get(self.span.currentText())
        if width is None or width >= rec.duration:
            self.spectrogram.setXRange(0, rec.duration, padding=0)
        else:
            # the cursor a third of the way in, the view moving with it
            left = min(max(self.position - width / 3, 0.0), rec.duration - width)
            self.spectrogram.setXRange(left, left + width, padding=0)
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.position * 10))
        self.slider.blockSignals(False)
        events = rec.events
        label = ""
        if events is not None and len(events):
            before = np.flatnonzero(events.onsets <= self.position)
            if before.size:
                k = before[-1]
                label = "  ·  {:.1f} s after {}".format(self.position - events.onsets[k], events.labels[k])
        self.clock.setText("{:.1f} / {:.0f} s{}".format(self.position, rec.duration, label))

    # ---------------- playing ----------------

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            if self.position >= self.recording.duration - 1e-6:
                self.position = min(SIGNAL_SECONDS, self.recording.duration)
            self.timer.start()
            self.play_button.setText("Pause")

    def _step(self):
        speed = SPEEDS[max(0, self.speed.currentIndex())]
        self.seek(self.position + speed * FRAME_MS / 1000.0)
        if self.position >= self.recording.duration - 1e-6:
            self.toggle()

    def seek(self, seconds):
        self.position = float(min(max(seconds, 0.0), self.recording.duration))
        self.draw_moment()

    def _spectrogram_clicked(self, event):
        view = self.spectrogram.getPlotItem().getViewBox()
        if view.sceneBoundingRect().contains(event.scenePos()):
            self.seek(view.mapSceneToView(event.scenePos()).x())

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
