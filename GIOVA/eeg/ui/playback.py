"""Playback transport: a player wrapping a file source, and its control bar."""
from pyqtgraph.Qt import QtCore, QtWidgets
from .widgets import NoScrollComboBox, NoScrollSlider

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SKIP_SECONDS = 10


def format_time(seconds):
    """Seconds -> m:ss, or h:mm:ss past an hour."""
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "{}:{:02d}:{:02d}".format(hours, minutes, secs)
    return "{}:{:02d}".format(minutes, secs)


class Player:
    """Wraps a seekable source with play/pause and a playback speed.

    The source itself stays dumb; everything about *when* to read lives here.
    """

    def __init__(self, source, fs):
        self.source = source
        self.fs = fs
        self.playing = True
        self.speed = 1.0
        self._fraction = 0.0

    # ---- position ----

    @property
    def position(self):
        return getattr(self.source, "pos", 0)

    @property
    def duration_samples(self):
        return self.source.n_total

    @property
    def at_end(self):
        return self.duration_samples and self.position >= self.duration_samples

    # ---- transport ----

    def play(self):
        if self.at_end:
            self.seek_samples(0)
        self.playing = True

    def pause(self):
        self.playing = False

    def toggle(self):
        self.pause() if self.playing else self.play()

    def seek_samples(self, sample):
        self.source.seek(sample)
        self._fraction = 0.0

    def skip(self, seconds):
        self.seek_samples(self.position + int(seconds * self.fs))

    def advance(self, dt_seconds):
        """Read the samples due after dt of wall time. Empty while paused."""
        if not self.playing:
            return self.source.read(0)
        due = max(float(dt_seconds), 0.0) * self.fs * self.speed + self._fraction
        count = int(due + 1e-9)
        self._fraction = max(due - count, 0.0)
        return self.source.read(count)


class JumpSlider(NoScrollSlider):
    """A time line: a click anywhere on it goes to that moment.

    A plain QSlider steps a page towards the click, so reaching a moment took
    several clicks or a drag of the small handle. Here the handle is put under
    the pointer first; the press then lands on it, and dragging goes on from there.
    """

    def value_at(self, x):
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        style, control = self.style(), QtWidgets.QStyle.ComplexControl.CC_Slider
        groove = style.subControlRect(control, option, QtWidgets.QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(control, option, QtWidgets.QStyle.SubControl.SC_SliderHandle, self)
        if handle.left() <= x <= handle.right():
            return self.value()                     # on the handle: an ordinary drag
        return QtWidgets.QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), int(x - groove.x() - handle.width() / 2),
            max(groove.width() - handle.width(), 1), option.upsideDown)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            self.setValue(self.value_at(event.position().x()))
        super().mousePressEvent(event)


class TransportBar(QtWidgets.QWidget):
    """Play/pause, scrub, skip and speed - laid out like a media player."""

    play_toggled = QtCore.Signal()
    seek_requested = QtCore.Signal(int)  # samples
    speed_changed = QtCore.Signal(float)

    def __init__(self):
        super().__init__()
        self._fs = 1.0
        self._scrubbing = False

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)

        self.start_button = self._button("|<", "Jump to start",
                                         lambda: self.seek_requested.emit(0))
        self.back_button = self._button("<<", "Back 10 s",
                                        lambda: self._skip(-SKIP_SECONDS))
        self.play_button = self._button("Play", "Play / pause",
                                        self.play_toggled.emit, width=64)
        self.forward_button = self._button(">>", "Forward 10 s",
                                           lambda: self._skip(SKIP_SECONDS))
        self.end_button = self._button(">|", "Jump to end",
                                       lambda: self.seek_requested.emit(self._total))
        for widget in (self.start_button, self.back_button, self.play_button,
                       self.forward_button, self.end_button):
            row.addWidget(widget)

        self.elapsed = QtWidgets.QLabel("0:00")
        self.elapsed.setMinimumWidth(46)
        self.elapsed.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                  | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.elapsed)

        self.slider = JumpSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setToolTip("Click anywhere to go to that moment, or drag")
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._press)
        self.slider.sliderReleased.connect(self._release)
        self.slider.sliderMoved.connect(self._moved)
        row.addWidget(self.slider, 1)

        self.total = QtWidgets.QLabel("0:00")
        self.total.setMinimumWidth(46)
        row.addWidget(self.total)

        self.speed = NoScrollComboBox()
        for value in SPEEDS:
            self.speed.addItem("{:g}x".format(value), value)
        self.speed.setCurrentIndex(SPEEDS.index(1.0))
        self.speed.currentIndexChanged.connect(
            lambda: self.speed_changed.emit(self.speed.currentData()))
        row.addWidget(self.speed)

        self._total = 0
        self.setEnabled(False)

    def _button(self, text, tip, slot, width=40):
        button = QtWidgets.QPushButton(text)
        button.setToolTip(tip)
        button.setFixedWidth(width)
        button.clicked.connect(slot)
        return button

    def _skip(self, seconds):
        self.seek_requested.emit(
            int(self.slider.value() + seconds * self._fs))

    def _press(self):
        self._scrubbing = True

    def _release(self):
        self._scrubbing = False
        self.seek_requested.emit(self.slider.value())

    def _moved(self, value):
        self.elapsed.setText(format_time(value / self._fs))

    # ---- state from the player ----

    def configure(self, total_samples, fs):
        self._fs = fs
        self._total = total_samples
        self.slider.setRange(0, total_samples)
        self.total.setText(format_time(total_samples / fs))
        self.setEnabled(total_samples > 0)

    def update_position(self, sample):
        if self._scrubbing:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(int(sample))
        self.slider.blockSignals(False)
        self.elapsed.setText(format_time(sample / self._fs))

    def set_playing(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
