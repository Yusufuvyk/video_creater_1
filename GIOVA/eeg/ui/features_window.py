"""Collapsible Monitor panel using the shared feature definitions and sample clock."""
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from .preferences import data_dir
from .scaling import HeldScale
from ..core.features import FeatureSet, FeatureHistory, FeatureLog, default_log_path
from ..core.feature_profiles import FeatureProfile
from .feature_view import ProfileDialog, fmt
from .plots import MultiCurveView
from .widgets import spin_box, combo_box


class FeaturesWindow(QtWidgets.QWidget):
    window_changed = QtCore.Signal(float)

    #: seconds of feature history kept on screen
    HISTORY_SECONDS = 60.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fs = None
        self.names = []
        self.profile = FeatureProfile()
        self.features = None
        self.history = None
        self.log = None
        self.last_row = None
        self.revision = 0
        self._last_compute = None
        self._now = None
        self.monitor = None
        self._history_key = None
        # The same held scale as the spectra: it grows at once and shrinks only
        # when the values stay small, so the curve moves and the axis does not.
        self._top = HeldScale()
        self._bottom = HeldScale()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.channel = combo_box()
        self.feature = combo_box()
        self.window_seconds = spin_box(.1, 60, 2, .5, 2)
        # Twice a second was a step every other frame; four times is a line.
        self.interval = spin_box(.01, 10, .25, .1, 2)
        for label, widget in (("Channel", self.channel), ("Feature", self.feature),
                              ("Window s", self.window_seconds), ("Every s", self.interval)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(widget)
        edit = QtWidgets.QPushButton("Choose features…")
        edit.clicked.connect(self.edit_profile)
        row.addWidget(edit)
        self.record = QtWidgets.QPushButton("Record CSV")
        self.record.setCheckable(True)
        self.record.toggled.connect(self._toggle_record)
        row.addWidget(self.record)
        layout.addLayout(row)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        splitter.addWidget(self.table)
        self.plot = MultiCurveView()
        splitter.addWidget(self.plot)
        splitter.setSizes([280, 600])
        layout.addWidget(splitter, 1)
        self.status = QtWidgets.QLabel("Waiting for new samples")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.feature.currentIndexChanged.connect(self._redraw_history)
        self.channel.currentIndexChanged.connect(self.reset)
        self.window_seconds.valueChanged.connect(self._settings_changed)
        self.interval.valueChanged.connect(self.reset)

    def attach(self, monitor):
        self.monitor = monitor
        monitor.features_ready.connect(self.on_frame)
        monitor.source_changed.connect(self.on_source)
        self.window_changed.connect(monitor.resize_feature_buffer)
        self.on_source(monitor.fs, monitor.panel.channel_names)

    def on_source(self, fs, names):
        self.fs = float(fs)
        self.names = list(names)
        index = self.channel.currentIndex()
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems(self.names)
        self.channel.setCurrentIndex(max(0, min(index, len(names)-1)) if index >= 0 else 0)
        self.channel.blockSignals(False)
        self.reset()

    def reset(self):
        if self.fs is None:
            return
        logging = self.log is not None
        if logging:
            self.log.close()
            self.log = None
        self.revision += 1
        self.features = FeatureSet(self.fs, self.window_seconds.value(), profile=self.profile)
        self.history = FeatureHistory(len(self.names), len(self.features.names), 1200)
        self.last_row = None
        self._last_compute = None
        self._now = None
        current = self.feature.currentIndex()
        self.feature.blockSignals(True)
        self.feature.clear()
        self.feature.addItems(self.features.names)
        self.feature.setCurrentIndex(max(0, min(current, self.feature.count()-1)))
        self.feature.blockSignals(False)
        self.table.setRowCount(len(self.features.names))
        for row, name in enumerate(self.features.names):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("—"))
        self.plot.draw([])
        self._history_key = None
        self._top.reset()
        self._bottom.reset()
        self.status.setText("Waiting for a complete window of new samples; causal Monitor filters")
        if logging:
            self._start_log()

    def _settings_changed(self):
        self.reset()
        self.window_changed.emit(self.window_seconds.value())

    def edit_profile(self):
        dialog = ProfileDialog(self, self.profile)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.profile = dialog.profile
            self._settings_changed()

    def on_frame(self, block, fs, names, source_seconds=None):
        if not self.isVisible() and self.log is None:
            return
        if fs != self.fs or list(names) != self.names:
            self.on_source(fs, names)
        if source_seconds is None:
            source_seconds = block.shape[1]/fs
        # The time axis follows the signal clock every frame, like the traces.
        # Moving it only when a value arrives (every 0.25 s) made it jump.
        self._now = source_seconds
        self._scroll()
        if self._last_compute is not None and source_seconds-self._last_compute < self.interval.value()-1e-9:
            return
        n = int(round(self.window_seconds.value()*fs))
        if block.shape[1] < n+int(fs):
            self.status.setText(f"Filling feature buffer: {block.shape[1]/fs:.1f} / {n/fs+1:.1f} s")
            return
        index = self.channel.currentIndex()
        if not 0 <= index < len(names):
            return
        row = self.features.compute(block[index:index+1, -n:])
        self._last_compute = source_seconds
        self.last_row = row
        full = np.full((len(names), row.shape[1]), np.nan)
        full[index] = row[0]
        self.history.append(source_seconds, full)
        for r, value in enumerate(row[0]):
            self.table.item(r, 1).setText(fmt(value))
            self.table.item(r, 1).setToolTip(self.features.last_values[0].reasons.get(self.profile.selected[r], ""))
        if self.log is not None:
            self.log.append(source_seconds, [index], row,
                            [[self.features.last_values[0].reasons.get(key, "") for key in self.profile.selected]])
        self._redraw_history()
        self.status.setText(f"{names[index]} · {n/fs:g} s window · {fs:g} Hz · revision {self.revision} · "
                            f"source time {source_seconds:.3f} s · causal filters · "
                            + ("recording "+self.log.path.name if self.log else self.profile.name))

    def _redraw_history(self):
        if self.history is None or not self.history.filled:
            return
        index = max(0, self.feature.currentIndex())
        channel = max(0, self.channel.currentIndex())
        times, values = self.history.series(index)
        key = (channel, index, self.feature.currentText(), self.names[channel])
        if key != self._history_key:
            self.plot.draw([(self.names[channel], times, values[:, channel], 0)])
            self.plot.setLabel("bottom", "Source time", units="s")
            self.plot.setLabel("left", self.feature.currentText())
            self._history_key = key
        else:
            self.plot._curves[0].setData(times, values[:, channel])
        self.plot._curves[0].setSymbol("o" if len(times) == 1 else None)
        self._scroll()
        shown = values[:, channel]
        if len(times):
            shown = shown[times >= float(times[-1])-self.HISTORY_SECONDS]
        finite = shown[np.isfinite(shown)]
        if finite.size:
            padding = max(float(np.ptp(finite))*.1, abs(float(finite.mean()))*.05, 1e-6)
            now = float(times[-1]) if len(times) else None
            top = self._top.update(float(finite.max())+padding, now)
            bottom = -self._bottom.update(-(float(finite.min())-padding), now)
            self.plot.setYRange(bottom, top, padding=0)

    def _scroll(self):
        """A rolling window ending at the current signal time.

        The whole history squeezed the newest values into the last few pixels.
        Display only: values, logging and the y scale are untouched.
        """
        if self.history is None or not self.history.filled or not self.plot._curves:
            return
        times, _ = self.history.series(max(0, self.feature.currentIndex()))
        if not len(times):
            return
        newest = max(float(times[-1]), float(self._now) if self._now is not None else float(times[-1]))
        self.plot.setXRange(max(float(times[0]), newest-self.HISTORY_SECONDS), max(newest, 1e-3), padding=0)

    def _start_log(self):
        metadata = dict(version=2, revision=self.revision, fs=self.fs, window=self.window_seconds.value(),
                        every=self.interval.value(), profile=self.profile.to_dict(), filter_mode="causal",
                        filters=self.monitor.panel.filter_kwargs if self.monitor else {})
        if self.monitor and self.monitor.source:
            metadata["recording"] = str(getattr(getattr(self.monitor.source, "recording", None), "path", "live"))
        try:
            self.log = FeatureLog(default_log_path(data_dir()), self.names, self.features.names, metadata)
            self.record.setText("Stop CSV")
        except OSError as exc:
            self.log = None
            self.record.setChecked(False)
            self.status.setText(str(exc))

    def _toggle_record(self, on):
        if on and self.features is not None:
            self._start_log()
        else:
            if self.log:
                self.log.close()
                self.log = None
            self.record.setText("Record CSV")

    def closeEvent(self, event):
        self.record.setChecked(False)
        event.accept()
