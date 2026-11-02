"""The filter controls, once, for both pages.

They used to be written out twice and had drifted apart: the monitor let you
choose 50 or 60 Hz mains, set the notch Q and the filter order, while the
analysis page hard-coded 50 Hz and order 4. That is not a cosmetic difference.
Cho 2017 was recorded in Korea, where the mains is 60 Hz, so the analysis page
was notching a frequency that was not there and leaving the one that was.

One widget, one set of defaults, one place to change them. The offline pages
get more of it: high-pass and low-pass switch on separately, the mains
harmonics are optional, and trials can be detrended and re-referenced. The
monitor keeps its compact band-pass layout.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import sosfreqz

from ..core.processing import design_filter, mains_harmonics
from .widgets import NoScrollComboBox, spin_box
from .plots import TRACE_COLOUR
from .theme import error, plot_background

MAINS = ("50 Hz", "60 Hz")
DETRENDS = (("Remove mean", "constant"), ("Remove linear trend", "linear"))
REFERENCES = (("As recorded", "none"), ("Common average (CAR)", "average"),
              ("Mean of chosen channels", "channels"))


class FilterGroup(QtWidgets.QGroupBox):
    """Notch + band settings, and the kwargs `design_filter` wants.

    `offline=True` adds the analysis-only controls; their values travel in the
    same dict under keys `design_filter` does not take (`detrend`,
    `reference`, `reference_channels`), which `core.trials` handles.
    """

    changed = QtCore.Signal()

    def __init__(self, fs, low=1.0, high=45.0, title="Filters", offline=False):
        super().__init__(title)
        self.fs = float(fs)
        self.offline = offline
        self.channel_names = []
        self.reference_channels = []
        form = QtWidgets.QFormLayout(self)

        self.notch_on = QtWidgets.QCheckBox("Notch")
        self.notch_on.setChecked(True)
        self.notch_freq = NoScrollComboBox()
        self.notch_freq.addItems(MAINS)
        self.notch_freq.setToolTip(
            "Mains frequency where the data was recorded - 50 Hz in Europe, "
            "60 Hz in the Americas, Korea and Japan.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.notch_on)
        row.addWidget(self.notch_freq)
        self.harmonics = QtWidgets.QCheckBox("+ harmonics")
        self.harmonics.setChecked(True)
        self.harmonics.setToolTip("Also notch 2x, 3x ... the mains frequency below Nyquist.")
        if offline:
            row.addWidget(self.harmonics)
        form.addRow(row)

        self.notch_q = spin_box(5, 60, 30)
        self.notch_q.setToolTip("Higher Q is a narrower notch: less of the "
                                "neighbouring signal is removed with the hum.")
        form.addRow("Notch Q", self.notch_q)

        self.low_cut = spin_box(0.01 if offline else 0.1, 1e5 if offline else 60, low, 0.5, 2 if offline else 1, " Hz")
        # Replaced with the recording's own Nyquist as soon as one is loaded.
        self.high_cut = spin_box(0.1 if offline else 1.0, 1e5, high, 1.0, 1, " Hz")
        self.order = spin_box(1, 8, 4)
        self.order.setToolTip(
            "Butterworth order. Steeper roll-off, but more ringing around sharp "
            "transients. 4 is the usual compromise. Offline filtering runs "
            "forwards and backwards (zero phase), which doubles the effective order.")

        self.bandpass_on = QtWidgets.QCheckBox("Bandpass")
        self.bandpass_on.setChecked(True)
        self.highpass_on = QtWidgets.QCheckBox("High-pass")
        self.lowpass_on = QtWidgets.QCheckBox("Low-pass")
        for box in (self.highpass_on, self.lowpass_on):
            box.setChecked(True)
        if offline:
            for check, spin in ((self.highpass_on, self.low_cut), (self.lowpass_on, self.high_cut)):
                row = QtWidgets.QHBoxLayout()
                row.addWidget(check)
                row.addWidget(spin, 1)
                form.addRow(row)
                check.toggled.connect(spin.setEnabled)
            self.low_cut.setToolTip("Removes slow drift below this frequency.")
            self.high_cut.setToolTip("Removes activity above this frequency.")
        else:
            form.addRow(self.bandpass_on)
            form.addRow("Low cut", self.low_cut)
            form.addRow("High cut", self.high_cut)
        form.addRow("Order", self.order)

        self.detrend = NoScrollComboBox()
        self.reference = NoScrollComboBox()
        self.reference_button = QtWidgets.QPushButton("Reference channels…")
        self.response = None
        self.warning = QtWidgets.QLabel()
        if offline:
            for label, key in DETRENDS:
                self.detrend.addItem(label, key)
            self.detrend.setToolTip("Applied to each trial before FFT, PSD and features.")
            form.addRow("Detrend", self.detrend)
            for label, key in REFERENCES:
                self.reference.addItem(label, key)
            self.reference.setToolTip("Subtracted from every channel of the continuous "
                                      "recording before filtering.")
            form.addRow("Reference", self.reference)
            self.reference_button.clicked.connect(self._choose_reference)
            form.addRow(self.reference_button)
            self.response = pg.PlotWidget(background=plot_background())
            self.response.setMinimumHeight(120)
            self.response.setMaximumHeight(150)
            self.response.setMouseEnabled(x=False, y=False)
            self.response.getAxis("left").enableAutoSIPrefix(False)
            self.response.setLabel("left", "dB")
            self.response.setLabel("bottom", "Hz")
            self.response.showGrid(x=True, y=True, alpha=0.2)
            self.response.setTitle("Zero-phase response |H|²", size="8pt")
            self._response_curve = self.response.plot(pen=pg.mkPen(TRACE_COLOUR, width=2))
            form.addRow(self.response)
            self.reference.currentIndexChanged.connect(self._sync_reference)
            self._sync_reference()

        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(error())
        form.addRow(self.warning)

        for widget in (self.notch_on, self.bandpass_on, self.highpass_on, self.lowpass_on, self.harmonics):
            widget.toggled.connect(self.changed)
        for combo in (self.notch_freq, self.detrend, self.reference):
            combo.currentIndexChanged.connect(self.changed)
        for box in (self.notch_q, self.low_cut, self.high_cut, self.order):
            box.valueChanged.connect(self.changed)
        self.changed.connect(self._update_preview)
        self.toggled.connect(lambda _: self._update_preview())
        self._update_preview()

    def set_sampling_rate(self, fs):
        """Follow the recording: a filter edge above Nyquist is meaningless."""
        self.fs = float(fs)
        limit = fs / 2.0 - 0.5
        self.high_cut.setMaximum(limit)
        if self.high_cut.value() > limit:
            self.high_cut.setValue(limit)
        self.low_cut.setMaximum(limit)
        self._update_preview()

    def set_channel_names(self, names):
        """The recording's channels, for choosing reference channels."""
        self.channel_names = list(names)
        self.reference_channels = [n for n in self.reference_channels if n in self.channel_names]
        self._sync_reference()

    @property
    def mains(self):
        return 50.0 if self.notch_freq.currentIndex() == 0 else 60.0

    @property
    def kwargs(self):
        """What `processing.design_filter` takes, plus the offline-only keys."""
        if not self.offline:
            enabled = self.bandpass_on.isChecked()
            return dict(
                low=self.low_cut.value() if enabled else None,
                high=self.high_cut.value() if enabled else None,
                order=int(self.order.value()),
                notches=mains_harmonics(self.mains, self.fs)
                if self.notch_on.isChecked() else (),
                notch_q=self.notch_q.value(),
            )
        notches = ()
        if self.notch_on.isChecked():
            notches = (list(mains_harmonics(self.mains, self.fs)) if self.harmonics.isChecked()
                       else [self.mains])
        values = dict(
            low=self.low_cut.value() if self.highpass_on.isChecked() else None,
            high=self.high_cut.value() if self.lowpass_on.isChecked() else None,
            order=int(self.order.value()),
            notches=list(notches),
            notch_q=self.notch_q.value(),
            detrend=self.detrend.currentData() or "constant",
            reference=self.reference.currentData() or "none",
        )
        if values["reference"] == "channels":
            values["reference_channels"] = list(self.reference_channels)
        return values

    def set_kwargs(self, filters):
        """Show a saved filter dict. Older dicts (low/high/notches only) work."""
        filters = dict(filters or {})
        widgets = (self.notch_on, self.bandpass_on, self.highpass_on, self.lowpass_on, self.harmonics,
                   self.notch_freq, self.detrend, self.reference, self.notch_q, self.low_cut,
                   self.high_cut, self.order)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            low, high = filters.get("low"), filters.get("high")
            self.highpass_on.setChecked(low is not None)
            self.lowpass_on.setChecked(high is not None)
            self.bandpass_on.setChecked(low is not None or high is not None)
            self.low_cut.setEnabled(low is not None or not self.offline)
            self.high_cut.setEnabled(high is not None or not self.offline)
            if low is not None:
                self.low_cut.setValue(float(low))
            if high is not None:
                self.high_cut.setValue(float(high))
            self.order.setValue(int(filters.get("order", 4)))
            notches = [float(f) for f in filters.get("notches") or []]
            self.notch_on.setChecked(bool(notches))
            if notches:
                base = min(notches)
                self.notch_freq.setCurrentIndex(1 if abs(base - 60) < 1 else 0)
                self.harmonics.setChecked(len(notches) > 1)
            self.notch_q.setValue(float(filters.get("notch_q", 30)))
            index = self.detrend.findData(filters.get("detrend", "constant"))
            self.detrend.setCurrentIndex(max(index, 0))
            index = self.reference.findData(filters.get("reference") or "none")
            self.reference.setCurrentIndex(max(index, 0))
            self.reference_channels = list(filters.get("reference_channels") or [])
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._sync_reference()
        self.changed.emit()

    # ---------------- offline extras ----------------

    def _choose_reference(self):
        if not self.channel_names:
            return
        from .headmap import ChannelSelectorDialog
        chosen = ChannelSelectorDialog.ask(
            self, self.channel_names,
            [self.channel_names.index(n) for n in self.reference_channels if n in self.channel_names])
        if chosen is not None:
            self.reference_channels = [self.channel_names[i] for i in chosen]
            self._sync_reference()
            self.changed.emit()

    def _sync_reference(self):
        by_channels = self.reference.currentData() == "channels"
        self.reference_button.setVisible(self.offline and by_channels)
        shown = ", ".join(self.reference_channels[:4]) + (" …" if len(self.reference_channels) > 4 else "")
        self.reference_button.setText("Reference channels: " + (shown or "choose…"))

    def problem(self):
        """Why the current settings cannot be used, or ''."""
        values = self.kwargs
        nyquist = self.fs / 2
        low, high = values.get("low"), values.get("high")
        if low is not None and low >= nyquist:
            return "High-pass must be below Nyquist ({:g} Hz)".format(nyquist)
        if high is not None and high >= nyquist:
            return "Low-pass must be below Nyquist ({:g} Hz)".format(nyquist)
        if low is not None and high is not None and low >= high:
            return "High-pass must be below low-pass"
        if values.get("reference") == "channels" and not values.get("reference_channels"):
            return "Choose reference channels"
        return ""

    def _update_preview(self):
        problem = self.problem() if self.isChecked() or not self.isCheckable() else ""
        self.warning.setText(problem)
        self.warning.setVisible(bool(problem))
        if self.response is None:
            return
        active = self.isChecked() or not self.isCheckable()
        values = self.kwargs
        sos = None
        if active and not problem:
            try:
                sos = design_filter(self.fs, low=values["low"], high=values["high"], order=values["order"],
                                    notches=values["notches"], notch_q=values["notch_q"])
            except ValueError as exc:
                self.warning.setText(str(exc))
                self.warning.setVisible(True)
        if sos is None:
            freqs = np.array([0.0, self.fs / 2])
            gain = np.zeros(2)
        else:
            freqs, h = sosfreqz(sos, worN=2048, fs=self.fs)
            gain = np.maximum(20 * np.log10(np.maximum(np.abs(h), 1e-12)) * 2, -120)
        self._response_curve.setData(freqs, gain)
        self.response.setXRange(0, self.fs / 2, padding=0)
        self.response.setYRange(-80, 5, padding=0)
