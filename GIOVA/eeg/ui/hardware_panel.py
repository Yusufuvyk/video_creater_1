"""Board channel settings: which amplifier channels are on and which feed BIAS.

Unlike the graph selection, these boxes change the hardware, and only when
Apply is pressed (or at the next Start). Until then the board keeps what it
had, so the boxes are not a description of the board: the line under them is.
"""
from pyqtgraph.Qt import QtCore, QtWidgets

from ..core import hardware as hw
from .theme import muted
from .widgets import NoScrollSpinBox

IMPEDANCE_HELP = (
    "6 nA at 31.25 Hz into each ON channel in turn (~6 s each, keep still). The value is "
    "electrode + head + BIAS electrode, so put the BIAS electrode on the skin.\n"
    "Green < 750 kΩ, yellow < 2500 kΩ, red above: the OpenBCI GUI defaults for this "
    "same Cyton measurement. Open? ≥ 4 MΩ: OpenBCI reports ~5000 kΩ with no electrode "
    "attached. Green means contact is adequate; still check the mains (50 / 60 Hz) column and the "
    "signal. Your channel/BIAS settings are restored afterwards.")


class HardwareGroup(QtWidgets.QGroupBox):
    apply_clicked = QtCore.Signal()
    resend_clicked = QtCore.Signal()
    impedance_clicked = QtCore.Signal()
    short_clicked = QtCore.Signal()

    def __init__(self, names):
        super().__init__("Board channels (hardware)")
        self.names = list(names)
        self.active_count = len(self.names)
        outer = QtWidgets.QVBoxLayout(self)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        self.on_boxes, self.bias_boxes, self.labels = [], [], []
        self.column_headers = []
        half = (len(self.names) + 1) // 2
        for block in range(2):
            base = block * 4
            for column, text in enumerate(("", "On", "BIAS")):
                header = QtWidgets.QLabel(text)
                grid.addWidget(header, 0, base + column)
                self.column_headers.append(header)
        for i, name in enumerate(self.names):
            row, base = 1 + i % half, 4 * (i // half)
            label = QtWidgets.QLabel(name)
            on = QtWidgets.QCheckBox()
            on.setChecked(True)
            bias = QtWidgets.QCheckBox()
            bias.setChecked(True)
            on.toggled.connect(self._changed)
            bias.toggled.connect(self._changed)
            grid.addWidget(label, row, base)
            grid.addWidget(on, row, base + 1)
            grid.addWidget(bias, row, base + 2)
            self.labels.append(label)
            self.on_boxes.append(on)
            self.bias_boxes.append(bias)
        grid.setColumnMinimumWidth(3, 12)
        outer.addLayout(grid)

        presets = QtWidgets.QHBoxLayout()
        for text, callback in [("All on", self.all_on),
                               ("Only selected…", self.choose_only),
                               ("No BIAS", self.no_bias)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            presets.addWidget(button)
        outer.addLayout(presets)

        buttons = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton("Apply to board")
        self.apply_button.setProperty("primary", True)
        self.apply_button.setToolTip("Sends only the channels you changed, in the background.")
        self.apply_button.clicked.connect(self.apply_clicked.emit)
        self.resend_button = QtWidgets.QPushButton("Resend all")
        self.resend_button.setToolTip(
            "Sends all 16 channels again (a few seconds). Use if the board may have "
            "reset itself, e.g. after a power dip.")
        self.resend_button.clicked.connect(self.resend_clicked.emit)
        buttons.addWidget(self.apply_button, 2)
        buttons.addWidget(self.resend_button, 1)
        outer.addLayout(buttons)

        note = QtWidgets.QLabel(
            "Off = channel powered down, out of BIAS and SRB2. With no channel "
            "in BIAS the feedback input is open and BIASOUT may saturate; unplug "
            "the lead to remove the BIAS electrode.")
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        outer.addWidget(note)
        self.state = QtWidgets.QLabel("Board: not connected")
        self.state.setWordWrap(True)
        outer.addWidget(self.state)

        self.impedance_button = QtWidgets.QPushButton("Impedance check (ON channels)")
        self.impedance_button.setToolTip(IMPEDANCE_HELP)
        self.impedance_button.clicked.connect(self.impedance_clicked.emit)
        outer.addWidget(self.impedance_button)
        self.short_button = QtWidgets.QPushButton("ADS1299 internal short (20 s)")
        self.short_button.setToolTip(
            "Internally shorts every ADS1299 input for 20 seconds, measures ADC/PGA noise "
            "and mains (50 / 60 Hz) pickup, then restores all channel/BIAS settings. Electrodes may stay "
            "connected because the input mux disconnects them during the test.")
        self.short_button.clicked.connect(self.short_clicked.emit)
        # A hardware diagnostic, rarely needed: it lives in Settings -> Monitor
        # (which borrows this widget), not in the sidebar.
        self.diagnostics = QtWidgets.QWidget()
        diagnostics = QtWidgets.QVBoxLayout(self.diagnostics)
        diagnostics.setContentsMargins(0, 0, 0, 0)
        diagnostics.addWidget(self.short_button)
        # The colour limits are set once and then left alone: they live in
        # Settings -> Monitor, which borrows this widget.
        self.limits = QtWidgets.QWidget()
        limits = QtWidgets.QFormLayout(self.limits)
        limits.setContentsMargins(0, 0, 0, 0)
        self.green_limit = NoScrollSpinBox()
        self.yellow_limit = NoScrollSpinBox()
        for spin, value in ((self.green_limit, 750), (self.yellow_limit, 2500)):
            spin.setRange(1, 4000)
            spin.setDecimals(0)
            spin.setSuffix(" kΩ")
            spin.setValue(value)
        limits.addRow("Green below", self.green_limit)
        limits.addRow("Yellow below", self.yellow_limit)
        self.impedance = QtWidgets.QLabel("Impedance: not measured")
        self.impedance.setWordWrap(True)
        self.impedance.setTextFormat(QtCore.Qt.TextFormat.RichText)
        outer.addWidget(self.impedance)
        self.short_result = QtWidgets.QLabel("Internal short: not measured")
        self.short_result.setWordWrap(True)
        self.short_result.setTextFormat(QtCore.Qt.TextFormat.RichText)
        diagnostics.addWidget(self.short_result)
        self._changed()

    def set_channel_count(self, count):
        """Show and return only channels physically present on the live board."""
        count = max(0, min(int(count), len(self.names)))
        self.active_count = count
        for i, widgets in enumerate(zip(self.labels, self.on_boxes, self.bias_boxes)):
            visible = i < count
            for widget in widgets:
                widget.setVisible(visible)
        for header in self.column_headers[3:]:
            header.setVisible(count > (len(self.names) + 1) // 2)
        self.setTitle("Board channels (hardware) · {} channel".format(count))
        self.resend_button.setToolTip(
            "Sends all {} physical channels again. Use if the board may have reset itself, "
            "e.g. after a power dip.".format(count))
        self._changed()

    # ---- values ----

    def settings(self):
        return [hw.ChannelSetting(on.isChecked(), bias.isChecked())
                for on, bias in zip(self.on_boxes[:self.active_count],
                                    self.bias_boxes[:self.active_count])]

    def all_settings(self):
        """All saved controls, including Daisy channels hidden in 8-channel mode."""
        return [hw.ChannelSetting(on.isChecked(), bias.isChecked())
                for on, bias in zip(self.on_boxes, self.bias_boxes)]

    def set_settings(self, settings):
        for on, bias, s in zip(self.on_boxes, self.bias_boxes, settings):
            on.setChecked(s.on)
            bias.setChecked(s.bias)
        self._changed()

    def on_channels(self):
        return [i for i, box in enumerate(self.on_boxes[:self.active_count]) if box.isChecked()]

    def set_names(self, names):
        """Relabel the pins (a channel profile's wiring); the count never changes."""
        if len(names) != len(self.labels):
            return
        self.names = list(names)
        for label, name in zip(self.labels, self.names):
            label.setText(name)

    # ---- presets ----

    def all_on(self):
        self.set_settings(hw.default_settings(self.active_count))

    def choose_only(self):
        """Pick the channels to keep powered on the head map."""
        from .headmap import ChannelSelectorDialog
        names = self.names[:self.active_count]
        current = [i for i, box in enumerate(self.on_boxes[:self.active_count]) if box.isChecked()]
        chosen = ChannelSelectorDialog.ask(self, names, current)
        if chosen is not None:
            self.only([names[i] for i in chosen])

    def only(self, keep):
        self.set_settings([hw.ChannelSetting(name in keep, self.bias_boxes[i].isChecked())
                           for i, name in enumerate(self.names[:self.active_count])])

    def no_bias(self):
        self.set_settings([hw.ChannelSetting(box.isChecked(), False)
                           for box in self.on_boxes[:self.active_count]])

    def _changed(self):
        # BIAS on a powered-down channel means nothing; show that, keep the tick.
        for on, bias in zip(self.on_boxes[:self.active_count],
                            self.bias_boxes[:self.active_count]):
            bias.setEnabled(on.isChecked())

    def show_applied(self, settings):
        self.state.setText("Board: " + hw.describe(settings, self.names))

    def show_pending(self, text):
        self.state.setText(text)


class SoftwareChannelGroup(QtWidgets.QGroupBox):
    """Channels in use on a board with no channel commands (the Unicorn).

    Neither BrainFlow nor g.tec's API powers a Unicorn electrode down: g.tec's
    configuration only chooses which channels are transmitted. So this is
    bookkeeping, stated as such: an unticked channel still streams, but it is
    logged as off and left out of the quality table, analysis and games.
    """
    apply_clicked = QtCore.Signal()

    def __init__(self, names, label):
        super().__init__("Channels in use (software) · {}".format(label))
        self.names = list(names)
        outer = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        self.on_boxes = []
        half = (len(self.names) + 1) // 2
        for i, name in enumerate(self.names):
            box = QtWidgets.QCheckBox(name)
            box.setChecked(True)
            grid.addWidget(box, i % half, i // half)
            self.on_boxes.append(box)
        outer.addLayout(grid)
        row = QtWidgets.QHBoxLayout()
        for text, callback in [("All on", self.all_on), ("Only selected…", self.choose_only)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_clicked.emit)
        row.addWidget(self.apply_button)
        outer.addLayout(row)
        note = QtWidgets.QLabel(
            "{} has no channel commands: an unticked channel is not powered down, it still "
            "streams. It is logged as off and left out of the quality table, analysis, brain "
            "map and games.".format(label))
        note.setWordWrap(True)
        note.setStyleSheet(muted())
        outer.addWidget(note)
        self.state = QtWidgets.QLabel("All channels in use")
        self.state.setWordWrap(True)
        outer.addWidget(self.state)

    def settings(self):
        return [hw.ChannelSetting(box.isChecked(), False) for box in self.on_boxes]

    def set_settings(self, settings):
        for box, setting in zip(self.on_boxes, settings):
            box.setChecked(setting.on)

    def on_channels(self):
        return [i for i, box in enumerate(self.on_boxes) if box.isChecked()]

    def all_on(self):
        for box in self.on_boxes:
            box.setChecked(True)

    def choose_only(self):
        from .headmap import ChannelSelectorDialog
        chosen = ChannelSelectorDialog.ask(self, self.names, self.on_channels())
        if chosen is not None:
            self.only([self.names[i] for i in chosen])

    def only(self, keep):
        for name, box in zip(self.names, self.on_boxes):
            box.setChecked(name in keep)

    def show_applied(self, settings):
        off = [name for name, s in zip(self.names, settings) if not s.on]
        self.state.setText("Off (software): " + ", ".join(off) if off else "All channels in use")
