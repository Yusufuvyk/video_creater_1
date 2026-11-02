"""The control sidebar. Owns the widgets and exposes their values; it does
not touch data or plots."""
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from ..core.boards import BOARDS, CYTON, CYTON_DAISY
from ..core.config import CH_NAMES, list_recordings
from .preferences import data_dir
from .channelnames import ChannelNamesDialog
from .filters import FilterGroup
from .hardware_panel import HardwareGroup, SoftwareChannelGroup
from .headmap import ChannelButton
from .theme import muted
from .icons import only, put
from .widgets import NoScrollComboBox, check_box, spin_box


#: Traces are stacked, so beyond this many the rows are a few pixels each and
#: the monitor shows nothing. A 64-electrode cap opens on a readable subset and
#: the head map is there to widen it.
MAX_DEFAULT_TRACES = 16

#: Seconds on the traces, and (by default) the stretch the spectra use: 5 s gives
#: an FFT grid of 0.2 Hz and shows a change - eyes closed - within 5 s.
DEFAULT_WINDOW_S = 5


class PortCombo(NoScrollComboBox):
    def text(self):
        return self.currentText()

    def setText(self, value):
        self.setCurrentText(value)


class ControlPanel(QtWidgets.QWidget):
    start_clicked = QtCore.Signal()
    remap_clicked = QtCore.Signal()
    features_clicked = QtCore.Signal()
    names_changed = QtCore.Signal(object)
    profile_chosen = QtCore.Signal(str)
    profile_save_clicked = QtCore.Signal()
    profile_edit_clicked = QtCore.Signal()
    profile_delete_clicked = QtCore.Signal()

    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self._running = False
        self.setFixedWidth(300)
        # Board defaults until a file says otherwise. A recording carries its
        # own channel names, and they win: labelling a 64-electrode cap
        # EEG1..EEG16 is not a cosmetic problem, it renames the electrodes.
        self.channel_names = list(CH_NAMES)

        layout = QtWidgets.QVBoxLayout(self)
        self.source_bar = self._source_group()
        layout.addWidget(self.remap_button)
        layout.addWidget(self._filter_group())
        layout.addWidget(self._display_group())
        layout.addWidget(self._channel_group())
        self.hardware = HardwareGroup(CH_NAMES)
        layout.addWidget(self.hardware)
        from ..core.boards import UNICORN
        self.unicorn_channels = SoftwareChannelGroup(UNICORN.names, UNICORN.label)
        layout.addWidget(self.unicorn_channels)
        layout.addStretch(1)
        self._sync_mode()

    # ---------------- groups ----------------

    def _source_group(self):
        group = QtWidgets.QWidget()
        form = QtWidgets.QHBoxLayout(group)
        form.setContentsMargins(0, 0, 0, 0)

        self.mode = NoScrollComboBox()
        # Index 1 stays the Cyton+Daisy; each board in BOARDS gets its own entry.
        self.mode.addItems(["File playback"] + ["Live ({})".format(b.label) for b in BOARDS])
        self.mode.currentIndexChanged.connect(self._sync_mode)
        # the longest board name must not set the window's width: shown cut, the list itself opens whole
        self.mode.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.mode.setMinimumContentsLength(16)
        form.addWidget(self.mode)

        self.file_combo = NoScrollComboBox()
        self.refresh_files()
        self.file_combo.setMaximumWidth(260)
        self.file_combo.setMinimumWidth(120)
        form.addWidget(self.file_combo)

        browse = only(QtWidgets.QToolButton(), "folder-open", "Browse", more="Open a recording from any folder")
        self.browse_button = browse
        browse.clicked.connect(self._browse)
        form.addWidget(browse)

        self.port = PortCombo()
        self.port.setEditable(True)
        self.port.setMaximumWidth(150)
        self.port.setCurrentText('COM7')
        #: what the connection box holds for each board: a COM port, a serial number
        self.addresses = {b.key: '' for b in BOARDS}
        self.addresses[CYTON_DAISY.key] = 'COM7'
        self.addresses[CYTON.key] = 'COM7'
        self._address_board = CYTON_DAISY
        form.addWidget(self.port)
        self.refresh_port_button = only(QtWidgets.QToolButton(), "refresh-cw", "Refresh ports")
        self.refresh_port_button.clicked.connect(self.refresh_ports)
        form.addWidget(self.refresh_port_button)
        self.refresh_ports()

        self.remap_button = QtWidgets.QPushButton("Mat layout...")
        self.remap_button.setToolTip(
            "Change how a .mat file is read: which condition, which rows are "
            "EEG, what the units are.")
        self.remap_button.clicked.connect(self.remap_clicked.emit)
        self.file_combo.currentIndexChanged.connect(self._sync_remap)
        self._sync_remap()

        # one button, like the Workbench's Run: Start, Stop while streaming (show_start)
        self.start_button = QtWidgets.QPushButton()
        self.start_button.setProperty("primary", True)
        self.show_start(False)
        self.start_button.clicked.connect(self.start_clicked.emit)
        form.addWidget(self.start_button)

        features = QtWidgets.QPushButton("Live features...")
        features.setToolTip(
            "Band powers, Hjorth descriptors and spectral shape, recomputed "
            "while streaming - and loggable to CSV.")
        features.clicked.connect(self.features_clicked.emit)
        self.features_button = features
        form.addWidget(features)
        form.addStretch(1)

        return group

    def show_start(self, running):
        """The Start button's face: ▶ Start, or ■ Stop while streaming; an icon alone, its name on hover."""
        word = "Stop" if running else "Start"
        put(self.start_button, "square" if running else "play", "", primary=True)
        self.start_button.setAccessibleName(word)
        self.start_button.setToolTip(word)
        self.start_button.setProperty("icon_only", True)

    def _filter_group(self):
        # 3 Hz rather than the analysis page's 1 Hz: a live electrode
        # drifts, and the drift is what the monitor has to stay readable
        # through.
        self.filters = FilterGroup(self.fs, low=3.0)
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QtWidgets.QPushButton("Filter settings…")
        button.setCheckable(True)
        button.toggled.connect(self.filters.setVisible)
        layout.addWidget(button)
        summary = QtWidgets.QLabel()
        summary.setWordWrap(True)
        summary.setStyleSheet(muted())
        layout.addWidget(summary)

        def describe():
            settings = self.filters.kwargs
            summary.setText(f"Causal filters · {settings.get('low')}–{settings.get('high')} Hz · "
                            f"notch {settings.get('notches', [])} Hz")
        self.filters.changed.connect(describe)
        layout.addWidget(self.filters)
        self.filters.hide()
        describe()
        return wrapper

    def _display_group(self):
        group = QtWidgets.QGroupBox("Display")
        form = QtWidgets.QFormLayout(group)

        self.window_seconds = spin_box(1, 60, DEFAULT_WINDOW_S, 1, 0, " s")
        self.window_seconds.setToolTip(
            "Seconds of signal on the traces, and the stretch the FFT/PSD spectra are computed "
            "from. Longer: finer frequency resolution (Δf = 1 / window) but a slower reaction. "
            "The quality table always uses the last 4 s.")
        form.addRow("Window", self.window_seconds)

        self.auto_amplitude = QtWidgets.QCheckBox("Auto / channel")
        self.auto_amplitude.setToolTip("Each channel has an independent display scale, shown beside its trace. "
                                       "Power calculations use the original microvolt samples.")
        self.auto_amplitude.setChecked(True)
        self.auto_amplitude.toggled.connect(self._sync_amplitude)
        self.amplitude = spin_box(1, 200_000, 100, 10, 0, " uV")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.amplitude)
        row.addWidget(self.auto_amplitude)
        form.addRow("Amplitude", row)

        self.psd_channel = NoScrollComboBox()
        self.psd_channel.addItems(self.channel_names)

        # A checkbox, matching the analysis page. The same idea should not be a
        # dropdown here and a tick box there.
        self.log_scale = check_box(
            "Logarithmic power axis", True,
            "Power spans decades; a linear axis hides everything below the peak.")
        self.log_scale.hide()  # Spectrum controls live beside the graphs.

        self._sync_amplitude()
        return group

    def _channel_group(self):
        group = QtWidgets.QGroupBox("Channels")
        outer = QtWidgets.QVBoxLayout(group)

        # Cyton+Daisy wiring: pin names plus which pins are powered. The last
        # one picked is reopened, so a headset is described once, not per Start.
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Profile"))
        self.profile_combo = NoScrollComboBox()
        self.profile_combo.setToolTip(
            "Cyton+Daisy channel profile: which electrode is on which pin and which "
            "pins are on. Picking one applies it; the last one opens next time.")
        self.profile_combo.activated.connect(
            lambda _: self.profile_chosen.emit(self.profile_combo.currentText()))
        row.addWidget(self.profile_combo, 1)
        from .profile_box import menu_button
        self.profile_menu = menu_button("Edit, save or delete channel profiles", [
            ("Edit profiles…", self.profile_edit_clicked.emit),
            ("Save current as profile…", self.profile_save_clicked.emit),
            ("Delete this profile", self.profile_delete_clicked.emit)])
        row.addWidget(self.profile_menu)
        outer.addLayout(row)

        self.channel_button = ChannelButton()
        self.channel_button.set_channels(self.channel_names,
                                         range(len(self.channel_names)))
        outer.addWidget(self.channel_button)

        rename = QtWidgets.QPushButton("Channel names...")
        rename.setToolTip(
            "A BrainFlow CSV stores no electrode names. Say which electrode is "
            "on which channel and it is remembered beside the recording.")
        rename.clicked.connect(self._rename_channels)
        outer.addWidget(rename)

        self.bad_label = QtWidgets.QLabel("-")
        self.bad_label.setWordWrap(True)
        self.bad_label.setStyleSheet(muted())
        outer.addWidget(self.bad_label)
        return group

    # ---------------- reactions ----------------

    def _sync_mode(self):
        live = self.is_live
        board = self.board
        if board is not None and board is not self._address_board:
            self._swap_address(board)
        serial = board is not None and not board.channel_settings
        self.port.setEnabled(live and not self._running)
        self.refresh_port_button.setEnabled(live and not serial and not self._running)
        self.file_combo.setEnabled(not live and not self._running)
        self.browse_button.setEnabled(not live and not self._running)
        # Only what the chosen source uses is shown: a file has no port, a board
        # no file list. Greyed-out boxes for the other kind of source were most
        # of what made this bar hard to read.
        self.file_combo.setVisible(not live)
        self.browse_button.setVisible(not live)
        self.port.setVisible(live)
        self.refresh_port_button.setVisible(live and not serial)
        # Channel power/BIAS and lead-off are Cyton SDK commands.
        self.hardware.setEnabled(live and board.channel_settings)
        self.hardware.set_channel_count(
            board.n_channels if board is not None and board.channel_settings
            else len(self.hardware.names))
        # The Cyton's channel commands, or the Unicorn's software choice; a file has neither.
        software = board is not None and not board.channel_settings
        self.hardware.setVisible(live and not software)
        self.unicorn_channels.setVisible(software)
        self.hardware.setToolTip("" if board is None or board.channel_settings else
                                 "{} has no channel or impedance commands in BrainFlow.".format(board.label))
        self._sync_remap()

    def _swap_address(self, board):
        """Keep the Cyton's COM port and the Unicorn's serial apart in one box."""
        self.addresses[self._address_board.key] = self.port.text().strip()
        self._address_board = board
        self.port.setToolTip("{}: {}".format(board.address_label, board.address_hint))
        self.port.lineEdit().setPlaceholderText(board.address_label)
        if board.channel_settings:
            self.port.setText(self.addresses[board.key])
            self.refresh_ports()
        else:
            self.port.clear()
            self.port.setText(self.addresses[board.key])

    def set_address(self, board, text):
        self.addresses[board.key] = text
        if board is self._address_board:
            self.port.setText(text)

    def address(self, board):
        """What was typed for `board`, whether or not it is selected now."""
        if board is self._address_board:
            return self.port.text().strip()
        return self.addresses[board.key]

    def refresh_ports(self):
        if not self._address_board.channel_settings:
            return
        wanted = self.port.text()
        try:
            from PySide6.QtSerialPort import QSerialPortInfo
            ports = [p.portName() for p in QSerialPortInfo.availablePorts()]
        except ImportError:
            ports = []
        self.port.clear()
        self.port.addItems(ports)
        self.port.setCurrentText(wanted or (ports[0] if ports else 'COM7'))

    def set_running(self, running):
        self._running = running
        self.mode.setEnabled(not running)
        self._sync_mode()

    def _sync_remap(self):
        """Only a .mat has a layout to argue about."""
        path = self.selected_file
        self.remap_button.setEnabled(
            not self._running and not self.is_live and bool(path) and str(path).lower().endswith(".mat"))

    def _sync_amplitude(self):
        self.amplitude.setEnabled(not self.auto_amplitude.isChecked())

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open recording", str(data_dir()),
            "Recordings (*.csv *.mat);;All files (*)")
        if path:
            self.file_combo.addItem(Path(path).name, path)
            self.file_combo.setCurrentIndex(self.file_combo.count() - 1)

    def refresh_files(self):
        self.file_combo.clear()
        for path in list_recordings(data_dir()):
            self.file_combo.addItem(path.name, str(path))

    # ---------------- values ----------------

    def set_sampling_rate(self, fs):
        """Follow the loaded file: filter edges above Nyquist are meaningless."""
        self.fs = fs
        self.filters.set_sampling_rate(fs)

    def _rename_channels(self):
        names = ChannelNamesDialog.ask(self, self.channel_names)
        if names is not None:
            self.set_channels(names)
            self.names_changed.emit(names)

    def set_profiles(self, names, current):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(names)
        self.profile_combo.setCurrentText(current)
        self.profile_combo.blockSignals(False)

    def set_channels(self, names):
        """Follow the loaded file's channels, however many and whatever named."""
        self.channel_names = list(names)
        keep = [i for i in self.channel_button.selected if i < len(names)]
        self.channel_button.set_channels(
            self.channel_names,
            keep or list(range(len(names)))[:MAX_DEFAULT_TRACES])

        wanted = self.psd_channel.currentText()
        self.psd_channel.blockSignals(True)
        self.psd_channel.clear()
        self.psd_channel.addItems(self.channel_names)
        if wanted in self.channel_names:
            self.psd_channel.setCurrentText(wanted)
        self.psd_channel.blockSignals(False)

    @property
    def is_live(self):
        return self.mode.currentIndex() >= 1

    @property
    def board(self):
        """The live board chosen, or None in file playback."""
        index = self.mode.currentIndex()
        return BOARDS[index - 1] if 1 <= index <= len(BOARDS) else None

    @property
    def selected_file(self):
        return self.file_combo.currentData()

    @property
    def log_psd(self):
        return self.log_scale.isChecked()

    @property
    def selected_channels(self):
        return self.channel_button.selected

    @property
    def filter_kwargs(self):
        return self.filters.kwargs
