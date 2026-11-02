"""The amplifiers live mode can stream from, and what differs between them.

Everything board-specific that the monitor, the live source and the CSV loader
need sits in one `Board` record, read from BrainFlow's own board description
rather than typed in: rows, rate and electrode names are then whatever the
installed BrainFlow writes to disk.

What BrainFlow cannot tell us is the amplifier's rail and which controls the
board accepts; those are stated here with their source.

No Qt, no I/O - safe to import anywhere.
"""
from dataclasses import dataclass

from brainflow.board_shim import BoardIds, BoardShim


@dataclass(frozen=True)
class Board:
    key: str
    label: str
    board_id: int
    fs: int
    eeg_rows: tuple
    timestamp_row: int
    marker_row: int
    n_rows: int
    names: tuple
    #: amplifier saturation in microvolts, for the "Saturated" quality check
    rail_uv: float
    #: per-channel power/BIAS commands and the ADS1299 lead-off test
    channel_settings: bool
    #: what the connection box asks for
    address_label: str
    address_hint: str
    #: motion sensor rows: accelerometer (g) and, on the Unicorn, gyroscope (°/s)
    accel_rows: tuple = ()
    gyro_rows: tuple = ()

    @property
    def n_channels(self):
        return len(self.eeg_rows)


def _board(key, label, board_id, rail_uv, channel_settings, address_label, address_hint):
    descr = BoardShim.get_board_descr(board_id)
    return Board(
        key=key, label=label, board_id=int(board_id),
        fs=int(descr["sampling_rate"]),
        eeg_rows=tuple(descr["eeg_channels"]),
        timestamp_row=int(descr["timestamp_channel"]),
        marker_row=int(descr["marker_channel"]),
        n_rows=int(descr["num_rows"]),
        names=tuple(descr["eeg_names"].split(",")),
        rail_uv=rail_uv, channel_settings=channel_settings,
        address_label=address_label, address_hint=address_hint,
        accel_rows=tuple(descr.get("accel_channels", ())),
        gyro_rows=tuple(descr.get("gyro_channels", ())))


def motion_layout(board):
    """(rows, names, units) of a board's motion sensor; empty when it has none."""
    rows = list(board.accel_rows) + list(board.gyro_rows)
    names = (["Accel {}".format(a) for a in "XYZ"][:len(board.accel_rows)]
             + ["Gyro {}".format(a) for a in "XYZ"][:len(board.gyro_rows)])
    units = ["g"] * len(board.accel_rows) + ["°/s"] * len(board.gyro_rows)
    return rows, names, units


#: OpenBCI Cyton+Daisy: 16 ch, 125 Hz over the RF dongle. Rail ±187,500 µV is
#: 4.5 V / gain 24 (Cyton data format docs).
CYTON_DAISY = _board(
    "cyton_daisy", "Cyton+Daisy", BoardIds.CYTON_DAISY_BOARD, 187_500.0, True,
    "Port", "COM port of the OpenBCI dongle")

#: OpenBCI Cyton without the Daisy daughter board: eight channels at 250 Hz.
CYTON = _board(
    "cyton", "Cyton only - 8 channels", BoardIds.CYTON_BOARD, 187_500.0, True,
    "Port", "COM port of the OpenBCI dongle")

#: g.tec Unicorn Hybrid Black: 8 ch, 250 Hz over Bluetooth, electrodes at
#: Fz C3 Cz C4 Pz PO7 Oz PO8. Input range ±750 mV (g.tec Unicorn Hybrid Black
#: specifications). BrainFlow streams it through g.tec's Unicorn.dll and offers
#: no channel or impedance commands for it.
UNICORN = _board(
    "unicorn", "Unicorn Hybrid Black", BoardIds.UNICORN_BOARD, 750_000.0, False,
    "Serial", "Unicorn serial (UN-XXXX.XX.XX) · blank = first paired headset")

# Keep the two established live-mode indices stable; Cyton-only is appended.
BOARDS = (CYTON_DAISY, UNICORN, CYTON)

# File-only layouts. Adding a reader must not add unsupported live controls.
SYNTHETIC = _board("synthetic", "Synthetic", BoardIds.SYNTHETIC_BOARD, None, False, "", "")
CSV_BOARDS = (CYTON_DAISY, UNICORN, CYTON, SYNTHETIC)


def csv_board(board_id):
    for board in CSV_BOARDS:
        if board.board_id == board_id:
            return board
    raise ValueError("Unsupported recording device: {}".format(board_id))


def for_csv_columns(n_columns):
    """The board that wrote a BrainFlow CSV with this many columns.

    A BrainFlow export has one column per board row and no header, so the
    A 32-column file can be Daisy or Synthetic; it needs device metadata
    or a usable sample clock. Never guess its rate from the shape alone.
    """
    for board in (UNICORN, CYTON):
        if board.n_rows == n_columns:
            return board
    raise ValueError("Cannot identify a {}-column recording without device metadata".format(n_columns))
