"""Where samples come from: a recorded file, or the board itself.

A `Recording` carries its own sampling rate, channel count and channel names.
Nothing downstream may assume the board's 125 Hz: a file from another system
would still load, and every duration, frequency axis and filter edge would be
quietly wrong.

Live capture keeps using the board constants, because there the board *is* the
authority.
"""
import time
import json
from pathlib import Path

import numpy as np

from . import boards
from .config import FS
from .hardware import apply_settings
from .recording_clock import estimated_rate, sample_positions, event_seconds


class Recording:
    """A loaded EEG file: microvolts, plus everything needed to interpret them."""

    __slots__ = ("data", "fs", "channel_names", "timestamps", "path", "rail_uv", "events", "gap_samples",
                 "events_are_raw_markers", "channel_log", "motion", "motion_names", "motion_units",
                 "original_fs", "unit_note", "bad_spans", "reading")

    def __init__(self, data, fs, channel_names=None, timestamps=None, path=None,
                 rail_uv=None, events=None):
        self.data = np.asarray(data, dtype=float)
        self.fs = float(fs)
        self.channel_names = list(channel_names) if channel_names is not None else [
            "CH{}".format(i + 1) for i in range(self.data.shape[0])]
        self.timestamps = timestamps
        self.path = Path(path) if path else None
        #: amplifier saturation in µV when the file's board is known, else None
        self.rail_uv = rail_uv
        #: core.epochs.EventTable read from the file itself, or None
        self.events = events
        self.gap_samples = 0
        #: True when `events` are bare codes from a BrainFlow marker channel;
        #: a labelled sidecar beside the file then describes them better.
        self.events_are_raw_markers = False
        #: [(unix time, on/off per channel)] from the live session's hardware
        #: log, or None: a powered-down channel records ~0 µV, not EEG
        self.channel_log = None
        #: motion sensor samples (n_axes, n_samples) on the EEG's clock, or None
        self.motion = None
        self.motion_names = []
        self.motion_units = []
        #: the file's own rate when this recording was downsampled on opening, else None
        self.original_fs = None
        #: how the file's numbers became µV, in words ("BrainVision header: each value × 0.0406901 µV"); "" unknown
        self.unit_note = ""
        #: [(start s, end s, label)] the file marks as not to be used (MNE's BAD_ / EDGE / boundary annotations;
        #: a boundary has no length): a trial over one is left out
        self.bad_spans = []
        #: What the program did on its own while reading the file, in words, for the user and the report: which
        #: layout, which runs or channels were left out, where the events and the bad marks came from
        self.reading = []

    @property
    def n_channels(self):
        return self.data.shape[0]

    @property
    def n_samples(self):
        return self.data.shape[1]

    @property
    def duration(self):
        return self.n_samples / self.fs

    @property
    def start_time(self):
        """First sample's unix timestamp, or None if the file has no clock."""
        if self.timestamps is None or len(self.timestamps) == 0:
            return None
        valid = np.flatnonzero(np.isfinite(self.timestamps))
        return float(self.timestamps[valid[0]] - valid[0] / self.fs) if valid.size else None

    def describe(self):
        return "{} channels, {:.1f} s at {:g} Hz, uV".format(
            self.n_channels, self.duration, self.fs) + (
                " · downsampled from {:g} Hz".format(self.original_fs) if self.original_fs else "") + (
                " · {} missing samples preserved".format(self.gap_samples) if self.gap_samples else "")

    def marked_bad(self, start, end):
        """The label of the first span the file marks as bad between start and end (seconds), else None."""
        for first, last, label in self.bad_spans:
            if first < end and (last > start if last > first else first >= start):
                return label
        return None

    def switched_off(self, start=0.0, end=None):
        """Channels powered down at any moment between start and end (seconds)."""
        if not self.channel_log:
            return set()
        end = self.duration if end is None else end
        states = sorted(((self.marker_seconds(t) if self.start_time is not None else 0.0), on)
                        for t, on in self.channel_log)
        current, off = states[0][1], set()      # before the first entry: its state
        for seconds, on in states:
            if seconds <= start:
                current = on
            elif seconds <= end:
                off |= {i for i, o in enumerate(on) if not o}
        off |= {i for i, o in enumerate(current) if not o}
        return {i for i in off if i < self.n_channels}

    def marker_seconds(self, when):
        if self.start_time is None:
            raise ValueError("Recording has no clock for Unix-time markers")
        return event_seconds(self.timestamps, self.fs, when, self.start_time)


# ----------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------


def board_metadata_path(path):
    return Path(path).with_suffix(".board.json")


def load_channel_log(record_path, n_channels):
    """[(unix time, on per channel)] from a live recording's hardware log, or None.

    Older Unicorn logs carry '-' (no channel choice was logged) and give None.
    """
    path = hardware_log_path(record_path)
    if not path.is_file():
        return None
    entries = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        pairs = fields[3].split(";")
        if len(pairs) != n_channels or any(len(p) != 2 or p[0] not in "01" for p in pairs):
            continue
        try:
            entries.append((float(fields[0]), tuple(p[0] == "1" for p in pairs)))
        except ValueError:
            continue
    return entries or None


def write_board_metadata(path, board):
    """Keep the device identity alongside every newly streamed recording."""
    board_metadata_path(path).write_text(json.dumps(dict(
        version=1, board_id=board.board_id, sampling_rate=board.fs, columns=board.n_rows)), encoding="utf-8")


def _csv_board(path, frame, board_id):
    metadata = board_metadata_path(path)
    if metadata.is_file():
        try:
            raw = json.loads(metadata.read_text(encoding="utf-8"))
            board = boards.csv_board(raw["board_id"])
            if raw.get("version") != 1 or raw.get("sampling_rate") != board.fs or raw.get("columns") != board.n_rows:
                raise ValueError("Device metadata does not match its layout")
            if board_id is not None and board.board_id != board_id:
                raise ValueError("Selected device conflicts with recording metadata")
            board_id = board.board_id
        except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Cannot read recording device metadata: {}".format(exc)) from exc
    if board_id is not None:
        board = boards.csv_board(board_id)
    elif frame.shape[1] == 32:
        rate = estimated_rate(frame.iloc[:, boards.CYTON_DAISY.timestamp_row])
        choices = [b for b in (boards.CYTON_DAISY, boards.SYNTHETIC)
                   if rate is not None and abs(rate / b.fs - 1) < .1]
        if len(choices) != 1:
            raise ValueError("This 32-column recording has no reliable device/sample-rate information")
        board = choices[0]
    else:
        board = boards.for_csv_columns(frame.shape[1])
    if frame.shape[1] != board.n_rows:
        raise ValueError("{} requires {} columns, but this file has {}".format(board.label, board.n_rows, frame.shape[1]))
    rate = estimated_rate(frame.iloc[:, board.timestamp_row])
    if rate is not None and abs(rate / board.fs - 1) >= .1:
        raise ValueError("Recording clock does not match the {} Hz device rate".format(board.fs))
    return board


def load_brainflow_csv(path, *, board_id=None):
    """Tab-separated BrainFlow export with device identity and sample-clock gaps.

    Metadata or an explicit board_id wins. Legacy 32-column recordings need
    enough clock information to distinguish Daisy (125 Hz) from Synthetic (250).
    """
    import pandas as pd            # here, not at the top: importing it costs 0.4 s at every start
    frame = pd.read_csv(path, sep="\t", header=None, skip_blank_lines=False)
    # Keep one row per original sample, including damaged EEG samples. Dropping
    # rows shifts all later markers and corrupts the sampling clock.
    frame = frame.apply(pd.to_numeric, errors="coerce")
    board = _csv_board(path, frame, board_id)

    data = frame.iloc[:, list(board.eeg_rows)].to_numpy().T.astype(float)
    motion_rows, motion_names, motion_units = boards.motion_layout(board)
    motion = frame.iloc[:, motion_rows].to_numpy().T.astype(float) if motion_rows else None

    # Kept so event markers, stamped in unix time, can be placed against
    # samples. Older exports may not have the column.
    timestamps = frame.iloc[:, board.timestamp_row].to_numpy().astype(float)
    step = 2 if board is boards.CYTON_DAISY else 1 if board is boards.CYTON else None
    positions = sample_positions(timestamps, board.fs, frame.iloc[:, 0], step)
    added = int(positions[-1] + 1 - len(positions)) if positions.size else 0
    if added:
        padded = np.full((data.shape[0], int(positions[-1]) + 1), np.nan)
        padded[:, positions] = data
        padded_times = np.full(padded.shape[1], np.nan)
        padded_times[positions] = timestamps
        data, timestamps = padded, padded_times
        if motion is not None:
            padded_motion = np.full((motion.shape[0], padded.shape[1]), np.nan)
            padded_motion[:, positions] = motion
            motion = padded_motion

    # The board's electrode names, as live capture uses. A `.channels.json`
    # sidecar beside the file overrides them when a session used a different
    # layout.
    names = (list(board.names) if data.shape[0] == len(board.names)
             else ["EEG{}".format(i + 1) for i in range(data.shape[0])])
    # Marker pulses are instants, not inferred trial durations. Labels remain
    # raw codes for the Analysis page's existing event mapping controls.
    from .epochs import EventTable
    markers = frame.iloc[:, board.marker_row].to_numpy()
    present = np.isfinite(markers) & (markers != 0)
    events = EventTable(positions[present] / board.fs, None, [format(v, "g") for v in markers[present]])
    recording = Recording(data, board.fs, names, timestamps, path, board.rail_uv, events)
    recording.gap_samples = added
    recording.events_are_raw_markers = True
    recording.unit_note = "BrainFlow CSV: µV as BrainFlow writes them, not rescaled"
    recording.channel_log = load_channel_log(path, data.shape[0])
    if motion is not None:
        recording.motion, recording.motion_names, recording.motion_units = motion, motion_names, motion_units
    return recording


# ----------------------------------------------------------------------
# downsampling
# ----------------------------------------------------------------------

#: Anti-alias filter length on each side, in output samples (as scipy's resample_poly).
DOWNSAMPLE_HALF_TAPS = 10


def downsample_factor(fs, target_hz):
    """The whole number to divide `fs` by so the rate stays at or above `target_hz`.

    1 when nothing is to be done: no target, or a file already at or below it.
    A whole factor keeps every kept sample on the original clock, so markers
    and gaps stay where they were (500 Hz, target 250 -> 2; 512 Hz -> 2, 256 Hz).
    """
    if not target_hz or not np.isfinite(target_hz) or target_hz <= 0 or not np.isfinite(fs):
        return 1
    return max(1, int(np.floor(fs / float(target_hz) + 1e-9)))


def _decimate(block, factor):
    """Low-pass at the new Nyquist (zero phase), then every `factor`-th sample.

    Finite runs are filtered on their own, mirrored at their ends, so a gap of
    missing samples neither spreads nor rings into its neighbours.
    """
    from scipy.ndimage import convolve1d
    from scipy.signal import firwin

    block = np.asarray(block, dtype=float)
    taps = firwin(2 * DOWNSAMPLE_HALF_TAPS * factor + 1, 1.0 / factor, window=("kaiser", 5.0))
    finite = np.isfinite(block)
    if finite.all():
        return convolve1d(block, taps, axis=-1, mode="mirror")[:, ::factor]
    from .processing import finite_runs
    out = np.full(block.shape, np.nan)
    for row, mask, target in zip(block, finite, out):
        for start, end in finite_runs(mask):
            target[start:end] = convolve1d(row[start:end], taps, mode="mirror")
    return out[:, ::factor]


def downsample(recording, target_hz):
    """`recording` at a lower rate, or `recording` itself when there is nothing to do.

    The file is never touched. Events are in seconds and keep their times; the
    clock keeps every `factor`-th stamp; the motion sensor is reduced the same way.
    """
    factor = downsample_factor(recording.fs, target_hz)
    if factor < 2 or recording.n_samples < 2 * factor:
        return recording
    stamps = recording.timestamps
    reduced = Recording(_decimate(recording.data, factor), recording.fs / factor, recording.channel_names,
                        None if stamps is None else np.asarray(stamps)[::factor], recording.path,
                        recording.rail_uv, recording.events)
    reduced.original_fs = recording.original_fs or recording.fs
    reduced.gap_samples = (int(np.count_nonzero(~np.isfinite(reduced.data).all(axis=0)))
                           if recording.gap_samples else 0)
    reduced.events_are_raw_markers = recording.events_are_raw_markers
    reduced.channel_log = recording.channel_log
    reduced.unit_note = recording.unit_note
    reduced.bad_spans = list(recording.bad_spans)
    reduced.reading = list(recording.reading) + ["Downsampled from {:g} Hz to {:g} Hz on opening (Settings, "
                                                 "Analysis).".format(recording.fs, reduced.fs)]
    if recording.motion is not None:
        reduced.motion = _decimate(recording.motion, factor)
        reduced.motion_names = list(recording.motion_names)
        reduced.motion_units = list(recording.motion_units)
    return reduced


# ----------------------------------------------------------------------
# playback
# ----------------------------------------------------------------------


class EEGSource:
    """Interface shared by every source."""

    #: total samples available, or 0 when the stream is unbounded
    n_total = 0
    fs = FS

    def read(self, n):
        raise NotImplementedError

    def seek(self, sample):
        pass

    def close(self):
        pass


class FileSource(EEGSource):
    """Replays a `Recording` at real time."""

    def __init__(self, recording):
        # A Recording, never a path: choosing how to read a file is the
        # formats package's job, and doing it here is what made `core` depend
        # on the very layer that depends on it.
        if not isinstance(recording, Recording):
            raise TypeError(
                "FileSource needs a Recording; use formats.recordings."
                "load_recording(path) to make one")
        self.recording = recording
        self.pos = 0
        #: motion samples of the latest read, or None when the file has none
        self.last_motion = None

    # Reading through to the recording keeps older callers working unchanged.
    @property
    def data(self):
        return self.recording.data

    @property
    def fs(self):
        return self.recording.fs

    @property
    def timestamps(self):
        return self.recording.timestamps

    @property
    def start_time(self):
        return self.recording.start_time

    @property
    def n_total(self):
        return self.recording.n_samples

    @property
    def motion_names(self):
        return list(self.recording.motion_names) if self.recording.motion is not None else []

    @property
    def motion_units(self):
        return list(self.recording.motion_units) if self.recording.motion is not None else []

    def seek(self, sample):
        self.pos = int(np.clip(sample, 0, self.n_total))

    def read(self, n):
        if self.pos >= self.n_total:
            self.last_motion = None
            return np.empty((self.recording.n_channels, 0))
        end = min(self.pos + n, self.n_total)
        chunk = self.recording.data[:, self.pos:end]
        motion = self.recording.motion
        self.last_motion = motion[:, self.pos:end] if motion is not None else None
        self.pos = end
        return chunk


def ascii_stand_in(path):
    """An ASCII-only file for BrainFlow to write `path` into, or None when `path` is fine.

    BrainFlow's C++ file streamer cannot open a Windows path with non-ASCII
    letters (Masaüstü, LAST_VERSİON): it fails with "unable to add streamer".
    The caller streams into the stand-in and moves it to `path` afterwards.
    """
    path = Path(path)
    if str(path).isascii():
        return None
    import os
    import tempfile
    for folder in (Path(tempfile.gettempdir()), Path(os.environ.get("PUBLIC", "C:/Users/Public"))):
        folder = folder / "EEGStudio_recording"
        if not str(folder).isascii():
            continue
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        stem = path.stem.encode("ascii", "replace").decode().replace("?", "_")
        return folder / "{}_{}{}".format(stem, int(time.time() * 1000), path.suffix)
    raise RuntimeError("no folder with plain (ASCII) letters to record into; move the program "
                       "or game to a folder without letters such as ü or İ")


#: Beside the ASCII stand-ins: one line per recording that could not be moved to its folder.
STRANDED_LOG = "NOT_MOVED.txt"


def _note_stranded(stand_in, target, error):
    """A lasting note, beside the stand-in, of a recording left there; the window may be long closed."""
    try:
        with open(Path(stand_in).with_name(STRANDED_LOG), "a", encoding="utf-8") as f:
            f.write("{}\t{}\t-> {}\t{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), stand_in, target, error))
    except OSError:
        pass    # the window still shows where the file is


def live_record_path(directory, prefix="live"):
    """A new, unused raw-recording name for a live session started now.

    `prefix` is the operator's (Settings -> Recording): the timestamp stays, so
    recordings still sort by time and never overwrite one another.
    """
    prefix = safe_stem(prefix) or "live"
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(directory) / "{}_{}.csv".format(prefix, stamp)
    n = 1
    while path.exists() or hardware_log_path(path).exists() or board_metadata_path(path).exists():
        n += 1
        path = Path(directory) / "{}_{}_{}.csv".format(prefix, stamp, n)
    return path


#: What may be typed into a file name. Anything else (slashes, colons, quotes)
#: either cannot be written on Windows or would point somewhere else entirely.
_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ")


def safe_stem(text):
    """`text` reduced to what a file name may hold, or '' if nothing is left."""
    cleaned = "".join(c for c in str(text) if c in _NAME_OK).strip(" .")
    return cleaned


#: Files written beside a recording that belong to it. A rename that leaves one
#: behind loses the channel names, the events or the hardware log.
SIDECAR_SUFFIXES = (".hardware.tsv", ".board.json", ".events.json", ".eventmap.json",
                    ".paradigm.json", ".channels.json", ".mapping.json", "_events.csv")


def sidecar_paths(path):
    """Every existing file that belongs to this recording, the recording aside."""
    path = Path(path)
    found = []
    for suffix in SIDECAR_SUFFIXES:
        candidate = path.with_name(path.stem + suffix)
        if candidate.is_file() and candidate != path:
            found.append(candidate)
    return found


def rename_recording(path, new_stem):
    """Rename a recording and its sidecars together; returns the new path.

    Nothing is moved unless every target is free: half a renamed recording is
    worse than none, because the leftovers still carry the old name.
    """
    path = Path(path)
    stem = safe_stem(new_stem)
    if not stem:
        raise ValueError("A recording needs a name")
    if stem == path.stem:
        return path
    moves = [(path, path.with_name(stem + path.suffix))]
    moves += [(s, s.with_name(stem + s.name[len(path.stem):])) for s in sidecar_paths(path)]
    taken = [str(target.name) for _, target in moves if target.exists()]
    if taken:
        raise ValueError("Already there: " + ", ".join(taken))
    done = []
    try:
        for source, target in moves:
            source.rename(target)
            done.append((source, target))
    except OSError as exc:
        for source, target in reversed(done):       # put back what was moved
            try:
                target.rename(source)
            except OSError:
                pass
        raise ValueError("Cannot rename: {}".format(exc)) from exc
    return moves[0][1]


def hardware_log_path(record_path):
    """Sidecar holding which channels were on / in BIAS, and from when.

    Not .csv, so the recording list never offers it as a recording.
    """
    record_path = Path(record_path)
    return record_path.with_name(record_path.stem + ".hardware.tsv")


class LiveSource(EEGSource):
    """Streams from a live board (Cyton+Daisy or Unicorn). The board sets the rate.

    `address` is the Cyton dongle's serial port, or the Unicorn's serial number
    (blank: BrainFlow opens the first paired Unicorn).

    With `record_dir`, every sample the board sends is written to a BrainFlow
    CSV there (the same tab-separated layout `load_brainflow_csv` reads), and
    each amplifier setting that landed is logged beside it with its unix time.
    A live session that is not on disk cannot be checked afterwards, so a
    recording that cannot be opened stops the start instead of being skipped.
    """

    def __init__(self, address, hardware=None, record_dir=None, board=boards.CYTON_DAISY,
                 record_prefix="live", names=None):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams

        self.info = board
        # Electrode names for the logs: the operator's channel profile when it
        # fits the board, else the board's own.
        self.names = (list(names) if names is not None and len(names) == len(board.names)
                      else list(board.names))
        self.fs = board.fs
        self.motion_rows, self.motion_names, self.motion_units = boards.motion_layout(board)
        self.last_motion = None
        params = BrainFlowInputParams()
        if board.channel_settings:
            params.serial_port = address
        else:
            params.serial_number = address
        self.board = BoardShim(board.board_id, params)
        self.record_path = None
        self._stream_path = None
        try:
            self.board.prepare_session()
            # After prepare_session, which resets the board to its defaults.
            # A reset auto-detects an attached Daisy and enables its ADS1299.
            # Cyton-only must explicitly detach it in firmware: `c` sends the
            # Daisy ADS1299 RESET then STANDBY, which also leaves CONFIG3
            # PD_BIAS=0.  The on-board Cyton ADS1299 is not reset or stopped.
            if board is boards.CYTON:
                from . import hardware as hw
                response = hw._send(self.board, "c")
                if "failure" in response.lower():
                    raise RuntimeError("board refused Cyton-only mode: {}".format(
                        response.strip()))
            if hardware is not None and board.channel_settings:
                apply_settings(self.board, hardware)
            if record_dir is not None:
                Path(record_dir).mkdir(parents=True, exist_ok=True)
                self.record_path = live_record_path(record_dir, record_prefix)
                write_board_metadata(self.record_path, board)
                # BrainFlow writes from its acquisition thread, so the file does
                # not depend on how often the display calls read(). A path it
                # cannot open (non-ASCII letters) is written beside, then moved.
                self._stream_path = ascii_stand_in(self.record_path)
                self.board.add_streamer("file://{}:w".format(self._stream_path or self.record_path))
                self._log_hardware(hardware, "start {} {}={}".format(
                    board.key, board.address_label.lower(), address or "auto"))
            self.board.start_stream()
        except Exception:
            self.close()
            raise

    def _names(self):
        """Profile names when given, else the board's own (also for bare instances)."""
        return getattr(self, "names", None) or list(self.info.names)

    def _log_hardware(self, hardware, note):
        if self.record_path is None:
            return
        from . import hardware as hw
        path = hardware_log_path(self.record_path)
        new = not path.exists()
        if self.info.channel_settings:
            settings = (hardware if hardware is not None
                        else hw.default_settings(self.info.n_channels))
            encoded, summary = hw.encode(settings), hw.describe(settings, self._names())
        elif hardware is not None and len(hardware) == self.info.n_channels:
            # No channel commands on this board: the choice is bookkeeping, and
            # is logged in the same form so analysis leaves those channels out.
            off = [name for name, s in zip(self._names(), hardware) if not s.on]
            encoded = hw.encode(hardware)
            summary = "{}: software only (no channel commands) · off: {}".format(
                self.info.label, ", ".join(off) or "none")
        else:
            encoded, summary = "-", "{}: no channel settings".format(self.info.label)
        with path.open("a", encoding="utf-8", newline="") as handle:
            if new:
                handle.write("unix_time\tlocal_time\tnote\ton_bias_per_channel\tsummary\n")
            handle.write("{:.3f}\t{}\t{}\t{}\t{}\n".format(
                time.time(), time.strftime("%H:%M:%S"), note, encoded, summary))

    def _require_channel_settings(self):
        if not self.info.channel_settings:
            raise RuntimeError("{} has no channel or impedance commands".format(self.info.label))

    def configure(self, hardware, previous=None):
        """Change channel power/BIAS while live: pause, send, resume.

        The pause is short and the monitor's filters settle through it; the
        samples on either side belong to different amplifier states anyway.
        With `previous`, only channels that changed are sent.
        """
        self._require_channel_settings()
        self.board.stop_stream()
        time.sleep(0.2)  # let the last packets arrive before commands read replies
        try:
            sent = apply_settings(self.board, hardware, previous)
            self._log_hardware(hardware, "applied ({} command{})".format(
                len(sent), "" if len(sent) == 1 else "s"))
        finally:
            self.board.start_stream()

    def impedance_step(self, previous, channel, hardware):
        """End the lead-off test on `previous`, start it on `channel`.

        Either may be None. The finished channel gets the operator's own
        setting back (not the board default the OpenBCI GUI restores, which
        silently switches every channel on and into BIAS).
        """
        from . import hardware as hw
        from . import impedance as imp
        self._require_channel_settings()
        settings = (hardware if hardware is not None
                    else hw.default_settings(self.info.n_channels))
        self.board.stop_stream()
        time.sleep(0.2)
        try:
            if previous is not None:
                for command in (imp.stop_command(previous),
                                hw.channel_command(previous, settings[previous])):
                    reply = hw._send(self.board, command)
                    if "failure" in reply.lower():
                        raise RuntimeError("board refused {}: {}".format(command, reply.strip()))
            if channel is not None:
                command = imp.test_command(channel)
                reply = hw._send(self.board, command)
                if "failure" in reply.lower():
                    raise RuntimeError("board refused {}: {}".format(command, reply.strip()))
                self._log_hardware(settings, "impedance test {}".format(self._names()[channel]))
            else:
                self._log_hardware(settings, "impedance test ended")
        finally:
            self.board.start_stream()

    def internal_short(self, enabled, hardware):
        """Enter or leave the ADS1299 input-short diagnostic on all channels.

        The test state is never left as the operator's configuration: leaving
        restores every channel, including channels that were powered down.
        """
        from . import hardware as hw
        from . import shorted_test
        self._require_channel_settings()
        settings = (hardware if hardware is not None
                    else hw.default_settings(self.info.n_channels))
        self.board.stop_stream()
        time.sleep(0.2)
        try:
            if enabled:
                for index in range(self.info.n_channels):
                    command = shorted_test.channel_command(index)
                    reply = hw._send(self.board, command)
                    if "failure" in reply.lower():
                        raise RuntimeError("board refused {}: {}".format(command, reply.strip()))
                self._log_hardware(settings, "ADS1299 internal short test started")
            else:
                hw.apply_settings(self.board, settings)
                self._log_hardware(settings, "ADS1299 internal short test ended")
        except Exception:
            if enabled:
                try:
                    hw.apply_settings(self.board, settings)
                except Exception:
                    pass
            raise
        finally:
            self.board.start_stream()

    def log_note(self, note, hardware):
        """Free text in the hardware log (e.g. impedance results)."""
        self._log_hardware(hardware, note)

    def read(self, n):
        data = self.board.get_board_data()
        if data.size == 0:
            self.last_motion = None
            return np.empty((self.info.n_channels, 0))
        self.last_motion = data[self.motion_rows, :] if self.motion_rows else None
        return data[list(self.info.eeg_rows), :]

    def close(self):
        try:
            self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass  # already down, or never came up
        stand_in = getattr(self, "_stream_path", None)
        self.move_error = None
        if stand_in is not None and Path(stand_in).exists() and self.record_path is not None:
            import shutil
            try:
                shutil.move(str(stand_in), str(self.record_path))
            except OSError as exc:
                # The recording stays where it was written, not lost: keep that path and say so.
                self.move_error = "{}: {}".format(type(exc).__name__, exc)
                _note_stranded(stand_in, self.record_path, self.move_error)
                return
        self._stream_path = None

    @property
    def stranded_path(self):
        """Where the recording still is when moving it to `record_path` failed, else None."""
        return self._stream_path if getattr(self, "move_error", None) else None
