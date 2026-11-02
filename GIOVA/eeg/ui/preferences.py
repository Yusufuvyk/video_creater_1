"""Where the user's settings live: one store, read and written by every page.

`config.py` holds no Qt, so the QSettings factory lives here instead. Keeping
it in one place matters because the Settings dialog, the Monitor and the
Analysis page all read the same keys: two stores would drift apart.
"""
import os
from pathlib import Path

from pyqtgraph.Qt import QtCore

from ..core.config import DATA_DIR

#: Everything the Settings dialog owns: key -> (default, type).
#: Per-session choices (board, filters, channel selection, window length) are
#: not here; they belong to the panel and are remembered separately.
DEFAULTS = {
    'analysis/tf_max_channels': (4, int),
    # Files are reduced to about this rate when opened; 0 keeps the file's own.
    'files/downsample_hz': (0.0, float),
    # What PSD graphs draw: power (µV²/Hz), amplitude_density (µV/√Hz) or amplitude (µV).
    'graphs/psd_unit': ('power', str),
    'recording/folder': ('', str),
    'recording/prefix': ('live', str),
    'monitor/views/layout': ('auto', str),
    'monitor/display_fps': (30, int),
    'monitor/trace_width': (1, int),
    # Traces scroll (False), or stand still while a cursor writes over them (True).
    'monitor/trace_sweep': (False, bool),
    # Monitor spectra: 0 s = the display window; Welch segment and overlap.
    'monitor/spectrum_seconds': (0.0, float),
    'monitor/welch_seconds': (2.0, float),
    'monitor/welch_overlap': (50.0, float),
    # Hover explanations on buttons and fields (the Tips button, top right).
    'ui/tooltips': (True, bool),
    # The window's colours, "dark" or "light", from the next start. Reports and saved figures are always light.
    'ui/theme': ('dark', str),
    # The Games tab: festival games on the Monitor's stream. Off hides the tab only.
    'ui/show_games': (True, bool),
    # A plain drag on the Workbench's empty sheet: "pan" moves the sheet (Shift + drag selects), "select" draws a
    # selection box (Space or the middle button + drag moves the sheet).
    'workbench/drag': ('pan', str),
}

_store = None


def user_settings():
    """The one QSettings the whole program uses.

    Offscreen validation must not overwrite the operator's desktop settings,
    so a test run gets its own ini file. Version 3 keeps its own settings, so
    trying it leaves version 2 exactly as it was; GIOVA (v3.1) likewise leaves version 3.
    """
    global _store
    offscreen = os.environ.get('QT_QPA_PLATFORM') == 'offscreen'
    wanted = 'tests' if offscreen else 'user'
    if _store is None or getattr(_store, 'origin', None) != wanted:
        _store = (QtCore.QSettings(str(Path(__file__).resolve().parents[2] / 'tests' / '.tmp' / 'monitor.ini'),
                                   QtCore.QSettings.Format.IniFormat)
                  if offscreen else QtCore.QSettings('GIOVA', 'GIOVA'))
        _store.origin = wanted
    return _store


def setting(key, default=None, kind=None):
    """One stored value, with the default and type from DEFAULTS when known."""
    fallback, expected = DEFAULTS.get(key, (default, kind))
    if default is not None:
        fallback = default
    if expected is None:
        return user_settings().value(key, fallback)
    try:
        return user_settings().value(key, fallback, type=expected)
    except (TypeError, ValueError):     # a hand-edited ini must not stop the program
        return fallback


def set_setting(key, value):
    user_settings().setValue(key, value)


def data_dir():
    """Where recordings are written and looked for.

    The operator's folder when they chose one and it is still usable, else the
    one the program found on its own (next to the exe when packaged).
    """
    chosen = str(setting('recording/folder', '') or '')
    if chosen:
        path = Path(chosen)
        if path.is_dir():
            return path
    return DATA_DIR


def psd_unit():
    """The unit PSD graphs are drawn in; a hand-edited value falls back to the density."""
    from ..core.spectral import POWER_DENSITY, PSD_UNITS
    unit = str(setting('graphs/psd_unit') or POWER_DENSITY)
    return unit if unit in PSD_UNITS else POWER_DENSITY


def downsample_hz():
    """The rate files are reduced to when opened, or 0 to keep their own."""
    try:
        return max(0.0, float(setting('files/downsample_hz')))
    except (TypeError, ValueError):
        return 0.0


def recording_prefix():
    """What a new live recording's name starts with, before the timestamp."""
    return str(setting('recording/prefix', 'live') or 'live')
