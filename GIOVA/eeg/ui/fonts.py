"""GIOVA's typefaces: Inter for everything, JetBrains Mono for code and the Log.

Both are open (SIL Open Font License) and carried in assets/fonts/, so the program looks the same on every
computer instead of borrowing whatever the system has (Segoe UI exists only on Windows). When the files are
not there, the system's own font is used and nothing else changes.
"""
from pathlib import Path

from pyqtgraph.Qt import QtGui

FONTS = Path(__file__).resolve().parent / "assets" / "fonts"
UI, MONO = "Inter", "JetBrains Mono"
_loaded = None


def load_fonts(app=None):
    """Register the carried fonts once; with `app`, make Inter its font. The families found, as a set."""
    global _loaded
    if _loaded is None:
        _loaded = set()
        for path in sorted(FONTS.glob("*.[ot]tf")) if FONTS.is_dir() else ():
            index = QtGui.QFontDatabase.addApplicationFont(str(path))
            if index >= 0:
                _loaded.update(QtGui.QFontDatabase.applicationFontFamilies(index))
    if app is not None and UI in _loaded:
        font = QtGui.QFont(UI)
        font.setPointSizeF(app.font().pointSizeF())
        app.setFont(font)
    return set(_loaded)


def mono():
    """Code and Log text: JetBrains Mono when carried, else the system's fixed font."""
    if MONO in load_fonts():
        return QtGui.QFont(MONO)
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
