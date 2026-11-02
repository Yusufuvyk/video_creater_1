"""The program's colours: a dark window by default, light on request, and always white on paper.

Every note and read-out used to carry a hard-coded light-theme colour: `color: #555` on a
label, `background: #f2f2f2` on the mapping preview. Under a dark window those become dark
grey text on dark grey. Ask here instead: `color("subtle")`, `muted()`, `plot_background()`.

The window follows Settings → Interface → Theme (dark unless chosen otherwise), read once
at start. What goes into a file - a report, a saved figure - is built inside `paper()`:
PAPER's colours and white plots whatever the window shows, because a page is read on white.

The values are GIOVA's: one Sky Blue accent, colour
only where it means something, red only for an error. A colour the window needs that is not here
goes into that page first.
"""
from contextlib import contextmanager

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import axes  # noqa: F401  (tick labels never overlap, on every plot)

#: GIOVA's accent: selection, the current page, the run in progress. On a filled button it is a
#: darker blue (`primary`) so white text on it stays readable.
SKY = "#3AA9D8"
SKY_FILL = "#0A7FAE"
#: Two conditions compared (A, B: odor and air, left and right hand): mid tones that read on the dark
#: window and on white paper alike, away from the blue accent and from the logo's teal.
CONDITIONS = ("#2FA67A", "#8A76E6")
#: The report's dark blue: headings and marks on paper.
NAVY = "#1b2140"

#: The white page reports and saved figures are drawn on: the light look as it was before GIOVA.
PAPER = dict(
    canvas="#f6f7f9", card="#ffffff", text="#1f2328", subtle="#57606a", muted="#616e7c", line="#dde1e6",
    field="#cfd6de", hover="#f2f4f7", hover_border="#b6bfca", pressed="#e6eaf0", checked="#e9ecf5",
    disabled_text="#a3aab3", disabled_border="#e3e7ec", disabled_field="#f3f5f7", header="#f4f6f8",
    header_text="#3c4650", grid="#eceff3", scroll="#cdd3db", scroll_hover="#aeb7c2", accent=NAVY,
    primary=NAVY, primary_hover="#2b3363", primary_disabled="#8e93a8", primary_text="#ffffff",
    selection="#cfe3ff", selection_text="#102a43", plot="#ffffff", plot_text="#1f2328", ink=NAVY,
    strong="#243b53", panel="#edf2f6", panel_hover="#dde5ec", panel_text="#263b50", good="#1b6e3c",
    bad="#c0392b", stale="#9a7a10", warning="#a04000", link="#1e3a8a", dim="#9aa5b1", tooltip=NAVY,
    sheet="#fbfcfd", sheet_line="#edf0f4", sheet_major="#dfe4ea", block="#ffffff", block_border="#c9d1da",
    input="#ffffff")
#: Dimmed dark with a faint blue (30 October 2026): the ground lifted off black and the text kept below
#: white, so long hours do not glare (#D3DAE3 on #1C222B, not #F1F5F9). Panels sit one step above.
DARK = dict(
    canvas="#1C222B", card="#232A34", text="#D3DAE3", subtle="#8B97A8", muted="#8B97A8", line="#323B48",
    field="#3A4452", hover="#2B3440", hover_border="#4A5566", pressed="#303A47", checked="#253949",
    disabled_text="#5A6472", disabled_border="#2A313B", disabled_field="#20262F", header="#191E26",
    header_text="#AEB8C5", grid="#262D37", scroll="#3A4452", scroll_hover="#4A5566", accent=SKY,
    primary=SKY_FILL, primary_hover="#1690C2", primary_disabled="#24394A", primary_text="#FFFFFF",
    selection="#1F4A63", selection_text="#E8EEF4", plot="#1F252E", plot_text="#AEB8C5", ink="#8FA3BC",
    strong="#E3E8EE", panel="#232A34", panel_hover="#2B3440", panel_text="#AEB8C5", good="#6FBF8B",
    bad="#E07B6E", stale="#D6B35A", warning="#D69A55", link="#6FC0E6", dim="#6B7685", tooltip="#3A4452",
    sheet="#1C222B", sheet_line="#323A46", sheet_major="#434D5B", block="#2A323D", block_border="#3A4452",
    input="#1A2028")
#: Light theme B (30 October 2026): untinted greys, softer than white - blue-grey read "too cold", warm
#: ivory suited an orange accent, not Sky Blue. White stays inside blocks and plots only.
LIGHT = dict(
    canvas="#E6E6E6", card="#F1F1F1", text="#2A2A2A", subtle="#6E6E6E", muted="#6E6E6E", line="#D0D0D0",
    field="#C4C4C4", hover="#DCDCDC", hover_border="#B5B5B5", pressed="#D4D4D4", checked="#D3E4EE",
    disabled_text="#A6A6A6", disabled_border="#DADADA", disabled_field="#EBEBEB", header="#DEDEDE",
    header_text="#444444", grid="#E2E2E2", scroll="#C4C4C4", scroll_hover="#A8A8A8", accent=SKY_FILL,
    primary=SKY_FILL, primary_hover="#086E97", primary_disabled="#9FC3D3", primary_text="#FFFFFF",
    selection="#BFDCEB", selection_text="#10303F", plot="#F8F8F8", plot_text="#2A2A2A", ink="#4A5D75",
    strong="#1A1A1A", panel="#E6E6E6", panel_hover="#DCDCDC", panel_text="#444444", good="#1B6E3C",
    bad="#C0392B", stale="#9A7A10", warning="#A04000", link="#0A6E96", dim="#9A9A9A", tooltip="#3A4452",
    sheet="#EDEDED", sheet_line="#CDCDCD", sheet_major="#B4B4B4", block="#F8F8F8", block_border="#D0D0D0",
    input="#FAFAFA")

THEMES = ("dark", "light")
_dark = None        # read from the settings on first use
_paper = 0          # > 0 inside paper()


def set_theme(name):
    """The window's theme from now on ("dark" or "light"); widgets built earlier keep theirs."""
    global _dark
    _dark = name != "light"


def window_is_dark():
    """The window's theme, whatever is being drawn now."""
    if _dark is None:
        from .preferences import setting
        set_theme(setting("ui/theme"))
    return _dark


def is_dark():
    """True when what is being built now is drawn dark: the window, unless on paper."""
    return not _paper and window_is_dark()


@contextmanager
def paper():
    """Widgets built and drawn inside are light, for a file: a report, a saved figure."""
    import pyqtgraph as pg
    global _paper
    before = {key: pg.getConfigOption(key) for key in ("background", "foreground")}
    _paper += 1
    # plots made inside take their axes' colours from here
    pg.setConfigOptions(background=PAPER["plot"], foreground=PAPER["plot_text"])
    try:
        yield
    finally:
        _paper -= 1
        pg.setConfigOptions(**before)


def colours():
    """Every role's colour for what is being built now: the window's theme, or the page inside `paper()`."""
    return PAPER if _paper else DARK if window_is_dark() else LIGHT


def color(role):
    """A colour by role (LIGHT's keys) for what is being built now."""
    return colours()[role]


def soften(hex_colour):
    """A category or status colour as text on the window: as it is on light, half grey on dark."""
    if not is_dark():
        return hex_colour
    a, b = QtGui.QColor(hex_colour), QtGui.QColor(DARK["subtle"])
    return QtGui.QColor((a.red() + b.red()) // 2, (a.green() + b.green()) // 2, (a.blue() + b.blue()) // 2).name()


def plot_background():
    return color("plot")


def plot_foreground():
    return color("plot_text")


def _pick(pair):
    return pair[1] if is_dark() else pair[0]


#: (light, dark) for each role.
MUTED = (LIGHT["muted"], DARK["muted"])
WARNING = (LIGHT["warning"], DARK["warning"])
ERROR = (LIGHT["bad"], DARK["bad"])

_INFO_BOX = (("#f2f2f2", "#dddddd", "#333333"), ("#2f3133", "#4a4d50", "#dcdcdc"))
_ERROR_BOX = (("#fdeeee", "#e8c4c0", "#8e2b20"), ("#402725", "#6b3a35", "#f0a79c"))


def muted():
    """Secondary text: present, but not competing with the controls."""
    return "color: {};".format(color("muted"))


def warning():
    return "color: {};".format(color("warning"))


def error():
    return "color: {};".format(color("bad"))


def note_box(bad=False):
    """A boxed read-out, e.g. the .mat mapping preview.

    `bad` turns it red for a value the user needs to notice - a unit scale that
    is out by a factor of 32, say.
    """
    background, border, text = _pick(_ERROR_BOX if bad else _INFO_BOX)
    return ("padding: 6px; background: {}; border: 1px solid {}; "
            "color: {};".format(background, border, text))


# ----------------------------------------------------------------------
# application look
# ----------------------------------------------------------------------


def palette():
    """A Fusion palette in the colours of what is being built now."""
    c = colours()
    out = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    for role, key in ((roles.Window, "canvas"), (roles.WindowText, "text"), (roles.Base, "card"),
                      (roles.AlternateBase, "header"), (roles.Text, "text"), (roles.Button, "card"),
                      (roles.ButtonText, "text"), (roles.ToolTipBase, "tooltip"), (roles.Highlight, "selection"),
                      (roles.HighlightedText, "selection_text"), (roles.PlaceholderText, "subtle"),
                      (roles.Link, "link"), (roles.Mid, "line"), (roles.Midlight, "hover"), (roles.Dark, "field"),
                      (roles.Light, "card"), (roles.Shadow, "canvas")):
        out.setColor(role, QtGui.QColor(c[key]))
    out.setColor(roles.ToolTipText, QtGui.QColor("#ffffff"))
    out.setColor(roles.BrightText, QtGui.QColor("#ffffff"))
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        out.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(c["disabled_text"]))
    return out


def use_theme(app):
    """The window's theme on the whole application: Fusion, its palette, the stylesheet and the
    plots' defaults. Windows' own dark or light mode does not matter; the setting decides."""
    import pyqtgraph as pg
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):  # Qt 6.8+
        hints.setColorScheme(QtCore.Qt.ColorScheme.Dark if is_dark() else QtCore.Qt.ColorScheme.Light)
    app.setStyle("Fusion")
    app.setPalette(palette())
    app.setStyleSheet(app_stylesheet())
    pg.setConfigOptions(antialias=True, background=plot_background(), foreground=plot_foreground())


def for_paper(widget):
    """A widget built inside `paper()`, drawn as on paper though the window around it is dark."""
    widget.setPalette(palette())
    widget.setStyleSheet(app_stylesheet())
    return widget


def chevron(colour):
    """A file holding Lucide's chevron in `colour`, for a list's arrow (a stylesheet takes a picture by path)."""
    import tempfile
    from pathlib import Path
    from .icons import SHAPES, SVG
    folder = Path(tempfile.gettempdir()) / "giova"
    path = folder / "chevron-{}.svg".format(colour.lstrip("#").lower())
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(SVG.format(colour=colour, shape=SHAPES["chevron-down"]), encoding="utf-8")
    return path.as_posix()


def app_stylesheet():
    """One restrained look for every page: cards on a canvas, rounded controls, one accent colour.
    Visual only - no widget is resized by more than its padding, and no behaviour depends on it.
    Fields (text, numbers, lists) are GIOVA's: filled a step off the
    panel, no frame until the mouse is on them, the accent when typed in. Number fields have no arrow
    buttons - typing and the arrow keys change them - and a list opens from Lucide's chevron.

    Primary buttons opt in with `setProperty("primary", True)`.
    """
    c = colours()
    return f"""
QMainWindow, QDialog {{ background: {c['canvas']}; }}
QToolTip {{ background: {c['tooltip']}; color: #E8EEF4; border: none; padding: 6px 8px; }}

QTabWidget::pane {{ border: 1px solid {c['line']}; border-radius: 8px; background: {c['card']}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {c['subtle']}; padding: 7px 16px; margin-right: 2px;
               border: none; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {c['strong']}; border-bottom: 2px solid {c['accent']}; }}
QTabBar::tab:hover:!selected {{ color: {c['strong']}; }}

QPushButton {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['field']}; border-radius: 6px;
              padding: 4px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {c['hover']}; border-color: {c['hover_border']}; }}
QPushButton:pressed {{ background: {c['pressed']}; }}
QPushButton:checked {{ background: {c['checked']}; border-color: {c['accent']}; color: {c['accent']}; }}
QPushButton:disabled {{ background: {c['canvas']}; color: {c['disabled_text']}; border-color: {c['disabled_border']}; }}
QPushButton[primary="true"] {{ background: {c['primary']}; color: {c['primary_text']}; border-color: {c['primary']};
    font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {c['primary_hover']}; border-color: {c['primary_hover']}; }}
QPushButton[primary="true"]:disabled {{ background: {c['primary_disabled']}; border-color: {c['primary_disabled']};
    color: {c['disabled_text']}; }}
QToolButton {{ border: none; border-radius: 6px; padding: 3px 8px; color: {c['text']}; }}
QToolButton:hover {{ background: {c['hover']}; }}
QToolButton:checked {{ background: {c['checked']}; color: {c['accent']}; }}

QGroupBox {{ background: {c['card']}; border: 1px solid {c['line']}; border-radius: 8px;
            margin-top: 14px; padding: 4px 0px 0px 0px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 10px;
                   padding: 0 4px; color: {c['strong']}; font-weight: 600; }}
QGroupBox:disabled {{ color: {c['disabled_text']}; }}

QLineEdit, QAbstractSpinBox, QComboBox {{ background: {c['input']}; color: {c['text']};
    border: 1px solid {c['line']}; border-radius: 6px; padding: 3px 8px; min-height: 18px;
    selection-background-color: {c['selection']}; selection-color: {c['selection_text']}; }}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {{ border-color: {c['hover_border']}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QComboBox:on {{ border-color: {c['accent']}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ background: {c['disabled_field']};
    color: {c['disabled_text']}; border-color: {c['disabled_border']}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0; border: none; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({chevron(c['subtle'])}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['line']};
    border-radius: 6px; padding: 4px; selection-background-color: {c['checked']};
    selection-color: {c['accent']}; outline: 0; }}

QHeaderView::section {{ background: {c['header']}; color: {c['header_text']}; border: none;
    border-bottom: 1px solid {c['line']}; border-right: 1px solid {c['grid']}; padding: 3px 4px; font-weight: 600; }}
QTableView, QTableWidget, QListView, QListWidget, QTextBrowser, QTreeView, QTreeWidget {{
    background: {c['card']}; color: {c['text']}; border: 1px solid {c['line']}; border-radius: 6px;
    gridline-color: {c['grid']}; }}
QListView::item:selected, QListWidget::item:selected, QTreeView::item:selected {{ background: {c['checked']};
    color: {c['accent']}; }}
QTableCornerButton::section {{ background: {c['header']}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {c['scroll']}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {c['scroll_hover']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMenu {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['line']}; padding: 4px; }}
QMenu::item {{ padding: 5px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['checked']}; color: {c['accent']}; }}
QMenu::item:disabled {{ color: {c['disabled_text']}; }}
QMenu::separator {{ height: 1px; background: {c['line']}; margin: 4px 6px; }}
QStatusBar {{ background: {c['canvas']}; color: {c['subtle']}; }}
QSplitter::handle {{ background: transparent; }}
"""


def brand_label():
    """The small mark and the program's name, for a corner that names the program."""
    from .splash import icon_pixmap
    label = QtWidgets.QLabel()
    ratio = 2.0
    pixmap = icon_pixmap(int(20 * ratio), light=not window_is_dark())
    pixmap.setDevicePixelRatio(ratio)
    label.setPixmap(pixmap)
    from .. import NAME, TAGLINE
    text = QtWidgets.QLabel("{} <span style='color:{}; font-weight:400'>· {}</span>".format(
        NAME, color("subtle"), TAGLINE))
    text.setStyleSheet("color: {}; font-weight: 700; letter-spacing: 0.5px;".format(color("strong")))
    box = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(box)
    row.setContentsMargins(8, 0, 14, 0)
    row.setSpacing(6)
    row.addWidget(label)
    row.addWidget(text)
    return box
