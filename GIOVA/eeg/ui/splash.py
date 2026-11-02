"""GIOVA's start-up screen, and the window icon.

Qt only - no numpy, scipy or BrainFlow - so it can be on screen before the
heavy imports that make up most of the start-up time.

The mark is GIOVA's logo (a blue-to-teal ribbon wave on a
rounded tile, corners transparent), kept in assets/: the dark tile for the taskbar, the
exe and the dark theme, the light tile for the light theme. The .ico carries the small
sizes Windows shows (16 to 48 px) as their own images. The start-up screen is the
same layout as before: the mark centred above the name, the tagline under it, a quiet
line saying what is being prepared, a thin bar that fills as the program loads, the
version and what the program is for.

The bar follows the modules imported so far against how many the last start needed
(no number is shown): a finder at the front of sys.meta_path, which finds nothing
itself, redraws the screen at most every FRAME seconds while the imports go on.
"""
import sys
import time
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

#: The logo's two ends: the ribbon runs from this blue to this teal.
BLUE = QtGui.QColor("#2563EB")
TEAL = QtGui.QColor("#14C8AE")
#: The dark tile's navy.
NAVY = QtGui.QColor("#0E1A2E")

ASSETS = Path(__file__).resolve().parent / "assets"
ICO = ASSETS / "giova.ico"
#: The light tile as an .ico, for the light theme's taskbar and title bar (the exe file keeps the dark one).
ICO_LIGHT = ASSETS / "giova_light.ico"
#: The sizes the .ico holds, each its own image.
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)

_cache = {}


def _logo(light):
    key = "light" if light else "dark"
    if key not in _cache:
        _cache[key] = QtGui.QPixmap(str(ASSETS / "giova_{}.png".format(key)))
    return _cache[key]


def icon_pixmap(size, light=False):
    """The logo at `size` pixels: the dark tile, or the light one for the light theme. Small sizes come from
    the .ico's own images, drawn for them, not shrunk from the large one."""
    ico = ICO_LIGHT if light else ICO
    if size <= 48 and ico.is_file():
        pixmap = QtGui.QIcon(str(ico)).pixmap(size, size)
        if not pixmap.isNull() and pixmap.width() == size:
            return pixmap
    logo = _logo(light)
    if logo.isNull():
        raise FileNotFoundError(ASSETS / "giova_{}.png".format("light" if light else "dark"))
    return logo.scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                       QtCore.Qt.TransformationMode.SmoothTransformation)


def app_icon(light=False):
    """The window and taskbar icon: the .ico, every size its own image; the light tile for the light theme."""
    icon = QtGui.QIcon(str(ICO_LIGHT if light else ICO))
    if icon.isNull():
        for size in ICON_SIZES:
            icon.addPixmap(icon_pixmap(size, light=light))
    return icon


def write_ico(path):
    """The exe's icon: the approved .ico, as it is."""
    try:
        Path(path).write_bytes(ICO.read_bytes())
    except OSError:
        return False
    return True


#: Where the number of modules a full start needed is kept (the exe's own, as it loads a different set),
#: and the guess before the first start.
MODULES_KEY = "ui/startup_modules" + ("_exe" if getattr(sys, "frozen", False) else "")
MODULES_GUESS = 1180
#: The bar is redrawn at most this often (seconds), and moves this share of the way to the true value each time.
FRAME = 0.03
EASE = 0.35


class _Pacer:
    """A meta-path finder that finds nothing: every import asks it first, which is when the bar moves on."""

    def __init__(self, splash):
        self.splash = splash

    def find_spec(self, name, path=None, target=None):
        self.splash.tick()
        return None


class GiovaSplash(QtWidgets.QSplashScreen):
    """The start-up screen: 600 x 338, drawn at the screen's pixel ratio; `set_status` for real milestones only."""

    WIDTH, HEIGHT = 600, 338
    #: the bar's track: x, y, width, height
    BAR = (234, 261, 132, 3)
    FOOTER = "Biosignal recording and analysis"
    #: background, text, quiet text, border, the accent's track: GIOVA's dark and light themes (theme.py)
    DARK = ("#1C222B", "#D3DAE3", "#8B97A8", "#323B48", "#2B3440")
    LIGHT = ("#F1F1F1", "#2A2A2A", "#6E6E6E", "#D0D0D0", "#DCDCDC")
    ACCENT = (BLUE.name(), TEAL.name())

    def __init__(self, icon=None, dark=True, version=None, pixel_ratio=None):
        super().__init__(QtGui.QPixmap(), QtCore.Qt.WindowType.FramelessWindowHint)
        from .. import NAME, TAGLINE, VERSION
        self.NAME, self.SUBTITLE = NAME, TAGLINE
        self.setWindowTitle(NAME)
        self.setAccessibleName("{} startup screen".format(NAME))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.dark = bool(dark)
        self.version = str(version or VERSION)
        self.status = "Preparing your workspace…"
        screen = QtGui.QGuiApplication.primaryScreen()
        self._ratio = float(pixel_ratio if pixel_ratio is not None else screen.devicePixelRatio() if screen else 1.0)
        if self._ratio <= 0:
            raise ValueError("pixel_ratio must be positive")
        self._icon = QtGui.QPixmap(icon) if icon is not None else icon_pixmap(192, light=not self.dark)
        self.progress = 0.0                      # the share of the bar filled, 0 to 1
        self._first = len(sys.modules)
        self._expected = max(self._first + 1, _stored_count())
        self._last = 0.0
        self._pacer = None
        self.setPixmap(self._render())

    @staticmethod
    def _font(size, weight=QtGui.QFont.Weight.Normal, spacing=0.0):
        font = QtGui.QFont()
        font.setFamilies(["Inter", "Segoe UI"])       # Inter when the program carries it (fonts.py)
        font.setPixelSize(size)
        font.setWeight(weight)
        if spacing:
            font.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, spacing)
        return font

    def _render(self):
        ratio = self._ratio
        image = QtGui.QPixmap(round(self.WIDTH * ratio), round(self.HEIGHT * ratio))
        image.setDevicePixelRatio(ratio)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                               QtGui.QPainter.RenderHint.TextAntialiasing |
                               QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        background, foreground, muted, border, track = self.DARK if self.dark else self.LIGHT
        painter.setPen(QtGui.QPen(QtGui.QColor(border), 1.0))
        painter.setBrush(QtGui.QColor(background))
        painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, self.WIDTH - 1, self.HEIGHT - 1), 18, 18)
        painter.drawPixmap(QtCore.QRectF(266, 40, 68, 68), self._icon, QtCore.QRectF(self._icon.rect()))
        centred = QtCore.Qt.AlignmentFlag.AlignCenter
        painter.setPen(QtGui.QColor(foreground))
        painter.setFont(self._font(40, QtGui.QFont.Weight.DemiBold, 2.0))
        painter.drawText(QtCore.QRectF(32, 119, 536, 49), centred, self.NAME)
        painter.setPen(QtGui.QColor(muted))
        painter.setFont(self._font(14, spacing=1.0))
        painter.drawText(QtCore.QRectF(32, 173, 536, 24), centred, self.SUBTITLE)
        painter.setFont(self._font(12))
        status = painter.fontMetrics().elidedText(self.status, QtCore.Qt.TextElideMode.ElideRight, 536)
        painter.drawText(QtCore.QRectF(32, 230, 536, 21), centred, status)
        # the bar fills with the loading; the ribbon's blue to teal runs along the whole track
        x, y, width, height = self.BAR
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(track))
        painter.drawRoundedRect(QtCore.QRectF(x, y, width, height), 1.5, 1.5)
        gradient = QtGui.QLinearGradient(x, 0, x + width, 0)
        gradient.setColorAt(0, QtGui.QColor(self.ACCENT[0]))
        gradient.setColorAt(1, QtGui.QColor(self.ACCENT[1]))
        painter.setBrush(gradient)
        painter.drawRoundedRect(QtCore.QRectF(x, y, max(height, width * self.progress), height), 1.5, 1.5)
        painter.setPen(QtGui.QColor(muted))
        painter.setFont(self._font(11))
        painter.drawText(QtCore.QRectF(32, 298, 100, 18),
                         QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                         "Version " + self.version)
        painter.drawText(QtCore.QRectF(182, 298, 386, 18),
                         QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, self.FOOTER)
        painter.end()
        return image

    def set_status(self, message):
        """Say what is being prepared now; for real milestones, it adds no delay."""
        self.status = str(message)
        self.setPixmap(self._render())
        self.repaint()

    def follow_imports(self):
        """Move the bar on while modules load, until `finish`."""
        if self._pacer is None:
            self._pacer = _Pacer(self)
            sys.meta_path.insert(0, self._pacer)

    def tick(self):
        """Bring the bar nearer to the share of modules loaded; at most one redraw per FRAME, never past 97 %."""
        now = time.monotonic()
        if now - self._last < FRAME:
            return
        self._last = now
        loaded = (len(sys.modules) - self._first) / (self._expected - self._first)
        target = min(0.97, max(0.0, loaded))
        if target <= self.progress:
            return
        self.progress += max(0.004, (target - self.progress) * EASE)
        self.progress = min(self.progress, target)
        self.setPixmap(self._render())
        self.repaint()

    def finish(self, window):
        """The bar full, the count kept for the next start, then the screen goes."""
        if self._pacer is not None:
            if self._pacer in sys.meta_path:
                sys.meta_path.remove(self._pacer)
            self._pacer = None
            _store_count(len(sys.modules))
        self.progress = 1.0
        self.setPixmap(self._render())
        self.repaint()
        super().finish(window)


def _stored_count():
    """How many modules the last full start had loaded; a guess before the first start."""
    try:
        from .preferences import setting
        return int(setting(MODULES_KEY, MODULES_GUESS) or MODULES_GUESS)
    except Exception:
        return MODULES_GUESS


def _store_count(count):
    try:
        from .preferences import set_setting
        set_setting(MODULES_KEY, int(count))
    except Exception:
        pass


def _dark_theme():
    """The theme chosen in Settings → Interface, read before the window exists; dark when unknown."""
    try:
        from .preferences import setting
        return str(setting("ui/theme", "dark") or "dark") != "light"
    except Exception:
        return True


def show_splash(app):
    """Put the start-up screen up at once and return it (call .finish(window))."""
    from .fonts import load_fonts
    load_fonts(app)
    splash = GiovaSplash(dark=_dark_theme())
    splash.show()
    app.processEvents()
    splash.follow_imports()
    return splash
