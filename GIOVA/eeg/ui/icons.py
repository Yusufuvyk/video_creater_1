"""The program's action icons: one outline family, tinted to the theme, sharp at any scaling.

Buttons used font characters as icons (a house, a gear, a triangle, a circular arrow), whose shape and
weight depend on the fonts installed. These are the matching Lucide outlines instead, 24 px viewBox,
stroke 2, round caps: neutral text colour, the primary button's text colour on a primary button, a
quieter grey when disabled. Labels stay; an icon goes beside them, or alone only on a familiar compact
button (refresh, previous / next) that keeps its tooltip and accessible name.

Lucide 0.468.0 (https://lucide.dev), the subset reviewed in BioTrace_UI_Icons_Proposal; its licence:

    ISC License

    Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
from functools import lru_cache

from pyqtgraph.Qt import QtCore, QtGui, QtSvg

from .theme import color

#: Toolbar icons are 18 logical pixels; 16 in dense rows.
SIZE = 18

#: Each icon's drawing inside <svg viewBox="0 0 24 24" ...>, as Lucide publishes it.
SHAPES = {
    'chevron-down': (
        '<path d="m6 9 6 6 6-6" />'
    ),
    'chevron-left': (
        '<path d="m15 18-6-6 6-6" />'
    ),
    'chevron-right': (
        '<path d="m9 18 6-6-6-6" />'
    ),
    'circle-check': (
        '<circle cx="12" cy="12" r="10" /> <path d="m9 12 2 2 4-4" />'
    ),
    'circle-help': (
        '<circle cx="12" cy="12" r="10" /> <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /> <path d="M12'
        ' 17h.01" />'
    ),
    'circle-x': (
        '<circle cx="12" cy="12" r="10" /> <path d="m15 9-6 6" /> <path d="m9 9 6 6" />'
    ),
    'code-xml': (
        '<path d="m18 16 4-4-4-4" /> <path d="m6 8-4 4 4 4" /> <path d="m14.5 4-5 16" />'
    ),
    'columns-2': (
        '<rect width="18" height="18" x="3" y="3" rx="2" /> <path d="M12 3v18" />'
    ),
    'copy': (
        '<rect width="14" height="14" x="8" y="8" rx="2" ry="2" /> <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1'
        '.9-2 2-2h10c1.1 0 2 .9 2 2" />'
    ),
    'download': (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /> <polyline points="7 10 12 15 17 10" /> <l'
        'ine x1="12" x2="12" y1="15" y2="3" />'
    ),
    'external-link': (
        '<path d="M15 3h6v6" /> <path d="M10 14 21 3" /> <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V'
        '8a2 2 0 0 1 2-2h6" />'
    ),
    'file-text': (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" /> <path d="M14 2v4a2 2 0 0'
        ' 0 2 2h4" /> <path d="M10 9H8" /> <path d="M16 13H8" /> <path d="M16 17H8" />'
    ),
    'folder': (
        '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 '
        '0 0 0-2 2v13a2 2 0 0 0 2 2Z" />'
    ),
    'folder-open': (
        '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 '
        '0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2" />'
    ),
    'house': (
        '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8" /> <path d="M3 10a2 2 0 0 1 .709-1.528l7-5.'
        '999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />'
    ),
    'image-down': (
        '<path d="M10.3 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v10l-3.1-3.1a2 2 0 0 0-2.814.0'
        '14L6 21" /> <path d="m14 19 3 3v-5.5" /> <path d="m17 22 3-3" /> <circle cx="9" cy="9" r="2" />'
    ),
    'maximize-2': (
        '<polyline points="15 3 21 3 21 9" /> <polyline points="9 21 3 21 3 15" /> <line x1="21" x2="14" '
        'y1="3" y2="10" /> <line x1="3" x2="10" y1="21" y2="14" />'
    ),
    'message-circle': (
        '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" />'
    ),
    'play': (
        '<polygon points="6 3 20 12 6 21 6 3" />'
    ),
    'plus': (
        '<path d="M5 12h14" /> <path d="M12 5v14" />'
    ),
    'refresh-cw': (
        '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /> <path d="M21 3v5h-5" /> <path d='
        '"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /> <path d="M8 16H3v5" />'
    ),
    'rotate-ccw': (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /> <path d="M3 3v5h5" />'
    ),
    'save': (
        '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2'
        ' 0 0 1 2-2z" /> <path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7" /> <path d="M7 3v4a1 1 0 0 0'
        ' 1 1h7" />'
    ),
    'settings': (
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0'
        ' 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 '
        '0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0'
        ' 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22'
        '-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.7'
        '3l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" '
        '/> <circle cx="12" cy="12" r="3" />'
    ),
    'sliders-horizontal': (
        '<line x1="21" x2="14" y1="4" y2="4" /> <line x1="10" x2="3" y1="4" y2="4" /> <line x1="21" x2="1'
        '2" y1="12" y2="12" /> <line x1="8" x2="3" y1="12" y2="12" /> <line x1="21" x2="16" y1="20" y2="2'
        '0" /> <line x1="12" x2="3" y1="20" y2="20" /> <line x1="14" x2="14" y1="2" y2="6" /> <line x1="8'
        '" x2="8" y1="10" y2="14" /> <line x1="16" x2="16" y1="18" y2="22" />'
    ),
    'square': (
        '<rect width="18" height="18" x="3" y="3" rx="2" />'
    ),
    'triangle-alert': (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" /> <path d="M'
        '12 9v4" /> <path d="M12 17h.01" />'
    ),
    'x': (
        '<path d="M18 6 6 18" /> <path d="m6 6 12 12" />'
    ),
    'chevron-up': (
        '<path d="m18 15-6-6-6 6" />'
    ),
    'minus': (
        '<path d="M5 12h14" />'
    ),
    'scan': (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2" /> <path d="M17 3h2a2 2 0 0 1 2 2v2" /> <path d="M21 17v2a2 2 0 0 1-2 '
        '2h-2" /> <path d="M7 21H5a2 2 0 0 1-2-2v-2" />'
    ),
    'fast-forward': (
        '<polygon points="13 19 22 12 13 5 13 19" /> <polygon points="2 19 11 12 2 5 2 19" />'
    ),
    # GIOVA's own, drawn to the same rules (24 grid, stroke 2, round ends) for what Lucide has no icon for:
    # the block categories
    'recording': (
        '<path d="M3 12h3l2-5 3 10 3-8 2 3h5" />'
    ),
    'filter-wave': (
        '<path d="M3 7h18" /> <path d="M3 12c4 0 5-5 9-5s5 10 9 10" opacity=".55" /> <path d="M6 17h12" />'
    ),
    'trials': (
        '<rect x="3" y="5" width="5" height="14" rx="1" /> <rect x="10" y="5" width="5" height="14" rx="1" /> '
        '<rect x="17" y="5" width="4" height="14" rx="1" />'
    ),
    'spectrum': (
        '<path d="M4 20V10M8 20V6M12 20v-8M16 20v-5M20 20v-3" />'
    ),
    'features': (
        '<path d="M4 6h6M4 12h6M4 18h6" /> <path d="M14 6l4 6-4 6" />'
    ),
    'statistics': (
        '<path d="M4 20h16" /> <rect x="6" y="10" width="4" height="10" rx="1" /> <rect x="14" y="6" width="4" '
        'height="14" rx="1" /> <path d="M8 5v5M16 2v4" />'
    ),
    'classifier': (
        '<circle cx="7" cy="8" r="1.5" /> <circle cx="9" cy="15" r="1.5" /> <circle cx="16" cy="7" r="1.5" /> '
        '<circle cx="17" cy="15" r="1.5" /> <path d="M4 20 20 4" />'
    ),
    'augmentation': (
        '<rect x="4" y="8" width="10" height="10" rx="1.5" /> <path d="M8 4h10a2 2 0 0 1 2 2v10" />'
    ),
    'spatial-filter': (
        '<circle cx="12" cy="12" r="8" /> <path d="M12 4v16M5 9l14 6" />'
    ),
    'riemann': (
        '<path d="M3 17c4-8 14-8 18 0" /> <circle cx="7" cy="13" r="1.5" /> <circle cx="17" cy="13" r="1.5" /> '
        '<path d="M8.5 12.2c2-1 5-1 7 0" />'
    ),
    'evaluation': (
        '<path d="M4 20h16" /> <path d="M7 16l3-4 3 2 5-7" />'
    ),
    'head-map': (
        '<circle cx="12" cy="13" r="8" /> <path d="M10 5.3 12 2.5l2 2.8" /> <path d="M4 11.5c-1.3.3-1.3 3.2 0 '
        '3.5" /> <path d="M20 11.5c1.3.3 1.3 3.2 0 3.5" /> <ellipse cx="14.5" cy="11.5" rx="3.6" ry="2.8" /> '
        '<circle cx="14.8" cy="11.4" r=".9" fill="{colour}" />'
    ),
    # the icon rail (rail.py): modes and side panels
    'activity': (
        '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.'
        '35 8.36A2 2 0 0 1 4.49 12H2" />'
    ),
    'workflow': (
        '<rect width="8" height="8" x="3" y="3" rx="2" /> <path d="M7 11v4a2 2 0 0 0 2 2h4" /> <rect width="8" '
        'height="8" x="13" y="13" rx="2" />'
    ),
    'gamepad-2': (
        '<line x1="6" x2="10" y1="11" y2="11" /> <line x1="8" x2="8" y1="9" y2="13" /> <line x1="15" x2="15.01" '
        'y1="12" y2="12" /> <line x1="18" x2="18.01" y1="10" y2="10" /> <path d="M17.32 5H6.68a4 4 0 0 0-3.978 '
        '3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 '
        '0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258'
        '-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z" />'
    ),
    'folder': (
        '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0'
        '-2 2v13a2 2 0 0 0 2 2Z" />'
    ),
    'blocks': (
        '<rect width="7" height="7" x="14" y="3" rx="1" /> <path d="M10 21V8a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v12a1 1'
        ' 0 0 0 1 1h12a1 1 0 0 0 1-1v-5a1 1 0 0 0-1-1H3" />'
    ),
    'library': (
        '<path d="m16 6 4 14" /> <path d="M12 6v14" /> <path d="M8 8v12" /> <path d="M4 4v16" />'
    ),
    'zoom-in': (
        '<circle cx="11" cy="11" r="8" /> <line x1="21" x2="16.65" y1="21" y2="16.65" /> <line x1="11" x2'
        '="11" y1="8" y2="14" /> <line x1="8" x2="14" y1="11" y2="11" />'
    ),
}

SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
       'stroke="{colour}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{shape}</svg>')


@lru_cache(maxsize=256)
def _icon(name, foreground, disabled):
    icon = QtGui.QIcon()
    for mode, colour in ((QtGui.QIcon.Mode.Normal, foreground), (QtGui.QIcon.Mode.Disabled, disabled)):
        svg = SVG.format(colour=colour, shape=SHAPES[name].replace("{colour}", colour)).encode("ascii")
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg))
        for size in (16, 18, 20, 24):
            for ratio in (1.0, 1.25, 1.5, 2.0):
                pixmap = QtGui.QPixmap(round(size * ratio), round(size * ratio))
                pixmap.setDevicePixelRatio(ratio)
                pixmap.fill(QtCore.Qt.GlobalColor.transparent)
                painter = QtGui.QPainter(pixmap)
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                renderer.render(painter, QtCore.QRectF(0, 0, size, size))
                painter.end()
                icon.addPixmap(pixmap, mode)
    return icon


def tinted(name, colour):
    """The icon `name` drawn in `colour` (a block category's, say)."""
    return QtGui.QIcon(_icon(name, colour, color("disabled_text")))


def icon(name, primary=False, role=None):
    """The icon `name` (a key of SHAPES) in the theme's text colour, a primary button's, or the theme
    colour `role` (e.g. "subtle" for a quiet mark)."""
    if name not in SHAPES:
        raise KeyError("No icon named {}".format(name))
    foreground = color(role) if role else color("primary_text") if primary else color("text")
    return QtGui.QIcon(_icon(name, foreground, color("disabled_text")))


def only(button, name, title, shortcut="", more="", size=SIZE):
    """`button` as an icon alone (GIOVA's toolbars): its name and shortcut in
    the tooltip ("Save (Ctrl+S)", then `more` on the next line) and as its accessible name. The name stays
    shown on hover even with hover tips switched off: without it an icon has no words at all."""
    button.setText("")
    button.setIcon(icon(name))
    button.setIconSize(QtCore.QSize(size, size))
    button.setAccessibleName(title)
    button.setToolTip("{} ({})".format(title, shortcut) if shortcut else title)
    if more:
        button.setToolTip(button.toolTip() + "\n" + more)
    button.setProperty("icon_only", True)
    if hasattr(button, "setToolButtonStyle"):
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
    if hasattr(button, "setAutoRaise"):
        button.setAutoRaise(True)
    return button


def put(button, name, text=None, size=SIZE, primary=False):
    """`button` with the icon beside its label (`text` replaces the label; "" leaves the icon alone, the
    label kept as its accessible name). A tool button is told to show both."""
    if text is not None:
        button.setAccessibleName(button.text() if text == "" and button.text() else text or name)
        button.setText(text)
    button.setIcon(icon(name, primary))
    button.setIconSize(QtCore.QSize(size, size))
    if hasattr(button, "setToolButtonStyle") and button.text():
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    return button
