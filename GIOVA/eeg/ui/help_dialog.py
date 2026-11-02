"""The Help window: topics and the block library on the left, the text on the right, a search box on top.

Read-only and non-modal, built once and kept, like the Settings dialog: it can
stay open beside the Monitor while the operator follows the steps.
"""
import html
import re

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ..pipeline.graph import block
from . import help_blocks
from .help_content import SECTIONS, TAB_TOPICS, TOPICS
from .theme import color
from .widgets import fit_to_screen

def style():
    """The help page's look, in the window's theme and GIOVA's type: Inter, headings in the
    strong text colour (not the accent: it marks only what is selected), tables without a grid - a quiet header
    row and one hairline under each row - and code in JetBrains Mono on the fields' ground."""
    c = {k: color(k) for k in ("text", "strong", "line", "field", "subtle", "input", "link")}
    return """
body {{ font-family: Inter, "Segoe UI"; font-size: 10pt; color: {text}; }}
p, li {{ line-height: 140%; }}
h2 {{ color: {strong}; font-size: 15pt; font-weight: 600; margin-top: 2px; margin-bottom: 10px; }}
h3 {{ color: {strong}; font-size: 11pt; font-weight: 600; margin-top: 18px; margin-bottom: 6px; }}
table {{ border-collapse: collapse; margin-top: 4px; margin-bottom: 12px; border-style: none; }}
th, td {{ border-style: none; border-bottom-style: solid; border-bottom-width: 1px; border-bottom-color: {line};
          padding: 6px 12px 6px 0; vertical-align: top; }}
th {{ color: {subtle}; font-weight: 600; text-align: left; border-bottom-color: {field}; }}
code {{ background: {input}; color: {text}; font-family: "JetBrains Mono", Consolas, monospace; font-size: 9pt; }}
a {{ color: {link}; }}
.note {{ color: {subtle}; }}
""".format(**c)


def page(markup):
    """A topic as the browser shows it: tables across the page's width, no cell spacing."""
    return markup.replace("<table>", '<table width="100%" cellspacing="0">')


def feedback_url():
    """A new issue on GIOVA's repository, its text started with the version and Windows release: what helps
    to reproduce a problem, and nothing about the person or their files."""
    import platform
    from urllib.parse import urlencode
    from .. import NAME, REPOSITORY, VERSION
    body = ("**What happened, or what would help:**\n\n\n**Steps to see it:**\n1. \n\n"
            "---\n{} {} · {} {} · Python {}".format(NAME, VERSION, platform.system(), platform.release(),
                                                 platform.python_version()))
    return "{}/issues/new?{}".format(REPOSITORY, urlencode({"body": body}))


def plain_text(markup):
    """Visible words of an HTML topic, for searching."""
    return html.unescape(re.sub(r"<[^>]+>", " ", markup)).lower()


#: A block page's picture: "block-picture:<type>", drawn when the page first asks for it.
PICTURE = "block-picture"


def picture_size(type_name):
    """(width, height) of a block's picture: the card, its ports and a little margin."""
    from .workbench.canvas import HEADER, PORT, ROW, WIDTH
    kind = block(type_name)
    return WIDTH + 2 * PORT + 8, HEADER + 22 + max(len(kind.inputs), len(kind.outputs), 1) * ROW + 8


def block_picture(type_name, ratio=2.0):
    """The block as it looks on the sheet - the Blocks list's own preview, its ports' rows only - drawn sharp
    at `ratio`, on a transparent ground."""
    from .workbench.canvas import BlockPreview
    preview = BlockPreview()
    preview.setFixedSize(*picture_size(type_name))
    preview.set_block(type_name)
    image = QtGui.QImage(round(preview.width() * ratio), round(preview.height() * ratio),
                         QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(ratio)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    preview.render(image, QtCore.QPoint(), QtGui.QRegion(), QtWidgets.QWidget.RenderFlag.DrawChildren)
    return image


def with_picture(type_name, markup):
    """A block's page with its picture under the heading and its category line."""
    # the page's 140 % line height would add 40 % of the picture's height under it
    tag = '<p style="line-height: 100%;"><img src="{}:{}" width="{}" height="{}"></p>'.format(PICTURE, type_name, *picture_size(type_name))
    end = markup.find("</p>")
    return markup[:end + 4] + tag + markup[end + 4:] if end >= 0 else tag + markup


class HelpBrowser(QtWidgets.QTextBrowser):
    """The page view; it draws a block's picture when the page asks for it, in the theme of that moment."""

    def loadResource(self, kind, url):
        if url.scheme() == PICTURE and kind == QtGui.QTextDocument.ResourceType.ImageResource:
            return block_picture(url.path(), ratio=max(2.0, self.devicePixelRatioF()))
        return super().loadResource(kind, url)


class HelpDialog(QtWidgets.QDialog):
    """Topics grouped on the left (the guides, then the block library, one page per block), the page on the right.

    Every entry is found by its key: a topic's title, "block:<type>" for a block's page, "category:<name>" for
    the blocks of one category. Searching keeps the entries containing every word typed, and their groups.
    """

    BLOCKS = "Block library"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GIOVA · Help")
        fit_to_screen(self, 980, 700)
        self.setModal(False)
        self._html = dict(TOPICS)
        self._html[self.BLOCKS] = help_blocks.LIBRARY_PAGE.format(help_blocks.overview())
        self._items = {}

        layout = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search help (e.g. BIAS, alpha, CSP, COM port)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        split = QtWidgets.QSplitter()
        self.topics = QtWidgets.QTreeWidget()
        self.topics.setHeaderHidden(True)
        self.topics.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._fill()
        self.topics.expandAll()
        # the widest entry, deepest indent included; the column itself stretches to the view, so not its width
        list_width = self.topics.sizeHintForColumn(0) + 28
        self.topics.setMinimumWidth(list_width)
        self.topics.collapseItem(self._items[self.BLOCKS])         # 40 blocks: open when wanted
        self.topics.currentItemChanged.connect(lambda item, _: self._show(item))
        split.addWidget(self.topics)
        self.browser = HelpBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.document().setDefaultStyleSheet(style())
        self.browser.document().setDocumentMargin(20)          # the page breathes inside its frame
        split.addWidget(self.browser)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([list_width, 10_000])
        layout.addWidget(split, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        feedback = buttons.addButton("Send feedback…", QtWidgets.QDialogButtonBox.ButtonRole.HelpRole)
        feedback.setToolTip("Opens a new issue on GIOVA's page in your browser, with the version filled in. "
                            "Nothing is sent until you submit it there.")
        feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(feedback_url())))
        layout.addWidget(buttons)
        self._text = {key: plain_text(body) for key, body in self._html.items()}
        self.topics.setCurrentItem(self._items[TOPICS[0][0]])

    def _add(self, parent, label, key=None):
        item = QtWidgets.QTreeWidgetItem(parent, [label])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, key)
        if key is None:                                     # a heading: it groups, it has no page
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        else:
            self._items[key] = item
        if parent is self.topics:                           # the sections, the library's own page among them
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
        return item

    def _fill(self):
        """The guides' sections, and the block library before Reference, grouped as the block list is."""
        for section, titles in SECTIONS:
            if section == "Reference":
                self._fill_blocks()
            heading = self._add(self.topics, section)
            for title in titles:
                self._add(heading, title, title)

    def _fill_blocks(self):
        library = self._add(self.topics, self.BLOCKS, self.BLOCKS)
        groups = {}
        for group, category, types in help_blocks.by_category():
            parent = library
            if group:
                if group not in groups:
                    groups[group] = self._add(library, group, "category:" + group)
                    self._html["category:" + group] = "<h2>{}</h2>{}".format(html.escape(group), help_blocks.overview(
                        [c for g, c, _ in help_blocks.by_category() if g == group]))
                parent = groups[group]
            self._add(parent, category, "category:" + category)
            self._html["category:" + category] = "<h2>{}</h2>{}".format(html.escape(category),
                                                                        help_blocks.overview([category]))
            for type_name in types:
                self._add(self._items["category:" + category], block(type_name).title, "block:" + type_name)
                self._html["block:" + type_name] = help_blocks.page(type_name)

    def key(self, item=None):
        """The key of `item` (the current entry by default), or None for a heading."""
        item = item if item is not None else self.topics.currentItem()
        return None if item is None else item.data(0, QtCore.Qt.ItemDataRole.UserRole)

    def _show(self, item):
        key = self.key(item)
        if key in self._html:
            markup = self._html[key]
            if key.startswith("block:"):
                markup = with_picture(key[len("block:"):], markup)
            self.browser.setHtml(page(markup))
            words = self.search.text().strip()
            if words:
                self.browser.find(words.split()[0])

    def _filter(self, text):
        """Keep only entries containing every word typed, with the headings above them."""
        words = text.lower().split()

        def keep(item):
            key = self.key(item)
            hit = key is not None and all(word in self._text[key] for word in words)
            children = [keep(item.child(k)) for k in range(item.childCount())]
            shown = hit or any(children)
            item.setHidden(not shown)
            if words and any(children):
                item.setExpanded(True)
            return shown

        for k in range(self.topics.topLevelItemCount()):
            keep(self.topics.topLevelItem(k))
        current = self.topics.currentItem()
        if current is None or current.isHidden():
            first = next((i for i in self._items.values() if not i.isHidden()
                          and all(w in self._text[self.key(i)] for w in words)), None)
            if first is not None:
                self.topics.setCurrentItem(first)
        else:
            self._show(current)

    def visible(self):
        """Keys of the entries shown now, top to bottom."""
        out = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.topics)
        while iterator.value():
            item = iterator.value()
            if not item.isHidden() and self.key(item) is not None:
                out.append(self.key(item))
            iterator += 1
        return out

    def open_at(self, title=None):
        """Show the window at `title` (a topic, a main tab name, or "block:<type>"), raised."""
        title = TAB_TOPICS.get(title, title)
        if title in self._items:
            self.search.clear()
            item = self._items[title]
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            self.topics.setCurrentItem(item)
            self.topics.scrollToItem(item)
        self.show()
        self.raise_()
        self.activateWindow()
        return self
