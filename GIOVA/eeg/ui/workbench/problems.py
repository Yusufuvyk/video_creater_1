"""What went wrong in a block, said plainly, and what can be done about it from where the user is.

A block's error used to be one red line under its settings, cut off on a narrow screen, often on the
block where the trouble showed rather than where it began (no events: seen at Epochs, caused by the
Recording). Here a message becomes a headline, the rest of its explanation, and buttons that do the
usual remedies: open the setting, choose how a file is read, go to the block where it began.

`fixes` only offers what fits the block and the message; the message itself stays the one the
computation gave, so nothing is said here that the code did not find.
"""
from pyqtgraph.Qt import QtCore, QtWidgets

from ...pipeline import compute
from ..theme import color

#: The start of the Epochs block's message when the recording has no events (compute.cut_epochs).
NO_EVENTS = "This recording has no events to cut trials around"


def split(message, title=""):
    """(headline, detail) of a block's message: its first sentence, the rest; the block's own name in front
    (the Runner puts it there) is dropped, the box already says which block it is."""
    text = str(message or "").strip()
    if title and text.startswith(title + ": "):
        text = text[len(title) + 2:]
    head, found, rest = text.partition(". ")
    if found and len(head) > 12:
        return head + ".", rest.strip()
    return text, ""


def fixes(page, node_id, message):
    """[(label, tooltip, action)] the Workbench can do about `message` on `node_id`; the most likely first."""
    graph = page.graph
    node = graph.nodes.get(node_id)
    if node is None:
        return []
    out = []
    text = str(message or "")
    params = graph.params(node_id)
    if node.type == "load":
        path = str(params.get("path") or "")
        if compute.MAT_LAYOUT in text or (path.lower().endswith(".mat") and "MatLayoutUnknown" in text):
            out.append(("Choose how to read this file…",
                        "Which variable is the signal, its sampling rate, its channel names and where the events "
                        "are; kept for this file, and for every file laid out the same way if you like",
                        lambda: page.choose_layout(node_id)))
        if not path or "No such file" in text or "not found" in text.lower() or "Choose a recording" in text:
            out.append(("Choose a file…", "Pick the recording this block reads",
                        lambda: page.inspector.browse(node_id, "path")))
    if node.type == "epochs" and NO_EVENTS in text:
        out.append(("Cut a trial every few seconds…",
                    "For a recording without markers: Options, 'Without markers: every'",
                    lambda: page.inspector.focus_field(node_id, "every")))
        out.append(("Type the trials in a table",
                    "Trials from: manual - each trial's start and end, written or pasted",
                    lambda: page.inspector.set_value(node_id, "source", "manual")))
    if node.type == "epochs" and "No events labelled" in text:
        out.append(("Choose the conditions…", "Leave it empty for every label the recording has",
                    lambda: page.inspector.focus_field(node_id, "conditions")))
    source = _upstream_load(graph, node_id)
    if node.type != "load" and source is not None and (NO_EVENTS in text or "events" in text.lower()):
        out.append(("Show the Recording", "Its result lists the events the file holds",
                    lambda: page.canvas.select(source)))
    failed = [n for n in graph.upstream(node_id) if n in page.errors]
    if failed:
        out.append(("Show the block that stopped", "This block waits for its result",
                    lambda: page.canvas.select(failed[0])))
    out.append(("Help on this block", "What it does, what it needs, what it cannot do",
                lambda: page.help_for(node.type)))
    return out


def _upstream_load(graph, node_id):
    loads = [n for n in graph.upstream(node_id) if graph.nodes[n].type == "load"]
    return loads[0] if len(loads) == 1 else None


class ProblemBox(QtWidgets.QFrame):
    """A block's error at the top of its settings: what happened, why, and buttons for what to do."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("problemBox")
        self.setStyleSheet("QFrame#problemBox {{ border: 1px solid {0}; border-left: 4px solid {0}; "
                           "border-radius: 4px; }}".format(color("bad")))
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(10, 8, 10, 8)
        column.setSpacing(4)
        self.headline = QtWidgets.QLabel()
        self.headline.setWordWrap(True)
        self.headline.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.headline.setStyleSheet("color: {}; font-weight: 600;".format(color("bad")))
        self.headline.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.detail.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.actions = QtWidgets.QVBoxLayout()
        self.actions.setSpacing(4)
        column.addWidget(self.headline)
        column.addWidget(self.detail)
        column.addLayout(self.actions)
        self.hide()

    def show_problem(self, message, title="", actions=()):
        """Show `message` (empty: hide), with a button per (label, tooltip, action)."""
        while self.actions.count():
            item = self.actions.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if not message:
            self.hide()
            return
        headline, detail = split(message, title)
        self.headline.setText(headline)
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        if actions:
            what = QtWidgets.QLabel("What you can do:")
            what.setStyleSheet("color: {}; margin-top: 4px;".format(color("subtle")))
            self.actions.addWidget(what)
        for label, tip, action in actions:
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tip)
            button.setStyleSheet("text-align: left; padding: 3px 8px;")
            button.clicked.connect(lambda _c=False, a=action: a())
            self.actions.addWidget(button)
        self.show()

    def text(self):
        """The headline and detail as shown, for tests and accessibility."""
        return (self.headline.text() + " " + self.detail.text()).strip() if self.isVisible() or \
            self.headline.text() else ""
