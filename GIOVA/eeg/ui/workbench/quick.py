"""Quick analysis on the Workbench: a recording and a ready profile, and the result.

Home's "Analyse a recording" and the Projects pane's "Quick analysis" ask for the two things the old Analysis
page started from - the file and a profile (the built-in ones and the user's own, the same list) - and open
the profile's pipeline (`pipeline.convert.from_profile`) in a new tab on that file, already running. The
sheet that was in view is left as it is.

Qt only; the pipeline is made by `pipeline.convert`.
"""
from pathlib import Path

from pyqtgraph.Qt import QtWidgets

from ...core import analysis_profiles as ap
from ...core.config import RECORDING_FILTER
from ..widgets import NoScrollComboBox

#: The profile chosen last, offered first next time (this session).
LAST = {"profile": "Eyes open / closed"}


class QuickDialog(QtWidgets.QDialog):
    """[recording …] [profile ▾] [description]  Open / Cancel."""

    def __init__(self, path="", parent=None, root=None):
        super().__init__(parent)
        self.setWindowTitle("Quick analysis")
        self.profiles = {p.name: p for p in ap.available(root)}
        form = QtWidgets.QFormLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit(str(path or ""))
        self.path.setPlaceholderText("The recording to analyse")
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        form.addRow("Recording", row)
        self.profile = NoScrollComboBox()
        self.profile.addItems(list(self.profiles))
        if LAST["profile"] in self.profiles:
            self.profile.setCurrentText(LAST["profile"])
        self.profile.currentTextChanged.connect(self._describe)
        form.addRow("Profile", self.profile)
        self.about = QtWidgets.QLabel()
        self.about.setWordWrap(True)
        form.addRow("", self.about)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Open |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._describe(self.profile.currentText())

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Recording", self.path.text(), RECORDING_FILTER)
        if path:
            self.path.setText(path)

    def _describe(self, name):
        profile = self.profiles.get(name)
        if profile is not None:
            kind = "built in, read-only" if name in ap.BUILTIN_NAMES else "your profile"
            self.about.setText("{} ({}).".format(profile.description or "No description", kind))

    def _accept(self):
        if not self.path.text().strip() or not Path(self.path.text().strip()).exists():
            self.about.setText("Choose a recording that exists.")
            return
        self.accept()

    def chosen(self):
        """(recording path, AnalysisProfile)."""
        LAST["profile"] = self.profile.currentText()
        return self.path.text().strip(), self.profiles[self.profile.currentText()]
