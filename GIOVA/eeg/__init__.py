"""GIOVA: live monitoring and offline analysis of EEG recordings.

Three layers, and the dependencies only ever point downwards:

    ui       -> pipeline -> formats -> core
    ui       -> formats -> core
    ui       -> core

    core     what an EEG recording is and what can be computed from one.
             Pure numpy/scipy/mne - no Qt, no dialogs, testable headless.
    formats  how to get a Recording out of a file on disk, including the
             awkward cases: .mat layouts, unit scales, dataset presets.
    pipeline analyses as graphs of blocks (the Workbench's engine), its
             Python API and script export. No Qt.
    core never imports formats or ui; formats never imports ui. Anything
    that breaks that is a bug, not a shortcut - it was already fixed once,
    when the file-opening dispatcher sat in core and pulled the .mat reader
    in behind it.

Launch with eeg_studio.py (or run_dashboard.py) in the folder above.
"""
__all__ = ["core", "formats", "ui"]

#: The program's name and version, wherever they are shown.
NAME = "GIOVA"
VERSION = "1.0.0"
#: What it is, in a few words: under the name on Home and the start-up screen.
TAGLINE = "Open-source biosignal lab"
#: Where the code, the downloads and the issue tracker live (Help → Send feedback opens it).
REPOSITORY = "https://github.com/OWNER/GIOVA"
