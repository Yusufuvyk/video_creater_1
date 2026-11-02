"""Launching GIOVA from source.

Kept deliberately thin, and free of Qt imports until the interpreter has been
checked - PyCharm readily launches this with an interpreter from some other
project, and an opaque ImportError is a poor way to find that out. The packaged
exe starts from eeg_studio.py instead.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The project's own environment, made as the README says (`python -m venv .venv` in this folder).
VENV_PYTHON = HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
GUARD = "_GIOVA_RELAUNCHED"


def relaunch_if_needed():
    """Re-run under the project venv when the current interpreter lacks Qt."""
    if os.environ.get(GUARD):
        return  # already retried once; do not loop
    try:
        import pyqtgraph.Qt  # noqa: F401  (loads Qt itself: a broken Qt DLL fails here, not later)
        return  # this interpreter is fine
    except Exception:
        pass
    if not VENV_PYTHON.exists():
        return  # nothing better to offer - let the real error surface

    print("[GIOVA] {} has no Qt binding".format(Path(sys.executable).name))
    print("[GIOVA] relaunching with {}".format(VENV_PYTHON))
    result = subprocess.run(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=dict(os.environ, **{GUARD: "1"}),
    )
    sys.exit(result.returncode)


def main():
    relaunch_if_needed()
    sys.path.insert(0, str(HERE))
    from pyqtgraph.Qt import QtWidgets
    from eeg.ui.splash import show_splash
    app = QtWidgets.QApplication(sys.argv)
    splash = show_splash(app)
    from eeg.ui.app import main as run
    sys.exit(run(splash))


if __name__ == "__main__":
    main()
