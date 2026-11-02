"""The game process: put `StudioEEG` where the game expects its board class, then
run the game script exactly as `python ball_game.py` would."""
import importlib
import os
from pathlib import Path
import runpy
import sys

from . import manifest as mf
from .studio_eeg import StudioEEG


def install(game):
    """Import the game's acquisition module and replace only its EEG class."""
    folder = str(Path(game.folder).resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    module = importlib.import_module(game.eeg["module"])
    setattr(module, game.eeg["class"], StudioEEG)
    return module


def run(folder):
    game = mf.load(folder)
    os.chdir(game.folder)
    install(game)
    sys.argv = [str(Path(game.folder) / game.entry)]
    runpy.run_path(sys.argv[0], run_name="__main__")


def main(argv):
    folder = argv[argv.index("--game") + 1]
    try:
        run(folder)
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return 1
    return 0
