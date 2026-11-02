"""The mod file that makes a game folder playable from GIOVA.

A game folder (alpha_ball_game_openbci and its copies) holds config.py, acquisition.py,
processing.py and the game script. Beside them goes one small JSON file:

    {
      "name": "Alpha ball (v2)",
      "description": "...",
      "entry": "ball_game.py",
      "eeg": {"module": "acquisition", "class": "EEG"},
      "asks": ["Participant name", "Session number"],
      "requires": ["pygame"]
    }

Copy the folder, change the name, and the copy is a second game. Nothing in
the game's own code has to change.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

FILENAME = "eegstudio_game.json"


@dataclass
class GameManifest:
    name: str
    entry: str = "ball_game.py"
    description: str = ""
    eeg: dict = field(default_factory=lambda: {"module": "acquisition", "class": "EEG"})
    asks: list = field(default_factory=lambda: ["Participant name", "Session number"])
    requires: list = field(default_factory=lambda: ["pygame"])
    folder: str = ""

    def validate(self):
        folder = Path(self.folder)
        if not self.name.strip():
            raise ValueError("The game needs a name")
        if not (folder / self.entry).is_file():
            raise ValueError("{} not found in {}".format(self.entry, folder))
        module = self.eeg.get("module", "")
        if not module or not (folder / (module + ".py")).is_file():
            raise ValueError("{}.py not found in {}".format(module or "(module)", folder))
        if not self.eeg.get("class"):
            raise ValueError("Name the EEG class to replace")
        return self

    def to_json(self):
        data = asdict(self)
        data.pop("folder")
        return json.dumps(data, indent=2, ensure_ascii=False)


def load(folder):
    folder = Path(folder)
    path = folder / FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read {}: {}".format(path, exc)) from exc
    known = set(GameManifest.__dataclass_fields__) - {"folder"}
    try:
        manifest = GameManifest(**{k: v for k, v in raw.items() if k in known}, folder=str(folder))
    except TypeError as exc:
        raise ValueError("Malformed {}".format(path)) from exc
    return manifest.validate()


def create(folder, name=None):
    """Write a manifest for a Demo_basic-style folder and return it."""
    folder = Path(folder)
    entry = next((c for c in ("ball_game.py", "game.py", "main.py") if (folder / c).is_file()), None)
    if entry is None:
        raise ValueError("No ball_game.py, game.py or main.py in {}".format(folder))
    manifest = GameManifest(name=name or folder.name, entry=entry, folder=str(folder)).validate()
    (folder / FILENAME).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def find(folders):
    """[(manifest or None, folder, problem)] for each folder and each subfolder with a mod file."""
    found, seen = [], set()
    for folder in folders:
        folder = Path(folder)
        candidates = [folder] if (folder / FILENAME).is_file() else (
            sorted(p for p in folder.iterdir() if (p / FILENAME).is_file()) if folder.is_dir() else [])
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                found.append((load(candidate), candidate, ""))
            except ValueError as exc:
                found.append((None, candidate, str(exc)))
    return found
