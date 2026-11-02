"""Projects: a project is a folder on disk, shown as a tree of its recordings.

Nothing is copied and there is no database. The program remembers which folders
were added; everything else is read from the folder each time, so a project
moved or tidied in the file manager is still the same project. Recordings keep
their sidecars beside them, as they always did.

A project may carry a few words about itself (participant, notes) in
`eegstudio_project.json`; a folder without that file is just as much a project.

No Qt - safe to import anywhere.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

from .config import EVENTS_SUFFIX, RECORDING_SUFFIXES

INFO_FILE = "eegstudio_project.json"
#: What a project may say about itself. All optional.
INFO_FIELDS = ("name", "participant", "notes")
ANALYSIS_SUFFIX = ".analysis.json"

FOLDER, RECORDING, ANALYSIS = "folder", "recording", "analysis"

#: Folders that never hold recordings and can be large: environments, builds, caches.
SKIP_FOLDERS = frozenset({"__pycache__", "node_modules", "site-packages", "_internal", "build", "dist",
                          "venv", "env", "envs"})
#: A tree is for finding a recording, not for mirroring a disk.
MAX_DEPTH = 5
MAX_ENTRIES = 3000


@dataclass
class Node:
    """One line of the tree: a folder, a recording, or a saved analysis of one."""
    name: str
    path: Path
    kind: str
    children: list = field(default_factory=list)


def _is_brainflow_csv(path):
    """A BrainFlow recording is tab-separated numbers with no header.

    A project folder also holds exported tables and notes saved as .csv; listing
    those as recordings only produces a failure when one is opened.
    """
    try:
        with open(path, "rb") as handle:
            first = handle.readline(4096)
    except OSError:
        return False
    if b"\t" not in first:
        return False
    try:
        float(first.split(b"\t", 1)[0])
    except ValueError:
        return False
    return True


def is_recording(path):
    """True for a file the program can open as a recording (sidecars are not)."""
    path = Path(path)
    name = path.name
    if path.suffix.lower() not in RECORDING_SUFFIXES:
        return False
    if name.endswith(EVENTS_SUFFIX) or name.startswith("features_") or name.endswith(".mapping.json"):
        return False
    return path.suffix.lower() != ".csv" or _is_brainflow_csv(path)


def tree(folder, max_depth=MAX_DEPTH, max_entries=MAX_ENTRIES):
    """The recordings under `folder` as nested nodes: folders first, then recordings by name.

    Folders with no recording anywhere below them are left out, as are hidden
    ones and environments. A saved analysis named after a recording
    (`<recording>.analysis.json`, the name Analysis proposes) hangs under it.
    """
    budget = [int(max_entries)]

    def walk(directory, depth):
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return []
        folders, recordings = [], []
        names = {entry.name for entry in entries}
        for entry in entries:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            if entry.is_dir():
                if depth < max_depth and not entry.name.startswith(".") and entry.name.lower() not in SKIP_FOLDERS:
                    children = walk(entry, depth + 1)
                    if children:
                        folders.append(Node(entry.name, entry, FOLDER, children))
            elif is_recording(entry):
                node = Node(entry.name, entry, RECORDING)
                saved = entry.stem + ANALYSIS_SUFFIX
                if saved in names:
                    node.children.append(Node(saved, entry.with_name(saved), ANALYSIS))
                recordings.append(node)
        return folders + recordings

    folder = Path(folder)
    return walk(folder, 1) if folder.is_dir() else []


def count_recordings(nodes):
    return sum(1 if node.kind == RECORDING else count_recordings(node.children)
               for node in nodes if node.kind != ANALYSIS)


def load_info(folder):
    """What the project says about itself; {} when it says nothing or the file is damaged."""
    try:
        raw = json.loads((Path(folder) / INFO_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {key: str(raw[key]).strip() for key in INFO_FIELDS if str(raw.get(key) or "").strip()}


def save_info(folder, info):
    """Write the project's words about itself; with nothing to say, the file is removed."""
    path = Path(folder) / INFO_FILE
    kept = {key: str(info.get(key) or "").strip() for key in INFO_FIELDS if str(info.get(key) or "").strip()}
    if not kept:
        try:
            path.unlink()
        except OSError:
            pass
        return {}
    path.write_text(json.dumps(dict(version=1, **kept), ensure_ascii=False, indent=2), encoding="utf-8")
    return kept


def display_name(folder, info=None):
    """The name given in the project's info, else the folder's own."""
    info = load_info(folder) if info is None else info
    return info.get("name") or Path(folder).name or str(folder)


def same_folder(a, b):
    """Whether two paths name one folder, however they are spelled."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)
