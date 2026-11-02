"""What was used last - recordings, pipelines, reports - for Home's "Continue".

Kept in the settings as a short list per kind, newest first. A file that has since been
moved or deleted is left out when the list is read, not when it is written, so a drive
plugged back in brings its entries back.

No widgets.
"""
import json
from pathlib import Path
import time

from .preferences import user_settings

RECORDINGS, PIPELINES, REPORTS = "recordings", "pipelines", "reports"
#: entries kept per kind
KEEP = 6


def _key(kind):
    return "recent/" + kind


def _load(kind):
    try:
        entries = json.loads(str(user_settings().value(_key(kind), "[]") or "[]"))
    except ValueError:
        return []
    return [e for e in entries if isinstance(e, dict) and e.get("path")]


def remember(kind, path, **info):
    """`path` first in its kind's list; `info` (a name, an accuracy) kept with it, earlier info too."""
    if not path:
        return
    path = str(Path(path).resolve())
    entries = _load(kind)
    old = next((e for e in entries if e["path"] == path), {})
    entry = {**old, **{k: v for k, v in info.items() if v is not None}, "path": path, "when": time.time()}
    entries = [entry] + [e for e in entries if e["path"] != path]
    user_settings().setValue(_key(kind), json.dumps(entries[:KEEP]))


def entries(kind):
    """The kind's entries whose files are still there, newest first."""
    return [e for e in _load(kind) if Path(e["path"]).exists()]


def forget(kind=None):
    for name in ((kind,) if kind else (RECORDINGS, PIPELINES, REPORTS)):
        user_settings().remove(_key(name))


def ago(when, now=None):
    """'just now', '5 min ago', '3 h ago', 'yesterday', '4 days ago'."""
    seconds = max(0.0, (now or time.time()) - float(when))
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return "{:.0f} min ago".format(seconds / 60)
    if seconds < 86400:
        return "{:.0f} h ago".format(seconds / 3600)
    days = int(seconds // 86400)
    return "yesterday" if days == 1 else "{} days ago".format(days)
