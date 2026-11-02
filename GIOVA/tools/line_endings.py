"""One line ending for every text file of the project: LF, except Windows batch files (CRLF).

The project had both: some files written on Windows with CRLF, others with LF. Two copies of a file then
differed on every line though their text was the same, and a change could not be carried from one to the
other by comparing them. `.gitattributes` keeps git from bringing it back; this keeps the folder itself
right, and tests/test_text_files.py fails as soon as a file breaks the rule.

Run: .venv/Scripts/python tools/line_endings.py [--fix] [folder]
     without --fix it only lists the files that break the rule (exit code 1 if any).
"""
from pathlib import Path
import sys

#: suffixes read as text; everything else (images, recordings, zips, the exe) is left alone
TEXT = {".py", ".md", ".txt", ".json", ".cff", ".spec", ".ps1", ".toml", ".cfg", ".ini", ".yml", ".yaml",
        ".html", ".css", ".js", ".svg", ".csv", ".bat", ".cmd", ".gitignore", ".gitattributes", ".editorconfig"}
#: cmd.exe reads batch files with CRLF; given LF, labels and `goto` can misbehave
CRLF = {".bat", ".cmd"}
#: folders that are not the project's own text: environments, builds, data, caches
SKIP = {".venv", "venv", "build", "dist", "__pycache__", ".tmp", "Data", ".git", "build_icon", ".pytest_cache",
        ".ruff_cache", ".hypothesis"}


def suffix(path):
    return path.suffix.lower() if path.suffix else path.name.lower()


def text_files(root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and suffix(path) in TEXT and not SKIP.intersection(path.relative_to(root).parts[:-1]):
            yield path


def wanted(path):
    return b"\r\n" if suffix(path) in CRLF else b"\n"


def problem(path):
    """Why `path` breaks the rule, or None."""
    data = path.read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "not UTF-8"
    lone = data.replace(b"\r\n", b"")
    if wanted(path) == b"\n" and b"\r" in data:
        return "CRLF or CR line endings (want LF)"
    if wanted(path) == b"\r\n" and b"\n" in lone:
        return "LF line endings (want CRLF)"
    return None


def fix(path):
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if wanted(path) == b"\r\n":
        data = data.replace(b"\n", b"\r\n")
    path.write_bytes(data)


def main(argv):
    fixing = "--fix" in argv
    folders = [a for a in argv if not a.startswith("--")] or [Path(__file__).resolve().parents[1]]
    bad = [(path, why) for folder in folders for path in text_files(folder) for why in [problem(path)] if why]
    for path, why in bad:
        if fixing and why != "not UTF-8":
            fix(path)
        print("{} {}: {}".format("fixed" if fixing and why != "not UTF-8" else "wrong", path, why))
    print("{} file(s) {}".format(len(bad), "fixed" if fixing else "break the rule"))
    return 1 if bad and not fixing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
