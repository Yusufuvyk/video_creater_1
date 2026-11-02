"""THIRD_PARTY_NOTICES.md from what the program carries, and their licence texts beside the exe.

Every package of requirements-lock.txt, read from its installed metadata (name, version, licence,
home page), and the fonts and icons kept in eeg/ui/assets. With --copy DIR the licence files each
package ships are copied to DIR/<package>/, as the packaged program must carry them.

Run: .venv/Scripts/python tools/third_party.py [--copy dist/GIOVA/licenses]
"""
import argparse
from importlib import metadata
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
LICENCE_FILE = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE|AUTHORS)[^/]*$", re.IGNORECASE)

#: what is kept inside the program itself rather than installed as a package
BUNDLED = [
    ("Inter 4.1", "SIL Open Font License 1.1", "https://rsms.me/inter/", "eeg/ui/assets/fonts/Inter-OFL.txt"),
    ("JetBrains Mono 2.304", "SIL Open Font License 1.1", "https://www.jetbrains.com/lp/mono/",
     "eeg/ui/assets/fonts/JetBrainsMono-OFL.txt"),
    ("Lucide icons 0.468.0 (a subset, as SVG paths)", "ISC", "https://lucide.dev", "eeg/ui/icons.py (header)"),
]
#: what the packaged exe itself is made of besides the packages: Python, and PyInstaller's bootloader
RUNTIME = [
    ("Python {}".format(sys.version.split()[0]), "Python Software Foundation License", "https://www.python.org",
     "licenses/Python/LICENSE.txt"),
    ("PyInstaller bootloader (starts the exe)", "GPL-2.0-or-later with the bootloader exception",
     "https://pyinstaller.org", "licenses/pyinstaller/"),
]


def locked():
    """[(name, version)] of requirements-lock.txt."""
    out = []
    for line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            out.append((name.strip(), version.strip()))
    return out


#: packages whose metadata names no licence, read from the licence file they ship (checked by hand)
STATED_IN_FILE = {"brainflow": "MIT"}


def licence(meta):
    """The licence as the package states it: its SPDX expression, else a short License field, else the
    trove classifiers."""
    if meta["Name"].lower() in STATED_IN_FILE:
        return STATED_IN_FILE[meta["Name"].lower()] + " (its LICENSE.txt)"
    expression = meta.get("License-Expression")
    if expression:
        return expression
    field = (meta.get("License") or "").strip()
    if field and len(field) < 80 and "\n" not in field:
        return field
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    return field.splitlines()[0][:80] if field else "see its licence file"


def home(meta):
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        if label.strip().lower() in ("homepage", "home", "source", "repository", "source code"):
            return url.strip()
    return (meta.get("Home-page") or "").strip()


def licence_files(dist):
    return [f for f in dist.files or [] if LICENCE_FILE.search(str(f).replace("\\", "/"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", help="copy each package's licence files into this folder")
    args = parser.parse_args()
    rows, missing = [], []
    for name, version in locked():
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        meta = dist.metadata
        if dist.version != version:
            missing.append("{} (installed {}, locked {})".format(name, dist.version, version))
        rows.append((meta["Name"], dist.version, licence(meta), home(meta)))
        if args.copy:
            target = Path(args.copy) / meta["Name"]
            for entry in licence_files(dist):
                source = Path(dist.locate_file(entry))
                if source.is_file():
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target / source.name)
    if missing:
        raise SystemExit("Not installed as locked: " + ", ".join(missing))
    lines = ["# Third-party notices", "",
             "GIOVA is licensed under the GNU General Public License v3.0 (see LICENSE). It is built on the",
             "open-source work below, each under its own licence. The packaged program carries their licence",
             "texts in its `licenses` folder. Qt for Python (PySide6, shiboken6) is used under the LGPL-3.0:",
             "the packaged program keeps Qt as separate libraries, which can be replaced with other builds of",
             "the same version.", "",
             "## Python packages", "",
             "| Package | Version | Licence | Home |", "|---|---|---|---|"]
    for name, version, text, url in sorted(rows, key=lambda r: r[0].lower()):
        lines.append("| {} | {} | {} | {} |".format(name, version, text.replace("|", "/"), url))
    lines += ["", "## Kept inside the program", "", "| Work | Licence | Home | Licence text |", "|---|---|---|---|"]
    for work, text, url, where in BUNDLED:
        lines.append("| {} | {} | {} | `{}` |".format(work, text, url, where))
    lines += ["", "## The packaged program itself", "", "| Work | Licence | Home | Licence text |", "|---|---|---|---|"]
    for work, text, url, where in RUNTIME:
        lines.append("| {} | {} | {} | `{}` (in the download) |".format(work, text, url, where))
    if args.copy:
        target = Path(args.copy)
        for _, _, _, where in BUNDLED:
            source = ROOT / where.split(" ")[0]
            if source.suffix == ".txt":
                shutil.copyfile(source, target / source.name)
        python_licence = Path(sys.base_prefix) / "LICENSE.txt"
        if python_licence.is_file():
            (target / "Python").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(python_licence, target / "Python" / "LICENSE.txt")
        try:
            builder = metadata.distribution("pyinstaller")
        except metadata.PackageNotFoundError:
            raise SystemExit("PyInstaller is not installed: its licence cannot be copied")
        (target / "pyinstaller").mkdir(parents=True, exist_ok=True)
        for entry in licence_files(builder):
            source = Path(builder.locate_file(entry))
            if source.is_file():
                shutil.copyfile(source, target / "pyinstaller" / source.name)
    (ROOT / "THIRD_PARTY_NOTICES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("THIRD_PARTY_NOTICES.md: {} packages".format(len(rows)))


if __name__ == "__main__":
    main()
