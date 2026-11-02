"""Every text file of the project is UTF-8 with one line ending: LF, Windows batch files CRLF.

Mixed endings made two copies of the same file differ on every line, and a change could no longer be carried
from one to the other by comparing them (1 Nov 2026). tools/line_endings.py --fix puts a folder right.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import line_endings  # noqa: E402


class TextFiles(unittest.TestCase):
    def test_one_line_ending_and_utf8_everywhere(self):
        files = list(line_endings.text_files(ROOT))
        self.assertGreater(len(files), 50)                       # the walk found the project
        bad = ["{}: {}".format(path.relative_to(ROOT), why) for path in files
               for why in [line_endings.problem(path)] if why]
        self.assertEqual(bad, [], "run tools/line_endings.py --fix")

    def test_the_rule_itself(self):
        self.assertEqual(line_endings.wanted(Path("build_exe.bat")), b"\r\n")
        self.assertEqual(line_endings.wanted(Path("eeg/core/epochs.py")), b"\n")
        self.assertIn(".gitattributes", {line_endings.suffix(Path(".gitattributes"))})


if __name__ == "__main__":
    unittest.main()
