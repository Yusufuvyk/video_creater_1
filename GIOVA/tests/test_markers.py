"""Event markers from the files beside a recording: `core/events.py`, `config.events_for` and `epochs.raw_events_for`.

A marker CSV holds `unix_time,label` rows. Each marker runs until the next one and the last to the end of the
recording; a paradigm written down instead gives one block per repeat of its cycle. Which file wins when there are
several is fixed: events stored in the recording, an `.events.json` sidecar, a marker CSV, a paradigm, and the
bare codes of a marker channel last.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from test_trial_analysis import scratch  # noqa: E402
from eeg.core import config, epochs as ep, events as ev, paradigm as pdm  # noqa: E402
from eeg.core.sources import Recording  # noqa: E402

T0 = 1_700_000_000.0


def clocked(seconds=20.0, fs=100.0, events=None):
    """A recording whose samples carry unix times from T0, as a live session writes them."""
    n = int(seconds * fs)
    return Recording(np.zeros((1, n)), fs, ["Cz"], timestamps=T0 + np.arange(n) / fs, events=events)


class MarkerFileTests(unittest.TestCase):
    def test_rows_are_sorted_and_what_is_not_a_marker_is_skipped(self):
        with scratch() as folder:
            path = folder / "rec_events.csv"
            path.write_text("# written by hand\n\ntime,label\n{0}.5, task \n{0},rest\nnot a time,x\n{1}\n"
                            "{2},end,extra column\n".format(int(T0) + 2, int(T0) + 3, int(T0) + 4), encoding="utf-8")
            self.assertEqual(ev.load_events(path), [(T0 + 2, "rest"), (T0 + 2.5, "task"), (T0 + 4, "end")])

    def test_a_file_saved_by_a_spreadsheet_reads_the_same(self):
        # Excel's "CSV UTF-8" starts with a byte order mark, and Windows ends lines with CR LF: the first
        # marker was read as "﻿1700000001", not a number, and dropped as if it were a header
        with scratch() as folder:
            plain, excel = folder / "a_events.csv", folder / "b_events.csv"
            text = "{},rest\n{},task\n".format(T0 + 1, T0 + 2)
            plain.write_text(text, encoding="utf-8")
            excel.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
            self.assertEqual(ev.load_events(excel), ev.load_events(plain))
            self.assertEqual(len(ev.load_events(excel)), 2)

    def test_each_marker_runs_until_the_next_and_the_last_to_the_end(self):
        markers = [(T0 - 1, "before"), (T0, "rest"), (T0 + 2, "task"), (T0 + 2, "cue"), (T0 + 5, "rest"),
                   (T0 + 8, "after")]
        self.assertEqual(ev.to_blocks(markers, T0, 8.0),
                         [ev.Block(0.0, 2.0, "rest"), ev.Block(2.0, 5.0, "cue"), ev.Block(5.0, 8.0, "rest")])
        self.assertEqual(ev.to_blocks([], T0, 8.0), [])

    def test_a_paradigm_gives_one_block_per_whole_cycle(self):
        paradigm = pdm.Paradigm("trial", [("rest", 2.0), ("cue", 1.0), ("task", 3.0)], repeats=4, start_offset=1.0)
        blocks = ev.blocks_by_cycle(paradigm, 20.0)
        self.assertEqual(blocks, [ev.Block(1.0, 7.0, "trial"), ev.Block(7.0, 13.0, "trial"),
                                  ev.Block(13.0, 19.0, "trial")])         # the fourth would end at 25 s: not a trial
        self.assertEqual(ev.blocks_by_cycle(pdm.Paradigm("empty"), 20.0), [])
        with scratch() as folder:
            pdm.save(pdm.path_for(folder / "rec.csv"), paradigm)
            again = pdm.load_for(folder / "rec.csv")
            self.assertEqual(again.to_dict(), paradigm.to_dict())
            pdm.path_for(folder / "bad.csv").write_text("{not json", encoding="utf-8")
            self.assertIsNone(pdm.load_for(folder / "bad.csv"))          # a broken file never blocks the recording
            self.assertIsNone(pdm.load_for(folder / "none.csv"))


class WhichFileTests(unittest.TestCase):
    def test_a_recordings_own_marker_file_then_its_subjects(self):
        with scratch() as folder:
            run = folder / "alpha_run1.csv"
            self.assertIsNone(config.events_for(run))
            (folder / "alpha_events.csv").write_text("", encoding="utf-8")
            self.assertEqual(config.events_for(run).name, "alpha_events.csv")       # one file for the subject
            (folder / "alpha_run1_events.csv").write_text("", encoding="utf-8")
            self.assertEqual(config.events_for(run).name, "alpha_run1_events.csv")  # the recording's own wins

    def test_where_events_come_from_in_order(self):
        with scratch() as folder:
            path = folder / "rec.csv"
            pdm.save(pdm.path_for(path), pdm.Paradigm("trial", [("rest", 2.0), ("task", 3.0)], repeats=2))
            table, source = ep.raw_events_for(clocked(), path)
            self.assertEqual((table.labels, list(table.onsets), source), (["trial", "trial"], [0.0, 5.0],
                                                                           "rec.paradigm.json"))
            (folder / "rec_events.csv").write_text("{},open\n{},closed\n".format(T0 + 1, T0 + 11), encoding="utf-8")
            table, source = ep.raw_events_for(clocked(), path)
            self.assertEqual((table.labels, source), (["open", "closed"], "rec_events.csv"))
            # unix times placed on the sample clock: well inside one sample (10 ms here)
            np.testing.assert_allclose(table.onsets, [1.0, 11.0], atol=1e-6)
            np.testing.assert_allclose(table.durations, [10.0, 9.0], atol=1e-6)
            ep.EventTable([4.0], None, ["named"]).save(ep.EventTable.path_for(path))
            self.assertEqual(ep.raw_events_for(clocked(), path)[1], "rec.events.json")
            stored = clocked(events=ep.EventTable([2.0], None, ["769"]))
            self.assertEqual(ep.raw_events_for(stored, path), (stored.events, "events stored in the file"))
            # bare marker-channel codes yield to any sidecar that names them, and are the last resort
            stored.events_are_raw_markers = True
            self.assertEqual(ep.raw_events_for(stored, path)[1], "rec.events.json")
            self.assertEqual(ep.raw_events_for(stored, folder / "other.csv"), (stored.events,
                                                                               "marker channel in the file"))
            self.assertEqual(ep.raw_events_for(clocked(), folder / "other.csv")[0], None)

    def test_markers_need_the_recordings_clock(self):
        with scratch() as folder:
            path = folder / "rec.csv"
            (folder / "rec_events.csv").write_text("{},open\n".format(T0 + 1), encoding="utf-8")
            unclocked = Recording(np.zeros((1, 2000)), 100.0, ["Cz"])
            self.assertIsNone(ep.raw_events_for(unclocked, path)[0])        # unix times mean nothing without one
            self.assertEqual(ep.raw_events_for(clocked(), path)[0].labels, ["open"])


if __name__ == "__main__":
    unittest.main()
