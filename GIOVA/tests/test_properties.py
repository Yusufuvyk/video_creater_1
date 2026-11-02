"""Round trips checked on many generated inputs: property-based testing (Hypothesis; MacIver, Hatfield-Dodds et al.,
2019, Journal of Open Source Software 4(43), 1891).

What a person types into GIOVA, and every pipeline file, is read back by a parser. An example test checks a few
inputs chosen by hand; here Hypothesis generates hundreds inside the domain each format allows, and shrinks any
failure to its smallest case:

    trial numbers  "1-3, 5, T1"         trial_numbers(trial_text(ids)) gives the ids back, numbers sorted, once each
    sequence       "Rest:30; Task:2.5"   parse_sequence reads back the phases the text was written from
    MAT rows       "1-32, 65"            parse_rows gives the rows typed, in order, or None when they are all rows
    text fields    names, numbers, bands, ranges, ratios and trials: the value a field shows reads back as itself
    pipelines      any chain of blocks the Workbench can wire is the same after saving and opening it

The examples are drawn from a fixed seed (`derandomize`), so a run is repeatable and CI never fails by chance.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))

from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

from eeg.core.analysis_profiles import parse_sequence  # noqa: E402
from eeg.formats.matfile import MatError, parse_rows  # noqa: E402
from eeg.pipeline import REGISTRY, Graph, block  # noqa: E402
from eeg.pipeline.compute import trial_numbers, trial_text  # noqa: E402
from eeg.ui.workbench.inspector import from_text, to_text  # noqa: E402

settings.register_profile("giova", derandomize=True, deadline=None, database=None, max_examples=200,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("giova")

#: what a field holds besides its separators: letters, digits, punctuation and spaces, any script
TEXT = st.characters(categories=("L", "N", "P", "Zs"))


def words(exclude="", min_size=1, max_size=12, spaces=True):
    """Text without the characters a format uses to separate values, and without spaces at its ends."""
    alphabet = st.characters(categories=("L", "N", "P", "Zs") if spaces else ("L", "N", "P"),
                             exclude_characters=exclude)
    return st.text(alphabet, min_size=min_size, max_size=max_size).map(str.strip).filter(bool)


#: seconds, hertz or any other setting, from the very small to hours of recording
NUMBERS = st.floats(min_value=-1e7, max_value=1e7, allow_nan=False, allow_infinity=False)
POSITIVE = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)


def close(test, read, written):
    """What a field reads back is what it showed, to 12 significant digits: far below a sample at any rate."""
    test.assertEqual(len(read), len(written))
    for a, b in zip(read, written):
        test.assertLessEqual(abs(a - b), 1e-11 * max(abs(b), 1e-300))


class TrialNumbersTests(unittest.TestCase):
    @given(st.lists(st.integers(1, 5000), max_size=80))
    def test_numbers_come_back_sorted_once_each(self, numbers):
        ids = [str(n) for n in numbers]
        self.assertEqual(trial_numbers(trial_text(ids)), [str(n) for n in sorted(set(numbers))])

    # a trial typed by hand carries a letter (T1, rest-2): an id like "1-3" would read as a range
    @given(st.lists(st.one_of(st.integers(1, 500).map(str), words(",").filter(lambda w: any(c.isalpha() for c in w))),
                    max_size=40))
    def test_ids_that_are_not_numbers_follow_as_they_are(self, ids):
        numeric = sorted({int(i) for i in ids if i.isdigit()})
        others = list(dict.fromkeys(i for i in ids if not i.isdigit()))
        self.assertEqual(trial_numbers(trial_text(ids), known=ids), [str(n) for n in numeric] + others)

    @given(st.integers(1, 500), st.integers(1, 500))
    def test_a_range_written_backwards_is_refused(self, a, b):
        if a > b:
            with self.assertRaisesRegex(ValueError, "must come before"):
                trial_numbers("{}-{}".format(a, b))
        else:
            self.assertEqual(trial_numbers("{}-{}".format(a, b)), [str(n) for n in range(a, b + 1)])


class SequenceTests(unittest.TestCase):
    # a condition may hold a colon ("Task: left"): the duration is what follows the last one
    @given(st.lists(st.tuples(words(";"), st.floats(1e-3, 1e5)), min_size=1, max_size=8), st.booleans())
    def test_the_phases_written_are_the_phases_read(self, phases, comma):
        text = "; ".join("{}:{}".format(name, repr(seconds).replace(".", "," if comma else "."))
                         for name, seconds in phases)
        self.assertEqual(parse_sequence(text), phases)

    @given(words(";:"), st.floats(max_value=0.0) | st.just(float("nan")) | st.just(float("inf")))
    def test_a_phase_needs_a_positive_duration(self, name, seconds):
        with self.assertRaises(ValueError):
            parse_sequence("{}:{}".format(name, seconds))


@st.composite
def rows_typed(draw):
    """(n_rows, text, the 0-based rows it names in the order typed)."""
    n_rows = draw(st.integers(1, 300))
    parts, rows = [], []
    for _ in range(draw(st.integers(1, 6))):
        first = draw(st.integers(1, n_rows))
        last = draw(st.integers(first, n_rows))
        parts.append(str(first) if first == last else "{}-{}".format(first, last))
        rows += range(first - 1, last)
    return n_rows, draw(st.sampled_from([",", ", ", " , "])).join(parts), list(dict.fromkeys(rows))


class MatRowsTests(unittest.TestCase):
    @given(rows_typed())
    def test_the_rows_typed_in_their_order(self, typed):
        n_rows, text, rows = typed
        self.assertEqual(parse_rows(text, n_rows), None if len(rows) == n_rows else rows)

    @given(rows_typed(), st.integers(1, 50))
    def test_a_row_past_the_last_is_refused_not_dropped(self, typed, past):
        n_rows, text, _ = typed
        with self.assertRaises(MatError):
            parse_rows("{},{}".format(text, n_rows + past), n_rows)


def edges(low, high, minimum=0.0):
    """(low, high) with low < high, apart by more than the 12 digits a field shows."""
    low, high = sorted((abs(low) + minimum, abs(high) + minimum))
    return low, max(high, low + max(abs(low), 1.0) * 1e-9)


class TextFieldTests(unittest.TestCase):
    @given(st.lists(words(","), max_size=12))
    def test_names(self, names):
        self.assertEqual(from_text("names", to_text("names", names)), names)

    @given(st.lists(NUMBERS, max_size=12))
    def test_numbers(self, numbers):
        shown = to_text("numbers", numbers)
        close(self, from_text("numbers", shown), numbers)
        self.assertEqual(to_text("numbers", from_text("numbers", shown)), shown)      # a second trip changes nothing

    # a band's name is one word: "mu 8-13"
    @given(st.lists(st.tuples(words(",", spaces=False), POSITIVE, POSITIVE), min_size=1, max_size=8))
    def test_bands(self, bands):
        bands = [[name, *edges(low, high)] for name, low, high in bands]
        read = from_text("bands", to_text("bands", bands))
        self.assertEqual([b[0] for b in read], [b[0] for b in bands])
        close(self, [v for b in read for v in b[1:]], [v for b in bands for v in b[1:]])

    @given(st.lists(st.tuples(POSITIVE, POSITIVE), min_size=1, max_size=8))
    def test_ranges(self, ranges):
        ranges = [list(edges(low, high)) for low, high in ranges]
        close(self, [v for r in from_text("ranges", to_text("ranges", ranges)) for v in r],
              [v for r in ranges for v in r])

    @given(st.lists(st.tuples(words(",=/"), words(",=/"), words(",=/")), max_size=6))
    def test_ratios(self, ratios):
        ratios = [list(r) for r in ratios]
        self.assertEqual(from_text("ratios", to_text("ratios", ratios)), ratios)

    @given(st.lists(st.tuples(words("|;"), words("|;"), NUMBERS, NUMBERS, st.booleans()), max_size=10))
    def test_trials(self, trials):
        rows = [list(t) for t in trials]
        read = from_text("trials", to_text("trials", rows))
        self.assertEqual([r[:2] + r[4:] for r in read], [r[:2] + r[4:] for r in rows])
        close(self, [v for r in read for v in r[2:4]], [v for r in rows for v in r[2:4]])

    def test_a_long_recording_keeps_its_milliseconds(self):
        # 6 significant digits wrote 3600.125 s as 3600.12: a trial edited as text moved by 5 ms
        rows = [["1", "Eyes open", 3600.125, 3630.125, True], ["2", "Eyes closed", 12345.678, 12375.678, False]]
        self.assertEqual(from_text("trials", to_text("trials", rows)), rows)
        self.assertEqual(to_text("trials", rows[:1]), "1 | Eyes open | 3600.125-3630.125")
        self.assertEqual(to_text("numbers", [-1.0, 0.5, 30.0]), "-1, 0.5, 30")     # the usual values look the same


def wired_chain(draw):
    """[(block type, params)]: a source, then blocks each of which takes what the one before gives."""
    starts = sorted(t for t, b in REGISTRY.items() if all(p.optional for p in b.inputs))
    chain = [draw(st.sampled_from(starts))]
    for _ in range(draw(st.integers(0, 6))):
        out = block(chain[-1]).outputs[0]
        following = sorted(t for t, b in REGISTRY.items() if any(out.types[0] in p.types for p in b.inputs))
        if not following:
            break
        chain.append(draw(st.sampled_from(following)))
    steps = []
    for type_name in chain:
        params = {p.name: draw(st.sampled_from(p.choices)) for p in block(type_name).params
                  if p.choices and draw(st.booleans())}
        steps.append((type_name, params))
    return steps


class PipelineFileTests(unittest.TestCase):
    @given(st.data())
    @settings(max_examples=100)
    def test_any_chain_the_workbench_wires_is_the_same_after_saving(self, data):
        graph = Graph(data.draw(words(min_size=0) | st.just("Pipeline")), data.draw(st.text(TEXT, max_size=40)))
        ids = graph.chain(*wired_chain(data.draw))
        for node_id in ids:
            graph.nodes[node_id].position = (data.draw(NUMBERS), data.draw(NUMBERS))
            graph.nodes[node_id].label = data.draw(st.text(TEXT, max_size=20))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pipeline.json"
            graph.save(path)
            again = Graph.load(path)
        # what a file holds: tuples are written as lists, so the two are compared as the JSON they make
        self.assertEqual(json.loads(json.dumps(again.to_dict())), json.loads(json.dumps(graph.to_dict())))
        self.assertEqual(json.dumps(again.copy().to_dict()), json.dumps(again.to_dict()))


if __name__ == "__main__":
    unittest.main()
