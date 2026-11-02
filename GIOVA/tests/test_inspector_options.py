"""A block shows only the settings that mean something with its others (an STFT window under STFT, not under
Morlet); what is hidden keeps its value. And a message that stands in for a scalp map fits its view.

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

from pyqtgraph.Qt import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from eeg.core import epochs as ep  # noqa: E402
from eeg.pipeline import REGISTRY, Graph  # noqa: E402
from eeg.ui.topomap import TopoGrid  # noqa: E402
from eeg.ui.workbench.inspector import Inspector  # noqa: E402


class WhenTests(unittest.TestCase):
    def test_every_condition_names_a_setting_of_its_block_and_values_it_can_take(self):
        for type_name, kind in REGISTRY.items():
            params = {p.name: p for p in kind.params}
            for param in kind.params:
                for name, values in (param.when or ()) + (param.unless or ()):
                    self.assertIn(name, params, "{}.{}".format(type_name, param.name))
                    choices = params[name].choices
                    if choices is not None:
                        self.assertLessEqual(set(values), set(choices), "{}.{}".format(type_name, param.name))
                    elif isinstance(params[name].default, bool):
                        self.assertLessEqual(set(values), {True, False}, "{}.{}".format(type_name, param.name))
                # with the block's own defaults, a setting that does not apply is behind Options or conditional
                self.assertIsInstance(param.applies(kind.defaults()), bool)

    def test_a_hidden_setting_keeps_its_value(self):
        graph = Graph()
        node = graph.add("erd", {"window": 0.25})
        graph.set(node, method=ep.WAVELET)
        self.assertEqual(graph.params(node)["window"], 0.25)          # the pipeline still holds it
        graph.set(node, method=ep.STFT)
        self.assertEqual(graph.params(node)["window"], 0.25)


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.inspector = Inspector()
        self.inspector.resize(420, 900)
        self.inspector.show()
        self.addCleanup(self.inspector.close)

    def shown(self):
        """Names of the settings whose rows are visible (Options counts as open)."""
        inspector = self.inspector
        out = set()
        for name, field in inspector.fields.items():
            form = inspector.advanced_form if field.param.advanced else inspector.basic_form
            if form.isRowVisible(field):
                out.add(name)
        return out

    def test_time_frequency_shows_the_settings_of_its_method_only(self):
        node = self.graph.add("erd")
        self.inspector.show_node(self.graph, node)
        # the STFT's window and overlap stay with every method: the head map and the statistics use them
        method_settings = {"window", "overlap", "width", "n_cycles", "wavelet"}
        self.assertEqual(self.shown() & method_settings, {"window", "overlap"})
        self.inspector.set_value(node, "method", ep.WAVELET)
        self.assertEqual(self.shown() & method_settings, {"window", "overlap", "width", "n_cycles"})  # per Hz
        self.inspector.set_value(node, "width", ep.FIXED)
        self.assertEqual(self.shown() & method_settings, {"window", "overlap", "width", "wavelet"})  # cmorB-C
        self.assertIn("baseline", self.shown())
        self.assertNotIn("reference", self.shown())
        self.inspector.set_value(node, "baseline_mode", ep.CONDITION)
        self.assertNotIn("baseline", self.shown())
        self.assertIn("reference", self.shown())

    def test_filter_shows_what_its_design_reads(self):
        node = self.graph.add("filter")
        self.inspector.show_node(self.graph, node)
        design = {"order", "ripple", "attenuation", "transition", "phase"}
        self.assertEqual(self.shown() & design, {"order", "phase"})                  # Butterworth
        self.inspector.set_value(node, "design", "elliptic")
        self.assertEqual(self.shown() & design, {"order", "ripple", "attenuation", "phase"})
        self.inspector.set_value(node, "design", "fir")
        self.assertEqual(self.shown() & design, {"transition", "phase"})
        self.inspector.set_value(node, "design", "ideal")
        self.assertEqual(self.shown() & design, set())                               # zero-phase only

    def test_options_go_when_nothing_behind_them_applies(self):
        node = self.graph.add("logistic")
        self.inspector.show_node(self.graph, node)
        app.processEvents()                                  # the new settings are shown on the next turn
        self.assertFalse(self.inspector.options_toggle.isVisibleTo(self.inspector))  # untuned: nothing behind
        self.assertIn("C", self.shown())
        self.inspector.set_value(node, "tune", True)
        self.assertTrue(self.inspector.options_toggle.isVisibleTo(self.inspector))
        self.assertNotIn("C", self.shown())
        self.assertIn("C_values", self.shown())


class ScalpMapNoteTests(unittest.TestCase):
    def test_the_message_instead_of_a_map_fits_a_narrow_view(self):
        grid = TopoGrid()
        grid.resize(240, 220)
        grid.show()
        self.addCleanup(grid.close)
        self.assertFalse(grid.draw(["C3"], [("A", [1.0])], "x"))
        app.processEvents()
        self.assertIn("at least three", grid.note)
        note = grid._note
        self.assertTrue(grid.viewport().rect().contains(note.geometry()))
        self.assertLessEqual(note.heightForWidth(note.width()), note.height())     # wrapped, every line in view
        self.assertTrue(grid.draw(["C3", "Cz", "C4", "Pz"], [("A", [1.0, 2.0, 3.0, 4.0])], "x"))
        self.assertEqual(grid.note, "")


if __name__ == "__main__":
    unittest.main()
