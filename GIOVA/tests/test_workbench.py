"""The Workbench page: sheet, wiring, settings, running, previews, files.

Run: .venv/Scripts/python -m unittest discover -s tests -p "test_*.py"
"""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / "tests" / ".tmp"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

from test_epochs import lateralised  # noqa: E402
from test_pipeline import save_fif  # noqa: E402
from test_trial_analysis import scratch  # noqa: E402
from eeg.pipeline import REGISTRY, Graph, templates  # noqa: E402
from eeg.ui.workbench import WorkbenchPage  # noqa: E402
from eeg.ui.workbench.page import FIRST_TEMPLATE  # noqa: E402
from eeg.ui.workbench.canvas import NodeItem, WireItem  # noqa: E402
from eeg.ui.workbench.inspector import from_text, to_text  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def fresh_sheet(test):
    """The Workbench starts on its first template, nothing unsaved; the remembered sheet is put back after `test`."""
    from eeg.ui.preferences import user_settings
    store = user_settings()
    keys = ("workbench/sheet", "workbench/file", "workbench/modified", "workbench/sheets", "workbench/current")
    saved = {k: store.value(k) for k in keys}
    for key in keys:
        store.remove(key)

    def restore():
        for key, value in saved.items():
            store.remove(key) if value is None else store.setValue(key, value)
    test.addCleanup(restore)


def wait_for_run(page, seconds=60):
    loop = QtCore.QEventLoop()
    page.ran.connect(loop.quit)
    QtCore.QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()
    page.ran.disconnect(loop.quit)


def mi_page(path):
    page = WorkbenchPage()
    graph = templates.csp_lda(str(path))
    graph.set("csp1", n_components=2)
    graph.set("cross_validate1", folds=4)
    page.set_graph(graph)
    return page


class TextFieldTests(unittest.TestCase):
    def test_lists_read_back_from_their_text(self):
        bands = [["mu", 8.0, 13.0], ["beta", 13.0, 30.0]]
        self.assertEqual(from_text("bands", to_text("bands", bands)), bands)
        self.assertEqual(from_text("numbers", "-1, 0.5", 2), [-1.0, 0.5])
        self.assertEqual(from_text("names", " C3, Cz ,C4,"), ["C3", "Cz", "C4"])
        with self.assertRaisesRegex(ValueError, "low-high"):
            from_text("bands", "mu 8 to 13")
        with self.assertRaisesRegex(ValueError, "below"):
            from_text("bands", "mu 13-8")
        with self.assertRaisesRegex(ValueError, "2 numbers"):
            from_text("numbers", "1", 2)


#: Scrolls a sheet at 150 % display scaling and counts the pixels that differ from the same view drawn afresh.
SCROLL_PROBE = """
import os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from pyqtgraph.Qt import QtWidgets
app = QtWidgets.QApplication([])
from eeg.ui.workbench.canvas import Canvas

def pixels(view):
    image = view.viewport().grab().toImage()
    return np.frombuffer(image.constBits(), np.uint8, image.sizeInBytes()).copy()

view = Canvas()
view.resize(600, 400)
view.show()
view.scale(1.13, 1.13)
app.processEvents()
for _ in range(25):
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + 37)
    app.processEvents()
scrolled = pixels(view)
view.resetCachedContent()
view.viewport().repaint()
app.processEvents()
print(int((scrolled != pixels(view)).sum()))
"""


class ScrollTests(unittest.TestCase):
    def test_the_sheet_scrolled_at_150_percent_is_drawn_as_afresh(self):
        # Windows at 150 %: a cached background was shifted by whole pixels while the squares sit on
        # fractional ones - stray lines and missing pieces of the sheet when it was panned
        import subprocess
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR="1.5")
        out = subprocess.run([sys.executable, "-c", SCROLL_PROBE, str(ROOT)], env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "0")


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.page = WorkbenchPage()
        self.page.set_graph(Graph("t"))
        self.canvas = self.page.canvas

    def test_palette_lists_every_block(self):
        listed, under = set(), {}

        def walk(item, heading):
            for k in range(item.childCount()):
                child = item.child(k)
                type_name = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if type_name:
                    listed.add(type_name)
                    under[type_name] = heading
                else:
                    walk(child, child.text(0))
        for i in range(self.page.palette.topLevelItemCount()):
            walk(self.page.palette.topLevelItem(i), self.page.palette.topLevelItem(i).text(0))
        self.assertEqual(listed, set(REGISTRY))
        # by what each block does: CSP is a spatial filter, not a model; a random forest is a classifier
        expected = {"band_power": "Feature extraction", "features": "Feature extraction",
                    "augment": "Data augmentation", "csp": "Spatial filters", "trcsp": "Spatial filters",
                    "fbcsp": "Spatial filters", "covariance": "Riemannian geometry", "tangent": "Riemannian geometry",
                    "standardize": "Scaling & selection", "select": "Scaling & selection",
                    "mibif": "Scaling & selection"}
        for type_name, heading in expected.items():
            self.assertEqual(under[type_name], heading, type_name)
        for model in ("lda", "svm", "logistic", "forest", "knn", "mdm", "nbpw"):
            self.assertEqual(under[model], "Classifiers", model)
        self.assertEqual(under["cross_validate"], "Evaluation")

    def test_double_click_adds_after_the_selected_block_already_wired(self):
        load = self.canvas.add_block("load")
        self.canvas.select(load)
        self.page.palette.chosen.emit("filter")
        filt = self.canvas.selected()
        self.assertEqual(self.page.graph.nodes[filt].type, "filter")
        self.assertEqual([(w.source, w.target) for w in self.page.graph.wires], [(load, filt)])
        self.assertGreater(self.canvas.cards[filt].pos().x(), self.canvas.cards[load].pos().x())

    def test_a_wire_of_the_wrong_kind_is_refused_with_a_reason(self):
        messages = []
        self.canvas.message.connect(messages.append)
        load, psd = self.canvas.add_block("load"), self.canvas.add_block("psd")
        out = self.canvas.cards[load].port("raw", True)
        inp = self.canvas.cards[psd].port("epochs", False)
        self.canvas.start_wire(out)
        self.assertFalse(inp.scale() > 1)                 # not lit: it would not accept
        self.assertFalse(self.canvas.finish_wire(inp))
        self.assertEqual(self.page.graph.wires, [])
        self.assertIn("needs epochs", messages[-1])

    def test_dragging_from_an_input_back_to_an_output_wires_too(self):
        load, filt = self.canvas.add_block("load"), self.canvas.add_block("filter")
        inp = self.canvas.cards[filt].port("raw", False)
        out = self.canvas.cards[load].port("raw", True)
        self.canvas.start_wire(inp)
        self.assertGreater(out.scale(), 1)                # lit: it fits
        self.assertTrue(self.canvas.finish_wire(out))
        self.assertEqual(len([i for i in self.canvas.scene().items() if isinstance(i, WireItem)]), 1)

    def test_delete_removes_blocks_with_their_wires(self):
        load, filt = self.canvas.add_block("load"), self.canvas.add_block("filter", after=None)
        self.canvas.wire(load, "raw", filt, "raw")
        self.canvas.select(filt)
        self.canvas.delete_selected()
        self.assertEqual(list(self.page.graph.nodes), [load])
        self.assertEqual(self.page.graph.wires, [])
        self.assertFalse([i for i in self.canvas.scene().items() if isinstance(i, WireItem)])

    def mouse(self, kind, point, buttons, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
        viewport = self.canvas.viewport()
        button = QtCore.Qt.MouseButton.NoButton if kind == QtCore.QEvent.Type.MouseMove else \
            QtCore.Qt.MouseButton.LeftButton
        event = QtGui.QMouseEvent(kind, QtCore.QPointF(point), QtCore.QPointF(viewport.mapToGlobal(point)),
                                  button, buttons, modifiers)
        QtWidgets.QApplication.sendEvent(viewport, event)

    def wheel(self, point, pixels=QtCore.QPoint(), angle=QtCore.QPoint(),
              modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
        viewport = self.canvas.viewport()
        event = QtGui.QWheelEvent(QtCore.QPointF(point), QtCore.QPointF(viewport.mapToGlobal(point)), pixels, angle,
                                  QtCore.Qt.MouseButton.NoButton, modifiers, QtCore.Qt.ScrollPhase.NoScrollPhase,
                                  False)
        QtWidgets.QApplication.sendEvent(viewport, event)

    def test_a_touchpad_moves_around_the_sheet_without_a_middle_button(self):
        self.page.resize(1200, 700)
        self.page.show()
        app.processEvents()
        load = self.canvas.add_block("load")
        self.canvas.select(load)
        h, v = self.canvas.horizontalScrollBar(), self.canvas.verticalScrollBar()
        empty = QtCore.QPoint(30, self.canvas.viewport().height() - 30)
        self.assertFalse(self.canvas.items_at(empty))
        left, none = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton
        # click and drag on empty sheet: the sheet follows, nothing is selected or moved
        start = (h.value(), v.value())
        self.mouse(QtCore.QEvent.Type.MouseButtonPress, empty, left)
        self.mouse(QtCore.QEvent.Type.MouseMove, empty + QtCore.QPoint(80, -50), left)
        self.mouse(QtCore.QEvent.Type.MouseButtonRelease, empty + QtCore.QPoint(80, -50), none)
        self.assertEqual((h.value(), v.value()), (start[0] - 80, start[1] + 50))
        self.assertEqual(self.canvas.selected(), load)                  # a drag is not a click
        # a click on empty sheet without moving still clears the selection
        self.mouse(QtCore.QEvent.Type.MouseButtonPress, empty, left)
        self.mouse(QtCore.QEvent.Type.MouseButtonRelease, empty, none)
        self.assertIsNone(self.canvas.selected())
        # two-finger scroll moves, sideways too; a mouse wheel notch moves 60 pixels
        start = (h.value(), v.value())
        self.wheel(empty, pixels=QtCore.QPoint(30, -20), angle=QtCore.QPoint(30, -20))
        self.assertEqual((h.value(), v.value()), (start[0] - 30, start[1] + 20))
        self.wheel(empty, angle=QtCore.QPoint(0, -120))
        self.assertEqual(v.value(), start[1] + 80)
        # pinch (Ctrl + wheel) and + / - zoom; each eases to its level, settle() goes there at once
        scale = self.canvas.transform().m11()
        self.wheel(empty, angle=QtCore.QPoint(0, 120), modifiers=QtCore.Qt.KeyboardModifier.ControlModifier)
        self.canvas.settle()
        self.assertAlmostEqual(self.canvas.transform().m11(), scale * 1.15)
        QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Minus)
        self.canvas.settle()
        self.assertAlmostEqual(self.canvas.transform().m11(), scale)
        # Shift + drag on empty sheet still draws a selection box
        card = self.canvas.cards[load]
        corner = self.canvas.mapFromScene(card.sceneBoundingRect().bottomRight()) + QtCore.QPoint(20, 20)
        top = self.canvas.mapFromScene(card.sceneBoundingRect().topLeft()) - QtCore.QPoint(20, 20)
        shift = QtCore.Qt.KeyboardModifier.ShiftModifier
        self.mouse(QtCore.QEvent.Type.MouseButtonPress, corner, left, shift)
        self.mouse(QtCore.QEvent.Type.MouseMove, top, left, shift)
        self.mouse(QtCore.QEvent.Type.MouseButtonRelease, top, none, shift)
        self.assertEqual(self.canvas.selected(), load)

    def test_the_sheet_is_free_to_move_and_zoom(self):
        """No edge and no scroll bars; zoom toward the mouse within 5-400 %, its level in the zoom bar; 0, 1, 2."""
        from eeg.ui.workbench.canvas import ZOOM_MAX, ZOOM_MIN
        self.page.resize(1200, 700)
        self.page.show()
        app.processEvents()
        canvas = self.canvas
        load = canvas.add_block("load")
        self.assertEqual(canvas.horizontalScrollBarPolicy(), QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bar = canvas.zoom_bar
        self.assertTrue(bar.isVisible())
        self.assertLess(bar.geometry().left(), canvas.width() / 2)               # lower left corner
        self.assertGreater(bar.geometry().top(), canvas.height() / 2)
        # far beyond where the old sheet ended (5000 px from its middle), and back
        centre = canvas.mapToScene(canvas.viewport().rect().center())
        for _ in range(40):
            canvas.scroll_by(900, 0)
        self.assertGreater(canvas.mapToScene(canvas.viewport().rect().center()).x(), centre.x() + 35000)
        for _ in range(40):
            canvas.scroll_by(-900, 0)
        self.assertAlmostEqual(canvas.mapToScene(canvas.viewport().rect().center()).x(), centre.x(), delta=1)
        # the point under the mouse stays put while zooming toward it
        point = QtCore.QPoint(300, 200)
        under = canvas.mapToScene(point)
        canvas.zoom(2.0, point)
        canvas.settle()
        self.assertLess((canvas.mapToScene(point) - under).manhattanLength(), 2)
        self.assertEqual(bar.level.text(), "200 %")
        canvas.zoom(1000)
        canvas.settle()
        self.assertAlmostEqual(canvas.zoom_level, ZOOM_MAX)                      # kept within its range
        canvas.zoom(1e-6)
        canvas.settle()
        self.assertAlmostEqual(canvas.zoom_level, ZOOM_MIN)
        QtTest.QTest.keyClick(canvas, QtCore.Qt.Key.Key_0)
        canvas.settle()
        self.assertAlmostEqual(canvas.zoom_level, 1.0)
        self.assertEqual(bar.level.text(), "100 %")
        # 1: the whole pipeline, even a long one, in view; 2: the selection
        far = canvas.add_block("filter", QtCore.QPointF(6000, 3000))
        QtTest.QTest.keyClick(canvas, QtCore.Qt.Key.Key_1)
        canvas.settle()
        view = canvas.mapToScene(canvas.viewport().rect()).boundingRect()
        for node in (load, far):
            self.assertTrue(view.contains(canvas.cards[node].sceneBoundingRect()), node)
        canvas.select(far)
        QtTest.QTest.keyClick(canvas, QtCore.Qt.Key.Key_2)
        canvas.settle()
        self.assertTrue(canvas.viewport().rect().contains(
            canvas.mapFromScene(canvas.cards[far].sceneBoundingRect()).boundingRect()))
        self.assertFalse(canvas.mapToScene(canvas.viewport().rect()).boundingRect().contains(
            canvas.cards[load].sceneBoundingRect()))

    def test_dragging_on_the_sheet_follows_the_setting(self):
        from eeg.ui.preferences import set_setting, user_settings
        self.addCleanup(lambda: user_settings().remove("workbench/drag"))
        self.page.resize(1200, 700)
        self.page.show()
        app.processEvents()
        load = self.canvas.add_block("load")
        card = self.canvas.cards[load]
        self.canvas.scene().clearSelection()
        h, v = self.canvas.horizontalScrollBar(), self.canvas.verticalScrollBar()
        left, none = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton
        corner = self.canvas.mapFromScene(card.sceneBoundingRect().bottomRight()) + QtCore.QPoint(30, 30)
        top = self.canvas.mapFromScene(card.sceneBoundingRect().topLeft()) - QtCore.QPoint(30, 30)
        set_setting("workbench/drag", "select")
        start = (h.value(), v.value())
        self.mouse(QtCore.QEvent.Type.MouseButtonPress, corner, left)
        self.mouse(QtCore.QEvent.Type.MouseMove, top, left)
        self.mouse(QtCore.QEvent.Type.MouseButtonRelease, top, none)
        self.assertEqual((h.value(), v.value()), start)                          # the sheet stayed
        self.assertEqual(self.canvas.selected(), load)                           # a box selected the block
        set_setting("workbench/drag", "pan")
        self.mouse(QtCore.QEvent.Type.MouseButtonPress, corner, left)
        self.mouse(QtCore.QEvent.Type.MouseMove, corner + QtCore.QPoint(40, 0), left)
        self.mouse(QtCore.QEvent.Type.MouseButtonRelease, corner + QtCore.QPoint(40, 0), none)
        self.assertEqual(h.value(), start[0] - 40)                               # the sheet moved

    def test_a_block_dragged_to_the_edge_moves_the_sheet_along(self):
        self.page.resize(1200, 700)
        self.page.show()
        app.processEvents()
        canvas = self.canvas
        node = canvas.add_block("load")
        canvas.scene().clearSelection()
        card = canvas.cards[node]
        grab = canvas.mapFromScene(card.mapToScene(QtCore.QPointF(60, 14)))
        edge = QtCore.QPoint(canvas.viewport().width() - 4, grab.y())
        h = canvas.horizontalScrollBar()
        left, none = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton
        self.mouse(QtCore.QEvent.Type.MouseButtonPress, grab, left)
        self.mouse(QtCore.QEvent.Type.MouseMove, edge, left)
        before, x_before = h.value(), card.pos().x()
        for _ in range(10):
            canvas._edge_step()
        self.assertGreater(h.value(), before)                                    # the sheet moved along
        self.assertGreater(card.pos().x(), x_before)                             # and the block with it
        self.mouse(QtCore.QEvent.Type.MouseButtonRelease, edge, none)
        self.assertFalse(canvas._edge.isActive())

    def test_a_sheet_opens_where_it_was_left(self):
        canvas = self.canvas
        self.page.resize(1200, 700)
        self.page.show()
        app.processEvents()
        canvas.add_block("load")
        canvas.zoom_to(1.5, smooth=False)
        canvas.scroll_by(120, -40)
        state = canvas.view_state()
        from eeg.ui.workbench.canvas import Canvas
        other = Canvas()
        other.resize(canvas.size())
        other.show()
        other.set_graph(canvas.graph.copy())
        other.saved_view = state
        other.first_view()
        self.addCleanup(other.close)
        self.assertEqual(other.view_state()[0], state[0])
        self.assertAlmostEqual(other.view_state()[1], state[1], delta=1)
        self.assertAlmostEqual(other.view_state()[2], state[2], delta=1)

    def test_a_block_is_dragged_by_the_mouse_even_when_wired(self):
        """Real mouse events through the view: the path a hand takes, not a call to setPos."""
        self.page.resize(1200, 700)
        self.page.show()
        load, filt = self.canvas.add_block("load"), self.canvas.add_block("filter")
        self.canvas.wire(load, "raw", filt, "raw")
        self.canvas.scene().clearSelection()
        card = self.canvas.cards[filt]
        start = self.canvas.mapFromScene(card.mapToScene(QtCore.QPointF(60, 14)))
        before = card.pos()
        left, none = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton
        self.mouse(QtCore.QEvent.Type.MouseButtonPress, start, left)
        self.assertIsNone(self.page.inspector.node_id)      # settings wait until the button is up
        for k in range(1, 6):
            self.mouse(QtCore.QEvent.Type.MouseMove, start + QtCore.QPoint(16 * k, 12 * k), left)
        self.mouse(QtCore.QEvent.Type.MouseButtonRelease, start + QtCore.QPoint(80, 60), none)
        moved = card.pos() - before
        self.assertGreater(moved.x(), 40)
        self.assertGreater(moved.y(), 30)
        position = self.page.graph.nodes[filt].position
        self.assertEqual((position[0] % 10, position[1] % 10), (0, 0))   # landed on the grid
        self.assertEqual(self.page.inspector.node_id, filt)
        wire = next(i for i in self.canvas.scene().items() if isinstance(i, WireItem))
        self.assertEqual(wire.path().elementAt(wire.path().elementCount() - 1).x,
                         card.port("raw", False).scenePos().x())      # the wire followed
        self.assertTrue(self.page.modified)

    def test_electrodes_are_picked_on_the_head_map(self):
        from unittest import mock
        from eeg.ui.headmap import ChannelSelectorDialog
        names = ["Fz", "C3", "Cz", "C4", "Pz"]
        node = self.canvas.add_block("channels")
        self.page.graph.set(node, channels=["C4", "Cz"])
        self.page.choices_for = lambda node_id, name: names
        self.page.inspector.choices_for = lambda node_id, name: names
        self.canvas.scene().clearSelection()            # selected when added: select again to show the new list
        self.canvas.select(node)
        # the settings shown now (an earlier block's, hidden, waits to be deleted)
        field = next(f for f in self.page.inspector.findChildren(QtWidgets.QWidget) if getattr(f, "param", None)
                     and f.param.name == "channels" and f.isVisibleTo(self.page.inspector))
        button = next(b for b in field.findChildren(QtWidgets.QPushButton) if b.text() == "Head…")
        with mock.patch.object(ChannelSelectorDialog, "ask", return_value=[1, 2, 3]) as ask:
            button.click()
        self.assertEqual(ask.call_args[0][2], [3, 2])                    # opened with C4 and Cz marked
        self.assertEqual(self.page.graph.params(node)["channels"], ["C4", "Cz", "C3"])   # kept order, C3 added

    def test_a_filter_shows_its_response_and_redraws_it_on_every_change(self):
        filt = self.canvas.add_block("filter")
        self.canvas.select(filt)
        response = self.page.inspector.response
        self.assertIsNotNone(response)
        freqs, gain = response.curve.getData()
        self.assertAlmostEqual(float(np.interp(20, freqs, gain)), 0.0, delta=0.5)
        self.page.inspector._edited("design", "elliptic")
        self.assertIn("elliptic", self.page.inspector.methods.text())
        self.page.inspector._edited("high", 200.0)         # above Nyquist at the default 250 Hz
        self.assertIn("Nyquist", response.note.text())

    def test_selecting_shows_the_blocks_settings_and_editing_changes_the_graph(self):
        filt = self.canvas.add_block("filter")
        self.canvas.select(filt)
        self.assertIn("Butterworth", self.page.inspector.methods.text())
        self.page.inspector._edited("high", 30.0)
        self.assertEqual(self.page.graph.params(filt)["high"], 30.0)
        self.assertIn("1–30 Hz", self.page.inspector.methods.text())


class RunTests(unittest.TestCase):
    def test_run_marks_blocks_shows_results_and_tracks_changes(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            page = mi_page(path)
            page.run()
            wait_for_run(page)
            items = page.canvas.cards
            self.assertEqual({n: i.state for n, i in items.items()}, {n: "done" for n in items})
            self.assertIn("accuracy", items["cross_validate1"].result)
            for node in page.graph.nodes:
                page.canvas.select(node)
                self.assertIsNotNone(page.inspector.result, node)
                window = page.inspector.open_larger(page.outputs()[node])
                self.assertGreater(window.findChildren(QtWidgets.QWidget).__len__(), 3, node)
                window.close()
            # what is chosen beside the settings opens in the larger window, comes back from it, and stays
            # when the block is shown again
            page.canvas.select("load1")
            small = page.inspector.result
            small.channel.setCurrentIndex(small.channel.count() - 1)
            small.span.setCurrentText("5 s")
            window = page.inspector.open_larger(page.outputs()["load1"])
            large = window.previews["out"] if "out" in window.previews else next(iter(window.previews.values()))
            self.assertEqual(large.channel.currentText(), small.channel.currentText())
            self.assertEqual(large.span.currentText(), "5 s")
            large.channel.setCurrentIndex(0)
            window.done(0)
            self.assertEqual(small.channel.currentIndex(), 0)
            page.canvas.select("epochs1")
            page.canvas.select("load1")
            self.assertEqual(page.inspector.result.channel.currentIndex(), 0)
            self.assertEqual(page.inspector.result.span.currentText(), "5 s")
            page.canvas.select("epochs1")
            box = page.inspector.result.findChild(QtWidgets.QComboBox)
            self.assertEqual(box.maxVisibleItems(), 15)
            self.assertIn("combobox-popup: 0", box.styleSheet())
            # a change upstream: that block and everything after it wait for the next run
            page.canvas.select("filter1")
            page.inspector._edited("low", 7.0)
            states = {n: i.state for n, i in items.items()}
            self.assertEqual(states["load1"], "done")
            self.assertEqual(states["csp1"], "done")
            for node in ("filter1", "epochs1", "cross_validate1"):
                self.assertEqual(states[node], "stale", node)
            page.run()
            wait_for_run(page)
            self.assertTrue(all(i.state == "done" for i in items.values()))
            self.assertEqual(sorted(page.choices_for("epochs1", "channels")), ["C3", "C4", "Cz"])
            self.assertEqual(page.choices_for("epochs1", "conditions"), ["left", "right"])

    def test_ica_and_a_riemannian_model_run_on_the_sheet(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=40), directory)
            page = WorkbenchPage()
            graph = templates.riemann(str(path))
            ica = graph.add("ica", {"eog_channels": [], "components": 2, "exclude": ["0"]})
            graph.connect("filter1", "raw", ica, "raw")
            graph.connect(ica, "raw", "epochs1", "raw")
            page.set_graph(graph)
            page.run()
            wait_for_run(page)
            self.assertEqual({n: c.state for n, c in page.canvas.cards.items()},
                             {n: "done" for n in page.canvas.cards})
            self.assertIn("1 removed", page.canvas.cards[ica].result)
            page.canvas.select(ica)
            self.assertIsNotNone(page.inspector.result)
            page.canvas.select("cross_validate1")
            texts = " ".join(label.text() for label in page.inspector.findChildren(QtWidgets.QLabel))
            self.assertIn("Tuning chose C = ", texts)
            self.assertIn("balanced accuracy", texts)

    def test_a_failing_block_turns_red_with_its_reason(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=10), directory)
            page = mi_page(path)
            page.graph.set("epochs1", conditions=["up"])
            page.run()
            wait_for_run(page)
            self.assertEqual(page.canvas.cards["epochs1"].state, "error")
            self.assertIn("left, right", page.canvas.cards["epochs1"].message)
            self.assertEqual(page.canvas.cards["load1"].state, "done")

    def test_a_recording_from_the_projects_pane_runs_in_the_pipeline(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=20), directory)
            page = mi_page("")
            self.assertTrue(page.open_recording(path))
            wait_for_run(page)
            self.assertEqual(page.graph.params("load1")["path"], str(path))
            self.assertEqual(page.canvas.cards["cross_validate1"].state, "done")

    def test_incomplete_blocks_do_not_stop_the_rest(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=10), directory)
            page = mi_page(path)
            page.canvas.add_block("psd")              # nothing feeds it
            page.run()
            wait_for_run(page)
            self.assertEqual(page.canvas.cards["cross_validate1"].state, "done")
            self.assertEqual(page.canvas.cards["psd1"].state, "")

    def test_a_permutation_test_is_one_click_under_the_result(self):
        with scratch() as directory:
            path = save_fif(lateralised(trials=20), directory)
            page = mi_page(path)
            page.run()
            wait_for_run(page)
            page.canvas.select("cross_validate1")
            texts = " ".join(label.text() for label in page.inspector.findChildren(QtWidgets.QLabel))
            self.assertIn("so it is approximate", texts)                     # the limit alone: said so
            button = next(b for b in page.inspector.findChildren(QtWidgets.QToolButton)
                          if b.text().startswith("Permutation test"))
            self.assertEqual(button.text(), "Permutation test (200 shuffles)")
            self.assertEqual([a.text() for a in button.menu().actions()],
                             ["100 shuffles", "200 shuffles", "500 shuffles", "1000 shuffles", "5000 shuffles"])
            with mock.patch.object(page, "run") as run:
                button.click()                                                # a click: 200
            run.assert_called_once_with(["cross_validate1"])
            self.assertEqual(page.graph.params("cross_validate1")["permutations"], 200)
            button.menu().actions()[0].trigger()                              # 100, from the menu, and run
            self.assertEqual(page.graph.params("cross_validate1")["permutations"], 100)
            self.assertEqual(page.inspector.fields["permutations"].widget.value(), 100)   # kept in Options
            wait_for_run(page)
            scores = page.outputs()["cross_validate1"]["scores"]
            self.assertEqual(scores.permutations, 100)
            page.canvas.select("cross_validate1")
            self.assertFalse(any(b.text().startswith("Permutation test")
                                 for b in page.inspector.findChildren(QtWidgets.QToolButton)))
            texts = " ".join(label.text() for label in page.inspector.findChildren(QtWidgets.QLabel))
            self.assertIn("Permutation test: p = ", texts)
            self.assertNotIn("so it is approximate", texts)


class FileTests(unittest.TestCase):
    def setUp(self):
        fresh_sheet(self)          # these replace the sheet: an unsaved one left by another test would be asked about

    def test_save_open_and_the_sheet_survives_a_restart(self):
        page = WorkbenchPage()
        graph = templates.band_power("rec.edf")
        page.set_graph(graph)
        with scratch() as directory:
            path = directory / "mine.pipeline.json"
            self.assertTrue(page.save_pipeline(str(path)))
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["name"], "Band power per condition")
            page.canvas.add_block("window")
            self.assertTrue(page.modified)
            again = WorkbenchPage()                    # as at the next start
            self.assertIn("window1", again.graph.nodes)
            self.assertTrue(again.modified)            # still unsaved after the restart
            discard = QtWidgets.QMessageBox.StandardButton.Discard
            with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=discard) as ask:
                self.assertTrue(again.open_pipeline(str(path)))
            self.assertEqual(ask.call_count, 1)
            self.assertNotIn("window1", again.graph.nodes)
            self.assertFalse(again.modified)
        self.assertEqual(again.graph.to_dict()["wires"], graph.to_dict()["wires"])

    def test_plus_and_new_offer_an_empty_sheet_and_the_two_pipelines_saved_last(self):
        import time
        page = WorkbenchPage()
        with scratch() as directory:
            for k, name in enumerate(("oldest", "middle", "newest")):
                graph = templates.band_power("rec.edf")
                graph.name = name
                path = directory / "{}.pipeline.json".format(name)
                graph.save(str(path))
                os.utime(path, (time.time() - 100 + k, time.time() - 100 + k))
            page.library.project_folders = lambda: [str(directory)]
            self.assertEqual([p.name for p in page.last_saved()], ["newest.pipeline.json", "middle.pipeline.json"])
            plus = page.tabs.cornerWidget(QtCore.Qt.Corner.TopRightCorner).menu()
            plus.aboutToShow.emit()
            entries = [a.text() for a in plus.actions() if not a.isSeparator()]
            # templates and the literature are in the Library, not here
            self.assertEqual(entries, ["Empty sheet", "newest", "middle"])
            before = len(page.sheets)
            next(a for a in plus.actions() if a.text() == "newest").trigger()
            self.assertEqual((len(page.sheets), page.graph.name), (before + 1, "newest"))     # in a new tab
            for sheet in list(page.sheets):
                sheet.modified = False

    def test_the_library_lists_mine_templates_and_literature_and_opens_in_a_new_tab(self):
        from eeg.pipeline import literature
        from eeg.ui.workbench import library as lib
        page = WorkbenchPage()
        with scratch() as directory:
            (directory / "subject01").mkdir()
            path = directory / "subject01" / "mine.pipeline.json"
            graph = templates.band_power("rec.edf")
            graph.name = "My band power"
            graph.save(str(path))
            page.library.project_folders = lambda: [str(directory)]
            page.library.refresh()
            entries = {item.text(0): item for item in page.library.entries()}
            self.assertIn("My band power", entries)
            self.assertIn(entries["My band power"].data(0, lib.KIND_ROLE), lib.MINE)
            self.assertTrue(set(templates.TEMPLATES) <= set(entries))
            self.assertTrue(set(literature.LIBRARY) <= set(entries))
            # the search keeps only what has every word typed
            page.library.search.setText("band my")
            shown = [i.text(0) for i in page.library.entries() if not i.isHidden()]
            self.assertEqual(shown, ["My band power"])
            page.library.search.clear()
            # opened in a tab of its own; the sheet in front is not replaced, and twice is still one tab
            before = len(page.sheets)
            first = page.sheet
            page.library.choose(entries["My band power"])
            self.assertEqual(len(page.sheets), before + 1)
            self.assertIn(first, page.sheets)
            self.assertEqual(page.graph.name, "My band power")
            page.library.choose(entries["My band power"])
            self.assertEqual(len(page.sheets), before + 1)
            page.library.choose(entries[FIRST_TEMPLATE])
            self.assertEqual(len(page.sheets), before + 2)
            for sheet in list(page.sheets):
                sheet.modified = False

    def test_a_new_template_keeps_the_recording_in_use(self):
        page = WorkbenchPage()
        page.set_graph(templates.band_power("C:/data/s01.mat"))
        page.new_pipeline("ERD / ERS")
        self.assertEqual(page.graph.params("load1")["path"], "C:/data/s01.mat")


class WindowTests(unittest.TestCase):
    def test_main_window_has_the_workbench_with_its_help(self):
        from eeg.ui.app import MainWindow
        from eeg.ui.help_content import TAB_TOPICS, TOPICS
        window = MainWindow()
        names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        self.assertEqual(names, ["Home", "Monitor", "Workbench", "Games"])
        self.assertIn(TAB_TOPICS["Workbench"], dict(TOPICS))
        self.assertTrue(any(isinstance(i, NodeItem) for i in window.workbench.canvas.scene().items()))
        window.close()


if __name__ == "__main__":
    unittest.main()
