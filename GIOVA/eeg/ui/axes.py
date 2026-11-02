"""Tick labels that never overlap, on every plot in the program.

pyqtgraph labels a log axis's minor ticks (2, 3, ... 9 between decades) whether or not there is room, so on a
small plot "7 8 9 10" or "0.8 0.9 1" were printed over each other (found filming tutorial 3). Here every axis
keeps a label only if it does not touch one already placed. On a log axis the decades are placed first, then
5s, then 2s, then the rest (pyqtgraph lists them left to right, so "0.9" could push "1" out); on a linear axis
the labels keep pyqtgraph's order.

Imported once by `theme`, so it holds for every plot, the Workbench's and the Monitor's alike.
tests/test_plot_labels.py checks every result view for overlapping labels.
"""
import pyqtgraph as pg

#: pixels kept free between two labels
GAP = 2.0

_drawn_specs = pg.AxisItem.generateDrawSpecs


def apart(rects, gap=GAP):
    """Indices of `rects` kept, first come first kept, none touching another kept one (with `gap` around)."""
    kept = []
    for i, rect in enumerate(rects):
        wide = rect.adjusted(-gap, -gap, gap, gap)
        if not any(wide.intersects(rects[k]) for k in kept):
            kept.append(i)
    return kept


SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def label_value(text):
    """The number a tick label shows ("0.5", "10³", "2×10⁻¹", "−3"), or None."""
    text = str(text).replace("−", "-").replace(" ", "")
    try:
        if "10" in text and text.translate(SUPERSCRIPT) != text:
            mantissa, _, power = text.partition("×")
            if not power:
                mantissa, power = "1", mantissa
            return float(mantissa) * 10 ** float(power[2:].translate(SUPERSCRIPT))
        return float(text)
    except ValueError:
        return None


def roundness(value):
    """0 for a decade (1, 10, 0.1), 1 for 5s, 2 for 2s, 3 for anything else: the order log labels are placed in."""
    import math
    if value is None or value <= 0:
        return 3
    mantissa = round(value / 10 ** math.floor(math.log10(value) + 1e-12), 6)
    return {1.0: 0, 5.0: 1, 2.0: 2}.get(mantissa, 3)


def _readable(self, p):
    specs = _drawn_specs(self, p)
    if not specs:
        return specs
    axis, ticks, texts = specs
    order = list(range(len(texts)))
    if self.logMode:
        order.sort(key=lambda i: roundness(label_value(texts[i][2])))
    kept = apart([texts[i][0] for i in order])
    return axis, ticks, [texts[i] for i in sorted(order[k] for k in kept)]


if getattr(pg.AxisItem.generateDrawSpecs, "__name__", "") != "_readable":
    pg.AxisItem.generateDrawSpecs = _readable
