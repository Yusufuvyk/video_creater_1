"""Reports: what a pipeline did and found, as one HTML file or a PDF.

A report holds the pipeline as drawn on the sheet, the methods paragraph
built from every block's own sentence, each block's result with the same
figure the Workbench shows beside it, the software versions and the papers
the methods rest on. The HTML carries its pictures inside it, so it is one
file to send; the PDF is the same text laid out on A4.

A comparison of pipelines over recordings has its own report: per-recording
accuracy, the summary, the paired tests and each pipeline's methods.
"""
import base64
from datetime import datetime
import html
import json
from pathlib import Path
import platform

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ... import NAME, VERSION
from ...pipeline import block
from ...pipeline.batch import score_nodes
from ...pipeline import provenance
from ...pipeline.data import Contrast
from ...pipeline.learn import APPROXIMATE, Scores
from .previews import CHANNEL_COLORS, picture, preview, result_line
from ..theme import for_paper, paper, plot_background

#: Width of a figure in the report, in screen pixels (each drawn at twice that for sharpness).
FIGURE_WIDTH = 940
#: The formats a report is saved in, as a file dialog lists them; the first is the default.
FORMATS = "PDF (*.pdf);;Word (*.docx);;Web page (*.html);;Figures only, PNG (*.png)"

#: The papers a block's method rests on.
REFERENCES = {
    "csp": "Ramoser H, Müller-Gerking J, Pfurtscheller G (2000). Optimal spatial filtering of single trial EEG "
           "during imagined hand movement. IEEE Trans Rehabil Eng 8(4):441-446.",
    "trcsp": "Lotte F, Guan C (2011). Regularizing common spatial patterns to improve BCI designs: unified "
             "theory and new algorithms. IEEE Trans Biomed Eng 58(2):355-362.",
    "fbcsp": "Ang KK, Chin ZY, Wang C, Guan C, Zhang H (2012). Filter bank common spatial pattern algorithm on "
             "BCI Competition IV datasets 2a and 2b. Front Neurosci 6:39.",
    "lda": "Ledoit O, Wolf M (2004). A well-conditioned estimator for large-dimensional covariance matrices. "
           "J Multivar Anal 88(2):365-411.",
    "tangent": "Barachant A, Bonnet S, Congedo M, Jutten C (2012). Multiclass brain-computer interface "
               "classification by Riemannian geometry. IEEE Trans Biomed Eng 59(4):920-928.",
    "forest": "Breiman L (2001). Random forests. Mach Learn 45(1):5-32.",
    "sklearn": "Pedregosa F et al. (2011). Scikit-learn: machine learning in Python. J Mach Learn Res "
               "12:2825-2830.",
    "cross_validate": "Müller-Putz GR, Scherer R, Brunner C, Leeb R, Pfurtscheller G (2008). Better than random? "
                      "A closer look on BCI results. Int J Bioelectromagn 10(1):52-55. Combrisson E, Jerbi K "
                      "(2015). Exceeding chance level by chance: the caveat of theoretical chance levels in brain "
                      "signal classification and statistical assessment of decoding accuracy. J Neurosci Methods "
                      "250:126-136. Noirhomme Q, Lesenfants D, Gomez F, Soddu A, Schrouff J, Garraux G, Luxen A, "
                      "Phillips C, Laureys S (2014). Biased binomial assessment of cross-validated estimation of "
                      "classification accuracies illustrated in diagnosis predictions. NeuroImage Clin 4:687-694. "
                      "Cohen J (1960). A coefficient of agreement for nominal scales. Educ Psychol Meas 20(1):37-46.",
    "permutation": "Ojala M, Garriga GC (2010). Permutation tests for studying classifier performance. J Mach "
                   "Learn Res 11:1833-1863.",
    "fir": "Widmann A, Schröger E, Maess B (2015). Digital filter design for electrophysiological data - a "
           "practical approach. J Neurosci Methods 250:34-46.",
    "mne": "Gramfort A et al. (2013). MEG and EEG data analysis with MNE-Python. Front Neurosci 7:267.",
    "fastica": "Hyvärinen A (1999). Fast and robust fixed-point algorithms for independent component analysis. "
               "IEEE Trans Neural Netw 10(3):626-634.",
    "erd": "Pfurtscheller G, Lopes da Silva FH (1999). Event-related EEG/MEG synchronization and "
           "desynchronization: basic principles. Clin Neurophysiol 110(11):1842-1857.",
    "laplacian": "Hjorth B (1975). An on-line transformation of EEG scalp potentials into orthogonal source "
                 "derivations. Electroencephalogr Clin Neurophysiol 39(5):526-530.",
    "psd": "Welch P (1967). The use of fast Fourier transform for the estimation of power spectra. IEEE Trans "
           "Audio Electroacoust 15(2):70-73.",
}


def references(graphs):
    """The references the blocks of `graphs` call for, in a fixed order, without repeats."""
    keys = set()
    for graph in graphs:
        for node_id, node in graph.nodes.items():
            params = graph.params(node_id)
            keys.add(node.type)
            if node.type in ("tangent", "mdm"):
                keys.add("tangent")
            if node.type in ("filter_bank", "mibif", "nbpw"):
                keys.add("fbcsp")
            if node.type in ("svm", "logistic", "knn", "forest", "select"):
                keys.add("sklearn")
            if node.type == "filter" and params.get("design") == "fir":
                keys |= {"fir", "mne"}
            if node.type == "ica":
                keys.add("mne")
                if params.get("method") == "fastica":
                    keys.add("fastica")
            if node.type == "reference" and params.get("mode") == "laplacian":
                keys.add("laplacian")
            if node.type == "cross_validate" and params.get("permutations"):
                keys.add("permutation")
            if node.type == "lda" and params.get("shrinkage") != "auto":
                keys.discard("lda")
    return [REFERENCES[k] for k in REFERENCES if k in keys]


def versions():
    import mne
    import scipy
    import sklearn
    rows = [("GIOVA", "v3"), ("Python", platform.python_version()), ("NumPy", np.__version__),
            ("SciPy", scipy.__version__), ("MNE-Python", mne.__version__), ("scikit-learn", sklearn.__version__)]
    try:
        import pyriemann
        rows.append(("pyRiemann", pyriemann.__version__))
    except ImportError:
        pass
    return rows


def prepare(widget, width=FIGURE_WIDTH):
    """A widget made ready to be pictured, off screen, `width` pixels wide: on white, a preview without its
    controls. It can be pictured again after a chooser changes, as Save every trial does."""
    for_paper(widget)
    if hasattr(widget, "for_report"):
        widget.for_report()
    widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(width, max(widget.sizeHint().height(), 120))
    widget.show()
    QtWidgets.QApplication.processEvents()
    widget.resize(width, max(widget.sizeHint().height(), 120))
    QtWidgets.QApplication.processEvents()
    return widget


def render(widget, width=FIGURE_WIDTH):
    """A widget drawn into an image, off screen, `width` pixels wide at PICTURE_SCALE times the
    pixels (sharp when printed or zoomed); a preview without its controls."""
    image = picture(prepare(widget, width))
    widget.close()
    widget.deleteLater()
    return image


def ask_path(parent, suggested):
    """Where to save a report and in which format, asked in a file dialog; None when cancelled.

    A name typed without an extension gets the chosen format's.
    """
    path, chosen = QtWidgets.QFileDialog.getSaveFileName(parent, "Save report",
                                                         str(Path(suggested).with_suffix(".pdf")), FORMATS)
    if not path:
        return None
    path = Path(path)
    if path.suffix.lower() not in (".pdf", ".docx", ".html", ".png"):
        found = chosen[chosen.find("*.") + 1:chosen.find(")")] if "*." in chosen else ".pdf"
        path = path.with_name(path.name + found)
    return path


def saved_text(saved):
    """What a status line says after a report was saved: one file, or how many figures."""
    if isinstance(saved, list):
        return "Saved {} figures as {}…".format(len(saved), saved[0].name[:-6]) if saved else "No figures to save"
    return "Saved {}".format(Path(saved).name)


def _png(image):
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def _shown_width(image):
    """An image's width on the page, in screen pixels: its pixels over their sharpness."""
    return image.width() / max(image.devicePixelRatio(), 1.0)


def scene_image(canvas):
    """The pipeline as drawn on the sheet, without the squared paper."""
    scene = canvas.scene()
    box = scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
    scale = min(2.0, 1600 / max(box.width(), 1))
    image = QtGui.QImage(int(box.width() * scale), int(box.height() * scale), QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor("#ffffff"))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    selected = scene.selectedItems()
    scene.clearSelection()
    # the squared paper is the view's, not the scene's: rendering the scene leaves it out
    scene.render(painter, QtCore.QRectF(image.rect()), box)
    for item in selected:
        item.setSelected(True)
    painter.end()
    return image


class Report:
    """Headings, paragraphs, tables and images, written out as HTML or PDF."""

    def __init__(self, title):
        self.title = title
        self.parts = []          # ("h2"|"h3"|"p"|"note"|"table"|"image"|"pre", content)
        self.images = []

    def heading(self, text, level=2):
        self.parts.append(("h{}".format(level), text))

    def paragraph(self, text, note=False):
        self.parts.append(("note" if note else "p", text))

    def table(self, header, rows):
        self.parts.append(("table", (header, rows)))

    def image(self, image, width=None):
        self.images.append(image)
        self.parts.append(("image", (len(self.images) - 1, width)))

    def code(self, text):
        self.parts.append(("pre", text))

    def _body(self, source, image_width=FIGURE_WIDTH):
        out = []
        for kind, content in self.parts:
            if kind in ("h2", "h3"):
                out.append("<{0}>{1}</{0}>".format(kind, html.escape(content)))
            elif kind == "p":
                out.append("<p>{}</p>".format(html.escape(content)))
            elif kind == "note":
                out.append("<p class='note'>{}</p>".format(html.escape(content)))
            elif kind == "pre":
                out.append("<pre>{}</pre>".format(html.escape(content)))
            elif kind == "table":
                header, rows = content
                cells = "".join("<th>{}</th>".format(html.escape(str(h))) for h in header)
                body = "".join("<tr>{}</tr>".format("".join("<td>{}</td>".format(html.escape(str(c))) for c in row))
                               for row in rows)
                out.append("<table><tr>{}</tr>{}</table>".format(cells, body))
            elif kind == "image":
                index, width = content
                image = self.images[index]
                shown = min(width or _shown_width(image), image_width)
                out.append("<p><img src='{}' width='{}'></p>".format(source(index, image), int(shown)))
        return "\n".join(out)

    STYLE = ("body { font-family: 'Segoe UI', Arial, sans-serif; color: #1f2328; max-width: 1000px; margin: 24px auto; "
             "line-height: 1.45; } h1 { color: #1b2140; } h2 { color: #1b2140; border-bottom: 1px solid #dde1e6; "
             "padding-bottom: 4px; margin-top: 28px; } h3 { color: #1b2140; margin-bottom: 4px; } "
             "table { border-collapse: collapse; margin: 6px 0 10px 0; } th, td { border: 1px solid #dde1e6; "
             "padding: 3px 8px; text-align: left; } th { background: #f4f6f8; } .note { color: #57606a; } "
             "pre { background: #f6f7f9; padding: 8px; font-size: 9pt; white-space: pre-wrap; } "
             "img { max-width: 100%; }")

    def to_html(self):
        def data_uri(_index, image):
            return "data:image/png;base64," + base64.b64encode(_png(image)).decode("ascii")
        return ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>{0}</title><style>{1}</style></head>"
                "<body><h1>{0}</h1>\n{2}\n</body></html>").format(html.escape(self.title), self.STYLE,
                                                                  self._body(data_uri))

    def save_html(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        return path

    def save_pdf(self, path):
        document = QtGui.QTextDocument()
        for index, image in enumerate(self.images):
            document.addResource(QtGui.QTextDocument.ResourceType.ImageResource,
                                 QtCore.QUrl("image{}".format(index)), image)
        document.setDefaultStyleSheet("h1, h2, h3 { color: #1b2140; } th { background-color: #f4f6f8; } "
                                      "td, th { padding: 2px 6px; } .note { color: #57606a; }")
        writer = QtGui.QPdfWriter(str(path))
        writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
        writer.setPageMargins(QtCore.QMarginsF(15, 15, 15, 15), QtGui.QPageLayout.Unit.Millimeter)
        writer.setResolution(110)
        writer.setTitle(self.title)
        # figures across the page; their pixels are twice that, so they stay sharp when printed
        body = self._body(lambda index, _image: "image{}".format(index), image_width=writer.width())
        document.setHtml("<h1>{}</h1>{}".format(html.escape(self.title), body.replace(
            "<table>", "<table border='1' cellspacing='0' cellpadding='3'>")))
        document.setPageSize(QtCore.QSizeF(writer.width(), writer.height()))
        document.print_(writer)
        return path

    def save_docx(self, path):
        """A Word document: headings, paragraphs, tables and the figures, editable in Word or LibreOffice."""
        from .docx import write_docx
        return write_docx(path, self.title, self.parts, [(_png(image), _shown_width(image), image.height() /
                                                          max(image.devicePixelRatio(), 1.0)) for image in self.images])

    def save_figures(self, path):
        """Only the figures, one PNG each: `path` without its .png, then _01, _02…; returns their paths."""
        path = Path(path)
        stem = path.with_suffix("") if path.suffix.lower() == ".png" else path
        saved = []
        for k, image in enumerate(self.images, start=1):
            target = stem.with_name("{}_{:02d}.png".format(stem.name, k))
            image.save(str(target), "PNG")
            saved.append(target)
        return saved

    def save(self, path):
        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            return self.save_pdf(path)
        if suffix == ".docx":
            return self.save_docx(path)
        if suffix == ".png":
            return self.save_figures(path)
        return self.save_html(path)


def _profile(report, graph, level=2):
    """Where a literature profile comes from, how it was checked and how it differs from the paper."""
    source = graph.source
    if not source:
        return
    report.heading("Literature profile", level)
    rows = [("Paper", source.get("citation", "")), ("DOI", source.get("doi", "")),
            ("Profile", source.get("id", "")), ("Checked", source.get("status", "design"))]
    if graph.changed_from_source():
        rows.append(("Changed", "Blocks or settings were changed after the profile was taken from the library: "
                                "this is an adaptation of the profile, not the profile."))
    report.table(("", ""), [r for r in rows if r[1]])
    if source.get("adaptations"):
        report.paragraph("How it differs from the paper:")
        for line in source["adaptations"]:
            report.paragraph("· " + line, note=True)
    report.paragraph("A profile is the paper's method as GIOVA builds it; its accuracy here is not the "
                     "paper's result.", note=True)


def _closing(report, graphs):
    report.heading("Software")
    report.table(("Package", "Version"), versions())
    cited = references(graphs)
    if cited:
        report.heading("References")
        for line in cited:
            report.paragraph(line)


def _handling(report, handled):
    """What the program did on its own in one run: each file's reading notes, each trial block's trials used and
    the ones it left out, every one with its reason."""
    if not handled:
        return
    report.heading("Data handling")
    report.paragraph("What GIOVA did on its own: how each file was read, which trials were used, and which "
                     "were left out and why.", note=True)
    for heading, lines, rows in handled:
        report.heading(heading, 3)
        for line in lines:
            report.paragraph(line)
        if rows:
            report.table(("Trial", "Condition", "Reason"), rows)


def _comparison_handling(report, result):
    """A comparison's trials per recording and how each file was read; notes every file shares said once."""
    reading = getattr(result, "reading", {}) or {}
    trials = getattr(result, "trials", {}) or {}
    if not reading and not trials:
        return
    report.heading("Data handling")
    report.paragraph("What GIOVA did on its own with each recording: the trials used and left out (and why), "
                     "and how each file was read. Recordings that failed are listed under Failures.", note=True)
    if trials:
        report.heading("Trials per recording", 3)
        report.table(("Recording", "Pipeline", "Trials used", "Left out", "Why"),
                     [(subject, pipeline, used, left, why or "-")
                      for (subject, pipeline), (used, left, why) in trials.items()])
    if reading:
        report.heading("How the files were read", 3)
        notes = list(reading.values())
        common = [n for n in notes[0] if all(n in other for other in notes[1:])] if len(notes) > 1 else []
        if common:
            report.paragraph("Every recording: " + " ".join(common))
        for subject, lines in reading.items():
            own = [n for n in lines if n not in common]
            if own:
                report.paragraph("{}: {}".format(subject, " ".join(own)))


def pipeline_report(graph, outputs, canvas=None):
    """The report of one pipeline's run: `outputs` as `WorkbenchPage.outputs()` gives them. On white, as on paper."""
    with paper():
        return _pipeline_report(graph, outputs, canvas)


def _pipeline_report(graph, outputs, canvas=None):
    """The report of one pipeline's run: `outputs` as `WorkbenchPage.outputs()` gives them."""
    report = Report(graph.name)
    files = sorted({graph.params(n)["path"] for n in graph.sources("load") if graph.params(n)["path"]})
    report.paragraph("{} · {} {} · {}".format(datetime.now().strftime("%d %B %Y, %H:%M"), NAME, VERSION,
                                             ", ".join(files) or "no recording"), note=True)
    if graph.description:
        report.paragraph(graph.description)
    _profile(report, graph)
    if canvas is not None and canvas.cards:
        report.heading("Pipeline")
        report.image(scene_image(canvas))
    report.heading("Methods")
    report.paragraph(" ".join(graph.methods()))
    _handling(report, provenance.summary(graph, outputs))
    report.heading("Results")
    for node_id in graph.order():
        if node_id not in outputs:
            continue
        node = graph.nodes[node_id]
        kind = block(node.type)
        values = outputs[node_id]
        if kind.outputs[0].types[0] == "model":
            continue
        report.heading("{} · {}".format(node.label or kind.title, kind.short(graph.params(node_id)) or node_id), 3)
        report.paragraph(" · ".join(result_line(v) for v in values.values() if result_line(v)), note=True)
        kept = graph.views.get(node_id, {})
        for port, value in values.items():
            if isinstance(value, Scores):
                _scores_tables(report, value)
            if isinstance(value, Contrast):
                report.table(value.table()[0], [[cell if isinstance(cell, str) else "{:.4g}".format(cell)
                                                 for cell in row] for row in value.table()[1:]])
            widget = preview(value)
            if widget is not None:
                if hasattr(widget, "choose"):
                    # drawn as it was chosen on the sheet (channel, condition, average, view), and said so
                    widget.choose(kept.get(port))
                report.image(render(widget))
    _closing(report, [graph])
    report.heading("Pipeline file")
    report.paragraph("Open in GIOVA's Workbench by saving this as a .pipeline.json.", note=True)
    report.code(json.dumps(graph.to_dict(), indent=1))
    return report


def _scores_tables(report, scores):
    spread = "" if len(scores.fold_accuracy) < 2 else " ± {:.1%}".format(scores.accuracy_sd)
    rows = [("Accuracy", "{:.1%}{}".format(scores.accuracy, spread)),
            ("Balanced accuracy", "{:.1%}".format(scores.balanced_accuracy)),
            ("Cohen's κ", "{:.3f}".format(scores.kappa) if np.isfinite(scores.kappa) else "-"),
            ("F1 (macro)", "{:.3f}".format(scores.f1) if np.isfinite(scores.f1) else "-"),
            ("ROC AUC", "{:.3f}".format(scores.auc) if scores.fold_auc else "-"),
            ("Compared with chance", scores.chance_level[0].upper() + scores.chance_level[1:]),
            ("Random limit, alpha 0.05 / 0.01", "{:.1%} / {:.1%} ({} tested trials; Müller-Putz et al., 2008; "
                                                "Combrisson & Jerbi, 2015)".format(
                scores.random_limit, scores.random_limit_01, scores.tested)),
            *([("Unequal conditions", "always guessing {} scores {:.1%}; balanced accuracy random limit {:.1%} "
                "(alpha 0.05)".format(*scores.largest, scores.balanced_limit))] if scores.unequal else []),
            ("Split", "{}, seed {}".format(scores.split_text(), scores.seed)),
            ("Model", scores.model),
            ("Trials", ", ".join("{} {}".format(c, n) for c, n in scores.counts.items()))]
    rows.append(("Permutation test", "p = {:.3g} ({} shuffles)".format(scores.permutation_p, scores.permutations)
                 if scores.permutations else "not run: " + APPROXIMATE))
    if scores.tuning_text():
        rows.append(("Tuning chose", scores.tuning_text()))
    report.table(("Measure", "Value"), rows)
    report.table(["true \\ predicted"] + list(scores.classes),
                 [[c] + [int(v) for v in row] for c, row in zip(scores.classes, scores.confusion)])


#: More recordings than this: their names are left off the plot (the table below has them) and their lines fade.
NAMED_RECORDINGS = 10


def mean_interval(values):
    """(mean, half width of its 95% confidence interval, Student's t); half width NaN for one value."""
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return (float(values.mean()) if values.size else float("nan")), float("nan")
    half = t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
    return float(values.mean()), float(half)


def side_by_side(accuracy, close=0.004, step=0.07):
    """Each point's x: its pipeline's column, recordings with (nearly) the same accuracy there set side by side
    instead of on top of each other - the lines move with them, so they still meet their points."""
    x = np.tile(np.arange(accuracy.shape[1], dtype=float), (accuracy.shape[0], 1))
    for j in range(accuracy.shape[1]):
        rows = [i for i in np.argsort(accuracy[:, j]) if np.isfinite(accuracy[i, j])]
        groups, current = [], []
        for i in rows:
            if current and accuracy[i, j] - accuracy[current[-1], j] > close:
                groups.append(current)
                current = []
            current.append(i)
        groups.append(current)
        for group in groups:
            for k, i in enumerate(group):
                x[i, j] += (k - (len(group) - 1) / 2) * step
    return x


def comparison_plot(result, measure="accuracy"):
    """One measure (batch.MEASURES, accuracy unless chosen) per recording, one column per pipeline: each
    recording's points on one line (the comparison is paired), each pipeline's mean with its 95% confidence
    interval. The axis starts at chance - 0.5 for the accuracies, F1 and AUC of two conditions, 0 for kappa - so
    that differences look their real size; pointing at a point names its recording and says whether its accuracy
    is above chance (the tables below say it for every recording)."""
    from ...pipeline import batch
    from ..theme import color
    title = dict((key, name) for name, key in batch.MEASURES).get(measure, measure)
    plot = pg.PlotWidget()
    plot.setBackground(plot_background())
    plot.setFixedHeight(340)
    plot.showGrid(y=True, alpha=0.15)
    plot.setLabel("left", title)
    plot.getAxis("left").enableAutoSIPrefix(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    names = [result.column_name(c) for c in result.columns]
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
    accuracy = np.asarray(result.values(measure), dtype=float).reshape(len(result.subjects), len(names))
    many = len(result.subjects) > NAMED_RECORDINGS
    line = QtGui.QColor(color("subtle"))
    line.setAlpha(45 if many else 110)
    x = side_by_side(accuracy)
    for row, at in zip(accuracy, x):                         # one recording: its points joined
        finite = np.isfinite(row)
        if finite.sum() > 1:
            plot.plot(at[finite], row[finite], pen=pg.mkPen(line, width=1.2))
    for j, name in enumerate(names):
        colour = CHANNEL_COLORS[j % len(CHANNEL_COLORS)]
        rows = np.flatnonzero(np.isfinite(accuracy[:, j]))
        values = accuracy[rows, j]
        if not values.size:
            continue
        mean, half = mean_interval(values)
        if np.isfinite(half):                                # the mean's 95% confidence interval
            plot.plot([j + 0.22] * 2, [mean - half, mean + half], pen=pg.mkPen(colour, width=2))
        plot.plot([j + 0.1, j + 0.34], [mean, mean], pen=pg.mkPen(colour, width=4))
        tips = ["{} · {} · {} {}{}".format(Path(result.subjects[i]).stem, name, title,
                                           batch.shown(accuracy[i, j], measure),
                                           " (above chance)" if result.better(i, j) else "") for i in rows]
        points = pg.ScatterPlotItem(x=x[rows, j], y=values, size=9, data=tips,
                                    brush=pg.mkBrush(colour), pen=pg.mkPen(plot_background(), width=1),
                                    hoverable=True, hoverSize=13, tip=lambda x, y, data: data)
        plot.addItem(points)
        if j == len(names) - 1 and not many:                 # the recordings named at the right, never overlapping
            placed = []
            for i in sorted(rows, key=lambda r: accuracy[r, j]):
                y = max(accuracy[i, j], placed[-1] + 0.03) if placed else accuracy[i, j]
                placed.append(y)
                label = pg.TextItem(Path(result.subjects[i]).stem, color=color("subtle"), anchor=(0, 0.5))
                label.setPos(j + 0.4, y)
                plot.addItem(label)
    finite = accuracy[np.isfinite(accuracy)]
    floor = 0.0 if measure == "kappa" else 0.5               # chance, or none (kappa 0 is chance)
    low = min(floor, float(finite.min()) - 0.03) if finite.size else floor
    plot.setYRange(max(-1.0 if measure == "kappa" else 0.0, low), 1.0, padding=0.02)
    plot.setXRange(-0.5, len(names) - (0.2 if many else 0.05))
    return plot


def comparison_report(result, recordings_note=""):
    """The report of a comparison of pipelines over recordings, its methods those of the pipelines that were run
    (`result.graphs`). On white, as on paper."""
    if [c[0] for c in result.columns] != [g.name for g in result.graphs for _ in score_nodes(g)]:
        raise ValueError("These results do not say which pipelines produced them: run the comparison again")
    with paper():
        return _comparison_report(result, result.graphs, recordings_note)


#: The Methods sentence of a Cross-validate block run under Across, in place of its own.
ACROSS_METHODS = ("Leave one recording out: each recording tested with the model trained on all the other "
                  "recordings' trials (seed {}), every step fitted on those only; the Cross-validate block's folds "
                  "and permutation test were not run. Accuracy judged against the upper limit of a random "
                  "classifier's accuracy for the number of tested trials (alpha 0.05; Müller-Putz et al., 2008), "
                  "taking them as independent.")


def _comparison_report(result, graphs, recordings_note=""):
    report = Report("Comparison: " + ", ".join(g.name for g in graphs))
    report.paragraph("{} · {} {} · {} recordings{}".format(
        datetime.now().strftime("%d %B %Y, %H:%M"), NAME, VERSION, len(result.subjects),
        " · " + recordings_note if recordings_note else ""), note=True)
    if result.evaluation == "across":
        report.paragraph("Each recording was tested with a model trained on all the other recordings (leave one "
                         "recording out): how well what was learned from others carries over to a recording it "
                         "never saw. Each file is one recording: two files of the same person are not kept apart.")
    else:
        report.paragraph("Each recording was cross-validated on its own trials.")
    report.heading("Summary")
    better = result.better_counts()
    report.table(("Pipeline", "Mean accuracy", "SD", "Median", "Above chance", "Recordings"),
                 [(name, "{:.1%}".format(mean), "{:.1%}".format(sd) if np.isfinite(sd) else "-",
                   "{:.1%}".format(median), "{} of {}".format(better[j], n), n)
                  for j, (name, mean, sd, median, n) in enumerate(result.summary())])
    report.paragraph("Above chance: the recording's accuracy is above the upper limit a random classifier "
                     "reaches with its number of tested trials (alpha 0.05; Müller-Putz et al., 2008); "
                     "marked * below. The limit takes the tested trials as independent, which cross-validated "
                     "ones are not, so it is approximate (Noirhomme et al., 2014).", note=True)
    report.image(render(comparison_plot(result)))
    paired = result.paired_holm()
    if paired:
        report.heading("Paired comparison")
        report.paragraph("Each recording's accuracy under one pipeline minus the other; Wilcoxon signed-rank test "
                         "over recordings (needs at least six). Holm: p adjusted for testing every pair (Holm, "
                         "1979); the recording, not the trial, is the unit.", note=True)
        shown = lambda v: "{:.3g}".format(v) if np.isfinite(v) else "-"
        report.table(("A", "B", "Mean A - B", "Wilcoxon p", "p (Holm)", "Recordings"),
                     [(a, b, "{:+.1%}".format(d), shown(p), shown(q), n) for a, b, d, p, q, n in paired])
    report.heading("Per recording")
    names = [result.column_name(c) for c in result.columns]
    report.table(["Recording"] + names,
                 [[s] + [result.cell(i, j) for j in range(len(names))] for i, s in enumerate(result.subjects)])
    if result.errors:
        report.heading("Failures", 3)
        for (subject, pipeline), message in result.errors.items():
            report.paragraph("{} · {}: {}".format(subject, pipeline, message), note=True)
    _comparison_handling(report, result)
    report.heading("Methods")
    for graph in graphs:
        report.heading(graph.name, 3)
        # under Across the Cross-validate block's folds and permutations were not run: say what was
        instead = ({node: ACROSS_METHODS.format(graph.params(node)["seed"]) for node in score_nodes(graph)}
                   if result.evaluation == "across" else None)
        report.paragraph(" ".join(m for m in graph.methods(instead) if not m.startswith("Recording ")))
        if graph.source:
            report.paragraph("From {} (doi {}); checked: {}.{} Differs from the paper: {}".format(
                graph.source.get("citation", ""), graph.source.get("doi", "-"), graph.source.get("status", "design"),
                " Changed after it was taken from the library." if graph.changed_from_source() else "",
                " ".join(graph.source.get("adaptations", []))), note=True)
    _closing(report, graphs)
    return report
