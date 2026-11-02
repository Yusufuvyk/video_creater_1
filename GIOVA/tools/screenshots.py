"""Pictures of GIOVA for the README and the website: the window drawn off screen, with settings of its own
(not the user's).

Home; the Workbench running CSP + LDA on Cho2017 subject 1 with its results; band power per condition
(left and right hand) on the same subject with its result; the Live monitor before a board is
connected; Help on a block's page. Writes <out>/<name>-<theme>.png.

Run: .venv/Scripts/python tools/screenshots.py <folder with Cho2017 s01.mat> [dark|light] [out folder] [scale]
A scale of 1.5 draws every picture at 1.5 times its pixels, for sharp images on the web.
"""
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"            # also gives the tests' own settings file, not the user's
if len(sys.argv) > 4:
    os.environ["QT_SCALE_FACTOR"] = sys.argv[4]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIZE = (1440, 900)


def settle(app, seconds=0.5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def run_and_show(app, page, graph, block_type):
    """`graph` in a tab of its own, run to the end, fitted, with `block_type`'s block selected."""
    for sheet in list(page.sheets):
        page.close_sheet(sheet)
    page.new_tab(graph=graph)
    settle(app)
    page.run()
    end = time.monotonic() + 300
    settle(app, 2.0)
    while page.sheet.busy and time.monotonic() < end:     # the run is in the background
        settle(app, 1.0)
    for sheet in [s for s in page.sheets if s is not page.sheet]:
        page.close_sheet(sheet)
    page.canvas.fit()
    page.canvas.select(next(n for n, node in page.sheet.graph.nodes.items() if node.type == block_type))
    settle(app, 3.0)


def main():
    cho = Path(sys.argv[1]) / "s01.mat" if len(sys.argv) > 1 else None
    theme = sys.argv[2] if len(sys.argv) > 2 else "dark"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "docs" / "images"
    from pyqtgraph.Qt import QtWidgets
    from eeg.ui import app as giova
    from eeg.ui.fonts import load_fonts
    from eeg.ui.theme import set_theme, use_theme
    from eeg.ui.preferences import user_settings
    from eeg.pipeline import templates
    user_settings().clear()
    user_settings().setValue("ui/theme", theme)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    load_fonts(application)
    set_theme(theme)
    use_theme(application)
    out.mkdir(parents=True, exist_ok=True)
    window = giova.MainWindow()
    window.resize(*SIZE)
    window.show()

    def save(name, widget=window):
        widget.grab().save(str(out / "{}-{}.png".format(name, theme)))

    window.open_mode("Home")
    settle(application, 1.5)
    save("home")
    window.open_mode("Monitor")
    settle(application, 2.0)
    save("monitor")
    if cho is not None and cho.is_file():
        window.open_mode("Workbench")
        settle(application)
        page = window.page("Workbench")
        run_and_show(application, page, templates.csp_lda(str(cho)), "cross_validate")
        save("workbench")
        # band power per condition: left and right hand imagery, the imagery period, bars per band and hand
        graph = templates.band_power(str(cho))
        loader = next(n for n, node in graph.nodes.items() if node.type == "load")
        trials = next(n for n, node in graph.nodes.items() if node.type == "epochs")
        graph.nodes[loader].params["condition"] = templates.CHO2017_IMAGERY
        graph.nodes[trials].params.update(tmin=2.0, tmax=5.0)
        run_and_show(application, page, graph, "band_power")
        save("results")
    help_dialog = window.open_help("block:csp")
    help_dialog.resize(1100, 720)
    settle(application, 1.0)
    save("help", help_dialog)
    print("saved in", out)


if __name__ == "__main__":
    main()
