"""The Help window opens, finds topics, and quotes the numbers the code uses."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import html
import inspect
import unittest

from pyqtgraph.Qt import QtWidgets

from eeg.core import channel_profiles as cp
from eeg.core import impedance as imp
from eeg.core.config import BANDS
from eeg.core.monitoring import EEG_BAND, IDENTICAL_DIFFERENCE, QualityLimits
from eeg.ui import help_content as hc

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def text(title):
    from eeg.ui.help_dialog import plain_text
    return " ".join(plain_text(dict(hc.TOPICS)[title]).split())


class HelpContentTests(unittest.TestCase):
    def test_the_workbench_help_says_how_it_is_now(self):
        words = " ".join(text(t) for t, _ in hc.TOPICS if t.startswith("Workbench"))
        for now in ("library", "mine", "head maps over time", "add to block", "save every trial", "larger",
                    "fp, af, f, fc, c, cp, p, po, o", "requirements.txt", "run.py", "in this block",
                    "double-click a recording", "roc curve", "mirrored waves", "turns into stop",
                    "results and log", "compared with chance", "point at it"):
            self.assertIn(now, words, now)
        # templates and the literature moved from New ▾ to the Library
        for gone in ("new ▾ → from the literature", "new ▾ → cho2017", "a template or from the literature"):
            self.assertNotIn(gone, words, gone)

    def test_every_setting_is_in_the_settings_table(self):
        words = text("Parameters explained")
        for setting in ("psd graphs in", "downsample files to",
                        "hover tips", "theme", "show the games tab", "folder, file name"):
            self.assertIn(setting, words, setting)

    def test_every_topic_has_a_heading_and_body(self):
        self.assertEqual(len({t for t, _ in hc.TOPICS}), len(hc.TOPICS))
        for title, body in hc.TOPICS:
            self.assertIn("<h2>", body, title)
            self.assertGreater(len(body), 400, title)
        for topic in hc.TAB_TOPICS.values():
            self.assertIn(topic, dict(hc.TOPICS))

    def test_monitor_numbers_match_the_code(self):
        words = text("Monitor: how it computes")
        limits = QualityLimits()
        for quoted in ("{:g}–{:g} hz".format(*EEG_BAND), "< {:g}".format(limits.flat_uv),
                       "> {:g}".format(limits.high_uv), "> {:g} %".format(limits.mains_ratio * 100),
                       "> {:g} %".format(limits.saturation_ratio * 100),
                       "< {:g} %".format(IDENTICAL_DIFFERENCE * 100)):
            self.assertIn(quoted, words)
        for name, low, high in BANDS:
            self.assertIn("{} {:g}–{:g}".format(name, low, high), words)

    def test_monitor_filter_defaults_match_the_panel(self):
        from eeg.ui import panel
        source = inspect.getsource(panel.ControlPanel._filter_group)
        self.assertIn("low=3.0", source)
        self.assertIn("3–45 hz", text("Monitor: how it computes"))

    def test_impedance_numbers_match_the_code(self):
        words = text("Impedance check")
        for quoted in ("{:g} na at {:g} hz".format(imp.DRIVE_AMPS * 1e9, imp.LEAD_OFF_HZ),
                       "{:g} kω".format(imp.SERIES_OHMS / 1e3), "{:g} s settling".format(imp.SETTLE_SECONDS),
                       "{:g} s are analysed".format(imp.MEASURE_SECONDS), "{:g}×".format(imp.MIN_SNR),
                       "< {:g} kω".format(imp.GREEN_OHMS / 1e3), "< {:g} kω".format(imp.YELLOW_OHMS / 1e3),
                       "≥ {:g} mω".format(imp.OPEN_OHMS / 1e6)):
            self.assertIn(quoted, words)

    def test_quality_table_explained_matches_the_code(self):
        from eeg.ui.live_views import QualityTable
        words = text("Quality table explained")
        limits = QualityLimits()
        for quoted in ("> {:g} %".format(limits.saturation_ratio * 100), "< {:g}".format(limits.flat_uv),
                       "> {:g}".format(limits.high_uv), "> {:g} %".format(limits.mains_ratio * 100),
                       "less than {:g} %".format(IDENTICAL_DIFFERENCE * 100),
                       "{:g}–{:g} hz".format(*EEG_BAND)):
            self.assertIn(quoted, words)
        table = QualityTable()
        for col in range(table.columnCount()):          # every column is explained
            self.assertIn(table.horizontalHeaderItem(col).text().lower(), words)
        for status in ("collecting data", "missing data", "saturated", "flat", "check signal",
                       "identical signal", "line noise", "no issues detected"):
            self.assertIn(status, words)

    def test_snr_numbers_match_the_code(self):
        from eeg.core.spectral import SNR_NEIGHBOURS_HZ
        words = text("Monitor: how it computes")
        self.assertIn("{:g}–{:g} hz away".format(*SNR_NEIGHBOURS_HZ), words)
        self.assertIn("{:g}×".format(imp.MIN_SNR), words)

    def test_alpha_profile_pins_match_the_code(self):
        profile = next(p for p in cp.builtin_profiles() if p.name.startswith("Alpha 4"))
        self.assertIn(profile.name.lower(), text("Channel profiles"))
        on = [(i + 1, profile.names[i]) for i, s in enumerate(profile.settings) if s.on]
        self.assertEqual(on, [(5, "PO3"), (6, "PO4"), (7, "O2"), (15, "O1")])
        self.assertIn("5 6 7 15", text("Channel profiles"))


class ParameterTopicTests(unittest.TestCase):
    """Every number in 'Parameters explained' is the one the code uses."""

    def setUp(self):
        self.words = text("Parameters explained")

    def test_spectral_windows(self):
        from eeg.core.spectral import welch_psd
        from eeg.ui import live_views, monitor
        signature = inspect.signature(welch_psd).parameters
        self.assertEqual((signature["window"].default, signature["overlap"].default), (2.0, 0.5))
        # Spectra follow the display window; quality keeps its 4 s.
        self.assertIn("self._spectral_seconds() * self.fs",
                      inspect.getsource(monitor.MonitorPage._spectral_samples))
        self.assertIn("self.panel.window_seconds.value()",
                      inspect.getsource(monitor.MonitorPage._spectral_seconds))
        self.assertIn("capacity4 = round(self.fs * 4)", inspect.getsource(monitor.MonitorPage._ingest))
        self.assertEqual(live_views.GLIDE_S, 0.3)
        for quoted in ("= display window", "δf = 1 / t", "10 s → 0.1 hz", "4 s → 0.25 hz",
                       "2 s segments, 50 % overlap", "δf = 0.5 hz", "every 0.5 s", "2 s time constant",
                       "0.3 s"):
            self.assertIn(quoted, self.words)

    def test_fft_steps_match_the_code(self):
        """The four FFT steps in Help are the ones spectra_snapshot performs."""
        import numpy as np
        from eeg.core import monitoring
        source = inspect.getsource(monitoring.spectra_snapshot)
        for step in ("safe.mean(axis=1", "np.hanning(n)", "np.fft.rfft(", "/ max(window.sum()", "*= 2"):
            self.assertIn(step, source)
        words = text("Monitor: how it computes")
        for quoted in ("remove the mean", "0.5 − 0.5·cos(2πn / (n−1))", "δf = fs / n = 1 / t",
                       "÷ σ w[n]", "× 2", "a 20 µv sine at 10 hz therefore reads 20 µv",
                       "3 for 4 s, 9 for 10 s", "quality table does not follow the window"):
            self.assertIn(quoted, words)
        # The worked numbers: amplitude, bin spacing and Welch segment count.
        from eeg.core.spectral import welch_sizes
        for seconds, spacing, segments in ((4, 0.25, 3), (10, 0.1, 9)):
            t = np.arange(int(seconds * 125)) / 125
            snapshot = monitoring.spectra_snapshot((20 * np.sin(2 * np.pi * 10 * t))[None, :], 125.0)
            self.assertAlmostEqual(snapshot["fft_freqs"][1], spacing)
            self.assertAlmostEqual(snapshot["fft"][0, int(round(10 / spacing))], 20.0, places=3)
            size, overlap = welch_sizes(t.size, 125.0)
            self.assertEqual(1 + (t.size - size) // (size - overlap), segments)

    def test_default_window_is_5_s(self):
        from eeg.ui.panel import ControlPanel, DEFAULT_WINDOW_S
        from eeg.ui.preferences import user_settings
        import numpy as np
        from eeg.core import monitoring
        from eeg.core.spectral import welch_sizes
        self.assertEqual(DEFAULT_WINDOW_S, 5)
        self.assertEqual(ControlPanel(125.0).window_seconds.value(), 5)
        user_settings().remove("trace_window")                   # a first start
        from eeg.ui.monitor import MonitorPage
        page = MonitorPage()
        self.assertEqual(page.panel.window_seconds.value(), 5)
        self.assertEqual(page._spectral_samples(), 5 * 125)
        page.close()
        self.assertIn("(default 5 s)", text("Monitor: how it computes"))
        self.assertIn("window 5 s", self.words)
        t = np.arange(5 * 125) / 125
        snapshot = monitoring.spectra_snapshot(np.sin(2 * np.pi * 10 * t)[None, :], 125.0)
        self.assertAlmostEqual(snapshot["fft_freqs"][1], 0.2)
        size, overlap = welch_sizes(t.size, 125.0)
        self.assertEqual(1 + (t.size - size) // (size - overlap), 4)
        self.assertIn("5 s (default) 0.2 hz 4", text("Monitor: how it computes"))

    def test_welch_settings_and_segment_formula(self):
        """Defaults in Help = preference defaults; K formula = what welch_sizes does."""
        from eeg.ui.preferences import DEFAULTS
        from eeg.core.spectral import welch_sizes
        self.assertEqual(DEFAULTS["monitor/spectrum_seconds"][0], 0.0)      # = display window
        self.assertEqual(DEFAULTS["monitor/welch_seconds"][0], 2.0)
        self.assertEqual(DEFAULTS["monitor/welch_overlap"][0], 50.0)
        words = text("Monitor: how it computes")
        for quoted in ("settings → monitor → spectra (monitor)", "welch segment l 2 s",
                       "welch overlap o 50 %", "δf = 1 / l", "also while streaming"):
            self.assertIn(quoted, words)
        fs = 125.0
        for seconds, segment, overlap in ((10, 2, .5), (10, 1, 0), (10, 4, .75), (4, 2, .5), (7, 3, .5)):
            n = int(seconds * fs)
            size, step_back = welch_sizes(n, fs, segment, overlap)
            real = 1 + (n - size) // (size - step_back)
            formula = int((seconds - segment) / (segment * (1 - overlap)) + 1e-9) + 1
            self.assertEqual(real, formula, (seconds, segment, overlap))

    def test_notch(self):
        import numpy as np
        from scipy.signal import freqz, iirnotch
        from eeg.core.processing import design_filter
        q = inspect.signature(design_filter).parameters["notch_q"].default
        self.assertEqual(q, 30.0)
        b, a = iirnotch(50.0, q, 125.0)
        freqs, response = freqz(b, a, worN=200000, fs=125.0)
        band = freqs[np.abs(response) < 1 / np.sqrt(2)]
        self.assertIn("≈{:.1f}–{:.1f} hz".format(band.min(), band.max()), self.words)
        self.assertIn("q = f₀ ÷ bandwidth", self.words)

    def test_filter_and_analysis_defaults(self):
        from eeg.ui.filters import FilterGroup
        group = FilterGroup(125.0)
        self.assertTrue(group.harmonics.isChecked())
        self.assertEqual((group.order.value(), group.notch_q.value()), (4, 30))
        from eeg.core.trials import Settings
        self.assertEqual((Settings().duration, Settings().quality_mad), (30.0, 6.0))
        for quoted in ("+ harmonics on", "order 4", "notch q 30", "common duration (to − from) 30 s", "6 mad"):
            self.assertIn(quoted, self.words)

    def test_live_features_and_display_rate(self):
        from eeg.ui import features_window, monitor
        source = inspect.getsource(features_window.FeaturesWindow.__init__)
        self.assertIn("spin_box(.1,60,2,", source)
        self.assertIn("spin_box(.01,10,.25,", source)
        self.assertEqual(features_window.FeaturesWindow.HISTORY_SECONDS, 60.0)
        self.assertEqual((monitor.FRAME_MS, monitor.SMOOTH_FRAME_MS, monitor.FAST_FRAME_MS), (50, 33, 16))
        self.assertEqual(monitor.DISPLAY_RATES, {60: 16, 30: 33})
        for quoted in ("window s 2 s", "every s 0.25 s", "history 60 s", "30 fps (or 60 fps)",
                       "60 → 30 → 20 fps", "30 % late"):
            self.assertIn(quoted, self.words)


class BlockLibraryTests(unittest.TestCase):
    def test_every_block_has_a_page_with_every_part(self):
        from eeg.pipeline import graph
        from eeg.ui import help_blocks as hb
        self.assertEqual(set(hb.BLOCKS), set(graph.REGISTRY))
        for type_name, kind in graph.REGISTRY.items():
            does, use, cannot = hb.BLOCKS[type_name]
            self.assertTrue(does and use and cannot, type_name)
            page = hb.page(type_name)
            for part in ("<h2>", "Use it for", "It cannot"):
                self.assertIn(part, page, type_name)
            for param in kind.params:
                self.assertIn(param.label or param.name, html.unescape(page), type_name)
        listed = [t for _, _, types in hb.by_category() for t in types]
        self.assertEqual(sorted(listed), sorted(graph.REGISTRY))

    def test_a_page_names_the_templates_that_use_the_block(self):
        from eeg.ui import help_blocks as hb
        self.assertIn("Motor imagery: CSP + LDA", hb.used_in("csp"))
        self.assertIn("See it in", hb.page("csp"))

    def test_right_click_help_opens_the_blocks_page(self):
        from eeg.ui.workbench import WorkbenchPage
        page = WorkbenchPage()
        dialog = page.help_for("lda")
        self.assertEqual(dialog.key(), "block:lda")
        self.assertIn("Linear discriminant analysis", dialog.browser.toPlainText())
        dialog.close()


class HelpDialogTests(unittest.TestCase):
    def test_opens_at_a_topic_or_a_tab(self):
        from eeg.ui.help_dialog import HelpDialog
        dialog = HelpDialog()
        dialog.open_at("Troubleshooting")
        self.assertEqual(dialog.key(), "Troubleshooting")
        self.assertIn("unable to add streamer", dialog.browser.toPlainText())
        dialog.open_at("Analysis")
        self.assertEqual(dialog.key(), "Analysis & features")
        dialog.open_at("block:csp")
        self.assertEqual(dialog.key(), "block:csp")
        self.assertTrue(dialog.topics.currentItem().parent().isExpanded())
        self.assertIn("Separate more than two conditions", dialog.browser.toPlainText())
        dialog.close()

    def test_every_block_page_shows_the_block_as_it_looks_on_the_sheet(self):
        from pyqtgraph.Qt import QtCore, QtGui
        from eeg.ui.help_dialog import HelpDialog, PICTURE, picture_size
        dialog = HelpDialog()
        blocks = [key for key in dialog._items if key.startswith("block:")]
        self.assertGreater(len(blocks), 30)
        for key in blocks:
            dialog.open_at(key)
            type_name = key[len("block:"):]
            self.assertIn('src="{}:{}"'.format(PICTURE, type_name), dialog.browser.toHtml())
        image = dialog.browser.loadResource(QtGui.QTextDocument.ResourceType.ImageResource,
                                            QtCore.QUrl("{}:csp".format(PICTURE)))
        width, height = picture_size("csp")
        self.assertEqual((image.width(), image.height()), (2 * width, 2 * height))      # sharp at 2x
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)                              # on the page's own ground
        self.assertTrue(any(image.pixelColor(x, image.height() // 2).alpha() == 255 for x in range(image.width())))
        self.assertNotIn(PICTURE, dialog._text["block:csp"])                            # searching reads the words only
        dialog.close()

    def test_about_says_research_only_and_feedback_opens_an_issue(self):
        from urllib.parse import parse_qs, urlparse
        from pyqtgraph.Qt import QtWidgets
        from eeg import REPOSITORY, VERSION
        from eeg.ui.help_dialog import HelpDialog, feedback_url
        dialog = HelpDialog()
        dialog.open_at("About GIOVA")
        text = dialog.browser.toPlainText()
        for words in ("not a medical device", "sends nothing anywhere", "GPL-3.0", VERSION):
            self.assertIn(words, text)
        url = urlparse(feedback_url())
        self.assertEqual("{}://{}{}".format(url.scheme, url.netloc, url.path), REPOSITORY + "/issues/new")
        body = parse_qs(url.query)["body"][0]
        self.assertIn("GIOVA " + VERSION, body)
        self.assertNotIn(__import__("os").environ.get("USERNAME", "\0"), body)      # nothing about the person
        names = [b.text() for b in dialog.findChildren(QtWidgets.QPushButton)]
        self.assertIn("Send feedback…", names)
        dialog.close()

    def test_search_keeps_matching_topics(self):
        from eeg.ui.help_dialog import HelpDialog
        dialog = HelpDialog()
        everything = dialog.visible()
        dialog.search.setText("hjorth")
        visible = dialog.visible()
        self.assertIn("Analysis & features", visible)
        self.assertIn("block:features", visible)                # the block library is searched too
        self.assertNotIn("Troubleshooting", visible)
        self.assertIn(dialog.key(), visible)
        dialog.search.clear()
        self.assertEqual(dialog.visible(), everything)
        dialog.close()

    def test_main_window_button_and_f1(self):
        from eeg.ui.app import MainWindow
        window = MainWindow()
        window.open_mode("Workbench")
        window.help_button.click()
        self.assertTrue(window.help_dialog.isVisible())
        self.assertEqual(window.help_dialog.key(), "Workbench")
        self.assertIs(window.open_help("Channel profiles"), window.help_dialog)   # built once
        self.assertEqual(window.help_dialog.key(), "Channel profiles")
        window.help_dialog.close()
        from pyqtgraph.Qt import QtCore, QtGui
        f1 = [s for s in window.findChildren(QtGui.QShortcut)
              if s.key() == QtGui.QKeySequence(QtCore.Qt.Key.Key_F1)]
        self.assertEqual(len(f1), 1)
        f1[0].activated.emit()
        self.assertTrue(window.help_dialog.isVisible())
        window.help_dialog.close()
        window.close()


if __name__ == "__main__":
    unittest.main()
