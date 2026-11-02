"""GIOVA's start-up screen and the window icon: shown without the heavy app imports."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

HERE = Path(__file__).resolve().parents[1]


class Splash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_the_start_up_screen_is_giovas(self):
        from eeg.ui.splash import GiovaSplash, BLUE, TEAL, show_splash
        splash = GiovaSplash(dark=True, pixel_ratio=2.0)
        self.assertEqual((splash.NAME, splash.SUBTITLE, splash.FOOTER, splash.status, splash.version),
                         ('GIOVA', 'Open-source biosignal lab', 'Biosignal recording and analysis',
                          'Preparing your workspace…', '1.0.0'))
        image = splash.pixmap().toImage()
        self.assertEqual((image.width(), image.height()), (1200, 676))        # 600 x 338, sharp at 2x
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)                    # rounded corners
        self.assertEqual(image.pixelColor(600, 20).name(), '#1c222b')          # GIOVA's dark surface
        colours = {image.pixelColor(x, y).name() for x in range(0, 1200, 2) for y in range(0, 676, 2)}

        def near(target):
            return any(abs(QtGui.QColor(c).red() - target.red()) < 30 and abs(QtGui.QColor(c).green() - target.green()) < 30
                       and abs(QtGui.QColor(c).blue() - target.blue()) < 30 for c in colours)
        self.assertTrue(near(BLUE))
        self.assertTrue(near(TEAL))            # the logo's ribbon and the accent
        light = GiovaSplash(dark=False, pixel_ratio=1.0).pixmap().toImage()
        self.assertEqual(light.pixelColor(300, 10).name(), '#f1f1f1')
        before = splash.pixmap().cacheKey()
        splash.set_status('Loading the Workbench…')
        self.assertNotEqual(splash.pixmap().cacheKey(), before)
        shown = show_splash(self.app)
        self.assertIsInstance(shown, GiovaSplash)
        shown.close()

    def test_the_bar_fills_as_modules_load(self):
        import eeg.ui.splash as sp
        splash = sp.GiovaSplash(dark=True, pixel_ratio=1.0)
        splash._expected = splash._first + 4
        splash.follow_imports()
        self.assertIs(sys.meta_path[0], splash._pacer)
        self.assertEqual(splash.progress, 0.0)
        before = splash.pixmap().cacheKey()
        for name in ('wave', 'colorsys'):                        # small stdlib modules, loaded afresh
            sys.modules.pop(name, None)
        import wave                                              # noqa: F401 - each import asks the pacer first
        import colorsys                                          # noqa: F401
        splash._last = 0.0
        splash.tick()
        self.assertGreater(splash.progress, 0.0)
        self.assertLessEqual(splash.progress, 0.97)              # never full before the window is there
        self.assertNotEqual(splash.pixmap().cacheKey(), before)
        window = QtWidgets.QWidget()
        splash.finish(window)
        self.assertEqual(splash.progress, 1.0)
        self.assertNotIn(splash._pacer, sys.meta_path)
        self.assertFalse(any(isinstance(f, sp._Pacer) for f in sys.meta_path))

    def test_icon_has_windows_sizes_and_writes_ico(self):
        from eeg.ui.splash import app_icon, write_ico, ICON_SIZES
        icon = app_icon()
        for size in (16, 24, 32, 48, 256):
            self.assertIn(QtCore.QSize(size, size), icon.availableSizes())
        path = Path(tempfile.mkdtemp()) / 'giova.ico'
        self.assertTrue(write_ico(path))
        data = path.read_bytes()
        self.assertEqual(int.from_bytes(data[4:6], 'little'), len(ICON_SIZES))
        # Every size is its own image (the approved .ico), and Qt can read the file back.
        self.assertEqual(QtGui.QImageReader(str(path)).imageCount(), len(ICON_SIZES))
        self.assertFalse(QtGui.QIcon(str(path)).isNull())

    def test_the_light_theme_has_its_own_window_icon(self):
        from eeg.ui.splash import app_icon, ICON_SIZES
        dark, light = app_icon(), app_icon(light=True)
        for size in ICON_SIZES:
            self.assertIn(QtCore.QSize(size, size), light.availableSizes())

        def middle(icon):                                                       # the tile, above the wave
            image = icon.pixmap(48, 48).toImage()
            return image.pixelColor(image.width() // 2, image.height() // 12)
        self.assertLess(middle(dark).lightness(), 80)                           # navy tile
        self.assertGreater(middle(light).lightness(), 180)                      # pale tile

    def test_icon_trace_stays_inside_the_tile_at_every_size(self):
        from eeg.ui.splash import icon_pixmap, ICON_SIZES
        for size in ICON_SIZES:
            image = icon_pixmap(size).toImage()
            corner = image.pixelColor(0, 0).alpha()
            self.assertLessEqual(corner, 8)                  # rounded, not a square (smoothing leaves a trace)
            border = [image.pixelColor(x, y) for x in range(size) for y in (0, size - 1)]
            self.assertFalse(any(c.red() > 200 and c.green() > 200 and c.blue() > 200 for c in border))

    def test_splash_has_no_university_line(self):
        import inspect
        import eeg.ui.splash as splash
        self.assertNotIn("Universit", inspect.getsource(splash))

    def test_splash_module_does_not_pull_in_the_heavy_stack(self):
        # The point of the splash is to be on screen before scipy/brainflow load.
        code = ("import sys; sys.path.insert(0, r'{}'); import eeg.ui.splash; "
                "print(','.join(m for m in ('scipy', 'brainflow', 'numpy', 'pandas') if m in sys.modules))").format(HERE)
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             env=dict(os.environ, QT_QPA_PLATFORM='offscreen'), timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        loaded = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
        self.assertNotIn('scipy', loaded)
        self.assertNotIn('brainflow', loaded)


if __name__ == '__main__':
    unittest.main()
