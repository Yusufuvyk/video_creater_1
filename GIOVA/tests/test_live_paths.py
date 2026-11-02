"""Live recordings in folders BrainFlow cannot open, and the names in their logs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path

from eeg.core import boards
from eeg.core import hardware as hw
from eeg.core.sources import LiveSource, ascii_stand_in, hardware_log_path, load_brainflow_csv


class StandInTests(unittest.TestCase):
    def test_only_non_ascii_paths_get_a_stand_in(self):
        self.assertIsNone(ascii_stand_in(Path("C:/Data/live.csv")))
        stand_in = ascii_stand_in(Path("C:/Users/x/Masaüstü/LAST_VERSİON/live.csv"))
        self.assertTrue(str(stand_in).isascii())
        self.assertEqual(stand_in.suffix, ".csv")


class LiveRecordingTests(unittest.TestCase):
    def test_synthetic_board_records_into_a_turkish_folder(self):
        """Real BrainFlow: the CSV ends up at the requested path, complete."""
        from brainflow.board_shim import BoardShim
        BoardShim.disable_board_logger()
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "Masaüstü" / "LAST_VERSİON" / "Data"
            source = LiveSource("", None, record_dir=folder, board=boards.SYNTHETIC)
            try:
                time.sleep(1.0)
                source.read(50)
            finally:
                source.close()
            self.assertTrue(source.record_path.is_file())
            self.assertEqual(source.record_path.parent, folder)
            self.assertGreater(load_brainflow_csv(source.record_path).data.shape[1], 100)

    def test_hardware_log_uses_the_profile_names(self):
        with tempfile.TemporaryDirectory() as root:
            source = LiveSource.__new__(LiveSource)
            source.info = boards.CYTON_DAISY
            source.names = ["CH{}".format(i + 1) for i in range(16)]
            source.names[14] = "O1"
            source.record_path = Path(root) / "live.csv"
            settings = [hw.ChannelSetting(i == 14, i == 14) for i in range(16)]
            source._log_hardware(settings, "start")
            log = hardware_log_path(source.record_path).read_text(encoding="utf-8")
            self.assertIn("BIAS from: O1", log)
            self.assertNotIn("P3", log)            # the default name of pin 15


if __name__ == "__main__":
    unittest.main()
