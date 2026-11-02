"""The channel profile editor works with no board, in any mode."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import patch

from pyqtgraph.Qt import QtWidgets

from eeg.core import channel_profiles as cp
from eeg.ui.preferences import user_settings

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ALPHA = "Alpha 4 · O1 O2 PO3 PO4"


class ProfileEditorTests(unittest.TestCase):
    def setUp(self):
        user_settings().remove('channel_profiles')
        from eeg.ui.monitor import MonitorPage
        self.page = MonitorPage()
        self.page.panel.mode.setCurrentIndex(0)          # file playback: no board at all
        self.dialogs = []
        real_exec = QtWidgets.QDialog.exec

        def capture(dialog):                              # keep it, do not block
            self.dialogs.append(dialog)
            return 0
        self._patch = patch.object(QtWidgets.QDialog, 'exec', capture)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        user_settings().remove('channel_profiles')

    def open(self):
        self.page.edit_profiles()
        return self.dialogs[-1]

    def test_opens_without_a_board_and_builtins_are_read_only(self):
        dialog = self.open()
        dialog.list.setCurrentRow([dialog.list.item(r).text() for r in range(dialog.list.count())].index(ALPHA))
        self.assertFalse(dialog.electrodes[0].isEnabled())
        self.assertFalse(dialog.save_button.isEnabled())
        self.assertEqual(dialog.electrodes[14].text(), "O1")

    def test_new_edit_save_use_rename_delete(self):
        dialog = self.open()
        rows = [dialog.list.item(r).text() for r in range(dialog.list.count())]
        dialog.list.setCurrentRow(rows.index(ALPHA))
        with patch.object(QtWidgets.QInputDialog, 'getText', return_value=("Lab cap", True)):
            dialog._new()
        self.assertEqual(dialog.list.currentItem().text(), "Lab cap")
        self.assertTrue(dialog.electrodes[0].isEnabled())

        dialog.electrodes[0].setText("Fz")
        dialog.on_boxes[0].setChecked(True)
        dialog._check()
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.electrodes[1].setText("O1")
        dialog._check()             # O1 is already on pin 15
        self.assertFalse(dialog.save_button.isEnabled())
        self.assertIn("same name", dialog.message.text())
        dialog.electrodes[1].setText("CH2")
        dialog._check()

        dialog._use()                                                   # save & use
        stored = {p.name: p for p in self.page._channel_profiles()}["Lab cap"]
        self.assertEqual(stored.names[0], "Fz")
        self.assertTrue(stored.settings[0].on)
        self.assertEqual(self.page.panel.profile_combo.currentText(), "Lab cap")
        self.assertEqual(self.page.panel.hardware.names[0], "Fz")
        self.assertEqual([i for i in self.page.panel.hardware.on_channels()], [0, 4, 5, 6, 14])

        dialog = self.open()                                            # rename
        dialog.name.setText("Lab cap 2")
        dialog._check()
        dialog._save()
        names = [p.name for p in self.page._channel_profiles()]
        self.assertIn("Lab cap 2", names)
        self.assertNotIn("Lab cap", names)

        with patch.object(QtWidgets.QMessageBox, 'question',
                          return_value=QtWidgets.QMessageBox.StandardButton.Yes):
            dialog._delete()
        self.assertEqual([p.name for p in self.page._channel_profiles()],
                         [p.name for p in cp.builtin_profiles()])

    def test_builtin_names_cannot_be_taken(self):
        dialog = self.open()
        with patch.object(QtWidgets.QInputDialog, 'getText', return_value=("Mine", True)):
            dialog._new()
        dialog.name.setText(cp.builtin_profiles()[0].name)
        dialog._check()
        self.assertFalse(dialog.save_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
