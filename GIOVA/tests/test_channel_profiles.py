"""Cyton+Daisy channel profiles: pin names and on/BIAS, stored and reopened."""
import unittest

from eeg.core import channel_profiles as cp


class ChannelProfileTests(unittest.TestCase):
    def test_alpha_four_uses_the_wired_pins_only(self):
        profile = next(p for p in cp.builtin_profiles() if "PO3" in p.name)
        on = {i + 1: profile.names[i] for i, s in enumerate(profile.settings) if s.on}
        self.assertEqual(on, {5: "PO3", 6: "PO4", 7: "O2", 15: "O1"})
        self.assertEqual(len(set(profile.names)), cp.N_PINS)   # no duplicate O1/O2

    def test_user_profiles_round_trip_and_cannot_shadow_builtins(self):
        names = ["P{}".format(i) for i in range(cp.N_PINS)]
        mine = cp.ChannelProfile("Mine", names, cp.hw.default_settings(cp.N_PINS))
        fake = cp.ChannelProfile("OpenBCI default (16)", names, cp.hw.default_settings(cp.N_PINS))
        loaded = cp.decode_user(cp.encode_user([mine, fake]))
        self.assertEqual([p.name for p in loaded], ["Mine"])
        self.assertEqual(loaded[0].names, names)

    def test_broken_storage_is_ignored(self):
        self.assertEqual(cp.decode_user("not json"), [])
        self.assertEqual(cp.decode_user('[{"name": "x"}]'), [])

    def test_duplicate_pin_names_are_refused(self):
        with self.assertRaises(ValueError):
            cp.ChannelProfile("dup", ["O1"] * cp.N_PINS).validate()


if __name__ == "__main__":
    unittest.main()
