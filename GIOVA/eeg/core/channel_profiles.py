"""Channel profiles for the Cyton+Daisy: which electrode is on which pin, and
which pins are powered.

A headset is wired once and then used for days, so the wiring should be chosen
once, not re-typed at every Start. A profile holds the sixteen pin names and the
sixteen on/BIAS settings; the Monitor remembers the last one picked and opens
with it.

Stored as JSON text (in QSettings by the UI), so this module stays pure - no Qt.
"""
import json
from dataclasses import dataclass, field

from . import hardware as hw
from .montages import CYTON_DAISY_1020

N_PINS = len(hw.CHANNEL_CHARS)


def _generic(pin):
    return "CH{}".format(pin)


def _only(pins):
    """16 names and settings: `pins` {pin number: electrode} on, every other pin off.

    An unused pin is named CH<pin>, not its default electrode: the default O1 on
    pin 7 would otherwise sit beside an O1 moved to pin 15.
    """
    out_names = [pins.get(pin, _generic(pin)) for pin in range(1, N_PINS + 1)]
    settings = [hw.ChannelSetting(pin in pins, pin in pins) for pin in range(1, N_PINS + 1)]
    return out_names, settings


@dataclass
class ChannelProfile:
    name: str
    names: list = field(default_factory=lambda: list(CYTON_DAISY_1020))
    settings: list = field(default_factory=lambda: hw.default_settings(N_PINS))

    def validate(self):
        if not str(self.name).strip():
            raise ValueError("A profile needs a name")
        if len(self.names) != N_PINS or len(self.settings) != N_PINS:
            raise ValueError("A Cyton+Daisy profile has {} pins".format(N_PINS))
        cleaned = [str(n).strip() for n in self.names]
        if not all(cleaned):
            raise ValueError("Every pin needs a name")
        if len({n.lower() for n in cleaned}) != len(cleaned):
            raise ValueError("Two pins have the same name")

    def to_dict(self):
        return {"name": self.name, "names": list(self.names),
                "settings": hw.encode(self.settings)}

    @classmethod
    def from_dict(cls, payload):
        try:
            profile = cls(str(payload["name"]), [str(n) for n in payload["names"]],
                          hw.decode(payload["settings"], n=N_PINS))
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed channel profile") from exc
        profile.validate()
        return profile


def builtin_profiles():
    alpha_names, alpha_settings = _only({5: "PO3", 6: "PO4", 7: "O2", 15: "O1"})
    return [
        ChannelProfile("OpenBCI default (16)"),
        ChannelProfile("Alpha 4 · O1 O2 PO3 PO4", alpha_names, alpha_settings),
    ]


BUILTIN_NAMES = frozenset(p.name for p in builtin_profiles())


def decode_user(text):
    """[ChannelProfile] from stored JSON; unreadable entries are skipped."""
    try:
        entries = json.loads(text) if text else []
    except (TypeError, ValueError):
        return []
    out = []
    for entry in entries if isinstance(entries, list) else []:
        try:
            profile = ChannelProfile.from_dict(entry)
        except ValueError:
            continue
        if profile.name not in BUILTIN_NAMES:
            out.append(profile)
    return out


def encode_user(profiles):
    return json.dumps([p.to_dict() for p in profiles if p.name not in BUILTIN_NAMES])


def all_profiles(user_text):
    return builtin_profiles() + decode_user(user_text)
