"""The stand-in for a game's `acquisition.EEG`, reading GIOVA's stream.

Same attributes and methods as the class in alpha_ball_game_openbci (and v3-v5), so the
game script runs unchanged. Where the original opened the board, this asks EEG
Studio: for samples, to insert markers, to start a recording, to apply the
game's channel settings. Channel names are the ones GIOVA streams, so a
feature channel the headset does not have still stops the game with the
game's own error.
"""
import json
import os
from pathlib import Path
import sys
import time

from ..core.game_bridge import BridgeClient

ADDRESS_ENV = "EEGSTUDIO_BRIDGE"
KEY_ENV = "EEGSTUDIO_BRIDGE_KEY"


def _client():
    address = os.environ.get(ADDRESS_ENV)
    key = os.environ.get(KEY_ENV)
    if not address or not key:
        raise RuntimeError("This game must be started from GIOVA (Games tab)")
    host, port = address.rsplit(":", 1)
    return BridgeClient((host, int(port)), bytes.fromhex(key))


def _game_config():
    return sys.modules.get("config")


class StudioEEG:
    """A live stream from GIOVA's Monitor: open it, ask for the last few seconds, close it."""

    #: seconds to wait for the Monitor's first samples before giving up
    START_TIMEOUT = 20.0

    def __init__(self):
        self._bridge = _client()
        deadline = time.time() + self.START_TIMEOUT
        info = self._bridge.request("hello")
        while not info["fs"] and time.time() < deadline:
            time.sleep(0.2)
            info = self._bridge.request("hello")
        if not info["fs"]:
            raise RuntimeError("GIOVA's Monitor is not streaming - start it (live or file playback)")
        self.fs = int(round(info["fs"])) if float(info["fs"]).is_integer() else info["fs"]
        self.names = list(info["names"])
        self.rows = list(range(len(self.names)))
        self.source = info["source"]
        self.recording_file = None
        self.now = None
        config = _game_config()
        disabled = list(getattr(config, "DISABLED_CHANNELS", []) or [])
        if disabled:
            self.rows_for(disabled)                  # stop on a misspelt name, as the original does
        self.configure_channels()

    # ---------------- the original interface ----------------

    def configure_channels(self):
        config = _game_config()
        self._bridge.request("configure", disabled=list(getattr(config, "DISABLED_CHANNELS", []) or []),
                             use_bias=bool(getattr(config, "USE_BIAS", True)))

    def rows_for(self, names):
        unknown = [name for name in names if name not in self.names]
        if unknown:
            raise ValueError(f"{unknown} not in the channels GIOVA streams ({', '.join(self.names)}) "
                             "- check config.py against the headset")
        return [self.names.index(name) for name in names]

    def last_seconds(self, seconds):
        return self._bridge.request("last", seconds=float(seconds)).astype(float)

    def window(self, seconds):
        data, stamp, reason = self._bridge.request("window", seconds=float(seconds))
        return data.astype(float), stamp, reason

    def start_recording(self, filename):
        self.recording_file = filename
        config = _game_config()
        settings = {
            "names": self.names,
            "disabled_channels": list(getattr(config, "DISABLED_CHANNELS", []) or []),
            "use_bias": bool(getattr(config, "USE_BIAS", True)),
            "reference": "SRB2",
            "source": "GIOVA ({})".format(self.source),
        }
        path = Path(filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".channels.json"), "w") as file:
            json.dump(settings, file, indent=2)
        self._bridge.request("start_recording", path=str(path))

    def mark(self, value):
        self._bridge.request("mark", value=float(value), time=time.time())

    def close(self):
        try:
            self._bridge.request("stop_recording")
        except Exception:
            pass
        self._bridge.close()
