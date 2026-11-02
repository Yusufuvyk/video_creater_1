"""Per-channel amplifier settings on the Cyton+Daisy: power and BIAS membership.

Everything here goes to the ADS1299 chips through the Cyton SDK channel command

    x <channel> <power_down> <gain> <input> <bias> <srb2> <srb1> X

so a channel switched off here is powered down in the amplifier, not merely
hidden on screen, and a channel taken out of BIAS no longer feeds the bias
drive. Changing what is plotted never does this; only `apply_settings` does.

What software cannot do: disconnect the BIAS pin. With every channel out of
BIAS the drive amplifier has no feedback input; on the Cyton hardware its
output can float or saturate. To have no BIAS electrode at all, unplug the lead.

The board forgets all of this on every connect (BrainFlow soft-resets it in
`prepare_session`), so settings are sent again each time a session opens.
"""
import time
from dataclasses import dataclass

#: SDK channel characters, Cyton 1-8 then Daisy 9-16.
CHANNEL_CHARS = "12345678QWERTYUI"
GAIN_24 = "6"


@dataclass(frozen=True)
class ChannelSetting:
    on: bool = True
    bias: bool = True


def default_settings(n=len(CHANNEL_CHARS)):
    """What the board boots with: every channel on and in BIAS and SRB2."""
    return [ChannelSetting() for _ in range(n)]


def channel_command(index, setting):
    """The SDK string for one channel.

    A channel that is off is also taken out of BIAS and SRB2 and its input
    shorted, as the firmware's own deactivate command does: a powered-down
    channel left in the bias loop would still inject whatever its open lead
    picks up.
    """
    if not 0 <= index < len(CHANNEL_CHARS):
        raise ValueError("Cyton+Daisy has channels 1-16, not {}".format(index + 1))
    if setting.on:
        fields = ("0", GAIN_24, "0", "1" if setting.bias else "0", "1", "0")
    else:
        fields = ("1", GAIN_24, "1", "0", "0", "0")
    return "x" + CHANNEL_CHARS[index] + "".join(fields) + "X"


def changed_channels(settings, previous):
    """Indices whose setting differs; every channel when `previous` is unknown."""
    if previous is None or len(previous) != len(settings):
        return list(range(len(settings)))
    return [i for i, (new, old) in enumerate(zip(settings, previous)) if new != old]


def apply_settings(board, settings, previous=None):
    """Send channel settings; raise if the firmware refuses one.

    With `previous` (what the board already has), only the channels that differ
    are sent. Each command waits for the firmware's reply, so switching one
    channel off costs one round trip instead of sixteen.

    Call with the stream stopped: the firmware only answers ("Success: ..." or
    "Failure: ...") when it is not streaming, and the answer is the only proof
    the setting landed.
    """
    sent = []
    for index in changed_channels(settings, previous):
        setting = settings[index]
        command = channel_command(index, setting)
        response = _send(board, command)
        if "failure" in response.lower():
            raise RuntimeError("board refused {}: {}".format(command, response.strip()))
        sent.append((command, response.strip()))
    return sent


def _send(board, command, attempts=3):
    """One command, read back as text.

    Right after stop_stream the serial buffer still holds binary sample
    packets (they start with 0xA0), and BrainFlow returns them as the reply to
    the next command - its Python wrapper then fails to decode them as UTF-8.
    That error comes only after the command was written, and a channel command
    is idempotent, so sending it again reads a reply from a drained buffer.
    """
    for attempt in range(attempts):
        try:
            return board.config_board(command) or ""
        except UnicodeDecodeError:
            if attempt == attempts - 1:
                raise RuntimeError("no readable reply to {} (serial buffer still "
                                   "holds stream data)".format(command))
            time.sleep(0.1)
    return ""


def describe(settings, names):
    off = [n for n, s in zip(names, settings) if not s.on]
    bias = [n for n, s in zip(names, settings) if s.on and s.bias]
    return "off: {} · BIAS from: {}".format(
        ", ".join(off) or "none", ", ".join(bias) or "none (feedback inputs open)")


def encode(settings):
    """Compact text for QSettings: one 'on,bias' pair per channel."""
    return ";".join("{}{}".format(int(s.on), int(s.bias)) for s in settings)


def decode(text, n=len(CHANNEL_CHARS)):
    try:
        pairs = str(text).split(";")
        settings = [ChannelSetting(p[0] == "1", p[1] == "1") for p in pairs]
    except (IndexError, TypeError):
        return default_settings(n)
    return settings if len(settings) == n else default_settings(n)
