"""Named, saveable analysis set-ups: channels, how trials are made, filters,
spectral / time-frequency settings, features and the view to open.

"Eyes open / closed on O1 and O2" used to be written into the page - a combo
that searched for "O1", a sequence field that started as "Eyes open:30; Eyes
closed:30". That made every other experiment a matter of retyping. Now it is
one profile among others: choosing a profile applies it, and anyone can save
their own (an SSVEP set-up on Oz, motor imagery on C3/C4).

A profile stores names, not indices: it is applied to recordings with other
channel layouts, and a channel the recording lacks is reported, not guessed.

Pure stdlib/numpy - no Qt.
"""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import re

import numpy as np

from .config import DATA_DIR
from .feature_profiles import FeatureProfile

#: Where trials come from.
MANUAL, SEQUENCE, EVENTS = "manual", "sequence", "events"
TRIAL_SOURCES = (MANUAL, SEQUENCE, EVENTS)
#: Result tabs a profile can open with.
TABS = ("signal", "spectrum", "time-frequency", "band", "brain map", "features")
SUFFIX = ".profile.json"
VERSION = 1

# Kept so profiles written by the two-page version still load.
BLOCKS, EPOCHS = "blocks", "epochs"


@dataclass
class AnalysisProfile:
    name: str = "Blank"
    description: str = ""
    #: channel names; empty means "the first channel of the recording"
    channels: list = field(default_factory=list)
    filters_enabled: bool = True
    filters: dict = field(default_factory=dict)
    # trials
    trial_source: str = MANUAL
    condition: str = ""
    sequence: str = ""
    sequence_start: float = 0.0
    repeats: int = 1
    common_duration: float = 30.0
    #: seconds before / after each event when trials come from events
    event_window: list = field(default_factory=lambda: [0.0, 4.0])
    #: use each event's own length (block markers) instead of the window
    event_length: bool = False
    # spectrum
    window: float = 2.0
    overlap: float = 0.5
    quality_mad: float = 6.0
    psd_scale: str = "linear"
    feature_profile: dict = field(default_factory=dict)
    #: core.epochs.EpochSettings fields (time-frequency, ERD, bands, baseline)
    epoch_settings: dict = field(default_factory=dict)
    tab: str = "spectrum"
    kind: str = BLOCKS          # ignored; read for compatibility
    version: int = VERSION

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("Malformed analysis profile")
        known = {f.name for f in fields(cls)}
        try:
            profile = cls(**{k: v for k, v in payload.items() if k in known})
        except TypeError as exc:
            raise ValueError("Malformed analysis profile") from exc
        if "trial_source" not in payload and payload.get("kind") == EPOCHS:
            profile.trial_source = EVENTS
            settings = profile.epoch_settings
            profile.event_window = [-float(settings.get("tmin", 0.0)), float(settings.get("tmax", 4.0))]
            profile.tab = "band"
        profile.validate()
        return profile

    def validate(self):
        if self.version != VERSION:
            raise ValueError("Unsupported analysis profile version")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A profile needs a name")
        if not isinstance(self.channels, list) or not all(isinstance(c, str) and c.strip() for c in self.channels):
            raise ValueError("Profile channels must be channel names")
        if not isinstance(self.filters, dict) or not isinstance(self.epoch_settings, dict):
            raise ValueError("Profile filters and epoch settings must be dictionaries")
        if self.trial_source not in TRIAL_SOURCES:
            raise ValueError("Trial source must be manual, sequence or events")
        if self.tab not in TABS:
            raise ValueError("Unknown result tab '{}'".format(self.tab))
        numbers = (self.sequence_start, self.common_duration, self.window, self.overlap, self.quality_mad,
                   *self.event_window)
        if len(self.event_window) != 2 or not np.isfinite(numbers).all():
            raise ValueError("Profile times must be finite numbers")
        if self.common_duration <= 0 or self.window <= 0 or sum(self.event_window) <= 0:
            raise ValueError("Profile durations must be positive")
        if not 0 <= self.overlap < 1 or self.quality_mad < 0 or self.sequence_start < 0:
            raise ValueError("Profile overlap must be in [0, 1) and MAD non-negative")
        if int(self.repeats) != self.repeats or self.repeats < 1:
            raise ValueError("Profile repeats must be a positive integer")
        if self.psd_scale not in ("linear", "log", "db", "snr"):
            raise ValueError("PSD scale must be linear, log, db or snr")
        if self.feature_profile:
            FeatureProfile.from_dict(self.feature_profile)
        if self.sequence.strip():
            parse_sequence(self.sequence)

    def resolve_channels(self, names):
        """(indices found in `names`, names that are missing), in profile order.

        Matching ignores case and surrounding spaces, because files spell the
        same electrode 'OZ', 'Oz' and ' Oz'.
        """
        lookup = {str(n).strip().upper(): i for i, n in enumerate(names)}
        found, missing = [], []
        for name in self.channels:
            index = lookup.get(name.strip().upper())
            if index is None:
                missing.append(name)
            elif index not in found:
                found.append(index)
        return found, missing


def parse_sequence(text):
    """'Eyes open:30; Eyes closed:30' -> [('Eyes open', 30.0), ('Eyes closed', 30.0)]."""
    phases = []
    for part in str(text).split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError("Each phase is Condition:seconds")
        name, seconds = part.rsplit(":", 1)
        try:
            value = float(seconds.replace(",", "."))
        except ValueError as exc:
            raise ValueError("Phase duration must be a number") from exc
        if not name.strip() or not np.isfinite(value) or value <= 0:
            raise ValueError("Each phase needs a condition and a positive duration")
        phases.append((name.strip(), value))
    if not phases:
        raise ValueError("The sequence has no phases")
    return phases


# ----------------------------------------------------------------------
# built-in profiles
# ----------------------------------------------------------------------


def builtin_profiles():
    """Read-only starting points. Change settings and 'Save as' to make them yours."""
    eyes = AnalysisProfile(
        name="Eyes open / closed",
        description="Occipital alpha: O1 and O2 as an ROI, 30 s blocks from markers "
                    "(or the sequence), eyes closed relative to eyes open.",
        channels=["O1", "O2"],
        filters=dict(low=1.0, high=45.0, order=4, notches=[50.0], notch_q=30.0,
                     detrend="constant", reference="none"),
        trial_source=EVENTS, event_length=True, event_window=[0.0, 30.0],
        condition="Eyes open", sequence="Eyes open:30; Eyes closed:30", repeats=4,
        common_duration=30.0, window=2.0, overlap=0.5, feature_profile=FeatureProfile().to_dict(),
        epoch_settings=dict(method="stft", fmin=2.0, fmax=40.0, window=2.0, overlap=0.5, smooth=0.0,
                            baseline_mode="condition", reference="", active=[0.0, 30.0],
                            erd_mode="pooled", bands=[["alpha", 8.0, 13.0], ["beta", 13.0, 30.0]]),
        tab="spectrum")
    mu_beta = FeatureProfile(name="Mu / beta", bands=[["mu", 8.0, 13.0], ["beta", 13.0, 30.0]],
                             ratios=[["mu/beta", "mu", "beta"]],
                             selected=["band:mu", "band:beta", "relative:mu", "relative:beta"],
                             total_range=[1.0, 40.0])
    cho = AnalysisProfile(
        name="Motor imagery (Cho2017)",
        description="Left vs right hand imagery: 7 s trials from events, cue at 2 s, "
                    "baseline 0.5-1.5 s, imagery 2-5 s, sensorimotor channels.",
        channels=["C3", "Cz", "C4"],
        filters=dict(low=1.0, high=40.0, order=4, notches=[60.0], notch_q=30.0,
                     detrend="constant", reference="none"),
        trial_source=EVENTS, event_window=[0.0, 7.0], common_duration=7.0,
        window=1.0, overlap=0.5, feature_profile=mu_beta.to_dict(),
        epoch_settings=dict(method="stft", fmin=4.0, fmax=40.0, window=0.5, overlap=0.9, n_cycles=0.5,
                            smooth=0.5, baseline_mode="window", baseline=[0.5, 1.5], active=[2.0, 5.0],
                            erd_mode="pooled", bands=[["mu", 8.0, 13.0], ["beta", 13.0, 30.0]]),
        tab="band")
    bands = FeatureProfile(name="Band powers")
    bands.selected = ["variance"] + ["band:" + b[0] for b in bands.bands]
    blank = AnalysisProfile(
        name="Blank",
        description="No channel or trial assumptions: first channel, manual trials, band powers.",
        filters=dict(low=1.0, high=45.0, order=4, notches=[50.0], notch_q=30.0),
        common_duration=10.0, feature_profile=bands.to_dict(),
        epoch_settings=dict(baseline_mode="window", baseline=[0.0, 1.0], active=[1.0, 10.0]))
    return [eyes, cho, blank]


BUILTIN_NAMES = frozenset(p.name for p in builtin_profiles())


# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------


def profiles_dir(root=None):
    return Path(root or DATA_DIR) / "profiles"


def slug(name):
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return text or "profile"


def save_profile(path, profile):
    profile.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(profile.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    return Path(path)


def load_profile(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read analysis profile: {}".format(exc)) from exc
    return AnalysisProfile.from_dict(payload)


def save_user_profile(profile, root=None):
    """Write into the profiles folder. Built-in names are reserved."""
    if profile.name in BUILTIN_NAMES:
        raise ValueError("'{}' is built in; choose another name".format(profile.name))
    return save_profile(profiles_dir(root) / (slug(profile.name) + SUFFIX), profile)


def delete_user_profile(name, root=None):
    if name in BUILTIN_NAMES:
        raise ValueError("Built-in profiles cannot be deleted")
    for path, profile in user_profiles(root):
        if profile.name == name:
            path.unlink()
            return True
    return False


def user_profiles(root=None):
    """[(path, profile)] for every readable profile file; broken files are skipped."""
    directory = profiles_dir(root)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*" + SUFFIX)):
        try:
            out.append((path, load_profile(path)))
        except ValueError:
            continue
    return out


def available(root=None):
    """Built-in then user profiles; a user file never shadows a built-in."""
    return builtin_profiles() + [p for _, p in user_profiles(root) if p.name not in BUILTIN_NAMES]
