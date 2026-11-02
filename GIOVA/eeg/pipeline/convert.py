"""The Analysis page's set-up as a pipeline, so it can be opened on the Workbench.

The page's profile (`core.analysis_profiles.AnalysisProfile`) already says
everything a pipeline needs: the file, the reference, the filter, the
channels, how trials are cut, the Welch settings, the features and the
time-frequency settings. Each becomes a block with the same numbers.

Trials come from the same source the page used: around the recording's
events, each event's own length, a labelled sequence, or a table typed by
hand (a profile has no table; a saved session has, see `from_session`).
The Condition field names trials added by hand and never filters the
events. Notches are the page's list, without added harmonics. What cannot
be carried over is reported back, not hidden.

Pure stdlib - no Qt.
"""
import json
from pathlib import Path

from ..core import analysis_profiles as ap
from .graph import Graph

#: Profile tabs that open a time-frequency view.
TF_TABS = ("time-frequency", "band", "brain map")


def from_session(path, recording=None, source_path=None):
    """(Graph, notes) for a saved Analysis session (`.analysis.json`, versions 1 to 3), without the page.

    The session is checked by `formats.trial_analysis.load_session` against the recording it names (its
    path, rate, length and channels), so a session of another recording is refused and nothing is changed.
    The trials are the session's own table - ids, conditions, start / end, included - cut for the common
    duration it saved, from the time zero its profile says; the filters, Welch settings and feature profile
    are those it was computed with (`Settings`), not the profile's, when the two differ. A profile, when the
    session has one, adds what `Settings` does not hold: the time zero and the time-frequency settings.
    The session file and the recording are only read.

    `recording` / `source_path`: the recording already read and where it is; by default the path the session
    names, read as the Recording block reads it (and reduced to the session's rate when it was reduced).
    """
    from ..core.sources import downsample
    from ..formats.trial_analysis import load_session
    from . import compute
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        saved = payload["recording"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Not a readable analysis session: {}".format(exc)) from None
    source_path = str(source_path or saved.get("path") or "")
    if not source_path or not Path(source_path).exists():
        raise ValueError("The recording this session belongs to is not at {}: put it back there, or open it "
                         "and choose the session again".format(source_path or "(no path saved)"))
    notes, reduced = [], None
    if recording is None:
        recording = compute.read_recording(source_path)
    try:
        if abs(float(saved["fs"]) - recording.fs) > 1e-9:
            # opened with Settings -> Files -> reduce to a rate: the same whole-factor reduction
            smaller = downsample(recording, float(saved["fs"]))
            if abs(smaller.fs - float(saved["fs"])) <= 1e-9:
                recording, reduced = smaller, float(saved["fs"])
    except (KeyError, TypeError, ValueError):
        pass
    channels, trials, settings, display = load_session(path, recording, source_path)
    profile = ap.AnalysisProfile.from_dict(display["profile"]) if display.get("profile") else None

    name = Path(path).name.removesuffix(".json").removesuffix(".analysis")
    graph = Graph(name or "Analysis session", "Opened from the analysis session {}.".format(Path(path).name))
    previous = graph.add("load", {"path": source_path})

    def then(type_name, params, port="raw"):
        nonlocal previous
        node = graph.add(type_name, params)
        graph.connect(previous, port, node, port)
        previous = node
        return node

    if reduced:
        then("resample", {"target": reduced})
        notes.append("The session was made from the recording reduced to {:g} Hz: a Downsample block does the "
                     "same.".format(reduced))
    filters = dict(settings.filters or {})
    reference = filters.get("reference") or "none"
    if reference != "none":
        then("reference", {"mode": reference, "channels": list(filters.get("reference_channels") or [])})
    if filters.get("low") or filters.get("high") or filters.get("notches"):
        params = {"low": float(filters.get("low") or 0.0), "high": float(filters.get("high") or 0.0),
                  "order": int(filters.get("order", 4)), "notch": 0.0,
                  "notches": [float(f) for f in filters.get("notches") or []]}
        if "notch_q" in filters:
            params["notch_q"] = float(filters["notch_q"])
        then("filter", params)

    # where each trial's time zero is: the event, `pre` s after the start, when trials were cut around events
    zero = 0.0
    if profile is not None and profile.trial_source == ap.EVENTS and not profile.event_length:
        zero = float(profile.event_window[0])
    elif profile is None:
        notes.append("This session has no profile (an older file): each trial's time zero is its start.")
    epochs = then("epochs", {"source": "manual", "trials": [[t.id, t.condition, t.start, t.end, t.included]
                                                           for t in trials],
                             "tmin": 0.0 - zero, "tmax": float(settings.duration) - zero,
                             "quality_mad": float(settings.quality_mad)})
    analysed = graph.add("trial_channels", {"channels": list(channels)})
    graph.connect(epochs, "epochs", analysed, "epochs")

    detrend = filters.get("detrend", "constant")
    psd = graph.add("psd", {"window": float(settings.window), "overlap": float(settings.overlap),
                            "detrend": detrend, "signal_mean": True})
    graph.connect(analysed, "epochs", psd, "epochs")
    fft = graph.add("fft", {"detrend": detrend, "signal_mean": True})
    graph.connect(analysed, "epochs", fft, "epochs")
    if settings.profile:
        from ..core.feature_profiles import FeatureProfile
        whole = FeatureProfile.from_dict(settings.profile)
        features = graph.add("features", {
            "selected": list(whole.selected), "window": float(settings.window), "overlap": float(settings.overlap),
            "detrend": detrend, "bands": [list(b) for b in whole.bands], "ratios": [list(r) for r in whole.ratios],
            "total_range": [float(v) for v in whole.total_range], "roi": len(channels) > 1})
        graph.connect(analysed, "epochs", features, "epochs")

    if profile is not None and profile.epoch_settings:
        settings_tf = dict(profile.epoch_settings)
        erd = {"channels": list(channels), "conditions": []}
        for key in ("baseline", "active", "method", "fmin", "fmax", "window", "overlap", "smooth", "bands",
                    "baseline_mode", "reference", "erd_mode", "n_cycles", "wavelet", "reject_uv"):
            if key in settings_tf:
                erd[key] = settings_tf[key]
        if erd.get("baseline_mode") == "condition" and not str(erd.get("reference") or "").strip():
            # the page took the first condition of the trial table when none was named
            erd["reference"] = trials[0].condition if trials else ""
            notes.append("No reference condition was named: '{}', the first in the trial table, as the page "
                         "chose.".format(erd["reference"]))
        erd["topography"] = profile.tab == "brain map"
        node = graph.add("erd", erd)
        graph.connect(epochs, "epochs", node, "raw")

    hidden = [i for i, shown in (display.get("visibility") or {}).items() if not shown]
    if hidden:
        notes.append("{} trial curve{} hidden on the page ({}): hiding was a drawing choice only, so every "
                     "included trial is analysed here as it was there.".format(
                         len(hidden), "" if len(hidden) == 1 else "s", ", ".join(hidden[:8])))
    excluded = [t.id for t in trials if not t.included]
    if excluded:
        notes.append("Left out of the selection, as on the page: {}.".format(", ".join(excluded[:12])))
    # what the page showed, kept as the spectrum preview's choice (a view, not a setting of the computation)
    graph.views = {psd: {"spectrum": {"choosers": [["Average", "Mean signal" if display.get("calculation") ==
                                                    "mean_signal" else "Mean of trials"]]}}}
    return graph.arrange(), notes


def from_profile(profile, path=None):
    """(Graph, notes) for an AnalysisProfile and the recording the page has open."""
    notes = []
    name = profile.name if profile.name and profile.name != "Blank" else "From the Analysis page"
    graph = Graph(name, "Opened from the Analysis page{}.".format(
        " ({})".format(Path(path).name) if path else ""))
    load = graph.add("load", {"path": str(path) if path else ""})
    previous = load
    filters = dict(profile.filters or {})

    def then(type_name, params):
        nonlocal previous
        node = graph.add(type_name, params)
        graph.connect(previous, "raw", node, "raw")
        previous = node
        return node

    reference = filters.get("reference") or "none"
    if profile.filters_enabled and reference != "none":
        then("reference", {"mode": reference, "channels": list(filters.get("reference_channels") or [])})
    if profile.filters_enabled and (filters.get("low") or filters.get("high") or filters.get("notches")):
        # the page's notches exactly as listed (core.processing.design_filter): no harmonics added
        params = {"low": float(filters.get("low") or 0.0), "high": float(filters.get("high") or 0.0),
                  "order": int(filters.get("order", 4)), "notch": 0.0,
                  "notches": [float(f) for f in filters.get("notches") or []]}
        if "notch_q" in filters:
            params["notch_q"] = float(filters["notch_q"])
        then("filter", params)
    whole = previous                       # every electrode, for a head map, before any channel is picked
    windowed = profile.trial_source == ap.EVENTS and not profile.event_length
    if profile.channels and windowed:
        then("channels", {"channels": list(profile.channels)})

    pre, post = (float(v) for v in profile.event_window)
    # the page's Condition field names trials added by hand; it never filtered the recording's events, so
    # every condition the trials have is kept
    conditions = []
    epoch_params = {"conditions": conditions}
    if windowed:
        tmin, tmax = -pre, post
    else:
        tmin, tmax = 0.0, float(profile.common_duration)
        if profile.trial_source == ap.EVENTS:
            epoch_params["event_length"] = True
        elif profile.trial_source == ap.SEQUENCE:
            epoch_params.update(source="sequence", sequence=profile.sequence,
                                sequence_start=float(profile.sequence_start), repeats=int(profile.repeats))
        else:
            epoch_params.update(source="manual", trials=[])
            notes.append("Trials entered by hand are not part of a profile: add them in the Epochs block's trial "
                         "table, or open the saved session (.analysis.json), which has them.")
    epoch_params.update(tmin=tmin, tmax=tmax)
    epochs = graph.add("epochs", epoch_params)
    graph.connect(previous, "raw", epochs, "raw")
    analysed = epochs
    if profile.channels and not windowed:
        # trials cut once from every electrode (the time-frequency block reads them all, for a head map);
        # spectra and features of the page's channels only
        analysed = graph.add("trial_channels", {"channels": list(profile.channels)})
        graph.connect(epochs, "epochs", analysed, "epochs")

    psd_params = {"window": float(profile.window), "overlap": float(profile.overlap)}
    if filters.get("detrend") == "linear" and profile.filters_enabled:
        psd_params["detrend"] = "linear"
    psd = graph.add("psd", psd_params)
    graph.connect(analysed, "epochs", psd, "epochs")
    feature = dict(profile.feature_profile or {})
    bands = [list(b) for b in feature.get("bands") or []]
    if bands:
        power = graph.add("band_power", {"bands": bands, "log": False})
        graph.connect(psd, "spectrum", power, "spectrum")
    if feature.get("selected"):
        # the whole feature profile: its bands, ratios and total range define band:, relative:, ratio: and peak:
        # features; a region of several channels is measured channel by channel and averaged, as on the page
        from ..core.feature_profiles import FeatureProfile
        whole_profile = FeatureProfile.from_dict(feature)
        features = graph.add("features", {
            "selected": list(whole_profile.selected), "window": float(profile.window),
            "overlap": float(profile.overlap), "detrend": psd_params.get("detrend", "constant"),
            "bands": [list(b) for b in whole_profile.bands], "ratios": [list(r) for r in whole_profile.ratios],
            "total_range": [float(v) for v in whole_profile.total_range], "roi": len(profile.channels) > 1})
        graph.connect(analysed, "epochs", features, "epochs")

    settings = dict(profile.epoch_settings or {})
    if settings and (windowed or profile.tab in TF_TABS):
        erd = {"channels": list(profile.channels) or ["C3", "Cz", "C4"], "conditions": conditions,
               "tmin": tmin, "tmax": tmax}
        for key in ("baseline", "active", "method", "fmin", "fmax", "window", "overlap", "smooth", "bands",
                    "baseline_mode", "reference", "erd_mode", "n_cycles", "wavelet", "reject_uv"):
            if key in settings:
                erd[key] = settings[key]
        if profile.tab == "brain map":
            erd["topography"] = True
        node = graph.add("erd", erd)
        if windowed:
            # cut around the events from the referenced, filtered recording with every electrode
            graph.connect(whole, "raw", node, "raw")
        else:
            # the same trials as the Epochs block: one table, one set of trials
            graph.connect(epochs, "epochs", node, "raw")
    return graph.arrange(), notes
