"""Versioned analysis selections and self-describing numeric CSV exports."""
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..core.trials import Settings, Trial, validate_settings, welch_plan


#: 3: `channels` list (one channel or an ROI) replaces the single `channel`.
VERSION = 3


def recording_identity(recording, path):
    return dict(path=str(Path(path).resolve()), fs=float(recording.fs),
                n_samples=int(recording.data.shape[1]),
                channels=list(recording.channel_names))


def save_session(path, recording, source_path, channels, trials, settings, display=None):
    channels = [channels] if isinstance(channels, str) else list(channels)
    payload = dict(version=VERSION, recording=recording_identity(recording, source_path),
                   channels=channels, trials=[asdict(t) for t in trials],
                   settings=asdict(settings), display=display or {})
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_session(path, recording, source_path):
    """Validate everything before the UI replaces its current selection."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["version"] not in (1, 2, VERSION):
            raise ValueError("Unsupported analysis file version")
        expected = recording_identity(recording, source_path)
        saved = payload["recording"]
        if (not np.isclose(saved["fs"], expected["fs"], rtol=0, atol=1e-9)
                or saved["n_samples"] != expected["n_samples"]
                or saved["channels"] != expected["channels"]
                or Path(saved["path"]).resolve() != Path(expected["path"])):
            raise ValueError("Analysis belongs to a different recording or channel layout")
        channels = payload["channels"] if "channels" in payload else [payload["channel"]]
        if not channels or any(name not in expected["channels"] for name in channels):
            raise ValueError("Saved channel is unavailable")
        settings = Settings(**payload["settings"])
        validate_settings(settings, recording.fs)
        trials = [Trial(**item) for item in payload["trials"]]
        if len({t.id for t in trials}) != len(trials):
            raise ValueError("Duplicate trial IDs")
        for t in trials:
            if (not isinstance(t.id, str) or not t.id.strip()
                    or not isinstance(t.condition, str) or not t.condition.strip()
                    or not isinstance(t.included, bool)
                    or not np.isfinite([t.start, t.end]).all()):
                raise ValueError("Invalid trial definition")
        return list(channels), trials, settings, payload.get("display", {})
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed analysis file") from exc


def power_comparison(conditions, calculation, reference):
    """Rows for the UI and exports; undefined relative differences stay empty."""
    ref = next((c for c in conditions if c.name == reference), None)
    ref_metrics = getattr(ref, calculation) if ref is not None else None
    rows = []
    for condition in conditions:
        metrics = getattr(condition, calculation)
        values = {"Mean square": metrics.power, **metrics.bands}
        reference_values = ({"Mean square": ref_metrics.power, **ref_metrics.bands}
                            if ref_metrics is not None else {})
        for name, value in values.items():
            base = reference_values.get(name)
            delta = value - base if base is not None else None
            percent = 100 * delta / base if base is not None and base != 0 else None
            rows.append((condition.name, len(condition.trial_ids), name, value, delta, percent))
    return rows


def export_csv(path, result, source_path, reference="", time_rows=None):
    """Long-form CSV: individual trials, both condition calculations and contrasts."""
    headers = ["recording", "channel", "fs_hz", "duration_s", "welch_window_s",
               "welch_overlap", "filters", "condition", "trial_ids", "n_trials",
               "calculation", "quantity", "x", "x_unit", "value", "unit",
               "reference_condition", "difference_uv2", "difference_percent",
               "trial_start_s", "trial_end_s", "welch_samples", "welch_overlap_samples",
               "feature_id", "definition", "profile", "reason", "quality_flags", "sd", "median", "q1", "q3",
               "difference", "cliffs_delta", "feature_window_s", "feature_step_s"]
    nperseg, noverlap = welch_plan(len(result.times), result.fs, result.settings)
    common = dict(recording=str(Path(source_path).resolve()), channel=result.channel,
                  fs_hz=result.fs, duration_s=len(result.times) / result.fs,
                  welch_window_s=nperseg / result.fs,
                  welch_overlap=result.settings.overlap,
                  welch_samples=nperseg, welch_overlap_samples=noverlap,
                  filters=json.dumps(result.settings.filters, sort_keys=True))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()

        def write_metrics(condition, ids, calculation, signal, metrics, trial=None):
            base = dict(common, condition=condition, trial_ids=";".join(ids),
                        n_trials=len(ids), calculation=calculation)
            if trial is not None:
                base.update(trial_start_s=trial.start, trial_end_s=trial.end)
            series = [("FFT amplitude", metrics.fft_freqs, metrics.fft, "Hz", "uV"),
                      ("PSD", metrics.psd_freqs, metrics.psd, "Hz", "uV^2/Hz")]
            if signal is not None:
                series.insert(0, ("Signal", result.times, signal, "s", "uV"))
            for quantity, xs, ys, x_unit, unit in series:
                for x, y in zip(xs, ys):
                    writer.writerow(dict(base, quantity=quantity, x=x, x_unit=x_unit,
                                         value=y, unit=unit))
            for quantity, value in {"Mean square": metrics.power, **metrics.bands}.items():
                writer.writerow(dict(base, quantity=quantity, value=value, unit="uV^2"))

        for item in result.trials:
            write_metrics(item.trial.condition, [item.trial.id], "Individual trial",
                          item.signal, item.metrics, item.trial)
        for condition in result.conditions:
            write_metrics(condition.name, condition.trial_ids, "Analysis of mean signal",
                          condition.signal, condition.mean_signal)
            write_metrics(condition.name, condition.trial_ids, "Mean of trial analyses",
                          None, condition.mean_trials)
        for calculation in ("mean_signal", "mean_trials"):
            for name, n, metric, value, delta, percent in power_comparison(
                    result.conditions, calculation, reference):
                writer.writerow(dict(common, condition=name, n_trials=n, calculation=calculation,
                                     quantity="Comparison: " + metric, value=value, unit="uV^2",
                                     reference_condition=reference, difference_uv2=delta,
                                     difference_percent=percent))
        if result.features is not None:
            from ..core.feature_profiles import summary, condition_values, cliffs_delta
            profile = result.features.profile
            specs = profile.visible_specs()
            def finite(value):
                return value if np.isfinite(value) else ""
            feature_common = dict(common, profile=json.dumps(profile.to_dict(), sort_keys=True),
                                  feature_window_s=profile.time_window, feature_step_s=profile.time_step)
            for entry in result.trials:
                values = result.features.trials[entry.trial.id]
                base = dict(feature_common, condition=entry.trial.condition, trial_ids=entry.trial.id, n_trials=1,
                            trial_start_s=entry.trial.start, trial_end_s=entry.trial.end,
                            quality_flags="; ".join(result.quality.get(entry.trial.id, [])))
                for spec in specs:
                    writer.writerow(dict(base, calculation="Trial feature", feature_id=spec.id, quantity=spec.name,
                                         definition=spec.description, value=finite(values.values[spec.id]), unit=spec.unit,
                                         reason=values.reasons.get(spec.id, "")))
                if time_rows and entry.trial.id in time_rows:
                    times, windows = time_rows[entry.trial.id]
                    for timestamp, values in zip(times, windows):
                        for spec in specs:
                            writer.writerow(dict(base, calculation="Within-trial feature", feature_id=spec.id,
                                                 quantity=spec.name, definition=spec.description, x=timestamp, x_unit="s",
                                                 value=finite(values.values[spec.id]), unit=spec.unit,
                                                 reason=values.reasons.get(spec.id, "")))
            for condition in result.conditions:
                for spec in specs:
                    vals = condition_values(result, condition.name, spec.id)
                    ref = condition_values(result, reference, spec.id)
                    stats, ref_stats = summary(vals), summary(ref)
                    base = dict(feature_common, condition=condition.name, trial_ids=";".join(condition.trial_ids),
                                feature_id=spec.id, quantity=spec.name, definition=spec.description, unit=spec.unit)
                    writer.writerow(dict(base, calculation="Condition feature summary", n_trials=stats["n"],
                                         value=finite(stats["mean"]), sd=finite(stats["sd"]), median=finite(stats["median"]),
                                         q1=finite(stats["q1"]), q3=finite(stats["q3"]), reference_condition=reference,
                                         difference=finite(stats["mean"]-ref_stats["mean"]),
                                         cliffs_delta=finite(cliffs_delta(ref,vals)) if reference!=condition.name else ""))
                    values = result.features.mean_signals[condition.name]
                    writer.writerow(dict(base, calculation="Mean signal feature", n_trials=len(condition.trial_ids),
                                         value=finite(values.values[spec.id]), reason=values.reasons.get(spec.id,"")))
