"""The Workbench's blocks as Python functions, for scripts.

Every block is a function of the same name. Its positional arguments are the
block's inputs in order, its keywords the block's settings; it returns the
block's output (a tuple when it has several). The code that runs is the code
the Workbench runs, so a script and the sheet give the same numbers.

    from eeg.pipeline import api as ee
    raw = ee.load(r"s01.mat", condition="imagery_left + imagery_right")
    raw = ee.filter(raw, low=8.0, high=30.0, notch=0.0)
    trials = ee.epochs(raw, tmin=2.0, tmax=5.0)
    model = ee.lda(ee.csp(n_components=6))
    scores = ee.cross_validate(trials, model, folds=5)
    print(scores.describe())
    ee.show(scores)

`help(ee.filter)` lists a block's settings with their defaults and meaning.

No Qt. `show` needs matplotlib.
"""
from . import REGISTRY, block
from .graph import Context, PipelineError

__all__ = sorted(REGISTRY) + ["run", "show"]


def run(type_name, *inputs, **params):
    """Run one block: `run("filter", raw, low=1.0)` is `filter(raw, low=1.0)`."""
    kind = block(type_name)
    params = kind.upgrade(params)
    if len(inputs) > len(kind.inputs):
        raise TypeError("{} takes {} input(s): {}".format(
            type_name, len(kind.inputs), ", ".join(p.name for p in kind.inputs)))
    unknown = set(params) - set(kind.defaults())
    if unknown:
        raise TypeError("{} has no setting {}".format(type_name, ", ".join(sorted(unknown))))
    feed = {port.name: value for port, value in zip(kind.inputs, inputs) if value is not None}
    missing = [p.name for p in kind.inputs if not p.optional and p.name not in feed]
    if missing:
        raise TypeError("{} needs its input: {}".format(type_name, ", ".join(missing)))
    try:
        outputs = kind.run(feed, {**kind.defaults(), **params}, Context())
    except PipelineError:
        raise
    except (ValueError, KeyError, IndexError, ArithmeticError, OSError) as exc:
        raise PipelineError(type_name, "{}: {}".format(kind.title, exc)) from exc
    values = [outputs[p.name] for p in kind.outputs]
    return values[0] if len(values) == 1 else tuple(values)


def _function(type_name):
    kind = block(type_name)

    def call(*inputs, **params):
        return run(type_name, *inputs, **params)

    call.__name__ = call.__qualname__ = type_name
    lines = ["{} ({}).".format(kind.title, kind.category), "",
             "Inputs: " + (", ".join("{}{}".format(p.name, " (optional)" if p.optional else "")
                                     for p in kind.inputs) or "none"),
             "Returns: " + ", ".join(p.name for p in kind.outputs), "", "Settings:"]
    for p in kind.params:
        lines.append("    {} = {!r}{}".format(p.name, p.default, "  - " + p.help if p.help else ""))
    call.__doc__ = "\n".join(lines)
    return call


def __getattr__(name):
    if name in REGISTRY:
        return _function(name)
    raise AttributeError("module 'eeg.pipeline.api' has no attribute {!r}".format(name))


def __dir__():
    return __all__


def show(value, channel=0, kind="erd"):
    """A quick matplotlib figure of a block's output: what the Workbench shows beside it.

    For a time-frequency result, `kind` is "erd", "power" (dB) or "amplitude" (µV),
    one map per condition, averaged over its trials.
    """
    from ..core.epochs import EpochResult
    from ..core.sources import Recording
    from .data import Epochs, Features, Spectrum
    from .learn import Scores
    if isinstance(value, EpochResult):
        return show_tf(value, channel, kind)
    for kinds, draw in ((Scores, show_scores), (Spectrum, show_spectrum), (Epochs, show_epochs),
                        (Recording, show_recording), (Features, show_features)):
        if isinstance(value, kinds):
            return draw(value) if draw in (show_scores, show_features) else draw(value, channel)
    raise TypeError("show() draws recordings, trials, spectra, features, scores and time-frequency maps")


def _shown(figure):
    import matplotlib.pyplot as plt
    figure.tight_layout()
    plt.show()
    return figure


def show_scores(scores):
    """Accuracy per fold, titled with the verdict against a random classifier."""
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(np.arange(1, len(scores.fold_accuracy) + 1), scores.fold_accuracy, color="#1b2140")
    axis.set(xlabel="Fold", ylabel="Accuracy", ylim=(0, 1), title=scores.random_text())
    return _shown(figure)


def show_spectrum(spectrum, channel=0):
    """One channel's spectrum, averaged per condition."""
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = np.asarray(spectrum.labels)
    for condition in dict.fromkeys(spectrum.labels):
        axis.semilogy(spectrum.freqs, spectrum.power[labels == condition, channel].mean(axis=0), label=condition)
    axis.set(xlabel="Frequency (Hz)", ylabel="{} ({})".format(spectrum.title, spectrum.unit),
             xlim=(0, min(60, spectrum.freqs[-1])), title=spectrum.names[channel])
    axis.legend()
    return _shown(figure)


def show_epochs(epochs, channel=0):
    """One channel's trials, averaged per condition."""
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = np.asarray(epochs.labels)
    for condition in dict.fromkeys(epochs.labels):
        axis.plot(epochs.times, epochs.data[labels == condition, channel].mean(axis=0), label=condition)
    axis.set(xlabel="Time from event (s)", ylabel="Mean (µV)", title=epochs.names[channel])
    axis.legend()
    return _shown(figure)


def show_recording(recording, channel=0):
    """The first 10 s of one channel."""
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axis = plt.subplots(figsize=(8, 4))
    n = min(recording.n_samples, int(10 * recording.fs))
    axis.plot(np.arange(n) / recording.fs, recording.data[channel, :n])
    axis.set(xlabel="Time (s)", ylabel="µV", title=recording.channel_names[channel])
    return _shown(figure)


def show_features(features):
    """Each feature's mean per condition."""
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = np.asarray(features.labels)
    conditions = list(dict.fromkeys(features.labels))
    means = [features.values[labels == c].mean(axis=0) for c in conditions]
    width = 0.8 / len(conditions)
    for k, (condition, mean) in enumerate(zip(conditions, means)):
        axis.bar(np.arange(len(features.names)) + k * width, mean, width, label=condition)
    axis.set_xticks(np.arange(len(features.names)), features.names, rotation=90)
    axis.legend()
    return _shown(figure)


def show_tf(result, channel=0, kind="erd"):
    """One channel's time-frequency map per condition: "erd", "power" (dB) or "amplitude" (µV)."""
    import matplotlib.pyplot as plt
    from ..core import epochs as ep
    tf = result.tf[channel]
    figure, axes = plt.subplots(1, len(result.conditions), figsize=(5 * len(result.conditions), 3.6), squeeze=False)
    unit = {ep.ERD_MAP: "ERD / ERS (%)", ep.POWER_MAP: "dB re 1 " + tf.units, ep.AMPLITUDE_MAP: "µV"}[kind]
    for axis, condition in zip(axes[0], result.conditions):
        values = ep.condition_map(result, tf, condition, kind)
        if kind == ep.ERD_MAP:
            bound = max(float(abs(values[values == values]).max()), 1e-9)
            options = dict(cmap="RdBu", vmin=-bound, vmax=bound)
        else:
            options = dict(cmap="viridis", vmin=0 if kind == ep.AMPLITUDE_MAP else None)
        image = axis.pcolormesh(tf.times, tf.freqs, values, shading="nearest", **options)
        for edge in result.settings.active:
            axis.axvline(edge, color="crimson", ls="--", lw=1.1)
        axis.set(xlabel="Time from event (s)", ylabel="Frequency (Hz)",
                 title="{} · {} (n = {})".format(tf.name, condition, len(result.epoch_ids[condition])))
        figure.colorbar(image, ax=axis, label=unit)
    figure.tight_layout()
    plt.show()
    return figure
