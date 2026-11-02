"""The Help window's block library: one short page per Workbench block.

Each page has the same parts: what the block does, what goes in and comes out, what it is for,
what it cannot do, its settings and the templates that use it. The words are written here (BLOCKS);
the ports, settings and templates are read from the blocks themselves, so a page never disagrees
with the block it describes. Every registered block must have an entry (tests/test_help.py).
"""
import html

from ..pipeline import graph as g
from ..pipeline.blocks import CATEGORIES, GROUPS
from ..pipeline.literature import LIBRARY
from ..pipeline.templates import TEMPLATES

#: Templates named under "See it in"; a block every pipeline has would list them all.
SHOW_EXAMPLES = 3

#: type -> (what it does, use it for, it cannot / watch out)
BLOCKS = {
    # input
    "load": ("Reads one recording file: its channels, sampling rate and events.",
             ["The start of every pipeline.",
              "Renaming events (Primary/P 1 → odour) or moving them by a stimulator's delay, in Options.",
              "A MATLAB file of any layout: 'How this file is read…' chooses its signal and events, once for "
              "every file laid out alike."],
             ["Read several files at once: use Compare for many recordings.",
              "Guess units the file does not state: check the µV note in its result, and use Units / gain if wrong."]),
    # preprocessing
    "units": ("Converts the samples to µV when the file holds volts, millivolts or amplified values.",
              ["A file whose amplitudes look a thousand or a million times off.",
               "Recordings made through an amplifier with a known gain."],
              ["Detect the right unit by itself: you state what the numbers are.",
               "Fix a file already read in µV: a second conversion makes it wrong."]),
    "channels": ("Keeps the channels named, in that order; the rest are dropped.",
                 ["Working on a region (C3, Cz, C4) or leaving out EOG and accelerometer channels.",
                  "Smaller, faster pipelines over many recordings."],
                 ["Bring back a dropped channel later in the pipeline.",
                  "Keep a full head map: for that, use Channels of trials after Epochs."]),
    "trial_channels": ("Keeps the channels named, after the trials are cut.",
                       ["Spectra or features of a few channels while the head map still has every electrode."],
                       ["Work on a recording: it takes trials. Before Epochs, use Channels."]),
    "bad_channels": ("Rebuilds channels that did not record EEG from their neighbours, or removes them.",
                     ["A loose or noisy electrode (P8, say) that would spoil an average reference or a head map."],
                     ["Find bad channels by itself: you name them, after looking at the Recording's result.",
                      "Interpolate a channel without a standard 10-20 / 10-05 name, or with fewer than three good "
                      "channels around it."]),
    "reference": ("Re-references every channel: to the mean of all channels, to the channels named, or to its "
                  "four nearest electrodes (Laplacian).",
                  ["An average reference before a head map.",
                   "Linked mastoids (TP9, TP10) or any other reference the study uses.",
                   "A Laplacian for local rhythms: mu and beta over C3 and C4 in motor imagery."],
                  ["Undo a reference applied in the amplifier.",
                   "Average reference fairly with a bad channel in: rebuild it with Bad channels first.",
                   "Laplacian channels without a 10-20 / 10-05 name; at the edge of the cap it is less exact, "
                   "its neighbours lying on one side."]),
    "filter": ("Filters the whole recording: band-pass, band-stop, and a mains notch.",
               ["Removing drift (low cut) and muscle or mains noise (high cut, notch) before trials are cut.",
                "Zero-phase for offline work, or causal to see what a live system would get."],
               ["Filter trials one by one: use Filter trials.",
                "Remove artefacts that share the EEG's frequencies (blinks under 4 Hz): see ICA."]),
    "resample": ("Lowers the sampling rate by a whole factor, after an anti-alias low-pass.",
                 ["Long or many recordings: less memory and faster blocks after it."],
                 ["Raise the rate.", "Keep frequencies above half the new rate."]),
    "ica": ("Splits the recording into independent components and removes those that follow the eye channels.",
            ["Blinks and eye movements in the frontal channels."],
            ["Clean muscle or movement artefacts reliably.",
             "Work without EOG or frontal channels unless you name the components by hand.",
             "Be part of the cross-validation: it is fitted on the whole recording, test trials included."]),
    # trials
    "epochs": ("Cuts the recording into trials: around events, from a trial table, from a labelled sequence, or "
               "every few seconds.",
               ["Every analysis by condition: spectra, ERD, features, classification.",
                "Leaving out trials with too large a signal or too much head movement.",
                "Files marked in MNE or another program: a trial over a span marked BAD_… or a boundary is left "
                "out, with the mark as its reason."],
               ["Cut trials of different lengths: every trial has the same span.",
                "Know whether the events are on time: a trigger delay must be set in the Recording block."]),
    "window": ("Keeps a part of every trial, from one time to another.",
               ["The active part of an imagery trial (0.5-2.5 s after the cue) before CSP or features."],
               ["Reach outside the trials: cut longer trials in Epochs first."]),
    "trial_filter": ("The Filter block's filter applied to each trial and channel on its own.",
                     ["Trials stored end to end (Cho2017), so no trial rings into the next."],
                     ["Remove slow drift well from short trials: a 1 Hz cut needs seconds to settle. Filter the "
                      "recording when you can."]),
    "select_trials": ("Keeps the trials named by number; the rest are left out.",
                      ["Leaving out trials you judged bad after looking at them (click them in its result)."],
                      ["Choose trials by itself: you decide which go.",
                       "Leave out trials blind to the outcome if you pick them after seeing the scores."]),
    "wavelet_denoise": ("Removes small wavelet coefficients from each trial's detail levels and rebuilds it.",
                        ["Reducing broadband noise while keeping the slow part of the signal."],
                        ["Remove blinks or other large artefacts: they survive the threshold.",
                         "Touch the lowest band (the approximation): it is never thresholded."]),
    "baseline": ("Subtracts each trial's mean over a window, channel by channel.",
                 ["ERPs: every trial measured from the same pre-stimulus level."],
                 ["Correct power: for ERD, the Time-frequency block has its own baseline.",
                  "Use a window outside the trials."]),
    # spectral
    "dwt_bands": ("Splits each trial into dyadic wavelet bands (0-4, 4-8, 8-16 … Hz at 512 Hz).",
                  ["Band envelopes over time, and each band's share of the energy as features."],
                  ["Choose band edges: they halve with each level and depend on the sampling rate."]),
    "filter_bank": ("Filters every trial into several bands, one after another, for FBCSP.",
                    ["Filter bank CSP (Ang et al., 2012)."],
                    ["Feed blocks other than FBCSP sensibly: its channels are band copies of each channel."]),
    "psd": ("Welch power spectral density of every trial (µV²/Hz).",
            ["Band power, peak frequencies, comparing conditions' spectra."],
            ["Show when power changes within a trial: use Time-frequency / ERD.",
             "Resolve finer than 1 / segment length (1 s: 1 Hz)."]),
    "fft": ("Amplitude spectrum of every trial (µV), one FFT of the whole trial.",
            ["Seeing the amplitude of a rhythm, as teaching scripts compute it."],
            ["Feed Band power: band power integrates a density, so use Welch PSD."]),
    "erd": ("Time-frequency map of a recording or of trials: ERD / ERS, power or amplitude over time.",
            ["Motor imagery: when and where mu and beta power drop.",
             "Head maps over time, and one image per trial (Save every trial…)."],
            ["Be wired to a classifier: it is for looking, not for features.",
             "Give a reliable ERD from a baseline with almost no power: percentages then grow very large."]),
    # features and statistics
    "band_power": ("Power in named bands from a Welch spectrum, one number per band and channel.",
                   ["Simple features for LDA (mu, beta at C3 and C4)."],
                   ["Take an FFT: it needs Welch PSD."]),
    "features": ("Signal features per trial: band power, ratios, RMS, Hjorth, entropy, peak frequency.",
                 ["Describing trials or conditions with interpretable numbers.",
                  "Features for a classifier."],
                 ["Choose features for you: pick them, or put Select features in the model."]),
    "compare_windows": ("Compares the band power of two windows of the same trials, channel by channel, "
                        "with a paired test and Holm's adjustment.",
                        ["Stimulus against rest (odour 0-2 s against air 4-10 s)."],
                        ["Compare two conditions or two groups of people: both windows come from the same trials.",
                         "Work with fewer than 6 trials."]),
    # machine learning
    "augment": ("Adds noisy, rescaled and shifted copies of the training trials, inside each fold.",
                ["Few trials, as a test of whether more data would help."],
                ["Make new information: copies only vary what is already there.",
                 "Touch the test trials: they are never copied."]),
    "csp": ("Common spatial patterns: channel weightings whose power differs most between two conditions.",
            ["Motor imagery, left against right hand.",
             "Average-referenced channels: it works in their rank, one direction fewer, so at most "
             "channels − 1 components."],
            ["Classify on its own: a classifier must follow.",
             "Separate more than two conditions."]),
    "trcsp": ("CSP with a penalty on large weights, tuned inside each training fold (Lotte & Guan, 2011).",
              ["Many channels and few trials, where plain CSP overfits."],
              ["Separate more than two conditions.", "Classify on its own."]),
    "fbcsp": ("CSP in each band of a Filter bank; the first and last filters of every band (Ang et al., 2012).",
              ["Motor imagery whose best band differs from person to person."],
              ["Run without a Filter bank before the trials.", "Separate more than two conditions."]),
    "standardize": ("Scales every feature to mean 0 and standard deviation 1, learned on the training trials.",
                    ["Features on different scales (power and ratios) before SVM, kNN or logistic regression."],
                    ["Tame an outlier trial: one very large trial still sets the spread."]),
    "select": ("Keeps the features whose ANOVA F between conditions is largest on the training trials.",
               ["Many features and few trials."],
               ["Find features that only work together: each is judged alone."]),
    "mibif": ("Keeps the features with the most mutual information with the condition, with their CSP partners.",
              ["The FBCSP-MIBIF-NBPW method (Ang et al., 2012)."],
              ["Work on features other than FBCSP's."]),
    "covariance": ("Each trial's channel covariance matrix, shrunk so it stays usable with many channels.",
                   ["The first step of every Riemannian pipeline."],
                   ["Classify on its own: follow it with MDM, or Tangent space and a classifier."]),
    "tangent": ("Maps covariance matrices to ordinary feature vectors on the tangent space.",
                ["Riemannian features for logistic regression, SVM or LDA."],
                ["Run without Covariance before it."]),
    "lda": ("Linear discriminant analysis, with Ledoit-Wolf shrinkage by default.",
            ["CSP or band power features: fast, robust, the usual BCI baseline."],
            ["Draw curved boundaries between conditions."]),
    "svm": ("Support vector machine, linear or rbf kernel.",
            ["Many features; an rbf kernel for curved boundaries (with Tune).",
             "Matching a published setting: C, and gamma for rbf, typed in, or the values Tune searches "
             "set in Options."],
            ["Choose C well without Tune: the default is a guess."]),
    "logistic": ("Logistic regression with an l1 or l2 penalty.",
                 ["Tangent-space features; l1 when only a few features should count."],
                 ["Draw curved boundaries between conditions."]),
    "forest": ("Random forest of decision trees.",
               ["Many mixed features; no scaling needed.",
                "Shallower trees or larger leaves (Options) when it fits the training trials too closely."],
               ["Do well with very few trials and many features.",
                "Give the same result with another seed: it is random."]),
    "knn": ("Classifies a trial by its nearest training trials.",
            ["A simple check of whether the conditions form clusters."],
            ["Cope with unscaled features: put Standardize before it."]),
    "mdm": ("Minimum distance to mean: each trial goes to the condition whose mean covariance is nearest "
            "(Riemannian distance).",
            ["Covariance matrices directly (Barachant et al., 2012)."],
            ["Take features: it needs Covariance before it."]),
    "nbpw": ("Naive Bayes with Parzen window densities (Ang et al., 2012).",
             ["The FBCSP-MIBIF-NBPW method."],
             ["Model features that depend on each other: each is treated as independent."]),
    "cross_validate": ("Trains the model on part of the trials and tests it on the rest, fold after fold; "
                       "accuracy, balanced accuracy, Cohen's κ, F1, ROC AUC and the confusion matrix.",
                       ["Asking whether the trials hold information about the condition.",
                        "Comparing pipelines on the same folds (Compare)."],
                       ["Say that the result works for a new person: that is Compare's Across recordings, one file per "
                        "person.",
                        "Prove significance from the random limit alone: run a permutation test (Options).",
                        "Keep trials of one block or session together in a fold: trials are treated as "
                        "independent."]),
}


def ports(kind, which):
    """The block's inputs or outputs, each as a coloured kind of data."""
    from .workbench.canvas import TYPE_COLOURS, TYPE_NAMES
    out = []
    for port in getattr(kind, which):
        names = " or ".join('<span style="color:{}">{}</span>'.format(TYPE_COLOURS.get(t, ""), TYPE_NAMES.get(t, t))
                            for t in port.types)
        out.append(names + (" (optional)" if port.optional else ""))
    return ", ".join(out) or "nothing"


def used_in(type_name):
    """Templates and literature pipelines that contain the block."""
    found = []
    for library in (TEMPLATES, LIBRARY):
        for name, make in library.items():
            try:
                graph = make()
            except Exception:           # a template that cannot be built is no reason to break Help
                continue
            if any(node.type == type_name for node in graph.nodes.values()):
                found.append(name)
    return found


def _items(lines):
    return "<ul>" + "".join("<li>{}</li>".format(html.escape(line)) for line in lines) + "</ul>"


def _only_with(kind, param):
    """For a setting the Workbench shows only with certain values of others: " (with Method stft)", " (not with
    Method wavelet, Head map and statistics from maps)"; else ""."""
    labels = {p.name: p.label or p.name for p in kind.params}

    def said(value):
        return {True: "on", False: "off"}[value] if isinstance(value, bool) else str(value)

    def words(conditions):
        return ", ".join("{} {}".format(labels.get(name, name), " or ".join(said(v) for v in values))
                         for name, values in conditions or ())
    out = " <i>(with {})</i>".format(html.escape(words(param.when))) if param.when else ""
    return out + (" <i>(not with {})</i>".format(html.escape(words(param.unless))) if param.unless else "")


def page(type_name):
    """The block's Help page."""
    kind = g.block(type_name)
    does, use, cannot = BLOCKS[type_name]
    parts = ["<h2>{}</h2>".format(html.escape(kind.title)),
             '<p class="note">{}</p>'.format(html.escape(GROUPS.get(kind.category, "") and
                                                         GROUPS[kind.category] + " › ") + html.escape(kind.category)),
             "<p>{}</p>".format(html.escape(does)),
             "<table><tr><th>In</th><td>{}</td></tr><tr><th>Out</th><td>{}</td></tr></table>".format(
                 ports(kind, "inputs"), ports(kind, "outputs")),
             "<h3>Use it for</h3>", _items(use),
             "<h3>It cannot</h3>", _items(cannot)]
    if kind.params:
        rows = "".join("<tr><td>{}</td><td>{}</td></tr>".format(
            html.escape(p.label or p.name) + (" <i>(Options)</i>" if p.advanced else "") + _only_with(kind, p),
            html.escape(p.help or ""))
            for p in sorted(kind.params, key=lambda p: p.advanced))
        parts += ["<h3>Settings</h3>", "<table><tr><th>Setting</th><th>What it does</th></tr>{}</table>".format(rows)]
    examples = used_in(type_name)
    if examples:
        shown = "; ".join(html.escape(e) for e in examples[:SHOW_EXAMPLES])
        if len(examples) > SHOW_EXAMPLES:
            shown += " and {} more".format(len(examples) - SHOW_EXAMPLES)
        parts += ["<h3>See it in</h3>", "<p>Library → {}</p>".format(shown)]
    return "\n".join(parts)


def by_category():
    """[(heading, category, [type, ...])] in the block list's order; heading is the group above, or ''."""
    out = []
    for category in CATEGORIES:
        types = [t for t, kind in g.REGISTRY.items() if kind.category == category and t in BLOCKS]
        if types:
            out.append((GROUPS.get(category, ""), category, types))
    return out


def overview(categories=None):
    """A table of blocks and what each does: all of them, or those of `categories`."""
    rows = []
    for _, category, types in by_category():
        if categories and category not in categories:
            continue
        rows.append('<tr><th colspan="2">{}</th></tr>'.format(html.escape(category)))
        rows += ["<tr><td>{}</td><td>{}</td></tr>".format(html.escape(g.block(t).title), html.escape(BLOCKS[t][0]))
                 for t in types]
    return "<table>{}</table>".format("".join(rows))


LIBRARY_PAGE = """<h2>Block library</h2>
<p>Every Workbench block: what it does, what goes in and out, what it is for and what it cannot do.
Pick one on the left, or right-click a block on the sheet → <b>Help</b>.</p>
""" + "{}"
