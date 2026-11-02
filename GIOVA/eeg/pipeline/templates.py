"""Ready-made pipelines: what the Workbench opens with instead of an empty sheet.

Timings follow Cho et al. (2017): each 7 s trial starts 2 s before the cue and
imagery lasts until 5 s, so the analysed window is 2-5 s from the trial's event
(0-3 s after the cue, the interval MOABB uses for this dataset).
"""
from .graph import Graph

#: the part of a Cho2017 file holding both imagery sides, one event per trial
CHO2017_IMAGERY = "imagery_left + imagery_right"
MOTOR = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]


def csp_lda(path="", condition=CHO2017_IMAGERY):
    graph = Graph("CSP + LDA", "Left vs right hand imagery: 8-30 Hz, CSP (6 filters), shrinkage LDA, "
                               "5-fold cross-validation.")
    load, _, epochs = graph.chain(("load", {"path": path, "condition": condition}),
                                  ("filter", {"low": 8.0, "high": 30.0, "notch": 0.0}),
                                  ("epochs", {"tmin": 2.0, "tmax": 5.0}))
    _, lda = graph.chain(("csp", {"n_components": 6}), "lda")
    cv = graph.add("cross_validate")
    graph.connect(epochs, "epochs", cv, "data")
    graph.connect(lda, "model", cv, "model")
    return graph.arrange()


def band_power_lda(path="", condition=CHO2017_IMAGERY):
    graph = Graph("Band power + LDA", "Left vs right hand imagery: mu and beta log band power over the "
                                      "motor cortex, standardised, shrinkage LDA, 5-fold cross-validation.")
    *_, features = graph.chain(("load", {"path": path, "condition": condition}),
                               ("filter", {"low": 1.0, "high": 40.0, "notch": 0.0}),
                               ("channels", {"channels": list(MOTOR)}),
                               ("epochs", {"tmin": 2.0, "tmax": 5.0}),
                               ("psd", {"window": 1.0}),
                               "band_power")
    _, lda = graph.chain("standardize", "lda")
    cv = graph.add("cross_validate")
    graph.connect(features, "features", cv, "data")
    graph.connect(lda, "model", cv, "model")
    return graph.arrange()


def riemann(path="", condition=CHO2017_IMAGERY):
    graph = Graph("Riemann + logistic regression", "Left vs right hand imagery: 8-30 Hz, covariance per "
                                                  "trial, tangent space, logistic regression with C tuned "
                                                  "inside each fold, 5-fold cross-validation.")
    *_, epochs = graph.chain(("load", {"path": path, "condition": condition}),
                             ("filter", {"low": 8.0, "high": 30.0, "notch": 0.0}),
                             ("epochs", {"tmin": 2.0, "tmax": 5.0}))
    *_, classifier = graph.chain(("covariance", {"estimator": "oas"}), "tangent", ("logistic", {"tune": True}))
    cv = graph.add("cross_validate")
    graph.connect(epochs, "epochs", cv, "data")
    graph.connect(classifier, "model", cv, "model")
    return graph.arrange()


def band_power(path=""):
    graph = Graph("Band power per condition", "Filter, trials around every event, Welch PSD and "
                                              "log band power per channel and trial.")
    graph.chain(("load", {"path": path}), ("filter", {"low": 1.0, "high": 40.0}),
                ("epochs", {"tmin": 0.0, "tmax": 4.0}), ("psd", {"window": 1.0}),
                ("band_power", {"bands": [["theta", 4.0, 8.0], ["alpha", 8.0, 13.0], ["beta", 13.0, 30.0]]}))
    return graph.arrange()


def erd(path=""):
    graph = Graph("ERD / ERS", "Filter, then time-frequency power and ERD / ERS around every event.")
    graph.chain(("load", {"path": path}), ("filter", {"low": 1.0, "high": 40.0}), "erd")
    return graph.arrange()


def trial_lab(path=""):
    """Cho2017's lab walk-through on one channel: look at the trials, keep the good ones, then FFT, Welch,
    STFT and wavelet scalogram of each kept trial, averaged, wavelet denoising and the DWT's bands. The same
    steps and settings as `run_pipeline_trials.py` of a lab's Python conversion (not public) (subject 1, C3,
    right-hand imagery)."""
    graph = Graph("Cho2017 trials: C3, right hand", "Right-hand imagery on C3: whole 7 s trials, baseline 0-2 s, "
                                                   "ideal 8-40 Hz band-pass per trial, the good trials picked by "
                                                   "hand, then FFT, Welch PSD, STFT and complex Morlet wavelet of each "
                                                   "trial, averaged; db4 wavelet denoising; DWT bands.")
    *_, kept = graph.chain(("load", {"path": path, "condition": "imagery_right"}),
                           ("channels", {"channels": ["C3"]}),
                           ("epochs", {"tmin": 0.0, "tmax": 7.0}),
                           ("baseline", {"start": 0.0, "end": 2.0}),
                           ("trial_filter", {"low": 8.0, "high": 40.0, "notch": 0.0, "design": "ideal"}),
                           ("select_trials", {"keep": ["1-3", "5", "7", "8"]}))
    maps = {"channels": ["C3"], "baseline": [0.0, 2.0], "active": [2.0, 5.0], "fmin": 1.0, "fmax": 45.0,
            "smooth": 0.0}
    for type_name, params, port in (("fft", {}, "epochs"), ("psd", {"window": 2.0, "overlap": 0.75}, "epochs"),
                                    ("erd", dict(maps, window=1.0, overlap=0.5), "raw"),
                                    ("erd", dict(maps, method="wavelet", width="one wavelet"), "raw"),
                                    ("dwt_bands", {"level": 6}, "epochs")):
        graph.connect(kept, "epochs", graph.add(type_name, params), port)
    # band-pass, then wavelet: what the wavelet removes on top of the filter (the lab's FIG 6)
    denoised = graph.add("wavelet_denoise", {"level": 4, "mode": "hard", "threshold": 10.0})
    graph.connect(kept, "epochs", denoised, "epochs")
    graph.connect(denoised, "epochs", graph.add("fft"), "epochs")
    return graph.arrange()


TEMPLATES = {"Band power per condition": band_power,
             "ERD / ERS": erd,
             "Cho2017 trials: look, pick, STFT, wavelets": trial_lab,
             "Motor imagery: CSP + LDA": csp_lda,
             "Motor imagery: band power + LDA": band_power_lda,
             "Motor imagery: Riemann + logistic regression": riemann}
