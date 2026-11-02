"""Pipelines from published papers, kept ready to compare a method against: a library.

Each profile is an ordinary pipeline built from the Workbench's blocks, with a `source`
that says which paper it comes from, how far it has been checked, and every way it
differs from the paper - the dataset, the trial window, the evaluation. A profile is
not the paper's result: its accuracy on your recordings is what Compare measures.

The profiles are set up for Cho et al. (2017), the motor imagery dataset this project
validates on (cue 2 s after each trial's event). On another dataset, set the Epochs
window to 0.5-2.5 s after that dataset's cue. See docs/LITERATURE_PIPELINES_MI.md.

Status, from least to most: "design" (built from the paper, not yet compared with
anything), "checked against its formula", "matches a reference implementation" (same
predictions as an independent implementation on the same folds). None of them says the
paper's accuracy was reproduced.
"""
from .graph import Graph
from .templates import CHO2017_IMAGERY

#: Cho2017: the cue comes 2 s after each trial's event; the papers analyse 0.5-2.5 s after the cue.
CUE = 2.0
WINDOW = (0.5, 2.5)


def _trials(graph, path, condition):
    """Recording → 8-30 Hz 5th-order Butterworth → trials 0.5-2.5 s after the cue: both papers' signal path."""
    *_, epochs = graph.chain(("load", {"path": path, "condition": condition}),
                             ("filter", {"low": 8.0, "high": 30.0, "notch": 0.0, "order": 5, "phase": "causal"}),
                             ("epochs", {"tmin": CUE + WINDOW[0], "tmax": CUE + WINDOW[1]}))
    return epochs


def _finish(graph, epochs, model):
    cv = graph.add("cross_validate")
    graph.connect(epochs, "epochs", cv, "data")
    graph.connect(model, "model", cv, "model")
    graph.arrange()
    graph.source["fingerprint"] = graph.method_fingerprint()
    return graph


COMMON = ["Data: Cho et al. (2017) left vs right hand imagery, all EEG channels, instead of the paper's "
          "BCI Competition datasets.",
          "Trials 0.5-2.5 s after the cue (2.5-4.5 s from Cho2017's trial event).",
          "Filter: 5th-order Butterworth band-pass 8-30 Hz as in the paper, applied forward only (causal).",
          "Evaluation: stratified 5-fold cross-validation within each recording, not the paper's "
          "separate training and test sessions."]


def mdrm(path="", condition=CHO2017_IMAGERY):
    graph = Graph("Barachant 2012 · MDRM",
                  "Minimum distance to the Riemannian mean of each condition's trial covariances.",
                  dict(id="MI/Barachant2012/MDRM",
                       citation="Barachant A, Bonnet S, Congedo M, Jutten C (2012). Multiclass brain-computer "
                                "interface classification by Riemannian geometry. IEEE Trans Biomed Eng "
                                "59(4):920-928.",
                       doi="10.1109/TBME.2011.2172210",
                       status="matches a reference implementation: pyRiemann Covariances('scm') → MDM gives the "
                              "same predictions on the same folds (tests/test_literature.py)",
                       adaptations=COMMON + ["Two conditions; the paper classifies four.",
                                             "Sample covariance (SCM) of each trial, as in the paper; the "
                                             "Riemann + logistic regression template uses OAS instead."]))
    epochs = _trials(graph, path, condition)
    *_, model = graph.chain(("covariance", {"estimator": "scm"}), "mdm")
    return _finish(graph, epochs, model)


def trcsp_lda(path="", condition=CHO2017_IMAGERY):
    graph = Graph("Lotte & Guan 2011 · TRCSP + LDA",
                  "Tikhonov-regularised CSP, 3 filters per condition, alpha tuned inside each fold, plain LDA.",
                  dict(id="MI/Lotte2011/TRCSP-LDA",
                       citation="Lotte F, Guan C (2011). Regularizing common spatial patterns to improve BCI "
                                "designs: unified theory and new algorithms. IEEE Trans Biomed Eng "
                                "58(2):355-362.",
                       doi="10.1109/TBME.2010.2082539",
                       status="checked against its formula: the filters solve the paper's penalised ratio, and "
                              "with a vanishing alpha they span plain CSP's (tests/test_literature.py)",
                       adaptations=COMMON + ["Trial covariances normalised by their trace, so alpha (10^-10 to "
                                             "10^-1, chosen by 10-fold cross-validation inside each training "
                                             "fold, as in the paper) does not depend on the µV scale.",
                                             "LDA without shrinkage, as in the paper."]))
    epochs = _trials(graph, path, condition)
    *_, model = graph.chain(("trcsp", {"n_filters": 3}), ("lda", {"shrinkage": "none"}))
    return _finish(graph, epochs, model)


#: FBCSP's trials start this long before the analysed window: the filter bank's filters settle in it.
#: Their impulse responses fall to 0.1% in about 3.8 s; with 4 s the kept part is within 0.3% of
#: filtering the continuous recording (tests/test_literature.py).
SETTLE = 4.0
#: Cho2017 is recorded at 512 Hz; nine bands of 64 channels at that rate would hold over 900 MB of trials.
FBCSP_RATE = 128.0


def fbcsp(path="", condition=CHO2017_IMAGERY):
    graph = Graph("Ang 2012 · FBCSP-MIBIF-NBPW",
                  "Nine 4 Hz Chebyshev II bands from 4 to 40 Hz, CSP in each, the most informative features by "
                  "mutual information, a naive Bayesian Parzen window classifier.",
                  dict(id="MI/Ang2012/FBCSP-MIBIF-NBPW",
                       citation="Ang KK, Chin ZY, Wang C, Guan C, Zhang H (2012). Filter bank common spatial "
                                "pattern algorithm on BCI Competition IV datasets 2a and 2b. Front Neurosci 6:39.",
                       doi="10.3389/fnins.2012.00039",
                       status="checked against its formulas: mutual information and the Parzen densities match "
                              "independent computations (SciPy gaussian_kde), the filter bank inside each trial "
                              "matches filtering the continuous recording once its filters have settled "
                              "(tests/test_literature.py); not compared with the FBCSP Toolbox",
                       adaptations=[COMMON[0], COMMON[1], COMMON[3],
                                    "Filter bank: Chebyshev II, causal, order 4 and 40 dB stop band; the paper names "
                                    "the filter type, not these values.",
                                    "Each trial is filtered from {:g} s before the analysed window, which is then "
                                    "kept, instead of filtering the continuous recording (within 1% of it); a trial "
                                    "whose lead-in would start before the recording is left out.".format(SETTLE),
                                    "Downsampled to {:g} Hz first (the bands end at 40 Hz).".format(FBCSP_RATE),
                                    "Two conditions; the paper's four-class version combines one-versus-rest "
                                    "classifiers, which is not built.",
                                    "Mutual information's conditional entropy averaged over trials, not summed: "
                                    "the same ranking."]))
    start, end = CUE + WINDOW[0], CUE + WINDOW[1]
    *_, bank = graph.chain(("load", {"path": path, "condition": condition}),
                           ("resample", {"target": FBCSP_RATE}),
                           ("epochs", {"tmin": start - SETTLE, "tmax": end}),
                           ("filter_bank", {"start": start, "end": end}))
    *_, model = graph.chain(("fbcsp", {"n_bands": 9, "n_pairs": 2}), ("mibif", {"k": 4, "pair_size": 4}), "nbpw")
    return _finish(graph, bank, model)


#: title -> builder(path="", condition=...), as templates.TEMPLATES
LIBRARY = {"Barachant 2012 · MDRM (Riemannian MDM)": mdrm,
           "Lotte & Guan 2011 · TRCSP + LDA": trcsp_lda,
           "Ang 2012 · FBCSP-MIBIF-NBPW": fbcsp}
