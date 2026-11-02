# Changelog

## 1.0.0 (2026-10-31)

The first public release.

- **Live monitor** for OpenBCI Cyton / Cyton+Daisy and g.tec Unicorn: signal quality, impedance, events, recording.
  Line noise is measured at the mains chosen in the filters (50 or 60 Hz).
- **Workbench**: analyses built from about 40 blocks, computed as you go, saved as `.pipeline.json`; ready-made
  pipelines and three from the literature (MDRM, TRCSP + LDA, FBCSP-MIBIF-NBPW). Each block shows only the settings
  that apply (an STFT's window with STFT, a Chebyshev's ripple with that design); hidden ones keep their values.
- **A sheet that moves freely**: no edge and no scroll bars, zoom from 5 % to 400 % easing toward the mouse, a zoom
  bar in the lower left (50-200 %, the whole pipeline, the selected blocks; keys 0, 1, 2), the sheet carried along
  when a block or a wire is dragged to its edge, and each tab opening where it was left. Settings → Interface
  chooses what a plain drag on empty sheet does: move the sheet, or select.
- **Re-reference** to the average, chosen channels or a **Laplacian** (each channel minus the mean of its four
  nearest electrodes, standard 10-05 positions), for local rhythms such as mu over C3 and C4.
- **Time-frequency**: STFT, or complex Morlet wavelets whose width is given as cycles per Hz or as one cmorB-C for
  every frequency. Each trial is transformed with the recording around it, so the edges of its map hold no
  artefact. The head map and the statistics take their window means from a separate STFT of each electrode (the
  default) or from the maps' own transform; methods, captions and Help say which, how many electrodes a head map
  is drawn from, and that their positions are the standard 10-05 template.
- **Classification** with every step fitted inside the training folds, and nested tuning whose values, inner folds
  and score (accuracy or balanced accuracy) are yours to set. Accuracy is judged against the exact binomial limit
  of a random classifier for the number of tested trials (Combrisson & Jerbi, 2015), said to be approximate under
  cross-validation, whose test trials are not independent (Noirhomme et al., 2014); a **permutation test** runs
  from under any result, with as many shuffles as you choose.
- **CSP and FBCSP after an average reference**: the filters are found in the channels' true rank.
- **Compare** pipelines over many recordings, or train on the others and test on each.
- **Reports** as PDF, Word, HTML or figures, their tables laid out as in papers (APA: blank where a value does not
  apply, a dash where it could not be computed); **Python export** that runs without GIOVA.
- **Files**: MATLAB (v7 and v7.3), EEGLAB, EDF / BDF / GDF, BrainVision, FIF and CSV recordings; events from the
  file, a marker CSV beside it (as a spreadsheet saves it too) or a paradigm written down; MNE's bad annotations
  in either case.
- **Help** with a page and a picture for every block; Help → Send feedback.
- Light and dark themes.
- **Tested**: about 650 automated tests, including ones drawn from the literature: no learning step learns from
  the test trials (Kapoor & Narayanan, 2023), the permutation test's false positive rate on noise, metamorphic
  relations (amplitude, channel order, time shift, condition names, units) and property-based round trips of
  every text field and pipeline file. They run on Windows and Linux for every change.
