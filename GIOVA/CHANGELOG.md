# Changelog

## Unreleased

- **Re-reference: Laplacian.** Each channel minus the mean of its four nearest electrodes (standard 10-05
  positions), for local rhythms such as mu over C3 and C4. In the report, the Python export and Help.
- **Fixed: Morlet and CWT maps at the edges of a trial.** Each trial was transformed on its own, so the wavelets
  saw zeros past its ends; the edges swelled and, with a baseline near them, the whole map read as a drop. Each
  trial now sees the recording around it (mirrored at the recording's ends).
- **Fixed: overlapping tick labels** on log axes (e.g. "7 8 9 10" printed over each other). Every plot now keeps
  a label only where it fits, decades first.
- **Spectra are drawn to 70 Hz** (was 60), so a 50 or 60 Hz mains peak stands clear of the edge.

## 1.0.0 (2026-10-31)

The first public release.

- **Live monitor** for OpenBCI Cyton / Cyton+Daisy and g.tec Unicorn: signal quality, impedance, events, recording.
- **Workbench**: analyses built from about 40 blocks, computed as you go, saved as `.pipeline.json`; ready-made
  pipelines and three from the literature (MDRM, TRCSP + LDA, FBCSP-MIBIF-NBPW).
- **Classification** with every step fitted inside the training folds, nested tuning whose values, inner folds
  and score (accuracy or balanced accuracy) are yours to set, chance levels for the number of trials, and an
  optional permutation test.
- **CSP and FBCSP after an average reference**: the filters are found in the channels' true rank.
- **Compare** pipelines over many recordings, or train on the others and test on each.
- **Reports** as PDF, Word, HTML or figures; **Python export** that runs without GIOVA.
- **Help** with a page and a picture for every block; Help → Send feedback.
- Light and dark themes.
