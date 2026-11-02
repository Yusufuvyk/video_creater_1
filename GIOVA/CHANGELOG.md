# Changelog

## Unreleased

- **Time-frequency: the head map and the statistics from either transform.** A new setting (Head map and
  statistics from) takes their window means from a separate STFT of each electrode (fast, the default, the same
  numbers as before) or from the maps' own transform. Their source is said in the methods, the captions and Help;
  the head map counts the electrodes it is drawn from, says that the positions are the standard 10-05 template,
  and gives its power in its own units (it said µV² for an STFT density under the wavelet method).
- **Empty table cells as in papers (APA)**: blank where a value does not apply, a dash where it could not be
  computed, explained beside the table.
- **The Workbench's sheet moves freely.** It has no edge and no scroll bars; zooming eases toward the mouse, from
  5 % to 400 %. A zoom bar in its lower left corner shows the level and offers 50-200 %, the whole pipeline and the
  selected blocks (keys 0, 1, 2). A block or a wire dragged to the sheet's edge moves it along, and each tab opens
  where it was left. Settings → Interface chooses what a plain drag on empty sheet does: move the sheet, or select.
- **Time-frequency: STFT and Wavelet.** Morlet and CWT were one transform, complex Morlet wavelets, offered as two
  methods. Now there is one Wavelet method; its width is given as cycles per Hz, or as any complex Morlet cmorB-C
  for every frequency (cmor1.5-1.0 spans 5.44 cycles). Pipelines saved with morlet or cwt open as these and give
  the same numbers.
- **The Workbench shows only the settings that apply**: an STFT's window with STFT, a Chebyshev's ripple with that
  design, a tuned classifier's search values when it is tuned. Hidden settings keep their values.
- **Fixed: recordings made from source landed inside the `eeg` package** (eeg/Data), which the build packs into
  the exe. They go to the project's Data folder, and the build stops while an eeg/Data folder exists.
- **Fixed: line noise at 60 Hz.** The Monitor's quality table and the internal-short test measure at the mains
  chosen in the filters (50 or 60 Hz).
- **Fixed: Stop in the Monitor could wait for a Workbench analysis** for up to 20 s.
- **Fixed: a scalp map's message ran off the left edge** of a narrow view; it wraps now.
- Annotations marked bad in lower case (`bad_motion`) are bad spans, as MNE reads them.
- The MATLAB test files (v7, v7.3, EEGLAB) are in the repository, so every reader test runs in a fresh checkout.

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
