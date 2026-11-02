"""What an EEG recording is, and what can be computed from one.

    config      board constants, data paths, recording/marker file pairing
    sources     Recording, and the file and live sample sources
    montages    electrode names, and where they sit on a head
    processing  filters, electrode quality, auto-scaling, live spectra
    analysis    Welch / STFT / Morlet, ERD, trial averaging, features
    features    the live feature stream, one row per window
    events      marker files, labelled blocks, epoch pairing
    paradigm    trial structure written down instead of read from markers

No Qt anywhere in here: every one of these is importable and testable without
a display, which is what keeps the analysis honest and the tests fast.
"""
__all__ = ["config", "sources", "montages", "processing", "analysis",
           "features", "events", "paradigm"]
