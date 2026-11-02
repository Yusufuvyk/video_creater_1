"""The Qt application: Home, the Monitor, the Workbench, Games, a live feature window, and shared widgets.

    theme       palette-aware colours, so text stays readable in either theme
    widgets     small shared Qt helpers (spin box, combo, check box)
    plots       time-series, spectrum, band-power, multi-curve, image grid
    headmap     the head-map channel picker
    matdialog   asking how to read a .mat, and remembering the answer
    panel       the monitor control sidebar
    playback    Player and the media-style transport bar
    monitor     the live monitor page
    topomap     scalp maps
    profile_box choosing and saving analysis profiles
    features_window live feature extraction, in its own window
    app         main window: Home, Monitor, Workbench, Games

Everything here may import core and formats; nothing in those may import this.
"""
__all__ = ["theme", "widgets", "plots", "headmap", "matdialog", "panel",
           "playback", "monitor", "topomap", "profile_box", "feature_view", "features_window", "app"]
