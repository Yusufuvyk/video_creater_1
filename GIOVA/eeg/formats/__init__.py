"""Getting a Recording out of a file on disk.

    recordings  the dispatcher: open any supported file
    matfile     reading .mat, and the Mapping that describes how
    channels    electrode names for a recording that carries none
    presets     datasets we recognise on sight, and the facts they imply

A .mat file does not say which of its arrays is the data, where the EEG rows
stop, or what its integers mean in microvolts. That is the whole reason this
layer exists: `core` should never have to guess.
"""
__all__ = ["recordings", "matfile", "channels", "presets"]
