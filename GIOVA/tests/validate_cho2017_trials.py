"""The Workbench template "Cho2017 trials: look, pick, STFT, wavelets" against the lab script it copies.

A lab's Python conversion (not public: `python_conversion/cho2017/run_pipeline_trials.py`)
walks one channel of Cho2017 subject 1 through baseline, an ideal band-pass per
trial, a hand-picked set of trials, then FFT, Welch, STFT and CWT averaged over
the kept trials, wavelet denoising and the DWT's bands. This runs the same file both ways and prints how far apart the
numbers are. Not a unit test: it needs the lab's `lib` package and the data.

    python tests/validate_cho2017_trials.py [--lab FOLDER] [--data s01.mat]

Run it with GIOVA's Python. The lab side runs in the lab's own .venv (it
imports PyWavelets, which GIOVA does not have) and hands its numbers back
in a file.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parents[1]
#: the lab's Python conversion (not public) and Cho2017 subject 1, from the environment
LAB = Path(os.environ.get("CHO2017_LAB", "python_conversion"))
DATA = Path(os.environ.get("CHO2017_DIR", "cho2017")) / "s01.mat"
GOOD = [1, 2, 3, 5, 7, 8]


def lab_side(lab, data):
    """The lab script's steps, with its own functions, on C3 right-hand imagery."""
    sys.path.insert(0, str(lab))
    from scipy.io import loadmat
    from lib.segmentation import baseline_correct, trial_time
    from lib.filters import design_bandpass, filter_trials
    from lib.artefact_removal import remove_bad_segments
    from lib.spectral import trial_fft_average, trial_welch_average
    from lib.wavelet import denoise_trials, trial_cwt_average, trial_dwt_bands, trial_stft_average, wavelet_bands
    sys.path.insert(0, str(lab / "cho2017"))
    from run_pipeline_trials import COUNTS_TO_UV, EEG_CH

    eeg = loadmat(str(data), squeeze_me=True, struct_as_record=False)["eeg"]
    fs = int(eeg.srate)
    row = np.asarray(eeg.imagery_right[EEG_CH.index("C3")], dtype=float) * COUNTS_TO_UV
    n = row.size // int(eeg.n_imagery_trials)
    trials = row.reshape(-1, n)[:10]
    t = trial_time(n, fs, t0=0.0)
    good = filter_trials(design_bandpass(fs, 8.0, 40.0),
                         remove_bad_segments(baseline_correct(trials, t, (0.0, 2.0)), GOOD))
    f_fft, p_fft = trial_fft_average(good, fs)
    f_w, p_w, _ = trial_welch_average(good, fs, nperseg=2 * fs, noverlap=int(0.75 * 2 * fs))
    f_st, t_st, a_st, _, _ = trial_stft_average(good, fs, t0=0.0, win_sec=1.0, overlap=0.5)
    f_cwt, a_cwt = trial_cwt_average(good, fs, f_max=45.0, n_scales=128)
    denoised, _, _ = denoise_trials(good, wavelet="db4", level=4, mode="hard", thr=10.0)
    dwt = trial_dwt_bands(good, fs, wavelet="db4", level=6)
    # each trial's share of energy per band, as the Workbench gives it (the lab averages energies first)
    shares = []
    for x in good:
        energy = [float(np.sum(b["signal"] ** 2)) for b in wavelet_bands(x, fs, "db4", 6)]
        shares.append(100 * np.asarray(energy) / sum(energy))
    return dict(fs=fs, good=good, f_fft=f_fft, p_fft=p_fft, f_w=f_w, p_w=p_w, f_st=f_st, t_st=t_st, a_st=a_st,
                f_cwt=f_cwt, a_cwt=a_cwt, denoised=denoised, envelopes=np.array([b["envelope"] for b in dwt]),
                shares=np.array(shares), band_names=np.array([b["name"] for b in dwt]))


def lab_numbers(lab, data):
    """`lab_side` run by the lab's own Python, read back from a file."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "lab.npz"
        subprocess.run([str(lab / ".venv" / "Scripts" / "python.exe"), "-I", __file__, "--lab", str(lab),
                        "--data", str(data), "--lab-only", str(path)], check=True)
        with np.load(path) as saved:
            return {k: saved[k] for k in saved.files}


def workbench_side(data):
    sys.path.insert(0, str(HERE))
    from eeg.pipeline import Runner, templates
    graph = templates.trial_lab(str(data))
    out = Runner(graph).run()
    return graph, out


def worst(a, b, scale=None):
    """Largest |a - b|, and the same relative to the largest |b| (or `scale`)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = float(np.max(np.abs(a - b)))
    return diff, diff / float(scale if scale is not None else np.max(np.abs(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lab", type=Path, default=LAB)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--lab-only", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.lab_only:
        np.savez(args.lab_only, **lab_side(args.lab, args.data))
        return
    lab = lab_numbers(args.lab, args.data)
    graph, out = workbench_side(args.data)
    fs = int(lab["fs"])

    kept = out["select_trials1"]["epochs"]
    print("Trials kept: Workbench {} | lab {}".format(", ".join(kept.ids), ", ".join(map(str, GOOD))))
    print("Kept trials, µV           max |diff| {:.3g} ({:.2g} of the largest sample)".format(
        *worst(kept.data[:, 0], lab["good"])))

    # FFT: the Workbench gives the amplitude A of each trial (a sine of amplitude A reads A); the lab
    # gives the one-sided power density |X/fs|² · 2 / T. For bins between 0 and Nyquist, A² · T / 2 is it.
    fft = out["fft1"]["spectrum"]
    duration = kept.data.shape[-1] / fs
    density = (fft.power[:, 0] ** 2 * duration / 2).mean(axis=0)
    p_lab = lab["p_fft"]
    inner = slice(1, -1)
    print("FFT density, µV²/Hz       max |diff| {:.3g} ({:.2g} of the peak)".format(
        *worst(density[inner], p_lab[inner])))

    # Welch: both a one-sided Hann density; the lab's np.hanning is the symmetric window, scipy's periodic.
    psd = out["psd1"]["spectrum"]
    f_w, p_w = lab["f_w"], lab["p_w"]
    band = (f_w >= 8) & (f_w <= 40)
    print("Welch PSD 8-40 Hz, µV²/Hz max |diff| {:.3g} ({:.2g} of the peak)".format(
        *worst(psd.power[:, 0].mean(axis=0)[band], p_w[band])))

    # STFT: the Workbench's amplitude reads A, scipy.signal.stft's half of it; the lab pads each trial so
    # its frames also sit at 0 s and 7 s, the Workbench keeps the frames that lie inside the trial.
    tf = out["erd1"]["tf"]
    channel = tf.tf[0]
    f_st, t_st, a_st = lab["f_st"], lab["t_st"], lab["a_st"]
    rows = [int(np.argmin(np.abs(f_st - f))) for f in channel.freqs]
    columns = [int(np.argmin(np.abs(t_st - t))) for t in channel.times]
    lab_map = a_st[np.ix_(rows, columns)]
    ours = channel.amplitude["right"] / 2
    # The Workbench takes each frame's mean out before its FFT (scipy's spectrogram detrend), the lab
    # does not: that changes the lowest rows only, where a frame's offset leaks through the window.
    above = channel.freqs >= 2
    print("STFT |X| from 2 Hz, µV    max |diff| {:.3g} ({:.2g} of the map's largest)".format(
        *worst(ours[above], lab_map[above], np.max(lab_map))))
    print("STFT |X| at {:g} Hz        max |diff| {:.3g} ({:.2g}): each frame's mean removed here, not in "
          "the lab".format(channel.freqs[0], *worst(ours[~above], lab_map[~above], np.max(lab_map))))
    print("  frames compared: {} of the lab's {} ({:g} to {:g} s), {} frequencies {:g}-{:g} Hz".format(
        len(columns), t_st.size, channel.times[0], channel.times[-1], channel.freqs.size,
        channel.freqs[0], channel.freqs[-1]))
    mu = (channel.freqs >= 8) & (channel.freqs <= 13)
    print("  mu 8-13 Hz:", " ".join("{:.2f}".format(v) for v in ours[mu].mean(axis=0)), "(Workbench)")
    print("             ", " ".join("{:.2f}".format(v) for v in lab_map[mu].mean(axis=0)), "(lab)")

    # Wavelet denoising after the band-pass (db4, 4 levels, hard, 10 µV): the lab's FIG 6.
    denoised = out["wavelet_denoise1"]["epochs"].data[:, 0]
    print("Wavelet denoised, µV      max |diff| {:.3g} ({:.2g} of the largest sample)".format(
        *worst(denoised, lab["denoised"])))

    # CWT: the Workbench reads A, the lab A/2; it keeps every 10th sample (50 frames a second at 512 Hz).
    # The lab's pywt.cwt is approximate where the wavelet is long (core.wavelets.cwt_amplitude): compare by band.
    cwt = out["erd2"]["tf"].tf[0]
    decim = int(round((cwt.times[1] - cwt.times[0]) * fs))
    np.testing.assert_allclose(cwt.freqs, lab["f_cwt"])
    ours, theirs = cwt.amplitude["right"] / 2, lab["a_cwt"][:, ::decim]
    peak = float(np.max(theirs))
    for low, high in ((1, 8), (8, 13), (13, 30), (30, 45)):
        rows = (cwt.freqs >= low) & (cwt.freqs <= high)
        print("CWT |X| {:2d}-{:2d} Hz, µV       max |diff| {:.3g} ({:.2g} of the map's largest); mean {:.3f} here, "
              "{:.3f} lab".format(low, high, *worst(ours[rows], theirs[rows], peak), ours[rows].mean(),
                                  theirs[rows].mean()))

    # DWT bands: the mean amplitude envelope of each band, and each trial's share of energy.
    bands = out["dwt_bands1"]
    envelopes = bands["epochs"].data.mean(axis=0)
    print("DWT envelopes, µV         max |diff| {:.3g} ({:.2g} of the largest)".format(
        *worst(envelopes, lab["envelopes"])))
    print("DWT energy share, %       max |diff| {:.3g}".format(worst(bands["features"].values, lab["shares"])[0]))
    for name, share in zip(lab["band_names"], bands["features"].values.mean(axis=0)):
        print("  {:5s} {:5.1f} % of the energy (mean over trials)".format(str(name), share))


if __name__ == "__main__":
    main()
