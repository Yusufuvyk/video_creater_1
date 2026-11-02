"""Both motor imagery templates over every Cho2017 subject, side by side.

Writes validation-output/cho2017_compare.csv (accuracy and ROC AUC per subject
and pipeline) and prints the summary and the paired comparison. Takes about a
minute per subject.

Run: .venv/Scripts/python tests/validate_cho2017.py [folder with s01.mat ...] [--subjects 5]
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg.pipeline import batch, templates  # noqa: E402

DEFAULT = Path("cho2017")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(DEFAULT))
    parser.add_argument("--subjects", type=int, default=0, help="only the first N; 0: all")
    args = parser.parse_args()
    files = sorted(Path(args.folder).glob("s[0-9][0-9].mat"))
    if args.subjects:
        files = files[:args.subjects]
    result = batch.compare([templates.csp_lda(), templates.band_power_lda()], files,
                           progress=lambda text: print(text, flush=True))
    out = ROOT.parent / "validation-output"
    out.mkdir(exist_ok=True)
    result.save_csv(out / "cho2017_compare.csv")
    print()
    for name, mean, sd, median, n in result.summary():
        print("{:20s} accuracy {:.1%} ± {:.1%} (median {:.1%}, {} subjects)".format(name, mean, sd, median, n))
    for a, b, diff, p, n in result.paired():
        print("{} - {}: {:+.1%} (Wilcoxon p = {:.3g}, {} subjects)".format(a, b, diff, p, n))
    for (subject, pipeline), message in result.errors.items():
        print("FAILED", subject, pipeline, message)
    print("Saved", out / "cho2017_compare.csv")


if __name__ == "__main__":
    main()
