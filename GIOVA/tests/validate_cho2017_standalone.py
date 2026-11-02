"""Exported scripts against the Workbench on every Cho2017 subject.

For each subject and pipeline (CSP + LDA, and FBCSP-MIBIF-NBPW from the
literature library), the sheet is run in this process and its exported script in
another Python (-I) where `eeg` cannot be imported. Compared: every fold's
accuracy and every tested trial's predicted condition (exactly), and every
trial's classifier score (largest difference reported). Writes
validation-output/cho2017_standalone.csv. Takes about a minute per subject.

Run: .venv/Scripts/python tests/validate_cho2017_standalone.py [folder] [--subjects 5]
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg.pipeline import Runner, batch, literature, templates  # noqa: E402
from eeg.pipeline.export import to_python  # noqa: E402

DEFAULT = Path("cho2017")
PIPELINES = {"CSP + LDA": templates.csp_lda, "FBCSP-MIBIF-NBPW": literature.fbcsp}

#: run in the other Python: the script's run(), its scores as JSON
ALONE = r"""
import importlib.abc, json, runpy, sys
class NoEEGStudio(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "eeg" or name.startswith("eeg."):
            raise ImportError("the exported script imported GIOVA's " + name)
sys.meta_path.insert(0, NoEEGStudio())
out = runpy.run_path(sys.argv[1], run_name="exported")["run"]()
scores = out[sys.argv[2]]
json.dump(dict(fold_accuracy=scores.fold_accuracy, predictions=scores.predictions, describe=scores.describe()),
          open(sys.argv[3], "w", encoding="utf-8"))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(DEFAULT))
    parser.add_argument("--subjects", type=int, default=0, help="only the first N; 0: all")
    args = parser.parse_args()
    files = sorted(Path(args.folder).glob("s[0-9][0-9].mat"))
    if args.subjects:
        files = files[:args.subjects]
    out = ROOT.parent / "validation-output"
    work = ROOT / "tests" / ".tmp" / "standalone_cho2017"
    work.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    rows, failed = [], 0
    for path in files:
        for name, build in PIPELINES.items():
            started = time.time()
            graph = build(str(path))
            cv = batch.score_nodes(graph)[0]
            sheet = Runner(graph).run()[cv]["scores"]
            script = work / "pipeline.py"
            script.write_text(to_python(graph), encoding="utf-8")
            result = work / "scores.json"
            done = subprocess.run([sys.executable, "-I", "-c", ALONE, str(script), cv, str(result)], cwd=str(work),
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
            if done.returncode:
                print("FAILED", path.stem, name, done.stderr[-1500:], flush=True)
                failed += 1
                continue
            alone = json.loads(result.read_text(encoding="utf-8"))
            same_folds = alone["fold_accuracy"] == sheet.fold_accuracy
            same_labels = [p[:3] for p in alone["predictions"]] == [list(p[:3]) for p in sheet.predictions]
            gap = max(abs(a[3] - b[3]) for a, b in zip(alone["predictions"], sheet.predictions))
            rows.append(dict(subject=path.stem, pipeline=name, sheet_accuracy=sheet.accuracy,
                             script_accuracy=sum(alone["fold_accuracy"]) / len(alone["fold_accuracy"]),
                             same_fold_accuracies=same_folds, same_predictions=same_labels,
                             largest_score_difference=gap, seconds=round(time.time() - started, 1)))
            failed += not (same_folds and same_labels)
            print("{} {:18s} sheet {:.1%} script {:.1%} folds {} predictions {} score gap {:.2g}".format(
                path.stem, name, sheet.accuracy, rows[-1]["script_accuracy"], "same" if same_folds else "DIFFER",
                "same" if same_labels else "DIFFER", gap), flush=True)
    out.mkdir(exist_ok=True)
    with (out / "cho2017_standalone.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print("{} runs, {} differing or failed; largest score difference {:.2g}".format(
        len(rows), failed, max(r["largest_score_difference"] for r in rows)))
    print("Saved", out / "cho2017_standalone.csv")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
