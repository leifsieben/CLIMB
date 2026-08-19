"""Give the fig A1/A2 anchor arms three model-seed replicates instead of one.

ECFP, ECFP+desc and CheMeleon-frozen each had exactly ONE run behind their BACE / CBS / QM7 /
Tox21 bars, so sd_seeds was literally 0.0 -- no model-seed error bar at all. That is the worst
place for the gap: ECFP+XGBoost is the arm that beats the CLMs, so it is the first bar a reviewer
checks. (Ames and MoleculeACE already carry three EVAL seeds for these arms; the aggregator
prints n_seeds=1 there only because it counts directories. Those panels need no new compute.)

The replicate has to be a whole 3-head-seed ENSEMBLE, not a single head seed. The published point
estimate is eval_v2's ensembled fold row -- it averages the 3 head-seeds' predictions BEFORE
scoring -- so exposing the existing s0/s1/s2 `_cell` rows as "three seeds" would silently swap in
a different, worse estimator and move every anchor bar. That was the 2026-08-16 bug. Instead each
replicate re-runs the full pipeline on a disjoint head-seed triple: {0,1,2} (existing), {3,4,5},
{6,7,8}. Same estimator as the CLIMB arms; the spread across the three is an honest model-seed SD.

Runs LOCALLY on purpose. The laptop is the Tox21 reference environment (77,864 masked prediction
rows, 7,822 reference molecules); a fresh EC2 box parses 7,831 molecules and lands ~0.008 off,
which is the drift that cost us the fig_C2/fig_D Tox21 column. Re-introducing it into fig_A's
anchors would be worse than having no error bar.

Per-dataset output subdirs mirror the base run's, so the resolver keeps preferring the corrected
copies and units never mix:
    BACE  -> moleculenet_cv/            Tox21 -> moleculenet_cv_tox21fixed/
    QM7   -> moleculenet_cv_qm7native/  cbs   -> cbs_benchmark/<run>/moleculenet_cv/

chemeleon_frozen_s1/_s2 ALREADY hold QM7 (extra head seeds on partition 0, deliberately pooled --
see figures/arms.py). eval_v2 opens moleculenet_summary.csv with "w", so writing BACE into those
dirs would DELETE that QM7. New rows are therefore merged in, never written straight over.
"""
from __future__ import annotations
import csv, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = sys.executable
P2 = Path("figure_data/climb_v2_phase2")
CBS = Path("figure_data/cbs_benchmark")
CBS_CSV = "data/cbs.csv"

REPLICATES = {"_s1": [3, 4, 5], "_s2": [6, 7, 8]}
# dataset -> output subdir, matching what the base run already writes
SUBDIR = {"BACE": "moleculenet_cv", "Tox21": "moleculenet_cv_tox21fixed",
          "QM7": "moleculenet_cv_qm7native", "HIV": "moleculenet_cv"}

ANCHORS = {
    # base dir            featurizer  head   datasets to replicate      preserve existing rows?
    "ecfp4_anchor":    dict(feat="ecfp4",     head="xgb", ds=["BACE", "Tox21", "QM7", "HIV"], merge=True),
    "fp_desc_anchor":  dict(feat="fp_desc",   head="xgb", ds=["BACE", "Tox21", "QM7", "HIV"], merge=True),
    # CheMeleon's vectors are precomputed on a chemprop>=2.2 box (see scripts/embed_chemeleon_box.py):
    # chemprop needs Python >=3.11, deepchem 2.8.0 -- which defines our Tox21 parse -- has no 3.12
    # wheel, so the two cannot share an interpreter. Splitting featurization keeps every scoring
    # decision in the reference environment instead of accepting a cross-environment offset.
    "chemeleon_frozen": dict(feat="chemeleon", head="mlp", ds=["BACE", "Tox21", "HIV"],  merge=True,
                             npz="figure_data/_chemeleon_features.npz"),
}


VALID_POOL = ("cls", "mean", "cls_mean")


def base_settings(base: str) -> list:
    """Read pool/standardize off the base run so a replicate cannot drift from it.

    eval_v2 writes "-" in the pool column for every non-encoder featurizer (there is no pooling
    to record), and "-" is not a value the CLI accepts back -- so it is dropped rather than
    echoed, letting the default stand exactly as it did for the base run.
    """
    f = P2 / base / "moleculenet_cv" / "moleculenet_summary.csv"
    for r in csv.DictReader(f.open()):
        args = ["--standardize", r["standardize"]]
        if r["pool"] in VALID_POOL:
            args += ["--pool", r["pool"]]
        return args
    raise SystemExit(f"no rows in {f}")


def run(args, label) -> bool:
    for d in Path(tempfile.gettempdir()).glob("*-featurized"):
        shutil.rmtree(d, ignore_errors=True)   # DeepChem cache collides across runs
    print(f"[anchor] === {label} ===", flush=True)
    r = subprocess.run([PY, "eval_v2.py"] + args, capture_output=True, text=True)
    ok = "[eval_v2] wrote" in r.stdout
    print(f"[anchor] {label}: {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("STDOUT:", r.stdout[-1500:], "\nSTDERR:", r.stderr[-1500:], flush=True)
    return ok


def merge_summary(new_dir: Path, dest: Path, datasets):
    """Replace only `datasets`' rows in dest, keeping every other dataset already there."""
    src = new_dir / "moleculenet_summary.csv"
    rows_new = list(csv.DictReader(src.open()))
    keep = []
    if dest.exists():
        keep = [r for r in csv.DictReader(dest.open()) if r["dataset"] not in datasets]
    fields = list(rows_new[0].keys())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep + rows_new:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[anchor] merged {len(rows_new)} rows into {dest} (kept {len(keep)})", flush=True)


def main(only=None) -> int:
    fails = []
    # ANCHOR_DS restricts the pass to named datasets, so a later addition (HIV joining the
    # canonical six) can be topped up without re-running what already landed. merge=True keeps
    # the earlier datasets' rows in place; eval_v2 itself opens the summary with "w".
    ds_filter = [d for d in os.environ.get("ANCHOR_DS", "").split(",") if d]
    run_cbs = os.environ.get("ANCHOR_CBS", "1") == "1"
    for base, cfg in ANCHORS.items():
        if only and base not in only:
            continue
        common = ["--featurizer", cfg["feat"], "--head", cfg["head"],
                  "--cv_folds", "5"] + base_settings(base)
        if cfg.get("npz"):
            common += ["--features_npz", cfg["npz"]]
        for suf, seeds in REPLICATES.items():
            name = base + suf
            seed_args = ["--head_seeds"] + [str(s) for s in seeds]
            for ds in (ds_filter or cfg["ds"]):
                dest_dir = P2 / name / SUBDIR[ds]
                if cfg["merge"]:
                    tmp = Path(tempfile.mkdtemp(prefix="anchorrep-"))
                    ok = run(common + seed_args + ["--datasets", ds, "--output_dir", str(tmp)],
                             f"{name} {ds}")
                    if ok:
                        merge_summary(tmp, dest_dir / "moleculenet_summary.csv", {ds})
                        for extra in ("test_predictions.csv", "suite_summary.json"):
                            if (tmp / extra).exists() and not (dest_dir / extra).exists():
                                shutil.copy2(tmp / extra, dest_dir / extra)
                    shutil.rmtree(tmp, ignore_errors=True)
                else:
                    ok = run(common + seed_args + ["--datasets", ds, "--output_dir", str(dest_dir)],
                             f"{name} {ds}")
                if not ok:
                    fails.append(f"{name}/{ds}")
            # CBS lives in its own tree and uses the benchmark's OWN fold column
            if not run_cbs:
                continue
            ok = run(common + seed_args + ["--task_csv", CBS_CSV, "--task_name", "cbs",
                                           "--task_type", "classification", "--cv_scheme", "provided",
                                           "--output_dir", str(CBS / name / "moleculenet_cv")],
                     f"{name} cbs")
            if not ok:
                fails.append(f"{name}/cbs")
    print(f"\n[anchor] DONE  {len(fails)} failure(s): {', '.join(fails) if fails else 'none'}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
