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
# sys.path[0] is scripts/ when this is run as `python3 scripts/anchor_seed_replicates.py`, so the
# repo root has to be added explicitly or `from figures.allsuites import ...` cannot resolve.
# chdir alone does NOT put the cwd on the import path.
sys.path.insert(0, str(ROOT))
PY = sys.executable
P2 = Path("figure_data/climb_v2_phase2")
CBS = Path("figure_data/cbs_benchmark")
CBS_CSV = "data/cbs.csv"

# ANCHOR_REPL picks which dirs to (re)build. "" is the BASE run on the canonical triple {0,1,2};
# the stereo fix (2026-08-19) invalidated it too, so it is no longer just the replicates that
# need rebuilding.
_ALL_REPL = {"": [0, 1, 2], "_s1": [3, 4, 5], "_s2": [6, 7, 8]}
# The base dir's suffix is the empty string, which is falsy -- so the selector uses the literal
# token "base" for it. Reading os.environ.get("ANCHOR_REPL") directly would silently fall through
# to the default and rebuild the replicates instead of the base.
_ALIAS = {"base": ""}
_want = os.environ.get("ANCHOR_REPL", "").strip()
REPLICATES = ({_ALIAS.get(k, k): _ALL_REPL[_ALIAS.get(k, k)] for k in _want.split(",")} if _want
              else {"_s1": _ALL_REPL["_s1"], "_s2": _ALL_REPL["_s2"]})
# dataset -> output subdir, matching what the base run already writes
SUBDIR = {"BACE": "moleculenet_cv", "Tox21": "moleculenet_cv_tox21fixed",
          "QM7": "moleculenet_cv_qm7native", "HIV": "moleculenet_cv",
          "BBBP": "moleculenet_cv", "ESOL": "moleculenet_cv"}

ANCHORS = {
    # base dir            featurizer  head   datasets to replicate      preserve existing rows?
    "ecfp4_anchor":    dict(feat="ecfp4",     head="xgb", ds=["BACE", "Tox21", "QM7", "HIV"], merge=True),
    "fp_desc_anchor":  dict(feat="fp_desc",   head="xgb", ds=["BACE", "Tox21", "QM7", "HIV"], merge=True),
    # CheMeleon's vectors are precomputed on a chemprop>=2.2 box (see scripts/embed_chemeleon_box.py):
    # chemprop needs Python >=3.11, deepchem 2.8.0 -- which defines our Tox21 parse -- has no 3.12
    # wheel, so the two cannot share an interpreter. Splitting featurization keeps every scoring
    # decision in the reference environment instead of accepting a cross-environment offset.
    # BBBP and ESOL added 2026-08-21: _s1/_s2 carry neither, so those two cells read one seed
    # against three everywhere else (audit check 19). ESOL routes to moleculenet_cv_regnative via
    # target_subdir, with its standardize read from that same row -- see base_settings.
    "chemeleon_frozen": dict(feat="chemeleon", head="mlp",
                             ds=["BACE", "Tox21", "HIV", "BBBP", "ESOL"], merge=True,
                             npz="figure_data/_chemeleon_features.npz"),
}


VALID_POOL = ("cls", "mean", "cls_mean")


def target_subdir(base: str, ds: str) -> str:
    """The subdir the RESOLVER will actually read this (arm, dataset) from.

    The declared SUBDIR map is global, and for ESOL it is wrong for all three anchors. ESOL lives
    in moleculenet_cv_regnative (native units) AND in moleculenet_cv (z-scored); figures.allsuites
    prefers regnative, so a replicate written to moleculenet_cv is silently DROPPED and the seed
    gap stays open with every directory count passing. Resolve the same way the figure does --
    first subdir in the resolver's own preference order that the BASE actually has the dataset in.
    """
    from figures.allsuites import NATIVE_SUBDIRS
    for sub in NATIVE_SUBDIRS.get(ds, ("moleculenet_cv",)):
        f = P2 / base / sub / "moleculenet_summary.csv"
        if f.exists() and any(r["dataset"] == ds for r in csv.DictReader(f.open())):
            if sub != SUBDIR.get(ds):
                print(f"[anchor] {base}/{ds}: writing to {sub}, NOT the declared "
                      f"SUBDIR {SUBDIR.get(ds)} -- that is where the resolver reads it", flush=True)
            return sub
    return SUBDIR[ds]


def base_settings(base: str, sub: str = "moleculenet_cv", ds: str | None = None) -> list:
    """Read pool/standardize off the base run so a replicate cannot drift from it.

    MUST be read from the SAME subdir the replicate will be written to, and from that DATASET's
    own row. ESOL is the case that forces it: moleculenet_cv says standardize=zscore (value
    1.9356) and moleculenet_cv_regnative says native (1.8022). Taking the setting from
    moleculenet_cv while writing into regnative puts z-scored folds beside native ones -- the QM7
    129.9 failure, reproduced on ESOL, and landing in a subdir a correct arm already depends on.

    eval_v2 writes "-" in the pool column for every non-encoder featurizer (there is no pooling
    to record), and "-" is not a value the CLI accepts back -- so it is dropped rather than
    echoed, letting the default stand exactly as it did for the base run.
    """
    f = P2 / base / sub / "moleculenet_summary.csv"
    for r in csv.DictReader(f.open()):
        if ds and r["dataset"] != ds:
            continue
        args = ["--standardize", r["standardize"]]
        if r["pool"] in VALID_POOL:
            args += ["--pool", r["pool"]]
        return args
    raise SystemExit(f"no {ds or 'rows'} in {f}")


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


def merge_preds(new_dir: Path, dest: Path, datasets):
    """Replace only `datasets`' prediction rows in dest, keeping every other dataset's."""
    src = new_dir / "test_predictions.csv"
    if not src.exists():
        return
    rows_new = list(csv.DictReader(src.open()))
    keep = []
    if dest.exists():
        keep = [r for r in csv.DictReader(dest.open()) if r["dataset"] not in datasets]
    fields = list(rows_new[0].keys()) if rows_new else None
    if not fields:
        return
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep + rows_new:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[anchor] merged {len(rows_new)} pred rows into {dest} (kept {len(keep)})", flush=True)


def merge_suite(new_dir: Path, dest: Path):
    """Union the suite keys, so the point estimates stay in step with the merged rows."""
    src = new_dir / "suite_summary.json"
    if not src.exists():
        return
    import json
    d = json.loads(dest.read_text()) if dest.exists() else {}
    d.update(json.loads(src.read_text()))
    dest.write_text(json.dumps(d, indent=1))


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
        common0 = ["--featurizer", cfg["feat"], "--head", cfg["head"], "--cv_folds", "5"]
        if cfg.get("npz"):
            common0 += ["--features_npz", cfg["npz"]]
        for suf, seeds in REPLICATES.items():
            # ANCHOR_TAG keeps a second featurizer variant in its OWN dirs, so the orthodox
            # ECFP4 anchor and the max-information Morgan variant can be produced concurrently
            # without either writing over the other.
            name = base + suf + os.environ.get("ANCHOR_TAG", "")
            seed_args = ["--head_seeds"] + [str(s) for s in seeds]
            for ds in (ds_filter or cfg["ds"]):
                sub = target_subdir(base, ds)
                common = common0 + base_settings(base, sub, ds)
                dest_dir = P2 / name / sub
                if cfg["merge"]:
                    tmp = Path(tempfile.mkdtemp(prefix="anchorrep-"))
                    ok = run(common + seed_args + ["--datasets", ds, "--output_dir", str(tmp)],
                             f"{name} {ds}")
                    if ok:
                        merge_summary(tmp, dest_dir / "moleculenet_summary.csv", {ds})
                        # The prediction dump has to be merged on the SAME rule as the summary.
                        # Skipping it when the file already exists leaves a dump covering fewer
                        # datasets than the summary claims, and a2_bootstrap then pools 3 seed
                        # dirs for the bar while the CI sees 1 -- a mismatch that reads as a
                        # tight CI rather than as missing data.
                        merge_preds(tmp, dest_dir / "test_predictions.csv", {ds})
                        merge_suite(tmp, dest_dir / "suite_summary.json")
                    shutil.rmtree(tmp, ignore_errors=True)
                else:
                    ok = run(common + seed_args + ["--datasets", ds, "--output_dir", str(dest_dir)],
                             f"{name} {ds}")
                if not ok:
                    fails.append(f"{name}/{ds}")
            # CBS lives in its own tree and uses the benchmark's OWN fold column
            if not run_cbs:
                continue
            # CBS takes its OWN settings, not whatever the dataset loop left in `common`.
            # Since settings became per-dataset (ESOL forced it), reusing the loop variable here
            # would hand CBS the last dataset's standardize -- native, if ESOL ran last.
            cbs_common = common0 + base_settings(base, "moleculenet_cv")
            ok = run(cbs_common + seed_args + ["--task_csv", CBS_CSV, "--task_name", "cbs",
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
