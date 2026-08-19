"""Fill CheMeleon e2e's two missing six-panel cells: MoleculeACE (30 targets) and hERG (Polaris).

CheMeleon e2e = native chemprop D-MPNN initialised from the CheMeleon foundation
(`--from-foundation CHEMELEON`), identical to the arm already on BACE/Tox21/QM7/CBS. Both cells
use the benchmark's OWN provided train/test split (the `split` column), 3 seeds, matching the
frozen CheMeleon arm and every CLIMB arm on those panels.

  MoleculeACE -> figure_data/chemeleon_suite/moleculeace/chemeleon_e2e/{results.csv,
                 test_predictions.csv,verified.json}   (scored locally: RMSE overall/cliff/noncliff)
  hERG        -> figure_data/chemeleon_suite/polaris/chemeleon_e2e/test_predictions.csv
                 then scored OFF-model by scripts/chemeleon_suite_score_polaris.py in .venv_polaris
                 (Polaris withholds test labels).

Writes into NEW prefixes ("chemeleon_e2e"), never into an existing multi-task Polaris dir — see
notes/polaris-clobber-recovery-2026-08-17.md for why that matters.

Run on the box with the chemeleon venv:  ~/venvs/chemeleon/bin/python scripts/chemeleon_e2e_gaps.py
Env: CHEM_SEEDS ("42 117 709"), CHEM_EPOCHS (50), CHEM_ONLY ("mace"|"herg"|both).
"""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
CHEMPROP = os.environ.get("CHEMPROP_BIN", str(Path.home() / "venvs" / "chemeleon" / "bin" / "chemprop"))
SEEDS = [int(s) for s in os.environ.get("CHEM_SEEDS", "42 117 709").split()]
EPOCHS = int(os.environ.get("CHEM_EPOCHS", "50"))
ONLY = os.environ.get("CHEM_ONLY", "")
# CHEM_RUN lets a replicate write to chemeleon_e2e_s1 / _s2 instead of overwriting the mainline
# dir (2026-08-19). Three model seeds on the suite tracks needs three OUTPUT dirs, and this script
# hardcoded one -- so a second run at a different CHEM_SEEDS would have silently replaced the
# published one rather than sitting beside it.
RUN = os.environ.get("CHEM_RUN", "chemeleon_e2e")
MACE_DIR = ROOT / "chemeleon_suite" / "data" / "moleculeace"
POL_DIR = ROOT / "chemeleon_suite" / "data" / "polaris"
S3 = "s3://climb-s3-bucket/experiments/chemeleon_suite"
LOG = ROOT / "analysis" / "chemeleon_e2e_gaps.log"
LOG.parent.mkdir(exist_ok=True)


def log(m):
    print(f"[chem-e2e] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[chem-e2e] {m}\n")


def sh(c, **kw):
    return subprocess.run(c, **kw)


def train_predict(task_type, tr_smi, tr_y, te_smi, seed, td):
    """Train one chemprop-from-CheMeleon model and predict te_smi. Returns 1-D predictions."""
    trp, tep, outp, predp = td / "tr.csv", td / "te.csv", td / f"o{seed}", td / f"p{seed}.csv"
    with trp.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles", "y"])
        w.writerows([[s, v] for s, v in zip(tr_smi, tr_y)])
    with tep.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles"]); w.writerows([[s] for s in te_smi])
    cmd = [CHEMPROP, "train", "--data-path", str(trp), "--task-type", task_type,
           "--smiles-columns", "smiles", "--target-columns", "y", "--output-dir", str(outp),
           "--epochs", str(EPOCHS), "--patience", "15", "--split-sizes", "0.9", "0.1", "0.0",
           "--pytorch-seed", str(seed), "--data-seed", str(seed), "--num-workers", "0",
           "--from-foundation", "CHEMELEON"]
    if task_type == "classification":
        cmd += ["--class-balance"]
    r = sh(cmd, capture_output=True, text=True)
    model = None
    for pat in ("**/best*.pt", "**/last*.pt", "**/*.ckpt"):
        hits = sorted(outp.glob(pat))
        if hits:
            model = str(hits[0]); break
    if model is None:
        raise RuntimeError(f"train failed seed{seed}: {r.stderr[-800:]}")
    pr = sh([CHEMPROP, "predict", "--test-path", str(tep), "--model-path", model,
             "--preds-path", str(predp), "--smiles-columns", "smiles"], capture_output=True, text=True)
    if not predp.exists():
        raise RuntimeError(f"predict failed seed{seed}: {pr.stderr[-800:]}")
    rows = list(csv.DictReader(predp.open()))
    col = [c for c in rows[0] if c != "smiles"][0]
    return np.array([float(r[col]) for r in rows])


# ------------------------------------------------------------------ MoleculeACE ---------------
def do_moleculeace():
    out = ROOT / "figure_data" / "chemeleon_suite" / "moleculeace" / RUN
    out.mkdir(parents=True, exist_ok=True)
    tasks = (ROOT / "chemeleon_suite" / "tasks" / "moleculeace_tasks.txt").read_text().split()
    res_rows, pred_rows = [], []
    for ti, task in enumerate(tasks, 1):
        rows = list(csv.DictReader((MACE_DIR / f"{task}.csv").open()))
        ycol = "y [pEC50/pKi]"
        smi = [r["smiles"] for r in rows]
        y = np.array([float(r[ycol]) for r in rows])
        split = [r["split"] for r in rows]
        cliff = np.array([r["cliff_mol"] in ("1", "1.0", "True") for r in rows])
        tr = [i for i, s in enumerate(split) if s == "train"]
        te = [i for i, s in enumerate(split) if s == "test"]
        for seed in SEEDS:
            with tempfile.TemporaryDirectory() as t:
                pred = train_predict("regression", [smi[i] for i in tr], y[tr],
                                     [smi[i] for i in te], seed, Path(t))
            yt = y[te]
            for sub, mask in (("overall", np.ones(len(te), bool)),
                              ("cliff", cliff[te]), ("noncliff", ~cliff[te])):
                if mask.sum() == 0:
                    continue
                e = pred[mask] - yt[mask]
                res_rows.append(dict(task=task, seed=seed, subset=sub, metric="rmse",
                                     value=float(np.sqrt((e ** 2).mean())), n_test=int(mask.sum())))
            for j, i in enumerate(te):
                pred_rows.append(dict(task=task, seed=seed, test_index=j, smiles=smi[i],
                                      y_pred=float(pred[j])))
        log(f"moleculeace {ti}/{len(tasks)} {task} done")
        with (out / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "seed", "subset", "metric", "value", "n_test"])
            w.writeheader(); w.writerows(res_rows)
        with (out / "test_predictions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "seed", "test_index", "smiles", "y_pred"])
            w.writeheader(); w.writerows(pred_rows)
    if len({r["task"] for r in res_rows}) >= len(tasks):
        (out / "verified.json").write_text(json.dumps(
            {"track": "moleculeace", "model": RUN, "featurizer": "chemprop_from_foundation",
             "head": "e2e", "seeds": SEEDS, "n_tasks": len(tasks)}))
        log("moleculeace VERIFIED")
    sh(["aws", "s3", "cp", "--recursive", str(out), f"{S3}/moleculeace/{RUN}", "--only-show-errors"])


# ------------------------------------------------------------------ hERG ----------------------
def do_herg():
    out = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / RUN
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((POL_DIR / "tdcommons__herg.csv").open()))
    smi = [r["smiles"] for r in rows]
    split = [r["split"] for r in rows]
    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    ytr = np.array([float(rows[i]["y"]) for i in tr])
    pred_rows = []
    for seed in SEEDS:
        with tempfile.TemporaryDirectory() as t:
            pred = train_predict("classification", [smi[i] for i in tr], ytr,
                                 [smi[i] for i in te], seed, Path(t))
        for j, i in enumerate(te):
            pred_rows.append(dict(task="tdcommons/herg", seed=seed, test_index=j,
                                  smiles=smi[i], y_pred=float(pred[j])))
        log(f"herg seed{seed} done")
    with (out / "test_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "test_index", "smiles", "y_pred"])
        w.writeheader(); w.writerows(pred_rows)
    log("herg predictions written — score with .venv_polaris/bin/python "
        f"scripts/chemeleon_suite_score_polaris.py {out}")


def do_polaris_all():
    """FULL 28-task Polaris track for chemeleon_e2e (ASK 6).

    do_herg() scored exactly one task, which left this arm at 39/66 all-suites coverage -- below
    fig_A1's >=60/66 admission floor, so the e2e variant was excluded from the ranking panel while
    the FROZEN variant appeared in it. One comparator name, two different models 65 kcal/mol apart
    on QM7, in one figure. Scoring the whole track takes it to 66/66 and removes that split.

    Writes ALL tasks in ONE pass: test_predictions.csv is opened "w", so a per-task run would
    replace the file each time and leave only the last task -- the same rewrite trap that has bitten
    this repo repeatedly (see notes/polaris-clobber-recovery-2026-08-17.md).
    """
    out = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / RUN
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((POL_DIR / "polaris_manifest.json").read_text())
    tasks = (ROOT / "chemeleon_suite" / "tasks" / "polaris_tasks.txt").read_text().split()
    log(f"polaris ALL: {len(tasks)} tasks x {len(SEEDS)} seeds")
    pred_rows = []
    for ti, task in enumerate(tasks, 1):
        fname = task.replace("/", "__") + ".csv"
        fp = POL_DIR / fname
        if not fp.exists():
            log(f"  [{ti}/{len(tasks)}] SKIP {task}: {fname} missing"); continue
        rows = list(csv.DictReader(fp.open()))
        smi = [r["smiles"] for r in rows]
        split = [r["split"] for r in rows]
        tr = [i for i, sp in enumerate(split) if sp == "train"]
        te = [i for i, sp in enumerate(split) if sp == "test"]
        if not tr or not te:
            log(f"  [{ti}/{len(tasks)}] SKIP {task}: empty split"); continue
        ttype = man[task]["type"]
        ytr = np.array([float(rows[i]["y"]) for i in tr])
        for seed in SEEDS:
            with tempfile.TemporaryDirectory() as t:
                pred = train_predict(ttype, [smi[i] for i in tr], ytr,
                                     [smi[i] for i in te], seed, Path(t))
            for j, i in enumerate(te):
                pred_rows.append(dict(task=task, seed=seed, test_index=j,
                                      smiles=smi[i], y_pred=float(pred[j])))
        log(f"  [{ti}/{len(tasks)}] {task} done ({len(te)} test)")
    with (out / "test_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "test_index", "smiles", "y_pred"])
        w.writeheader(); w.writerows(pred_rows)
    log(f"wrote {len(pred_rows)} rows across {len({r['task'] for r in pred_rows})} tasks")


def main():
    if ONLY in ("", "mace"):
        do_moleculeace()
    if ONLY == "polaris_all":
        do_polaris_all()
    elif ONLY in ("", "herg"):
        do_herg()
    log("DONE")


if __name__ == "__main__":
    main()
