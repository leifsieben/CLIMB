"""Re-evaluate only the tasks affected by the two loader fixes and MERGE the
corrected results back into each run's existing eval outputs.

Affected tasks (everything else is byte-identical and left alone):
  - Tox21  : missing multitask labels were fed as fake negatives (w==0 -> NaN fix)
  - ESOL   : regression targets were DeepChem-normalized + cross-fold leaked
  - QM7    : same as ESOL
  (Lipophilicity is regression-affected too but excluded from every figure; we
   re-run it as well when a run scored it, purely for internal consistency.)

HIV / BBBP / BACE are single-task classification with rank-based metrics, so
neither fix touches them -- their rows are preserved untouched by the merge.

For each run and each split dir (moleculenet_cv/ = 5-fold CV, moleculenet/ =
hold-out) this:
  1. reads the run's own eval config (featurizer/pool/standardize/head + seeds)
     from the existing moleculenet_summary.csv, so the re-run matches exactly;
  2. shells out to eval_v2.py on the affected datasets into a temp dir;
  3. merges the affected datasets' rows into moleculenet_summary.csv,
     suite_summary.json and test_predictions.csv, dropping the stale rows for
     those datasets and keeping HIV/BBBP/BACE as-is.

Idempotent: writes a `.corrected_v2.json` marker in each split dir and skips it
on re-run unless --force. e2e runs (name contains 'e2e') are SKIPPED here --
they need finetune_e2e_v2 and are handled by a separate pass.

Usage:
  python scripts/rerun_corrected_tasks.py --waves climb_v2_phase2 [--dry-run]
  python scripts/rerun_corrected_tasks.py --waves climb_v2_phase2 climb_v2_ablation_dedup climb_v2_h1
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figure_data"
AFFECTED = ["Tox21", "ESOL", "QM7", "Lipophilicity"]
TOK_DEFAULT = DATA / "_tokenizer"
ENC_SEARCH = [DATA, ROOT / ".hf_staging" / "encoders"]   # <base>/<wave>/<run>/[encoder/]


def resolve_encoder(wave: str, run: str) -> Path | None:
    for base in ENC_SEARCH:
        for cand in (base / wave / run / "encoder", base / wave / run):
            if (cand / "model.safetensors").exists():
                return cand
    return None


def read_cfg(split_dir: Path):
    """(featurizer, pool, standardize, head, head_seeds, datasets_present) or None."""
    summ = split_dir / "moleculenet_summary.csv"
    if not summ.exists():
        return None
    d = pd.read_csv(summ)
    if not len(d):
        return None
    r = d.iloc[0]
    seeds = sorted({int(s) for s in d.head_seed.astype(str) if s.isdigit()}) or [0, 1, 2]
    present = [t for t in AFFECTED if t in set(d.dataset)]
    # bracket access: r["head"] would otherwise resolve to the Series.head() method
    pool = str(r["pool"])
    return dict(featurizer=str(r["featurizer"]), pool=(pool if pool != "-" else "mean"),
                standardize=str(r["standardize"]), head=str(r["head"]), seeds=seeds, datasets=present)


def run_eval(split_dir: Path, cfg: dict, enc: Path | None, tok: Path, cv: bool, tmp: Path) -> bool:
    cmd = [sys.executable, str(ROOT / "eval_v2.py"),
           "--output_dir", str(tmp),
           "--featurizer", str(cfg["featurizer"]),
           "--pool", str(cfg["pool"]),
           "--standardize", str(cfg["standardize"]),
           "--head", str(cfg["head"]),
           "--head_seeds", *[str(s) for s in cfg["seeds"]],
           "--max_length", "256",
           "--datasets", *cfg["datasets"]]
    if cfg["featurizer"] == "encoder":
        if enc is None:
            return False
        cmd += ["--encoder", str(enc), "--tokenizer", str(tok)]
    if cv:
        cmd += ["--cv_folds", "5"]
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        (tmp / "eval_stderr.log").write_text(r.stdout + "\n" + r.stderr)
        return False
    return (tmp / "moleculenet_summary.csv").exists()


def merge(dest: Path, tmp: Path, tasks: list[str]):
    # 1) summary csv: drop affected datasets' old rows, append the new ones
    dcsv, ncsv = dest / "moleculenet_summary.csv", tmp / "moleculenet_summary.csv"
    dd, nn = pd.read_csv(dcsv), pd.read_csv(ncsv)
    pd.concat([dd[~dd.dataset.isin(tasks)], nn[nn.dataset.isin(tasks)]], ignore_index=True).to_csv(dcsv, index=False)
    # 2) suite_summary.json: drop keys whose dataset prefix is affected, then add new
    djs, njs = dest / "suite_summary.json", tmp / "suite_summary.json"
    if djs.exists() and njs.exists():
        dj = json.loads(djs.read_text()); nj = json.loads(njs.read_text())
        merged = {k: v for k, v in dj.items() if k.split("_")[0] not in tasks}
        merged.update({k: v for k, v in nj.items() if k.split("_")[0] in tasks})
        djs.write_text(json.dumps(merged, indent=2))
    # 3) per-molecule dumps: drop affected datasets' rows, append the new ones
    dtp, ntp = dest / "test_predictions.csv", tmp / "test_predictions.csv"
    if dtp.exists() and ntp.exists():
        dd2, nn2 = pd.read_csv(dtp), pd.read_csv(ntp)
        pd.concat([dd2[~dd2.dataset.isin(tasks)], nn2[nn2.dataset.isin(tasks)]], ignore_index=True).to_csv(dtp, index=False)


def process_run(wave: str, run_dir: Path, force: bool, dry: bool) -> str:
    run = run_dir.name
    if "e2e" in run:
        return "skip-e2e"
    enc = resolve_encoder(wave, run)
    did = []
    for split, cv in (("moleculenet_cv", True), ("moleculenet", False)):
        sd = run_dir / split
        cfg = read_cfg(sd)
        if cfg is None or not cfg["datasets"]:
            continue
        if cfg["featurizer"] == "encoder" and enc is None:
            return "no-encoder"
        marker = sd / ".corrected_v2.json"
        if marker.exists() and not force:
            did.append(f"{split}:cached"); continue
        if dry:
            did.append(f"{split}:would-run[{cfg['featurizer']}/{cfg['head']} {cfg['datasets']}]"); continue
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ok = run_eval(sd, cfg, enc, TOK_DEFAULT, cv, tmp)
            if not ok:
                return f"FAIL:{split}"
            merge(sd, tmp, cfg["datasets"])
            marker.write_text(json.dumps({"corrected": cfg["datasets"], "ts": int(time.time())}))
            did.append(f"{split}:done")
    return ",".join(did) if did else "no-splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process at most N runs (debug)")
    args = ap.parse_args()

    n = 0
    for wave in args.waves:
        wdir = DATA / wave
        runs = sorted(p for p in wdir.iterdir() if p.is_dir() and not p.name.startswith("_"))
        print(f"\n=== {wave}: {len(runs)} runs ===", flush=True)
        for rd in runs:
            if args.limit and n >= args.limit:
                print("(limit reached)"); return
            t0 = time.time()
            status = process_run(wave, rd, args.force, args.dry_run)
            print(f"[{wave}] {rd.name:42} {status:28} ({time.time()-t0:.0f}s)", flush=True)
            n += 1
    print(f"\ndone: {n} runs processed", flush=True)


if __name__ == "__main__":
    main()
