"""Re-evaluate Tox21 for the runs whose PREDICTIONS predate the 2026-08-05 masking fix.

These 20 runs cannot be repaired from disk. Their test_predictions.csv carry 93,876 Tox21 rows
(the unmasked cell count) instead of 77,864, so the 16,012 missing-label cells DeepChem encodes as
y=0,w=0 were scored as true inactives. No re-scoring of that dump can undo it -- the corrected
number has to come from a fresh eval against the checkpoint.

Non-destructive: everything is written to moleculenet_cv_tox21fixed/ beside the original, matching
the subdir the figure layer resolves through. moleculenet_cv/ is never modified.

Protocol fidelity: featurizer / head / pool / standardize are read from each run's OWN existing
Tox21 summary row, so the re-eval reproduces that run's protocol rather than imposing a default.

THE GATE IS THE POINT: a result is accepted only if the freshly written dump has exactly 77,864
Tox21 rows. That is positive proof the masking fix was in force for this eval. Without it we would
be trusting that the code on the box is current -- which is the assumption that produced this whole
incident.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments"
TOK = "figure_data/_tokenizer"
MASKED_ROWS = 77864
OUTSUB = "moleculenet_cv_tox21fixed"
LOG = ROOT / "analysis" / "tox21_reeval.log"
LOG.parent.mkdir(exist_ok=True)

# e2e arms have no encoder of their own: they fine-tune the SAME saved weights the frozen control
# was scored at, exactly as run_e2e_random.py does.
E2E_FROM = {"e2e_random_00": "random_baseline_00",
            "e2e_random_01": "random_baseline_01",
            "e2e_random_02": "random_baseline_02"}


def log(m):
    print(f"[t21] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[t21] {m}\n")


def sh(c):
    return subprocess.run(c, check=False)


def run_cfg(run_dir: Path):
    """featurizer/pool/standardize/head from this run's own Tox21 row."""
    p = run_dir / "moleculenet_cv" / "moleculenet_summary.csv"
    for r in csv.reader(p.open()):
        if len(r) > 6 and r[0] == "Tox21":
            return dict(featurizer=r[2], pool=r[3], standardize=r[4], head=r[5])
    return None


def stage_encoder(run: str, waves):
    enc = ROOT / "figure_data" / "_stage_t21" / run / "encoder"
    if (enc / "model.safetensors").exists():
        return str(enc)
    enc.mkdir(parents=True, exist_ok=True)
    for w in waves:
        sh(["aws", "s3", "sync", f"{S3B}/{w}/{run}/encoder", str(enc), "--only-show-errors"])
        if (enc / "model.safetensors").exists():
            return str(enc)
    return None


def masked_ok(out_dir: Path):
    p = out_dir / "test_predictions.csv"
    if not p.exists():
        return False, 0
    n = sum(1 for line in p.open() if line.startswith("Tox21,"))
    return n == MASKED_ROWS, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="/tmp/reeval20.txt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()

    if not (ROOT / TOK / "tokenizer.json").exists():
        (ROOT / TOK).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK, "--only-show-errors"])

    items = [l.strip() for l in Path(a.list).read_text().split() if l.strip()]
    items = [x for i, x in enumerate(items) if i % a.nshard == a.shard]
    log(f"START {len(items)} runs (shard {a.shard}/{a.nshard}) -> {OUTSUB}")
    ok = 0
    for wr in items:
        wave, run = wr.split("/", 1)
        base = ROOT / "figure_data" / wave / run
        out = base / OUTSUB
        good, n = masked_ok(out)
        if good:
            log(f"SKIP {wr} (already masked, {n} rows)"); ok += 1; continue
        cfg = run_cfg(base)
        if not cfg:
            log(f"ERROR {wr}: no Tox21 row to read protocol from"); continue
        out.mkdir(parents=True, exist_ok=True)

        if run in E2E_FROM:
            src = E2E_FROM[run]
            enc = stage_encoder(src, [wave, "climb_v2_phase2", "climb_v2_ablation_dedup"])
            if not enc:
                log(f"ERROR {wr}: no encoder {src}"); continue
            log(f"e2e Tox21: {wr} (fine-tune from {src})")
            sh([PY, "finetune_e2e_v2.py", "--encoder", enc, "--tokenizer", TOK,
                "--output_dir", str(out), "--datasets", "Tox21", "--seeds", "0", "1", "2",
                "--cv_folds", "5", "--cv_scheme", "scaffold", "--subsample_seed", "0"])
        else:
            cmd = [PY, "eval_v2.py", "--output_dir", str(out), "--datasets", "Tox21",
                   "--featurizer", cfg["featurizer"], "--head", cfg["head"],
                   "--head_seeds", "0", "1", "2", "--cv_folds", "5", "--cv_scheme", "scaffold"]
            if cfg["pool"] in ("cls", "mean", "cls_mean"):
                cmd += ["--pool", cfg["pool"]]
            if cfg["featurizer"] == "encoder":
                enc = stage_encoder(run, [wave, "climb_v2_phase2", "climb_v2_ablation_dedup"])
                if not enc:
                    log(f"ERROR {wr}: no encoder"); continue
                cmd += ["--encoder", enc, "--tokenizer", TOK]
            log(f"frozen Tox21: {wr} ({cfg['featurizer']}/{cfg['head']})")
            sh(cmd)

        good, n = masked_ok(out)
        if good:
            sh(["aws", "s3", "cp", "--recursive", str(out),
                f"{S3B}/{wave}/{run}/{OUTSUB}", "--only-show-errors"])
            log(f"OK {wr} ({n} masked rows)"); ok += 1
        else:
            # never leave a wrong artefact wearing the 'fixed' name
            bad = base / f"{OUTSUB}.REJECTED_{n}rows"
            sh(["rm", "-rf", str(bad)])
            out.rename(bad)
            log(f"FAIL {wr}: dump had {n} Tox21 rows, expected {MASKED_ROWS} -> rejected")
        sh(["rm", "-rf", str(ROOT / "figure_data" / "_stage_t21" / run)])
    log(f"DONE {ok}/{len(items)}")
    if ok == len(items):
        (ROOT / "figure_data" / f"TOX21_REEVAL_DONE_{a.shard}").write_text("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
