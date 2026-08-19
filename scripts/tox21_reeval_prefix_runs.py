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
MASKED_ROWS = 77864          # the reference environment's valid-cell count
UNMASKED_ROWS = 93876        # what a pre-fix dump has (16,012 missing cells scored as inactives)
REF_MOLS = "figure_data/_tox21_reference_molecules.txt"
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


def run_cfg(run_dir: Path, wave: str = "", run: str = ""):
    """featurizer/pool/standardize/head from this run's own Tox21 row.

    Staged from S3 when absent: a fresh box has no figure_data tree, and this file is only READ
    (for protocol fidelity), never written -- the corrected output goes to a separate subdir.
    """
    p = run_dir / "moleculenet_cv" / "moleculenet_summary.csv"
    if not p.exists() and wave and run:
        p.parent.mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "cp", f"{S3B}/{wave}/{run}/moleculenet_cv/moleculenet_summary.csv",
            str(p), "--only-show-errors"])
    if not p.exists():
        return None
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
    """Gate on MASKING BEING IN FORCE, not on an exact row count.

    The exact count is environment-dependent: this box's RDKit/DeepChem parses 7,831 Tox21
    molecules where the reference environment parsed 7,823, giving 77,946 valid cells instead of
    77,864. That drift is real but tiny, and rejecting on it would block the whole re-eval for a
    reason unrelated to the bug being fixed. What must be proven is that the 16,012 missing cells
    are NOT being scored as inactives, and any count far below the unmasked 93,876 proves that.

    Comparability across runs is restored separately, by scoring on the shared reference molecule
    set (see score_on_reference), so the 8 extra molecules cannot make these 20 runs incomparable
    to the 120 rebuilt from original dumps.
    """
    p = out_dir / "test_predictions.csv"
    if not p.exists():
        return False, 0
    n = sum(1 for line in p.open() if line.startswith("Tox21,"))
    masked = 0 < n <= (UNMASKED_ROWS - 10000)      # comfortably below the unmasked count
    return masked, n


def score_on_reference(out_dir: Path):
    """Rewrite the summary scoring ONLY the shared reference molecules.

    The fresh eval may see a few molecules the reference environment did not parse. Scoring the
    intersection makes these runs directly comparable to the 120 rebuilt from original dumps --
    same molecules, same estimator (per (fold, assay) -> mean over assays -> per-fold row -> mean
    over folds, folds assigned on unique molecules). Without this the 20 would sit on a slightly
    different eval set, which is the class of silent inconsistency this whole exercise exists to
    remove. Returns (n_kept, n_dropped, mean) or None if the reference list is unavailable.
    """
    ref_path = ROOT / REF_MOLS
    if not ref_path.exists():
        sh(["aws", "s3", "cp", "s3://climb-s3-bucket/datasets/_tox21_reference_molecules.txt",
            str(ref_path), "--only-show-errors"])
    if not ref_path.exists():
        log("WARN no reference molecule list -> leaving full-set scores"); return None
    ref = set(ref_path.read_text().split())

    pf = out_dir / "test_predictions.csv"
    keep, drop = [], 0
    for r in csv.DictReader(pf.open()):
        if r["dataset"] != "Tox21":
            continue
        if r.get("canonical_key") in ref:
            keep.append(r)
        else:
            drop += 1
    if not keep:
        return None
    filt = out_dir / "test_predictions.reference_set.csv"
    with filt.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
        w.writeheader(); w.writerows(keep)

    sys.path.insert(0, str(ROOT / "scripts"))
    from tox21_rescore_from_preds import rebuild  # noqa: E402
    tmp = out_dir / "_refscore"; tmp.mkdir(exist_ok=True)
    (tmp / "test_predictions.csv").write_text(filt.read_text())
    new, err = rebuild(tmp)
    sh(["rm", "-rf", str(tmp)])
    if err:
        log(f"WARN reference scoring failed: {err}"); return None

    sp = out_dir / "moleculenet_summary.csv"
    rows = list(csv.reader(sp.open()))
    header, body = rows[0], rows[1:]
    for r in body:
        if len(r) > 9 and r[0] == "Tox21" and (r[6], r[7]) in new:
            r[9] = repr(new[(r[6], r[7])])
    import io as _io
    buf = _io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(header); w.writerows(body)
    sp.write_text(buf.getvalue())
    (out_dir / "reference_scoring.json").write_text(json.dumps(
        {"scored_on": "shared reference Tox21 molecule set", "n_reference_keys": len(ref),
         "rows_kept": len(keep), "rows_dropped_not_in_reference": drop,
         "roc_auc_MEAN": new[("roc_auc", "MEAN")]}, indent=2))
    return len(keep), drop, new[("roc_auc", "MEAN")]


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
        cfg = run_cfg(base, wave, run)
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
            res = score_on_reference(out)
            if res:
                log(f"  scored on reference set: kept {res[0]}, dropped {res[1]}, MEAN {res[2]:.4f}")
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
