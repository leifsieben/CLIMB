"""hERG (Polaris tdcommons/herg) on every scaling encoder — the top Fig-B blocker. hERG test
labels are withheld locally, so scoring must go through Polaris's benchmark.evaluate() in the
.venv_polaris (py3.12) env — a two-step per encoder:
  1) CLIMB venv: chemeleon_suite_run.py --track polaris  -> test_predictions.csv (herg only;
     polaris_tasks.txt has been reduced to the single line "tdcommons/herg" on the box)
  2) .venv_polaris: chemeleon_suite_score_polaris.py      -> polaris_scores.csv (roc_auc/seed)
Validated: reproduces the mainline unsup_8M herg (0.7996/0.8346/…) exactly.

Writes figure_data/chemeleon_suite/polaris/<prefix>/polaris_scores.csv (what the aggregator reads,
filter task=="tdcommons/herg", metric=="roc_auc"). Idempotent, S3-synced, gated marker.
"""
from __future__ import annotations
import csv, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
CLIMB_PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
POLARIS_PY = os.environ.get("POLARIS_PY", str(ROOT / ".venv_polaris" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments"
TOK_STD = "figure_data/_tokenizer"
STAGE = ROOT / "figure_data" / "_stage_herg"
LOG = ROOT / "analysis" / "six_panel_herg.log"
LOG.parent.mkdir(exist_ok=True)


def log(m):
    print(f"[herg] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(f"[herg] {m}\n")


def sh(c):
    return subprocess.run(c, check=False)


def _manifest():
    m = []
    a2 = {
        "unsup": ["unsup_2M", "unsup_8M", "unsup_24M", "unsup_48M", "unsup_50M", "unsup_100M"],
        "skip_dense": ["skip_dense_2M", "skip_dense_8M", "skip_dense_24M", "skip_dense_48M", "skip_dense_96M"],
        "skip_sparse_all": ["skip_sparse_all_2M", "skip_sparse_all_8M", "skip_sparse_all_24M", "skip_sparse_all_48M"],
        "skip_dps": ["skip_dense_plus_sparse_2M", "skip_dense_plus_sparse_8M", "skip_dense_plus_sparse_24M", "skip_dense_plus_sparse_48M"],
        "u2s_dense": ["u2s_dense_from2M", "u2s_dense_from8M", "u2s_dense_from24M", "u2s_dense_from48M"],
    }
    for prefs in a2.values():
        for p in prefs:
            m.append((p, "climb_v2_phase2", False))
    for mode in ("canonical", "enumerated"):
        for frac in ("0p001", "0p01", "0p1", "0p3", "full"):
            for s in ("s0", "s1", "s2"):
                m.append((f"scaling_{mode}_frac{frac}_{s}", "climb_v2_h1", False))  # standard vocab
    for p in ("bpe_261", "bpe_1000", "bpe_3000", "bpe_12000", "unigram_261", "unigram_700", "unigram_1200", "unigram_3000"):
        m.append((p, "climb_v2_vocab", True))  # own tokenizer (vocab differs)
    # the two remaining A2/six-panel cells with no Polaris coverage at all
    for p in ("s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2"):
        m.append((p, "climb_v2_phase2", False))
    return m


def _stage(prefix, wave, own_tok):
    dst = STAGE / prefix
    enc = dst / "encoder"
    if not (enc / "model.safetensors").exists():
        enc.mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", f"{S3B}/{wave}/{prefix}/encoder", str(enc), "--only-show-errors"])
    tok = TOK_STD
    if own_tok:
        td = dst / "tokenizer"
        if not (td / "tokenizer.json").exists():
            td.mkdir(parents=True, exist_ok=True)
            sh(["aws", "s3", "sync", f"{S3B}/{wave}/{prefix}/tokenizer", str(td), "--only-show-errors"])
        tok = str(td)
    return str(enc), tok


SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))


SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))


def _herg_done(prefix):
    """Full-track coverage. 2026-08-18 the Polaris panel moved hERG -> Ames; scoring one named task
    means a panel swap invalidates the whole wave (these encoders had hERG and NO Ames for exactly
    that reason). All 28 tasks is 47k molecules -- the same order as MoleculeACE."""
    f = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / prefix / "polaris_scores.csv"
    try:
        return len({r["task"] for r in csv.DictReader(open(f))}) >= 20
    except Exception:
        return False


def _would_clobber(prefix):
    """True if this prefix already holds a MULTI-task Polaris dir.

    chemeleon_suite_run.py REWRITES results.csv / test_predictions.csv for exactly the tasks in
    polaris_tasks.txt — it does not merge. This driver reduces that file to the single line
    "tdcommons/herg", so pointing it at a prefix that already has the full 28-task Polaris run
    destroys the other 27 (both the scores AND the per-seed predictions).

    That is what happened on 2026-08-16 to skip_dense_8M, skip_dense_plus_sparse_8M,
    skip_sparse_all_8M, unsup_8M and u2s_dense_from8M (notes/polaris-clobber-recovery-2026-08-17.md):
    only the means survived, in an older summary CSV. Refuse rather than repeat it — a prefix that
    already has full Polaris coverage already has hERG in it.
    """
    f = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / prefix / "polaris_scores.csv"
    if not f.exists():
        return False
    try:
        return len({r["task"] for r in csv.DictReader(open(f))}) > 1
    except Exception:
        return False


def main():
    if not (ROOT / TOK_STD / "tokenizer.json").exists():
        (ROOT / TOK_STD).mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", "s3://climb-s3-bucket/tokenizer_10M", TOK_STD, "--only-show-errors"])
    man = _manifest()
    man = [m for i, m in enumerate(man) if i % NSHARD == SHARD]
    log(f"START full-Polaris on {len(man)} encoders (shard {SHARD}/{NSHARD})")
    ok = 0
    for prefix, wave, own_tok in man:
        if _herg_done(prefix):
            log(f"SKIP {prefix} (herg done)"); ok += 1; continue
        if False:  # full track is a superset of any single-task run
            log(f"REFUSE {prefix}: already has a full multi-task Polaris dir — running the "
                f"herg-only task list here would destroy the other 27 tasks. Skipping.")
            ok += 1
            continue
        enc, tok = _stage(prefix, wave, own_tok)
        if not Path(enc, "model.safetensors").exists():
            log(f"ERROR {prefix}: encoder missing after sync"); continue
        outdir = f"figure_data/chemeleon_suite/polaris/{prefix}"
        log(f"predict herg: {prefix}")
        r1 = sh([CLIMB_PY, "scripts/chemeleon_suite_run.py", "--track", "polaris", "--featurizer",
                 "encoder", "--model", prefix, "--encoder", enc, "--tokenizer", tok,
                 "--head", "mlp", "--seeds", "42", "117", "709"])
        if r1.returncode == 0:
            log(f"score herg: {prefix}")
            sh([POLARIS_PY, "scripts/chemeleon_suite_score_polaris.py", outdir])
        if _herg_done(prefix):
            sh(["aws", "s3", "cp", "--recursive", outdir, f"{S3B}/chemeleon_suite/polaris/{prefix}",
                "--only-show-errors"])
            ok += 1
        else:
            log(f"FAIL {prefix}: no herg score produced")
        sh(["rm", "-rf", str(STAGE / prefix)])
    log(f"DONE {ok}/{len(man)}")
    if ok == len(man):
        (ROOT / "figure_data" / "SIX_PANEL_HERG_DONE").write_text("herg on all scaling encoders\n")
        log("SIX_PANEL_HERG_DONE written")


if __name__ == "__main__":
    main()
