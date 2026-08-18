"""(a) Fig-E arms on the canonical panels + (b) hERG on the mainline pretraining-seed replicates.

(a) 13 corrupted/synthetic-pretraining encoders were only ever evaluated on MoleculeNet, so Fig E
    cannot use the canonical six panels. Frozen-probe them on MoleculeACE + CBS + hERG.
(b) hERG currently has ONE Polaris run per mainline arm, so its A2 whisker degenerates to head-seed
    noise on a single 132-molecule split. Scoring the existing _s1/_s2 pretraining replicates
    upgrades it from 3 to 9 cells and restores the pretraining-seed component. (Split variance stays
    structurally unavailable — Polaris withholds test labels; that belongs in the caption.)

Frozen probes only, no pretraining. Writes exactly where the figure layer globs:
    MoleculeACE -> figure_data/chemeleon_suite/moleculeace/<run>/results.csv
    hERG        -> figure_data/chemeleon_suite/polaris/<run>/polaris_scores.csv
    CBS         -> figure_data/cbs_benchmark/<run>/moleculenet_cv/
Idempotent, S3-synced, refuses to clobber a full multi-task Polaris dir.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = os.environ.get("CLIMB_PY", str(Path.home() / "venvs" / "climb" / "bin" / "python"))
POLARIS_PY = os.environ.get("POLARIS_PY", str(ROOT / ".venv_polaris" / "bin" / "python"))
S3B = "s3://climb-s3-bucket/experiments"
TOK = "figure_data/_tokenizer"
STAGE = ROOT / "figure_data" / "_stage_figE"
LOG = ROOT / "analysis" / "figE_herg.log"; LOG.parent.mkdir(exist_ok=True)

# (a) Fig-E arms: (run_label, s3_wave)
FIGE = [("corrupt_mlm_8M", "climb_v2_phase2"), ("corrupt_mtr_8M", "climb_v2_phase2"),
        ("corrupt_mlm_8M_s1", "climb_v2_expA"), ("corrupt_mlm_8M_s2", "climb_v2_expA"),
        ("unigram_8M", "climb_v2_expA"), ("unigram_8M_s1", "climb_v2_expA"), ("unigram_8M_s2", "climb_v2_expA"),
        ("bigram_8M", "climb_v2_expA"), ("bigram_8M_s1", "climb_v2_expA"), ("bigram_8M_s2", "climb_v2_expA"),
        ("wiki_real_8M", "climb_v2_expB"), ("wiki_real_8M_s1", "climb_v2_expB"), ("wiki_real_8M_s2", "climb_v2_expB")]
# (b) mainline pretraining-seed replicates needing hERG
MAIN = ["unsup_8M", "skip_dense_8M", "skip_dense_plus_sparse_8M", "skip_sparse_all_8M",
        "skip_mixed_8M", "skip_minimol_full_8M", "u2s_dense_from8M", "u2s_dense_plus_sparse_from8M",
        "u2s_sparse_all_from8M", "u2s_mixed_from8M", "u2s_minimol_full_from8M"]
HERG_SEEDS = [(f"{a}_{s}", "climb_v2_phase2") for a in MAIN for s in ("s1", "s2")]


def log(m):
    print(f"[figE] {m}", flush=True)
    with LOG.open("a") as f: f.write(f"[figE] {m}\n")

def sh(c): return subprocess.run(c, check=False)

def stage(run, wave):
    enc = STAGE / run / "encoder"
    if not (enc / "model.safetensors").exists():
        enc.mkdir(parents=True, exist_ok=True)
        sh(["aws", "s3", "sync", f"{S3B}/{wave}/{run}/encoder", str(enc), "--only-show-errors"])
    return str(enc)

def mace_done(r): return (ROOT/"figure_data/chemeleon_suite/moleculeace"/r/"verified.json").exists()
def cbs_done(r):
    p = ROOT/"figure_data/cbs_benchmark"/r/"moleculenet_cv/suite_summary.json"
    try: return p.exists() and json.loads(p.read_text()).get("cbs_nef1_MEAN") is not None
    except Exception: return False
def herg_done(r):
    f = ROOT/"figure_data/chemeleon_suite/polaris"/r/"polaris_scores.csv"
    try: return f.exists() and any(x["task"]=="tdcommons/herg" for x in csv.DictReader(open(f)))
    except Exception: return False
def herg_would_clobber(r):
    f = ROOT/"figure_data/chemeleon_suite/polaris"/r/"polaris_scores.csv"
    try: return f.exists() and len({x["task"] for x in csv.DictReader(open(f))}) > 1
    except Exception: return False

def do_mace(run, enc):
    log(f"MoleculeACE {run}")
    r = sh([PY, "scripts/chemeleon_suite_run.py", "--track", "moleculeace", "--featurizer", "encoder",
            "--model", run, "--encoder", enc, "--tokenizer", TOK, "--head", "mlp",
            "--seeds", "42", "117", "709"])
    if r.returncode == 0 and mace_done(run):
        sh(["aws","s3","cp","--recursive",f"figure_data/chemeleon_suite/moleculeace/{run}",
            f"{S3B}/chemeleon_suite/moleculeace/{run}","--only-show-errors"])

def do_herg(run, enc):
    if herg_would_clobber(run):
        log(f"REFUSE herg {run}: full multi-task Polaris dir present"); return
    log(f"hERG {run}")
    out = f"figure_data/chemeleon_suite/polaris/{run}"
    r = sh([PY, "scripts/chemeleon_suite_run.py", "--track", "polaris", "--featurizer", "encoder",
            "--model", run, "--encoder", enc, "--tokenizer", TOK, "--head", "mlp",
            "--seeds", "42", "117", "709"])
    if r.returncode == 0:
        sh([POLARIS_PY, "scripts/chemeleon_suite_score_polaris.py", out])
        if herg_done(run):
            sh(["aws","s3","cp","--recursive",out,f"{S3B}/chemeleon_suite/polaris/{run}","--only-show-errors"])

def do_cbs(run, enc):
    log(f"CBS {run}")
    out = f"figure_data/cbs_benchmark/{run}/moleculenet_cv"
    r = sh([PY, "eval_v2.py", "--encoder", enc, "--tokenizer", TOK, "--output_dir", out,
            "--head","mlp","--head_seeds","0","1","2","--task_csv","data/cbs.csv",
            "--task_name","cbs","--task_type","classification","--cv_folds","5","--cv_scheme","provided"])
    if r.returncode == 0 and cbs_done(run):
        (ROOT/"figure_data/cbs_benchmark"/run/"verified.json").write_text(
            json.dumps({"run":run,"metric":"nef1","cv":"provided-5fold"}))
        sh(["aws","s3","cp","--recursive",f"figure_data/cbs_benchmark/{run}",
            f"{S3B}/cbs_benchmark/{run}","--only-show-errors"])

def main():
    if not (ROOT/TOK/"tokenizer.json").exists():
        (ROOT/TOK).mkdir(parents=True, exist_ok=True)
        sh(["aws","s3","sync","s3://climb-s3-bucket/tokenizer_10M",TOK,"--only-show-errors"])
    log(f"START figE={len(FIGE)} hergSeeds={len(HERG_SEEDS)}")
    # (a) Fig-E: MoleculeACE + hERG first (informative), CBS last (likely at the NEF1 floor)
    for run, wave in FIGE:
        enc = stage(run, wave)
        if not Path(enc,"model.safetensors").exists(): log(f"ERROR {run}: no encoder"); continue
        if not mace_done(run): do_mace(run, enc)
        if not herg_done(run): do_herg(run, enc)
        if not cbs_done(run):  do_cbs(run, enc)
        sh(["rm","-rf",str(STAGE/run)])
    # (b) hERG on mainline pretraining-seed replicates
    for run, wave in HERG_SEEDS:
        if herg_done(run): log(f"SKIP herg {run}"); continue
        enc = stage(run, wave)
        if not Path(enc,"model.safetensors").exists(): log(f"ERROR {run}: no encoder"); continue
        do_herg(run, enc)
        sh(["rm","-rf",str(STAGE/run)])
    nf = sum(1 for r,_ in FIGE if mace_done(r) and herg_done(r))
    nh = sum(1 for r,_ in HERG_SEEDS if herg_done(r))
    log(f"DONE figE(mace+herg)={nf}/{len(FIGE)}  hergSeeds={nh}/{len(HERG_SEEDS)}")
    if nf == len(FIGE) and nh == len(HERG_SEEDS):
        (ROOT/"figure_data/FIGE_HERG_DONE").write_text("done\n"); log("FIGE_HERG_DONE")

if __name__ == "__main__":
    main()
