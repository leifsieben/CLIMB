#!/usr/bin/env bash
# Two CheMeleon jobs on one box, using the SPLIT-FEATURIZATION pattern.
#
#   1. head comparison: chemeleon_frozen scored with an XGBoost probe (it already has the MLP half)
#   2. fig_F: CheMeleon in the concatenation test -- CheMel / desc+CheMel / fp+desc+CheMel
#
# WHY TWO VENVS. CheMeleon needs chemprop>=2.2 (python>=3.11); deepchem 2.8.0, which defines our
# Tox21 parse, has no 3.12 wheel. So py3.12 turns SMILES into vectors ONCE, and every decision that
# produces a number -- dataset parse, folds, scoring -- happens in the pinned py3.9 reference env
# reading that table. Same fix that made the chemeleon_frozen replicates comparable.
#
# FROZEN ONLY. chemeleon_e2e is never used here: it fine-tunes its whole network per task, so it is
# not a probe, and putting it on a probe axis would compare a fine-tune against three probes.
set -uo pipefail
cd /home/ec2-user/CLIMB
S3=s3://climb-s3-bucket/experiments
CF=$HOME/venvs/cf/bin/python          # py3.12 + chemprop, featurization ONLY
REF=$HOME/venvs/ref39/bin/python      # py3.9 pinned reference env, everything that scores
NPZ=figure_data/_chemeleon_allmols.npz
TASKS="ESOL QM7 BBBP BACE Tox21 HIV"
LOG=analysis/chemeleon_jobs.log
mkdir -p analysis figure_data
say(){ echo "[cjobs $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -x "$REF" ]; then
  say "building pinned reference venv (py3.9)"
  python3.9 -m venv "$HOME/venvs/ref39" && $HOME/venvs/ref39/bin/pip -q install --upgrade pip && \
  $HOME/venvs/ref39/bin/pip -q install "numpy==2.0.2" "rdkit==2025.9.2" "scikit-learn==1.6.1" \
      "deepchem==2.8.0" torch transformers xgboost pandas safetensors \
    || { say "FATAL ref venv failed -> staying UP"; exit 1; }
fi
if [ ! -x "$CF" ]; then
  say "building chemprop venv (py3.12)"
  python3.12 -m venv "$HOME/venvs/cf" && $HOME/venvs/cf/bin/pip -q install --upgrade pip && \
  $HOME/venvs/cf/bin/pip -q install "rdkit==2025.9.2" "chemprop>=2.2.0" torch numpy \
    || { say "FATAL cf venv failed -> staying UP"; exit 1; }
fi
$REF -c "import deepchem,rdkit,numpy;print('ref  deepchem',deepchem.__version__,'rdkit',rdkit.__version__)" | tee -a "$LOG"
$CF  -c "import chemprop,rdkit;print('cf   chemprop',chemprop.__version__,'rdkit',rdkit.__version__)" | tee -a "$LOG"

for p in datasets tokenizer_10M; do :; done
aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
aws s3 sync s3://climb-s3-bucket/datasets chemeleon_suite/data --only-show-errors
n_mace=$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l | tr -d ' ')
n_pol=$(ls chemeleon_suite/data/polaris/ 2>/dev/null | wc -l | tr -d ' ')
say "staged suite data: $n_mace moleculeace, $n_pol polaris entries"
[ "$n_mace" -ge 30 ] || { say "FATAL suite data incomplete -> staying UP"; exit 1; }

# ---- 1. collect every SMILES both jobs will need, IN THE REFERENCE ENV -------------------------
if [ ! -f "$NPZ" ]; then
  say "collecting SMILES in the reference environment"
  $REF - <<'PYEOF' 2>&1 | tail -3 | tee -a "$LOG"
import csv, glob, json, sys
sys.path.insert(0, ".")
from eval_v2 import _load_moleculenet_full, register_custom_task
mols = set()
for ds in ["ESOL", "QM7", "BBBP", "BACE", "Tox21", "HIV"]:
    s, _ = _load_moleculenet_full(ds); mols |= set(map(str, s))
for f in glob.glob("chemeleon_suite/data/moleculeace/*.csv") + glob.glob("chemeleon_suite/data/polaris/*/*.csv"):
    rows = list(csv.DictReader(open(f)))
    if not rows: continue
    col = next((c for c in rows[0] if c.lower() in ("smiles", "canonical_smiles")), None)
    if col: mols |= {str(r[col]) for r in rows if r[col]}
if __import__("os").path.exists("data/cbs.csv"):
    register_custom_task("cbs", "data/cbs.csv")
    s, _ = _load_moleculenet_full("cbs"); mols |= set(map(str, s))
mols = sorted(mols)
json.dump({"_all_unique": mols}, open("figure_data/_chemeleon_allmols.json", "w"))
print(f"COLLECTED {len(mols)} unique SMILES")
PYEOF
  say "embedding with CheMeleon (py3.12)"
  # PYTHONPATH=. because embed_chemeleon_box.py imports chemeleon_fingerprint from the REPO ROOT
  # and running it as scripts/... puts scripts/ on sys.path instead. It worked before only because
  # the file happened to be invoked from the root.
  PYTHONPATH=. $CF scripts/embed_chemeleon_box.py figure_data/_chemeleon_allmols.json "$NPZ" \
    2>&1 | tail -2 | tee -a "$LOG"
  [ -f "$NPZ" ] || { say "FATAL no npz -> staying UP"; exit 1; }
  aws s3 cp "$NPZ" "$S3/_chemeleon_allmols.npz" --only-show-errors
fi

export CONCAT_FEATURES_NPZ=$NPZ FP_VARIANT=ecfp4_stereo
ok=1

# ---- 2. head comparison: chemeleon_frozen with an XGBoost probe --------------------------------
out=figure_data/climb_v2_phase2/chemeleon_frozen__xgb/moleculenet_cv
say "head comparison: chemeleon_frozen __ xgb"
$REF eval_v2.py --output_dir "$out" --featurizer chemeleon --features_npz "$NPZ" --head xgb \
    --standardize zscore --cv_folds 5 --head_seeds 0 1 2 --datasets $TASKS >> "$LOG" 2>&1
grep -q "^BACE,.*,roc_auc,fold0," "$out/moleculenet_summary.csv" 2>/dev/null \
  && { say "chemeleon_frozen__xgb OK"; aws s3 sync "$(dirname $out)" "$S3/climb_v2_phase2/chemeleon_frozen__xgb" --only-show-errors; } \
  || { say "chemeleon_frozen__xgb FAILED"; ok=0; }

# ---- 3. fig_F: CheMeleon in the concatenation test ---------------------------------------------
# BOTH embeddings. fig_F's redesign needs the bare bases (fp, desc) and fp+CLM as well as the
# CheMeleon pairs, and feature_sets now emits all seven blocks per embedding, so one pass each
# covers every cell the new layout asks for.
for emb in climb chemeleon; do
  say "fig_F concat, MolNet, CONCAT_EMB=$emb"
  CONCAT_EMB=$emb CONCAT_OUT=concat_redundancy_${emb}_v2.csv \
    $REF scripts/concat_redundancy.py >> "$LOG" 2>&1 && say "concat MolNet $emb OK" \
    || { say "concat MolNet $emb FAILED"; ok=0; }
  say "fig_F concat, panels, CONCAT_EMB=$emb"
  CONCAT_EMB=$emb CONCAT_PANEL_OUT=concat_panels_${emb}_v2.csv \
    $REF scripts/concat_redundancy_panels.py >> "$LOG" 2>&1 && say "concat panels $emb OK" \
    || { say "concat panels $emb FAILED"; ok=0; }
done
aws s3 sync analysis/rigor "$S3/analysis_rigor" --exclude "*" --include "concat_*_v2.csv" --only-show-errors

if [ "$ok" = "1" ]; then say "COMPLETE -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE -> staying UP for inspection"; fi
