#!/usr/bin/env bash
# e2e_random_01 / _02 Tox21, end-to-end, IN THE REFERENCE ENVIRONMENT.
#
# Closes the last replication gap in the six panels: e2e_no_pretrain rests on ONE dir on Tox21
# where it has three everywhere else, because the existing _01/_02 copies came from the August
# boxes and carry the ~0.0075 environment offset.
#
# THE ENVIRONMENT IS THE WHOLE POINT. The August drift was deepchem 2.8.0 having no py3.12 wheel;
# on THIS AMI the system python is 3.9, where 2.8.0 installs fine, so the reference parse is
# reachable here -- it just has to be pinned and then PROVEN before a single GPU-hour is spent.
#
# Two gates, both before anything is written:
#   PREFLIGHT  the parse must yield 7,823 Tox21 molecules and 77,864 non-missing label cells.
#              If it does not, STOP. Do not adjust, do not write. A third unusable copy is worse
#              than no copy.
#   POSTCHECK  the prediction dump must carry exactly 77,864 Tox21 rows.
# And reference_scoring.json is never written -- its presence is what disqualifies a dir.
# e2e_random_00 is never touched: it is the one honest dir the current panel rests on.
set -uo pipefail
cd /home/ec2-user/CLIMB
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
PY=$HOME/venvs/ref39/bin/python
TOK=${TOK:-experiments/_tok_t21e2e}
REF_MOLS=7823
REF_CELLS=77864
LOG=analysis/tox21_e2e_reeval.log
mkdir -p analysis
say(){ echo "[t21e2e $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG" || {
  say "FATAL no GPU -> staying UP"; exit 1; }

if [ ! -x "$PY" ]; then
  say "building the PINNED reference venv on system python3.9"
  python3.9 -m venv "$HOME/venvs/ref39" || { say "FATAL no python3.9 -> staying UP"; exit 1; }
  $HOME/venvs/ref39/bin/pip -q install --upgrade pip
  $HOME/venvs/ref39/bin/pip -q install "numpy==2.0.2" "rdkit==2025.9.2" "scikit-learn==1.6.1" \
      "deepchem==2.8.0" torch transformers xgboost pandas safetensors \
    || { say "FATAL pinned install failed -> staying UP"; exit 1; }
fi
$PY -c "import numpy,rdkit,sklearn,deepchem,torch
print('numpy',numpy.__version__,'rdkit',rdkit.__version__,'sklearn',sklearn.__version__,
      'deepchem',deepchem.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  2>&1 | tee -a "$LOG" || { say "FATAL pinned imports broken -> staying UP"; exit 1; }

aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors

# ---- PREFLIGHT: prove the parse BEFORE spending GPU time -------------------------------------
say "preflight: checking the Tox21 parse against the reference"
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, numpy as np
sys.path.insert(0, ".")
from eval_v2 import _load_moleculenet_full
smiles, y = _load_moleculenet_full("Tox21")
y = np.asarray(y, dtype=float)
cells = int(np.sum(~np.isnan(y)))
print(f"PREFLIGHT molecules={len(smiles)} non_missing_cells={cells}")
sys.exit(0 if (len(smiles) == 7823 and cells == 77864) else 3)
PYEOF
if [ "${PIPESTATUS[0]}" != "0" ]; then
  say "PREFLIGHT FAILED -- this box does NOT reproduce the reference parse. Nothing written, "
  say "no GPU spent. Report the two numbers above; do not adjust and re-run."
  exit 1
fi
say "preflight PASSED: 7823 molecules / 77864 cells"

ok=1
for r in e2e_random_01 e2e_random_02; do
  src=$(echo "$r" | sed 's/e2e_random/random_baseline/')
  d=figure_data/climb_v2_phase2/$r
  mkdir -p "$d" figure_data/climb_v2_phase2/$src
  aws s3 sync "$S3/$src/encoder" "figure_data/climb_v2_phase2/$src/encoder" --only-show-errors
  [ -f figure_data/climb_v2_phase2/$src/encoder/model.safetensors ] || {
    say "$r: no encoder for $src -> skipping"; ok=0; continue; }

  tmp=$(mktemp -d)
  say "$r: fine-tuning Tox21 end-to-end from $src (5-fold scaffold CV)"
  $PY - "$src" "$tmp" "$TOK" >> "$LOG" 2>&1 <<'PYEOF'
import sys
sys.path.insert(0, ".")
from finetune_e2e_v2 import evaluate_finetuned
src, out, tok = sys.argv[1], sys.argv[2], sys.argv[3]
evaluate_finetuned(encoder_path=f"figure_data/climb_v2_phase2/{src}/encoder",
                   tokenizer_path=tok, output_dir=out, seeds=[0],
                   datasets=[("Tox21", "classification")], cv_folds=5, subsample_seed=0)
PYEOF
  n=$( [ -f "$tmp/test_predictions.csv" ] && awk -F, '$1=="Tox21"' "$tmp/test_predictions.csv" | wc -l | tr -d ' ' || echo 0 )
  folds=$(grep -c "^Tox21,.*,fold[0-9]," "$tmp/moleculenet_summary.csv" 2>/dev/null || echo 0)
  if [ "$n" != "$REF_CELLS" ] || [ "$folds" -lt 5 ]; then
    say "$r: POSTCHECK FAILED rows=$n (want $REF_CELLS) folds=$folds -- NOTHING WRITTEN"
    ok=0; rm -rf "$tmp"; continue
  fi
  dest="$d/moleculenet_cv_tox21fixed"
  mkdir -p "$dest"
  cp "$tmp/moleculenet_summary.csv" "$dest/moleculenet_summary.csv"
  cp "$tmp/test_predictions.csv"    "$dest/test_predictions.csv"
  [ -f "$tmp/suite_summary.json" ] && cp "$tmp/suite_summary.json" "$dest/suite_summary.json"
  rm -f "$dest/reference_scoring.json" "$dest/test_predictions.reference_set.csv"
  v=$(awk -F, '$1=="Tox21" && $8=="MEAN" && $7=="roc_auc"{printf "%.4f",$10}' "$dest/moleculenet_summary.csv")
  say "$r: OK  $n rows (== reference)  Tox21 roc_auc=$v"
  aws s3 sync "$dest" "$S3/$r/moleculenet_cv_tox21fixed" --delete --only-show-errors
  rm -rf "$tmp"
done

if [ "$ok" = "1" ]; then say "COMPLETE -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE -> staying UP for inspection"; fi
