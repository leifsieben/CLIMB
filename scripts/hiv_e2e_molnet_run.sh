#!/usr/bin/env bash
# SI fig a's empty HIV cell: end-to-end fine-tune on the MoleculeNet CV path.
#
# unsup_8M_e2e and skip_dense_8M_e2e exist but are SUITE-TRACK ONLY -- no moleculenet_cv/ -- so
# there is nothing for the MolNet reader to pick up. This writes that path for all three
# pretraining seeds of both encoders, so the panel can report a real replication depth.
#
# BOTH METRICS, ALWAYS. HIV is the only panel scored on two (nef1 and roc_auc), and that is exactly
# what let build_fig_E_table's .iloc[0] return a ROC-AUC where a NEF1 was expected and publish a
# +41% result for Wikipedia pretraining. eval_v2 emits both rows; nothing here collapses them and
# no bare "HIV" value is written.
set -uo pipefail
cd /home/ec2-user/CLIMB
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
PY=$HOME/venvs/ref39/bin/python
TOK=${TOK:-experiments/_tok_hive2e}
LOG=analysis/hiv_e2e_molnet.log
mkdir -p analysis
say(){ echo "[hive2e $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

nvidia-smi --query-gpu=name --format=csv,noheader | tee -a "$LOG" || { say "FATAL no GPU"; exit 1; }
if [ ! -x "$PY" ]; then
  say "building the pinned reference venv on system python3.9"
  python3.9 -m venv "$HOME/venvs/ref39" || { say "FATAL no python3.9 -> staying UP"; exit 1; }
  $HOME/venvs/ref39/bin/pip -q install --upgrade pip
  $HOME/venvs/ref39/bin/pip -q install "numpy==2.0.2" "rdkit==2025.9.2" "scikit-learn==1.6.1" \
      "deepchem==2.8.0" torch transformers xgboost pandas safetensors \
    || { say "FATAL pinned install failed -> staying UP"; exit 1; }
fi
$PY -c "import numpy,rdkit,deepchem,torch;print('numpy',numpy.__version__,'rdkit',rdkit.__version__,
'deepchem',deepchem.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tee -a "$LOG" \
  || { say "FATAL pinned imports broken -> staying UP"; exit 1; }
aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors

say "preflight: HIV parse"
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys; sys.path.insert(0,".")
from eval_v2 import _load_moleculenet_full
s,_=_load_moleculenet_full("HIV"); print(f"PREFLIGHT HIV molecules={len(s)}")
sys.exit(0 if len(s)>40000 else 3)
PYEOF
[ "${PIPESTATUS[0]}" = "0" ] || { say "PREFLIGHT FAILED -> staying UP, nothing written"; exit 1; }

ok=1; done_n=0
for base in unsup_8M skip_dense_8M; do
  for suf in "" _s1 _s2; do
    src=${base}${suf}; dst=${base}_e2e${suf}
    d=figure_data/climb_v2_phase2/$dst
    mkdir -p "$d" figure_data/climb_v2_phase2/$src
    aws s3 sync "$S3/$src/encoder" "figure_data/climb_v2_phase2/$src/encoder" --only-show-errors
    [ -f figure_data/climb_v2_phase2/$src/encoder/model.safetensors ] || {
      say "$dst: no encoder for $src -> skipping"; ok=0; continue; }
    # stage whatever the destination already has, so a merge has a real destination
    aws s3 sync "$S3/$dst" "$d" --only-show-errors
    tmp=$(mktemp -d)
    say "$dst: HIV end-to-end from $src"
    $PY - "$src" "$tmp" "$TOK" >> "$LOG" 2>&1 <<'PYEOF'
import sys; sys.path.insert(0,".")
from finetune_e2e_v2 import evaluate_finetuned
src,out,tok=sys.argv[1],sys.argv[2],sys.argv[3]
evaluate_finetuned(encoder_path=f"figure_data/climb_v2_phase2/{src}/encoder",
                   tokenizer_path=tok, output_dir=out, seeds=[0],
                   datasets=[("HIV","classification")], cv_folds=5, subsample_seed=0)
PYEOF
    n=$( [ -f "$tmp/test_predictions.csv" ] && awk -F, '$1=="HIV"' "$tmp/test_predictions.csv"|wc -l|tr -d ' ' || echo 0)
    nef=$(grep -c "^HIV,.*,nef1,fold[0-9]," "$tmp/moleculenet_summary.csv" 2>/dev/null || echo 0)
    auc=$(grep -c "^HIV,.*,roc_auc,fold[0-9]," "$tmp/moleculenet_summary.csv" 2>/dev/null || echo 0)
    if [ "$n" -lt 40000 ] || [ "$nef" -lt 5 ] || [ "$auc" -lt 5 ]; then
      say "$dst: REJECTED rows=$n nef1_folds=$nef auc_folds=$auc -- BOTH metrics required, nothing written"
      ok=0; rm -rf "$tmp"; continue
    fi
    mkdir -p "$d/moleculenet_cv"
    $PY scripts/merge_summary_rows.py "$tmp/moleculenet_summary.csv" \
        "$d/moleculenet_cv/moleculenet_summary.csv" HIV >> "$LOG" 2>&1
    cp "$tmp/test_predictions.csv" "$d/moleculenet_cv/test_predictions.csv"
    [ -f "$tmp/suite_summary.json" ] && cp "$tmp/suite_summary.json" "$d/moleculenet_cv/suite_summary.json"
    v1=$(awk -F, '$1=="HIV"&&$8=="MEAN"&&$7=="nef1"{printf "%.4f",$10}' "$d/moleculenet_cv/moleculenet_summary.csv")
    v2=$(awk -F, '$1=="HIV"&&$8=="MEAN"&&$7=="roc_auc"{printf "%.4f",$10}' "$d/moleculenet_cv/moleculenet_summary.csv")
    say "$dst: OK  $n rows  nef1=$v1  roc_auc=$v2"
    aws s3 sync "$d" "$S3/$dst" --only-show-errors
    done_n=$((done_n+1)); rm -rf "$tmp"
  done
done
say "wrote $done_n / 6 cells"
if [ "$ok" = "1" ] && [ "$done_n" = "6" ]; then say "COMPLETE -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE ($done_n/6) -> staying UP"; fi
