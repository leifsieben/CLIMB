#!/usr/bin/env bash
# GAP 1: SI fig a's only empty cell -- end-to-end HIV at 100% labels for the best-two CLIMB arms.
#
# Scope (agreed with the figures session): the SAME label-efficiency end-to-end protocol that
# produced six_panel_e2e.csv, extended to HIV at pct=100 only. 2 encoders x 3 fine-tune seeds = 6
# runs on 41,127 molecules. NOT the full fraction ladder -- SI fig a reads only the top rung and
# SI fig e's HIV line is already complete.
#
# THE GUARD THAT MATTERS MOST HERE. six_panel_e2e.py rebuilds its long CSV from `rows`, which
# starts as whatever analysis/rigor/six_panel_e2e.csv holds -- EMPTY on a fresh box. It then
# `aws s3 cp`s that file over s3://.../six_panel/six_panel_e2e.csv. Run unstaged, this job would
# publish a HIV-only CSV over the existing BACE/BBBP/Tox21/QM7 rows: the s2u clobber again, one
# layer up. So the CSV is staged first and the job REFUSES to start unless all four existing tasks
# are present in it.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
LONG=analysis/rigor/six_panel_e2e.csv
S3CSV=s3://climb-s3-bucket/experiments/six_panel/six_panel_e2e.csv
LOG=analysis/gap1_hiv_e2e.log
mkdir -p analysis/rigor figure_data/_tokenizer
say(){ echo "[gap1 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG" || {
  say "FATAL no GPU -> staying UP"; exit 1; }
$PY -c "import torch;assert torch.cuda.is_available()" || { say "FATAL torch/cuda -> staying UP"; exit 1; }

aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors

# ---- stage the long CSV and ASSERT it, before anything can overwrite it -----------------------
aws s3 cp "$S3CSV" "$LONG" --only-show-errors || { say "FATAL cannot stage $S3CSV -> staying UP"; exit 1; }
missing=""
for t in BACE BBBP Tox21 QM7; do
  grep -q ",$t," "$LONG" || missing="$missing $t"
done
if [ -n "$missing" ]; then
  say "FATAL staged CSV is missing:$missing -- refusing to run, it would be published over the real one"
  exit 1
fi
say "staged $LONG with $(wc -l < "$LONG") rows, all four existing tasks present"

# ---- assert HIV actually loads before spending an hour of GPU --------------------------------
$PY -c "
from eval_v2 import _load_moleculenet_full
smiles, y = _load_moleculenet_full('HIV')
print('HIV molecules:', len(smiles))
assert len(smiles) > 40000, f'HIV loaded only {len(smiles)} molecules'
" 2>&1 | tee -a "$LOG" || { say "FATAL HIV does not load -> staying UP"; exit 1; }

W3_TASKS=HIV W3_FRACTIONS=1.00 $PY scripts/six_panel_e2e.py >> "$LOG" 2>&1
rc=$?
say "driver rc=$rc"

# ---- completion is achieved work: HIV rows for BOTH arms, and the old tasks still present -----
ok=1
for a in unsup_only "sup_only:dense"; do
  n=$(awk -F, -v A="$a" '$1==A && $2=="HIV" && $8=="nef1"' "$LONG" | wc -l | tr -d ' ')
  say "$a HIV nef1 rows: $n"
  [ "$n" -ge 3 ] || ok=0
done
for t in BACE BBBP Tox21 QM7; do grep -q ",$t," "$LONG" || { say "LOST $t from the CSV"; ok=0; }; done

if [ "$ok" = "1" ]; then
  aws s3 cp "$LONG" "$S3CSV" --only-show-errors
  aws s3 sync figure_data/six_panel_e2e s3://climb-s3-bucket/experiments/six_panel_e2e --only-show-errors
  say "COMPLETE -> shutdown"
  sudo shutdown -h now
else
  say "INCOMPLETE -> staying UP for inspection"
fi
