#!/usr/bin/env bash
# Clean label-efficiency downsampling re-run for the big-enough tasks (HIV/QM7/Tox21).
# 4 arms: random (frozen), sup:dense (frozen), unsup (frozen), no_pretrain e2e. 5 seeds each for
# clean curves. Test performance only matters downstream. Env: LE_TASKS, LE_TAG. Gated shutdown.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis figure_data/six_panel
: "${LE_TASKS:?set LE_TASKS}"; : "${LE_TAG:?set LE_TAG}"
LOG=analysis/labeleff_clean_${LE_TAG}.log
echo "[le-clean $LE_TAG] start $(date -u +%FT%TZ) tasks=$LE_TASKS" >> "$LOG"

[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer; aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
for a in random_baseline_00 skip_dense_8M unsup_8M; do
  d=figure_data/climb_v2_phase2/$a/encoder
  [ -f "$d/model.safetensors" ] || { mkdir -p "$d"; aws s3 sync "s3://climb-s3-bucket/experiments/climb_v2_phase2/$a/encoder" "$d" --only-show-errors; }
done

export LE_SEEDS=5
FRO=figure_data/six_panel/labeleff_clean_${LE_TAG}_frozen.csv
E2E=figure_data/six_panel/labeleff_clean_${LE_TAG}_e2e.csv

# frozen arms (random, sup:dense, unsup) — cheap
LE_ARMS="random sup unsup" LE_LONG="$FRO" LE_SUMM="${FRO%.csv}_summary.csv" \
  ~/venvs/climb/bin/python scripts/label_eff_fractions.py >> "$LOG" 2>&1
echo "[le-clean $LE_TAG] frozen rc=$? $(date -u +%FT%TZ)" >> "$LOG"

# e2e arm (no_pretrain, random init fine-tuned) — the expensive one
LE_CELLROOT="figure_data/labeleff_clean_e2e_${LE_TAG}" LE_LONG="$E2E" LE_SUMM="${E2E%.csv}_summary.csv" \
  ~/venvs/climb/bin/python scripts/label_eff_fractions_e2e.py >> "$LOG" 2>&1
echo "[le-clean $LE_TAG] e2e rc=$? $(date -u +%FT%TZ)" >> "$LOG"

aws s3 cp "$FRO" s3://climb-s3-bucket/experiments/six_panel/ --only-show-errors
aws s3 cp "$E2E" s3://climb-s3-bucket/experiments/six_panel/ --only-show-errors

# completion from ACHIEVED WORK: every task must have a 100% test row in BOTH csvs
ok=$(~/venvs/climb/bin/python - "$FRO" "$E2E" <<'PY'
import sys, csv, os
tasks = os.environ["LE_TASKS"].split()
def has_full(path):
    if not os.path.exists(path): return set()
    got=set()
    for r in csv.DictReader(open(path)):
        if r.get("split")=="test" and int(float(r["pct"]))==100: got.add(r["task"])
    return got
fro, e2e = has_full(sys.argv[1]), has_full(sys.argv[2])
print(1 if all(t in fro and t in e2e for t in tasks) else 0)
PY
)
echo "[le-clean $LE_TAG] complete=$ok $(date -u +%FT%TZ)" >> "$LOG"
if [ "$ok" = "1" ]; then
  touch "figure_data/LABELEFF_CLEAN_${LE_TAG}_DONE"
  echo "[le-clean $LE_TAG] all done -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[le-clean $LE_TAG] incomplete -> staying UP for inspection $(date -u +%FT%TZ)" >> "$LOG"
fi
