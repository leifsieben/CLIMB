#!/usr/bin/env bash
# fig_F's missing null: the RANDOM encoder through the concatenation test, in the V2 environment.
#
# THE ENVIRONMENT IS THE WHOLE DIFFICULTY. experiments/_figF_v2/_environment.json records what the
# other V2 tables were computed in, and one field is a trap:
#
#     rdkit_EFFECTIVE: 2022.09.5     (rdkit-pypi shadowing rdkit 2025.9.2)
#
# Every OTHER job today repairs that shadowing, because MTR needs the 217-descriptor set. This job
# must NOT: the rdkit_sameenv tables were computed against the 208-descriptor set, so a repaired box
# produces a `desc` block of a different width and the CLMrand bars would not be comparable to the
# bars beside them. Measured cost of mixing fig_F environments: median 0.38 fold SD against lifts of
# 0.1-0.4. So this script asserts the RECORDED environment and refuses to run in any other.
set -u
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/_figF_v2
S3V1=s3://climb-s3-bucket/experiments/_figF
LOG=analysis/figF_clmrand.log
ENC=figure_data/climb_v2_phase2/random_baseline_00/encoder
mkdir -p analysis/rigor figure_data
say () { echo "[clmrand] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
die () { say "FATAL $*"; aws s3 cp "$LOG" "$S3/logs/figF_clmrand.log" --only-show-errors; exit 2; }

say "start on $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type || echo unknown)"
aws s3 cp "$S3/_environment.json" analysis/_figF_v2_env.json --only-show-errors || die "cannot read the V2 environment record"

# ---- become the recorded environment, then PROVE it ---------------------------------------------
want_xgb=$($PY -c "import json;print(json.load(open('analysis/_figF_v2_env.json'))['packages']['xgboost'])")
want_rdk=$($PY -c "import json;print(json.load(open('analysis/_figF_v2_env.json'))['packages']['rdkit_EFFECTIVE'])")
say "V2 recorded xgboost $want_xgb, EFFECTIVE rdkit $want_rdk"
$PY -m pip install -q "xgboost==$want_xgb" >/dev/null 2>&1
$PY -m pip list 2>/dev/null | grep -q "^rdkit-pypi" || {
  say "restoring the rdkit-pypi shadow the V2 tables were computed under"
  $PY -m pip install -q "rdkit-pypi==2022.9.5" >/dev/null 2>&1
}
$PY - <<PYEOF 2>&1 | tee -a "$LOG" || die "environment does not match the V2 record"
import json, rdkit, xgboost, numpy, sklearn
rec = json.load(open("analysis/_figF_v2_env.json"))["packages"]
import descriptors_v2 as dv
got = {"xgboost": xgboost.__version__, "rdkit_EFFECTIVE": rdkit.__version__,
       "numpy": numpy.__version__, "scikit-learn": sklearn.__version__}
bad = {k: (v, rec[k]) for k, v in got.items() if rec.get(k) and v != rec[k]}
print(f"[clmrand] effective rdkit exposes {dv.n_descriptors()} descriptors")
for k, v in got.items():
    print(f"[clmrand]   {k:16} {v:12} recorded {rec.get(k)}")
assert not bad, f"environment differs from the V2 record: {bad}"
assert dv.n_descriptors() == 208, (
    f"{dv.n_descriptors()} descriptors -- the rdkit_sameenv tables were computed against 208; "
    "a repaired rdkit makes this table incomparable to the bars beside it")
print("[clmrand] environment MATCHES the V2 record")
PYEOF

# ---- stage the same inputs the other V2 cells used ----------------------------------------------
aws s3 cp "$S3V1/_mordred_figF.npz"   figure_data/_mordred_figF.npz   --only-show-errors
aws s3 cp "$S3V1/_figF_smiles.json"   figure_data/_figF_smiles.json   --only-show-errors
aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s "$ENC/model.safetensors" ] || aws s3 cp s3://climb-s3-bucket/experiments/climb_v2_phase2/random_baseline_00/encoder "$ENC" --recursive --only-show-errors
[ -s "$ENC/model.safetensors" ] || die "random encoder absent after staging"
[ -s figure_data/_tokenizer/tokenizer.json ] || die "tokenizer absent"
say "inputs staged"

# ---- the two cells, rdkit family only -----------------------------------------------------------
B="CLMrand,fp+CLMrand,desc+CLMrand,fp+desc+CLMrand"
for sc in molnet panels; do
  if [ "$sc" = molnet ]; then
    out=concat_rdkit_sameenv_CLMrand_V2.csv
    env CONCAT_DESC=rdkit CONCAT_EMB=climb CONCAT_TAG=CLMrand CONCAT_OUT="$out" CONCAT_BLOCKS="$B" \
        CONCAT_ENC="$ENC" CONCAT_MORDRED_NPZ=figure_data/_mordred_figF.npz OMP_NUM_THREADS=8 \
        $PY scripts/concat_redundancy.py >> "$LOG" 2>&1 || die "molnet cell failed"
  else
    out=concat_panels_rdkit_sameenv_CLMrand_V2.csv
    env CONCAT_DESC=rdkit CONCAT_EMB=climb CONCAT_TAG=CLMrand CONCAT_PANEL_OUT="$out" CONCAT_BLOCKS="$B" \
        CONCAT_ENC="$ENC" CONCAT_MORDRED_NPZ=figure_data/_mordred_figF.npz \
        CONCAT_PANELS="MoleculeACE Ames" OMP_NUM_THREADS=8 \
        $PY scripts/concat_redundancy_panels.py >> "$LOG" 2>&1 || die "panels cell failed"
  fi
  f="analysis/rigor/$out"; ff="analysis/rigor/${out%.csv}_folds.csv"
  # PAIRED_READY needs the per-fold file, so a table without one is not a finished cell.
  [ -s "$f" ] && [ -s "$ff" ] || die "$out produced without its _folds.csv"
  aws s3 cp "$f" "$S3/$out" --only-show-errors
  aws s3 cp "$ff" "$S3/${out%.csv}_folds.csv" --only-show-errors
  say "DONE $out ($(wc -l < "$f") rows, $(wc -l < "$ff") fold rows)"
done

aws s3 cp analysis/_figF_v2_env.json "$S3/logs/_env_asserted_clmrand.json" --only-show-errors
aws s3 cp "$LOG" "$S3/logs/figF_clmrand.log" --only-show-errors
# The box may be wanted for the 100M rung afterwards, so shutting down is opt-in rather than the
# default: a job that takes a machine with it is wrong when the machine is the scarce thing.
if [ "${CLMRAND_SHUTDOWN:-0}" = 1 ]; then
  say "ALL FOUR FILES ON S3 -- shutting down"
  sudo shutdown -h now
else
  say "ALL FOUR FILES ON S3 -- leaving the box up (CLMRAND_SHUTDOWN=1 to shut down)"
fi
