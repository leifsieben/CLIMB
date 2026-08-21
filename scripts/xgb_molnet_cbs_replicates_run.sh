#!/usr/bin/env bash
# The MolNet + CBS half of the two __xgb arms' pretraining-seed replicates.
#
# The suite half (MoleculeACE + Polaris) is a different job on a different box. This one fills:
#
#   figure_data/climb_v2_phase2/<arm>__xgb_s{1,2}/moleculenet_cv   6 datasets, 220 rows
#   figure_data/cbs_benchmark/<arm>__xgb_s{1,2}/moleculenet_cv     cbs,        44 rows
#
# for arm in {unsup_8M, skip_dense_8M}. Four dirs, eight cells.
#
# MIND THE DOUBLE UNDERSCORE. The target is <arm>__xgb_s1, not <arm>_s1. <arm>_s1 is the FROZEN
# MLP arm and it already exists, complete, in both trees -- writing there would destroy finished
# work. eval_v2 opens moleculenet_summary.csv with "w".
#
# HEAD SEEDS ARE PINNED TO THE BASE TRIPLE 0/1/2; ONLY THE ENCODER MOVES. The nearest template,
# anchor_seed_replicates.py, uses DISJOINT triples ({0,1,2},{3,4,5},{6,7,8}) and is right to: its
# anchors have one representation, so the head is the only axis available. Our arms have three
# pretrainings, and that is the axis fig_A1 asks about. Copying the disjoint recipe here would
# move the encoder and the head seeds together and the spread would measure neither cleanly.
#
# EVERY OTHER KNOB IS LIFTED VERBATIM from scripts/head_comparison_run.sh, which produced the
# bases -- datasets, pool, standardize, cv_folds, cv_scheme, and the CBS task_csv/type. A
# replicate that differs from its base in any knob is not a replicate.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis figure_data

ARM=${ARM:?ARM must be set (unsup_8M | skip_dense_8M)}
SUFFIXES=${SUFFIXES:-"_s1 _s2"}
LOG="analysis/xgb_molnet_cbs_${ARM}.log"
S3=s3://climb-s3-bucket/experiments
PY=~/venvs/climb/bin/python
TASKS="ESOL QM7 BBBP BACE Tox21 HIV"

IMDS_TOKEN=$(curl -fs --max-time 5 -X PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
IID=$(curl -fs --max-time 5 -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
say () { echo "[xgbmn] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
[ -n "$IID" ] || { say "FATAL no instance id"; exit 3; }
( while true; do aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors 2>/dev/null; sleep 60; done ) &

say "start on $IID -- arm $ARM, suffixes $SUFFIXES, head seeds 0 1 2"

# ---------------------------------------------------------------- preflight
[ -x "$PY" ] || { say "FATAL no python at $PY"; exit 2; }
$PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "FATAL preflight imports"; exit 2; }
import torch, transformers, xgboost, sklearn, rdkit, numpy, deepchem
print(f"[preflight] torch={torch.__version__} xgboost={xgboost.__version__} "
      f"deepchem={deepchem.__version__} rdkit={rdkit.__version__} numpy={numpy.__version__}")
PYEOF
aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.probe" --only-show-errors || { say "FATAL cannot write S3"; exit 2; }
say "preflight OK"

# ---------------------------------------------------------------- inputs, all asserted
aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s figure_data/_tokenizer/tokenizer.json ] || { say "FATAL tokenizer.json absent"; exit 2; }
$PY -c "
from transformers import PreTrainedTokenizerFast
t = PreTrainedTokenizerFast.from_pretrained('figure_data/_tokenizer')
assert t.vocab_size > 0
print(f'[preflight] tokenizer OK, vocab={t.vocab_size}')" >> "$LOG" 2>&1 \
  || { say "FATAL tokenizer will not load"; exit 2; }

mkdir -p data
aws s3 cp s3://climb-s3-bucket/datasets/cbs.csv data/cbs.csv --only-show-errors
[ -s data/cbs.csv ] || { say "FATAL data/cbs.csv absent -- CBS cells would die 5 folds in"; exit 2; }

for sfx in $SUFFIXES; do
  aws s3 cp "$S3/climb_v2_phase2/${ARM}${sfx}/encoder" "figure_data/climb_v2_phase2/${ARM}${sfx}/encoder" \
      --recursive --only-show-errors
  [ -s "figure_data/climb_v2_phase2/${ARM}${sfx}/encoder/model.safetensors" ] \
      || { say "FATAL encoder ${ARM}${sfx} absent"; exit 2; }
done

# The base summaries are the PARSE FINGERPRINT for the tripwire below.
aws s3 cp "$S3/climb_v2_phase2/${ARM}__xgb/moleculenet_cv/moleculenet_summary.csv" \
    "/tmp/base_mn_${ARM}.csv" --only-show-errors
aws s3 cp "$S3/cbs_benchmark/${ARM}__xgb/moleculenet_cv/moleculenet_summary.csv" \
    "/tmp/base_cbs_${ARM}.csv" --only-show-errors
say "inputs staged"

# ---------------------------------------------------------------- the eight cells
for sfx in $SUFFIXES; do
  ENC="figure_data/climb_v2_phase2/${ARM}${sfx}/encoder"
  TAG="${ARM}__xgb${sfx}"

  out="figure_data/climb_v2_phase2/${TAG}/moleculenet_cv"
  if [ -s "$out/moleculenet_summary.csv" ]; then say "SKIP molnet $TAG"; else
    say "RUN  molnet $TAG (encoder ${ARM}${sfx})"
    $PY eval_v2.py --output_dir "$out" --datasets $TASKS --head xgb --head_seeds 0 1 2 \
        --pool mean --standardize zscore --cv_folds 5 --cv_scheme scaffold \
        --featurizer encoder --encoder "$ENC" --tokenizer figure_data/_tokenizer >> "$LOG" 2>&1
    say "molnet $TAG rc=$?"
  fi

  cout="figure_data/cbs_benchmark/${TAG}/moleculenet_cv"
  if [ -s "$cout/moleculenet_summary.csv" ]; then say "SKIP cbs $TAG"; else
    say "RUN  cbs $TAG"
    $PY eval_v2.py --output_dir "$cout" --head xgb --head_seeds 0 1 2 \
        --pool mean --standardize zscore --cv_folds 5 --cv_scheme provided \
        --task_csv data/cbs.csv --task_name cbs --task_type classification \
        --featurizer encoder --encoder "$ENC" --tokenizer figure_data/_tokenizer >> "$LOG" 2>&1
    say "cbs $TAG rc=$?"
  fi
done

# ---------------------------------------------------------------- verify: rows AND parse identity
# THE ROW COUNT IS NOT ENOUGH. A box whose deepchem parses Tox21 differently produces a complete,
# well-formed 220-row cell built on a DIFFERENT molecule set -- that is the drift that cost the
# fig_C2/fig_D Tox21 column, and it is invisible to any completeness check. n_train per
# (dataset, head_seed) is the parse fingerprint, and the base is the reference. If it does not
# match exactly, this environment is not the base's environment and the seed spread would be
# partly an environment measurement, so the cell is quarantined rather than published.
say "verifying row counts and parse identity against the base"
MISSING=0
for sfx in $SUFFIXES; do
  TAG="${ARM}__xgb${sfx}"
  for spec in "climb_v2_phase2:220:/tmp/base_mn_${ARM}.csv" "cbs_benchmark:44:/tmp/base_cbs_${ARM}.csv"; do
    tree=${spec%%:*}; rest=${spec#*:}; want=${rest%%:*}; ref=${rest#*:}
    f="figure_data/${tree}/${TAG}/moleculenet_cv/moleculenet_summary.csv"
    verdict=$($PY scripts/verify_replicate_parse.py "$f" "$ref" "$want" 2>&1)
    case "$verdict" in
      OK*)  say "VERIFIED $tree $TAG -- $verdict" ;;
      *)    say "REJECT   $tree $TAG -- $verdict"; MISSING=$((MISSING+1)); continue ;;
    esac
    # HELD, NOT PUBLISHED. The peer's stage-0 env test has not returned a verdict on whether the
    # BASES themselves reproduce in a fresh venv. If they do not, these replicates are measured
    # against a base that is about to be rebuilt. Backed up so nothing is lost, under a prefix
    # nothing reads, and promoted by hand once that verdict lands.
    aws s3 cp --recursive "figure_data/${tree}/${TAG}/moleculenet_cv" \
        "$S3/_pending_verdict/${tree}/${TAG}/moleculenet_cv" --only-show-errors
    say "HELD $tree $TAG -> _pending_verdict/ (awaiting stage-0 base verdict)"
  done
done

aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors
if [ "$MISSING" -eq 0 ]; then
  say "COMPLETE all cells verified and held -- staying UP until the cells are promoted"
else
  say "NOT COMPLETE ($MISSING cell(s) rejected) -- staying UP for inspection"
fi
