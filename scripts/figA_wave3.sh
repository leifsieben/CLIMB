#!/usr/bin/env bash
# wave 3: the two NEW datasets -- Wong and FartDB -- across all THIRTEEN arms.
#
# These are the only datasets nothing else in the project covers, so they are first. MolNet and CBS
# for the literature arms follow in wave 4.
#
# THE REPLICATE AXIS DIFFERS BY ARM AND THE CONVENTIONS ARE NOT INTERCHANGEABLE. Taken from
# figures/arms.py, which is the source the figure itself reads:
#   ecfp / ecfp_desc      DISJOINT HEAD-SEED TRIPLES on one featurizer (no encoder to vary)
#   literature CLMs       same -- one released checkpoint each
#   CLIMB encoder arms    the ENCODER is the axis; head seeds PINNED to the base triple.
#                         <stem>_s1 means encoder _s1, NOT a new head seed.
#   s2u_dense             _s0/_s1/_s2 -- no bare stem
#   random_encoder        random_baseline_00/01/02 -- two digits
# A script assuming <stem>{,_s1,_s2} silently produces NOTHING for the last two.
set -u
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_wave3.log
say () { echo "[w3] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

LOCK=analysis/.figA_wave3.lock; exec 9>"$LOCK"
flock -n 9 || { say "another wave3 holds the lock -- refusing"; exit 0; }

# arm|replicate-label|featurizer|source|seeds
spec_lines () {
cat <<'EOF'
ecfp|ecfp4_anchor|ecfp4|-|42 117 709
ecfp|ecfp4_anchor_s1|ecfp4|-|43 118 710
ecfp|ecfp4_anchor_s2|ecfp4|-|44 119 711
ecfp_desc|fp_desc_anchor|fp_desc|-|42 117 709
ecfp_desc|fp_desc_anchor_s1|fp_desc|-|43 118 710
ecfp_desc|fp_desc_anchor_s2|fp_desc|-|44 119 711
chemberta_mtr|chemberta_mtr|npz|figure_data/_chemberta_mtr.npz|42 117 709
chemberta_mtr|chemberta_mtr_s1|npz|figure_data/_chemberta_mtr.npz|43 118 710
chemberta_mtr|chemberta_mtr_s2|npz|figure_data/_chemberta_mtr.npz|44 119 711
molformer_c3|molformer_c3|npz|figure_data/_molformer_c3.npz|42 117 709
molformer_c3|molformer_c3_s1|npz|figure_data/_molformer_c3.npz|43 118 710
molformer_c3|molformer_c3_s2|npz|figure_data/_molformer_c3.npz|44 119 711
selfies_ted|selfies_ted|npz|figure_data/_selfies_ted.npz|42 117 709
selfies_ted|selfies_ted_s1|npz|figure_data/_selfies_ted.npz|43 118 710
selfies_ted|selfies_ted_s2|npz|figure_data/_selfies_ted.npz|44 119 711
sup_dense|skip_dense_8M|encoder|skip_dense_8M|42 117 709
sup_dense|skip_dense_8M_s1|encoder|skip_dense_8M_s1|42 117 709
sup_dense|skip_dense_8M_s2|encoder|skip_dense_8M_s2|42 117 709
sup_sparse|skip_sparse_all_8M|encoder|skip_sparse_all_8M|42 117 709
sup_sparse|skip_sparse_all_8M_s1|encoder|skip_sparse_all_8M_s1|42 117 709
sup_sparse|skip_sparse_all_8M_s2|encoder|skip_sparse_all_8M_s2|42 117 709
sup_mixed|skip_mixed_8M|encoder|skip_mixed_8M|42 117 709
sup_mixed|skip_mixed_8M_s1|encoder|skip_mixed_8M_s1|42 117 709
sup_mixed|skip_mixed_8M_s2|encoder|skip_mixed_8M_s2|42 117 709
unsup|unsup_8M|encoder|unsup_8M|42 117 709
unsup|unsup_8M_s1|encoder|unsup_8M_s1|42 117 709
unsup|unsup_8M_s2|encoder|unsup_8M_s2|42 117 709
u2s_dense|u2s_dense_from8M|encoder|u2s_dense_from8M|42 117 709
u2s_dense|u2s_dense_from8M_s1|encoder|u2s_dense_from8M_s1|42 117 709
u2s_dense|u2s_dense_from8M_s2|encoder|u2s_dense_from8M_s2|42 117 709
u2s_sparse|u2s_sparse_all_from8M|encoder|u2s_sparse_all_from8M|42 117 709
u2s_sparse|u2s_sparse_all_from8M_s1|encoder|u2s_sparse_all_from8M_s1|42 117 709
u2s_sparse|u2s_sparse_all_from8M_s2|encoder|u2s_sparse_all_from8M_s2|42 117 709
s2u_dense|s2u_dense_from8M_s0|encoder|s2u_dense_from8M_s0|42 117 709
s2u_dense|s2u_dense_from8M_s1|encoder|s2u_dense_from8M_s1|42 117 709
s2u_dense|s2u_dense_from8M_s2|encoder|s2u_dense_from8M_s2|42 117 709
random_encoder|random_baseline_00|encoder|random_baseline_00|42 117 709
random_encoder|random_baseline_01|encoder|random_baseline_01|42 117 709
random_encoder|random_baseline_02|encoder|random_baseline_02|42 117 709
EOF
}

stage_encoder () {  # $1 stem -> echoes the local path, or empty if it cannot be staged
  local stem=$1 d=figure_data/climb_v2_phase2/$1/encoder
  if [ ! -s "$d/model.safetensors" ]; then
    for pre in experiments/climb_v2_phase2 experiments/climb_v2_phase1 experiments; do
      aws s3 cp "s3://climb-s3-bucket/$pre/$stem/encoder" "$d" --recursive --only-show-errors 2>/dev/null
      [ -s "$d/model.safetensors" ] && break
    done
  fi
  [ -s "$d/model.safetensors" ] && echo "$d" || echo ""
}

n_ok=0; n_bad=0
while IFS='|' read -r arm label feat src seeds; do
  [ -n "${arm:-}" ] || continue
  encarg=(); npzarg=""
  if [ "$feat" = encoder ]; then
    p=$(stage_encoder "$src")
    # A MISSING ENCODER MUST BE LOUD. Silently skipping is how an arm ends up absent from a
    # ranking that still looks complete.
    [ -n "$p" ] || { say "MISSING ENCODER for $label ($src) -- arm cell NOT produced"; n_bad=$((n_bad+1)); continue; }
    encarg=(--encoder "$p" --tokenizer figure_data/_tokenizer)
  elif [ "$feat" = npz ]; then
    [ -s "$src" ] || { say "MISSING NPZ $src for $label"; n_bad=$((n_bad+1)); continue; }
    encarg=(--encoder "$src")
  fi
  for ds in wong fartdb; do
    if [ "$ds" = wong ]; then
      out=figure_data/wong_saureus/$label; script=scripts/wong_run.py; rem="$S3/wong/$label"
    else
      out=figure_data/fartdb/$label;       script=scripts/fartdb_multiclass.py; rem="$S3/fartdb/$label"
    fi
    [ -s "$out/verified.json" ] && { say "SKIP $ds/$label"; continue; }
    say "RUN $ds/$label (feat=$feat seeds=$seeds)"
    $PY "$script" --model "$label" --featurizer "$feat" "${encarg[@]}" --head mlp --seeds $seeds \
      >> "analysis/figA_w3_${ds}_${label}.log" 2>&1
    if [ -s "$out/verified.json" ]; then
      aws s3 cp "$out" "$rem" --recursive --only-show-errors
      say "DONE $ds/$label"; n_ok=$((n_ok+1))
    else
      say "FAILED $ds/$label (see analysis/figA_w3_${ds}_${label}.log)"; n_bad=$((n_bad+1))
    fi
  done
done < <(spec_lines)
say "WAVE 3 COMPLETE -- $n_ok cells done, $n_bad failed/missing"
