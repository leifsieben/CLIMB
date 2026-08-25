#!/usr/bin/env bash
# Extract all three literature-CLM arms in ONE environment, over the FULL 177,922-molecule universe.
#
# WHY ALL THREE HERE, INCLUDING ChemBERTa WHICH LOADS FINE IN THE PINNED VENV. ChemBERTa's suite
# results were produced through --featurizer hf_encoder under transformers 4.57.3; molformer_c3 and
# selfies_ted can only be featurized under 5.15.1. Leaving it that way would mean the ranking
# compares arms whose vectors came from DIFFERENT transformers versions -- the exact confound that
# moved 27 of 30 fig_F cells by a median 0.38 fold SD. One featurization environment for the whole
# axis; the probe still runs in the pinned venv for every arm.
#
# The 2022 ibm/MoLFormer-XL arm is deliberately ABSENT: Leif is being asked whether it stays
# alongside MoLFormer-c3 or is replaced by it, and three replicate sets across 66 datasets is not
# worth spending before that answer.
set -u
cd /home/ec2-user/CLIMB
NEW=~/venvs/clm_new/bin/python
LOG=analysis/figA_extract_all.log
SM=figure_data/_figA_smiles.json
say () { echo "[ext] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

n=$($NEW -c "import json;print(len(json.load(open('$SM'))['_all_unique']))")
say "universe: $n molecules"

run () {  # $1 out  $2 hf id  $3 extra args...
  local out=$1 hf=$2; shift 2
  # Coverage is COUNTED, not assumed: an npz from the old 113k universe is a valid file that would
  # raise a KeyError halfway through MolNet, after the cheap datasets were already paid for.
  if [ -s "$out" ]; then
    local have
    # SKIP ONLY IF IT IS BOTH COMPLETE AND SELF-DESCRIBING. Coverage alone was the wrong test: a
    # table extracted before the meta blob existed covers every molecule and still cannot say what
    # produced it, so a coverage-only check would keep an unprovenanced table forever.
    have=$($NEW -c "
import numpy as np, json
z=np.load('$out', allow_pickle=False)
have={str(s) for s in z['smiles']}
want=set(json.load(open('$SM'))['_all_unique'])
missing=len(want-have)
print('OK' if (missing==0 and 'meta' in z.files) else f'{missing}:{\"meta\" in z.files}')")
    if [ "$have" = "OK" ]; then say "SKIP $out (complete and carries meta)"; return 0; fi
    say "RE-EXTRACT $out (missing:has_meta = $have)"
    mv "$out" "figure_data/_superseded/$(basename "$out" .npz)_partial.npz" 2>/dev/null
  fi
  say "extract $hf -> $out"
  $NEW scripts/extract_clm_embeddings.py --hf_model "$hf" --smiles "$SM" --out "$out" \
       --batch_size 64 "$@" >> "analysis/figA_ext_$(basename "$out" .npz).log" 2>&1 \
    && say "DONE $out" || say "FAILED $out"
}

mkdir -p figure_data/_superseded
run figure_data/_chemberta_mtr.npz DeepChem/ChemBERTa-77M-MTR
run figure_data/_molformer_c3.npz  DeepChem/MoLFormer-c3-1.1B --tokenizer_from ibm/MoLFormer-XL-both-10pct
run figure_data/_selfies_ted.npz   ibm-research/materials.selfies-ted --selfies --encoder_only
say "EXTRACTION COMPLETE"
for f in figure_data/_chemberta_mtr.npz figure_data/_molformer_c3.npz figure_data/_selfies_ted.npz; do
  [ -s "$f" ] && say "  $f $(stat -c %s "$f") bytes" || say "  MISSING $f"
done
