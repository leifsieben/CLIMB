#!/usr/bin/env bash
# Run ONE fig_B rung end to end on a fresh box, then self-terminate on verified completion.
#
# Usage: figB_run.sh <run_id>
#
# The box is launched with instance-initiated-shutdown-behavior=terminate, so `shutdown -h` is all
# this needs and the instance role stays S3-only.
set -u
RUN=$1
# Resolve our own path BEFORE the cd -- the re-exec below runs after it, and "$0" was relative to
# whatever directory the launcher used.
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
LOG=analysis/figB_${RUN}.log
mkdir -p analysis
say () { echo "[figB:$RUN] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$S3/$RUN/figB_run.log" --only-show-errors; exit 1; }

say "start on $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type || echo unknown)"
# A script that git-resets its OWN source is reading a file that changes underneath it: bash reads
# lazily by byte offset, so an update that inserts lines makes it resume at the wrong place and
# SKIP whatever moved past the offset. That is how the MTR gate silently did not run on the launch
# that added it. Update, then re-exec once from the new file with a fresh offset.
if [ "${FIGB_REEXEC:-0}" != "1" ]; then
  git fetch -q origin v2-redux && git reset -q --hard origin/v2-redux
  say "code at $(git rev-parse --short HEAD) -- re-exec from updated source"
  FIGB_REEXEC=1 exec bash "$SELF" "$@"
fi
say "code at $(git rev-parse --short HEAD)"

# ---- fetch the TEMPLATE manifests: experiments/ is gitignored, so a fresh box has none ---------
# The first bridge launch died here in one second -- "no manifest entry for template skip_dense_8M"
# -- because build_figB_manifest.py clones from a real entry and those live only on the laptop.
# Publishing them to S3 keeps the CLONE and its gate on the box, which is where they belong: the
# alternative, generating the manifest locally and shipping the result, would move the check away
# from the machine that actually runs it.
mkdir -p experiments/climb_v2_phase2
# The cp errors were swallowed and the guard only asked "is the file non-empty?" -- and the worker
# bundle ships a STALE July experiments/ tree, so a failed fetch left an old manifest in place that
# passed the guard and died 1s later in the builder. Check the fetch, then TEST the condition that
# matters: does this manifest actually carry the template this rung clones from?
for f in manifest.json manifest_supplement.json; do
  aws s3 cp "s3://climb-s3-bucket/experiments/climb_v2_phase2/manifests/templates/$f"             "experiments/climb_v2_phase2/$f" --only-show-errors     || abort "template fetch failed for $f -- refusing to run against whatever was already on disk"
done
# Ask the builder which template THIS rung clones -- 50M and 100M both clone skip_dense_8M, so
# deriving the name by stripping a suffix would abort on a manifest that is perfectly fine.
TPL=$($PY -c "
import json,sys
sys.path.insert(0,'scripts')
from build_figB_manifest import SPEC
t=SPEC.get('$RUN')
if t is None: sys.exit(1)
ids=set()
for f in ('manifest.json','manifest_supplement.json'):
    try: ids |= {r['run_id'] for r in json.load(open('experiments/climb_v2_phase2/'+f))['runs']}
    except Exception: pass
if t[0] not in ids: sys.exit(2)
print(t[0])") || abort "template for '$RUN' missing from the fetched manifests -- stale or wrong file"
say "template manifests fetched -- '$TPL' present"

# ---- manifest, gated ------------------------------------------------------------------------
$PY scripts/build_figB_manifest.py --run "$RUN" --out "analysis/manifest_${RUN}.json" 2>&1 | tee -a "$LOG"
[ -s "analysis/manifest_${RUN}.json" ] || abort "manifest builder refused or failed"

# ---- assert the corpus is reachable BEFORE spending hours -------------------------------------
# A missing or misnamed corpus prefix returns empty and exits 0 from `aws s3 ls`, so COUNT.
n=$(aws s3 ls s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full/ | grep -c "\.parquet$")
[ "$n" -ge 124 ] || abort "expected >=124 corpus shards, counted $n"
say "corpus OK -- $n shards"
# And the one field whose presence would silently corrupt MTR on this corpus.
$PY -c "
import json
m=json.load(open('analysis/manifest_${RUN}.json'))
pc=m['runs'][0]['pretrain_config']
assert 'descriptor_precompute_dir' not in pc, 'descriptor_precompute_dir is set -- shard names collide across corpora'
print('[figB] descriptor_precompute_dir correctly absent')" 2>&1 | tee -a "$LOG" || abort "descriptor precompute check failed"

# ---- the MTR target space must be the one the TEMPLATE trained against -------------------------
# This box came off an April AMI carrying rdkit-pypi 2022.9.5, which SHADOWED the rdkit 2025.9.2 the
# July fleet used and exposes only 208 of the 217 descriptors. It surfaced as a broadcast error --
# but only because the counts differed. Had a version merely REORDERED the list, every descriptor
# would have been z-scored by another descriptor's mean and nothing would have raised. So check the
# names in order, and check them against what the template run RECORDED, not against a constant.
aws s3 cp "$S3/$TPL/metadata.json" analysis/tpl_meta.json --only-show-errors   || abort "cannot read template metadata -- no reference MTR width to check against"
$PY -c "
import json, descriptors_v2 as dv
want = json.load(open('analysis/tpl_meta.json'))['mtr_n_desc']
stats = json.load(open('configs/descriptor_stats.json'))
cur = dv.descriptor_names()
assert cur == stats['names'], f'descriptor names differ from the fitted stats ({len(cur)} vs {len(stats[\"names\"])}) -- this rdkit is not the fleet rdkit'
assert len(cur) == want, f'MTR width {len(cur)} != template {want}'
print(f'[figB] MTR target space matches template $TPL: {want} descriptors, names in order')" 2>&1 | tee -a "$LOG" || abort "MTR descriptor space does not match the template -- refusing to train an incomparable rung"

# ---- train ------------------------------------------------------------------------------------
# 11 ms/molecule single-core x 6 workers should be ~545 seq/s; the first attempt managed 90 with a
# load average of 47 on 16 vCPUs, which is thread oversubscription, not descriptor cost -- both
# corpora measure the same 11 ms. Pin the math libraries to one thread per worker. This is an env
# setting, not a config change: the manifest diff against the template stays exactly the eight
# fields the gate allows, and dataloader_num_workers stays at the template's 6 so the shard
# interleaving is unchanged.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say "launching wave"
$PY scripts/launch_v2_wave.py --manifest "analysis/manifest_${RUN}.json" --worker_name "figB_${RUN}" \
  >> "$LOG" 2>&1
rc=$?
say "wave exited rc=$rc"

# ---- completion is ACHIEVED forward passes, never a file ---------------------------------------
d=experiments/climb_v2_phase2/$RUN
want=$($PY -c "
import json; print(json.load(open('analysis/manifest_${RUN}.json'))['runs'][0]['selection']['total_forward_passes'])")
got=$($PY -c "
import json
try:
    print(json.loads(open('$d/metrics.jsonl').read().strip().split(chr(10))[-1])['forward_passes_seen'])
except Exception: print(0)" 2>/dev/null)
say "forward passes $got / $want"
$PY -c "import sys; sys.exit(0 if $got >= 0.98*$want else 1)" || abort "only $got of $want forward passes -- truncated"

# ---- upload BEFORE the gates, so a failing gate still leaves the work durable -------------------
aws s3 cp "$d" "$S3/$RUN" --recursive --only-show-errors || abort "upload failed"
aws s3 cp "$LOG" "$S3/$RUN/figB_run.log" --only-show-errors
say "uploaded"

# ---- reconcile against S3 rather than trusting the local state ----------------------------------
for f in encoder/model.safetensors metrics.jsonl config.yaml; do
  aws s3 ls "$S3/$RUN/$f" >/dev/null 2>&1 || abort "$f absent on S3 after upload"
done
say "ALL GATES PASSED -- terminating"
aws s3 cp "$LOG" "$S3/$RUN/figB_run.log" --only-show-errors
sudo shutdown -h now
