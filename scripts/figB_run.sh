#!/usr/bin/env bash
# Run ONE fig_B rung end to end on a fresh box, then self-terminate on verified completion.
#
# Usage: figB_run.sh <run_id>
#
# The box is launched with instance-initiated-shutdown-behavior=terminate, so `shutdown -h` is all
# this needs and the instance role stays S3-only.
set -u
# PIPEFAIL IS LOAD-BEARING HERE. Every gate below is written `python ... | tee -a "$LOG" || abort`,
# and without this the pipeline's exit status is TEE's -- which always succeeds. Three rungs
# launched on 2026-08-26 with their descriptor directory 20 shards complete: the completeness gate
# ran, printed "BAD descriptors_shard_00005.npy: ABSENT" seventy times, and the run proceeded to
# training anyway. The gate was not missing and it was not wrong; its verdict was discarded one
# character before it was read.
set -o pipefail
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
# The descriptor directory is now REQUIRED and must belong to this corpus -- absent would put the
# rung on the live pathway, which is how the broken skip_dense_48M behaved. Check the pairing, then
# check the directory is COMPLETE, then check the rows are the right molecules. A correct path over
# an incomplete or misaligned directory is the failure this is actually guarding against.
$PY -c "
import json, sys
sys.path.insert(0, 'scripts')
from precompute_descriptors import CORPORA
m = json.load(open('analysis/manifest_${RUN}.json'))
pc = m['runs'][0]['pretrain_config']
raw = pc['unsupervised_raw_smiles_paths']
corpus = [c for c in CORPORA if any(c in r for r in raw)]
assert len(corpus) == 1, f'cannot identify corpus from {raw}'
want = CORPORA[corpus[0]][1].rstrip('/')
got = str(pc.get('descriptor_precompute_dir', '')).rstrip('/')
assert got == want, f'descriptor_precompute_dir {got!r} does not belong to corpus {corpus[0]} ({want})'
print(f'[figB] descriptor dir matches corpus {corpus[0]}')" 2>&1 | tee -a "$LOG" || abort "descriptor directory does not match the corpus"

# ---- repair rdkit BEFORE any gate that uses it --------------------------------------------------
# Every box off the April AMI carries rdkit-pypi 2022.9.5 shadowing rdkit 2025.9.2, exposing 208 of
# 217 descriptors. This block used to sit BELOW the alignment gate -- under a comment saying "repair
# the environment before checking it" -- so the alignment gate ran on the broken rdkit and aborted
# with "descriptor rows are not these molecules". The rows were fine; the environment was not, and
# the message sent the reader to the data.
#
# The condition tested is the EFFECTIVE descriptor count, not `pip list`: a fresh box listed
# rdkit-pypi and the pip-list guard still did not fire. Uninstalling rdkit-pypi alone breaks both
# (they share paths), so force-reinstall the one we want, and let pip's errors reach the log.
aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json --only-show-errors \
  || abort "cannot fetch the canonical descriptor stats"
n_desc () { $PY -c "import descriptors_v2 as d; print(len(d.descriptor_names()))" 2>/dev/null || echo 0; }
WANT_DESC=$($PY -c "import json; print(len(json.load(open('configs/descriptor_stats.json'))['names']))")
if [ "$(n_desc)" != "$WANT_DESC" ]; then
  say "rdkit exposes $(n_desc) descriptors, need $WANT_DESC -- repairing"
  $PY -m pip uninstall -y rdkit-pypi 2>&1 | tail -2 | tee -a "$LOG"
  $PY -m pip install -q --force-reinstall --no-deps "rdkit==2025.9.2" 2>&1 | tail -2 | tee -a "$LOG"
  say "after repair: $(n_desc) descriptors"
fi
[ "$(n_desc)" = "$WANT_DESC" ] || abort "rdkit exposes $(n_desc) descriptors, need $WANT_DESC -- environment, not data"

# Complete: every shard the corpus has must have a descriptor file of the size its row count implies.
# Check the shards THIS RUNG OPENS, not the whole corpus: requiring shards it will never read would
# block a launch on work that does not exist yet, and the 50M rung's 72 land well before the 100M
# rung's 120. The budget comes from the manifest, so the gate and the run cannot disagree about it.
BUDGET=$($PY -c "
import json; print(json.load(open('analysis/manifest_${RUN}.json'))['runs'][0]['selection']['total_forward_passes'])")
$PY scripts/verify_descriptor_dir.py --corpus pubchem_124m_full --budget "$BUDGET" 2>&1 | tee -a "$LOG"   || abort "descriptor directory incomplete for a ${BUDGET} forward-pass run -- refusing to train against partial targets"
# Right molecules: a path and a size cannot see a row shift or a corpus swap.
$PY -c "
import sys; sys.path.insert(0,'scripts')
from c124_priority_order import needed
print(','.join(needed($BUDGET)))" > analysis/_probe_shards.txt
$PY scripts/verify_descriptor_alignment.py --corpus pubchem_124m_full --shard_list "$(cat analysis/_probe_shards.txt)" --n_probes 12 2>&1 | tee -a "$LOG"   || abort "descriptor rows are not these molecules"

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
# ---- the GPU must be OURS before we start -------------------------------------------------------
# skip_dense_100M_c124 OOMed twelve seconds into training because a second trainer was already
# holding 13.45 of the 22.06 GiB on that box, and the run then sat dead for 5.5 hours. The model is
# 41.4M params and trains fine on this instance type -- the arithmetic was a squatter, not the
# model. Discovering that from an OOM traceback costs a launch; discovering it here costs nothing.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)
say "GPU: ${FREE_MIB:-unknown} MiB free, $OTHER process(es) already on the device"
[ -n "$FREE_MIB" ] || abort "cannot read GPU memory -- refusing to train blind"
[ "$OTHER" -eq 0 ] || abort "$OTHER process(es) already holding this GPU: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';') -- refusing to share"
[ "$FREE_MIB" -ge 18000 ] || abort "only ${FREE_MIB} MiB free on the GPU, need >=18000 -- something is holding it"

# ---- warm-start base, staged and PROVEN present -------------------------------------------------
# A u2s rung names its base as a LOCAL path (experiments/.../unsup_50M/encoder). experiments/ is
# gitignored, so on a fresh box that path does not exist -- and from_pretrained then treats it as a
# HUGGING FACE REPO ID and dies on "Repo id must be in the form 'repo_name'". Both u2s boxes sat
# idle 1.5h on exactly that. phase2_worker.sh has staged these for months; this runner never did.
INIT_ENCODERS=$($PY -c "
import json
m = json.load(open('analysis/manifest_${RUN}.json'))
runs = m['runs'] if isinstance(m, dict) and 'runs' in m else m
seen = []
for r in runs:
    for sel in (r.get('selection') or {}, (r.get('pretrain_config') or {}).get('selection') or {}):
        p = sel.get('init_encoder_path')
        if p and p not in seen:
            seen.append(p)
print(' '.join(seen))
") || abort "cannot read init_encoder_path out of the manifest"
for e in $INIT_ENCODERS; do
  say "warm-start base: $e"
  mkdir -p "$e"
  aws s3 sync "s3://climb-s3-bucket/$e" "$e" --only-show-errors || abort "sync failed for warm-start base $e"
  # Test for WEIGHTS, not for the directory: mkdir -p above guarantees the directory exists, so a
  # directory test would pass on every box and prove nothing.
  [ -f "$e/model.safetensors" ] || [ -f "$e/pytorch_model.bin" ]     || abort "warm-start base $e has no weights after sync -- the run would die on startup"
done
[ -n "$INIT_ENCODERS" ] && say "warm-start bases staged and carry weights"

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
