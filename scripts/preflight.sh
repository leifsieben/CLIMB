#!/usr/bin/env bash
# Preflight gate — run ONCE before any training on a box. Asserts the environment
# is correct so we never again burn hours only to discover a bad cap / missing
# precompute / dead GPU. Exits 0 (and notifies INFO) if all checks pass; exits 1
# (and notifies ALERT) otherwise so the worker can refuse to launch.
#
# Usage: preflight.sh <worker_name> <manifest.json>
set -u
cd /home/ec2-user/CLIMB
WORKER="${1:-worker}"
MANIFEST="${2:-}"
export CLIMB_WORKER="$WORKER"
NOTIFY="bash scripts/notify.sh"
fail=0
report=""
add(){ report="${report}
  - $1"; }

# 1. git commit actually deployed
HASH=$(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)
add "git commit: ${HASH}"

# 2. the throughput-based cap fix is present (NOT the old flat 12h that truncated runs)
if grep -q "fps / 400" scripts/launch_v2_wave.py 2>/dev/null; then
  add "watchdog cap: throughput-based (fixed) OK"
else
  add "watchdog cap: OLD FLAT CAP PRESENT — would truncate long runs"; fail=1
fi

# 3. GPU visible
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  add "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
else
  add "gpu: NOT VISIBLE"; fail=1
fi

# 4. disk headroom (checkpoints + descriptor npy staging need room)
FREEG=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
add "disk free: ${FREEG:-?}G"
if [ "${FREEG:-0}" -lt 30 ]; then add "disk: LOW (<30G)"; fail=1; fi

# 5. python deps import
PY=/home/ec2-user/venvs/climb/bin/python
if $PY -c "import torch, transformers, deepchem" >/dev/null 2>&1; then
  add "python deps: OK"
else
  add "python deps: IMPORT FAILED"; fail=1
fi

# 6. descriptor precompute reachable IFF a run in this manifest actually USES the MTR objective.
# Decided by parsing the manifest, not by grepping for the substring "mtr": every config carries
# the hyperparameter `"mtr_loss": "mse"` whether or not MTR is among its objectives, so the old
# grep fired on manifests with no MTR run at all. finalize_manifest.py then (correctly) had not
# wired descriptor_precompute_dir into such a manifest, so the wiring check below failed and
# preflight refused to launch a perfectly valid supervised-only wave.
NEEDS_MTR=$($PY - "$MANIFEST" <<'PYEOF' 2>/dev/null || echo no
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    print("no"); raise SystemExit
runs = m["runs"] if isinstance(m, dict) and "runs" in m else m
def uses_mtr(r):
    for sel in ((r.get("selection") or {}), ((r.get("pretrain_config") or {}).get("selection") or {})):
        if float((sel.get("objectives") or {}).get("mtr", 0) or 0) > 0:
            return True
    return False
print("yes" if any(uses_mtr(r) for r in runs) else "no")
PYEOF
)
if [ -n "$MANIFEST" ] && [ "$NEEDS_MTR" = "yes" ]; then
  N=$(aws s3 ls s3://climb-s3-bucket/tokenized_sources/pubchem_descriptors/ 2>/dev/null | grep -c '\.npy')
  add "descriptor precompute: ${N}/12 shards in S3"
  if [ "${N:-0}" -lt 12 ]; then add "precompute: INCOMPLETE — dense runs would run 6x slow"; fail=1; fi
  # and that the manifest actually WIRES it (the exact bug that slowed the dense arm)
  if grep -q "descriptor_precompute_dir" "$MANIFEST" 2>/dev/null; then
    add "descriptor_precompute_dir: wired in manifest OK"
  else
    add "descriptor_precompute_dir: NOT WIRED — dense runs would compute on the fly (6x slow)"; fail=1
  fi
fi

if [ "$fail" -eq 0 ]; then
  $NOTIFY INFO "${WORKER} preflight PASSED — launching" "All preflight checks passed:${report}"
  echo "PREFLIGHT OK"
  exit 0
else
  $NOTIFY ALERT "${WORKER} PREFLIGHT FAILED — not launching" "One or more checks FAILED (fix before launch):${report}"
  echo "PREFLIGHT FAILED"
  exit 1
fi
