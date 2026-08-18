#!/usr/bin/env bash
# Takes shutdown ownership of a Polaris-shard box from polaris_shard_run.sh.
#
# Two reasons this exists:
#  1. The old gate was `rc -eq 0` from six_panel_herg.py — but that driver logs FAIL for a broken
#     encoder and still exits 0, so the box could power off with work silently missing. This gates
#     on the driver's own achieved-work line, "DONE a/b" with a == b.
#  2. GPU capacity is exhausted region-wide, so the seq_* ablation wave (the fig_C_D blocker) has
#     nowhere to launch. RUN_ABLATION=1 makes this box pick it up when its shard finishes.
#
# Usage: RUN_ABLATION=0|1 nohup setsid bash scripts/shard_supervisor.sh &
set -u
cd /home/ec2-user/CLIMB
LOG=analysis/shard_supervisor.log
RUN_ABLATION="${RUN_ABLATION:-0}"
say() { echo "[sup] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "supervisor start (RUN_ABLATION=$RUN_ABLATION)"

# 1. wait for the in-flight shard driver
while pgrep -f "six_panel_herg.py" > /dev/null; do sleep 60; done
say "shard driver exited"

gate_ok() {  # last "DONE a/b" in the shard log must have a == b
  local line a b
  line=$(grep -o "DONE [0-9]*/[0-9]*" analysis/polaris_shard.log 2>/dev/null | tail -1)
  [ -n "$line" ] || return 1
  a=${line#DONE }; a=${a%%/*}; b=${line##*/}
  [ "$a" = "$b" ] && [ "$a" != "0" ]
}
if gate_ok; then say "shard COMPLETE ($(grep -o 'DONE [0-9]*/[0-9]*' analysis/polaris_shard.log | tail -1))"; SHARD_OK=1
else say "shard INCOMPLETE -> will stay UP for inspection"; SHARD_OK=0; fi

ABL_OK=1
if [ "$RUN_ABLATION" = "1" ] && [ "$SHARD_OK" = "1" ]; then
  ABL_OK=0
  say "starting seq_* ablation wave (canonical six for fig_C_D)"
  ~/venvs/climb/bin/python scripts/six_panel_ablation.py >> analysis/ablation_run.log 2>&1
  say "ablation rc=$?"
  if [ -f figure_data/SIX_PANEL_ABLATION_DONE ]; then ABL_OK=1; say "ablation COMPLETE 6/6"
  else say "ablation INCOMPLETE -> staying UP"; fi
fi

if [ "$SHARD_OK" = "1" ] && [ "$ABL_OK" = "1" ]; then
  say "all assigned work verified complete -> shutdown"
  sudo shutdown -h now
else
  say "NOT shutting down (shard_ok=$SHARD_OK abl_ok=$ABL_OK)"
fi
