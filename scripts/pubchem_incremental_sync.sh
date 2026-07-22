#!/usr/bin/env bash
# Push PubChem shards to S3 AS THEY APPEAR, instead of only when the whole download finishes.
#
# download_pubchem_full.sh syncs once, after prepare_pubchem_124m.py returns. Its header claims
# shards upload incrementally; they do not. A run that has been going since yesterday therefore
# held 53 shards (1.8 GB) that existed on one instance store and nowhere else -- one stop, spot
# reclaim or crash from losing all of it. That is a lot of hours to re-spend for nothing.
#
# Upload-only, never deletes, and touches nothing the downloader is writing: `aws s3 sync` skips
# a file whose size and mtime already match, so re-running it costs a listing and nothing else.
# A shard still being written is uploaded partially and then corrected on the next pass, because
# its size changes and sync re-uploads it.
set -uo pipefail
cd /home/ec2-user/CLIMB
OUT=raw_data/pubchem_124m_full
S3=s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full
INTERVAL=${INTERVAL:-600}

say(){ echo "[pcsync $(date -u +%H:%M:%S)] $*"; }

say "watching $OUT -> $S3 every ${INTERVAL}s"
while true; do
    n=$(ls "$OUT" 2>/dev/null | wc -l)
    aws s3 sync "$OUT" "$S3" --only-show-errors
    m=$(aws s3 ls "$S3/" 2>/dev/null | wc -l)
    say "local=$n in-s3=$m"
    # Exit once the downloader is gone AND S3 has caught up: one last sync has just run, so the
    # final shards are safe before this watcher stops.
    if ! pgrep -f prepare_pubchem_124m.py > /dev/null; then
        aws s3 sync "$OUT" "$S3" --only-show-errors
        say "downloader finished; final sync done (local=$(ls "$OUT" | wc -l))"
        bash scripts/notify.sh DONE "PubChem download finished and is fully in S3" \
            "$(ls "$OUT" | wc -l) shards at $S3 . No follow-on job was started, as requested."
        break
    fi
    sleep "$INTERVAL"
done
say "PCSYNC_DONE"
