"""Fingerprint the figure_data/ snapshot the committed figures were built from.

figure_data/ is 15 GB and gitignored, so the notebook (versioned) reads inputs that are NOT
versioned. Two people can run the identical notebook against different data and get different
figures -- which already happened: an older snapshot was missing six A1.b CV cells, so its Table
A1.b counted over 1 task instead of 6.

This writes figure_data_manifest.json (committed): for every file the notebook opens, a content
sha256 + byte size, plus prediction row-counts per dataset, plus one snapshot hash over all of it.
mtime is recorded for humans but DELIBERATELY EXCLUDED from every hash -- copying/rsync/S3 pulls
rewrite mtimes without changing content, and hashing them would raise false mismatches.

    python scripts/build_data_manifest.py            # write figure_data_manifest.json
    python scripts/build_data_manifest.py --check     # recompute + diff vs the committed manifest

Exit 0 = manifest matches (or was written). Non-zero (--check) = local figure_data differs from
the snapshot the committed figures came from; the diff says exactly which cells.
"""
import json, hashlib, sys, time
from pathlib import Path

DATA_ROOT = Path("figure_data")
MANIFEST = Path("figure_data_manifest.json")
# The file types the notebook actually opens (see notebook_cells/*.py): per-run eval summaries and
# per-molecule OOF predictions, the trainer token counts A2 reads, and the Tanimoto tables.
PATTERNS = ["**/suite_summary.json", "**/test_predictions.csv", "**/metrics.jsonl",
            "_tanimoto/*.csv"]

def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def _rows_by_dataset(path):
    """Cheap per-dataset row counts for a predictions CSV -- the signal that catches a missing
    arm x task cell without parsing the whole frame."""
    try:
        import pandas as pd
        col = pd.read_csv(path, usecols=["dataset"])
        return {k: int(v) for k, v in col.dataset.value_counts().sort_index().items()}
    except Exception:
        return None

def build():
    files = {}
    seen = set()
    for pat in PATTERNS:
        for p in sorted(DATA_ROOT.glob(pat)):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            rel = str(p.relative_to(DATA_ROOT))
            rec = {"bytes": p.stat().st_size, "sha256": _sha256(p),
                   "mtime": time.strftime("%Y-%m-%dT%H:%M", time.localtime(p.stat().st_mtime))}
            if p.name == "test_predictions.csv":
                rec["rows_by_dataset"] = _rows_by_dataset(p)
            files[rel] = rec
    # snapshot hash: content only (sorted rel path + sha256), never mtime
    snap = hashlib.sha256(json.dumps(
        sorted((k, v["sha256"]) for k, v in files.items())).encode()).hexdigest()
    return {"note": "content fingerprint of the figure_data snapshot the committed figures were "
                    "built from; mtime is informational and excluded from all hashes",
            "data_root": str(DATA_ROOT), "n_files": len(files),
            "snapshot_hash": snap, "files": files}

def check():
    if not MANIFEST.exists():
        print("no figure_data_manifest.json to check against -- build it first.")
        return 1
    want = json.loads(MANIFEST.read_text())
    have = build()
    if have["snapshot_hash"] == want["snapshot_hash"]:
        print(f"DATA MANIFEST: OK  ({have['n_files']} files, snapshot {have['snapshot_hash'][:12]})")
        return 0
    wf, hf = want["files"], have["files"]
    missing = sorted(set(wf) - set(hf))          # in committed manifest, absent locally
    extra = sorted(set(hf) - set(wf))            # present locally, not in the manifest
    changed = sorted(k for k in set(wf) & set(hf) if wf[k]["sha256"] != hf[k]["sha256"])
    print("DATA MANIFEST: MISMATCH -- your figure_data differs from the snapshot the committed "
          "figures were built from.")
    print(f"  committed snapshot {want['snapshot_hash'][:12]} ({want['n_files']} files)  "
          f"vs local {have['snapshot_hash'][:12]} ({have['n_files']} files)")
    def _show(label, items):
        if items:
            print(f"  {label} ({len(items)}):")
            for k in items[:20]:
                print("    " + k)
            if len(items) > 20:
                print(f"    ... and {len(items) - 20} more")
    _show("MISSING locally (figures used these; you don't have them)", missing)
    _show("EXTRA locally (not part of the committed snapshot)", extra)
    _show("CHANGED content (same path, different bytes)", changed)
    return 1

def main():
    if "--check" in sys.argv:
        return check()
    MANIFEST.write_text(json.dumps(build(), indent=1))
    m = json.loads(MANIFEST.read_text())
    print(f"wrote {MANIFEST}: {m['n_files']} files, snapshot {m['snapshot_hash'][:12]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
