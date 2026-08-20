"""Merge fig_F's scored Ames cells into the v2 concat panel tables.

WHY THIS IS A SEPARATE STEP. Ames is Polaris tdcommons/ames, whose test labels are withheld, so
its cells cannot be scored where they are produced: concat_redundancy_panels.py writes predictions
and the scoring happens off-box in the polaris venv (scripts/score_polaris_concat.py), landing in
figure_data/chemeleon_suite/polaris/concat_<emb>/polaris_scores.csv. The panel tables that fig_F
reads are written by the box before that scoring exists, so their Ames rows have to be merged in
afterwards. That is the whole job of this script.

WHAT IT GUARDS AGAINST, each of which has actually happened in this project:

  * REWRITE INSTEAD OF MERGE. The panel table already holds MoleculeACE and CBS. Writing Ames rows
    with a plain open("w") drops them. This reads the file, replaces only its Ames rows, and keeps
    every other row in its original order.
  * APPENDING TWICE. Re-running must not leave two Ames blocks -- fig_F's .loc would return a
    Series and the panel would crash or silently take the first. Existing Ames rows are dropped
    before the new ones go in, so the script is idempotent.
  * POOLING 28 POLARIS TASKS INTO ONE PANEL. The scorer's file is per-task. Ames is ONE task; if a
    future scorer emits all 28 and this script averaged them, the Ames cell would be a suite mean
    wearing a task's name (exactly the SI fig f bug). Rows are filtered to tdcommons/ames and the
    filter is asserted to have found something.
  * A TAG RENAME EMPTYING THE PANEL. The feature-block names must match the suffixes fig_F builds
    its keys from (CLM / CLMsup / CheMel). If the scorer's tag drifts, the panel would go blank
    with no error. The expected block set is asserted per embedding.

The `features` column carries SEVEN FEATURE BLOCKS, not replicates. There is one scored value per
block and no seed axis, so `std` is deliberately empty -- an Ames error bar would have to come from
a bootstrap over the test set, which is not what the other panels' bars mean.

Usage:  python scripts/merge_concat_ames_panels.py [climb|climb_sup|chemeleon ...]
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = "tdcommons/ames"
PANEL = "Ames"
METRIC = "roc_auc"
FIELDS = ["task", "features", "metric", "mean", "std"]

# suffix -> the blocks that embedding's pass must produce. Mirrors ROLE_SUFFIX in figures/fig_F.py.
BLOCKS = {"climb": "CLM", "climb_sup": "CLMsup", "chemeleon": "CheMel"}
FREE = ["fp", "desc", "fp+desc"]                       # embedding-free, produced by every pass


def scores_for(emb: str) -> dict[str, float]:
    src = ROOT / "figure_data" / "chemeleon_suite" / "polaris" / f"concat_{emb}" / "polaris_scores.csv"
    if not src.exists():
        print(f"  {emb:10} polaris_scores.csv absent -- panel stays 'not run'")
        return {}
    out = {}
    for r in csv.DictReader(src.open()):
        if r["task"] != TASK:
            continue
        assert r["metric"] == METRIC, f"{emb}: unexpected metric {r['metric']!r}"
        assert r["features"] not in out, f"{emb}: duplicate block {r['features']!r}"
        out[r["features"]] = float(r["value"])
    assert out, f"{emb}: no {TASK} rows in {src} -- wrong task filter or an empty scoring run"
    sfx = BLOCKS[emb]
    want = set(FREE) | {sfx, f"fp+{sfx}", f"desc+{sfx}", f"fp+desc+{sfx}"}
    assert set(out) == want, f"{emb}: blocks {sorted(set(out) ^ want)} unexpected/missing (tag drift?)"
    return out


def merge(emb: str, sc: dict[str, float]) -> None:
    tbl = ROOT / "analysis" / "rigor" / f"concat_panels_{emb}_v2.csv"
    if not tbl.exists():
        print(f"  {emb:10} {tbl.name} absent -- nothing to merge into"); return
    kept = [r for r in csv.DictReader(tbl.open()) if r["task"] != PANEL]
    rows = kept + [{"task": PANEL, "features": f, "metric": METRIC,
                    "mean": f"{v:.4f}", "std": ""} for f, v in sorted(sc.items())]
    with tbl.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"  {emb:10} {len(sc)} Ames rows -> {tbl.name} ({len(kept)} other rows kept)")


def main(embs: list[str]) -> int:
    got = {e: scores_for(e) for e in embs}
    live = {e: s for e, s in got.items() if s}

    # The embedding-free blocks are run by EVERY pass, so they arrive once per embedding. No
    # embedding enters them, so the passes cannot influence each other -- the repeat is a free
    # reproducibility check. Report the spread rather than quietly keeping one.
    for blk in FREE:
        vals = {e: s[blk] for e, s in live.items() if blk in s}
        if len(vals) > 1:
            lo, hi = min(vals.values()), max(vals.values())
            rel = 100 * (hi - lo) / abs(lo) if lo else float("nan")
            print(f"  {blk:8} across {len(vals)} passes: {lo:.4f}..{hi:.4f}  ({rel:.3f}%)")

    for e, s in live.items():
        merge(e, s)
        sfx = BLOCKS[e]
        base, full = s.get("fp+desc"), s.get(f"fp+desc+{sfx}")
        if base and full:
            d = full - base
            print(f"  {e:10} verdict: fp+desc {base:.4f} -> +{sfx} {full:.4f} "
                  f"({d:+.4f}, {'gain' if d > 0 else 'COST'}; roc_auc, higher better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["climb", "climb_sup", "chemeleon"]))
