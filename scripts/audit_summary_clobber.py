"""Find summaries that were overwritten by a single-dataset top-up.

eval_v2 opens moleculenet_summary.csv with "w". Evaluating one extra dataset into a populated
run dir therefore DELETES every other dataset -- and it is invisible from the aggregate, because
suite_summary.json is left untouched, so point estimates still resolve while the per-fold rows
(and the SD that depends on them) are gone.

The tell, and the reason this check is cheap: suite_summary.json remembers the datasets the run
ONCE had. Any dataset with a key there but no rows in the CSV is a casualty.
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = re.compile(r"^(?P<ds>[A-Za-z0-9_]+?)_(?:[a-z0-9_]+_)?(?:MEAN|STD)$")


def datasets_in_suite(p: Path) -> set:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return set()
    out = set()
    for k in d:
        m = KEY.match(k)
        if m:
            out.add(m["ds"])
    return out


def main() -> int:
    bad = []
    for suite in sorted(ROOT.glob("figure_data/*/*/moleculenet_cv*/suite_summary.json")):
        csv_p = suite.parent / "moleculenet_summary.csv"
        if not csv_p.exists():
            continue
        have = {r["dataset"] for r in csv.DictReader(csv_p.open())}
        want = datasets_in_suite(suite)
        lost = {d for d in want if d not in have and d.lower() not in {h.lower() for h in have}}
        if lost:
            bad.append((suite.parent.relative_to(ROOT), sorted(lost), sorted(have)))
    for d, lost, have in bad:
        print(f"CLOBBERED {d}\n    lost: {', '.join(lost)}\n    has : {', '.join(have) or '(none)'}")
    print(f"\n{len(bad)} clobbered summary/summaries")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
