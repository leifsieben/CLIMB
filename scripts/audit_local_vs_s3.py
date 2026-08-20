"""Does every run dir a figure resolves actually exist in BOTH stores?

WHY THIS EXISTS. Coverage in this project is counted from figure_data/, and figure_data/ is a local
mirror of S3 that has drifted in BOTH directions. On 2026-08-20 the CBS cells for the three __xgb
arms were reported as a compute gap and queued for a box-hour; they had been on S3 for hours and
were simply not pulled down. The opposite drift is worse: a dir that exists only locally is one
laptop failure from being gone, against a standing rule that every result lives on S3.

So "not run" as read from a local tree is not a fact about the experiment, it is a fact about the
last sync. This resolves every source dir declared in figures/arms.py against both stores and says
which of the two is missing it:

  S3-ONLY    the figure under-reports coverage; sync down and re-count before queueing any compute
  LOCAL-ONLY the result is unprotected; push it up before the box or the laptop goes away

Listing is done ONCE PER TREE rather than once per dir -- 4 aws calls instead of ~200.

Run: python3 scripts/audit_local_vs_s3.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
S3 = "s3://climb-s3-bucket/experiments"
sys.path.insert(0, str(ROOT))
from figures.arms import ARMS                                                  # noqa: E402

# arms.py src key -> (local tree, S3 prefix). `cbs` is deliberately absent: arms.py's cbs field
# holds LABELS from the deprecated summary, not directory names -- allsuites._cbs_value resolves
# CBS from the arm's `mol` dirs under the cbs_benchmark tree instead, so that is how it is checked.
TREES = {"mol": "climb_v2_phase2",
         "mace": "chemeleon_suite/moleculeace",
         "pol": "chemeleon_suite/polaris"}
CBS_TREE = "cbs_benchmark"


def s3_dirs(prefix: str) -> set[str]:
    out = subprocess.run(["aws", "s3", "ls", f"{S3}/{prefix}/"],
                         capture_output=True, text=True).stdout
    return {ln.split("PRE ", 1)[1].rstrip("/ \n") for ln in out.splitlines() if " PRE " in ln}


def local_dirs(tree: str) -> set[str]:
    d = FD / tree
    return {p.name for p in d.iterdir() if p.is_dir()} if d.is_dir() else set()


def declared(key: str) -> dict[str, list[str]]:
    """{arm: [dir, ...]} for one src key, expanding the <base>/_s1/_s2 convention the same way the
    aggregator does -- a bare string means the arm MAY have replicate dirs beside it."""
    out = {}
    for a, m in ARMS.items():
        v = m.get("src", {}).get(key)
        if v is None:
            continue
        out[a] = list(v) if isinstance(v, (list, tuple)) else [v, f"{v}_s1", f"{v}_s2"]
    return out


def main() -> int:
    print("LOCAL vs S3 — every run dir figures/arms.py resolves")
    bad = 0
    checks = [(k, t, declared(k)) for k, t in TREES.items()]
    # CBS is resolved from the `mol` dir names under a different tree; check it that way.
    checks.append(("cbs(via mol)", CBS_TREE, declared("mol")))
    for key, tree, dec in checks:
        here, there = local_dirs(tree), s3_dirs(tree)
        s3_only, local_only = [], []
        for arm, dirs in dec.items():
            for d in dirs:
                inl, ins3 = d in here, d in there
                if ins3 and not inl:
                    s3_only.append((arm, d))
                elif inl and not ins3:
                    local_only.append((arm, d))
        print(f"\n  {key:14s} {tree:28s} local {len(here):3d}  s3 {len(there):3d}")
        for arm, d in s3_only:
            print(f"    S3-ONLY     {arm:22s} {d}  <- sync down; coverage is under-reported")
        for arm, d in local_only:
            print(f"    LOCAL-ONLY  {arm:22s} {d}  <- push up; this result is unprotected")
        bad += len(s3_only) + len(local_only)
        if not s3_only and not local_only:
            print("    OK - every declared dir that exists is in both stores")
    print(f"\n{'CLEAN' if not bad else str(bad) + ' DIR(S) IN ONLY ONE STORE'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
