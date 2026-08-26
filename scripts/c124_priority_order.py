"""The order a c124 rung will actually OPEN its shards -- so the precompute can serve training sooner.

Consumption order is deterministic and this script reproduces it exactly: data.py slices the file
list per worker (`paths[id::num_workers]`) and then shuffles each worker's own slice with
`random.Random(subset_seed)`, seed 0. Verified against a live run: the bridge's on-box cache after
1.4h contains exactly the union of the first three files per worker, nothing outside it.

PREFETCH is why the margin exists -- StreamingTokenizedDataset schedules `prefetch_files` ahead, so
a run opens a few shards beyond the ones it consumes. Predicting the consumed set exactly and
precomputing only that would crash the run on the first prefetch.

    python scripts/c124_priority_order.py --budget 50000000   # -> the shards that rung needs
"""
from __future__ import annotations
import argparse, random

NW, ROWS, MARGIN = 6, 1_000_000, 3


def worker_orders(n_shards: int, num_workers: int = NW, seed: int = 0):
    shards = [f"{i:05d}" for i in range(n_shards)]
    out = []
    for w in range(num_workers):
        sl = shards[w::num_workers]
        random.Random(seed).shuffle(sl)
        out.append(sl)
    return out


def needed(budget_fp: int, n_shards: int = 124) -> list[str]:
    """Shards a run of this budget will open, in the order the workers reach them."""
    orders = worker_orders(n_shards)
    per_worker = int(budget_fp / NW / ROWS) + 1 + MARGIN
    seen, ordered = set(), []
    for depth in range(per_worker):          # interleave: all workers advance together
        for w in range(NW):
            if depth < len(orders[w]) and orders[w][depth] not in seen:
                seen.add(orders[w][depth]); ordered.append(orders[w][depth])
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=50_000_000)
    ap.add_argument("--n_shards", type=int, default=124)
    ap.add_argument("--full", action="store_true", help="print the whole corpus, priority first")
    a = ap.parse_args()
    first = needed(a.budget, a.n_shards)
    if a.full:
        rest = [f"{i:05d}" for i in range(a.n_shards) if f"{i:05d}" not in set(first)]
        print(" ".join(first + rest))
    else:
        print(" ".join(first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
