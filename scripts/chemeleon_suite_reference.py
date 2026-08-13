"""Parse the published CheMeleon-suite baseline tables (chemeleon_suite/reference/{polaris,moleculeace}/*.md,
copied verbatim from JacksonBurns/chemeleon) into ONE tidy long CSV, so our own results can be compared to
14 baseline models without re-running them.

Output: chemeleon_suite/reference/reference_long.csv
  columns: track, model, task, seed, target_label, metric, value

Markdown shapes handled:
  * polaris: `## Random Seed N` sections; each `### `<task>`` has a table Test set|Target label|Metric|Score
  * moleculeace: `## Random Seed N` sections; each `### `<task>`` has a table metric|value
    (metrics: 'overall test rmse', 'noncliff test rmse', 'cliff test rmse')
Pure stdlib. Run from repo root."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFDIR = ROOT / "chemeleon_suite" / "reference"
SEED_RE = re.compile(r"^##\s+Random Seed\s+(\d+)")
TASK_RE = re.compile(r"^###\s+`([^`]+)`")
ROW_RE = re.compile(r"^\|")


def _split_row(line):
    # "| a | b | c |" -> ["a","b","c"]
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_md(path: Path, track: str, model: str, rows: list):
    seed = None
    task = None
    header = None
    for line in path.read_text().splitlines():
        m = SEED_RE.match(line)
        if m:
            seed = int(m.group(1)); continue
        m = TASK_RE.match(line)
        if m:
            task = m.group(1); header = None; continue
        if task and ROW_RE.match(line):
            cells = _split_row(line)
            if set("".join(cells)) <= set("-: "):   # separator row
                continue
            if header is None:
                header = [c.lower() for c in cells]
                continue
            rec = dict(zip(header, cells))
            if track == "polaris":
                # columns: (index) test set, target label, metric, score
                metric = rec.get("metric"); val = rec.get("score")
                tgt = rec.get("target label", "")
            else:  # moleculeace: metric | value
                metric = rec.get("metric"); val = rec.get("value"); tgt = ""
            if metric is None or val in (None, ""):
                continue
            try:
                fval = float(val)
            except ValueError:
                continue
            rows.append({"track": track, "model": model, "task": task, "seed": seed,
                         "target_label": tgt, "metric": metric, "value": fval})


def main():
    rows = []
    for track in ("polaris", "moleculeace"):
        d = REFDIR / track
        if not d.exists():
            continue
        for md in sorted(d.glob("*.md")):
            parse_md(md, track, md.stem, rows)
    out = REFDIR / "reference_long.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track", "model", "task", "seed", "target_label", "metric", "value"])
        w.writeheader(); w.writerows(rows)
    # quick provenance summary
    models = sorted({r["model"] for r in rows})
    tp = sorted({r["task"] for r in rows if r["track"] == "polaris"})
    tm = sorted({r["task"] for r in rows if r["track"] == "moleculeace"})
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    print(f"[ref] wrote {out}: {len(rows)} rows")
    print(f"[ref] models={len(models)}: {models}")
    print(f"[ref] polaris tasks={len(tp)}, moleculeace tasks={len(tm)}, seeds={seeds}")


if __name__ == "__main__":
    main()
