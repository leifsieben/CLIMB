"""E13 / H2c — build the CORRUPTED-PRETRAINING control manifest (source for Fig B2).

The control holds objective, data volume, compute, schedule and model fixed and removes
only the *chemical content*, so a downstream gain that survives corruption cannot be
chemical information — it isolates H2 mechanism (c) "adds information" from (a)/(b)
"initialization / regularization".

Two arms, each matched to its real counterpart at the 8M matched budget:

    corrupt_mlm_8M  (corruption=shuffle_tokens)  <-> unsup_8M       (real MLM)
    corrupt_mtr_8M  (corruption=shuffle_targets) <-> skip_dense_8M  (real dense MTR)

Fig B2 then reads as three bars per task:
    no_pretrain  |  corrupted pretraining  |  real pretraining
  - corrupted ≈ no_pretrain  and  real > both   -> pretraining adds INFORMATION (c)
  - corrupted ≈ real         and  both > none   -> the gain is the objective's STRUCTURE (a/b)

Usage:
    python scripts/make_e13_manifest.py --spec configs/v2_phase2.yaml --output e13_manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source run to clone, new run id, corruption mode, what it controls for)
ARMS = [
    ("unsup_8M", "corrupt_mlm_8M", "shuffle_tokens", "real MLM (unsup_only)"),
    ("skip_dense_8M", "corrupt_mtr_8M", "shuffle_targets", "real dense MTR (sup_only:dense)"),
]


def build(spec: Path, output: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        full_path = Path(tmp.name)
    subprocess.run([sys.executable, str(ROOT / "experiment_v2.py"),
                    "--spec", str(spec), "--output", str(full_path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    full = json.loads(full_path.read_text())
    by = {r["output_dir"].split("/")[-1]: r for r in full["runs"]}

    runs = []
    for src, new_id, mode, controls in ARMS:
        if src not in by:
            raise KeyError(f"source run {src!r} not in generated manifest")
        r = json.loads(json.dumps(by[src]))          # deep copy
        r["output_dir"] = r["output_dir"].replace(src, new_id)
        r["backup_s3_uri"] = r["backup_s3_uri"].replace(src, new_id)
        r["evaluation_output_dir"] = r["evaluation_output_dir"].replace(src, new_id)
        r["pretrain_config"]["run_id"] = new_id
        r["pretrain_config"]["selection"]["corruption"] = mode
        runs.append(r)
        sel = r["pretrain_config"]["selection"]
        print(f"  {new_id:16} corruption={mode:16} objectives={r['pretrain_config']['selection'].get('objectives')} "
              f"fp={sel['total_forward_passes']} (controls for {controls})")

    out = {k: full[k] for k in full if k != "runs"}
    out["runs"] = runs
    output.write_text(json.dumps(out, indent=2))
    full_path.unlink(missing_ok=True)
    print(f"[e13] wrote {len(runs)} corrupted-control runs to {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="configs/v2_phase2.yaml")
    p.add_argument("--output", default="e13_manifest.json")
    a = p.parse_args()
    build(Path(a.spec), Path(a.output))


if __name__ == "__main__":
    main()
