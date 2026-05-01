"""v2 experiment manifest generator.

One YAML spec → one resolved manifest.json with all run entries materialised.
Each run knows: run_id, run_type, output_dir, backup_s3_uri, pretrain_config (a
self-contained YAML dict ready to drop into pretrain_v2.py / random_baseline_v2.py),
and a `selection` block with the experiment cell coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import yaml

from config_v2 import (
    EvalConfigV2,
    ModelConfigV2,
    SUPERVISED_FAMILIES_V2,
    TrainingConfigV2,
)


RUN_TYPES = {
    "smoke",
    "random_baseline",
    "unsup_only",
    "sup_only",
    "mixed",
    "family_order",
    "stretch_unsup",
}


def _build_pretrain_config(
    spec: dict,
    *,
    mixing_ratio: float,
    n_families: int,
    family_order: List[str],
    pretraining_seed: int,
    total_forward_passes: int,
    run_id: str,
) -> dict:
    base = {
        "run_id": run_id,
        "tokenizer_path": spec["tokenizer_path"],
        "unsupervised_data_paths": spec["unsupervised_data_paths"],
        "supervised_tokenized_parquet_path": spec.get("supervised_tokenized_parquet_path"),
        "model": spec.get("model", {}),
        "training": spec.get("training", {}),
        "selection": {
            "mixing_ratio": mixing_ratio,
            "n_families": n_families,
            "family_order": family_order,
            "pretraining_seed": pretraining_seed,
            "total_forward_passes": total_forward_passes,
        },
        "evaluation": spec.get("evaluation", {}),
        "max_supervised_rows": spec.get("max_supervised_rows"),
        "unsupervised_subset_fraction": spec.get("unsupervised_subset_fraction"),
    }
    return base


def _build_baseline_config(spec: dict, *, pretraining_seed: int, run_id: str) -> dict:
    return {
        "run_id": run_id,
        "tokenizer_path": spec["tokenizer_path"],
        "model": spec.get("model", {}),
        "selection": {"pretraining_seed": pretraining_seed},
        "evaluation": spec.get("evaluation", {}),
    }


def _output_dir(spec: dict, run_id: str) -> str:
    return f"{spec['results_root']}/{run_id}"


def _backup_uri(spec: dict, run_id: str) -> str:
    return f"{spec['s3_backup_root']}/{run_id}"


def _emit(run_type: str, run_id: str, pretrain_config: dict, spec: dict, selection: dict, requires_pretrain: bool) -> dict:
    return {
        "run_id": run_id,
        "run_type": run_type,
        "stage": "main",
        "requires_pretrain": requires_pretrain,
        "output_dir": _output_dir(spec, run_id),
        "backup_s3_uri": _backup_uri(spec, run_id),
        "evaluation_output_dir": f"{_output_dir(spec, run_id)}/moleculenet",
        "pretrain_config": pretrain_config,
        "selection": selection,
    }


def _smoke_runs(spec: dict) -> List[dict]:
    out = []
    selection = {
        "mixing_ratio": 0.5, "n_families": 2,
        "family_order": SUPERVISED_FAMILIES_V2[:2],
        "pretraining_seed": 0, "total_forward_passes": 1_000_000,
    }
    cfg = _build_pretrain_config(
        spec, mixing_ratio=0.5, n_families=2,
        family_order=SUPERVISED_FAMILIES_V2[:2],
        pretraining_seed=0, total_forward_passes=1_000_000, run_id="smoke_v2",
    )
    # Smoke is quick-validate; cap supervised rows so the load doesn't dominate the run.
    cfg["max_supervised_rows"] = 50_000
    out.append(_emit("smoke", "smoke_v2", cfg, spec, selection, requires_pretrain=True))
    return out


def _random_baseline_runs(spec: dict) -> List[dict]:
    out = []
    for rep in range(spec.get("random_baseline_replicates", 3)):
        rid = f"random_baseline_{rep:02d}"
        sel = {"pretraining_seed": rep}
        cfg = _build_baseline_config(spec, pretraining_seed=rep, run_id=rid)
        out.append(_emit("random_baseline", rid, cfg, spec, sel, requires_pretrain=False))
    return out


def _unsup_only_runs(spec: dict) -> List[dict]:
    """5 fractions of B (1/5..5/5), 3 reps each."""
    out = []
    base_b = int(spec["training"]["total_forward_passes"])
    for k in range(1, 6):
        budget = base_b * k // 5
        for rep in range(3):
            rid = f"unsup_only_{k}of5_{rep:02d}"
            sel = {
                "mixing_ratio": 1.0, "n_families": 0, "family_order": [],
                "pretraining_seed": rep, "total_forward_passes": budget,
            }
            cfg = _build_pretrain_config(
                spec, mixing_ratio=1.0, n_families=0, family_order=[],
                pretraining_seed=rep, total_forward_passes=budget, run_id=rid,
            )
            out.append(_emit("unsup_only", rid, cfg, spec, sel, requires_pretrain=True))
    return out


def _sup_only_runs(spec: dict) -> List[dict]:
    """5 family counts (1..5), 3 reps each. Canonical alphabetical order."""
    out = []
    budget = int(spec["training"]["total_forward_passes"])
    for n_fam in range(1, 6):
        order = SUPERVISED_FAMILIES_V2[:n_fam]
        for rep in range(3):
            rid = f"sup_only_{n_fam}fam_{rep:02d}"
            sel = {
                "mixing_ratio": 0.0, "n_families": n_fam, "family_order": order,
                "pretraining_seed": rep, "total_forward_passes": budget,
            }
            cfg = _build_pretrain_config(
                spec, mixing_ratio=0.0, n_families=n_fam, family_order=order,
                pretraining_seed=rep, total_forward_passes=budget, run_id=rid,
            )
            out.append(_emit("sup_only", rid, cfg, spec, sel, requires_pretrain=True))
    return out


def _mixed_runs(spec: dict) -> List[dict]:
    """3 ratios, 3 reps each. Use all 5 families for the supervised side."""
    out = []
    budget = int(spec["training"]["total_forward_passes"])
    ratios = [(0.25, 0.75), (0.50, 0.50), (0.75, 0.25)]
    for unsup_pct, sup_pct in ratios:
        tag = f"{int(unsup_pct*100)}_{int(sup_pct*100)}"
        for rep in range(3):
            rid = f"mixed_{tag}_{rep:02d}"
            sel = {
                "mixing_ratio": unsup_pct, "n_families": 5,
                "family_order": SUPERVISED_FAMILIES_V2,
                "pretraining_seed": rep, "total_forward_passes": budget,
            }
            cfg = _build_pretrain_config(
                spec, mixing_ratio=unsup_pct, n_families=5,
                family_order=SUPERVISED_FAMILIES_V2,
                pretraining_seed=rep, total_forward_passes=budget, run_id=rid,
            )
            out.append(_emit("mixed", rid, cfg, spec, sel, requires_pretrain=True))
    return out


def _family_order_runs(spec: dict) -> List[dict]:
    """5 different orderings at family count 3, 3 reps each.
    The "order" is the family selection (which 3-of-5)."""
    out = []
    budget = int(spec["training"]["total_forward_passes"])
    # 5 distinct 3-element subsets of the 5-element family list
    fams = SUPERVISED_FAMILIES_V2
    orderings = [
        [fams[0], fams[1], fams[2]],
        [fams[0], fams[1], fams[3]],
        [fams[0], fams[2], fams[4]],
        [fams[1], fams[3], fams[4]],
        [fams[2], fams[3], fams[4]],
    ]
    for ord_idx, order in enumerate(orderings):
        for rep in range(3):
            rid = f"family_order_3of5_v{ord_idx}_{rep:02d}"
            sel = {
                "mixing_ratio": 0.0, "n_families": 3, "family_order": order,
                "pretraining_seed": rep, "total_forward_passes": budget,
                "ordering_index": ord_idx,
            }
            cfg = _build_pretrain_config(
                spec, mixing_ratio=0.0, n_families=3, family_order=order,
                pretraining_seed=rep, total_forward_passes=budget, run_id=rid,
            )
            out.append(_emit("family_order", rid, cfg, spec, sel, requires_pretrain=True))
    return out


def _stretch_runs(spec: dict) -> List[dict]:
    """500M and 1B FP unsup-only, 3 reps each. Stretch H6."""
    out = []
    budgets = [500_000_000, 1_000_000_000]
    for budget in budgets:
        for rep in range(3):
            tag = f"{budget // 1_000_000}M" if budget < 1_000_000_000 else f"{budget // 1_000_000_000}B"
            rid = f"stretch_unsup_{tag}_{rep:02d}"
            sel = {
                "mixing_ratio": 1.0, "n_families": 0, "family_order": [],
                "pretraining_seed": rep, "total_forward_passes": budget,
            }
            cfg = _build_pretrain_config(
                spec, mixing_ratio=1.0, n_families=0, family_order=[],
                pretraining_seed=rep, total_forward_passes=budget, run_id=rid,
            )
            out.append(_emit("stretch_unsup", rid, cfg, spec, sel, requires_pretrain=True))
    return out


def generate_manifest(spec: dict) -> dict:
    runs: List[dict] = []
    runs.extend(_smoke_runs(spec))
    runs.extend(_random_baseline_runs(spec))
    runs.extend(_unsup_only_runs(spec))
    runs.extend(_sup_only_runs(spec))
    runs.extend(_mixed_runs(spec))
    runs.extend(_family_order_runs(spec))
    if spec.get("include_stretch", False):
        runs.extend(_stretch_runs(spec))
    return {
        "name": spec.get("name", "robust_matrix_v2"),
        "results_root": spec["results_root"],
        "s3_backup_root": spec["s3_backup_root"],
        "tokenizer_path": spec["tokenizer_path"],
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    manifest = generate_manifest(spec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest['runs'])} runs to {out_path}")


if __name__ == "__main__":
    main()
