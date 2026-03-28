"""Experiment spec expansion for CLIMB pretraining studies."""

from __future__ import annotations

import copy
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from config import PretrainingConfig
from utils import git_commit_sha


DEFAULT_SUPERVISED_FAMILIES = [
    {"name": "PCQM", "prefix": "PCQM__"},
    {"name": "WONG", "prefix": "WONG__"},
    {"name": "L1000_MCF7", "prefix": "L1000_MCF7__"},
    {"name": "L1000_VCAP", "prefix": "L1000_VCAP__"},
    {"name": "PCBA", "prefix": "PCBA__"},
]
DEFAULT_UNSUPERVISED_BASELINES = [
    1_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    10_000_000_000,
]
DEFAULT_UNSUPERVISED_COVERAGE = [0.10, 0.25, 0.50, 0.75, 1.00]
DEFAULT_MIXED_RATIOS = [(0.10, 0.90), (0.20, 0.80), (0.50, 0.50), (0.80, 0.20), (0.90, 0.10)]
DEFAULT_MOLECULENET_DATASETS = [
    "QM7",
    "QM8",
    "QM9",
    "Tox21",
    "BBBP",
    "ToxCast",
    "SIDER",
    "ClinTox",
    "HIV",
    "BACE",
    "MUV",
    "PCBA",
    "ESOL",
    "FreeSolv",
    "Lipophilicity",
]


@dataclass(frozen=True)
class SequenceChoice:
    families: Tuple[str, ...]
    order_score: float


def load_spec(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def _slug(value: str) -> str:
    return value.lower().replace("/", "_").replace("%", "p").replace(".", "_")


def _s3_join(root: Optional[str], leaf: str) -> Optional[str]:
    if not root:
        return None
    return f"{root.rstrip('/')}/{leaf.lstrip('/')}"


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _balanced_representative_sequences(
    families: Sequence[str],
    k: int,
    n_sequences: int,
    seed: int,
) -> List[Tuple[str, ...]]:
    candidates = list(itertools.permutations(families, k))
    if n_sequences >= len(candidates):
        return candidates

    selected: List[Tuple[str, ...]] = []
    used = set()
    family_counts = Counter()
    position_counts = Counter()

    def score(seq: Tuple[str, ...]) -> Tuple[float, float, int, Tuple[str, ...]]:
        presence_cost = sum((family_counts[name] + 1) ** 2 for name in set(seq))
        position_cost = sum((position_counts[(idx, name)] + 1) ** 2 for idx, name in enumerate(seq))
        tie_break = (seed + sum(ord(ch) for item in seq for ch in item)) % 997
        return (presence_cost, position_cost, tie_break, seq)

    for _ in range(n_sequences):
        best = min((seq for seq in candidates if seq not in used), key=score)
        used.add(best)
        selected.append(best)
        for idx, name in enumerate(best):
            family_counts[name] += 1
            position_counts[(idx, name)] += 1
    return selected


def _runtime_estimate(
    token_budget: Optional[int],
    calibration: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    tokens_per_second = calibration.get("tokens_per_second")
    eval_hours = calibration.get("evaluation_hours")
    if token_budget is None or not tokens_per_second:
        return {
            "pretrain_hours": None,
            "evaluation_hours": eval_hours,
            "bundle_hours": None,
            "estimate_basis": "uncalibrated",
        }
    pretrain_hours = token_budget / float(tokens_per_second) / 3600.0
    bundle_hours = pretrain_hours + (eval_hours or 0.0)
    return {
        "pretrain_hours": round(pretrain_hours, 3),
        "evaluation_hours": eval_hours,
        "bundle_hours": round(bundle_hours, 3),
        "estimate_basis": "calibrated",
    }


def _build_base_pretrain_config(spec: Dict[str, Any], run_id: str, output_dir: str, run_metadata: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "name": run_id,
        "output_dir": output_dir,
        "tokenizer_path": spec["tokenizer_path"],
        "model": spec["model"],
        "compute_budget": spec["compute_budget_defaults"],
        "mlm_training": spec["mlm_training"],
        "unsupervised_data": spec["unsupervised_data"],
        "supervised_parquet_path": spec.get("supervised_parquet_path"),
        "supervised_tokenized_parquet_path": spec.get("supervised_tokenized_parquet_path"),
        "supervised_families": spec.get("supervised_families", DEFAULT_SUPERVISED_FAMILIES),
        "supervised_training": spec["supervised_training"],
        "validation_fraction": spec.get("validation_fraction", 0.0),
        "backup_s3_uri": run_metadata.get("backup_s3_uri"),
        "run_metadata": run_metadata,
    }
    return PretrainingConfig.from_dict(payload).to_dict()


def _smoke_runs(spec: Dict[str, Any], results_root: Path, calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = []
    smoke_defs = [
        {
            "run_id": "smoke_unsup_10pct_1m",
            "run_type": "smoke_unsupervised",
            "compute_budget": {"total_epochs": 1, "supervised_fraction": 0.0, "total_tokens": 1_000_000},
            "config_updates": {
                "unsupervised_subset_fraction": 0.10,
                "unsupervised_subset_seed": 101,
                "mlm_training": {"shuffle": True},
            },
        },
        {
            "run_id": "smoke_supervised_full_1epoch",
            "run_type": "smoke_supervised",
            "compute_budget": {"total_epochs": 1, "supervised_fraction": 1.0, "total_tokens": None},
            "config_updates": {
                "supervised_families": spec.get("supervised_families", DEFAULT_SUPERVISED_FAMILIES),
                "supervised_training": {"num_epochs": 1},
            },
        },
        {
            "run_id": "smoke_mixed_90_10_100m",
            "run_type": "smoke_mixed",
            "compute_budget": {"total_epochs": 1, "supervised_fraction": 0.10, "total_tokens": 100_000_000},
            "config_updates": {
                "unsupervised_subset_fraction": 1.0,
                "unsupervised_subset_seed": 303,
                "mlm_training": {"shuffle": True},
                "supervised_training": {"num_epochs": 1},
            },
        },
    ]

    for entry in smoke_defs:
        run_id = entry["run_id"]
        output_dir = results_root / run_id
        backup_s3_uri = _s3_join(spec.get("s3_backup_root"), run_id)
        run_metadata = {
            "stage": "smoke",
            "run_type": entry["run_type"],
            "backup_s3_uri": backup_s3_uri,
            "evaluation_datasets": spec.get("evaluation", {}).get("datasets", DEFAULT_MOLECULENET_DATASETS),
        }
        pretrain_config = _deep_merge(
            _build_base_pretrain_config(spec, run_id, str(output_dir), run_metadata),
            {
                "compute_budget": entry["compute_budget"],
                **entry["config_updates"],
            },
        )
        token_budget = pretrain_config["compute_budget"].get("total_tokens")
        runs.append(
            {
                "run_id": run_id,
                "stage": "smoke",
                "run_type": entry["run_type"],
                "output_dir": str(output_dir),
                "backup_s3_uri": backup_s3_uri,
                "evaluation_output_dir": str(output_dir / "moleculenet"),
                "pretrain_config": pretrain_config,
                "selection": {},
                "runtime_estimate": _runtime_estimate(token_budget, calibration),
            }
        )
    return runs


def _unsupervised_baseline_runs(spec: Dict[str, Any], results_root: Path, calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = []
    for token_budget in spec.get("unsupervised_baseline_budgets", DEFAULT_UNSUPERVISED_BASELINES):
        for replicate in range(spec.get("unsupervised_baseline_replicates", 3)):
            run_id = f"unsup_baseline_{token_budget}_{replicate:02d}"
            output_dir = results_root / run_id
            backup_s3_uri = _s3_join(spec.get("s3_backup_root"), run_id)
            run_metadata = {
                "stage": "main",
                "run_type": "unsupervised_baseline",
                "replicate": replicate,
                "backup_s3_uri": backup_s3_uri,
                "selection": {"token_budget": token_budget},
            }
            pretrain_config = _deep_merge(
                _build_base_pretrain_config(spec, run_id, str(output_dir), run_metadata),
                {
                    "compute_budget": {"total_epochs": 1, "supervised_fraction": 0.0, "total_tokens": token_budget},
                    "unsupervised_subset_seed": 10_000 + replicate,
                    "unsupervised_subset_fraction": 1.0,
                    "mlm_training": {"shuffle": True},
                },
            )
            runs.append(
                {
                    "run_id": run_id,
                    "stage": "main",
                    "run_type": "unsupervised_baseline",
                    "output_dir": str(output_dir),
                    "backup_s3_uri": backup_s3_uri,
                    "evaluation_output_dir": str(output_dir / "moleculenet"),
                    "pretrain_config": pretrain_config,
                    "selection": {"token_budget": token_budget, "replicate": replicate},
                    "runtime_estimate": _runtime_estimate(token_budget, calibration),
                }
            )
    return runs


def _supervised_order_runs(spec: Dict[str, Any], results_root: Path, calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
    families_cfg = spec.get("supervised_families", DEFAULT_SUPERVISED_FAMILIES)
    family_names = [item["name"] for item in families_cfg]
    families_by_name = {item["name"]: item for item in families_cfg}
    runs = []

    for k in range(1, len(family_names) + 1):
        sequences = _balanced_representative_sequences(
            family_names,
            k,
            n_sequences=spec.get("supervised_order_sequences_per_k", 5),
            seed=spec.get("seed", 0) + k,
        )
        for seq_idx, seq in enumerate(sequences):
            run_id = f"sup_order_{k}of{len(family_names)}_{seq_idx:02d}"
            output_dir = results_root / run_id
            backup_s3_uri = _s3_join(spec.get("s3_backup_root"), run_id)
            selected = [families_by_name[name] for name in seq]
            run_metadata = {
                "stage": "main",
                "run_type": "supervised_order_ramp",
                "backup_s3_uri": backup_s3_uri,
                "selection": {
                    "family_count": k,
                    "sequence_index": seq_idx,
                    "family_order": list(seq),
                },
            }
            pretrain_config = _deep_merge(
                _build_base_pretrain_config(spec, run_id, str(output_dir), run_metadata),
                {
                    "compute_budget": {"total_epochs": 1, "supervised_fraction": 1.0, "total_tokens": None},
                    "supervised_families": selected,
                    "supervised_training": {"num_epochs": 1},
                },
            )
            runs.append(
                {
                    "run_id": run_id,
                    "stage": "main",
                    "run_type": "supervised_order_ramp",
                    "output_dir": str(output_dir),
                    "backup_s3_uri": backup_s3_uri,
                    "evaluation_output_dir": str(output_dir / "moleculenet"),
                    "pretrain_config": pretrain_config,
                    "selection": {"family_count": k, "sequence_index": seq_idx, "family_order": list(seq)},
                    "runtime_estimate": _runtime_estimate(None, calibration),
                }
            )
    return runs


def _unsupervised_coverage_runs(spec: Dict[str, Any], results_root: Path, calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = []
    total_tokens = spec.get("fixed_compute_token_budget", 10_000_000_000)
    for pct in spec.get("unsupervised_coverage_fractions", DEFAULT_UNSUPERVISED_COVERAGE):
        pct_slug = str(int(round(pct * 100)))
        run_id = f"unsup_cov_{pct_slug}pct_10b"
        output_dir = results_root / run_id
        backup_s3_uri = _s3_join(spec.get("s3_backup_root"), run_id)
        run_metadata = {
            "stage": "main",
            "run_type": "unsupervised_fixed_budget",
            "backup_s3_uri": backup_s3_uri,
            "selection": {"unsupervised_fraction": pct, "token_budget": total_tokens},
        }
        pretrain_config = _deep_merge(
            _build_base_pretrain_config(spec, run_id, str(output_dir), run_metadata),
            {
                "compute_budget": {"total_epochs": 1, "supervised_fraction": 0.0, "total_tokens": total_tokens},
                "unsupervised_subset_fraction": pct,
                "unsupervised_subset_seed": 20_000 + int(pct * 100),
                "mlm_training": {"shuffle": True},
            },
        )
        runs.append(
            {
                "run_id": run_id,
                "stage": "main",
                "run_type": "unsupervised_fixed_budget",
                "output_dir": str(output_dir),
                "backup_s3_uri": backup_s3_uri,
                "evaluation_output_dir": str(output_dir / "moleculenet"),
                "pretrain_config": pretrain_config,
                "selection": {"unsupervised_fraction": pct, "token_budget": total_tokens},
                "runtime_estimate": _runtime_estimate(total_tokens, calibration),
            }
        )
    return runs


def _mixed_runs(spec: Dict[str, Any], results_root: Path, calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = []
    total_tokens = spec.get("fixed_compute_token_budget", 10_000_000_000)
    families_cfg = spec.get("supervised_families", DEFAULT_SUPERVISED_FAMILIES)
    family_names = [item["name"] for item in families_cfg]
    families_by_name = {item["name"]: item for item in families_cfg}
    family_orders = _balanced_representative_sequences(
        family_names,
        len(family_names),
        n_sequences=spec.get("mixed_replicates", 3),
        seed=spec.get("seed", 0) + 500,
    )

    for unsup_ratio, sup_ratio in spec.get("mixed_ratios", DEFAULT_MIXED_RATIOS):
        for replicate in range(spec.get("mixed_replicates", 3)):
            ratio_slug = f"{int(unsup_ratio * 100):02d}_{int(sup_ratio * 100):02d}"
            run_id = f"mixed_{ratio_slug}_{replicate:02d}"
            output_dir = results_root / run_id
            backup_s3_uri = _s3_join(spec.get("s3_backup_root"), run_id)
            family_order = family_orders[replicate % len(family_orders)]
            selected = [families_by_name[name] for name in family_order]
            run_metadata = {
                "stage": "main",
                "run_type": "mixed_fixed_budget",
                "backup_s3_uri": backup_s3_uri,
                "selection": {
                    "unsupervised_ratio": unsup_ratio,
                    "supervised_ratio": sup_ratio,
                    "replicate": replicate,
                    "family_order": list(family_order),
                    "token_budget": total_tokens,
                },
            }
            pretrain_config = _deep_merge(
                _build_base_pretrain_config(spec, run_id, str(output_dir), run_metadata),
                {
                    "compute_budget": {
                        "total_epochs": 1,
                        "supervised_fraction": sup_ratio,
                        "total_tokens": total_tokens,
                    },
                    "unsupervised_subset_fraction": 1.0,
                    "unsupervised_subset_seed": 30_000 + replicate,
                    "supervised_families": selected,
                    "mlm_training": {"shuffle": True},
                    "supervised_training": {"num_epochs": 1},
                },
            )
            runs.append(
                {
                    "run_id": run_id,
                    "stage": "main",
                    "run_type": "mixed_fixed_budget",
                    "output_dir": str(output_dir),
                    "backup_s3_uri": backup_s3_uri,
                    "evaluation_output_dir": str(output_dir / "moleculenet"),
                    "pretrain_config": pretrain_config,
                    "selection": {
                        "unsupervised_ratio": unsup_ratio,
                        "supervised_ratio": sup_ratio,
                        "replicate": replicate,
                        "family_order": list(family_order),
                        "token_budget": total_tokens,
                    },
                    "runtime_estimate": _runtime_estimate(total_tokens, calibration),
                }
            )
    return runs


def generate_manifest(spec: Dict[str, Any], spec_path: Optional[str] = None) -> Dict[str, Any]:
    results_root = Path(spec["results_root"])
    calibration = spec.get("calibration", {})
    runs = []
    runs.extend(_smoke_runs(spec, results_root, calibration))
    runs.extend(_unsupervised_baseline_runs(spec, results_root, calibration))
    runs.extend(_supervised_order_runs(spec, results_root, calibration))
    runs.extend(_unsupervised_coverage_runs(spec, results_root, calibration))
    runs.extend(_mixed_runs(spec, results_root, calibration))

    counts = Counter(run["run_type"] for run in runs if run["stage"] == "main")
    summary = {
        "main_counts": dict(counts),
        "expected": {
            "unsupervised_baseline": len(
                spec.get("unsupervised_baseline_budgets", DEFAULT_UNSUPERVISED_BASELINES)
            )
            * spec.get("unsupervised_baseline_replicates", 3),
            "supervised_order_ramp": len(spec.get("supervised_families", DEFAULT_SUPERVISED_FAMILIES))
            * spec.get("supervised_order_sequences_per_k", 5),
            "unsupervised_fixed_budget": len(
                spec.get("unsupervised_coverage_fractions", DEFAULT_UNSUPERVISED_COVERAGE)
            ),
            "mixed_fixed_budget": len(spec.get("mixed_ratios", DEFAULT_MIXED_RATIOS))
            * spec.get("mixed_replicates", 3),
        },
        "smoke_runs": 3,
        "total_runs": len(runs),
    }
    return {
        "name": spec.get("name", "climb_experiments"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "spec_path": spec_path,
        "results_root": str(results_root),
        "s3_backup_root": spec.get("s3_backup_root"),
        "cluster": spec.get("cluster", {}),
        "evaluation": spec.get("evaluation", {}),
        "summary": summary,
        "runs": runs,
    }


def write_manifest(manifest: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
