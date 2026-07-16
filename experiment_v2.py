"""v2 experiment manifest generator.

One YAML spec → one resolved manifest.json. Lean grid built around the paper's four
experiments (see the plan): a validation gate (A), the core comparison (B), the
pretraining-data-scaling curve (C), and the label-efficiency curve (D, which is a
pure eval sweep over B's encoders and needs no extra runs here).

Run families emitted:
  - smoke          : 1 canonical + 1 enumerated tiny MLM run (path sanity)
  - ecfp4_anchor   : classical ECFP4 + XGBoost baseline (no encoder)
  - random_baseline: untrained-encoder floor (N reps)
  - unsup_only / sup_only / mixed : the core comparison at a fixed budget B_core
  - scaling        : unsup_only, unique-molecule fraction × {canonical, enumerated}

Every pretrain run entry carries a self-contained `pretrain_config` ready for
pretrain_v2.py, plus a `selection` block with the experiment-cell coordinates and an
optional `eval_override` for anchor runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import yaml

from config_v2 import SUPERVISED_FAMILIES_V2


# ---------- config builders ----------

def _build_pretrain_config(
    spec: dict, *, run_id: str, mixing_ratio: float, n_families: int,
    family_order: List[str], pretraining_seed: int, total_forward_passes: int,
    augmentation: str = "canonical", unsupervised_subset_fraction: Optional[float] = None,
) -> dict:
    return {
        "run_id": run_id,
        "tokenizer_path": spec["tokenizer_path"],
        "unsupervised_data_paths": spec.get("unsupervised_data_paths"),
        "unsupervised_raw_smiles_paths": spec.get("unsupervised_raw_smiles_paths"),
        "supervised_tokenized_parquet_path": spec.get("supervised_tokenized_parquet_path"),
        "model": spec.get("model", {}),
        "training": spec.get("training", {}),
        "evaluation": spec.get("evaluation", {}),
        "max_supervised_rows": spec.get("max_supervised_rows"),
        "unsupervised_subset_fraction": unsupervised_subset_fraction,
        "selection": {
            "mixing_ratio": mixing_ratio,
            "n_families": n_families,
            "family_order": family_order,
            "pretraining_seed": pretraining_seed,
            "total_forward_passes": total_forward_passes,
            "augmentation": augmentation,
            "unsupervised_subset_fraction": unsupervised_subset_fraction,
        },
    }


def _build_baseline_config(spec: dict, *, pretraining_seed: int, run_id: str) -> dict:
    return {
        "run_id": run_id,
        "tokenizer_path": spec["tokenizer_path"],
        "model": spec.get("model", {}),
        "selection": {"pretraining_seed": pretraining_seed},
        "evaluation": spec.get("evaluation", {}),
    }


def _build_objective_config(spec: dict, *, run_id: str, objectives: dict, pretraining_seed: int,
                            total_forward_passes: int, supervised_families=None,
                            init_encoder_path=None, augmentation: str = "canonical",
                            unsupervised_subset_fraction=None) -> dict:
    """General objective-based pretrain config (mlm/mtr/supervised + warm-start)."""
    return {
        "run_id": run_id,
        "tokenizer_path": spec["tokenizer_path"],
        "unsupervised_data_paths": spec.get("unsupervised_data_paths"),
        "unsupervised_raw_smiles_paths": spec.get("unsupervised_raw_smiles_paths"),
        "supervised_tokenized_parquet_path": spec.get("supervised_tokenized_parquet_path"),
        "descriptor_stats_path": spec.get("descriptor_stats_path"),
        "descriptor_precompute_dir": spec.get("descriptor_precompute_dir"),
        "eval_blocklist_path": spec.get("eval_blocklist_path"),
        "model": spec.get("model", {}),
        "training": spec.get("training", {}),
        "evaluation": spec.get("evaluation", {}),
        "unsupervised_subset_fraction": unsupervised_subset_fraction,
        "selection": {
            "objectives": objectives,
            "supervised_families": supervised_families,
            "init_encoder_path": init_encoder_path,
            "pretraining_seed": pretraining_seed,
            "total_forward_passes": total_forward_passes,
            "augmentation": augmentation,
            "unsupervised_subset_fraction": unsupervised_subset_fraction,
        },
    }


def _output_dir(spec: dict, run_id: str) -> str:
    return f"{spec['results_root']}/{run_id}"


def _backup_uri(spec: dict, run_id: str) -> str:
    return f"{spec['s3_backup_root']}/{run_id}"


def _emit(run_type, run_id, pretrain_config, spec, selection, requires_pretrain,
          eval_override: Optional[dict] = None) -> dict:
    entry = {
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
    if eval_override is not None:
        entry["eval_override"] = eval_override
    return entry


# ---------- run emitters ----------

def _smoke_runs(spec: dict) -> List[dict]:
    out = []
    fps = 1_000_000
    for aug in ("canonical", "enumerated"):
        rid = f"smoke_{aug}"
        sel = {"mixing_ratio": 1.0, "n_families": 0, "family_order": [],
               "pretraining_seed": 0, "total_forward_passes": fps, "augmentation": aug}
        cfg = _build_pretrain_config(
            spec, run_id=rid, mixing_ratio=1.0, n_families=0, family_order=[],
            pretraining_seed=0, total_forward_passes=fps, augmentation=aug)
        out.append(_emit("smoke", rid, cfg, spec, sel, requires_pretrain=True))
    return out


def _ecfp4_anchor_run(spec: dict) -> List[dict]:
    rid = "ecfp4_anchor"
    cfg = {"run_id": rid, "tokenizer_path": spec["tokenizer_path"],
           "evaluation": spec.get("evaluation", {})}
    sel = {"note": "classical baseline; no encoder"}
    return [_emit("ecfp4_anchor", rid, cfg, spec, sel, requires_pretrain=False,
                  eval_override={"featurizer": "ecfp4", "head": "xgb"})]


def _random_baseline_runs(spec: dict) -> List[dict]:
    out = []
    for rep in range(spec.get("random_baseline_replicates", 1)):
        rid = f"random_baseline_{rep:02d}"
        cfg = _build_baseline_config(spec, pretraining_seed=rep, run_id=rid)
        out.append(_emit("random_baseline", rid, cfg, spec, {"pretraining_seed": rep},
                         requires_pretrain=False))
    return out


def _core_comparison_runs(spec: dict) -> List[dict]:
    """B: unsup_only / sup_only / mixed at the fixed B_core budget, canonical. One
    entry per (cell × core_pretraining_seed). Exploratory sweep uses a single seed;
    the headline replication passes core_pretraining_seeds=[0,1,2]."""
    out = []
    budget = int(spec["b_core_forward_passes"])
    seeds = spec.get("core_pretraining_seeds", [0])
    fams = SUPERVISED_FAMILIES_V2
    cells = [
        ("unsup_only", 1.0, 0, []),
        ("sup_only", 0.0, len(fams), fams),
        ("mixed", 0.5, len(fams), fams),
    ]
    for seed in seeds:
        for run_type, mix, n_fam, order in cells:
            rid = f"{run_type}_seed{seed}"
            sel = {"mixing_ratio": mix, "n_families": n_fam, "family_order": order,
                   "pretraining_seed": seed, "total_forward_passes": budget,
                   "augmentation": "canonical"}
            cfg = _build_pretrain_config(
                spec, run_id=rid, mixing_ratio=mix, n_families=n_fam, family_order=order,
                pretraining_seed=seed, total_forward_passes=budget, augmentation="canonical")
            out.append(_emit(run_type, rid, cfg, spec, sel, requires_pretrain=True))
    return out


def _scaling_runs(spec: dict) -> List[dict]:
    """C: unsup_only at fixed B_core, varying the UNIQUE-molecule fraction (compute
    held constant), for canonical vs enumerated. Single seed."""
    out = []
    budget = int(spec["b_core_forward_passes"])
    fractions = spec.get("scaling_fractions", [0.001, 0.01, 0.1, 0.3, None])
    augs = spec.get("scaling_augmentations", ["canonical", "enumerated"])
    for aug in augs:
        for frac in fractions:
            tag = "full" if frac is None else f"{frac:g}".replace(".", "p")
            rid = f"scaling_{aug}_frac{tag}"
            sel = {"mixing_ratio": 1.0, "n_families": 0, "family_order": [],
                   "pretraining_seed": 0, "total_forward_passes": budget,
                   "augmentation": aug, "unsupervised_subset_fraction": frac}
            cfg = _build_pretrain_config(
                spec, run_id=rid, mixing_ratio=1.0, n_families=0, family_order=[],
                pretraining_seed=0, total_forward_passes=budget, augmentation=aug,
                unsupervised_subset_fraction=frac)
            out.append(_emit("scaling", rid, cfg, spec, sel, requires_pretrain=True))
    return out


def _ablation_runs(spec: dict) -> List[dict]:
    """Dense-vs-sparse supervised ablation. An MLM base is trained once (or reused via
    base_encoder_path); every arm is unsup→sup[X] warm-started from that base."""
    from config_v2 import SUPERVISED_GROUPS
    out: List[dict] = []
    seed = int(spec.get("pretraining_seed", 0))
    base_budget = int(spec["base_forward_passes"])
    sup_budget = int(spec["sup_forward_passes"])
    aug = spec.get("augmentation", "canonical")

    # anchors
    if spec.get("include_ecfp_anchor", True):
        out.extend(_ecfp4_anchor_run(spec))
    for rep in range(spec.get("random_baseline_replicates", 1)):
        rid = f"random_baseline_{rep:02d}"
        out.append(_emit("random_baseline", rid,
                         _build_baseline_config(spec, pretraining_seed=rep, run_id=rid),
                         spec, {"pretraining_seed": rep}, requires_pretrain=False))

    # A0: MLM base (the unsup_only reference; reused by every arm)
    base_path = spec.get("base_encoder_path")
    if not base_path:
        base_id = spec.get("base_run_id", "base_unsup")
        cfg = _build_objective_config(spec, run_id=base_id, objectives={"mlm": 1.0},
                                      pretraining_seed=seed, total_forward_passes=base_budget,
                                      augmentation=aug)
        out.append(_emit("unsup_only", base_id, cfg, spec, cfg["selection"], requires_pretrain=True))
        base_path = f"{_output_dir(spec, base_id)}/encoder"

    # arms (unsup→sup[X]); each warm-starts from the MLM base
    G = SUPERVISED_GROUPS
    arms = [
        ("mtr", {"mtr": 1.0}, None),                                   # A1 dense-only
        ("pcba", {"supervised": 1.0}, G["pcba"]),                      # A2 sparse single
        ("l1000", {"supervised": 1.0}, G["l1000"]),                   # A3 sparse single
        ("sparse_all", {"supervised": 1.0}, G["sparse_all"]),         # A4 sparse-only combined
        ("dense_plus_sparse", {"mtr": 0.5, "supervised": 0.5}, G["sparse_all"]),  # A5 dense+sparse
        ("pcqm", {"supervised": 1.0}, G["pcqm"]),                     # A6 negative control
    ]
    for name, objs, fams in arms:
        rid = f"seq_{name}"
        cfg = _build_objective_config(spec, run_id=rid, objectives=objs, pretraining_seed=seed,
                                      total_forward_passes=sup_budget, supervised_families=fams,
                                      init_encoder_path=base_path, augmentation=aug)
        out.append(_emit("ablation", rid, cfg, spec, cfg["selection"], requires_pretrain=True))
    return out


def _fp_tag(b: int) -> str:
    b = int(b)
    if b >= 1_000_000_000:
        return f"{b // 1_000_000_000}B"
    return f"{b // 1_000_000}M"


def _compute_scaling_runs(spec: dict) -> List[dict]:
    """Workstream B: does more COMPUTE (not data) fix the 'saturation'? MLM unsup_only
    at increasing forward-pass budgets over full data, plus an optional data sweep at a
    fixed budget (the data×compute 2D check)."""
    out: List[dict] = []
    seed = int(spec.get("pretraining_seed", 0))
    if spec.get("include_ecfp_anchor", True):
        out.extend(_ecfp4_anchor_run(spec))
    for rep in range(spec.get("random_baseline_replicates", 1)):
        rid = f"random_baseline_{rep:02d}"
        out.append(_emit("random_baseline", rid,
                         _build_baseline_config(spec, pretraining_seed=rep, run_id=rid),
                         spec, {"pretraining_seed": rep}, requires_pretrain=False))
    # compute axis (full data, growing budget)
    for b in spec["compute_budgets"]:
        rid = f"cscale_{_fp_tag(b)}"
        cfg = _build_objective_config(spec, run_id=rid, objectives={"mlm": 1.0},
                                      pretraining_seed=seed, total_forward_passes=int(b))
        out.append(_emit("compute_scaling", rid, cfg, spec, cfg["selection"], requires_pretrain=True))
    # optional data×compute 2D (vary unique molecules at a fixed larger budget)
    if spec.get("data_sweep_budget"):
        b = int(spec["data_sweep_budget"])
        for frac in spec.get("data_sweep_fractions", [0.001, 0.01, 0.1]):
            rid = f"dsweep_frac{('%g' % frac).replace('.', 'p')}"
            cfg = _build_objective_config(spec, run_id=rid, objectives={"mlm": 1.0},
                                          pretraining_seed=seed, total_forward_passes=b,
                                          unsupervised_subset_fraction=frac)
            out.append(_emit("data_sweep", rid, cfg, spec, cfg["selection"], requires_pretrain=True))
    return out


def _sft_type_defs(spec: dict) -> List[dict]:
    """Resolve the SFT 'W' recipes (objectives + supervised family group) from spec."""
    from config_v2 import SUPERVISED_GROUPS
    out = []
    for t in spec["sft_types"]:
        grp = t.get("group")
        fams = None if grp in (None, "null") else SUPERVISED_GROUPS[grp]
        out.append({"name": t["name"], "objectives": t["objectives"], "families": fams})
    return out


def _phase2_runs(spec: dict) -> List[dict]:
    """Phase-2 scaling matrix answering 'does unsup even help?' across SFT types.

    Three blocks, all tagged with a `stage` so the launcher/splitter can order them:
      - stage=ladder : shared pure-unsup MLM ladder (from scratch, growing budget).
      - stage=skip   : skip-unsup[W] = random base + SFT[W], per-W budget ladder
                       (the SFT-from-scratch catch-up test). No dependency.
      - stage=u2s    : unsup->sup[W] = warm-start each ladder checkpoint on W at a fixed
                       small SFT budget. Depends on the matching ladder run's encoder.
    """
    out: List[dict] = []
    seed = int(spec.get("pretraining_seed", 0))
    aug = spec.get("augmentation", "canonical")
    types = _sft_type_defs(spec)

    # anchors (reference floor + classical baseline)
    if spec.get("include_ecfp_anchor", True):
        out.extend(_ecfp4_anchor_run(spec))
    for rep in range(spec.get("random_baseline_replicates", 1)):
        rid = f"random_baseline_{rep:02d}"
        out.append(_emit("random_baseline", rid,
                         _build_baseline_config(spec, pretraining_seed=rep, run_id=rid),
                         spec, {"pretraining_seed": rep}, requires_pretrain=False))

    # stage=ladder : shared pure-unsup MLM ladder
    ladder = [int(b) for b in spec["unsup_ladder"]]
    base_path_for = {}
    for b in ladder:
        rid = f"unsup_{_fp_tag(b)}"
        cfg = _build_objective_config(spec, run_id=rid, objectives={"mlm": 1.0},
                                      pretraining_seed=seed, total_forward_passes=b,
                                      augmentation=aug)
        e = _emit("unsup_ladder", rid, cfg, spec, cfg["selection"], requires_pretrain=True)
        e["stage"] = "ladder"
        out.append(e)
        base_path_for[b] = f"{_output_dir(spec, rid)}/encoder"

    # stage=u2s : warm-start each ladder checkpoint on each SFT type
    sup_budget = int(spec["sup_finetune_budget"])
    for W in types:
        for b in ladder:
            rid = f"u2s_{W['name']}_from{_fp_tag(b)}"
            cfg = _build_objective_config(
                spec, run_id=rid, objectives=W["objectives"], pretraining_seed=seed,
                total_forward_passes=sup_budget, supervised_families=W["families"],
                init_encoder_path=base_path_for[b], augmentation=aug)
            e = _emit("unsup_to_sup", rid, cfg, spec, cfg["selection"], requires_pretrain=True)
            e["stage"] = "u2s"
            e["depends_on"] = f"unsup_{_fp_tag(b)}"
            out.append(e)

    # stage=skip : skip-unsup[W] = random base + SFT[W], per-W budget ladder
    skip_ladders = spec["skip_ladders"]
    for W in types:
        for b in [int(x) for x in skip_ladders[W["name"]]]:
            rid = f"skip_{W['name']}_{_fp_tag(b)}"
            cfg = _build_objective_config(
                spec, run_id=rid, objectives=W["objectives"], pretraining_seed=seed,
                total_forward_passes=b, supervised_families=W["families"],
                init_encoder_path=None, augmentation=aug)
            e = _emit("skip_unsup", rid, cfg, spec, cfg["selection"], requires_pretrain=True)
            e["stage"] = "skip"
            out.append(e)

    return out


def generate_manifest(spec: dict) -> dict:
    if spec.get("wave") in ("ablation", "compute_scaling", "phase2_scaling"):
        runs = ({"ablation": _ablation_runs, "compute_scaling": _compute_scaling_runs,
                 "phase2_scaling": _phase2_runs}[spec["wave"]])(spec)
        return {
            "name": spec.get("name", "climb_v2_" + spec["wave"]),
            "results_root": spec["results_root"],
            "s3_backup_root": spec["s3_backup_root"],
            "tokenizer_path": spec["tokenizer_path"],
            "runs": runs,
        }
    runs: List[dict] = []
    if spec.get("include_smoke", True):
        runs.extend(_smoke_runs(spec))
    if spec.get("include_ecfp_anchor", True):
        runs.extend(_ecfp4_anchor_run(spec))
    runs.extend(_random_baseline_runs(spec))
    runs.extend(_core_comparison_runs(spec))
    if spec.get("include_scaling", True):
        runs.extend(_scaling_runs(spec))
    return {
        "name": spec.get("name", "climb_v2"),
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
    by_type = {}
    for r in manifest["runs"]:
        by_type[r["run_type"]] = by_type.get(r["run_type"], 0) + 1
    print(f"Wrote {len(manifest['runs'])} runs to {out_path}: {by_type}")


if __name__ == "__main__":
    main()
