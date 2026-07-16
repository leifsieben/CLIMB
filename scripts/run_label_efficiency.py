"""Exp D — label-efficiency curves.

Reuses the exploratory-wave encoders (no new pretraining). For each featurizer and
each downstream train-set size, trains the frozen-featurizer head and records the
per-task metric. Produces the "money plot": performance vs #labeled molecules, one
line per featurizer — showing whether unsupervised pretraining buys label efficiency
(separates at small N) or a higher ceiling (separates at large N) or nothing.

Featurizers (encoders read from an existing results_root):
    unsup   -> <root>/unsup_only_seed0/encoder
    sup     -> <root>/sup_only_seed0/encoder
    random  -> <root>/random_baseline_00/encoder
    ecfp4   -> classical Morgan + XGBoost anchor (no encoder)

Usage:
    python scripts/run_label_efficiency.py \
        --exploratory_root experiments/climb_v2 \
        --output_dir experiments/climb_v2_labeleff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_v2 import MOLECULENET_TASKS_V2
from eval_v2 import evaluate

SUBSAMPLES = [100, 300, 1000, 3000, None]  # None = full train split
COLORS = {"unsup": "#1f77b4", "sup": "#ff7f0e", "random": "#999999", "ecfp4": "#000000"}


def _featurizers(root: Path):
    """(name, featurizer, encoder_path, tokenizer_path, head)."""
    return [
        ("unsup", "encoder", root / "unsup_only_seed0/encoder", root / "unsup_only_seed0/tokenizer", "mlp"),
        ("sup", "encoder", root / "sup_only_seed0/encoder", root / "sup_only_seed0/tokenizer", "mlp"),
        ("random", "encoder", root / "random_baseline_00/encoder", root / "random_baseline_00/tokenizer", "mlp"),
        ("ecfp4", "ecfp4", None, None, "xgb"),
    ]


def run_sweep(exploratory_root: Path, output_dir: Path, head_seeds):
    for name, feat, enc, tok, head in _featurizers(exploratory_root):
        if feat == "encoder" and not Path(enc).exists():
            print(f"[label_eff] SKIP {name}: encoder not found at {enc}")
            continue
        for n in SUBSAMPLES:
            tag = "full" if n is None else str(n)
            out = output_dir / f"{name}_n{tag}" / "moleculenet"
            print(f"\n[label_eff] === {name} n={tag} ===")
            evaluate(
                encoder_path=str(enc) if enc else None,
                tokenizer_path=str(tok) if tok else None,
                output_dir=str(out), head_seeds=head_seeds,
                datasets=MOLECULENET_TASKS_V2, featurizer=feat, head=head,
                train_subsample=n, subsample_seed=0,
            )


def collect_and_plot(output_dir: Path):
    rows = []
    for summ in sorted(output_dir.glob("*/moleculenet/moleculenet_summary.csv")):
        name = summ.parent.parent.name.rsplit("_n", 1)[0]
        df = pd.read_csv(summ)
        for _, r in df[df.head_seed == "MEAN"].iterrows():
            rows.append({"featurizer": name, "dataset": r["dataset"],
                         "main_metric": r["main_metric"], "n_train": int(r["n_train"]),
                         "mean": r["main_value"]})
    if not rows:
        print("[label_eff] no results to plot")
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "label_efficiency.csv", index=False)

    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds]
        metric = d["main_metric"].iloc[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in ["ecfp4", "random", "sup", "unsup"]:
            sub = d[d.featurizer == name].sort_values("n_train")
            if len(sub) > 1:
                ax.plot(sub["n_train"], sub["mean"], marker="o",
                        label=name, color=COLORS.get(name))
        ax.set_xscale("log")
        ax.set_xlabel("# labeled training molecules")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{ds} — label efficiency ({metric}, {'lower' if metric=='rmse' else 'higher'} better)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"fig_labeleff_{ds}.png", dpi=120)
        plt.close(fig)
    print(f"[label_eff] wrote {output_dir}/label_efficiency.csv + {df.dataset.nunique()} figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exploratory_root", default="experiments/climb_v2")
    p.add_argument("--output_dir", default="experiments/climb_v2_labeleff")
    p.add_argument("--head_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--plot_only", action="store_true")
    args = p.parse_args()

    root = Path(args.exploratory_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run_sweep(root, out, args.head_seeds)
    collect_and_plot(out)


if __name__ == "__main__":
    main()
