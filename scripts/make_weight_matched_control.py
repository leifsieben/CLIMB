"""Build WEIGHT-MATCHED controls for a pretrained encoder: same weight statistics, no learned structure.

WHY. fig_E's unsupervised ladder shows a zero-chemistry corpus (English Wikipedia) recovering most
of real chemistry's benefit on some panels. The standard reviewer objection is that this is not
about corpus statistics at all: any gradient steps grow weight norms and re-calibrate LayerNorm,
so the arm may simply have a better-conditioned optimization endpoint than a fresh random init.

Two contrasts already argue against that, from data we have:
  * unigram-resampled pretraining travels 85% of ||W0|| and buys ~0 lift; and
  * wiki travels FURTHER from init than real chemistry (1.461 vs 1.331 of ||W0||, in every seed)
    and ends with LARGER weight norms, yet lifts less on all six panels.
Weight scale and distance therefore do not order the arms. But those are crude summaries, and the
objection can retreat to something subtler than norms. This script makes the test causal instead of
correlational: take the trained encoder and destroy what it learned while keeping its weight
statistics, then run the same probe.

TWO VARIANTS, deliberately both:
  gaussian  each 2-D tensor redrawn from N(mean, std) of that same tensor. Matches per-tensor first
            and second moments -- the plain reading of "matched weight statistics".
  permuted  each 2-D tensor's entries randomly permuted. Every scalar is preserved EXACTLY, so the
            full per-tensor value distribution, its norm and all higher moments are identical by
            construction. Immune to "your matching was approximate".

1-D tensors (LayerNorm gains and biases, all biases) are COPIED VERBATIM in both variants. That is
the point: the LayerNorm calibration named in the objection is handed to the control for free.

Interpretation is fixed in advance, so this cannot be read after the fact:
  control lands at the random-init floor  -> the lift lives in learned structure; the corpus claim stands
  control recovers the arm's lift         -> conditioning explains it, and THAT is the finding

Usage:
  python3 scripts/make_weight_matched_control.py --src <dir with model.safetensors+config.json> \
      --out <dir> --variant gaussian|permuted --seed N
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def build(src: Path, out: Path, variant: str, seed: int) -> dict:
    t = load_file(str(src / "model.safetensors"))
    rng = np.random.default_rng(seed)
    new, touched, copied = {}, 0, 0
    for k, v in t.items():
        if v.ndim == 2:
            a = v.astype(np.float64)
            if variant == "gaussian":
                w = rng.normal(a.mean(), a.std(), size=a.shape)
            elif variant == "permuted":
                flat = a.ravel().copy()
                rng.shuffle(flat)                      # exact same multiset of values
                w = flat.reshape(a.shape)
            else:
                raise SystemExit(f"unknown variant {variant!r}")
            new[k] = w.astype(v.dtype)
            touched += 1
        else:
            new[k] = v                                  # LayerNorm gains/biases and all biases: verbatim
            copied += 1
    out.mkdir(parents=True, exist_ok=True)
    save_file(new, str(out / "model.safetensors"))
    shutil.copy(src / "config.json", out / "config.json")

    # Prove the match rather than asserting it: report the statistics the control is supposed to
    # preserve, per tensor, against the source. A silent mismatch here would invalidate the whole
    # experiment and nothing downstream would notice.
    worst_rms, worst_name = 0.0, ""
    for k, v in t.items():
        if v.ndim != 2:
            continue
        a, b = v.astype(np.float64), new[k].astype(np.float64)
        ra, rb = np.sqrt((a**2).mean()), np.sqrt((b**2).mean())
        d = abs(rb - ra) / ra
        if d > worst_rms:
            worst_rms, worst_name = d, k
    allsrc = np.concatenate([v.ravel().astype(np.float64) for v in t.values() if v.ndim == 2])
    allnew = np.concatenate([v.ravel().astype(np.float64) for v in new.values() if v.ndim == 2])
    meta = dict(
        src=str(src), variant=variant, seed=seed,
        tensors_randomised=touched, tensors_copied_verbatim=copied,
        src_global_rms=float(np.sqrt((allsrc**2).mean())),
        ctl_global_rms=float(np.sqrt((allnew**2).mean())),
        worst_per_tensor_rms_reldiff=float(worst_rms), worst_tensor=worst_name,
        src_sha1=hashlib.sha1((src / "model.safetensors").read_bytes()).hexdigest()[:12],
    )
    (out / "control_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=["gaussian", "permuted"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    m = build(a.src, a.out, a.variant, a.seed)
    print(f"[ctl] {a.out.name}: {m['tensors_randomised']} tensors randomised, "
          f"{m['tensors_copied_verbatim']} copied verbatim; "
          f"global RMS {m['src_global_rms']:.6f} -> {m['ctl_global_rms']:.6f}; "
          f"worst per-tensor RMS drift {m['worst_per_tensor_rms_reldiff']:.2e} ({m['worst_tensor']})")
