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
                rng.shuffle(flat)
                w = flat.reshape(a.shape)
            else:
                raise SystemExit(f"unknown variant {variant!r}")
            new[k] = w.astype(v.dtype)
            touched += 1
        else:
            new[k] = v
            copied += 1
    out.mkdir(parents=True, exist_ok=True)
    save_file(new, str(out / "model.safetensors"))
    shutil.copy(src / "config.json", out / "config.json")


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
