"""Encoding throughput / latency benchmark: classical fingerprints vs the CLIMB encoder.

The paper argues about *compute cost* at virtual-screening scale, so we need a measured
number, not a hand-wave. This times the three featurizers we actually ship, on the SAME
1000 molecules, with the SAME settings production uses:

  ecfp4    featurize_v2.ecfp4_features      -> Morgan r=2, 2048 bits (the ECFP4 anchor)
  fp_desc  ecfp4 ++ descriptors_v2          -> 2048 bits ++ 217 RDKit descriptors
  encoder  eval_v2._encoder_features        -> ModernBERT (~41M) tokenize + forward + mean pool

Plus `rdkit_desc` on its own as a diagnostic, because the interesting result is how the
fp_desc cost splits between the fingerprint and the descriptors.

Protocol: warm up once (RDKit lazy imports, torch kernel autotune / MPS shader compile),
then time --repeats runs and report mean +/- sd. RDKit has no GPU path, so its GPU cells
are "n/a (CPU-only)"; it does parallelize embarrassingly, so we also time a process pool
over all cores (pool startup is outside the timed region -- at screening scale it amortizes
to zero). For the encoder we report tokenization separately, since tokenization is an
unavoidable CPU cost even when the forward pass is on a GPU.

Usage:
  # local: CPU + Apple MPS
  python scripts/bench_featurization.py --devices cpu,mps
  # on a CUDA box: GPU only, merged into the same JSON afterwards
  python scripts/bench_featurization.py --devices cuda --skip-rdkit --out /tmp/gpu.json
  python scripts/bench_featurization.py --merge-from /tmp/gpu.json --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------- molecules ----------

def load_smiles(source: str, n: int, seed: int) -> List[str]:
    """Deterministic sample of n unique SMILES from a source already in the repo."""
    if source == "zinc":
        import pandas as pd
        df = pd.read_parquet(REPO_ROOT / "raw_data" / "unsupervised_12M_ZINC.parquet")
        pool = sorted(set(df["SMILES"].astype(str).tolist()))
    elif source == "moleculenet":
        import csv
        path = (REPO_ROOT / "figure_data" / "climb_v2_phase2" / "ecfp4_anchor"
                / "moleculenet" / "test_predictions.csv")
        with open(path) as f:
            pool = sorted({row["raw_smiles"] for row in csv.DictReader(f) if row["raw_smiles"]})
    else:
        raise ValueError(f"Unknown --source {source!r} (expected zinc|moleculenet)")
    if len(pool) < n:
        raise RuntimeError(f"source {source} only has {len(pool)} unique SMILES, need {n}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx)]


def molecule_stats(smiles: List[str]) -> Dict:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    heavy, chars, rings = [], [], []
    n_bad = 0
    for s in smiles:
        chars.append(len(s))
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_bad += 1
            continue
        heavy.append(m.GetNumHeavyAtoms())
        rings.append(m.GetRingInfo().NumRings())

    def q(v):
        a = np.asarray(v, dtype=float)
        return {"mean": float(a.mean()), "sd": float(a.std()), "min": float(a.min()),
                "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                "max": float(a.max())}

    return {"n": len(smiles), "n_unparseable": n_bad, "heavy_atoms": q(heavy),
            "smiles_chars": q(chars), "num_rings": q(rings)}


# ---------- timing helpers ----------

def time_repeats(fn, repeats: int, warmup: int = 1) -> List[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def summarize(method: str, device: str, precision: str, notes: str,
              times: List[float], n_mol: int, extra: Dict = None) -> Dict:
    mean = statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    per_mol_s = mean / n_mol
    rec = {
        "method": method, "device": device, "precision": precision, "notes": notes,
        "n_molecules": n_mol, "repeats": len(times),
        "times_s": [round(t, 6) for t in times],
        "total_s_mean": round(mean, 6), "total_s_sd": round(sd, 6),
        "ms_per_mol": round(per_mol_s * 1e3, 6),
        "mol_per_s": round(1.0 / per_mol_s, 3),
        "hours_1M": round(per_mol_s * 1e6 / 3600.0, 6),
        "hours_1B": round(per_mol_s * 1e9 / 3600.0, 3),
    }
    if extra:
        rec.update(extra)
    return rec


def result_key(rec: Dict) -> str:
    return f"{rec['method']}|{rec['device']}|{rec['precision']}|{rec['notes']}"


# ---------- RDKit featurizers (exactly what eval_v2._featurize does) ----------

def feat_ecfp4(smiles: List[str]) -> np.ndarray:
    from featurize_v2 import ecfp4_features
    return np.asarray(ecfp4_features(smiles))


def feat_rdkit_desc(smiles: List[str]) -> np.ndarray:
    from descriptors_v2 import rdkit_descriptors
    x = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    return x


def feat_fp_desc(smiles: List[str]) -> np.ndarray:
    from featurize_v2 import ecfp4_features
    from descriptors_v2 import rdkit_descriptors
    fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
    d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
    d[~np.isfinite(d)] = np.nan
    return np.concatenate([fp, d], axis=1)


_RDKIT_FEATURIZERS = {"ecfp4": feat_ecfp4, "rdkit_desc": feat_rdkit_desc, "fp_desc": feat_fp_desc}


def _pool_worker(args):
    """Top-level so it pickles under macOS spawn."""
    name, chunk = args
    return _RDKIT_FEATURIZERS[name](chunk)


def bench_rdkit_parallel(name: str, smiles: List[str], workers: int, repeats: int) -> List[float]:
    """Chunk across a process pool. Pool creation and the warmup map happen outside the
    timed region; at screening scale the fixed startup amortizes away."""
    import multiprocessing as mp
    chunks = [list(c) for c in np.array_split(np.asarray(smiles, dtype=object), workers)]
    payload = [(name, c) for c in chunks]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        pool.map(_pool_worker, payload)  # warmup: pay import + fork cost once
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = pool.map(_pool_worker, payload)
            np.concatenate(out, axis=0)
            times.append(time.perf_counter() - t0)
    return times


# ---------- encoder ----------

def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_encoder(encoder_path: str, tokenizer_path: str, device_str: str, dtype_str: str):
    import torch
    from transformers import ModernBertModel, PreTrainedTokenizerFast
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    device = torch.device(device_str)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    # reference_compile=False mirrors eval_v2: no torch.compile/triton dependency.
    model = ModernBertModel.from_pretrained(
        encoder_path, attn_implementation="sdpa", reference_compile=False
    ).to(device=device, dtype=dtypes[dtype_str])
    model.eval()
    return model, tokenizer, device


def bench_encoder(model, tokenizer, device, smiles: List[str], batch_size: int,
                  max_length: int, padding: str, repeats: int) -> Dict:
    """Tokenize + forward + masked-mean pool, same as eval_v2._encoder_features.
    Returns the per-run wall times and the CPU-side tokenization time within them."""
    import torch
    from featurize_v2 import pool as pool_fn

    tok_times: List[float] = []

    def one_pass():
        tok_s = 0.0
        feats = []
        with torch.no_grad():
            for i in range(0, len(smiles), batch_size):
                chunk = smiles[i:i + batch_size]
                t_tok = time.perf_counter()
                enc = tokenizer(chunk, truncation=True, max_length=max_length,
                                padding=padding, return_tensors="pt")
                tok_s += time.perf_counter() - t_tok
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                out = model(input_ids=ids, attention_mask=mask)
                pooled = pool_fn(out.last_hidden_state, mask, "mean").float().cpu().numpy()
                feats.append(pooled)
        _sync(device)
        tok_times.append(tok_s)
        return np.concatenate(feats, axis=0)

    times = time_repeats(one_pass, repeats, warmup=1)
    # drop the warmup's tokenization entry so the lists line up
    tok = tok_times[-repeats:]
    return {"times": times, "tokenize_s_mean": statistics.mean(tok)}


def token_stats(tokenizer, smiles: List[str], max_length: int) -> Dict:
    lens = [len(tokenizer(s, truncation=True, max_length=max_length)["input_ids"]) for s in smiles]
    a = np.asarray(lens, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "n_truncated": int((a >= max_length).sum())}


# ---------- hardware ----------

def hardware_info() -> Dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "cpu_count_logical": os.cpu_count()}
    try:
        if platform.system() == "Darwin":
            info["cpu"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            for key, sysctl in [("cpu_cores_physical", "hw.physicalcpu"),
                                ("cpu_cores_perf", "hw.perflevel0.logicalcpu"),
                                ("cpu_cores_eff", "hw.perflevel1.logicalcpu"),
                                ("mem_bytes", "hw.memsize")]:
                try:
                    info[key] = int(subprocess.check_output(["sysctl", "-n", sysctl]).decode().strip())
                except Exception:
                    pass
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu"] = line.split(":", 1)[1].strip()
                        break
    except Exception as exc:
        info["cpu"] = f"unknown ({exc})"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    for mod in ("rdkit", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        info["ec2_instance_type"] = subprocess.check_output(
            ["curl", "-s", "--max-time", "1", "http://169.254.169.254/latest/meta-data/instance-type"]
        ).decode().strip() or None
    except Exception:
        pass
    return info


# ---------- table ----------

def print_table(results: List[Dict]) -> None:
    cols = [("method", 12), ("device", 10), ("precision", 10), ("notes", 34),
            ("total_s", 16), ("ms/mol", 10), ("mol/s", 12), ("h/1M", 9), ("h/1B", 11)]
    header = "  ".join(f"{c:<{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        total = f"{r['total_s_mean']:.4f} +/- {r['total_s_sd']:.4f}"
        row = [r["method"], r["device"], r["precision"], r["notes"][:34], total,
               f"{r['ms_per_mol']:.4f}", f"{r['mol_per_s']:.1f}",
               f"{r['hours_1M']:.3f}", f"{r['hours_1B']:.1f}"]
        print("  ".join(f"{v:<{w}}" for v, (_, w) in zip(row, cols)))


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_molecules", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=3)
    # moleculenet = the raw_smiles the paper's own eval actually scores, and a much wider
    # size distribution (4-118 heavy atoms) than our ZINC tranches (which are ~all 23).
    p.add_argument("--source", choices=["moleculenet", "zinc"], default="moleculenet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--devices", default="cpu",
                   help="comma-separated torch devices for the encoder: cpu,mps,cuda")
    p.add_argument("--gpu_precisions", default="bf16,fp16,fp32",
                   help="precisions to try on cuda/mps (cpu is always fp32)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--paddings", default="longest,max_length",
                   help="tokenizer padding modes to benchmark (eval_v2 uses 'longest')")
    p.add_argument("--encoder_path", default=str(REPO_ROOT / "figure_data" / "climb_v2_phase2"
                                                 / "unsup_8M" / "encoder"))
    p.add_argument("--tokenizer_path", default=str(REPO_ROOT / "figure_data" / "_tokenizer"))
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="processes for the parallel RDKit run (0 = skip)")
    p.add_argument("--skip_rdkit", action="store_true")
    p.add_argument("--skip_encoder", action="store_true")
    p.add_argument("--hardware_label", default=None,
                   help="short label for this machine, e.g. 'M4 Pro' or 'g5.2xlarge A10G'")
    p.add_argument("--out", default=str(REPO_ROOT / "figure_data" / "_bench" / "featurization_timing.json"))
    p.add_argument("--merge_from", default=None,
                   help="merge results from another run's JSON into --out")
    p.add_argument("--merge_only", action="store_true",
                   help="only merge --merge_from into --out; run no benchmarks")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        if not args.merge_from:
            p.error("--merge_only requires --merge_from")
        base = json.loads(out_path.read_text()) if out_path.exists() else {"results": []}
        incoming = json.loads(Path(args.merge_from).read_text())
        merged = {result_key(r): r for r in base.get("results", [])}
        for r in incoming.get("results", []):
            merged[result_key(r)] = r
        base["results"] = list(merged.values())
        base.setdefault("hardware", {})
        base["hardware"][incoming.get("hardware_label", "merged")] = incoming.get("hardware", {})
        out_path.write_text(json.dumps(base, indent=2))
        print(f"merged {len(incoming.get('results', []))} results into {out_path}")
        print_table(base["results"])
        return

    hw = hardware_info()
    label = args.hardware_label or hw.get("cpu", "unknown")
    print(f"[bench] hardware: {label}")
    print(f"[bench] {json.dumps(hw)}")

    smiles = load_smiles(args.source, args.n_molecules, args.seed)
    stats = molecule_stats(smiles)
    print(f"[bench] {len(smiles)} molecules from {args.source}: "
          f"heavy atoms mean {stats['heavy_atoms']['mean']:.1f} "
          f"(p50 {stats['heavy_atoms']['p50']:.0f}, p95 {stats['heavy_atoms']['p95']:.0f}), "
          f"SMILES chars mean {stats['smiles_chars']['mean']:.1f}")

    results: List[Dict] = []

    # ---- RDKit, single core ----
    if not args.skip_rdkit:
        for name in ("ecfp4", "fp_desc", "rdkit_desc"):
            fn = _RDKIT_FEATURIZERS[name]
            print(f"[bench] {name} single-core ...")
            times = time_repeats(lambda: fn(smiles), args.repeats)
            results.append(summarize(name, "cpu", "n/a", "single core", times, len(smiles)))
        if args.workers and args.workers > 1:
            for name in ("ecfp4", "fp_desc", "rdkit_desc"):
                print(f"[bench] {name} {args.workers} processes ...")
                times = bench_rdkit_parallel(name, smiles, args.workers, args.repeats)
                results.append(summarize(name, "cpu", "n/a", f"{args.workers} processes",
                                         times, len(smiles)))

    # ---- encoder ----
    tok_info = None
    if not args.skip_encoder:
        import torch
        for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
            if dev == "cuda" and not torch.cuda.is_available():
                print("[bench] cuda requested but not available - skipping")
                continue
            if dev == "mps" and not (getattr(torch.backends, "mps", None)
                                     and torch.backends.mps.is_available()):
                print("[bench] mps requested but not available - skipping")
                continue
            precisions = ["fp32"] if dev == "cpu" else [
                x.strip() for x in args.gpu_precisions.split(",") if x.strip()]
            for prec in precisions:
                try:
                    model, tokenizer, device = load_encoder(
                        args.encoder_path, args.tokenizer_path, dev, prec)
                except Exception as exc:
                    print(f"[bench] encoder {dev}/{prec} failed to load: {exc}")
                    continue
                if tok_info is None:
                    tok_info = token_stats(tokenizer, smiles, args.max_length)
                    n_params = sum(q.numel() for q in model.parameters())
                    print(f"[bench] encoder params {n_params/1e6:.1f}M; tokens/mol mean "
                          f"{tok_info['mean']:.1f} (p95 {tok_info['p95']:.0f}, max {tok_info['max']:.0f})")
                for padding in [x.strip() for x in args.paddings.split(",") if x.strip()]:
                    print(f"[bench] encoder {dev}/{prec} bs={args.batch_size} pad={padding} ...")
                    try:
                        got = bench_encoder(model, tokenizer, device, smiles, args.batch_size,
                                            args.max_length, padding, args.repeats)
                    except Exception as exc:
                        print(f"[bench]   failed: {exc}")
                        continue
                    pad_note = "dynamic pad" if padding == "longest" else f"pad to {args.max_length}"
                    tok_mean = got["tokenize_s_mean"]
                    results.append(summarize(
                        "encoder", dev, prec, f"bs{args.batch_size}, {pad_note}",
                        got["times"], len(smiles),
                        extra={"tokenize_s_mean": round(tok_mean, 6),
                               "tokenize_ms_per_mol": round(tok_mean / len(smiles) * 1e3, 6),
                               "tokenize_frac": round(tok_mean / statistics.mean(got["times"]), 4),
                               "batch_size": args.batch_size, "padding": padding,
                               "max_length": args.max_length}))
                del model
                if dev == "cuda":
                    torch.cuda.empty_cache()

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware_label": label,
        "hardware": hw,
        "config": {"n_molecules": args.n_molecules, "repeats": args.repeats,
                   "source": args.source, "seed": args.seed,
                   "batch_size": args.batch_size, "max_length": args.max_length,
                   "encoder_path": args.encoder_path, "workers": args.workers,
                   "ecfp4": "Morgan radius=2, 2048 bits (featurize_v2.ecfp4_features)",
                   "rdkit_desc": "Descriptors.descList (descriptors_v2.rdkit_descriptors)"},
        "molecule_stats": stats,
        "token_stats": tok_info,
        "results": results,
    }

    if args.merge_from:
        incoming = json.loads(Path(args.merge_from).read_text())
        merged = {result_key(r): r for r in results}
        for r in incoming.get("results", []):
            merged[result_key(r)] = r
        payload["results"] = list(merged.values())
        payload["hardware"] = {label: hw,
                               incoming.get("hardware_label", "merged"): incoming.get("hardware", {})}

    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print_table(payload["results"])
    print()
    print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
