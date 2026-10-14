from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np


def _log(m: str) -> None:
    print(f"[synth {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _shard_names(n: int) -> list[str]:
    return [f"shard_{i:05d}.pkl" for i in range(n)]


def _load_shard(path: Path) -> list:
    obj = pickle.load(open(path, "rb"))
    return obj["data"] if isinstance(obj, dict) else obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unigram", "bigram"], required=True)
    ap.add_argument("--n_shards", type=int, default=40)
    ap.add_argument("--local_src", required=True)
    ap.add_argument("--local_out", required=True)
    ap.add_argument("--gen-seed", type=int, default=12345)
    ap.add_argument("--vocab-size", type=int, default=1000)
    a = ap.parse_args()

    local_src = Path(a.local_src); local_src.mkdir(parents=True, exist_ok=True)
    local_out = Path(a.local_out); local_out.mkdir(parents=True, exist_ok=True)
    names = _shard_names(a.n_shards)
    V = a.vocab_size


    _log(f"pass 1: fitting {a.mode} statistics over {a.n_shards} shards")
    uni = np.zeros(V, dtype=np.int64)
    big = np.zeros((V, V), dtype=np.int64) if a.mode == "bigram" else None
    first = np.zeros(V, dtype=np.int64)
    n_seq = 0
    n_tok = 0
    t0 = time.time()
    for name in names:
        data = _load_shard(local_src / name)
        for d in data:
            ids = d["input_ids"]
            n_seq += 1
            n_tok += len(ids)
            first[ids[0]] += 1
            arr = np.asarray(ids, dtype=np.int64)
            np.add.at(uni, arr, 1)
            if big is not None and len(arr) > 1:
                np.add.at(big, (arr[:-1], arr[1:]), 1)
        _log(f"  fit {name}: seqs={n_seq:,} toks={n_tok:,} ({time.time()-t0:.0f}s)")

    uni_p = uni / uni.sum()
    support = np.where(uni > 0)[0]
    H_uni = float(-(uni_p[support] * np.log(uni_p[support])).sum())
    _log(f"corpus: {n_seq:,} sequences, {n_tok:,} content tokens, |support|={len(support)}")
    _log(f"unigram entropy H={H_uni:.4f} nats ({H_uni/np.log(2):.4f} bits) = MLM-loss floor for unigram arm")

    rng = np.random.default_rng(a.gen_seed)
    p_support = uni_p[support].astype(np.float64)
    p_support /= p_support.sum()


    if big is not None:
        row_sums = big.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            trans = np.where(row_sums > 0, big / np.maximum(row_sums, 1), uni_p[None, :])
        first_p = first / first.sum()
        trans_cdf = np.cumsum(trans, axis=1); trans_cdf[:, -1] = 1.0
        first_cdf = np.cumsum(first_p); first_cdf[-1] = 1.0


    _log("pass 2: emitting synthetic shards")
    syn_uni = np.zeros(V, dtype=np.int64)
    for name in names:
        data = _load_shard(local_src / name)
        out = []
        if a.mode == "unigram":
            for d in data:
                L = len(d["input_ids"])
                toks = rng.choice(support, size=L, p=p_support)
                out.append({"input_ids": toks.astype(np.int32).tolist(),
                            "attention_mask": [1] * L})
                np.add.at(syn_uni, toks, 1)
        elif a.mode == "bigram":
            for d in data:
                L = len(d["input_ids"])
                u = rng.random(L)
                seq = np.empty(L, dtype=np.int64)
                prev = int(np.searchsorted(first_cdf, u[0]))
                seq[0] = prev
                for j in range(1, L):
                    prev = int(np.searchsorted(trans_cdf[prev], u[j]))
                    seq[j] = prev
                out.append({"input_ids": seq.astype(np.int32).tolist(),
                            "attention_mask": [1] * L})
                np.add.at(syn_uni, seq, 1)
        pickle.dump({"data": out}, open(local_out / name, "wb"))
        _log(f"  wrote {name} ({len(out):,} seqs)")


    syn_p = syn_uni / syn_uni.sum()
    m = uni > 0
    kl = float((uni_p[m] * np.log(uni_p[m] / np.maximum(syn_p[m], 1e-12))).sum())
    diag = {
        "mode": a.mode,
        "gen_seed": a.gen_seed,
        "n_sequences": int(n_seq),
        "n_content_tokens": int(n_tok),
        "support_size": int(len(support)),
        "unigram_entropy_nats": H_uni,
        "unigram_entropy_bits": H_uni / float(np.log(2)),
        "mlm_loss_floor_nats": H_uni,
        "token_freq_KL_real_given_synth_nats": kl,
        "note": ("KL(real||synth) over the token marginal should be ~0 (matched by construction "
                 "for unigram; small for bigram/bagswap). MLM train loss plateauing at "
                 "unigram_entropy_nats certifies the unigram corruption worked."),
    }
    (local_out / "_diagnostics.json").write_text(json.dumps(diag, indent=2))
    _log(f"token-freq KL(real||synth) = {kl:.6e} nats   (want ~0)")
    _log(f"diagnostics -> {local_out/'_diagnostics.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
