from __future__ import annotations

import argparse
import glob
import json
import pickle
import time
from pathlib import Path

import numpy as np


def _log(m: str) -> None:
    print(f"[wiki {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _sample_smiles_lengths(real_pkl_dir: str, n: int, max_len: int) -> np.ndarray:

    shards = sorted(glob.glob(f"{real_pkl_dir}/shard_*.pkl"))
    if not shards:
        raise FileNotFoundError(f"No tokenized SMILES shards in {real_pkl_dir}")
    lens = []
    for sh in shards:
        obj = pickle.load(open(sh, "rb"))
        data = obj["data"] if isinstance(obj, dict) else obj
        for d in data:
            lens.append(min(len(d["input_ids"]), max_len))
            if len(lens) >= n:
                break
        if len(lens) >= n:
            break
    arr = np.asarray(lens, dtype=np.int32)
    _log(f"SMILES length dist from {len(arr):,} real seqs: median={int(np.median(arr))} "
         f"mean={arr.mean():.1f} p95={int(np.percentile(arr,95))} (capped at {max_len})")
    return arr


def _wiki_stream(n_articles_cap: int):
    """Yield article texts from wikimedia/wikipedia (pre-cleaned; no markup stripping needed)."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_articles_cap:
            return
        yield ex["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--real_pkl_dir", required=True)
    ap.add_argument("--n_real_len_sample", type=int, default=1_000_000)
    ap.add_argument("--target_chunks", type=int, default=12_000_000)
    ap.add_argument("--n_articles_cap", type=int, default=2_000_000)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--chunks_per_shard", type=int, default=300_000)
    ap.add_argument("--local_out", required=True)
    ap.add_argument("--gen-seed", type=int, default=12345)
    ap.add_argument("--vocab-size", type=int, default=1000)
    a = ap.parse_args()

    out = Path(a.local_out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.gen_seed)
    V = a.vocab_size

    len_pool = _sample_smiles_lengths(a.real_pkl_dir, a.n_real_len_sample, a.max_len)

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(a.tokenizer)

    uni = np.zeros(V, dtype=np.int64)
    buf: list[int] = []
    shard: list[dict] = []
    n_chunks = 0
    n_tokens = 0
    n_articles = 0
    shard_idx = 0
    chunk_lens: list[int] = []
    t0 = time.time()

    def flush_shard():
        nonlocal shard, shard_idx
        if shard:
            pickle.dump({"data": shard}, open(out / f"shard_{shard_idx:05d}.pkl", "wb"))
            _log(f"  wrote shard_{shard_idx:05d}.pkl ({len(shard):,} chunks; total {n_chunks:,})")
            shard = []
            shard_idx += 1

    for text in _wiki_stream(a.n_articles_cap):
        n_articles += 1

        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buf.extend(ids)

        while len(buf) >= a.max_len + 1:
            L = int(len_pool[rng.integers(len(len_pool))])
            if len(buf) < L:
                break
            chunk = buf[:L]; del buf[:L]
            np.add.at(uni, np.asarray(chunk, dtype=np.int64), 1)
            shard.append({"input_ids": [int(x) for x in chunk], "attention_mask": [1] * L})
            n_chunks += 1; n_tokens += L; chunk_lens.append(L)
            if len(shard) >= a.chunks_per_shard:
                flush_shard()
            if n_chunks >= a.target_chunks:
                break
        if n_articles % 50000 == 0:
            _log(f"  {n_articles:,} articles -> {n_chunks:,} chunks ({time.time()-t0:.0f}s)")
        if n_chunks >= a.target_chunks:
            break
    flush_shard()


    cl = np.asarray(chunk_lens)
    diag = {
        "corpus": "wikimedia/wikipedia 20231101.en, frozen SMILES byte-BPE, length-matched chunks",
        "gen_seed": a.gen_seed,
        "n_articles_consumed": n_articles,
        "n_chunks": int(n_chunks),
        "n_content_tokens": int(n_tokens),
        "chunk_len_median": int(np.median(cl)), "chunk_len_mean": float(cl.mean()),
        "smiles_len_median": int(np.median(len_pool)), "smiles_len_mean": float(len_pool.mean()),
        "support_size": int((uni > 0).sum()),
        "wiki_token_unigram": uni.tolist(),
        "note": ("Chunk lengths sampled from the real SMILES length distribution, so length is matched "
                 "and cannot confound the transfer signal. wiki_token_unigram enables the coverage guard."),
    }
    (out / "_diagnostics.json").write_text(json.dumps(diag, indent=2))
    _log(f"DONE: {n_chunks:,} chunks, {n_tokens:,} tokens from {n_articles:,} articles; "
         f"len median wiki={int(np.median(cl))} vs smiles={int(np.median(len_pool))}; "
         f"support={int((uni>0).sum())}/{V}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
