"""Experiment A — materialize synthetic pre-tokenized corpora that ablate WHICH statistic of the
real corpus a masked-LM can exploit. Each synthetic corpus mirrors the real tokenized corpus
1:1 (one synthetic sequence per real molecule, SAME length, SAME shard/order), so sequence count,
length marginal and non-padding token count are matched by construction. Only the token *content*
is resampled, according to --mode:

  unigram : each sequence = ℓ iid draws from the CORPUS-LEVEL unigram (token marginal) distribution.
            Destroys all within-molecule structure; keeps only the token frequency. This is the
            M_UG analogue — the operative arm of Experiment A.
  bigram  : each sequence = a length-ℓ walk of a first-order Markov chain fit on the corpus (token
            i -> i+1 transitions). Keeps LOCAL co-occurrence (adjacent-token statistics) but not
            global molecular structure (ring closures, branch/paren matching, valence). The
            intermediate rung between `unigram` (marginal only) and real SMILES (full structure).
  bagswap : each sequence = the token multiset of ANOTHER length-matched molecule, order shuffled.
            (Provided for completeness; NOTE this is distributionally identical to the existing
            shuffle_tokens arm — see the ladder discussion — so it is not expected to add
            information beyond it.)

Why a MATERIALIZED corpus and not an on-the-fly collator (like shuffle_tokens): the unigram/bigram
statistics are CORPUS-LEVEL, so they must be estimated over the whole corpus first; materializing
also lets us certify the corruption worked (token-frequency KL(synthetic||real) ≈ 0; for bigram,
bigram-KL small; MLM train loss will plateau at the unigram entropy floor) BEFORE spending any GPU.

IMPORTANT — corpus convention: the real pre-tokenized corpus stores PURE CONTENT tokens (no
BOS/EOS/PAD/MASK; min id 7, max 999). The MLM collator adds padding and masking downstream. So the
synthetic sequences must ALSO contain only content tokens (no specials), or the arm would differ
from the real arms in two things at once. We sample over the empirical content-token support, so no
special id is ever emitted and the token marginal is matched exactly.

The synthetic corpus is generated ONCE with a fixed --gen-seed and is SHARED by all 3 pretraining
seeds (which vary model init / masking / data order, not the corpus) — mirroring how
unsup_8M{,_s1,_s2} all read the one real corpus.

Usage (on the GPU box; reads/writes local disk, uploads to S3):
    python scripts/build_synthetic_corpus.py --mode unigram \
        --src_prefix s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_tokenized_pkl \
        --n_shards 40 --local_src /data/real_pkl --local_out /data/unigram_pkl \
        --s3_out s3://climb-s3-bucket/tokenized_sources/pubchem_filtered_unigram_pkl \
        --gen-seed 12345
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np


def _log(m: str) -> None:
    print(f"[synth {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _shard_names(n: int) -> list[str]:
    return [f"shard_{i:05d}.pkl" for i in range(n)]


def _dl(src_prefix: str, name: str, local_src: Path) -> Path:
    local = local_src / name
    if not local.exists():
        subprocess.run(["aws", "s3", "cp", f"{src_prefix.rstrip('/')}/{name}", str(local)],
                       check=True, capture_output=True)
    return local


def _load_shard(path: Path) -> list:
    obj = pickle.load(open(path, "rb"))
    return obj["data"] if isinstance(obj, dict) else obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unigram", "bigram", "bagswap"], required=True)
    ap.add_argument("--src_prefix", required=True)
    ap.add_argument("--n_shards", type=int, default=40)
    ap.add_argument("--local_src", required=True)
    ap.add_argument("--local_out", required=True)
    ap.add_argument("--s3_out", default=None)
    ap.add_argument("--gen-seed", type=int, default=12345)
    ap.add_argument("--vocab-size", type=int, default=1000)
    a = ap.parse_args()

    local_src = Path(a.local_src); local_src.mkdir(parents=True, exist_ok=True)
    local_out = Path(a.local_out); local_out.mkdir(parents=True, exist_ok=True)
    names = _shard_names(a.n_shards)
    V = a.vocab_size

    # ---- Pass 1: fit corpus statistics ----
    _log(f"pass 1: fitting {a.mode} statistics over {a.n_shards} shards")
    uni = np.zeros(V, dtype=np.int64)
    big = np.zeros((V, V), dtype=np.int64) if a.mode == "bigram" else None
    first = np.zeros(V, dtype=np.int64)          # empirical first-token distribution
    n_seq = 0
    n_tok = 0
    t0 = time.time()
    for name in names:
        data = _load_shard(_dl(a.src_prefix, name, local_src))
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
    H_uni = float(-(uni_p[support] * np.log(uni_p[support])).sum())          # nats
    _log(f"corpus: {n_seq:,} sequences, {n_tok:,} content tokens, |support|={len(support)}")
    _log(f"unigram entropy H={H_uni:.4f} nats ({H_uni/np.log(2):.4f} bits) = MLM-loss floor for unigram arm")

    rng = np.random.default_rng(a.gen_seed)
    p_support = uni_p[support].astype(np.float64)
    p_support /= p_support.sum()          # normalized once; sampled per sequence in pass 2

    # For bigram: row-normalized transition matrix with unigram back-off for unseen contexts.
    # Precompute CDFs so generation is inverse-transform sampling via searchsorted (one uniform draw
    # per token) — ~1000x faster than rng.choice(p=...) which rebuilds a cumsum on every call.
    if big is not None:
        row_sums = big.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            trans = np.where(row_sums > 0, big / np.maximum(row_sums, 1), uni_p[None, :])
        first_p = first / first.sum()
        trans_cdf = np.cumsum(trans, axis=1); trans_cdf[:, -1] = 1.0   # guard FP drift at the top
        first_cdf = np.cumsum(first_p); first_cdf[-1] = 1.0

    # ---- Pass 2: emit synthetic shards mirroring lengths/order ----
    _log("pass 2: emitting synthetic shards")
    syn_uni = np.zeros(V, dtype=np.int64)
    prev_leftover = []          # for bagswap length-matched pairing across shard boundary (unused: within-shard)
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
        elif a.mode == "bagswap":
            # length-matched multiset swap within the shard: group indices by length, derange within
            # each group, emit shuffled bag of the paired molecule.
            lengths = np.array([len(d["input_ids"]) for d in data])
            order = np.arange(len(data))
            partner = np.empty(len(data), dtype=np.int64)
            for L in np.unique(lengths):
                grp = order[lengths == L]
                perm = rng.permutation(grp)
                if len(grp) > 1:                       # avoid fixed points where possible
                    for k in range(len(grp)):
                        if perm[k] == grp[k]:
                            perm[k], perm[(k + 1) % len(grp)] = perm[(k + 1) % len(grp)], perm[k]
                partner[grp] = perm
            out = [None] * len(data)
            for k in range(len(data)):
                bag = np.asarray(data[partner[k]]["input_ids"], dtype=np.int64)
                rng.shuffle(bag)
                out[k] = {"input_ids": bag.astype(np.int32).tolist(),
                          "attention_mask": [1] * len(bag)}
                np.add.at(syn_uni, bag, 1)
        pickle.dump({"data": out}, open(local_out / name, "wb"))
        _log(f"  wrote {name} ({len(out):,} seqs)")

    # ---- Diagnostics: certify the corruption ----
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
        "mlm_loss_floor_nats": H_uni,     # a well-corrupted unigram arm plateaus here
        "token_freq_KL_real_given_synth_nats": kl,
        "note": ("KL(real||synth) over the token marginal should be ~0 (matched by construction "
                 "for unigram; small for bigram/bagswap). MLM train loss plateauing at "
                 "unigram_entropy_nats certifies the unigram corruption worked."),
    }
    (local_out / "_diagnostics.json").write_text(json.dumps(diag, indent=2))
    _log(f"token-freq KL(real||synth) = {kl:.6e} nats   (want ~0)")
    _log(f"diagnostics -> {local_out/'_diagnostics.json'}")

    # ---- Upload ----
    # Use `cp --recursive` (PutObject per file), NOT `sync`: the box's IAM role can PutObject but
    # lacks s3:ListBucket, so `sync` (which lists the destination to diff) fails with AccessDenied.
    if a.s3_out:
        _log(f"uploading to {a.s3_out}")
        subprocess.run(["aws", "s3", "cp", str(local_out), a.s3_out.rstrip("/"),
                        "--recursive", "--only-show-errors"], check=True)
        _log("upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
