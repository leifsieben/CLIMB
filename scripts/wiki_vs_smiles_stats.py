"""Experiment B — compare the real SMILES corpus vs the Wikipedia corpus (same tokenizer, lengths
matched by construction) on their TOKEN-FREQUENCY distributions and vocabulary usage. This is the
control panel for the transfer result: it makes explicit that the two corpora share length + tokenizer
but differ sharply in the token marginal (a low-level statistic the paper reasons about) and in which
vocab tokens each domain even uses (NLP vs SMILES).

Reports:
  * length: median/mean/p95 for each (matched by design — sanity check they really coincide).
  * token marginal divergence: JS divergence (bits, symmetric/finite) + KL both directions (smoothed).
  * vocabulary: |support| each, shared, SMILES-only ("chemistry tokens Wikipedia never fills"),
    Wiki-only ("English tokens SMILES never uses"), with the top tokens DECODED.
  * the most domain-divergent shared tokens (largest |log freq-ratio|).

Run on the box (needs the real pkl corpus + tokenizer + wiki diagnostics).

Usage:  python scripts/wiki_vs_smiles_stats.py --real_pkl_dir /home/ec2-user/synth/real_pkl \
            --wiki_diag /home/ec2-user/wiki/wiki_pkl/_diagnostics.json \
            --tokenizer experiments/climb_v2_expA/unigram_8M/tokenizer \
            --out analysis/rigor/wiki_vs_smiles_stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np


def _real_unigram(real_pkl_dir: str, n_seq: int, V: int):
    uni = np.zeros(V, dtype=np.int64)
    lens = []
    seen = 0
    for sh in sorted(glob.glob(f"{real_pkl_dir}/shard_*.pkl")):
        obj = pickle.load(open(sh, "rb"))
        data = obj["data"] if isinstance(obj, dict) else obj
        for d in data:
            ids = d["input_ids"]
            np.add.at(uni, np.asarray(ids, dtype=np.int64), 1)
            lens.append(len(ids)); seen += 1
            if seen >= n_seq:
                break
        if seen >= n_seq:
            break
    return uni, np.asarray(lens)


def _js_kl(p, q):
    eps = 1e-12
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b): return float(np.sum(a * (np.log2(a + eps) - np.log2(b + eps))))
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)          # bits, in [0,1]
    ps = p + eps; qs = q + eps; ps /= ps.sum(); qs /= qs.sum()
    return js, kl(ps, qs), kl(qs, ps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_pkl_dir", default="/home/ec2-user/synth/real_pkl")
    ap.add_argument("--wiki_diag", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n_real_seq", type=int, default=2_000_000)
    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--out", default="analysis/rigor/wiki_vs_smiles_stats.json")
    a = ap.parse_args()
    V = a.vocab_size

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(a.tokenizer)

    smi, smi_lens = _real_unigram(a.real_pkl_dir, a.n_real_seq, V)
    wiki = np.asarray(json.loads(Path(a.wiki_diag).read_text())["wiki_token_unigram"], dtype=np.int64)
    if len(wiki) < V:
        wiki = np.pad(wiki, (0, V - len(wiki)))

    js, kl_sw, kl_ws = _js_kl(smi.astype(float), wiki.astype(float))
    sp_s, sp_w = smi > 0, wiki > 0
    smiles_only = np.where(sp_s & ~sp_w)[0]
    wiki_only = np.where(sp_w & ~sp_s)[0]
    shared = np.where(sp_s & sp_w)[0]

    def decode(ids): return {int(i): tok.convert_ids_to_tokens(int(i)) for i in ids}
    def top(ids, freq, k=25):
        order = ids[np.argsort(-freq[ids])][:k]
        return [{"id": int(i), "token": tok.convert_ids_to_tokens(int(i)),
                 "count": int(freq[i])} for i in order]

    # most domain-divergent SHARED tokens (by log freq-ratio), each direction
    ps = smi / smi.sum(); pw = wiki / wiki.sum()
    lr = np.log2((ps[shared] + 1e-12) / (pw[shared] + 1e-12))
    smi_rich = shared[np.argsort(-lr)][:20]     # over-represented in SMILES
    wiki_rich = shared[np.argsort(lr)][:20]     # over-represented in Wikipedia

    out = {
        "length": {
            "smiles": {"median": int(np.median(smi_lens)), "mean": round(float(smi_lens.mean()), 3),
                       "p95": int(np.percentile(smi_lens, 95))},
            "note_wiki_matched_by_construction": "wiki chunk lengths are SAMPLED from this distribution",
        },
        "token_marginal_divergence": {
            "js_divergence_bits": round(js, 4),
            "kl_smiles_wiki_bits": round(kl_sw, 4),
            "kl_wiki_smiles_bits": round(kl_ws, 4),
            "interpretation": "same tokenizer, matched lengths, but the token MARGINALS are far apart",
        },
        "vocabulary": {
            "smiles_support": int(sp_s.sum()), "wiki_support": int(sp_w.sum()),
            "shared": int(len(shared)),
            "smiles_only_count": int(len(smiles_only)), "wiki_only_count": int(len(wiki_only)),
            "smiles_only_tokens_top": top(smiles_only, smi),   # chemistry tokens Wikipedia never fills
            "wiki_only_tokens_top": top(wiki_only, wiki),       # English tokens SMILES never uses
        },
        "most_divergent_shared": {
            "smiles_over_wiki": [{"id": int(i), "token": tok.convert_ids_to_tokens(int(i)),
                                  "smiles_freq": round(float(ps[i]), 5), "wiki_freq": round(float(pw[i]), 5)}
                                 for i in smi_rich],
            "wiki_over_smiles": [{"id": int(i), "token": tok.convert_ids_to_tokens(int(i)),
                                  "smiles_freq": round(float(ps[i]), 5), "wiki_freq": round(float(pw[i]), 5)}
                                 for i in wiki_rich],
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"length: smiles median={out['length']['smiles']['median']} mean={out['length']['smiles']['mean']}")
    print(f"token marginal: JS={js:.4f} bits | KL(smiles||wiki)={kl_sw:.3f} | KL(wiki||smiles)={kl_ws:.3f}")
    print(f"support: smiles={int(sp_s.sum())}  wiki={int(sp_w.sum())}  shared={len(shared)}  "
          f"smiles_only={len(smiles_only)}  wiki_only={len(wiki_only)}")
    print("chemistry tokens Wikipedia NEVER fills (top by SMILES freq):",
          [t["token"] for t in out["vocabulary"]["smiles_only_tokens_top"][:15]])
    print("English tokens SMILES never uses (top by wiki freq):",
          [t["token"] for t in out["vocabulary"]["wiki_only_tokens_top"][:15]])
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
