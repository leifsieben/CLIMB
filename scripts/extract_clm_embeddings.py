"""Extract frozen embeddings from a CLM that will not run in the pinned CLIMB environment.

WHY THIS EXISTS. Two of the fig_A arms cannot load under transformers 4.57.3, which every CLIMB
number is pinned to:

  MoLFormer-c3 (DeepChem, 2025)  its remote code imports transformers.masking_utils.create_bidirectional_mask
  selfies-ted  (IBM, 2024)       takes SELFIES, so it needs the `selfies` package

The tempting fix is to upgrade transformers on the shared venv. That is exactly how the fig_F
v1/v2 mismatch happened -- an unpinned dependency moved 27 of 30 embedding-free cells by a median
0.38 fold SD, larger than the effects being plotted, and nothing looked wrong until a
duplicate-block check caught it. So instead this runs in its OWN venv and hands over an npz keyed
on SMILES, the same split CheMeleon (chemprop) and Mordred already use: featurize in one
environment, score in another, and let no library reach across.

The probe itself -- folds, seeds, z-score, MLP -- still runs in the pinned venv on these vectors,
so these arms go through the identical head as every other arm in the ranking.

    ~/venvs/clm_new/bin/python scripts/extract_clm_embeddings.py \
        --hf_model DeepChem/MoLFormer-c3-1.1B --tokenizer_from ibm/MoLFormer-XL-both-10pct \
        --tokenizer_revision 7b12d946c181 --smiles figure_data/_figA_smiles.json \
        --out figure_data/_molformer_c3.npz

SMILES ARE SAVED AS '<U', NOT object: an object array is pickled, and a pickle written by a modern
numpy references numpy._core, which the numpy 1.23.5 in the pinned venv cannot import -- so the
table would become unreadable exactly where it is used.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np


def to_selfies(smiles: list[str]) -> tuple[list[str], list[str]]:
    """-> (encodable_smiles, selfies). A molecule selfies cannot encode is DROPPED and reported;
    the lookup on the other side raises on a miss rather than mean-filling, so a silent absence
    surfaces as a loud KeyError instead of a fabricated vector."""
    import selfies as sf
    keep, out = [], []
    for s in smiles:
        try:
            e = sf.encoder(s)
        except Exception:
            continue
        if e:
            # SPACE-SEPARATE THE TOKENS. selfies-ted's tokenizer expects "[C] [C] [O]", not
            # "[C][C][O]": fed the unspaced string it emits THREE tokens for any molecule and
            # every embedding comes out identical -- measured off-diagonal cosine 1.0000 and
            # per-dimension sd 0.0000. That arm scored 1.2171 macro RMSE on MoleculeACE, which
            # reads as "selfies-ted is terrible" rather than "the input was destroyed". Spaced,
            # the same molecules give 26 tokens, sd 0.24, cosine 0.62.
            keep.append(s); out.append(e.replace("][", "] ["))
    return keep, out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--tokenizer_from", default=None, help="take the tokenizer from a different repo")
    p.add_argument("--tokenizer_revision", default=None)
    p.add_argument("--smiles", required=True, help="JSON with _all_unique, or .txt one per line")
    p.add_argument("--out", required=True)
    p.add_argument("--selfies", action="store_true", help="model consumes SELFIES (selfies-ted)")
    p.add_argument("--encoder_only", action="store_true",
                   help="use model.encoder rather than the full model. Required for BART-style\n                        encoder-decoders: AutoModel(...).last_hidden_state is the DECODER\n                        output, which is not the representation the model is meant to expose.")
    p.add_argument("--max_length", type=int, default=0, help="0 = take it from the checkpoint")
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()

    src = Path(a.smiles)
    smiles = (json.loads(src.read_text())["_all_unique"] if src.suffix == ".json"
              else [l.strip() for l in src.read_text().splitlines() if l.strip()])
    print(f"[extract] {len(smiles)} molecules", flush=True)

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok_id = a.tokenizer_from or a.hf_model
    tok_kw = {"trust_remote_code": True}
    if a.tokenizer_revision or (a.revision and not a.tokenizer_from):
        tok_kw["revision"] = a.tokenizer_revision or a.revision
    tok = AutoTokenizer.from_pretrained(tok_id, **tok_kw)
    m_kw = {"trust_remote_code": True}
    if a.revision:
        m_kw["revision"] = a.revision
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(a.hf_model, **m_kw).to(dev).eval()

    # RESPECT THE CHECKPOINT'S OWN LIMIT rather than assuming ours applies to someone else's model.
    cand = [512]
    for attr in ("max_position_embeddings", "n_positions", "max_len"):
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and 0 < v < 4096:
            cand.append(v)
    mm = getattr(tok, "model_max_length", None)
    if isinstance(mm, int) and 0 < mm < 4096:
        cand.append(mm)
    max_len = a.max_length or min(cand)
    print(f"[extract] {a.hf_model} rev={a.revision or 'main'} hidden={model.config.hidden_size} "
          f"max_length={max_len} device={dev}", flush=True)

    kept, texts = (to_selfies(smiles) if a.selfies else (list(smiles), list(smiles)))
    if a.selfies:
        print(f"[extract] selfies: {len(kept)}/{len(smiles)} encodable, "
              f"{len(smiles) - len(kept)} dropped", flush=True)

    feats, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(texts), a.batch_size):
            chunk = texts[i:i + a.batch_size]
            enc = tok(chunk, truncation=True, max_length=max_len, padding="longest",
                      return_tensors="pt")
            ids = enc["input_ids"].to(dev)
            mask = enc["attention_mask"].to(dev)
            mod = model.encoder if (a.encoder_only and hasattr(model, "encoder")) else model
            out = mod(input_ids=ids, attention_mask=mask)
            h = out.last_hidden_state
            # MASKED MEAN -- the identical pooling eval_v2._encoder_features applies to the CLIMB
            # encoders. A different pooling here would make the ranking a pooling comparison.
            mw = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mw).sum(1) / mw.sum(1).clamp(min=1e-9)
            feats.append(pooled.float().cpu().numpy())
            if i and i % (a.batch_size * 50) == 0:
                r = i / max(time.time() - t0, 1e-9)
                print(f"[extract] {i}/{len(texts)}  {r:.0f} mol/s  "
                      f"eta {(len(texts) - i) / max(r, 1e-9) / 60:.1f} min", flush=True)
    X = np.concatenate(feats, 0).astype(np.float32)

    # DEGENERACY TRIPWIRE. A featurizer that returns the SAME vector for every molecule produces a
    # perfectly well-formed npz, a complete run, and a plausible-looking bad score -- there is no
    # error anywhere. That is exactly what a mis-tokenised selfies-ted did. So test the condition
    # rather than trust the pipeline: real embeddings separate molecules.
    sd = X.std(axis=0)
    n = min(512, len(X))
    sub = X[np.linspace(0, len(X) - 1, n).astype(int)]
    nrm = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-9)
    cos = nrm @ nrm.T
    off = float((cos.sum() - np.trace(cos)) / (n * n - n))
    print(f"[extract] sanity: median per-dim sd {np.median(sd):.4f}, "
          f"dead dims {int((sd < 1e-6).sum())}/{X.shape[1]}, mean off-diagonal cosine {off:.4f}",
          flush=True)
    if off > 0.99 or np.median(sd) < 1e-4:
        raise SystemExit(
            f"FATAL degenerate embeddings: off-diagonal cosine {off:.4f}, median sd "
            f"{np.median(sd):.2e}. The model is returning near-identical vectors for different "
            f"molecules -- check tokenisation (selfies-ted needs SPACE-SEPARATED tokens) and "
            f"whether --encoder_only is required. Refusing to write a table that would score as "
            f"a bad model rather than a broken input.")

    # THE TABLE CARRIES ITS OWN PROVENANCE. Routing an arm through --featurizer npz erased the
    # model identity from verified.json -- it recorded featurizer "npz" and a file path, which is
    # strictly less than the hf_model/hf_revision the direct path had just been fixed to write.
    # A vector table that does not say what produced it is the fig_F v1 problem in miniature.
    import transformers as _tf
    meta = {
        "hf_model": a.hf_model, "hf_revision": a.revision or "main",
        "tokenizer_from": a.tokenizer_from or a.hf_model,
        "tokenizer_revision": a.tokenizer_revision or a.revision or "main",
        "pooling": "masked_mean", "encoder_only": bool(a.encoder_only),
        "selfies_input": bool(a.selfies), "max_length": int(max_len),
        "hidden_size": int(X.shape[1]), "n_molecules": int(X.shape[0]),
        "n_params_M": round(sum(q.numel() for q in model.parameters()) / 1e6, 1),
        "transformers": _tf.__version__, "torch": torch.__version__,
        "sanity_off_diagonal_cosine": round(off, 4),
        "sanity_median_dim_sd": round(float(np.median(sd)), 6),
    }
    if "c3" in a.hf_model:
        meta["naming_note"] = (
            "MoLFormer-c3-1.1B: the 1.1B is the PRETRAINING DATA scale (molecules), NOT parameters. "
            f"This checkpoint is {meta['n_params_M']}M parameters. Reported as a parameter count it "
            "would turn a data-scale comparison into a fabricated scaling result.")
    np.savez_compressed(a.out, smiles=np.asarray([str(s) for s in kept]), X=X,
                        meta=np.asarray(json.dumps(meta)))
    print(f"[extract] meta: {json.dumps(meta)}", flush=True)
    print(f"[extract] wrote {a.out}: {X.shape}, {len(smiles) - len(kept)} dropped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
