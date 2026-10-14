from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np


def to_selfies(smiles: list[str]) -> tuple[list[str], list[str]]:

    import selfies as sf
    keep, out = [], []
    for s in smiles:
        try:
            e = sf.encoder(s)
        except Exception:
            continue
        if e:


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


    _kept = set(kept)
    unencodable = [s for s in smiles if s not in _kept]
    if a.selfies:
        print(f"[extract] selfies: {len(kept)}/{len(smiles)} encodable, "
              f"{len(unencodable)} kept as all-NaN rows", flush=True)

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


            mw = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mw).sum(1) / mw.sum(1).clamp(min=1e-9)
            feats.append(pooled.float().cpu().numpy())
            if i and i % (a.batch_size * 50) == 0:
                r = i / max(time.time() - t0, 1e-9)
                print(f"[extract] {i}/{len(texts)}  {r:.0f} mol/s  "
                      f"eta {(len(texts) - i) / max(r, 1e-9) / 60:.1f} min", flush=True)
    X = np.concatenate(feats, 0).astype(np.float32)
    if unencodable:
        pad = np.full((len(unencodable), X.shape[1]), np.nan, dtype=np.float32)
        X = np.concatenate([X, pad], 0)
        kept = list(kept) + list(unencodable)
        Path(a.out + ".unencodable.json").write_text(json.dumps(unencodable))
        print(f"[extract] added {len(unencodable)} all-NaN rows; listed in "
              f"{a.out}.unencodable.json", flush=True)


    finite = X[np.isfinite(X).all(axis=1)]
    sd = finite.std(axis=0)
    n = min(512, len(finite))
    sub = finite[np.linspace(0, len(finite) - 1, n).astype(int)]
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


    import transformers as _tf
    meta = {
        "hf_model": a.hf_model, "hf_revision": a.revision or "main",
        "tokenizer_from": a.tokenizer_from or a.hf_model,
        "tokenizer_revision": a.tokenizer_revision or a.revision or "main",
        "pooling": "masked_mean", "encoder_only": bool(a.encoder_only),
        "selfies_input": bool(a.selfies), "max_length": int(max_len),
        "hidden_size": int(X.shape[1]), "n_molecules": int(X.shape[0]),
        "n_all_nan_rows": len(unencodable),
        "n_params_M": round(sum(q.numel() for q in model.parameters()) / 1e6, 1),
        "transformers": _tf.__version__, "torch": torch.__version__,
        "sanity_off_diagonal_cosine": round(off, 4),
        "sanity_median_dim_sd": round(float(np.median(sd)), 6),
    }


    import re as _re
    _m = _re.search(r"-(\d+(?:\.\d+)?)([MB])\b", a.hf_model)
    if _m:
        meta["naming_note"] = (
            f"'{_m.group(0).lstrip('-')}' in the checkpoint name is the PRETRAINING DATA scale "
            f"(molecules), NOT parameters. This checkpoint is {meta['n_params_M']}M parameters. "
            "Reported as a parameter count it would turn a data-scale comparison into a fabricated "
            "scaling result.")
    np.savez_compressed(a.out, smiles=np.asarray([str(s) for s in kept]), X=X,
                        meta=np.asarray(json.dumps(meta)))
    print(f"[extract] meta: {json.dumps(meta)}", flush=True)
    print(f"[extract] wrote {a.out}: {X.shape}, {len(smiles) - len(kept)} dropped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
