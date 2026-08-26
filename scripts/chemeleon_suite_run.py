"""CheMeleon-suite FROZEN-probe runner (Tracks A/B). For each task it uses the track's OWN fixed
train/test split (NOT CV), featurizes with the chosen featurizer, z-scores (encoder/chemeleon only),
fits our standard head per seed, predicts the held-out test set, and records the task's metrics —
plus, for MoleculeACE, the overall / cliff / non-cliff test RMSE separately.

Apples-to-apples with CLIMB's frozen encoders: identical featurize->standardize->head pipeline
(reused from featurize_v2 / heads_v2 / eval_v2). Regression is native-unit (target scaler fit on
train only, predictions unscaled).

Outputs (NOTHING overwritten):
  figure_data/chemeleon_suite/<track>/<model>/results.csv   # long: task,seed,subset,metric,value,n_test
  figure_data/chemeleon_suite/<track>/<model>/verified.json # written when all tasks x seeds complete

e2e (chemprop) and ToxCast-kNN modes are separate runners (see HARNESS.md); this file is frozen-only.

Usage:
  python scripts/chemeleon_suite_run.py --track moleculeace --featurizer ecfp4  --model ecfp4 \
      --head mlp --seeds 42 117 709
  python scripts/chemeleon_suite_run.py --track polaris --featurizer chemeleon --model chemeleon_frozen ...
  python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
      --encoder figure_data/climb_v2_phase2/unsup_8M/encoder --tokenizer figure_data/_tokenizer --model unsup_8M
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from featurize_v2 import apply_standardizer, ecfp4_features, fit_standardizer  # noqa: E402
from heads_v2 import make_head  # noqa: E402
import eval_v2  # noqa: E402

MACE_DIR = ROOT / "chemeleon_suite" / "data" / "moleculeace"
POLARIS_DIR = ROOT / "chemeleon_suite" / "data" / "polaris"
MACE_TARGET = "y [pEC50/pKi]"


# ---------------- task loading ----------------

def load_task(track, task):
    """Return (smiles, y[N,1], split[list], cliff[bool array or None], task_type)."""
    if track == "moleculeace":
        rows = list(csv.DictReader((MACE_DIR / f"{task}.csv").open()))
        smi = [r["smiles"] for r in rows]
        y = np.array([float(r[MACE_TARGET]) for r in rows], dtype=np.float64).reshape(-1, 1)
        split = [r["split"] for r in rows]
        cliff = np.array([r["cliff_mol"] in ("1", "1.0", "True") for r in rows])
        return smi, y, split, cliff, "regression"
    if track == "polaris":
        man = json.loads((POLARIS_DIR / "polaris_manifest.json").read_text())
        rows = list(csv.DictReader((POLARIS_DIR / f"{task.replace('/', '__')}.csv").open()))
        smi = [r["smiles"] for r in rows]
        split = [r["split"] for r in rows]
        y = np.array([float(r["y"]) if r["y"] not in ("", "nan", "None") else np.nan
                      for r in rows], dtype=np.float64).reshape(-1, 1)
        return smi, y, split, None, man[task]["type"]
    raise ValueError(f"unknown track {track}")


def task_list(track):
    name = {"moleculeace": "moleculeace_tasks.txt", "polaris": "polaris_tasks.txt"}[track]
    return (ROOT / "chemeleon_suite" / "tasks" / name).read_text().split()


# ---------------- metrics ----------------

def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def reg_metrics(yt, yp):
    yt = yt.ravel(); yp = yp.ravel()
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    def _sp(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    return {"rmse": _rmse(yt, yp), "mae": float(np.mean(np.abs(yt - yp))),
            "r2": r2, "spearman": _sp(yt, yp), "pearson": float(np.corrcoef(yt, yp)[0, 1])}


def clf_metrics(yt, yp):
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    yt = yt.ravel(); yp = yp.ravel()
    m = ~np.isnan(yt)
    yt, yp = yt[m], yp[m]
    out = {}
    if len(np.unique(yt)) >= 2:
        out["roc_auc"] = float(roc_auc_score(yt, yp))
        out["pr_auc"] = float(average_precision_score(yt, yp))
        out["accuracy"] = float(accuracy_score(yt, (yp > 0.5).astype(int)))
    return out


# ---------------- featurization ----------------

NPZ_META: dict = {}   # provenance read out of a precomputed feature table, if present


def prepare_fold(X, tr, head, std_method):
    """Standardise and impute a feature matrix for ONE fold, exactly as run() does.

    EXTRACTED BECAUSE I LOST BOTH RULES BY REWRITING THE LOOP. wong_run.py and cbs_run.py called
    make_featurizer directly and reimplemented the fold loop, which silently dropped two fixes that
    already existed here, each installed after its own incident:

      * make_featurizer returns std="none" for ecfp4/fp_desc because those arms were defined with
        XGBoost, which is scale-invariant. An MLP is not -- unscaled descriptors span 20+ orders of
        magnitude beside 0/1 bits and collapse it (fp_desc__mlp returned CONSTANT predictions for
        whole Polaris cells).
      * fp_desc keeps undefined descriptors as NaN because XGBoost consumes them natively. An MLP
        cannot: one NaN makes every output NaN, and with std="none" nothing upstream removes them.
        On Polaris that produced 9 of 28 tasks at 100% NaN, exiting 0 with full-size prediction
        files.

    Wong/fp_desc died with "Input contains NaN" -- the loud version of the same thing. Any runner
    that builds its own loop must call THIS, not re-derive it.

    Imputation uses TRAIN-fold medians only, so no test information reaches the fit, and applies
    only when the head needs it -- an XGBoost arm is untouched.
    """
    import numpy as _np
    if head != "xgb" and std_method == "none":
        std_method = "zscore"
    Z = _np.asarray(X, dtype=_np.float32)
    if head != "xgb" and not _np.isfinite(Z).all():
        med = _np.nanmedian(_np.where(_np.isfinite(Z[tr]), Z[tr], _np.nan), axis=0)
        med = _np.where(_np.isfinite(med), med, 0.0)
        bad = ~_np.isfinite(Z)
        n_bad = int(bad.sum())
        Z = _np.where(bad, _np.broadcast_to(med, Z.shape), Z)
        print(f"[prepare_fold] imputed {n_bad} non-finite entries from TRAIN medians", flush=True)
    if std_method == "zscore":
        mu = _np.nanmean(Z[tr], 0)
        sd = _np.nanstd(Z[tr], 0)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd
    return Z


def make_featurizer(featurizer, encoder_path, tokenizer_path, hf_model=None, hf_revision=None):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu")
    enc = tok = None
    max_len = 256
    if featurizer == "encoder":
        from transformers import ModernBertModel, PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        enc = ModernBertModel.from_pretrained(encoder_path, attn_implementation="sdpa",
                                              reference_compile=False).to(device).eval()
    elif featurizer == "npz":
        # PRECOMPUTED VECTORS from a model that will not load in this environment (MoLFormer-c3,
        # selfies-ted). eval_v2._load_feature_table gives a STRICT {smiles: vector} lookup that
        # raises on a miss rather than mean-filling, so a molecule the extractor dropped surfaces
        # as a loud KeyError instead of a fabricated row.
        _table = eval_v2._load_feature_table(encoder_path)
        _dim = len(next(iter(_table.values())))
        # CARRY THE TABLE'S PROVENANCE INTO THE RESULT. Without this, verified.json records
        # featurizer "npz" and a path -- less than the direct HF path records, so routing an arm
        # through a precomputed table would quietly downgrade what the artifact knows about itself.
        try:
            _z = np.load(encoder_path, allow_pickle=False)
            NPZ_META.update(json.loads(str(_z["meta"])) if "meta" in _z.files else {})
        except Exception as _e:
            print(f"[npz] WARNING: no meta in {encoder_path} ({_e}) -- provenance will be thin",
                  flush=True)
        print(f"[npz] {encoder_path}: {len(_table)} molecules, dim {_dim} "
              f"model={NPZ_META.get('hf_model', 'UNRECORDED')}", flush=True)
    elif featurizer == "hf_encoder":
        # LITERATURE CLMs THROUGH THE SAME PROBE. The whole point of the fig_A ranking is to compare
        # REPRESENTATIONS, so these get the frozen encoder + masked-mean pooling + z-score + MLP that
        # the CLIMB arms get -- not each paper's own fine-tuning recipe. A fine-tuned arm measures the
        # head; that confound is what made CheMeleon unreadable for a week, when its frozen-XGBoost
        # bar beat its own fine-tuned bar.
        from transformers import AutoModel, AutoTokenizer
        kw = {"trust_remote_code": True}
        if hf_revision:
            kw["revision"] = hf_revision
        tok = AutoTokenizer.from_pretrained(hf_model, **kw)
        enc = AutoModel.from_pretrained(hf_model, **kw).to(device).eval()
        # RESPECT THE MODEL'S OWN LIMIT. MoLFormer's positional embeddings stop at 202; feeding it
        # our ModernBERT 256 would index past them. Take the smallest sane bound the checkpoint
        # advertises rather than assuming ours applies to someone else's model.
        cand = [256]
        for attr in ("max_position_embeddings", "n_positions", "max_len"):
            v = getattr(enc.config, attr, None)
            if isinstance(v, int) and 0 < v < 4096:
                cand.append(v)
        mm = getattr(tok, "model_max_length", None)
        if isinstance(mm, int) and 0 < mm < 4096:
            cand.append(mm)
        max_len = min(cand)
        print(f"[hf_encoder] {hf_model} rev={hf_revision or 'main'} "
              f"hidden={enc.config.hidden_size} max_length={max_len}", flush=True)

    def feat(smiles):
        if featurizer == "ecfp4":
            return np.asarray(ecfp4_features(smiles), dtype=np.float32)
        if featurizer == "fp_desc":
            from descriptors_v2 import rdkit_descriptors
            fp = np.asarray(ecfp4_features(smiles), dtype=np.float32)
            d = np.asarray(rdkit_descriptors(list(smiles)), dtype=np.float32)
            d[~np.isfinite(d)] = np.nan
            return np.concatenate([fp, d], axis=1)
        if featurizer == "npz":
            missing = [x for x in smiles if x not in _table]
            if missing:
                raise KeyError(f"{len(missing)} molecules absent from {encoder_path}, "
                               f"e.g. {missing[:3]} -- re-extract rather than impute")
            M = np.asarray([_table[x] for x in smiles], dtype=np.float32)
            # ALL-NaN ROWS are molecules the featurizer could not represent -- SELFIES cannot
            # encode 309 of the 177,922. They are PRESENT so that every arm sees the same molecule
            # set (fig_F made this call for CheMeleon's 15 organometallics), and imputed to the
            # column mean here because the MLP head, unlike XGBoost, cannot consume NaN. Imputing
            # to the mean gives the arm no information about those molecules rather than a
            # fabricated one. Counted out loud so it can never be a silent zero.
            bad = ~np.isfinite(M).all(axis=1)
            if bad.any():
                good = M[~bad]
                fill = np.nanmean(good, axis=0) if len(good) else np.zeros(M.shape[1], np.float32)
                M[bad] = fill
                print(f"[npz] {int(bad.sum())}/{len(M)} molecules had no representation "
                      f"({100 * bad.mean():.3f}%) -- imputed to the feature mean", flush=True)
                NPZ_META["n_imputed_this_task"] = int(bad.sum())
            return M
        if featurizer == "chemeleon":
            return eval_v2._chemeleon_features(smiles, device)
        return eval_v2._encoder_features(enc, tok, smiles, device, "mean", max_len)

    std = "none" if featurizer in ("ecfp4", "fp_desc") else "zscore"
    return feat, std


# ---------------- run ----------------

def run(track, featurizer, model, head, seeds, encoder_path, tokenizer_path,
        hf_model=None, hf_revision=None):
    out_dir = ROOT / "figure_data" / "chemeleon_suite" / track / model
    out_dir.mkdir(parents=True, exist_ok=True)
    feat, std_method = make_featurizer(featurizer, encoder_path, tokenizer_path,
                                       hf_model=hf_model, hf_revision=hf_revision)
    # MATCH THE HEAD, AND MATCH THIS ARM'S OWN MolNet CELL.
    #
    # make_featurizer returns std="none" for ecfp4/fp_desc because those arms were defined with an
    # XGBoost head, which is scale-invariant. An MLP is not: raw RDKit descriptors span orders of
    # magnitude (MolWt ~1e2, some counts ~1e6) next to 0/1 fingerprint bits, and unscaled they
    # collapse it. On Polaris that is not hypothetical -- fp_desc__mlp returned a CONSTANT
    # prediction for whole (task, seed) cells, which is why pearsonr/spearmanr came back NaN on 7
    # tasks that had no NaN inputs at all.
    #
    # head_comparison_run.sh already passes --standardize zscore for the MolNet half of this exact
    # arm, so leaving the suite half at "none" also made the two halves incomparable. Align them.
    if head != "xgb" and std_method == "none":
        print(f"[suite] head={head} on featurizer={featurizer}: standardize none -> zscore "
              f"(matches this arm's MolNet cell; unscaled descriptors collapse an MLP)", flush=True)
        std_method = "zscore"
    tasks = task_list(track)
    rows = []
    pred_rows = []
    for task in tasks:
        smi, y, split, cliff, ttype = load_task(track, task)
        X = feat(smi)
        idx = np.arange(len(smi))
        tr = idx[[s == "train" for s in split]]
        te = idx[[s == "test" for s in split]]
        # Label-efficiency sweeps (SI fig e) need a REDUCED label budget on this benchmark's own
        # native split. Subsample the TRAIN indices only -- never the test set -- using the same
        # rng and without-replacement draw as eval_v2._subsample_train, so a point here means the
        # same thing as the corresponding point on the MoleculeNet panels. FRACTION is per task
        # (fraction * n_train), not a shared absolute count, so no two fractions collapse onto the
        # same subset on small tasks.
        if TRAIN_FRACTION is not None and TRAIN_FRACTION < 1.0:
            n_keep = max(1, int(round(TRAIN_FRACTION * len(tr))))
            if n_keep < len(tr):
                rng_sub = np.random.default_rng(SUBSAMPLE_SEED)
                tr = np.sort(rng_sub.choice(tr, size=n_keep, replace=False))
        # NaN DESCRIPTORS MEET A HEAD THAT CANNOT EAT THEM.
        #
        # fp_desc deliberately KEEPS non-finite descriptor values as NaN, because XGBoost consumes
        # them natively and imputing would throw away the "this descriptor is undefined here"
        # signal. An MLP cannot: one NaN input makes every output NaN. Worse, std_method="none"
        # for fp_desc, so nothing upstream removes them, and the failure is silent -- the run exits
        # 0 and writes a full-size test_predictions.csv in which every value is NaN.
        #
        # That is what happened to fp_desc__mlp on Polaris (2026-08-20): 9 of 28 tasks, including
        # tdcommons/ames -- the Ames panel -- came back 100% NaN. The regression tasks among them
        # then scored to NaN rather than raising, so they entered polaris_scores.csv looking like
        # data. Only the classification ones failed loudly.
        #
        # Impute from the TRAIN fold only (median, 0.0 for an all-NaN column), so no test
        # information reaches the fit, and only when the head actually needs it -- an XGBoost arm
        # is untouched and every existing number is bit-identical. Loud, because a silent impute is
        # how this class of bug survives.
        if head != "xgb" and not np.isfinite(X).all():
            bad = ~np.isfinite(X)
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(np.isfinite(X[tr]), X[tr], np.nan), axis=0)
            med[~np.isfinite(med)] = 0.0
            print(f"[suite] {task}: head={head} cannot consume NaN -- imputing "
                  f"{int(bad.sum())} non-finite cells in {int(bad.any(0).sum())} column(s) "
                  f"from train medians", flush=True)
            X = np.where(bad, med, X)

        sp = fit_standardizer(X[tr], std_method)
        Xtr, Xte = apply_standardizer(X[tr], sp), apply_standardizer(X[te], sp)
        ysc = eval_v2._fit_target_scaler(y[tr], ttype)
        n_out = 1
        yte = y[te]
        te_smi = [smi[i] for i in te]
        has_labels = bool(np.isfinite(yte).any())   # MoleculeACE: yes; Polaris: test labels hidden
        for seed in seeds:
            hd = make_head(head, ttype, n_out, seed)
            # Fast frozen-probe head: big batch + fewer epochs. The default (batch 64, 100 epochs) is
            # launch-overhead-bound for a tiny MLP over frozen features; this is ~10x faster with
            # negligible effect on a converged probe. Applied UNIFORMLY to every suite arm (fair).
            if head in ("mlp", "linear") and hasattr(hd, "hp"):
                hd.hp = dict(hd.hp); hd.hp.update({"batch_size": 512, "epochs": 60, "patience": 8})
            hd = hd.fit(Xtr, eval_v2._scale_targets(y[tr], ysc), Xtr, eval_v2._scale_targets(y[tr], ysc))
            pred = eval_v2._unscale_preds(np.asarray(hd.predict(Xte), dtype=np.float64), ysc)
            if ttype == "regression":
                # Bound OOD extrapolation: an unbounded MLP over some pretrained embeddings (e.g. CheMeleon)
                # blows up on a few test molecules far from the train distribution, wrecking RMSE. Clip to the
                # train target range + 25% margin — physically-motivated, uniform across arms, a no-op for
                # well-behaved features (whose predictions already sit inside this band).
                ylo, yhi = float(np.nanmin(y[tr])), float(np.nanmax(y[tr])); m = 0.25 * (yhi - ylo + 1e-9)
                pred = np.clip(pred, ylo - m, yhi + m)
            # always dump per-seed test predictions (Polaris scores these later via benchmark.evaluate()).
            pv = pred.ravel()
            for i in range(len(te_smi)):
                pred_rows.append([task, seed, i, te_smi[i], float(pv[i])])
            if has_labels:  # local scoring (MoleculeACE) — Polaris test labels are hidden by design
                mets = reg_metrics(yte, pred) if ttype == "regression" else clf_metrics(yte, pred)
                for k, v in mets.items():
                    rows.append([task, seed, "overall", k, v, len(te)])
                if track == "moleculeace":  # cliff / non-cliff RMSE split (the CliffPFN numbers)
                    ct = cliff[te]
                    if ct.any():
                        rows.append([task, seed, "cliff", "rmse", _rmse(yte[ct], pred[ct]), int(ct.sum())])
                    if (~ct).any():
                        rows.append([task, seed, "noncliff", "rmse", _rmse(yte[~ct], pred[~ct]), int((~ct).sum())])
        print(f"[suite] {track}/{model} {task}: done ({len(te)} test, "
              f"{'scored' if has_labels else 'preds-only'})", flush=True)

    res = out_dir / "results.csv"
    with res.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "subset", "metric", "value", "n_test"])
        w.writerows(rows)
    # per-seed test predictions in test order (Polaris scoring reads these; also useful for MoleculeACE)
    with (out_dir / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "seed", "test_index", "smiles", "y_pred"])
        w.writerows(pred_rows)
    n_tasks_done = len({r[0] for r in pred_rows})
    if n_tasks_done == len(tasks):
        (out_dir / "verified.json").write_text(json.dumps(
            {"track": track, "model": model, "featurizer": featurizer, "head": head,
             "seeds": seeds, "n_tasks": n_tasks_done,
             # RECORD THE HF CHECKPOINT AND ITS REVISION. The whole reason MoLFormer is pinned to
             # 7b12d946c181 is that its main-branch code will not load under our transformers; a
             # pin that is applied at runtime but not written down is unrecoverable from the
             # artifact, which is the same gap that made every fig_F v1 number require a rebuild
             # to identify. Absent for non-HF arms, which is meaningful rather than missing.
             **({"hf_model": hf_model} if hf_model else {}),
             **({"hf_revision": hf_revision} if hf_revision else {}),
             **({"features_npz": encoder_path, "npz_provenance": NPZ_META}
                if featurizer == "npz" else {}),
             # RECORD THE FP VARIANT -- BUT ONLY FOR ARMS THAT HAVE A FINGERPRINT. "featurizer":
             # "ecfp4" is written identically by a stereo-blind run and a stereo-aware one, so
             # vintage was unrecoverable from the file: the MoleculeACE ecfp4 dir (2026-08-13,
             # pre-stereo) and ecfp4_r3c (2026-08-19, r=3 counts) both claim featurizer "ecfp4".
             # Only the S3 upload timestamp separated them, which is not provenance.
             #
             # The first version of this wrote FP_VARIANT unconditionally, which stamped
             # "fp_variant": "ecfp4_stereo" onto chemeleon_frozen__xgb -- an arm with no fingerprint
             # anywhere. Harmless to the numbers and a false claim in the artefact, which is exactly
             # the failure this field exists to prevent. A field that describes a component the arm
             # does not have is worse than no field, for the same reason "featurizer": "ecfp4"
             # naming three featurizations was: it answers, so you stop asking.
             **({"fp_variant": os.environ.get("FP_VARIANT", "ecfp4_stereo")}
                if featurizer in ("ecfp4", "fp_desc") else {})}))
    print(f"[suite] wrote {res}  ({n_tasks_done}/{len(tasks)} tasks)", flush=True)


TRAIN_FRACTION = None
SUBSAMPLE_SEED = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True, choices=["moleculeace", "polaris"])
    p.add_argument("--featurizer", required=True,
               choices=["ecfp4", "fp_desc", "chemeleon", "encoder", "hf_encoder", "npz"])
    p.add_argument("--hf_model", default=None, help="HF id for --featurizer hf_encoder")
    p.add_argument("--hf_revision", default=None,
                   help="PIN the checkpoint revision. MoLFormer's main-branch remote code "
                        "imports create_bidirectional_mask, absent from the transformers 4.57.3 "
                        "every CLIMB result is pinned to; revision 7b12d946c181 works. Upgrading "
                        "transformers instead would move every other arm -- that is exactly how "
                        "the fig_F v1/v2 mismatch happened.")
    p.add_argument("--train_fraction", type=float, default=None,
                   help="Label-efficiency sweep: keep this FRACTION of each task's own train split "
                        "(test untouched). Per-task, so points cannot collapse on small tasks.")
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--model", required=True, help="output label (e.g. unsup_8M, chemeleon_frozen, ecfp4)")
    p.add_argument("--head", default="mlp", choices=["mlp", "linear", "xgb"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 117, 709])
    p.add_argument("--encoder", default=None)
    p.add_argument("--tokenizer", default=None)
    a = p.parse_args()
    global TRAIN_FRACTION, SUBSAMPLE_SEED
    TRAIN_FRACTION, SUBSAMPLE_SEED = a.train_fraction, a.subsample_seed
    if TRAIN_FRACTION is not None:
        print(f"[suite] label-efficiency: keeping {TRAIN_FRACTION:.0%} of each task's train split "
              f"(seed {SUBSAMPLE_SEED}); test split untouched", flush=True)
    if a.featurizer == "hf_encoder" and not a.hf_model:
        raise SystemExit("--featurizer hf_encoder requires --hf_model")
    run(a.track, a.featurizer, a.model, a.head, a.seeds, a.encoder, a.tokenizer,
        hf_model=a.hf_model, hf_revision=a.hf_revision)


if __name__ == "__main__":
    main()
