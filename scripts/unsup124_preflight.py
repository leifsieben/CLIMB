#!/usr/bin/env python3
"""Pre-launch gate for the unsup124 runs. Refuses to launch on any failure.

Asserts the EFFECTIVE values the job will actually use, not what a config file says.
The step this exists for is the corpus: the 124M pre-tokenized mirror is brand new and
nothing has ever streamed it, so we pull real batches through the real dataset class
before committing 37 GPU-hours.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/ec2-user/CLIMB")

FAIL = []
def check(name, ok, detail=""):
    print(f"  {'ok    ' if ok else 'FAIL  '} {name}{(' :: ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def main():
    manifest_path = sys.argv[1]
    man = json.load(open(manifest_path))
    runs = man["runs"]
    print(f"[preflight] manifest={manifest_path} runs={[r['run_id'] for r in runs]}")

    # ---- 1. GPU visible to the framework, not just to nvidia-smi
    import torch
    check("cuda available", torch.cuda.is_available(), torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "NO GPU")
    check("bf16 supported", torch.cuda.is_bf16_supported())

    # ---- 2. deps import
    try:
        import transformers, deepchem, rdkit, sklearn, safetensors  # noqa
        import rdkit.rdBase
        check("imports (transformers/deepchem/rdkit/sklearn/safetensors)", True,
              f"rdkit {rdkit.rdBase.rdkitVersion} deepchem {deepchem.__version__}")
    except Exception as e:
        check("imports", False, repr(e)[:120])
    # xgboost is NOT required: it is imported lazily inside the 'xgb' head only, and these runs
    # evaluate with --head mlp. Reported for the record, never gating.
    try:
        import xgboost  # noqa
        print("  ok     xgboost present (unused by --head mlp)")
    except Exception:
        print("  note   xgboost absent - fine, --head mlp never imports it")

    # ---- 3. disk headroom
    import shutil
    free = shutil.disk_usage("/home/ec2-user").free / 1e9
    check("disk free > 120GB", free > 120, f"{free:.0f} GB")

    # ---- 4. S3 write access to the run prefix
    s3root = man["s3_backup_root"]
    probe = f"{s3root}/_preflight_probe.txt"
    w = subprocess.run(["bash", "-c", f"echo ok | aws s3 cp - {probe} --only-show-errors"],
                       capture_output=True, text=True)
    r = subprocess.run(["aws", "s3", "cp", probe, "-"], capture_output=True, text=True)
    check("S3 write+read back", w.returncode == 0 and r.stdout.strip() == "ok",
          (w.stderr or r.stderr)[:120])
    subprocess.run(["aws", "s3", "rm", probe, "--only-show-errors"], capture_output=True)

    # ---- 5. code version actually deployed
    import pretrain_v2, eval_v2  # noqa
    check("pretrain_v2/eval_v2 importable", True)
    git = subprocess.run(["git", "-C", "/home/ec2-user/CLIMB", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"         code version: {git or 'unknown'}")

    # ---- 6. EFFECTIVE config values, per run
    for r_ in runs:
        pc = r_["pretrain_config"]
        sel = pc["selection"]
        rid = r_["run_id"]
        check(f"[{rid}] MLM-only", sel["objectives"] == {"mlm": 1.0}, str(sel["objectives"]))
        check(f"[{rid}] canonical (pre-tokenized) path", sel["augmentation"] == "canonical")
        check(f"[{rid}] from scratch", sel["init_encoder_path"] is None)
        check(f"[{rid}] tokenizer is tokenizer_10M",
              pc["tokenizer_path"].rstrip("/").endswith("tokenizer_10M"), pc["tokenizer_path"])
        check(f"[{rid}] corpus is the 124M mirror",
              "pubchem_124m_full_tokenized_pkl" in pc["unsupervised_data_paths"][0],
              pc["unsupervised_data_paths"][0])
        steps = sel["total_forward_passes"] // pc["training"]["batch_size"]
        print(f"         [{rid}] budget {sel['total_forward_passes']:,} FP "
              f"= {steps:,} steps @ bs {pc['training']['batch_size']} "
              f"~= {sel['total_forward_passes']/749/3600:.1f} GPU-h")
        # no stale completion marker would let this run be silently skipped
        vj = Path(r_["output_dir"]) / "verified.json"
        check(f"[{rid}] no stale verified.json", not vj.exists(), str(vj))

    # ---- 7. THE UNTESTED STEP: stream real batches from the new corpus
    corpus = runs[0]["pretrain_config"]["unsupervised_data_paths"]
    from storage_utils import list_data_files
    files = list_data_files(corpus[0], suffixes=(".pkl", ".parquet"))
    check("corpus shards listed in S3", len(files) > 0, f"{len(files)} shards")
    expect_rows = len(files) * 300_000
    max_fp = max(r_["selection"]["total_forward_passes"] for r_ in runs)
    check("corpus >= largest budget (single epoch, no repetition)", expect_rows >= max_fp,
          f"~{expect_rows:,} rows vs budget {max_fp:,}")

    from data_v2 import make_mlm_dataset, MLMCollator
    from torch.utils.data import DataLoader
    from pretrain_v2 import materialize_tokenizer_dir
    from transformers import AutoTokenizer
    tokdir = materialize_tokenizer_dir(runs[0]["pretrain_config"]["tokenizer_path"])
    tok = AutoTokenizer.from_pretrained(tokdir)
    ds = make_mlm_dataset(corpus, subset_fraction=None, subset_seed=0)
    coll = MLMCollator(mask_token_id=tok.mask_token_id, vocab_size=tok.vocab_size,
                       mlm_probability=0.3, max_length=128, pad_token_id=tok.pad_token_id)
    dl = DataLoader(ds, batch_size=256, collate_fn=coll, num_workers=2)
    it = iter(dl)
    b = next(it)
    n_unk = int((b["input_ids"] == tok.unk_token_id).sum())
    check("streamed a real batch from the 124M corpus", b["input_ids"].shape[0] == 256,
          f"shape {tuple(b['input_ids'].shape)}")
    check("batch has MLM labels", "labels" in b and int((b["labels"] != -100).sum()) > 0)
    check("no UNK tokens in batch", n_unk == 0, f"{n_unk} UNK")
    for _ in range(4):
        next(it)
    print("         pulled 5 batches OK")

    print()
    if FAIL:
        print(f"PREFLIGHT FAIL ({len(FAIL)}): {FAIL}")
        return 1
    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
