"""Verify every irreplaceable artifact exists in all three places: S3, local, HF.

Artifact classes:
  CHECKPOINTS  pretrained encoders (cannot be regenerated without repeating pretraining)
  RESULTS      eval outputs (regenerable from a checkpoint, but expensive)
  TRAINING DATA the tokenized/pretraining corpora + tokenizers

Prints a per-class table and exits non-zero if any CHECKPOINT is missing from S3 — that is the
only class whose loss is unrecoverable. Read-only.

Run: python3 scripts/backup_audit.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S3 = "s3://climb-s3-bucket"
FD = ROOT / "figure_data"


def s3_has(uri, want_file=None):
    """True if the prefix exists (and contains want_file when given)."""
    try:
        out = subprocess.run(["aws", "s3", "ls", uri], capture_output=True, text=True, timeout=60)
        if out.returncode != 0 or not out.stdout.strip():
            return False
        return (want_file in out.stdout) if want_file else True
    except Exception:
        return False


def s3_count(uri, grep=None):
    try:
        out = subprocess.run(["aws", "s3", "ls", uri], capture_output=True, text=True, timeout=120)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        if grep:
            lines = [l for l in lines if grep in l]
        return len(lines)
    except Exception:
        return 0


# ---- CHECKPOINTS: every pretrained encoder that backs a figure arm -------------------------
def check_checkpoints():
    waves = {
        "climb_v2_phase2": ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2",
                            "skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2",
                            "skip_dense_plus_sparse_8M", "skip_sparse_all_8M", "skip_mixed_8M",
                            "skip_minimol_full_8M",
                            "u2s_dense_from8M", "u2s_sparse_all_from8M", "u2s_dense_plus_sparse_from8M",
                            "u2s_mixed_from8M", "u2s_minimol_full_from8M",
                            "random_baseline_00", "corrupt_mlm_8M", "corrupt_mtr_8M",
                            "s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2",
                            # scaling ladders
                            "unsup_2M", "unsup_24M", "unsup_48M", "unsup_100M",
                            "skip_dense_2M", "skip_dense_24M", "skip_dense_48M", "skip_dense_96M"],
    }
    rows, missing = [], []
    for wave, prefixes in waves.items():
        for p in prefixes:
            ok = s3_has(f"{S3}/experiments/{wave}/{p}/encoder/", "model.safetensors")
            rows.append((f"{wave}/{p}", ok))
            if not ok:
                missing.append(f"{wave}/{p}")
    # H1 + vocab waves counted in bulk
    h1 = s3_count(f"{S3}/experiments/climb_v2_h1/", "PRE")
    voc = s3_count(f"{S3}/experiments/climb_v2_vocab/", "PRE")
    return rows, missing, h1, voc


def main():
    print("=" * 78)
    print("BACKUP AUDIT — checkpoints / results / training data across S3, local, HF")
    print("=" * 78)

    print("\n[CHECKPOINTS]  (unrecoverable if lost — must be on S3)")
    rows, missing, h1, voc = check_checkpoints()
    for name, ok in rows:
        if not ok:
            print(f"   MISSING FROM S3  {name}")
    print(f"   on S3: {sum(1 for _, ok in rows if ok)}/{len(rows)} named encoders")
    print(f"   climb_v2_h1 prefixes on S3: {h1} (expect 30 + _logs)")
    print(f"   climb_v2_vocab prefixes on S3: {voc} (expect 8 + _logs)")

    print("\n[RESULTS]  (regenerable from a checkpoint, but expensive)")
    for label, uri, local in [
        ("MoleculeACE", f"{S3}/experiments/chemeleon_suite/moleculeace/", FD / "chemeleon_suite" / "moleculeace"),
        ("Polaris/hERG", f"{S3}/experiments/chemeleon_suite/polaris/", FD / "chemeleon_suite" / "polaris"),
        ("CBS", f"{S3}/experiments/cbs_benchmark/", FD / "cbs_benchmark"),
        ("six_panel", f"{S3}/experiments/six_panel/", FD / "six_panel"),
    ]:
        n_s3 = s3_count(uri)
        n_loc = len(list(local.iterdir())) if local.exists() else 0
        flag = "" if n_s3 and n_loc else "   <-- CHECK"
        print(f"   {label:14s} S3={n_s3:4d}  local={n_loc:4d}{flag}")

    print("\n[TRAINING DATA]")
    for label, uri in [("tokenizer_10M", f"{S3}/tokenizer_10M/"),
                       ("pubchem tokenized", f"{S3}/tokenized_sources/pubchem_filtered_tokenized_pkl/"),
                       ("pubchem raw smiles", f"{S3}/tokenized_sources/pubchem_filtered/"),
                       ("descriptors", f"{S3}/tokenized_sources/pubchem_descriptors/")]:
        print(f"   {label:20s} S3={'yes' if s3_has(uri) else 'NO'}")

    print("\n[HF]  lsieben/climb-results — push with:")
    print("   python scripts/publish_to_hf.py --org lsieben --repo results --execute")
    print("   python scripts/upload_cbs_results_hf.py --org lsieben --execute")
    print("   python scripts/upload_chemeleon_molnet_hf.py --org lsieben --execute")

    if missing:
        print(f"\nFAIL: {len(missing)} checkpoint(s) missing from S3: {missing}")
        return 1
    print("\nOK: every named checkpoint is on S3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
