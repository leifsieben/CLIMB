"""Build a one-run manifest for a fig_B rung, CLONING an existing run's entry.

WHY CLONE RATHER THAN HAND-WRITE. build_unsup124_manifest.py already learned this the hard way and
says so: "a hand-written config containing only {run_id, selection} is accepted by the manifest
loader and then dies inside pretrain_v2 on cfg['tokenizer_path']". The loader validates less than
the trainer needs, so the failure lands hours later on a box, not here.

So every field comes from a REAL manifest entry for the closest existing rung, and only the fields
that define this rung are changed. What may change is enumerated, and anything else differing is a
refusal -- the same gate as figB_config_gate.py, applied to the manifest rather than the config,
because the manifest is what actually runs.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S3_ROOT = "s3://climb-s3-bucket/experiments/climb_v2_phase2"
RESULTS_ROOT = "experiments/climb_v2_phase2"

# rung -> (template run to clone, forward passes, init encoder or None)
SPEC = {
    "skip_dense_8M_c124":   ("skip_dense_8M",     8_000_000,   None),
    "skip_dense_50M_c124":  ("skip_dense_8M",    50_000_000,   None),
    "skip_dense_100M_c124": ("skip_dense_8M",   100_000_000,   None),
    "u2s_dense_from50M":    ("u2s_dense_from8M",  2_000_000,   f"{RESULTS_ROOT}/unsup_50M/encoder"),
    "u2s_dense_from100M":   ("u2s_dense_from8M",  2_000_000,   f"{RESULTS_ROOT}/unsup_100M/encoder"),
}
sys.path.insert(0, str(Path(__file__).parent))
from precompute_descriptors import CORPORA  # one definition of corpus -> descriptor directory

C124 = "s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full_tokenized_pkl/"
R124 = "s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full/"


def find_entry(manifest_paths, run_id):
    for p in manifest_paths:
        try:
            m = json.loads(Path(p).read_text())
        except Exception:
            continue
        for r in m.get("runs", []):
            if r.get("run_id") == run_id:
                return copy.deepcopy(r), p
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(SPEC))
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", action="append", default=[
        "experiments/climb_v2_phase2/manifest.json",
        "experiments/climb_v2_phase2/manifest_supplement.json",
    ])
    a = ap.parse_args()
    tmpl, fp, init = SPEC[a.run]
    entry, src = find_entry(a.manifest, tmpl)
    if entry is None:
        print(f"FATAL: no manifest entry for template {tmpl} in {a.manifest}")
        return 2
    print(f"[figB] cloning {tmpl} from {src}")

    before = copy.deepcopy(entry)
    before_eval = entry.get("evaluation_output_dir", f"{RESULTS_ROOT}/{tmpl}")
    entry["run_id"] = a.run
    entry["output_dir"] = f"{RESULTS_ROOT}/{a.run}"
    # PRESERVE THE TEMPLATE'S SUFFIX. skip_dense_8M's evaluation_output_dir is
    # ".../skip_dense_8M/moleculenet", not the bare run dir; rewriting it to the bare dir would put
    # eval output somewhere nothing downstream reads. Substitute the run id INSIDE the template's
    # own path rather than reconstructing the path.
    entry["evaluation_output_dir"] = before_eval.replace(tmpl, a.run)
    entry["backup_s3_uri"] = f"{S3_ROOT}/{a.run}"
    pc = entry["pretrain_config"]
    pc["run_id"] = a.run
    pc["unsupervised_data_paths"] = [C124]
    pc["unsupervised_raw_smiles_paths"] = [R124]
    # The descriptor directory must BELONG TO the corpus. Both corpora name shards
    # shard_00000.parquet upward and data_v2.py resolves descriptors_{stem}.npy, so pointing a
    # 124M-corpus run at pubchem_filtered's directory attaches that corpus's descriptor rows to
    # these molecules -- 43 of the first 5,000 match, MTR trains against other compounds' targets,
    # and it converges. That is why this used to POP the field outright.
    #
    # Popping it is no longer right. pubchem_124m_full now has its OWN precomputed directory, and
    # leaving the field absent would put every c124 rung on the live pathway -- which is how the
    # broken skip_dense_48M behaved, and a difference we are removing from the ladder rather than
    # spreading. Set it from the same CORPORA table the precompute writes from, so the pairing has
    # one definition. figB_run.sh verifies the directory is complete AND row-aligned before training.
    pc["descriptor_precompute_dir"] = CORPORA["pubchem_124m_full"][1]
    for sel in (entry.get("selection"), pc.get("selection")):
        if not sel:
            continue
        sel["total_forward_passes"] = fp
        sel["pretraining_seed"] = 0
        if init:
            sel["init_encoder_path"] = init
    entry["depends_on"] = []

    # What differs from the template, enumerated -- anything unexpected is a refusal.
    def flat(d, p=()):
        o = {}
        for k, v in (d or {}).items():
            q = p + (k,)
            o.update(flat(v, q)) if isinstance(v, dict) else o.__setitem__(q, v)
        return o
    allowed_tails = {"run_id", "output_dir", "evaluation_output_dir", "backup_s3_uri",
                     "unsupervised_data_paths", "unsupervised_raw_smiles_paths",
                     "total_forward_passes", "init_encoder_path", "descriptor_precompute_dir",
                     "depends_on", "pretraining_seed"}
    fa, fb = flat(before), flat(entry)
    bad = []
    for k in sorted(set(fa) | set(fb), key=lambda x: ".".join(map(str, x))):
        x, y = fa.get(k, "<absent>"), fb.get(k, "<absent>")
        if x == y:
            continue
        ok = k[-1] in allowed_tails
        print(f"  [{'ok' if ok else 'FORBIDDEN'}] {'.'.join(map(str,k))}: {x!r} -> {y!r}")
        if not ok:
            bad.append(k)
    if bad:
        print(f"REFUSING: {len(bad)} unexpected difference(s) from {tmpl}")
        return 1
    Path(a.out).write_text(json.dumps({"runs": [entry]}, indent=2))
    print(f"[figB] wrote {a.out}: 1 run ({a.run}, {fp:,} FP, seed 0, NO replicates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
