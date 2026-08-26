"""Refuse to launch a fig_B rung that differs from skip_dense_48M in anything but corpus and budget.

A rung that differs in two things cannot answer a question about one. The figures session asked for
this gate explicitly and I would want it anyway: the whole value of skip_dense_*_c124 is that the
ONLY change from the existing supervised ladder is the corpus, so a stray field silently makes the
comparison uninterpretable rather than wrong-looking.

IT ALSO BLOCKS ONE SPECIFIC, SILENT CATASTROPHE. descriptor_precompute_dir looks like a harmless
speedup and would corrupt these runs completely:

    pubchem_filtered   12 shards x 1M molecules,  descriptors present for all 12
    pubchem_124m_full  124 shards x 1M molecules, descriptors present for only 12

  and the shards in BOTH corpora are named shard_00000.parquet .. -- so the loader, which resolves
  descriptors as `descriptors_{stem}.npy`, would happily attach pubchem_filtered's descriptor rows
  to pubchem_124m_full's molecules. They are not the same molecules: of the first 5,000 rows, 43
  match. The run would train MTR against targets belonging to other compounds, converge, and
  produce a plausible encoder. Nothing would fail.

skip_dense_48M does not set the field, so on-the-fly descriptors are what the measured 761 FP/s
already reflects. Keeping it absent is both correct AND identical to the reference.
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

REF = Path("figure_data/climb_v2_phase2/skip_dense_48M/config.yaml")
ALLOWED = {
    ("run_id",),
    ("unsupervised_data_paths",),
    ("unsupervised_raw_smiles_paths",),
    ("selection", "total_forward_passes"),
}


def flat(d, prefix=()):
    out = {}
    for k, v in (d or {}).items():
        p = prefix + (k,)
        if isinstance(v, dict):
            out.update(flat(v, p))
        else:
            out[p] = v
    return out


def main(path: str) -> int:
    ref, new = flat(yaml.safe_load(REF.read_text())), flat(yaml.safe_load(Path(path).read_text()))
    diffs = sorted(set(ref) | set(new), key=lambda k: ".".join(k))
    bad = []
    for k in diffs:
        a, b = ref.get(k, "<absent>"), new.get(k, "<absent>")
        if a == b:
            continue
        tag = "ALLOWED" if k in ALLOWED else "FORBIDDEN"
        print(f"  [{tag}] {'.'.join(k)}: {a!r} -> {b!r}")
        if k not in ALLOWED:
            bad.append(k)
    if "descriptor_precompute_dir" in {k[0] for k in new}:
        print("  [FORBIDDEN] descriptor_precompute_dir is set -- shard names collide across corpora; "
              "this would attach pubchem_filtered descriptors to pubchem_124m_full molecules")
        bad.append(("descriptor_precompute_dir",))
    if bad:
        print(f"REFUSING: {len(bad)} forbidden difference(s) from skip_dense_48M")
        return 1
    print("GATE PASSED: differs from skip_dense_48M in corpus and budget only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
