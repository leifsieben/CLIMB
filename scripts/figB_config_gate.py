"""Refuse to launch a fig_B rung that differs from skip_dense_8M in anything but corpus, budget and
its own descriptor directory.

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

THE REFERENCE USED TO BE skip_dense_48M AND THAT WAS WRONG. It is the only rung in its own ladder
that trained against 208 descriptors instead of 217 -- it started 2h43m before the canonical stats
file existed, refit its own normalizer, and is consequently also the only rung computing descriptors
live. Gating the c124 rungs against it meant gating them against the outlier, and the docstring's
"keeping the field absent is identical to the reference" was true of the reference and false of the
ladder. skip_dense_8M is the right reference: 217 descriptors, precompute set, and already the
bridge's comparison partner.

SO THE FIELD IS NO LONGER FORBIDDEN -- IT IS REQUIRED TO MATCH ITS CORPUS. The catastrophe above is
a MISMATCH between corpus and descriptor directory, not the presence of a directory. Forbidding the
field outright would now force every c124 rung onto the live pathway, which is exactly the
difference we are removing from the ladder. The gate therefore checks the pairing, using the same
CORPORA table the precompute writes from, so there is one definition of which directory belongs to
which corpus.
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from precompute_descriptors import CORPORA  # one definition of corpus -> descriptor directory

REF = Path("figure_data/climb_v2_phase2/skip_dense_8M/config.yaml")
ALLOWED = {
    ("run_id",),
    ("unsupervised_data_paths",),
    ("unsupervised_raw_smiles_paths",),
    ("selection", "total_forward_passes"),
    ("descriptor_precompute_dir",),
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
    # The descriptor directory must BELONG TO the corpus this run reads. Absent is allowed (the
    # live pathway); present-and-wrong is the silent catastrophe above.
    raw = new.get(("unsupervised_raw_smiles_paths",)) or []
    if isinstance(raw, str):
        raw = [raw]
    corpora = [c for c in CORPORA if any(c in str(r) for r in raw)]
    desc = new.get(("descriptor_precompute_dir",), "<absent>")
    if desc != "<absent>":
        if len(corpora) != 1:
            print(f"  [FORBIDDEN] cannot identify the corpus from {raw!r}, so the descriptor "
                  f"directory cannot be checked against it")
            bad.append(("descriptor_precompute_dir",))
        elif str(desc).rstrip("/") != CORPORA[corpora[0]][1].rstrip("/"):
            print(f"  [FORBIDDEN] descriptor_precompute_dir {desc!r} does not belong to corpus "
                  f"{corpora[0]} ({CORPORA[corpora[0]][1]}) -- shard names collide across corpora, "
                  f"so this would attach one corpus's descriptor rows to the other's molecules")
            bad.append(("descriptor_precompute_dir",))
        else:
            print(f"  [ok] descriptor_precompute_dir matches corpus {corpora[0]}")
    if bad:
        print(f"REFUSING: {len(bad)} forbidden difference(s) from skip_dense_8M")
        return 1
    print("GATE PASSED: differs from skip_dense_8M in corpus, budget and its own descriptor directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
