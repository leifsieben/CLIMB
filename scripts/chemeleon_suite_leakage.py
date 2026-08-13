"""Leakage gate for the CheMeleon suite: does our Phase-1 PubChem SSL pretraining corpus (~12M,
`pubchem_filtered`) contain any of the TEST compounds of the suite's tasks? A silent overlap would
invalidate treating this as a tuning-locked benchmark.

Reuses the exact corpus streaming + canonicalization from scripts/dedup_i1_reanalysis.py
(`canon_nosalt` = RDKit isomeric canonical, matching how the corpus stores SMILES_canonical).

Covers MoleculeACE test splits (always) + Polaris test splits (once chemeleon_suite/data/polaris exists).
Output: chemeleon_suite/leakage/pretrain_vs_testsets.json  (+ leaked_pairs.csv if any).

Run from repo root. CPU-only; streams the 12 corpus shards (cached under the dedup cache dir)."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dedup_i1_reanalysis import canon_nosalt, _shards, _shard_smiles  # noqa: E402

OUT = ROOT / "chemeleon_suite" / "leakage"
OUT.mkdir(parents=True, exist_ok=True)


def _test_smiles_by_task():
    tasks = {}
    md = ROOT / "chemeleon_suite" / "data" / "moleculeace"
    for f in sorted(md.glob("CHEMBL*.csv")):
        rows = list(csv.DictReader(f.open()))
        tasks[("moleculeace", f.stem)] = [r["smiles"] for r in rows if r["split"] == "test"]
    pol = ROOT / "chemeleon_suite" / "data" / "polaris"
    if pol.exists():
        for f in sorted(pol.glob("*.csv")):
            rows = list(csv.DictReader(f.open()))
            if not rows or "split" not in rows[0]:
                continue
            tasks[("polaris", f.stem)] = [r["smiles"] for r in rows if r.get("split") == "test"]
    return tasks


def main():
    tasks = _test_smiles_by_task()
    key2tasks = defaultdict(set)
    allkeys = set()
    n_test = 0
    for (track, task), smis in tasks.items():
        n_test += len(smis)
        for s in smis:
            k = canon_nosalt(s)
            if k:
                key2tasks[k].add(f"{track}/{task}")
                allkeys.add(k)
    print(f"[leak] test tasks={len(tasks)} test molecules={n_test} unique canonical keys={len(allkeys)}", flush=True)

    leaked = set()
    n_corpus = 0
    for shard in _shards():
        sset = set(_shard_smiles(shard))
        n_corpus += len(sset)
        leaked |= allkeys & sset
        print(f"[leak] {shard.name}: corpus so far {n_corpus:,}, leaked so far {len(leaked)}", flush=True)

    leaked_pairs = []
    for k in sorted(leaked):
        for t in sorted(key2tasks[k]):
            leaked_pairs.append({"canonical_key": k, "task": t})
    report = {
        "corpus": "pubchem_filtered (~12M, all 12 shards)",
        "corpus_molecules": n_corpus,
        "key": "canon_nosalt (RDKit isomeric canonical, matches corpus SMILES_canonical)",
        "n_test_tasks": len(tasks),
        "n_test_molecules": n_test,
        "n_test_unique_keys": len(allkeys),
        "n_leaked_keys": len(leaked),
        "leak_fraction_of_unique": round(len(leaked) / max(1, len(allkeys)), 6),
        "tasks_with_leak": sorted({t for k in leaked for t in key2tasks[k]}),
        "note": "Polaris covered only if chemeleon_suite/data/polaris exists at run time.",
    }
    (OUT / "pretrain_vs_testsets.json").write_text(json.dumps(report, indent=2))
    if leaked_pairs:
        with (OUT / "leaked_pairs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["canonical_key", "task"]); w.writeheader(); w.writerows(leaked_pairs)
    print(f"[leak] DONE: {len(leaked)}/{len(allkeys)} unique test keys in corpus "
          f"({report['leak_fraction_of_unique']*100:.3f}%). wrote {OUT/'pretrain_vs_testsets.json'}", flush=True)


if __name__ == "__main__":
    main()
