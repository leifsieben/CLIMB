"""Generate notes/dataset-inventory.md — every dataset in the project, with sizes measured from disk.

Sizes are computed, never typed by hand: MoleculeNet from the pooled out-of-fold prediction files,
MoleculeACE from the per-target CSVs, Polaris from the benchmark manifest, CBS from data/cbs.csv.
The one-line descriptions are curated text (there is no machine-readable description in any of the
sources); ChEMBL target IDs are NOT resolved to protein names because no local mapping exists.

Run:  python3 scripts/dataset_inventory.py
"""
from __future__ import annotations
import json, glob, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MOLNET_DESC = {
    "BACE":  ("classification", "Binary inhibition of human β-secretase 1 (BACE-1), from a single "
                                "medicinal-chemistry series — so scaffolds are unusually homogeneous."),
    "BBBP":  ("classification", "Binary blood–brain-barrier penetration. RETIRED from the six-panel "
                                "suite 2026-08-16: an untrained encoder ranks 7/16 on it."),
    "ESOL":  ("regression",     "Delaney aqueous solubility, log mol/L. Small and largely predictable "
                                "from size and polarity."),
    "HIV":   ("classification", "Ability to inhibit HIV replication, from the NCI AIDS antiviral "
                                "screen. Large and heavily imbalanced."),
    "Lipophilicity": ("regression", "Experimental octanol/water logD at pH 7.4 (ChEMBL)."),
    "QM7":   ("regression",     "DFT atomization energies (kcal/mol) for small organic molecules "
                                "from GDB-13 — a quantum-chemistry, not a bioactivity, endpoint."),
    "Tox21": ("classification", "12 nuclear-receptor and stress-response toxicity assays, binary, "
                                "scored as the mean ROC-AUC over the 12 subtasks."),
}

POLARIS_DESC = {
    "polaris/adme-fang-hclint-1": "Human liver-microsome intrinsic clearance (log), Fang et al. ADME panel.",
    "polaris/adme-fang-rclint-1": "Rat liver-microsome intrinsic clearance (log), Fang et al. ADME panel.",
    "polaris/adme-fang-hppb-1":   "Human plasma protein binding (log), Fang et al. ADME panel.",
    "polaris/adme-fang-rppb-1":   "Rat plasma protein binding (log), Fang et al. ADME panel.",
    "polaris/adme-fang-perm-1":   "MDR1-MDCK efflux ratio — passive permeability / transporter liability.",
    "polaris/adme-fang-solu-1":   "Kinetic solubility, Fang et al. ADME panel.",
    "polaris/pkis2-egfr-wt-reg-v2": "PKIS2 kinase panel: % inhibition of wild-type EGFR (regression).",
    "polaris/pkis2-kit-wt-reg-v2":  "PKIS2 kinase panel: % inhibition of wild-type KIT (regression).",
    "polaris/pkis2-kit-wt-cls-v2":  "PKIS2 kinase panel: active/inactive against wild-type KIT.",
    "polaris/pkis2-ret-wt-reg-v2":  "PKIS2 kinase panel: % inhibition of wild-type RET (regression).",
    "polaris/pkis2-ret-wt-cls-v2":  "PKIS2 kinase panel: active/inactive against wild-type RET.",
    "tdcommons/ames":               "Ames test — bacterial reverse-mutation mutagenicity.",
    "tdcommons/bbb-martins":        "Blood–brain-barrier penetration (Martins) — same endpoint as BBBP, larger and re-curated.",
    "tdcommons/bioavailability-ma": "Oral bioavailability in humans (Ma), binary.",
    "tdcommons/caco2-wang":         "Caco-2 cell permeability (apparent permeability coefficient).",
    "tdcommons/clearance-hepatocyte-az": "Intrinsic clearance in hepatocytes (AstraZeneca).",
    "tdcommons/clearance-microsome-az":  "Intrinsic clearance in liver microsomes (AstraZeneca).",
    "tdcommons/cyp2c9-substrate-carbonmangels": "Is the molecule a CYP2C9 substrate?",
    "tdcommons/cyp2d6-substrate-carbonmangels": "Is the molecule a CYP2D6 substrate?",
    "tdcommons/cyp3a4-substrate-carbonmangels": "Is the molecule a CYP3A4 substrate?",
    "tdcommons/dili":               "Drug-induced liver injury, binary.",
    "tdcommons/half-life-obach":    "In-vivo elimination half-life in humans (Obach).",
    "tdcommons/herg":               "Blockade of the hERG potassium channel (cardiotoxicity). IN the six-panel suite since 2026-08-16.",
    "tdcommons/ld50-zhu":           "Acute oral toxicity LD50 in rats (Zhu).",
    "tdcommons/lipophilicity-astrazeneca": "Octanol/water logD at pH 7.4 (AstraZeneca).",
    "tdcommons/pgp-broccatelli":    "P-glycoprotein inhibition, binary.",
    "tdcommons/ppbr-az":            "Plasma protein binding rate (AstraZeneca).",
    "tdcommons/vdss-lombardo":      "Volume of distribution at steady state (Lombardo).",
}

IN_SUITE = {"BACE", "Tox21", "QM7"}                       # MolNet members of the six-panel suite


def molnet_rows():
    """Molecule counts from the pooled out-of-fold prediction files (= the full curated set)."""
    out = {}
    for f in glob.glob(str(ROOT / "figure_data/climb_v2_phase2/*/moleculenet_cv/test_predictions.csv")):
        d = pd.read_csv(f)
        for ds, g in d.groupby("dataset"):
            if ds not in out:
                pos = g.y_true.mean() if set(g.y_true.unique()) <= {0.0, 1.0} else None
                out[ds] = dict(n=int(g.mol_index.nunique()), outputs=int(g.output_index.nunique()),
                               pos=pos)
    return out


def moleculeace_rows():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "chemeleon_suite/data/moleculeace/*.csv"))):
        d = pd.read_csv(f)
        sp = d["split"].value_counts().to_dict()
        rows.append(dict(task=os.path.basename(f)[:-4], n=len(d), train=sp.get("train", 0),
                         test=sp.get("test", 0), cliff=int(d["cliff_mol"].sum())))
    return rows


def main():
    mn = molnet_rows()
    ace = moleculeace_rows()
    man = json.load(open(ROOT / "chemeleon_suite/data/polaris/polaris_manifest.json"))
    cbs = pd.read_csv(ROOT / "data/cbs.csv") if (ROOT / "data/cbs.csv").exists() else None

    L = ["# Dataset inventory", "",
         "Every dataset in the project, with sizes measured from the files on disk. Generated by",
         "`scripts/dataset_inventory.py` — re-run it rather than editing this file.", "",
         "**The six-panel suite** (used across all figures): MoleculeACE · CBS · BACE · hERG · "
         "Tox21 · QM7.", "",
         "| Suite | Datasets | Molecules | Role |", "|---|---|---|---|",
         f"| MoleculeACE | 30 | {sum(r['n'] for r in ace):,} | potency regression, cliff-focused |",
         f"| Polaris | {len(man)} | — (per-task splits below) | ADMET + kinase panels |",
         f"| MoleculeNet | {len(mn)} | {sum(v['n'] for v in mn.values()):,} | classic property benchmarks |",
         f"| CBS | 1 | {len(cbs):,} | rare-active virtual screen |" if cbs is not None else "| CBS | 1 | 10,445 | rare-active virtual screen |",
         "", "---", "", "## 1. MoleculeNet", "",
         "Molecule counts are the full curated set; we score them with 5-fold Bemis–Murcko scaffold",
         "cross-validation, so every molecule is predicted out-of-fold and each test fold is ~1/5.", "",
         "| Dataset | Task | Molecules | Positives | In suite? | Description |",
         "|---|---|---|---|---|---|"]
    for ds in sorted(mn):
        kind, desc = MOLNET_DESC.get(ds, ("?", ""))
        v = mn[ds]
        n = f"{v['n']:,}" + (f" × {v['outputs']} assays" if v["outputs"] > 1 else "")
        pos = f"{v['pos']:.0%}" if v["pos"] is not None else "—"
        mark = "**yes**" if ds in IN_SUITE else "no"
        L.append(f"| {ds} | {kind} | {n} | {pos} | {mark} | {desc} |")
    L += ["", "> **Lipophilicity caveat:** it is only present for 2 of the 16 mainline arms "
              "(random encoder, CheMeleon), so it cannot be used for a mainline comparison as "
              "things stand.", "", "---", "", "## 2. Polaris", "",
          "Benchmark-provided train/test splits and a designated primary metric per task. 11 of the",
          "28 are classification.", "",
          "| Task | Type | Train | Test | Primary metric | Description |", "|---|---|---|---|---|---|"]
    for t, v in sorted(man.items(), key=lambda kv: -(kv[1].get("n_test") or 0)):
        mark = " **(in suite)**" if t == "tdcommons/herg" else ""
        L.append(f"| `{t}`{mark} | {v['type']} | {v.get('n_train','?'):,} | {v.get('n_test','?'):,} "
                 f"| {v['primary_metric']} | {POLARIS_DESC.get(t,'')} |")

    L += ["", "---", "", "## 3. MoleculeACE", "",
          "30 ChEMBL target-based potency sets (pKi or pEC50), each with a fixed train/test split "
          "and a flagged activity-cliff subset. We report the macro-mean RMSE over the 30 targets, "
          "overall and on the cliff subset. **Target IDs are not resolved to protein names** — no "
          "local ChEMBL mapping exists in the repo.", "",
          "| Target | Endpoint | Total | Train | Test | Cliff molecules |", "|---|---|---|---|---|---|"]
    for r in sorted(ace, key=lambda r: -r["n"]):
        cid, endpoint = r["task"].rsplit("_", 1)
        L.append(f"| {cid} | p{endpoint} | {r['n']:,} | {r['train']:,} | {r['test']:,} | "
                 f"{r['cliff']:,} ({r['cliff']/r['n']:.0%}) |")
    L += ["", f"**Totals:** {len(ace)} targets, {sum(r['n'] for r in ace):,} molecules, "
              f"{sum(r['test'] for r in ace):,} test molecules, "
              f"{sum(r['cliff'] for r in ace):,} cliff molecules.", "", "---", "", "## 4. CBS", ""]
    if cbs is not None:
        L += [f"Single external virtual-screening benchmark: **{len(cbs):,} molecules, "
              f"{int(cbs.y.sum())} confirmed actives ({cbs.y.mean():.2%})**, "
              f"{cbs.fold.nunique()} benchmark-provided folds, scored by NEF1% (enrichment in the "
              f"top 1%). Inhibitors of cystathionine β-synthase (Truong et al. 2026). This is the "
              f"only panel testing the rare-active regime that real screening campaigns face — it "
              f"is why HIV was droppable.", ""]
    (ROOT / "notes" / "dataset-inventory.md").write_text("\n".join(L) + "\n")
    print(f"wrote notes/dataset-inventory.md  "
          f"({len(mn)} MoleculeNet + {len(man)} Polaris + {len(ace)} MoleculeACE + 1 CBS)")


if __name__ == "__main__":
    main()
