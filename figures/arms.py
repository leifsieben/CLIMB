from __future__ import annotations


FAMILY_COLORS = {
    "anchor":   "#C8912F",
    "sup":      "#A3455E",
    "unsup":    "#3F6E9C",
    "u2s":      "#3D8073",


    "literature": "#7E6BA8",
    "s2u":      "#6B6494",
    "e2e":      "#8A8A8A",
    "random":   "#2B2B2B",
}


SHADES = {


    "anchor": ["#C8912F", "#8A5F1B", "#E0BC80", "#E8B86A", "#4E340B"],


    "sup":    ["#A3455E", "#B96A7E", "#CB8C9C", "#DBAEB9", "#E9CFD6", "#6E2437", "#8A3A50"],
    "unsup":  ["#3F6E9C", "#6B93B8", "#9AB6D0", "#C3D5E4"],
    "u2s":    ["#2A5C50", "#3D8073", "#5E9C90", "#84B7AD", "#ABD0C9"],


    "literature": ["#111111", "#767676", "#BFBFBF"],
    "s2u":    ["#6B6494", "#8F89B2", "#B7B3CE"],
    "e2e":    ["#8A8A8A", "#A8A8A8", "#C6C6C6"],
    "random": ["#2B2B2B", "#555555", "#808080"],
}


ARMS = {

    "ecfp": dict(


        label="ECFP4", short="ECFP4", family="anchor", color=SHADES["anchor"][1], probe="xgb", pretrain_replicates=False,
        in_ablation=True,


        src=dict(mace="ecfp4", cbs_legacy_label="ecfp4",
                 mol=["ecfp4_anchor", "ecfp4_anchor_s1", "ecfp4_anchor_s2"])),
    "ecfp_desc": dict(
        label="ECFP4+desc", short="ECFP4+desc", family="anchor", color=SHADES["anchor"][0], probe="xgb", pretrain_replicates=False,
        in_ablation=True,
        src=dict(mace="fp_desc", cbs_legacy_label="fp_desc",
                 mol=["fp_desc_anchor", "fp_desc_anchor_s1", "fp_desc_anchor_s2"])),


    "unsup_100M": dict(
        label="unsupervised", short="unsup 100M", family="unsup", system="CLIMB 100M",
        color=SHADES["unsup"][1], probe="frozen", pretrain_replicates=False,
        in_ablation=False, unique_molecules=100_000_000,
        src=dict(mace="unsup_100M", mol=["unsup_100M"])),
    "sup_dense_96M": dict(
        label="supervised, desc", short="sup 96M", family="sup", system="CLIMB 96M FP",
        color=SHADES["sup"][2], probe="frozen", pretrain_replicates=False,
        in_ablation=False, in_ranking=False, unique_molecules=12_000_000,
        src=dict(mace="skip_dense_96M", mol=["skip_dense_96M"])),


    "unsup_8M_c124": dict(
        label="unsupervised", short="unsup 8M c124", family="unsup", system="CLIMB 8M",
        color=SHADES["unsup"][2], probe="frozen", pretrain_replicates=False,
        in_ablation=False, in_ranking=False, unique_molecules=8_000_000,
        src=dict(mace="unsup_8M_c124", mol=["unsup_8M_c124"])),


    "sup_dense_100M": dict(
        label="supervised, desc", short="sup 100M", family="sup",
        system="CLIMB 100M", color=SHADES["sup"][5], probe="frozen",
        pretrain_replicates=False, in_ablation=False,
        unique_molecules=100_000_000,
        src=dict(mace="skip_dense_100M_c124", mol=["skip_dense_100M_c124"])),


    "chemberta_mtr": dict(
        label="MTR pretraining", short="ChemBERTa", family="chemberta",
        color=SHADES["literature"][0], probe="frozen", pretrain_replicates=False,
        in_ablation=False, params_m=3.4, hf="DeepChem/ChemBERTa-77M-MTR",
        objective="supervised", pretrain_mols=77e6, inputs="desc",
        src=dict(mace="chemberta_mtr",
                 mol=["chemberta_mtr", "chemberta_mtr_s1", "chemberta_mtr_s2"])),


    "molformer_c3": dict(
        label="c3, linear attention", short="MoLFormer", family="molformer",
        color=SHADES["literature"][1], probe="frozen", pretrain_replicates=False,
        in_ablation=False, params_m=44.4, hf="DeepChem/MoLFormer-c3-1.1B",
        objective="unsupervised", pretrain_mols=1.1e9, inputs="SMILES",
        src=dict(mace="molformer_c3",
                 mol=["molformer_c3", "molformer_c3_s1", "molformer_c3_s2"])),


    "selfies_ted": dict(
        label="SELFIES, enc-dec", short="SELFIES-TED", family="selfies_ted",
        color=SHADES["literature"][2], probe="frozen", pretrain_replicates=False,
        in_ablation=False, params_m=358.1, hf="ibm-research/materials.selfies-ted",
        objective="unsupervised", pretrain_mols=1e9, inputs="SELFIES",
        src=dict(mace="selfies_ted",
                 mol=["selfies_ted", "selfies_ted_s1", "selfies_ted_s2"])),


    "sup_dense": dict(
        label="supervised, desc", short="sup, desc", family="sup", color=SHADES["sup"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_dense_8M", mol=["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"], cbs_legacy_label="sup_only:dense")),
    "sup_dense_sparse": dict(
        label="supervised, desc+sparse", short="sup, desc+sparse", family="sup", color=SHADES["sup"][1],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_dense_plus_sparse_8M", mol=["skip_dense_plus_sparse_8M", "skip_dense_plus_sparse_8M_s1", "skip_dense_plus_sparse_8M_s2"],
                 cbs_legacy_label="sup_only:dense_plus_sparse")),
    "sup_sparse": dict(
        label="supervised, sparse", short="sup, sparse", family="sup", color=SHADES["sup"][3],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_sparse_all_8M", mol=["skip_sparse_all_8M", "skip_sparse_all_8M_s1", "skip_sparse_all_8M_s2"], cbs_legacy_label="sup_only:sparse_all")),


    "unsup": dict(
        label="unsupervised", short="unsup", family="unsup", color=SHADES["unsup"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="unsup_8M", mol=["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"], cbs_legacy_label="unsup_only")),


    "u2s_dense": dict(
        label="unsup→sup, desc", short="unsup→sup, desc", family="u2s", color=SHADES["u2s"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_dense_from8M", mol=["u2s_dense_from8M", "u2s_dense_from8M_s1", "u2s_dense_from8M_s2"], cbs_legacy_label="unsup2sup:dense")),
    "u2s_dense_sparse": dict(
        label="unsup→sup, desc+sparse", short="unsup→sup, desc+sparse", family="u2s",
        color=SHADES["u2s"][1], probe="frozen", in_ablation=True,
        src=dict(mace="u2s_dense_plus_sparse_from8M", mol=["u2s_dense_plus_sparse_from8M", "u2s_dense_plus_sparse_from8M_s1", "u2s_dense_plus_sparse_from8M_s2"],
                 cbs_legacy_label="unsup2sup:dense_plus_sparse")),
    "u2s_sparse": dict(
        label="unsup→sup, sparse", short="unsup→sup, sparse", family="u2s", color=SHADES["u2s"][3],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_sparse_all_from8M", mol=["u2s_sparse_all_from8M", "u2s_sparse_all_from8M_s1", "u2s_sparse_all_from8M_s2"], cbs_legacy_label="unsup2sup:sparse_all")),


    "s2u_dense": dict(
        label="sup→unsup, desc", short="sup→unsup, desc", family="s2u", color="#5B4E8C",
        probe="frozen", in_ablation=True,
        src=dict(mace=["s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2"],
                 mol=["s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2"],
                 cbs_legacy_label="sup2unsup:dense")),


    "unsup_e2e": dict(
        label="unsupervised, end2end", short="unsup, end2end", family="unsup",
        color=SHADES["unsup"][1], probe="e2e", in_ablation=False,


        suite_seed_axis="finetune",
        src=dict(mace="unsup_8M_e2e",
                 mol=["unsup_8M_e2e", "unsup_8M_e2e_s1", "unsup_8M_e2e_s2"],
                 cbs_legacy_label="unsup_8M_e2e")),
    "sup_dense_e2e": dict(
        label="supervised, desc end2end", short="sup, desc end2end", family="sup",
        color=SHADES["sup"][5], probe="e2e", in_ablation=False,


        suite_seed_axis="finetune",
        src=dict(mace="skip_dense_8M_e2e",
                 mol=["skip_dense_8M_e2e", "skip_dense_8M_e2e_s1", "skip_dense_8M_e2e_s2"],
                 cbs_legacy_label="skip_dense_8M_e2e")),


    "unsup_xgb": dict(
        label="unsupervised, XGBoost probe", short="unsup, XGBoost", family="unsup",
        color=SHADES["unsup"][2], probe="xgb", pretrain_replicates=False, in_ablation=False,


        in_ranking=True,


        src=dict(mace="unsup_8M__xgb",
                 mol=["unsup_8M__xgb", "unsup_8M__xgb_s1", "unsup_8M__xgb_s2"])),
    "sup_dense_xgb": dict(
        label="supervised, desc XGBoost probe", short="sup, desc XGBoost", family="sup",
        color=SHADES["sup"][6], probe="xgb", pretrain_replicates=False, in_ablation=False,


        in_ranking=True,


        src=dict(mace="skip_dense_8M__xgb",
                 mol=["skip_dense_8M__xgb", "skip_dense_8M__xgb_s1",
                      "skip_dense_8M__xgb_s2"])),


    "random_encoder": dict(


        label="no pretrain, random", short="no pretrain, random", family="random", color=FAMILY_COLORS["random"],
        probe="frozen", in_ablation=True,


        src=dict(mace=["random_baseline_00", "random_baseline_01", "random_baseline_02"],
                 mol=["random_baseline_00", "random_baseline_01", "random_baseline_02"], cbs_legacy_label="no_pretrain")),
    "e2e_no_pretrain": dict(
        label="no pretrain, end2end", short="no pretrain, end2end", family="e2e", color=FAMILY_COLORS["e2e"],
        probe="e2e", in_ablation=True,
        src=dict(mace="no_pretrain_e2e_e2e", mol=["e2e_random_00", "e2e_random_01", "e2e_random_02"], cbs_legacy_label="no_pretrain_e2e")),


}

def _wong_trained():

    import json as _json
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parent.parent
    out = set()
    for k, v in ARMS.items():
        for d in (v.get("src", {}).get("mol") or [])[:1]:
            f = root / "figure_data" / "climb_v2_phase2" / d / "metadata.json"
            if not f.exists():
                continue
            fams = _json.loads(f.read_text()).get("supervised_families") or []
            if any("WONG" in str(x).upper() for x in fams):
                out.add(k)
    return out


RETIRED = _wong_trained()
_missing = RETIRED - set(ARMS)
assert not _missing, f"arms.py: RETIRED names an arm that does not exist: {_missing}"


ARM_ORDER = [a for a in ARMS if a not in RETIRED]


def assert_not_retired(names, where):

    bad = sorted(set(names) & RETIRED)
    assert not bad, (
        f"{where} names retired arm(s) {bad}: pretraining includes the Wong benchmark target.")


PANELS = {
    "MoleculeACE": dict(marker="^", label="MoleculeACE", metric="macro_rmse", metric_label="macro RMSE (30 targets)",
                        metric_short="macro RMSE",
                        higher_better=False, group="potency regression", n_tasks=30),


    "HIV":         dict(marker="D", label="HIV", metric="nef1", metric_label="NEF1%",
                        metric_short="NEF1%",
                        higher_better=True, group="rare-active screen", n_tasks=1),
    "CBS":         dict(marker="D", label="CBS", metric="nef1", metric_label="NEF1%",
                        metric_short="NEF1%",
                        higher_better=True, group="rare-active screen", n_tasks=1),
    "BACE":        dict(marker="o", label="BACE", metric="roc_auc", metric_label="ROC-AUC",
                        metric_short="ROC-AUC",
                        higher_better=True, group="binding classification", n_tasks=1),


    "Ames":        dict(marker="s", label="Ames Mutagenicity", metric="roc_auc", metric_label="ROC-AUC",
                        metric_short="ROC-AUC",
                        higher_better=True, group="mutagenicity", n_tasks=1,
                        source="polaris", polaris_task="tdcommons/ames"),
    "Tox21":       dict(marker="v", label="Tox21", metric="roc_auc", metric_label="mean ROC-AUC (12 assays)",
                        metric_short="ROC-AUC (12)",
                        higher_better=True, group="toxicity classification", n_tasks=12),
    "QM7":         dict(marker="P", label="QM7", metric="rmse", metric_label="RMSE (kcal/mol)",
                        metric_short="RMSE",
                        higher_better=False, group="quantum regression", n_tasks=1),
}


PANEL_ORDER = ["MoleculeACE", "HIV", "BACE", "QM7", "Tox21", "Ames"]


TASK_COLORS = {

    "MoleculeACE":  "#3F6E9C",
    "HIV":          "#3D8073",
    "CBS":          "#4FA08F",
    "BACE":         "#A3455E",
    "Ames":         "#C8912F",
    "Tox21":        "#8A8A8A",
    "QM7":          "#6B6494",

    "BBBP":         "#7E6BA8",
    "ESOL":         "#5B8FBF",
    "Lipophilicity": "#4FA08F",
}


SYSTEM = {"anchor": "XGBoost",
          "chemberta": "ChemBERTa-2", "molformer": "MoLFormer", "selfies_ted": "SELFIES-TED"}


def system(arm_key: str) -> str:

    if arm_key not in ARMS:
        return ""
    return ARMS[arm_key].get("system") or SYSTEM.get(ARMS[arm_key]["family"], "CLIMB")


def _mols(n):
    """1.1e9 -> '1.1B', 7.7e7 -> '77M'. Trailing '.0' dropped so 77M never prints as 77.0M."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            v = n / div
            return f"{v:.1f}".rstrip("0").rstrip(".") + suf
    return str(int(n))


def label(arm_key: str) -> str:

    if arm_key not in ARMS:
        return arm_key
    a = ARMS[arm_key]
    if a.get("objective"):
        mols = f"{_mols(a['pretrain_mols'])} " if a.get("pretrain_mols") else ""
        return f"{a['objective']}, {mols}{a['inputs']}"
    return a["label"]


def two_line_label(arm_key: str) -> str:

    return f"{system(arm_key)}\n{label(arm_key)}"


def color(arm_key: str) -> str:
    return ARMS[arm_key]["color"] if arm_key in ARMS else "#999999"


E2E_PAIRS = [p for p in [("unsup", "unsup_e2e"),
                         ("sup_dense", "sup_dense_e2e")]
             if not (set(p) & RETIRED)]


def series_label(frozen_arm: str) -> str:

    return ARMS[frozen_arm]["label"]


def ablation_arms():
    """Arms used in the ablation and scaling figures."""
    return [k for k, v in ARMS.items() if v["in_ablation"]]


LIFT_FLOOR_LABEL = ARMS["random_encoder"]["label"]
LIFT_YLABEL = f"Lift over {LIFT_FLOOR_LABEL} (%)"
