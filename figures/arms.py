"""Single source of truth for the CLIMB paper figures: model nomenclature, colours, and the
canonical 6-panel benchmark suite.

Imported by BOTH the aggregation script (scripts/six_panel_aggregate.py, stdlib only) and every
figure script under figures/. Nothing else in the repo defines an arm label or an arm colour.

Nomenclature (fixed 2026-08-16, user decision) -- use these strings verbatim in every figure:
    ECFP / ECFP+desc                     XGBoost anchors
    supervised, <readout>                supervised multi-task-regression pretraining
    unsupervised                         MLM pretraining
    unsup->sup, <readout>                MLM then supervised (canonical short form)
    random encoder                       untrained encoder, frozen (probe only)
    no pretrain, end2end                 transformer trained from scratch on the downstream task
    CheMeleon                            external comparator (curiosity only -- NOT in ablation/scaling)

Colour scheme: orange = XGBoost anchors, red = supervised, blue = unsupervised,
green = unsup->supervised, purple = CheMeleon, grey = end2end, black = random encoder.
Shades within a family run dark (headline recipe) -> light (peripheral recipe).
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------------
# colour families
# --------------------------------------------------------------------------------------------
# Colour-vision-deficiency note: the requested semantic scheme (orange / red / green) is the one
# pairing that deuteranopes cannot separate by hue, so the hues are nudged to keep the intended
# reading while staying separable: "red" is a magenta-leaning crimson and "green" a bluish
# (teal) green, both anchored on the Okabe-Ito palette. Every family additionally spans a
# distinct lightness band, so the figures also survive greyscale printing.
# Muted, print-friendly versions of the same semantic families (user request 2026-08-16). Hues are
# unchanged so the scheme still reads orange / red / blue / green / purple / grey / black; only the
# saturation is pulled back. Every figure takes its colours from here, so they are consistent by
# construction — never hard-code a colour in a figure script.
FAMILY_COLORS = {
    "anchor":   "#C8912F",   # orange   (muted amber)
    "sup":      "#A3455E",   # red      (muted crimson, magenta-leaning for CVD)
    "unsup":    "#3F6E9C",   # blue     (muted)
    "u2s":      "#3D8073",   # green    (muted teal-green, bluish for CVD)
    "chemeleon": "#7E6BA8",  # purple   (muted violet)
    "s2u":      "#6B6494",   # slate/indigo = supervised -> unsupervised (forgetting mirror)
    "e2e":      "#8A8A8A",   # grey
    "random":   "#2B2B2B",   # near-black
}

# shade ladders (dark -> light) used for scaling/ablation plots that need more than one member
SHADES = {
    "anchor": ["#C8912F", "#8A5F1B", "#E0BC80"],          # ECFP+desc amber, ECFP dark amber
    "sup":    ["#A3455E", "#B96A7E", "#CB8C9C", "#DBAEB9", "#E9CFD6"],
    "unsup":  ["#3F6E9C", "#6B93B8", "#9AB6D0", "#C3D5E4"],
    "u2s":    ["#2A5C50", "#3D8073", "#5E9C90", "#84B7AD", "#ABD0C9"],
    "chemeleon": ["#7E6BA8", "#A093C0", "#C4BCD8"],
    "s2u":    ["#6B6494", "#8F89B2", "#B7B3CE"],
    "e2e":    ["#8A8A8A", "#A8A8A8", "#C6C6C6"],
    "random": ["#2B2B2B", "#555555", "#808080"],
}

# --------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------
# key            -> canonical key used in every CSV under figure_data/six_panel/ and every figure
# label          -> the ONLY string that may appear in a figure
# short          -> compact label for cramped axes
# family         -> colour family
# color          -> exact colour
# probe          -> how the downstream head sees the model ("frozen" / "e2e" / "xgb")
# in_ablation    -> include in ablation/scaling figures (CheMeleon is excluded by decision)
# src            -> where the raw numbers live, per suite
#                   mace: figure_data/chemeleon_suite/moleculeace/<src>/results.csv, AND the
#                         polaris dir of the same name. A bare string is expanded to
#                         <src>, <src>_s1, <src>_s2; pass an explicit LIST when an arm's replicate
#                         dirs do not follow that convention (s2u_dense is _s0/_s1/_s2, the
#                         controls are _00/_01/_02). Listing them beats renaming real result dirs
#                         to fit the resolver.
#                   mol : LIST of pretraining-seed dirs, each
#                         figure_data/climb_v2_phase2/<dir>/moleculenet_cv/moleculenet_summary.csv
#                         (CLIMB arms: <base>, <base>_s1, <base>_s2; controls: _00/_01/_02).
#                         All of them are pooled -- 3 pretraining seeds x 3 head seeds x 5 folds.
#                   cbs : arm column of experiment_cbs/cbs_nef1_summary.csv
ARMS = {
    # ---- XGBoost anchors (orange) -----------------------------------------------------------
    "ecfp": dict(
        label="ECFP", short="ECFP", family="anchor", color=SHADES["anchor"][1], probe="xgb",
        in_ablation=True,
        src=dict(mace="ecfp4", mol=["ecfp4_anchor"], cbs="ecfp4")),
    "ecfp_desc": dict(
        label="ECFP+desc", short="ECFP+desc", family="anchor", color=SHADES["anchor"][0], probe="xgb",
        in_ablation=True,
        src=dict(mace="fp_desc", mol=["fp_desc_anchor"], cbs="fp_desc")),

    # ---- supervised pretraining (red) -------------------------------------------------------
    "sup_dense": dict(
        label="supervised, dense", short="sup, dense", family="sup", color=SHADES["sup"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_dense_8M", mol=["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"], cbs="sup_only:dense")),
    "sup_dense_sparse": dict(
        label="supervised, dense+sparse", short="sup, dense+sparse", family="sup", color=SHADES["sup"][1],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_dense_plus_sparse_8M", mol=["skip_dense_plus_sparse_8M", "skip_dense_plus_sparse_8M_s1", "skip_dense_plus_sparse_8M_s2"],
                 cbs="sup_only:dense_plus_sparse")),
    "sup_mixed": dict(
        label="supervised, mixed", short="sup, mixed", family="sup", color=SHADES["sup"][2],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_mixed_8M", mol=["skip_mixed_8M", "skip_mixed_8M_s1", "skip_mixed_8M_s2"], cbs="sup_only:mixed")),
    "sup_sparse": dict(
        label="supervised, sparse", short="sup, sparse", family="sup", color=SHADES["sup"][3],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_sparse_all_8M", mol=["skip_sparse_all_8M", "skip_sparse_all_8M_s1", "skip_sparse_all_8M_s2"], cbs="sup_only:sparse_all")),
    "sup_minimol": dict(
        label="supervised, MiniMol tasks", short="sup, MiniMol", family="sup", color=SHADES["sup"][4],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_minimol_full_8M", mol=["skip_minimol_full_8M", "skip_minimol_full_8M_s1", "skip_minimol_full_8M_s2"], cbs="sup_only:minimol_full")),

    # ---- unsupervised pretraining (blue) ----------------------------------------------------
    "unsup": dict(
        label="unsupervised", short="unsup", family="unsup", color=SHADES["unsup"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="unsup_8M", mol=["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"], cbs="unsup_only")),

    # ---- unsupervised -> supervised (green) -------------------------------------------------
    "u2s_dense": dict(
        label="unsup→sup, dense", short="unsup→sup, dense", family="u2s", color=SHADES["u2s"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_dense_from8M", mol=["u2s_dense_from8M", "u2s_dense_from8M_s1", "u2s_dense_from8M_s2"], cbs="unsup2sup:dense")),
    "u2s_dense_sparse": dict(
        label="unsup→sup, dense+sparse", short="unsup→sup, dense+sparse", family="u2s",
        color=SHADES["u2s"][1], probe="frozen", in_ablation=True,
        src=dict(mace="u2s_dense_plus_sparse_from8M", mol=["u2s_dense_plus_sparse_from8M", "u2s_dense_plus_sparse_from8M_s1", "u2s_dense_plus_sparse_from8M_s2"],
                 cbs="unsup2sup:dense_plus_sparse")),
    "u2s_mixed": dict(
        label="unsup→sup, mixed", short="unsup→sup, mixed", family="u2s", color=SHADES["u2s"][2],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_mixed_from8M", mol=["u2s_mixed_from8M", "u2s_mixed_from8M_s1", "u2s_mixed_from8M_s2"], cbs="unsup2sup:mixed")),
    "u2s_sparse": dict(
        label="unsup→sup, sparse", short="unsup→sup, sparse", family="u2s", color=SHADES["u2s"][3],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_sparse_all_from8M", mol=["u2s_sparse_all_from8M", "u2s_sparse_all_from8M_s1", "u2s_sparse_all_from8M_s2"], cbs="unsup2sup:sparse_all")),
    "u2s_minimol": dict(
        label="unsup→sup, MiniMol tasks", short="unsup→sup, MiniMol", family="u2s",
        color=SHADES["u2s"][4], probe="frozen", in_ablation=True,
        src=dict(mace="u2s_minimol_full_from8M", mol=["u2s_minimol_full_from8M", "u2s_minimol_full_from8M_s1", "u2s_minimol_full_from8M_s2"],
                 cbs="unsup2sup:minimol_full")),

    # ---- supervised -> unsupervised (catastrophic-forgetting mirror) -------------------------
    # Mirror of unsup -> supervised: 8M supervised MTR, then a 2M MLM continuation. Isolates
    # whether the MLM phase forgets the supervised descriptor signal. Results land as
    # s2u_dense_from8M_s{0,1,2} (GPU run launched 2026-08-16); the entry is here so the arm is
    # picked up automatically the moment they appear. Colour: its own slate/indigo family (user
    # decision 2026-08-16) -- deliberately away from both the blue (unsupervised) and green
    # (unsup->supervised) families so the two mirror recipes never read as the same thing.
    "s2u_dense": dict(
        label="sup→unsup, dense", short="sup→unsup, dense", family="s2u", color="#5B4E8C",
        probe="frozen", in_ablation=True,
        src=dict(mace=["s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2"],
                 mol=["s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2"],
                 cbs="sup2unsup:dense")),

    # ---- controls ---------------------------------------------------------------------------
    "random_encoder": dict(
        label="random encoder", short="random enc.", family="random", color=FAMILY_COLORS["random"],
        probe="frozen", in_ablation=True,
        # MoleculeACE spelled out: the controls' replicates are _00/_01/_02, not <base>/_s1/_s2,
        # so the default resolver would find only the first dir and leave this arm at 1 seed
        # while every CLIMB arm has 3 (audit check 3). The _01/_02 dirs landed 2026-08-18.
        src=dict(mace=["random_baseline_00", "random_baseline_01", "random_baseline_02"],
                 mol=["random_baseline_00", "random_baseline_01", "random_baseline_02"], cbs="no_pretrain")),
    "e2e_no_pretrain": dict(
        label="no pretrain, end2end", short="no pretrain, e2e", family="e2e", color=FAMILY_COLORS["e2e"],
        probe="e2e", in_ablation=True,
        src=dict(mace="no_pretrain_e2e_e2e", mol=["e2e_random_00", "e2e_random_01", "e2e_random_02"], cbs="no_pretrain_e2e")),

    # ---- external comparator (curiosity only) -----------------------------------------------
    # SPLIT 2026-08-17 (was a single "chemeleon" arm labelled "end2end" but sourced entirely from
    # chemeleon_FROZEN — that mislabel is what put the frozen arm's broken QM7 value, 268.8, on the
    # e2e comparison in Fig A2). The two probes are now separate arms, each internally consistent:
    # never mix them in one series.
    #   e2e    = native chemprop D-MPNN from the CheMeleon foundation. 3 pretraining seeds.
    #            QM7 is healthy here (198.7 / 199.9 / 199.9, fold sd ~4).
    #   frozen = CheMeleon fingerprint + MLP probe. ONE seed.
    #            QM7 is genuinely broken here (mean 281, fold2 = 434 vs 212 elsewhere, i.e. worse
    #            than predicting the training mean, sigma = 228.7). Not a harness bug: predictions
    #            are correctly centred/scaled (pred mean -1540 vs true -1531, sd 229 vs 223), so the
    #            frozen embedding simply fails to encode atomization energy on some scaffold folds.
    #            Report it with the fold spread visible; do NOT quote it as "CheMeleon on QM7".
    # MoleculeACE: a NATIVE CheMeleon e2e run landed 2026-08-17 (chemeleon_e2e_gaps.py; 30 targets
    # x 3 eval seeds, chemprop_from_foundation) and supersedes Burns' published point estimates in
    # chemeleon_suite/reference/reference_long.csv -- ours has per-molecule OOF and can be
    # bootstrapped. hERG (polaris/chemeleon_e2e) is still in flight; that cell reads n/a until it
    # lands, at which point NO code change is needed (the same `mace` key names the polaris dir).
    "chemeleon_e2e": dict(
        label="CheMeleon (e2e)", short="CheMeleon e2e", family="chemeleon", color=SHADES["chemeleon"][0],
        probe="e2e", in_ablation=False,
        src=dict(mace="chemeleon_e2e", mol=["chemeleon_e2e", "chemeleon_e2e_s1", "chemeleon_e2e_s2"],
                 cbs="chemeleon_e2e")),
    "chemeleon_frozen": dict(
        label="CheMeleon (frozen)", short="CheMeleon frozen", family="chemeleon", color=SHADES["chemeleon"][1],
        probe="frozen", in_ablation=False,
        # QM7 REPLICATION (2026-08-18). chemeleon_frozen's original single run put a 427.7 fold in
        # the mean and shipped 268.8. Four probe runs settled it: the elevation is REAL (every run
        # has one or two badly degraded folds, under three different head-seed sets AND two
        # different scaffold partitions), but 268.8 overstates it. _s1/_s2 are EXTRA HEAD SEEDS on
        # PARTITION 0 -- CheMeleon is a fixed external encoder with no pretraining stage to
        # replicate, so do not read n_seeds=3 here as three pretrained models. They are pooled
        # because every other arm's QM7 cell is partition 0 and pooling more head seeds on the same
        # folds is strictly a better estimate of the same quantity.
        # chemeleon_frozen_part1/_part2 exist too and are DELIBERATELY EXCLUDED: they use fold
        # partitions 1 and 2, so pooling them would give this one bar a different estimand from the
        # other 17. They support the robustness statement in the caption only.
        # _s1/_s2 carry QM7 only, so BACE/Tox21 correctly stay at the base dir alone.
        src=dict(mace="chemeleon_frozen",
                 mol=["chemeleon_frozen", "chemeleon_frozen_s1", "chemeleon_frozen_s2"],
                 cbs="chemeleon_frozen")),
}

# display order: anchors, supervised, unsupervised, unsup->sup, controls, comparator
ARM_ORDER = list(ARMS)

# --------------------------------------------------------------------------------------------
# the canonical 6 panels
# --------------------------------------------------------------------------------------------
# key -> label, metric key, metric label, higher_better, task-type group
PANELS = {
    "MoleculeACE": dict(marker="^", label="MoleculeACE", metric="macro_rmse", metric_label="macro RMSE (30 targets)",
                        metric_short="macro RMSE",
                        higher_better=False, group="potency regression", n_tasks=30),
    "CBS":         dict(marker="D", label="CBS", metric="nef1", metric_label="NEF1%",
                        metric_short="NEF1%",
                        higher_better=True, group="rare-active screen", n_tasks=1),
    "BACE":        dict(marker="o", label="BACE", metric="roc_auc", metric_label="ROC-AUC",
                        metric_short="ROC-AUC",
                        higher_better=True, group="binding classification", n_tasks=1),
    # BBBP was dropped 2026-08-16: its whole field spans 1.8% of ROC-AUC and an UNTRAINED random
    # encoder ranks 7/16 there, so it separates featurization style rather than pretraining
    # quality (notes/bbbp-anchor-verification-2026-08-16.md). hERG replaces it: the most
    # discriminative classification task we have (random encoder ranks LAST, anchors first) and
    # the only small-data panel in the suite (523 train / 132 test, benchmark-provided split).
    # hERG was replaced by Ames on 2026-08-17. hERG has 132 test molecules, so its analytic
    # SE(AUC) is ~0.039 and its 0.215 field spans only ~5.6 SE -- it looked discriminative (the
    # untrained controls ranked last) but could not support that ranking. Ames has 1457 test
    # molecules: SE ~0.0116, field 0.141, i.e. ~12.2 SE of headroom, more than double hERG's, with
    # the untrained controls still in the bottom third (10-11 of 16). It is also a standard
    # regulatory tox endpoint. BBB-Martins was rejected (5.3 SE, and it re-introduces the BBB
    # endpoint already dropped as non-discriminative in its BBBP form).
    # display label only -- the panel KEY stays "Ames", which is what every CSV and every
    # figure_data path is written with; renaming the key would orphan them.
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
PANEL_ORDER = list(PANELS)

# Categorical colour per MoleculeNet task, for the similarity/transfer analysis figures (C2/D)
# where POINTS ARE TASKS, not arms. Muted hues drawn from the arm families; within those figures
# the legend defines them as tasks and no arm-coloured element shares the axes, so there is no
# semantic collision. The 6 canonical panels above keep their marker encoding and no colour.
# Per-EVAL-TASK colours for the figures that colour by task rather than by arm (fig_C2, fig_D).
# The canonical six come first and keep the hues the paper already uses for them; the three
# MoleculeNet tasks the canonical set drops (BBBP / ESOL / Lipophilicity) are retained so the
# pre-canonical figures still render while they are being migrated.
TASK_COLORS = {
    # --- canonical six -------------------------------------------------------------------
    "MoleculeACE":  "#3F6E9C",   # blue     (potency regression)
    "CBS":          "#3D8073",   # teal     (rare-active screen)
    "BACE":         "#A3455E",   # crimson  (binding classification)
    "Ames":         "#C8912F",   # amber    (mutagenicity)
    "Tox21":        "#8A8A8A",   # grey     (toxicity classification)
    "QM7":          "#6B6494",   # slate    (quantum regression)
    # --- pre-canonical MoleculeNet tasks, kept for the not-yet-migrated figures ------------
    "BBBP":         "#7E6BA8",   # violet
    "ESOL":         "#5B8FBF",   # light blue -- distinct from MoleculeACE, same family
    "HIV":          "#7E6BA8",   # violet
    "Lipophilicity": "#4FA08F",  # light teal -- distinct from CBS, same family
}


SYSTEM = {"anchor": "XGBoost", "chemeleon": "CheMeleon"}          # everything else is CLIMB


def system(arm_key: str) -> str:
    """Which model system the arm belongs to -- the first line of a two-line axis label."""
    return SYSTEM.get(ARMS[arm_key]["family"], "CLIMB") if arm_key in ARMS else ""


def label(arm_key: str) -> str:
    return ARMS[arm_key]["label"] if arm_key in ARMS else arm_key


def two_line_label(arm_key: str) -> str:
    """'XGBoost\nECFP+desc' / 'CLIMB\nsupervised, dense' / 'CheMeleon\nend2end'."""
    return f"{system(arm_key)}\n{label(arm_key)}"


def color(arm_key: str) -> str:
    return ARMS[arm_key]["color"] if arm_key in ARMS else "#999999"


def ablation_arms():
    """Arms allowed in ablation/scaling figures (CheMeleon excluded by decision)."""
    return [k for k, v in ARMS.items() if v["in_ablation"]]
