"""Single source of truth for the CLIMB paper figures: model nomenclature, colours, and the
canonical 6-panel benchmark suite.

Imported by BOTH the aggregation script (scripts/six_panel_aggregate.py, stdlib only) and every
figure script under figures/. Nothing else in the repo defines an arm label or an arm colour.

Nomenclature (fixed 2026-08-16, user decision) -- use these strings verbatim in every figure:
    end2end                              ALWAYS spelled out in anything a reader sees (user
                                         2026-08-19: "e2e that is not commonly understood").
                                         `probe="e2e"`, family keys and run-dir names keep the
                                         short form -- those are internal identifiers, not labels.
    ECFP4 / ECFP4+desc                   XGBoost anchors, Morgan r=2 BINARY, chirality on
    R3FP / R3FP+desc                     XGBoost anchors, Morgan r=3 COUNTS, chirality on
                                         (label fixed 2026-08-19, user: "let's use this label
                                         consistently from now on"). The RUN DIRS keep their own
                                         names -- ecfp4_r3c / fp_desc_r3c -- and the featurizer is
                                         still FP_VARIANT=morgan_r3_counts. Only what a reader
                                         SEES is R3FP; renaming result dirs to match a label is how
                                         provenance gets lost.
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
    # The three EXTERNAL literature CLMs. They take the retired CheMeleon hue rather than a new
    # one: it is the same semantic slot -- a pretrained model from outside this lab -- and the
    # violet band is free now. Giving them CLIMB blues/reds instead would imply they are our arms.
    "literature": "#7E6BA8",
    "s2u":      "#6B6494",   # slate/indigo = supervised -> unsupervised (forgetting mirror)
    "e2e":      "#8A8A8A",   # grey
    "random":   "#2B2B2B",   # near-black
}

# shade ladders (dark -> light) used for scaling/ablation plots that need more than one member
SHADES = {
    # [0] ECFP4+desc, [1] ECFP4, [2] spare, [3] r3-counts+desc, [4] r3-counts. FOUR XGBoost arms
    # as of 2026-08-19 (user: "I'd like to have our r3-fp + descriptors as the third XGBoost model.
    # Let's actually include r3-fp too"), so the anchor family needs four separable ambers rather
    # than two. The four DRAWN rungs span roughly even lightness steps -- [4] #4E340B, [1] #8A5F1B,
    # [0] #C8912F, [3] #E8B86A -- so the family also survives greyscale printing.
    #
    # [3] was #EBD3A6 for an hour and was too pale to see: r3c+desc sorts to the TOP row of fig_A1,
    # which is drawn on a light-grey banded background, and its marker and whisker nearly vanished
    # against it. A shade that disappears on the arm most likely to be quoted is not a shade.
    "anchor": ["#C8912F", "#8A5F1B", "#E0BC80", "#E8B86A", "#4E340B"],
    # Six, not five. The sixth is APPENDED rather than inserted so no existing
    # SHADES["sup"][i] reference moves; it is darker than [0] and sits out of the ladder's
    # dark->light order on purpose, because it belongs to the end2end arm rather than to the
    # frozen ladder the first five encode.
    "sup":    ["#A3455E", "#B96A7E", "#CB8C9C", "#DBAEB9", "#E9CFD6", "#6E2437", "#8A3A50"],
    "unsup":  ["#3F6E9C", "#6B93B8", "#9AB6D0", "#C3D5E4"],
    "u2s":    ["#2A5C50", "#3D8073", "#5E9C90", "#84B7AD", "#ABD0C9"],
    "chemeleon": ["#7E6BA8", "#A093C0", "#C4BCD8"],
    # [0] ChemBERTa-2, [1] MoLFormer, [2] SELFIES-TED. Three separable violets, dark -> light,
    # so the external block reads as one family in fig_A and still survives greyscale.
    "literature": ["#5C4A85", "#8B7BB5", "#B9AED5"],
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
# pretrain_replicates -> does this arm HAVE a pretraining stage that can be replicated? Default
#                   True. False for the XGBoost anchors (a fixed classical featurization) and for
#                   CheMeleon (a frozen, externally-supplied encoder). Those arms have exactly ONE
#                   run dir on the suite tracks and that is a fact about the model, not missing
#                   compute -- their full model variance is head/eval-seed variance, and 3 eval
#                   seeds of it live INSIDE the single dir. Declared here rather than as a name
#                   list inside the audit, which is how audit check 11 spent a day failing on six
#                   arms that were complete.
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
# LABEL PUNCTUATION -- one rule, because these strings sit next to each other on a page and an
# inconsistent comma reads as a distinction that is not there (Leif 2026-08-20).
#
#   The comma separates the ENCODER from its qualifier:  "unsupervised, end2end"
#                                                        "no pretrain, random"
#   When the encoder's own name already carries a comma, the probe appends with a SPACE rather
#   than a third comma:                                  "supervised, desc"  ->
#                                                        "supervised, desc end2end"
#                                                        "supervised, desc XGBoost probe"
#
# Never parentheses (user 2026-08-19). The two sup_dense probe arms read "supervised desc,
# end2end" until 2026-08-20 -- comma after the feature set instead of after the objective -- which
# put two different spellings of one encoder in the same fig_A1 column.
#
# THE CHEMELEON LABELS OMIT "CheMeleon" AND THE CHEMELEON SHORTS INCLUDE IT. That is not an
# oversight in either direction, it is the two renderers wanting different things:
#   fig_A1 draws system() on one line and label() on the next, so a label carrying "CheMeleon"
#          printed "CheMeleon / CheMeleon, end2end" -- the model named twice on one tick.
#   fig_A2 draws `short` ALONE, and "end2end" alone is indistinguishable from the
#          "no pretrain, end2end" control sitting a few rows away.
# So label="end2end" and short="CheMeleon, end2end", and both are correct where they are used.
# scripts/audit_figure_consistency.py check 17 enforces all of this.
#
# `cbs_legacy_label` IS NOT A PATH, WHICH IS WHY IT IS NO LONGER CALLED `cbs`. It holds an arm's
# name in the DEPRECATED experiment_cbs/cbs_nef1_summary.csv, and exactly one consumer reads it
# (scripts/audit_six_panel_sources.py, which audits that deprecated summary). The CBS PANEL every
# figure draws is resolved by allsuites._cbs_value from the arm's `mol` DIR NAMES under the
# cbs_benchmark tree -- so an arm with no entry here still has a CBS cell, and an entry here does
# not give it one.
#
# Renamed 2026-08-20 after a field called `cbs` that was not the CBS source cost a queued box-hour:
# the three __xgb arms have no entry here, were read as "no CBS", and were scheduled for a run whose
# results were already on disk. Same failure family as a verified.json writing "featurizer": "ecfp4"
# for three different featurizations -- a name that answers confidently and stops you looking.
ARMS = {
    # ---- XGBoost anchors (orange) -----------------------------------------------------------
    "ecfp": dict(
        # "ECFP4" IS correct again (user 2026-08-19: "let's please run ECFP4 with stereochemistry
        # and then out of curiosity let's also get your current version... what you came up with is
        # non-orthodox"). FP_VARIANT=ecfp4_stereo is the headline anchor -- Morgan r=2, BINARY,
        # chirality ON -- which is an ECFP4 in every respect that the name denotes, plus the stereo
        # flag RDKit leaves off by default. The label went to "Morgan" for ~1h while the default was
        # a radius-3 COUNT vector, where "ECFP4" would have been a factual error; it is not one now.
        # The r=3 count version is FP_VARIANT=morgan_r3_counts and is reported as a variant.
        label="ECFP4", short="ECFP4", family="anchor", color=SHADES["anchor"][1], probe="xgb", pretrain_replicates=False,
        in_ablation=True,
        # THREE MODEL SEEDS as of 2026-08-19 (peer session, commit 3c52686). The anchors used to
        # be a single run, so sd_seeds was literally 0.0 on the arms that beat the CLMs. Each
        # replicate is a full 3-head-seed ENSEMBLE on a disjoint head-seed triple ({3,4,5} and
        # {6,7,8}), i.e. the SAME estimator the mainline plots -- exposing the existing _cell rows
        # instead would have swapped in the pre-ensemble estimator and moved every anchor bar.
        # MoleculeACE and Polaris are still one run each: those tracks were not replicated.
        src=dict(mace="ecfp4", cbs_legacy_label="ecfp4",
                 mol=["ecfp4_anchor", "ecfp4_anchor_s1", "ecfp4_anchor_s2"])),
    "ecfp_desc": dict(
        label="ECFP4+desc", short="ECFP4+desc", family="anchor", color=SHADES["anchor"][0], probe="xgb", pretrain_replicates=False,
        in_ablation=True,
        src=dict(mace="fp_desc", cbs_legacy_label="fp_desc",
                 mol=["fp_desc_anchor", "fp_desc_anchor_s1", "fp_desc_anchor_s2"])),

    # ---------------------------------------------------------- the largest-corpus CLIMB runs --
    # Added 2026-08-26. These are SINGLE pretraining runs, so their replicate axis is the HEAD
    # SEED (3 seeds inside one directory), not three pretrainings -- `pretrain_replicates=False`,
    # the same convention the anchors and the literature CLMs use.
    #
    # unsup_100M is the largest CLIMB model on the axis that matters: 100M DISTINCT molecules from
    # the 124M corpus, under one epoch, no repetition. It is the only CLIMB arm that has seen more
    # unique chemistry than ChemBERTa-2's ~77M.
    #
    # skip_dense_96M is deliberately NOT its supervised counterpart. It is the largest supervised
    # run by COMPUTE (96M forward passes) but sees only 12M unique molecules -- eight epochs of the
    # small corpus -- and it scores WORSE than the 24M rung it repeats, 0.7748 against 0.7687 macro
    # RMSE on MoleculeACE, four times the compute for a worse number. (That comparison used to be
    # drawn against skip_dense_48M; that rung was dropped everywhere on 2026-08-26 for training on
    # 208 descriptors under a self-fit normalizer, so 24M is now the nearest valid rung below it.
    # The conclusion did not depend on which one: 96M is worse than every dense rung except 2M.)
    # Registered here so it can be quoted, and left out of fig_A's field: ranking a
    # repetition-saturated arm as "the largest supervised model" would say the opposite of what
    # fig_B now says. Its real counterpart is skip_dense_100M_c124, in flight.
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

    # skip_dense_100M_c124 lands here when the run finishes, as
    #     "sup_dense_100M": dict(label="supervised, desc", short="sup 100M", family="sup",
    #                            system="CLIMB 100M", color=SHADES["sup"][5], probe="frozen",
    #                            pretrain_replicates=False, in_ablation=False,
    #                            unique_molecules=100_000_000,
    #                            src=dict(mace="skip_dense_100M_c124",
    #                                     mol=["skip_dense_100M_c124"]))
    # -- same two-line shape as unsup_100M so the pair reads as one comparison: "CLIMB 100M" bold
    # on both, objective underneath. Written out here rather than left to be reinvented.

    # ------------------------------------------------------------ external literature CLMs ----
    # Three published chemical language models, added 2026-08-26 from the fig_A wave. All three
    # are FROZEN + MLP probe, featurized in ONE environment (scripts/figA_extract_all.sh) so no
    # arm in this group carries a different transformers version from its neighbours.
    #
    # REPLICATE AXIS IS THE HEAD SEED, not pretraining: each is a single released checkpoint, so
    # `pretrain_replicates=False` and the three dirs are disjoint head-seed triples
    # (42/117/709, 43/118/710, 44/119/711) -- the same convention the ECFP anchors use. See
    # notes/figA-seed-axis-is-not-uniform.md; "three seeds" means two different estimands on this
    # panel and the caption has to say so.
    #
    # THE NUMBER IN EACH NAME IS PRETRAINING DATA, NOT PARAMETERS, and the two orderings are
    # OPPOSITE. Measured from the checkpoints by the compute session: ChemBERTa-77M-MTR is 3.4M
    # parameters, MoLFormer-c3-1.1B is 44.4M, SELFIES-TED is 358.1M. So the names read
    # 77M < 1.1B while the models are 3.4M < 44.4M < 358.1M. Never print a name's number as a
    # size: "the 1.1B model loses to the 77M model" is a sentence someone would otherwise write,
    # and it would be backwards. `params_m` is here so a caption can use the real number.
    #
    # `mol` is a LITERAL list and must name all three dirs; `mace` is a STEM that _seed_dirs
    # expands. Getting that backwards has silently cost seeds four times in this repo.
    "chemberta_mtr": dict(
        label="MTR pretraining", short="ChemBERTa", family="chemberta",
        color=SHADES["literature"][0], probe="frozen", pretrain_replicates=False,
        in_ablation=False, params_m=3.4, hf="DeepChem/ChemBERTa-77M-MTR",
        src=dict(mace="chemberta_mtr",
                 mol=["chemberta_mtr", "chemberta_mtr_s1", "chemberta_mtr_s2"])),
    "molformer_c3": dict(
        label="c3, linear attention", short="MoLFormer", family="molformer",
        color=SHADES["literature"][1], probe="frozen", pretrain_replicates=False,
        in_ablation=False, params_m=44.4, hf="ibm-research/MoLFormer-XL-both-10pct",
        src=dict(mace="molformer_c3",
                 mol=["molformer_c3", "molformer_c3_s1", "molformer_c3_s2"])),
    "selfies_ted": dict(
        label="SELFIES, enc-dec", short="SELFIES-TED", family="selfies_ted",
        color=SHADES["literature"][2], probe="frozen", pretrain_replicates=False,
        in_ablation=False, params_m=358.1, hf="ibm-research/materials.selfies-ted",
        src=dict(mace="selfies_ted",
                 mol=["selfies_ted", "selfies_ted_s1", "selfies_ted_s2"])),

    # The r3-counts fingerprint as its OWN pair of arms rather than a replacement. Leif asked for
    # both generations reported (2026-08-19): the orthodox ECFP4+stereo is the headline anchor and
    # this is the variant, chosen by the peer session on a collision measure over 29,918 molecules
    # -- Morgan radius 3, COUNT vector, chirality on, 2048-d, FP_VARIANT=morgan_r3_counts.
    # Same XGBoost head, same splits, same seeds, same everything else, so the pair is a clean
    # read on what the fingerprint generation buys: counts help bare ECFP on the harder panels
    # (Tox21 +0.016, HIV +0.028, QM7 -0.96) and wash out once descriptors are present.
    # MoleculeACE and Ames are still missing here (peer session is running them); until they land
    # these two arms fall below fig_A1's 60-of-66 coverage floor and simply will not be drawn,
    # which is the correct behaviour rather than a partial ranking.
    "r3fp": dict(
        label="R3FP", short="R3FP", family="anchor", color=SHADES["anchor"][4],
        probe="xgb", pretrain_replicates=False, in_ablation=False,
        src=dict(mace="ecfp4_r3c", cbs_legacy_label="ecfp4_r3c",
                 mol=["ecfp4_anchor_r3c", "ecfp4_anchor_s1_r3c", "ecfp4_anchor_s2_r3c"])),
    "r3fp_desc": dict(
        label="R3FP+desc", short="R3FP+desc", family="anchor",
        color=SHADES["anchor"][3], probe="xgb", pretrain_replicates=False, in_ablation=False,
        src=dict(mace="fp_desc_r3c", cbs_legacy_label="fp_desc_r3c",
                 mol=["fp_desc_anchor_r3c", "fp_desc_anchor_s1_r3c", "fp_desc_anchor_s2_r3c"])),

    # ---- supervised pretraining (red) -------------------------------------------------------
    "sup_dense": dict(
        label="supervised, desc", short="sup, desc", family="sup", color=SHADES["sup"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_dense_8M", mol=["skip_dense_8M", "skip_dense_8M_s1", "skip_dense_8M_s2"], cbs_legacy_label="sup_only:dense")),
    "sup_dense_sparse": dict(
        label="supervised, desc+sparse", short="sup, desc+sparse", family="sup", color=SHADES["sup"][1],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_dense_plus_sparse_8M", mol=["skip_dense_plus_sparse_8M", "skip_dense_plus_sparse_8M_s1", "skip_dense_plus_sparse_8M_s2"],
                 cbs_legacy_label="sup_only:dense_plus_sparse")),
    "sup_mixed": dict(
        label="supervised, mixed", short="sup, mixed", family="sup", color=SHADES["sup"][2],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_mixed_8M", mol=["skip_mixed_8M", "skip_mixed_8M_s1", "skip_mixed_8M_s2"], cbs_legacy_label="sup_only:mixed")),
    "sup_sparse": dict(
        label="supervised, sparse", short="sup, sparse", family="sup", color=SHADES["sup"][3],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_sparse_all_8M", mol=["skip_sparse_all_8M", "skip_sparse_all_8M_s1", "skip_sparse_all_8M_s2"], cbs_legacy_label="sup_only:sparse_all")),
    "sup_minimol": dict(
        label="supervised, MiniMol tasks", short="sup, MiniMol", family="sup", color=SHADES["sup"][4],
        probe="frozen", in_ablation=True,
        src=dict(mace="skip_minimol_full_8M", mol=["skip_minimol_full_8M", "skip_minimol_full_8M_s1", "skip_minimol_full_8M_s2"], cbs_legacy_label="sup_only:minimol_full")),

    # ---- unsupervised pretraining (blue) ----------------------------------------------------
    "unsup": dict(
        label="unsupervised", short="unsup", family="unsup", color=SHADES["unsup"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="unsup_8M", mol=["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"], cbs_legacy_label="unsup_only")),

    # ---- unsupervised -> supervised (green) -------------------------------------------------
    "u2s_dense": dict(
        label="unsup→sup, desc", short="unsup→sup, desc", family="u2s", color=SHADES["u2s"][0],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_dense_from8M", mol=["u2s_dense_from8M", "u2s_dense_from8M_s1", "u2s_dense_from8M_s2"], cbs_legacy_label="unsup2sup:dense")),
    "u2s_dense_sparse": dict(
        label="unsup→sup, desc+sparse", short="unsup→sup, desc+sparse", family="u2s",
        color=SHADES["u2s"][1], probe="frozen", in_ablation=True,
        src=dict(mace="u2s_dense_plus_sparse_from8M", mol=["u2s_dense_plus_sparse_from8M", "u2s_dense_plus_sparse_from8M_s1", "u2s_dense_plus_sparse_from8M_s2"],
                 cbs_legacy_label="unsup2sup:dense_plus_sparse")),
    "u2s_mixed": dict(
        label="unsup→sup, mixed", short="unsup→sup, mixed", family="u2s", color=SHADES["u2s"][2],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_mixed_from8M", mol=["u2s_mixed_from8M", "u2s_mixed_from8M_s1", "u2s_mixed_from8M_s2"], cbs_legacy_label="unsup2sup:mixed")),
    "u2s_sparse": dict(
        label="unsup→sup, sparse", short="unsup→sup, sparse", family="u2s", color=SHADES["u2s"][3],
        probe="frozen", in_ablation=True,
        src=dict(mace="u2s_sparse_all_from8M", mol=["u2s_sparse_all_from8M", "u2s_sparse_all_from8M_s1", "u2s_sparse_all_from8M_s2"], cbs_legacy_label="unsup2sup:sparse_all")),
    "u2s_minimol": dict(
        label="unsup→sup, MiniMol tasks", short="unsup→sup, MiniMol", family="u2s",
        color=SHADES["u2s"][4], probe="frozen", in_ablation=True,
        src=dict(mace="u2s_minimol_full_from8M", mol=["u2s_minimol_full_from8M", "u2s_minimol_full_from8M_s1", "u2s_minimol_full_from8M_s2"],
                 cbs_legacy_label="unsup2sup:minimol_full")),

    # ---- supervised -> unsupervised (catastrophic-forgetting mirror) -------------------------
    # Mirror of unsup -> supervised: 8M supervised MTR, then a 2M MLM continuation. Isolates
    # whether the MLM phase forgets the supervised descriptor signal. Results land as
    # s2u_dense_from8M_s{0,1,2} (GPU run launched 2026-08-16); the entry is here so the arm is
    # picked up automatically the moment they appear. Colour: its own slate/indigo family (user
    # decision 2026-08-16) -- deliberately away from both the blue (unsupervised) and green
    # (unsup->supervised) families so the two mirror recipes never read as the same thing.
    "s2u_dense": dict(
        label="sup→unsup, desc", short="sup→unsup, desc", family="s2u", color="#5B4E8C",
        probe="frozen", in_ablation=True,
        src=dict(mace=["s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2"],
                 mol=["s2u_dense_from8M_s0", "s2u_dense_from8M_s1", "s2u_dense_from8M_s2"],
                 cbs_legacy_label="sup2unsup:dense")),

    # ---- CLIMB end-to-end (same encoders as `unsup` / `sup_dense`, fine-tuned) ----------------
    #
    # ADMITTED TO fig_A1 ONLY (user 2026-08-19: "add these two models to A1 (but not A2 please)").
    # fig_A2 selects from its own explicit MODELS list, so declaring them here cannot reach it.
    #
    # Coverage, verified 2026-08-20: MolNet 7 datasets x 3 pretraining seeds (42/42 cells),
    # MoleculeACE 30 tasks x 3 seeds, Polaris 28 tasks x 3 seeds, CBS 1 = 65 of 65 datasets, so
    # both clear fig_A1's >=60 admission threshold on real data rather than on a resolver quirk.
    #
    # THE SEED AXIS IS NOT UNIFORM ACROSS SUITES AND THE CAPTION MUST SAY SO. MolNet and CBS carry
    # three PRETRAINING seeds (CBS closed 2026-08-20); MoleculeACE and Polaris carry one pretrained
    # encoder with three fine-tune seeds. Every other CLIMB arm replicates on the pretraining axis everywhere. The
    # gap was left open deliberately rather than closed with ~700 fine-tunes, because the
    # measurement says it is small: on the two datasets where both axes exist, end-to-end
    # pretraining-seed SD is 0.44-0.97x the frozen arm's (scripts/pretrain_seed_variance.py), i.e.
    # fine-tuning DAMPS the initialisation it started from. Estimated MoleculeACE SD ~0.014.
    # That is an estimate from two datasets on a different suite, and it is stated as one.
    "unsup_e2e": dict(
        label="unsupervised, end2end", short="unsup, end2end", family="unsup",
        color=SHADES["unsup"][1], probe="e2e", in_ablation=False,
        # DECLARED, so audit checks 3 and 11 report it as a known asymmetry instead of a
        # failure, and so the reason travels with the arm rather than living in the
        # checkers. MolNet and CBS carry 3 pretraining seeds; MoleculeACE and Polaris
        # carry 1 pretrained encoder x 3 fine-tune seeds.
        suite_seed_axis="finetune",
        src=dict(mace="unsup_8M_e2e",
                 mol=["unsup_8M_e2e", "unsup_8M_e2e_s1", "unsup_8M_e2e_s2"],
                 cbs_legacy_label="unsup_8M_e2e")),
    "sup_dense_e2e": dict(
        label="supervised, desc end2end", short="sup, desc end2end", family="sup",
        color=SHADES["sup"][5], probe="e2e", in_ablation=False,
        # DECLARED, so audit checks 3 and 11 report it as a known asymmetry instead of a
        # failure, and so the reason travels with the arm rather than living in the
        # checkers. MolNet and CBS carry 3 pretraining seeds; MoleculeACE and Polaris
        # carry 1 pretrained encoder x 3 fine-tune seeds.
        suite_seed_axis="finetune",
        src=dict(mace="skip_dense_8M_e2e",
                 mol=["skip_dense_8M_e2e", "skip_dense_8M_e2e_s1", "skip_dense_8M_e2e_s2"],
                 cbs_legacy_label="skip_dense_8M_e2e")),

    # ---- XGBoost probe on a FROZEN embedding (the head-comparison arms) -----------------------
    #
    # Same three frozen representations that already appear above, read by gradient boosting
    # instead of an MLP. SI fig f is the two-point version of this and its finding is why these
    # belong in the ranking: the head CHANGES THE ORDER, so a ranking drawn with one head is a
    # statement about that head, not about the representations.
    #
    # ALL THREE, NOT JUST CheMeleon (user asked for CheMeleon; the other two come with it because
    # otherwise the figure is rigged). On MoleculeACE the head swap moves CheMeleon 0.826 -> 0.688
    # (much better) and moves the CLIMB arms the OTHER WAY, 0.778 -> 0.830 and 0.774 -> 0.813
    # (worse). Admitting only the comparator's better head while leaving CLIMB on its worse one
    # would manufacture a gap out of the probe. They cost nothing extra: identical coverage,
    # 63/65 datasets each.
    #
    # 63/65: MoleculeACE 30 + Polaris 28 + MolNet 5 of 6 + CBS 0. Clears the
    # >=60 admission floor.
    #
    # pretrain_replicates=False is a statement about THIS EXPERIMENT, not about the model. The
    # head comparison was run on ONE encoder per representation with three head seeds inside it,
    # so the pretraining-seed axis is not available here even though unsup and sup_dense have one
    # elsewhere. That is the same SHAPE as the anchors and CheMeleon -- one run dir, three head
    # seeds -- which is the group audit checks 3 and 11 compare them against.
    "chemeleon_frozen_xgb": dict(
        label="frozen, XGBoost probe", short="CheMeleon, frozen XGBoost", family="chemeleon",
        color=SHADES["chemeleon"][2], probe="xgb", pretrain_replicates=False, in_ablation=False,
        # The mol list is EXPLICIT and must name the replicate dirs. A bare string means "this
        # dir plus _s1/_s2 if they exist"; a list means "exactly these", which is what stopped the
        # aggregator seeing the replicates produced 2026-08-20 even after they were synced. The
        # list was written when only one dir existed and could not know about the other two.
        src=dict(mace="chemeleon_frozen__xgb",
                 mol=["chemeleon_frozen__xgb", "chemeleon_frozen__xgb_s1",
                      "chemeleon_frozen__xgb_s2"])),
    "unsup_xgb": dict(
        label="unsupervised, XGBoost probe", short="unsup, XGBoost", family="unsup",
        color=SHADES["unsup"][2], probe="xgb", pretrain_replicates=False, in_ablation=False,
        # NOT IN THE RANKING (user 2026-08-20). fig_A1 shows each representation at the head
        # that suits it -- CLIMB with its MLP probe, CheMeleon with XGBoost -- so the weaker
        # half of each pair is declared out. The arms stay defined because SI fig f is built
        # from the pair and needs both ends of every slope.
        in_ranking=True,
        # DECLARATION REMOVED 2026-08-22, all three of its reasons dead. It said this arm was
        # unranked (Leif ranked it), that no figure drew an interval for it (fig_A1 does), and
        # that its suite base came from a venv existing on no box -- which was true and is now
        # resolved: the whole trio was rebuilt in ONE environment, so the spread is a pretraining
        # measurement rather than partly an environment one. Measured cost of NOT doing that:
        # 0.235 max |delta| on 630 shared cells between the old base and an in-env rebuild.
        # Kept as a note rather than deleted silently, because this declaration expired on an
        # EVENT and nothing was watching -- see check 19, which tests the condition instead of
        # quoting it.
        # mol is a LITERAL list -- unlike `mace`, which _seed_dirs expands from a stem. A
        # single-element list here is the trap that has now bitten this repo four times:
        # it reads as "this arm has one dir" and _molnet silently drops anything not listed,
        # so replicate dirs land on disk and never reach the figure. Listed ahead of the
        # MolNet+CBS replicate run (2026-08-21); missing dirs are filtered until they exist.
        src=dict(mace="unsup_8M__xgb",
                 mol=["unsup_8M__xgb", "unsup_8M__xgb_s1", "unsup_8M__xgb_s2"])),
    "sup_dense_xgb": dict(
        label="supervised, desc XGBoost probe", short="sup, desc XGBoost", family="sup",
        color=SHADES["sup"][6], probe="xgb", pretrain_replicates=False, in_ablation=False,
        # NOT IN THE RANKING (user 2026-08-20). fig_A1 shows each representation at the head
        # that suits it -- CLIMB with its MLP probe, CheMeleon with XGBoost -- so the weaker
        # half of each pair is declared out. The arms stay defined because SI fig f is built
        # from the pair and needs both ends of every slope.
        in_ranking=True,
        # DECLARATION REMOVED 2026-08-22, all three of its reasons dead. It said this arm was
        # unranked (Leif ranked it), that no figure drew an interval for it (fig_A1 does), and
        # that its suite base came from a venv existing on no box -- which was true and is now
        # resolved: the whole trio was rebuilt in ONE environment, so the spread is a pretraining
        # measurement rather than partly an environment one. Measured cost of NOT doing that:
        # 0.235 max |delta| on 630 shared cells between the old base and an in-env rebuild.
        # Kept as a note rather than deleted silently, because this declaration expired on an
        # EVENT and nothing was watching -- see check 19, which tests the condition instead of
        # quoting it.
        # mol is a LITERAL list -- unlike `mace`, which _seed_dirs expands from a stem. A
        # single-element list here is the trap that has now bitten this repo four times:
        # it reads as "this arm has one dir" and _molnet silently drops anything not listed,
        # so replicate dirs land on disk and never reach the figure. Listed ahead of the
        # MolNet+CBS replicate run (2026-08-21); missing dirs are filtered until they exist.
        src=dict(mace="skip_dense_8M__xgb",
                 mol=["skip_dense_8M__xgb", "skip_dense_8M__xgb_s1",
                      "skip_dense_8M__xgb_s2"])),

    # ---- controls ---------------------------------------------------------------------------
    "random_encoder": dict(
        # "no pretrain, random" rather than "random encoder" (user 2026-08-19): this arm and
        # `e2e_no_pretrain` are BOTH random-init and the reader has to be able to tell which
        # question each one floors. Naming them as a pair -- "no pretrain, random" (frozen) and
        # "no pretrain, end2end" (fine-tuned) -- puts the shared "no pretrain" first and the
        # protocol second, which is the axis that actually differs.
        label="no pretrain, random", short="no pretrain, random", family="random", color=FAMILY_COLORS["random"],
        probe="frozen", in_ablation=True,
        # MoleculeACE spelled out: the controls' replicates are _00/_01/_02, not <base>/_s1/_s2,
        # so the default resolver would find only the first dir and leave this arm at 1 seed
        # while every CLIMB arm has 3 (audit check 3). The _01/_02 dirs landed 2026-08-18.
        src=dict(mace=["random_baseline_00", "random_baseline_01", "random_baseline_02"],
                 mol=["random_baseline_00", "random_baseline_01", "random_baseline_02"], cbs_legacy_label="no_pretrain")),
    "e2e_no_pretrain": dict(
        label="no pretrain, end2end", short="no pretrain, end2end", family="e2e", color=FAMILY_COLORS["e2e"],
        probe="e2e", in_ablation=True,
        src=dict(mace="no_pretrain_e2e_e2e", mol=["e2e_random_00", "e2e_random_01", "e2e_random_02"], cbs_legacy_label="no_pretrain_e2e")),

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
        label="end2end", short="CheMeleon, end2end", family="chemeleon", color=SHADES["chemeleon"][0],
        probe="e2e", pretrain_replicates=False, in_ablation=False,
        src=dict(mace="chemeleon_e2e", mol=["chemeleon_e2e", "chemeleon_e2e_s1", "chemeleon_e2e_s2"],
                 cbs_legacy_label="chemeleon_e2e")),
    "chemeleon_frozen": dict(
        label="frozen, MLP probe", short="CheMeleon, frozen", family="chemeleon", color=SHADES["chemeleon"][1],
        # OUT OF fig_A1 (user 2026-08-20): the same representation ranks 3rd under XGBoost and
        # 14th under this MLP probe, and showing both makes the ranking a statement about probes
        # rather than about representations. The XGBoost row is the one kept. This arm is still
        # drawn in fig_A2, SI fig a, SI fig f and fig_G, where the frozen MLP number is the
        # quantity those figures are actually about.
        in_ranking=True,
        probe="frozen", pretrain_replicates=False, in_ablation=False,
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
        # WHAT _s1/_s2 ACTUALLY CARRY, measured 2026-08-21 through the resolver rather than
        # asserted: BACE, HIV, Tox21 and QM7 -- NOT "QM7 only", and BACE/Tox21 do NOT "stay at the
        # base dir alone", which is what this comment claimed until the dirs grew content after it
        # was written. Tox21 and QM7 arrive via moleculenet_cv_tox21fixed and
        # moleculenet_cv_qm7clamped, which _s1/_s2 have.
        # The pooling argument above generalises unchanged -- more head seeds on the same folds is
        # a better estimate of the same quantity -- so this is fine where it happens. What is NOT
        # fine is that BBBP and ESOL get one dir while the other four get three: ESOL resolves
        # through moleculenet_cv_regnative and BBBP through plain moleculenet_cv, neither of which
        # _s1/_s2 carry. Audit check 19 reports it. Closing it means running those two datasets at
        # the same two extra head-seed sets, queued with the compute session and deliberately held
        # until the chemeleon featurizer is shown to reproduce across environments -- building them
        # in a drifted env would make two of the six cells an environment measurement.
        src=dict(mace="chemeleon_frozen",
                 mol=["chemeleon_frozen", "chemeleon_frozen_s1", "chemeleon_frozen_s2"],
                 cbs_legacy_label="chemeleon_frozen")),
}

# RETIRED FROM THE PAPER (Leif 2026-08-23): "remove Chemeleon from all figures. If it appears
# anywhere else it must be removed."
#
# The DEFINITIONS stay. Retiring here rather than deleting the entries keeps every source path,
# label and replicate list intact, so the CheMeleon results remain reproducible from this repo and
# the notes that cite them do not become dangling -- the decision is about the paper's narrative,
# not about the validity of the runs. What changes is that no figure DRAWS them.
#
# Excluded at ARM_ORDER, which is what every figure iterates, so one edit retires the arm
# everywhere instead of twelve edits that can disagree. Anything naming a CheMeleon key directly
# -- E2E_PAIRS below, fig_F's embedding roles, A2_ARMS in the bootstrap -- is fixed at its own
# site, because a name that is written down is not reached by a filter.
RETIRED = {"chemeleon_frozen", "chemeleon_frozen_xgb", "chemeleon_e2e"}
_missing = RETIRED - set(ARMS)
assert not _missing, f"arms.py: RETIRED names an arm that does not exist: {_missing}"

# display order: anchors, supervised, unsupervised, unsup->sup, controls, comparator
ARM_ORDER = [a for a in ARMS if a not in RETIRED]

# --------------------------------------------------------------------------------------------
# the canonical 6 panels
# --------------------------------------------------------------------------------------------
# key -> label, metric key, metric label, higher_better, task-type group
PANELS = {
    "MoleculeACE": dict(marker="^", label="MoleculeACE", metric="macro_rmse", metric_label="macro RMSE (30 targets)",
                        metric_short="macro RMSE",
                        higher_better=False, group="potency regression", n_tasks=30),
    # HIV replaced CBS in the rare-active-screen slot on 2026-08-19. Same question, ~3x the
    # resolution: CBS has 43 actives in 10,445 molecules and 8-10 per fold, so NEF1% is quantised
    # in steps of ~1/9 and 0 OF 28 arm pairs had non-overlapping CIs -- it could not separate any
    # two models, and a RANDOM encoder scored 0.758 against Truong's SOTA 0.764. Measured
    # signal-to-noise (across-arm range / mean within-arm SD): CBS NEF1 1.80, CBS ROC-AUC 1.33
    # (saturated, 0.92-0.995), HIV NEF1 3.00, HIV ROC-AUC 3.81. NEF1 is kept as the metric because
    # early enrichment is why this panel exists; HIV is 3.5% active, so it is a genuine rare-active
    # screen. CBS is NOT discarded -- it stays in the 66-dataset all-suites table and moves to an SI
    # external-validation panel, where "no ligand-based model separates from a random encoder" is
    # itself the finding.
    "HIV":         dict(marker="D", label="HIV", metric="nef1", metric_label="NEF1%",
                        metric_short="NEF1%",
                        higher_better=True, group="rare-active screen", n_tasks=1),
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
# The canonical SIX. CBS stays defined above (the all-suites table and the SI external-validation
# panel still use it) but is deliberately NOT one of the six, so `list(PANELS)` is not the order.
PANEL_ORDER = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]

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
    "HIV":          "#3D8073",   # teal     (rare-active screen -- inherits CBS's slot AND colour)
    "CBS":          "#4FA08F",   # light teal (external validation, SI only)
    "BACE":         "#A3455E",   # crimson  (binding classification)
    "Ames":         "#C8912F",   # amber    (mutagenicity)
    "Tox21":        "#8A8A8A",   # grey     (toxicity classification)
    "QM7":          "#6B6494",   # slate    (quantum regression)
    # --- pre-canonical MoleculeNet tasks, kept for the not-yet-migrated figures ------------
    "BBBP":         "#7E6BA8",   # violet
    "ESOL":         "#5B8FBF",   # light blue -- distinct from MoleculeACE, same family
    "Lipophilicity": "#4FA08F",  # light teal -- distinct from CBS, same family
}


SYSTEM = {"anchor": "XGBoost", "chemeleon": "CheMeleon",
          "chemberta": "ChemBERTa-2", "molformer": "MoLFormer", "selfies_ted": "SELFIES-TED"}
# everything else is CLIMB


def system(arm_key: str) -> str:
    """Which model system the arm belongs to -- the first line of a two-line axis label.

    A per-arm `system` key overrides the family default. That exists for the large-corpus runs,
    where the informative split is "CLIMB 100M" on the bold line and the objective underneath --
    the scale is what distinguishes them from every other CLIMB row, so it belongs on the line the
    eye reads first, and the family default of plain "CLIMB" buries it in the subtitle.
    """
    if arm_key not in ARMS:
        return ""
    return ARMS[arm_key].get("system") or SYSTEM.get(ARMS[arm_key]["family"], "CLIMB")


def label(arm_key: str) -> str:
    return ARMS[arm_key]["label"] if arm_key in ARMS else arm_key


def two_line_label(arm_key: str) -> str:
    """'XGBoost\nECFP4+desc' / 'CLIMB\nsupervised, desc' / 'CheMeleon\nend2end'."""
    return f"{system(arm_key)}\n{label(arm_key)}"


def color(arm_key: str) -> str:
    return ARMS[arm_key]["color"] if arm_key in ARMS else "#999999"


# THE FROZEN -> END2END PAIRS SI FIG A DRAWS, and the configuration the paper reports for each
# encoder. Declared here rather than in the figure or its builder because BOTH read it, and when
# they held separate literals they drifted: arms.py renamed sup_dense to "supervised, desc" while
# figures/SI_fig_a.py still asked for "supervised, dense", the join matched nothing, and that
# encoder's line disappeared from all six panels without any check firing.
#
# CheMeleon's frozen half is the XGBOOST probe (Leif 2026-08-20: the only two CheMeleon models the
# paper mentions are frozen+XGBoost and end-to-end-from-foundation; the MLP-probe frozen arm was
# run for our own understanding and is not reported). That matches fig_A1's convention of showing
# each representation at the head that suits it -- SI fig f measures that preference as a property
# in its own right -- so the two figures agree on what "CheMeleon, frozen" names.
# the CheMeleon pair is gone with the arm (RETIRED); built from ARM_ORDER membership so a retired
# arm cannot leave a dangling half-pair behind
E2E_PAIRS = [p for p in [("unsup", "unsup_e2e"),
                         ("sup_dense", "sup_dense_e2e"),
                         ("chemeleon_frozen_xgb", "chemeleon_e2e")]
             if not (set(p) & RETIRED)]


def series_label(frozen_arm: str) -> str:
    """The ENCODER's name for a line that joins its two probes. Taken from the FROZEN arm only.

    Deriving it from either arm was the obvious design and it does not work: the end2end labels do
    not share one comma convention with their frozen counterparts ("supervised, desc" pairs with
    "supervised, desc end2end"), so stripping the probe off each arm independently yields two
    different strings for one line. The frozen arm alone is unambiguous. CheMeleon's frozen arms
    are labelled by their probe ("frozen, XGBoost probe"), so they take the family name instead --
    which is also what SI fig f wants, where each line IS a representation and the probe is the
    x-axis."""
    if ARMS[frozen_arm]["family"] == "chemeleon":
        return SYSTEM["chemeleon"]
    return ARMS[frozen_arm]["label"]


def ablation_arms():
    """Arms allowed in ablation/scaling figures (CheMeleon excluded by decision)."""
    return [k for k, v in ARMS.items() if v["in_ablation"]]


# EVERY lift panel in the paper measures the same thing: lift over the no-pretraining end-to-end
# floor. fig_C1 a/b, fig_C2 c, fig_D e/f and fig_E a/b all resolve their floor from
# ARMS["e2e_no_pretrain"], so the axis label is one string, defined once. The compact (assembled)
# variants used to shorten it to "lift (%)" or drop it entirely, which left a reader of fig_C+D
# unable to tell whether the four panels shared a baseline -- they do (user 2026-08-19).
# The floor figs C1/C2/D lift against, changed 2026-08-20 from the fine-tuned random init to the
# FROZEN one so the floor's protocol matches the frozen arms being measured. This constant is what
# every one of those figures' axis labels is built from, so it must track the floor those figures
# actually use -- it said "no pretrain, end2end" for a while after the floor moved, which put a
# wrong comparator name on five panels of fig C+D.
LIFT_FLOOR_LABEL = ARMS["random_encoder"]["label"]
LIFT_YLABEL = f"lift over {LIFT_FLOOR_LABEL} (%)"
