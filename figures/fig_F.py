"""Fig F — are CLIMB embeddings redundant to classical features?

ONE script, ONE figure: figures_v2/fig_F.png / .pdf  (+ figure_data/fig_F/fig_F.csv as the data record)

The test is concatenation. If the CLIMB embedding carries signal the classical features do not,
gluing it onto ECFP4+descriptors must beat ECFP4+descriptors alone. If it carries nothing new the
concatenation is at best flat — and, because the extra dimensions cost degrees of freedom, may be
slightly worse.

THE AXIS IS "% OF THE ECFP4+desc ANCHOR", ONE UNIT FOR ALL SIX PANELS (user 2026-08-19). The
panels previously each carried their own raw metric -- ROC-AUC, NEF1 and two RMSEs -- with a
per-panel reference line, so nothing could be read across panels and the compact assembly in
fig_E+F dropped the y-label entirely, leaving unlabelled bars. Now 100% is parity with the
classical anchor, by construction, and it is the dotted line in every panel. For RMSE the ratio is
inverted (100*ref/v) so that "more" still means "better" everywhere; plotting the raw ratio for an
error would make the worst arm the tallest bar in two of the six.

Read that way the result is one sentence: NO BAR REACHES 100% ON ANY CANONICAL PANEL. The CLIMB
embedding alone recovers 82-95% of the classical anchor, and adding it on top of the anchor's own
features recovers 92-99% -- i.e. concatenation does not reach the anchor, let alone beat it.

LAYOUT: THREE CLASSICAL BASES x FOUR BARS (user 2026-08-20). One tick per classical block --
desc, ECFP4, desc+ECFP4 -- and inside each, the block ALONE followed by the block plus each
embedding. Redundancy is then read entirely WITHIN a group: an embedding that adds nothing leaves
its bar level with the plain block beside it. The old "none" group (embeddings with no classical
features) is gone; it answers "how good is the embedding by itself", which is a different question
and cost a quarter of the panel width. Those numbers are still in fig_F.csv.

THREE EMBEDDINGS, all frozen, all through the same XGBoost head on the same splits and seeds:

  + CLIMB unsup.        the MLM arm             (source tag CLM)
  + CLIMB sup., desc    the descriptor-SFT arm  (source tag CLMsup)
  + CheMeleon frozen    the external comparator (source tag CheMel)

THE SUPERVISED CLIMB ARM HAD NEVER BEEN RUN THROUGH THIS EXPERIMENT. scripts/concat_redundancy.py
hardcodes ENC = climb_v2_phase2/unsup_8M/encoder, so every "CLIMB" number this figure has ever
shown is the UNSUPERVISED arm, and the paper's other headline CLIMB arm was simply absent from the
redundancy test. Caught by the user 2026-08-20. Its cells draw as declared "not run" slots until
the run lands -- an absence the reader can see, rather than one they cannot.

THE RESULT: concatenation helps on 1 of the 8 tasks run (BBBP, which is not a canonical panel),
and on NONE of the 5 canonical panels currently drawn. Quote the qualified form: the unqualified
"1 of 9" counts tasks the figure does not plot, and its single positive is BBBP, whose control
cell is a NEF1 pinned at a quantised ceiling with zero spread.
`fp+desc+CLM` is worse than `fp+desc` alone on MoleculeACE (0.728 vs 0.690 macro RMSE), Ames
(-0.028), ESOL (-0.028), QM7 (-2.20), HIV (-0.018), Tox21 (-0.012) and BACE (-0.006), and EXACTLY
TIES on CBS (0.930 both). That tie is real rather than a duplicated row: NEF1 counts hits in the
top 1%, so it is quantised, and the two runs do differ on the continuous metric beside it
(ROC-AUC 0.9917 vs 0.9963).

The single exception is BBBP (+0.048 ROC-AUC, beyond its SD), and BBBP is exactly the dataset
dropped from the paper's panel set for failing to discriminate: its whole field spans 1.8% of
ROC-AUC and an UNTRAINED random encoder ranks 7 of 16 on it
(notes/bbbp-anchor-verification-2026-08-16.md). A gain on the one benchmark we already decided
cannot separate models does not rescue the conclusion. BBBP is also the only task where CLM alone
is NOT the weakest of the four feature sets — it is weakest on the other 8 of 9.

So on every benchmark that discriminates, the CLIMB embedding is redundant to the classical
featurization. This is a negative result and is reported as one. It is also the honest frame for
Fig A1, where the descriptor-bearing classical anchors rank first overall: the transformer is not
adding a missing view of the molecule.

THE NEGATIVE RESULT DOES NOT DEPEND ON THE FINGERPRINT, and that is now shown rather than hoped.
The whole test was re-run a second time on the SAME code at FP_VARIANT=ecfp4_legacy
(concat_*_legacy.csv), so legacy-vs-stereo differs by exactly one variable. Concatenation fails to
help under both featurizers on every panel that discriminates, and the fp+desc -> fp+desc+CLM gap
barely moves: Ames 0.0211 legacy against 0.0279 stereo, MoleculeACE 0.0411 against 0.0375. The
stereo fix makes the classical anchor slightly stronger and leaves the conclusion untouched.

THE ISOLATION IS PROVABLE FROM THE TABLES, WHICH IS WHY IT WAS DONE THIS WAY. `CLM` and `desc+CLM`
carry no fingerprint, so they CANNOT respond to FP_VARIANT: all 28 such rows across the two files
are identical to 8 decimal places, while 22 of the 28 fingerprint-bearing rows moved. Two arms that
must not move, and did not.

That check is also what caught the earlier mistake. An initial comparison against the 08-05/08-18
tables looked like a featurizer test and was not one -- the CLM-only rows had moved (MoleculeACE
0.840 -> 0.819, CBS NEF1 0.509 -> 0.768), which a featurizer cannot do, because three unrelated
commits had landed on the path between the runs (0ab0388, c4f1c23, 3c52686). GENERAL RULE WORTH
REUSING: any table containing an arm INVARIANT to the change under test carries its own isolation
check for free, and it costs nothing to look at it before quoting a delta.

One number not to over-read: CBS fp+desc NEF1 moves +0.0250, but NEF1 counts hits in the top 1% so
that is a single hit crossing the threshold; the ROC-AUC beside it moved +0.0010.

PANEL SCOPE: ALL SIX canonical panels are filled (2026-08-18). MoleculeACE, CBS and Ames come from
analysis/rigor/concat_panels_climb.csv; BACE, Tox21 and QM7 from the original MoleculeNet run.
Ames was the last to land, and it was never a missing RUN — the predictions had been written all
along and only the scoring failed, because that runner writes the FEATURE-SET name into the `seed`
column while the Polaris scorer calls int(seed). Its cells are 1 replicate per feature set, so they
are drawn WITHOUT a whisker rather than borrowing an SD from another panel (same rule as fig_E).
ESOL, BBBP and HIV were also run and are NOT shown here — they are outside the canonical panel set
— but they are in figure_data/fig_F/fig_F.csv and BBBP is the exception discussed above.

Error bars are +-1 SD across the seeds of that (task, feature set) cell.

Source: analysis/rigor/concat_redundancy.csv + concat_panels_climb.csv (git-tracked).

Run:  python3 -m figures.fig_F
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, mark_empty, row_ncol, LEGEND_BOX
from figures.arms import PANELS, PANEL_ORDER, SHADES

check_font()
INK = "#000000"

ROOT = Path(__file__).resolve().parent.parent
# TWO sources, concatenated. `concat_redundancy.csv` is the original MoleculeNet run
# (ESOL/BBBP/BACE/HIV/Tox21/QM7); `concat_panels_climb.csv` is the canonical-panel top-up that
# added MoleculeACE and CBS (landed 2026-08-18). Same four feature sets, same XGBoost head, each
# task on its own canonical split. Ames landed 2026-08-18 in concat_panels_climb.csv: its
# predictions had been written all along and only the SCORING failed, because that runner puts the
# feature-set name in the `seed` column and the Polaris scorer does int(seed). Scored per feature
# set and appended, so the MoleculeACE and CBS rows were not clobbered.
# THE _stereo TABLES ARE THE CURRENT ONES. Both were re-run on 2026-08-19 with the current code and
# FP_VARIANT=ecfp4_stereo; the un-suffixed files are the 2026-08-05 / 08-18 vintage and are kept
# only as the provenance trail (concat_panels_climb.PREFIX_BACKUP.csv is a byte-identical copy).
#
# DO NOT PRESENT old-vs-new AS A FEATURIZER COMPARISON. It is not one, and the numbers invite the
# claim: the CLM-only arm uses no fingerprint at all, yet it moves (MoleculeACE 0.8401 -> 0.8185,
# CBS nef1 0.5089 -> 0.7678). A featurizer cannot do that. Three commits also landed on this path
# between the two runs -- 0ab0388 (standardize + median-impute classical features for non-tree
# heads), c4f1c23 (OOD prediction bounding), 3c52686 (anchor re-runs) -- so the pair confounds the
# featurizer with the code version. The defensible statement is "re-run on current code with the
# current featurizer", full stop. A clean single-variable isolation needs the same code at
# FP_VARIANT=ecfp4_legacy, where the CLM rows matching to the digit IS the check that the
# isolation worked.
# FOUR SOURCES, TWO EMBEDDINGS x TWO TRACKS, and every one that exists is read.
#
# The v2 tables carry all SEVEN feature blocks (fp, desc, fp+desc alone, each + the embedding, and
# the triple) where the stereo pair carried four, which is what turns this figure from four
# populated cells into the full lattice.
#
# THEY COME FROM A DIFFERENT MACHINE THAN THE STEREO TABLES DID, and that is an accepted, recorded
# limitation rather than an oversight: XGBoost's floating-point reduction order differs between
# Apple silicon and x86, so the same code and the same library versions give ~1% different metrics
# (notes/concat-reproducibility.md, confirmed on both architectures). The user accepted the mix
# 2026-08-20. What it costs is stated plainly: values here may differ in the last percent from an
# arm64 rerun, and no cell in this figure should be quoted against a number from another figure.
# What it does NOT cost is any within-panel comparison, because a panel's cells all come from one
# table and therefore one machine.
# The source FILES are derived from ROLE_ORDER below, not listed here -- see SOURCES. A hardcoded
# list cannot know about a table written after it, which is the failure this figure has already had
# once: the four-file list held climb and chemeleon only, so the supervised arm's tables could have
# landed in analysis/rigor and the third bar would have kept drawing "not run" with nothing failing.
RIGOR = ROOT / "analysis" / "rigor"
OUTDIR = ROOT / "figures_v2"
# the per-task record is DATA, not a deliverable -- figures_v2/ holds only what goes in the paper
DATADIR = ROOT / "figure_data" / "fig_F"

# canonical panel -> task name in the source (None = experiment never run there)
PANEL_TASK = {"MoleculeACE": "MoleculeACE", "HIV": "HIV", "BACE": "BACE",
              "Ames": "Ames", "Tox21": "Tox21", "QM7": "QM7"}
PRIMARY = {"MoleculeACE": "macro_rmse", "HIV": "nef1", "BACE": "roc_auc",
           "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
LOWER_BETTER_METRIC = {"rmse", "macro_rmse"}
# panel metric -> the metric the PER-FOLD dumps record for it
FOLD_METRIC = {"macro_rmse": "rmse"}
# every task in the sources, for the CSV record (superset of the canonical panels)
ALL_TASKS = {"MoleculeACE": "macro_rmse", "CBS": "nef1", "ESOL": "rmse", "QM7": "rmse",
             "BACE": "roc_auc", "BBBP": "roc_auc", "Ames": "roc_auc", "Tox21": "roc_auc",
             "HIV": "roc_auc"}
# classical anchor keeps the anchor amber; anything containing CLIMB moves into the unsup blues,
# darkening as more classical information is added back
# THE FULL LATTICE (user 2026-08-19). Every combination of the two classical blocks with at most
# one embedding: 4 classical bases x {no embedding, + CLIMB, + CheMeleon}, minus the empty cell.
# CLIMB+CheMeleon is deliberately absent -- stacking two learned embeddings answers no question
# this paper asks.
#
# COLOUR IS THE EMBEDDING, GROUP IS THE CLASSICAL BASE, and that is what makes the panel answer
# both directions of the redundancy question at once:
#   read WITHIN a group   -> does the embedding add anything to this classical block?
#   read ONE COLOUR ACROSS groups -> does the classical block add anything to this embedding?
# Redundancy is directional, so a figure that only shows the first reading cannot support the
# claim: a flat within-group result is equally consistent with the head failing to exploit 512
# dense dims beside 2048 sparse bits, with added variance at fixed n, or with a task ceiling.
#
# The desc+ECFP4 group is also the POSITIVE CONTROL -- "does adding a second block ever help on
# this task at all" -- which is what turns a flat embedding result from unreadable into negative.
# It was a dotted reference line for one revision; back to bars so it sits on the same footing as
# everything else it is being compared against.
# LAYOUT (user 2026-08-20). One tick per CLASSICAL BASE -- desc, ECFP4, desc+ECFP4 -- and inside
# each, the classical block ALONE followed by the same block plus each embedding. The question the
# figure exists to answer is read entirely within a group: if an embedding is redundant to a
# classical block, its bar equals the block's own bar.
#
# THE "none" GROUP IS GONE (user). It held the embeddings with no classical features at all, which
# answers a different question -- how good is the embedding on its own -- and made the panel four
# groups wide for a comparison the redundancy claim does not use. The embedding-alone numbers are
# still in the CSV; they are just not what this panel is for.
#
# THREE EMBEDDINGS, NOT ONE. Until 2026-08-20 the only CLIMB embedding here was `CLM`, which is
# unsup_8M -- scripts/concat_redundancy.py hardcodes ENC = climb_v2_phase2/unsup_8M/encoder -- so
# the supervised arm had never been through this experiment at all. The user spotted the gap. Both
# CLIMB arms and CheMeleon frozen now have a slot; the supervised cells draw as "not run" until
# the run lands, which is the honest state rather than a silent absence.
NO_EMB = "no embedding"
EMB_CLM_U, EMB_CLM_S, EMB_CHE = "+ CLIMB unsup.", "+ CLIMB sup., desc", "+ CheMeleon frozen"
ROLE_COLOR = {NO_EMB: SHADES["anchor"][0], EMB_CLM_U: SHADES["unsup"][0],
              EMB_CLM_S: SHADES["sup"][0], EMB_CHE: SHADES["chemeleon"][0]}
# role -> the suffix its feature key carries in the source tables. "CLMsup" is the tag the
# supervised concat run must write; it is named here so the request and the reader agree.
ROLE_SUFFIX = {NO_EMB: None, EMB_CLM_U: "CLM", EMB_CLM_S: "CLMsup", EMB_CHE: "CheMel"}
# CheMeleon RETIRED from the paper (Leif 2026-08-23) -- see figures/arms.py RETIRED. Its role
# constant and colour stay defined so the concat tables, which still carry CheMel blocks, load
# without special-casing; it is simply not drawn.
ROLE_ORDER = [NO_EMB, EMB_CLM_U, EMB_CLM_S]

# TWO DESCRIPTOR FAMILIES ON ONE AXIS (Leif 2026-08-22). RDKit was the only descriptor block here.
# Mordred earns its own ticks because CheMeleon is a D-MPNN pretrained to regress exactly the 1,613
# Mordred 2D descriptors -- so "CheMeleon is redundant to fp+desc" against RDKit compares two
# unrelated feature families and is merely suggestive, while against MORDRED it is the sharp claim:
# if the embedding adds nothing on top of its OWN pretraining target, it compressed that target
# rather than learning past it.
#
# THE KEYS COLLIDE AND THAT IS THE TRAP. Both families' tables call the descriptor block "desc",
# so concatenating them keys "desc"/"fp+desc"/"desc+CLM" identically and load()'s drop_duplicates
# would keep whichever landed first and silently discard the other family ENTIRELY -- while every
# bar still drew and nothing failed. Mordred rows are therefore namespaced to `mdesc` on read; see
# _tag_family(). Blocks with no descriptor in them (fp, fp+CLM, the bare embeddings) are the same
# experiment in both families and stay unnamespaced, where the duplicate check turns them into a
# free cross-family reproducibility read.
MORDRED_KEY = "mdesc"

# ---------------------------------------------------------------------------------------------
# THE AXIS IS LIFT OVER A REFERENCE, AND EACH TICK IS A DIFFERENT REFERENCE (Leif 2026-08-22)
# ---------------------------------------------------------------------------------------------
# The question this figure exists to answer is "what ORTHOGONAL information does an embedding
# carry", and the previous layout made the reader answer it by subtracting two bars by eye. Now
# the reference block IS the zero line -- it is not drawn as a bar at all -- and every bar is what
# one addition buys on top of it. A redundant addition sits ON the line; an orthogonal one rises
# off it. The CheMeleon-vs-Mordred result becomes a bar of height ~0 next to a bar of visible
# height, in the same panel, instead of two absolute values a reader must difference.
#
# THE CLASSICAL ADDITIONS ARE NOT DECORATION, THEY ARE THE POSITIVE CONTROL. On the ECFP4 tick,
# "+ RDKit desc" and "+ Mordred" show that descriptors DO carry information ECFP4 lacks. Without
# them a flat embedding bar is unreadable: it is equally consistent with "the embedding is
# redundant" and with "nothing helps on this task". They are also what makes the pair legible --
# Mordred lifts ECFP4, and CheMeleon lifts ECFP4, but CheMeleon does not lift Mordred.
#
# Additions differ per reference because the experiment ran what it ran; a tick shows every
# addition that exists for it rather than padding to a uniform width.
# MORDRED RETIRED FROM THIS FIGURE (Leif 2026-08-23), to shorten the narrative -- NOT because the
# result was wrong. The Mordred axis showed CheMeleon adding +0.0009 over the 1,613 descriptors it
# was pretrained to regress, which is a real finding and stays in the tables and in
# notes/fig_F-mordred-axis-handoff.md. With CheMeleon out of the paper the comparison it existed to
# sharpen has no arm left to sharpen, so drawing it would cost a tick and answer nothing.
REF_ORDER = [("ECFP4", "fp"), ("RDKit desc", "desc"), ("ECFP4+desc", "fp+desc")]
CLASSICAL_ADD = {
    "fp":      [("fp+desc", "+ RDKit desc", SHADES["anchor"][0])],
    "desc":    [("fp+desc", "+ ECFP4", SHADES["anchor"][1])],
    "fp+desc": [],
}
EMB_ROLES = [r for r in ROLE_ORDER if ROLE_SUFFIX[r]]


def additions(ref_key):
    """[(feature_key, label, colour)] for one reference -- classical first, then the embeddings."""
    out = list(CLASSICAL_ADD.get(ref_key, []))
    out += [(f"{ref_key}+{ROLE_SUFFIX[r]}", r, ROLE_COLOR[r]) for r in EMB_ROLES]
    return out


GROUPS = [(tick, ref, additions(ref)) for tick, ref in REF_ORDER]
# every key the panel touches, reference keys included -- shared_ylims and the source audit read it
FEATURES = [(ref, None, None) for _, ref in REF_ORDER] + \
           [(k, lab, c) for _, _, adds in GROUPS for k, lab, c in adds]


BASE = "fp+desc"
# TWO HEADLINE DELTAS, NOT ONE (user 2026-08-20: "it'll become two numbers yes, everywhere report
# supervised and unsupervised"). The concatenation test is run separately for each CLIMB arm, so
# the figure reports the redundancy verdict for each rather than picking one arm to speak for
# both. Derived from ROLE_SUFFIX so a fourth embedding would be picked up automatically.
CONCAT_ARMS = [(r, f"{BASE}+{ROLE_SUFFIX[r]}") for r in (EMB_CLM_U, EMB_CLM_S)]
# short column-safe keys for the CSV
CONCAT_TAG = {EMB_CLM_U: "unsup", EMB_CLM_S: "sup"}

# EVERY EMBEDDING'S TABLES, DERIVED FROM ROLE_ORDER RATHER THAN LISTED.
#
# Each embedding contributes two tables to analysis/rigor: the MolNet lattice
# (concat_redundancy_<stem>_v2.csv) and the canonical-panel top-up (concat_panels_<stem>_v2.csv).
# Declaring the STEM per role and building the paths from ROLE_ORDER means adding a role to the
# figure adds its sources in the same edit. The alternative -- a literal list of four paths -- is
# the pattern that has cost this project a panel more than once: a name list written before an
# object exists cannot include it, and the failure is silent because "file absent" is also the
# legitimate state of a run still in flight.
ROLE_SRC_STEM = {EMB_CLM_U: "climb", EMB_CLM_S: "climb_sup", EMB_CHE: "chemeleon"}
_undeclared = [r for r in ROLE_ORDER if ROLE_SUFFIX[r] and r not in ROLE_SRC_STEM]
assert not _undeclared, f"fig_F: {_undeclared} have a feature suffix but no source-table stem"
SOURCES = [RIGOR / f"concat_{kind}_{ROLE_SRC_STEM[r]}_v2.csv"
           for r in ROLE_ORDER if ROLE_SUFFIX[r]
           for kind in ("redundancy", "panels")]

# --- the Mordred axis, and the same-environment rule it depends on ------------------------------
#
# The Mordred tables were computed tonight; the published RDKit tables were not. Measured on the
# one block that shares identical code and contains neither descriptors nor an embedding -- plain
# `fp` -- the environment term is 0.01 to 0.22 fold SD (QM7 rmse 216.1573 tonight vs 216.6266
# published). That is ~10x below the BBBP effect the figure would be used to read, so it does not
# threaten the finding -- but it would sit inside EVERY RDKit-vs-Mordred difference drawn here,
# and this figure's whole logic is "change exactly one thing". Descriptor family AND environment
# is two.
#
# So the rule is: Mordred is drawn ONLY against an RDKit arm regenerated in the same environment.
# If those tables are absent the Mordred groups draw as "not run" rather than being compared
# across environments -- refusing the comparison is the honest state, and it is visible on the
# canvas instead of buried in a caveat nobody reads.
# ---------------------------------------------------------------------------------------------
# V2: ONE ENVIRONMENT, RECORDED, AND PER-FOLD VALUES
# ---------------------------------------------------------------------------------------------
# v1 and v2 are two environments and are MUTUALLY UNMIXABLE. Measured on the 30 embedding-free
# cells they share: 27 of 30 differ, median 0.38 fold SD and max 1.82, against lifts of 0.1-0.4
# fold SD -- the shift is as large as the effect. The cause is that no v1 artifact recorded its
# own environment (the AMI pins deepchem/numpy/torch/rdkit but not xgboost or mordred). v2 is the
# first set that does: analysis/rigor/figF_v2/_environment.json carries the instance type, AMI and
# package versions. v1 is preserved under figF/ and is NOT read here.
#
# The embedding-free blocks live in a SHARED table rather than being recomputed per tag -- they
# contain no embedding, so they are identical across tags by construction, and computing them once
# is what makes that a fact rather than a hope.
FIGF_V2 = RIGOR / "figF_v2"
V2_TAGS = {EMB_CLM_U: "CLMunsup", EMB_CLM_S: "CLMsup"}
V2_STEMS = ["SHARED"] + [V2_TAGS[r] for r in ROLE_ORDER if ROLE_SUFFIX[r]]
SOURCES_V2 = [FIGF_V2 / f"{pre}_rdkit_sameenv_{stem}_V2.csv"
              for stem in V2_STEMS for pre in ("concat", "concat_panels")]
FOLD_SOURCES = [FIGF_V2 / f"{pre}_rdkit_sameenv_{stem}_V2_folds.csv"
                for stem in V2_STEMS for pre in ("concat", "concat_panels")]
V2_READY = all(f.exists() for f in SOURCES_V2)
PAIRED_READY = V2_READY and all(f.exists() for f in FOLD_SOURCES)

# ---------------------------------------------------------------------------------------------
# THE AXIS: percent of the classical anchor, so all six panels share one unit
# ---------------------------------------------------------------------------------------------
# Raw metrics cannot share an axis -- ROC-AUC, NEF1 and two RMSEs in different units -- so every
# panel used to carry its own scale and its own dotted reference line, and nothing could be read
# across panels. Expressing each feature set as a PERCENTAGE OF THE ECFP4+desc ANCHOR puts them
# on one axis: 100% is parity with the anchor, by construction, and is the dotted line.
#
# For higher-is-better metrics that is 100*v/ref. For RMSE it is 100*ref/v -- the RATIO IS
# INVERTED so that "more" still means "better" on the same axis; plotting 100*v/ref for an error
# would make the worst arm the tallest bar in two of the six panels.
#
# The anchor's OWN uncertainty is deliberately not propagated: it is the reference, it is 100% by
# definition, and folding its spread into every other bar would make the comparison to it noisier
# than it is. Each bar's error is its own spread rescaled by the same factor as its mean, so the
# error bars answer "how well is this feature set measured", not "is it distinguishable from the
# anchor" -- that second question is what the delta column of fig_F.csv is for.
PCT_YLABEL = "% of ECFP4+desc"
# Every panel is a percentage lift over ITS OWN tick reference, so one label serves all six
# and the number is comparable across panels in a way the raw metrics never were.
LIFT_YLABEL = "lift over reference (%)"
PCT_YLIM = (0, 132)


def as_pct_of_anchor(values, errs, metric):
    """(values, errs) rescaled so the anchor is 100. Returns NaNs unchanged."""
    ref = values[0]
    if not np.isfinite(ref) or ref == 0:
        return [np.nan] * len(values), [0.0] * len(values)
    lower_better = metric in LOWER_BETTER_METRIC
    out_v, out_e = [], []
    for v, e in zip(values, errs):
        if not np.isfinite(v) or v == 0:
            out_v.append(np.nan); out_e.append(0.0); continue
        if lower_better:
            out_v.append(100.0 * ref / v)
            out_e.append(100.0 * ref * e / (v * v))      # |d(ref/v)/dv| * e
        else:
            out_v.append(100.0 * v / ref)
            out_e.append(100.0 * e / ref)
    return out_v, out_e


def compute():
    """The concatenation table. Exposed so figures/fig_E_plus_F.py can assemble this figure with fig_E
    without re-implementing (and therefore drifting from) the analysis."""
    # Missing sources are SKIPPED, not fatal: the CheMeleon panels half is still running, and a
    # figure that refuses to draw until every cell exists cannot show progress. Cells with no
    # source render as the declared "not run" slot, which is the honest state.
    parts = []
    if not V2_READY:
        missing = [f.name for f in SOURCES_V2 if not f.exists()]
        raise FileNotFoundError(
            f"fig_F: v2 tables absent ({len(missing)}), e.g. {missing[:2]}. v1 under figF/ is a "
            f"DIFFERENT environment and is not a fallback -- 27 of its 30 shared cells differ from "
            f"v2 by a median 0.38 fold SD against lifts of 0.1-0.4 SD.")
    print(f"  fig_F: v2 tables, one recorded environment ({FIGF_V2 / '_environment.json'})")
    for f in SOURCES_V2:
        parts.append(_align_tags(pd.read_csv(f)))
    if not parts:
        raise FileNotFoundError("fig_F: no concat source table found")
    d = pd.concat(parts, ignore_index=True)

    # THE EMBEDDING-FREE BLOCKS ARE IN BOTH EMBEDDINGS' TABLES, so fp, desc and fp+desc arrive
    # twice per task. They are the SAME experiment run twice -- no embedding is involved, so
    # neither the CLIMB nor the CheMeleon pass can influence them -- which makes the duplicate a
    # free reproducibility check rather than a nuisance. Report the disagreement, then keep the
    # first occurrence so every downstream .loc gets a scalar.
    key = ["task", "features", "metric"]
    dup = d[d.duplicated(key, keep=False)]
    if len(dup):
        w = dup.groupby(key)["mean"].agg(["min", "max", "count"])
        w["rel"] = 100 * (w["max"] - w["min"]) / w["min"].abs().replace(0, float("nan"))
        worst = w.sort_values("rel", ascending=False).head(3)
        print(f"  fig_F: {len(w)} embedding-free cell(s) present in both tables; "
              f"max disagreement {w['rel'].max():.2f}% "
              f"({', '.join(f'{i[0]}/{i[1]} {r.rel:.2f}%' for i, r in worst.iterrows())})")
    return _fill_ames_se(d.drop_duplicates(key, keep="first").reset_index(drop=True))


# Panels that share a METRIC share a y-RANGE, so a bar of a given height means the same thing in
# each (user 2026-08-19: "some go to 1 others don't, seems inconsistent"). Panels on different
# metrics cannot share one -- macro RMSE, NEF1% and kcal/mol are not commensurable -- so the rule is
# per metric, not global. Computed from the drawn data rather than hardcoded.
def _align_tags(df):
    """CLMunsup -> CLM, for EVERY figF table regardless of descriptor family.

    The tag rename belongs to the SOURCE PIPELINE, not to one family: both the mordred and the
    rdkit_sameenv tables come off the same box and both label the unsupervised arm CLMunsup, where
    ROLE_SUFFIX and therefore every drawn key says CLM. Doing it only inside _tag_family left the
    RDKit half untouched, so desc+CLM / fp+CLM / fp+desc+CLM resolved to nothing and three of the
    four bars in three of the four ticks silently vanished from all six panels.

    It surfaced as a NaN axis limit rather than as blank bars, which is the only reason it was
    caught on sight -- the same absence would have drawn as "not run" in a slightly different code
    path and looked deliberate.
    """
    df = df.copy()
    df["features"] = df["features"].str.replace("CLMunsup", "CLM", regex=False)
    return df


def _tag_family(df):
    """Namespace a MORDRED table's descriptor keys, and align its embedding tag with ours.

    Two independent renames, both of which are silent corruption if skipped:

    * `desc` -> `mdesc`. Both families name the block "desc", so without this the two tables key
      identically on (task, features, metric) and load()'s drop_duplicates keeps ONE of them and
      drops the other family whole -- with every bar still drawn. Only keys that actually contain
      a descriptor are renamed; `fp`, `fp+<emb>` and the bare embeddings are the same experiment
      in both families and are left alone so the duplicate check reads them as a cross-family
      reproducibility check.
    * `CLMunsup` -> `CLM`. The Mordred run tags the unsupervised arm CLMunsup where the published
      tables use CLM; it is the same encoder (climb_v2_phase2/unsup_8M) under a different string,
      and ROLE_SUFFIX keys on that string directly. CLMsup and CheMel already agree.
    """
    df = _align_tags(df)
    # THE CONTRACT IS TESTED, NOT TRUSTED. Agreed with the compute session 2026-08-22: every table
    # it hands over says `desc`, MolNet and panels alike, and the rename lives HERE and nowhere
    # else. Two components independently "fixing" the same thing is the failure both halves of this
    # project hit today from opposite sides -- so if a table ever arrives pre-namespaced, this
    # fails loudly instead of silently no-opping and leaving the ownership question unresolved.
    already = sorted(f for f in df["features"].unique() if MORDRED_KEY in f)
    assert not already, (
        f"fig_F: Mordred table already namespaced ({already[:3]}). The rename is owned by "
        f"_tag_family; a source that pre-namespaces means both halves are doing it.")
    assert any("desc" in f for f in df["features"].unique()), \
        "fig_F: Mordred table has no `desc` key -- the agreed source format changed"
    df["features"] = df["features"].str.replace("desc", MORDRED_KEY, regex=False)
    return df


def _fill_ames_se(d):
    """Ames carries a mean and no spread; give it the analytic AUC SE instead of nothing.

    Polaris withholds the Ames test labels, so there is ONE held-out evaluation and no fold
    variance to report -- every Ames row arrives with std = NaN. That is not a defect in the
    tables, it is the shape of the benchmark.

    Hanley & McNeil (1982) gives the SE of an AUC from the AUC and the two class counts alone, so
    this panel can carry a real interval without labels. The definition lives in
    scripts/merge_concat_ames_panels.py and is imported rather than copied: the earlier version of
    this figure coerced the missing std to 0.0, which drew a zero-length capped whisker that read
    as PERFECT PRECISION on the one panel with no replicates at all. Leif caught it on sight
    ("barely visible, that's a bit sus"). A second hand-written copy of the formula is how that
    kind of thing comes back.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_ames_se", ROOT / "scripts" / "merge_concat_ames_panels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ames = (d.task == "Ames") & (d.metric == "roc_auc") & (~np.isfinite(d["std"]))
    n = int(ames.sum())
    if n:
        d.loc[ames, "std"] = d.loc[ames, "mean"].map(m.hanley_mcneil_se)
        print(f"  fig_F: Ames has no fold spread by construction (labels withheld); "
              f"{n} cell(s) given the Hanley-McNeil analytic AUC SE "
              f"(n1={m.N_POS}, n0={m.N_NEG})")
    return d


def fold_table():
    """{(task, features, metric): {fold: value}} from the v2 per-fold dumps.

    This is what makes a paired error bar possible. A bar in this figure is a DIFFERENCE between
    two arms that saw the SAME folds, so its uncertainty is the spread of the per-fold difference,
    in which fold difficulty -- the dominant term -- cancels. The marginal SD each summary row
    reports does NOT cancel it: measured on fp+desc -> fp+desc+CLM it ran 2.0x the lift on BACE and
    7.8x on Tox21, which is why this figure drew no intervals at all until the per-fold values
    existed.

    MoleculeACE keys its `fold` column on the TARGET name rather than an index; the arithmetic is
    identical, the pairing is per target. Ames emits no fold rows at all -- Polaris withholds the
    labels, so there is one evaluation and no spread to pair.
    """
    import collections
    out = collections.defaultdict(dict)
    for f in FOLD_SOURCES:
        if not f.exists():
            continue
        for r in _align_tags(pd.read_csv(f)).itertuples():
            out[(r.task, r.features, r.metric)][str(r.fold)] = float(r.value)
    return out


def shared_ylims(d):
    """{metric: (lo, hi)} over every canonical panel scored on that metric."""
    import collections
    acc = collections.defaultdict(list)
    for p in PANEL_ORDER:
        task = PANEL_TASK.get(p)
        if task is None:
            continue
        metric = PRIMARY[task]
        g = d[(d.task == task) & (d.metric == metric)].set_index("features")
        for f, _, _ in FEATURES:
            if f in g.index:
                v, e = float(g.loc[f, "mean"]), float(g.loc[f, "std"])
                e = 0.0 if e != e else e
                acc[metric] += [v - e, v + e]
    out = {}
    for metric, vs in acc.items():
        lo, hi = min(vs), max(vs)
        pad = 0.10 * max(hi - lo, 1e-9)
        out[metric] = (lo - pad, min(hi + pad, 1.0) if metric == "roc_auc" else hi + pad)
    return out


def draw_panel(ax, d, p, compact=False, tag=None, fig=None, ylims=None, xrot=None, bw=None,
               ylabel=True):
    """Draw ONE canonical panel onto an existing axes.

    `compact` narrows the bars and drops the y-label, for the assembled fig_E+F where six panels
    share half the canvas. `tag` puts the panel letter ON THE TITLE BASELINE, immediately left of
    the title, matching fig_E's tag/title pair -- so every panel letter in the assembled figure is
    positioned the same way instead of F's floating above its centred title.
    """
    meta = PANELS[p]
    task = PANEL_TASK[p]
    arrow = "↑" if meta["higher_better"] else "↓"
    if tag is None:
        ax.set_title(f"{meta['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=3)
    else:
        from matplotlib.transforms import ScaledTranslation
        ax.text(0.0, 1.04, tag, transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        # short label in the assembled figure: "Ames Mutagenicity" runs into the next panel's tag
        # short names in the assembled figure: the full labels run into the next panel's tag
        short = {"Ames": "Ames", "MoleculeACE": "MolACE"}.get(p, meta["label"])
        ax.text(0.0, 1.04, f"{short} {arrow}", fontsize=FS["title"] - (1 if compact else 0),
                fontweight="bold",
                va="bottom", ha="left", color=INK,
                transform=ax.transAxes + ScaledTranslation(11 / 72, 0, fig.dpi_scale_trans))
    # ABSOLUTE metric (user 2026-08-19: "I do think absolute performance was more meaningful").
    # The % -of-anchor axis was tried and dropped: it made the panels comparable to each other but
    # answered the wrong question, since the comparison that matters here is base vs base+embedding
    # WITHIN a group, and that is a like-for-like pair already. Labelled in every panel including
    # the compact assembly, where it used to be dropped and left the bars unlabelled.
    # ONE UNIT ACROSS ALL SIX PANELS, because every bar is now a relative lift rather than a raw
    # metric. The raw metric name moved into the panel title's arrow, which still says which
    # direction is better; the axis says what the bar measures.
    # ONE y-label per ROW, not per panel. All six read the same string, so five of them were
    # spending width to repeat the sixth -- and width is what buys a shorter figure at fixed A4
    # text-block width. `ylabel` is passed False by the callers that already have one to their left.
    if ylabel:
        ax.set_ylabel(LIFT_YLABEL, fontsize=FS["annot"] - (1 if compact else 0), color=INK)
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if task is None:
        ax.text(0.5, 0.5, "concatenation test\nnot run", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS["annot"], color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        # DECLARED empty, so style.check_no_empty_panels passes this one and still fails on any
        # panel that is empty because a key was resolved wrongly. Ames is the live case: its
        # concat run emits PREDICTIONS rather than scores, because Polaris withholds test labels,
        # so the panel waits on an off-box scoring step rather than on a GPU run.
        mark_empty(ax, f"{p}: concatenation test not run on this panel")
        return
    metric = PRIMARY[task]
    FOLDS = fold_table()
    g = d[(d.task == task) & (d.metric == metric)].set_index("features")

    def cell(k):
        if k not in g.index:
            return None, None
        v = float(g.loc[k, "mean"])
        e = float(g.loc[k, "std"])
        # NaN STAYS NaN. This coerced a missing spread to 0.0, and matplotlib duly drew a
        # zero-length error bar with caps -- a small tick at the top of the bar that reads as
        # "measured to within nothing at all" rather than "not measured". Leif caught it on the
        # Ames panel (2026-08-20: "barely visible, that's a bit sus"), where every cell had an
        # empty std because the panel has one scored value per feature block and no seed axis.
        # Ames now carries a real analytic SE (see scripts/merge_concat_ames_panels.py); this
        # guard is what stops the next panel with no spread from claiming perfect precision.
        return v, (float("nan") if not np.isfinite(e) else e)

    # LIFT, NOT LEVEL. Each bar is what its addition buys over the tick's reference; the
    # reference is the zero line and is never drawn as a bar. Signed so POSITIVE ALWAYS MEANS
    # BETTER, which for RMSE means the reference minus the arm rather than the other way round --
    # otherwise the best bar would point down in two of the six panels.
    def lift_pct(ref_key, add_key):
        v0, _ = cell(ref_key)
        v1, _ = cell(add_key)
        if v0 is None or v1 is None or not np.isfinite(v0) or v0 == 0:
            return None
        gain = (v0 - v1) if metric in LOWER_BETTER_METRIC else (v1 - v0)
        return 100.0 * gain / abs(v0)

    GAP = 0.8
    xs, ys, cs, present, labels = [], [], [], [], []
    xi = 0.0
    group_centres = []
    for _, ref_key, adds in GROUPS:
        start = xi
        for k, lab, colour in adds:
            xs.append(xi)
            ys.append(lift_pct(ref_key, k))
            cs.append(colour)
            labels.append(lab)
            present.append(ys[-1] is not None)
            xi += 1.0
        if xi == start:            # a reference with no addition present at all
            xi += 1.0
        group_centres.append((start + xi - 1.0) / 2.0)
        xi += GAP
    x = np.array(xs)
    # PAIRED interval: the SD of the per-fold LIFT, not of either arm's absolute score.
    def lift_sd(ref_key, add_key):
        if not PAIRED_READY:
            return float("nan")
        # MoleculeACE's panel metric is macro_rmse -- the MEAN over its 30 targets -- while the
        # per-fold dump records plain `rmse` per target, because a macro average has no per-fold
        # decomposition. The replicate unit there is the TARGET, which is what the rest of this
        # figure set already uses for MoleculeACE, so pairing per target is the same estimand.
        # Without this the lookup missed on every MoleculeACE cell and the panel silently drew no
        # intervals while every other panel had them.
        fmetric = FOLD_METRIC.get(metric, metric)
        r0 = FOLDS.get((task, ref_key, fmetric), {})
        r1 = FOLDS.get((task, add_key, fmetric), {})
        shared = sorted(set(r0) & set(r1))
        if len(shared) < 2:
            # AMES HAS NO REPLICATE AXIS AT ALL -- Polaris withholds the labels, so there is one
            # scored evaluation per feature block and nothing to pair. Propagate the two analytic
            # Hanley-McNeil AUC SEs through the lift instead of drawing nothing, because a bar with
            # no whisker reads as precise rather than as unmeasured.
            #
            # This is a CONSERVATIVE bound and is labelled as one. The two arms are scored on the
            # SAME test set, so their AUC estimates are positively correlated and the true SE of
            # their difference is smaller than independent propagation gives. The correlated form
            # (DeLong) needs the label vector, which is exactly what is withheld -- so an upper
            # bound is the strongest honest statement available here.
            v0, _ = cell(ref_key)
            v1, _ = cell(add_key)
            g0 = g.loc[ref_key, "std"] if ref_key in g.index else float("nan")
            g1 = g.loc[add_key, "std"] if add_key in g.index else float("nan")
            if None in (v0, v1) or not all(np.isfinite(x) for x in (v0, v1, g0, g1)) or v0 == 0:
                return float("nan")
            # lift = 100*(v1-v0)/v0  ->  partials 100/v0 and -100*v1/v0**2
            var = (100.0 / v0) ** 2 * float(g1) ** 2 + (100.0 * v1 / v0 ** 2) ** 2 * float(g0) ** 2
            return float(np.sqrt(var))
        per = []
        for f in shared:
            v0, v1 = r0[f], r1[f]
            if not np.isfinite(v0) or v0 == 0:
                continue
            gain = (v0 - v1) if fmetric in LOWER_BETTER_METRIC else (v1 - v0)
            per.append(100.0 * gain / abs(v0))
        return float(np.std(per, ddof=1)) if len(per) > 1 else float("nan")

    es = [lift_sd(rk, k) for _, rk, adds in GROUPS for k, _, _ in adds]

    base_v = 0.0               # the reference line, by construction
    drawn = [v for v in ys if v is not None]
    if not drawn:
        ax.text(0.5, 0.5, "concatenation test\nnot run", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS["annot"], color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        mark_empty(ax, f"{p}: concatenation test not run on this panel")
        return

    # y range spans the drawn lifts and always includes zero, since zero is the claim being read
    # against and a panel whose axis excluded it would hide the sign of every bar.
    span = [v for v in drawn] + [0.0]
    span += [v + e for v, e in zip(ys, es) if v is not None and np.isfinite(e)]
    span += [v - e for v, e in zip(ys, es) if v is not None and np.isfinite(e)]
    lo, hi = min(span), max(span)
    pad = 0.28 * max(hi - lo, 1e-9)
    y0, y1 = lo - pad, hi + pad
    ax.set_ylim(y0, y1)


    bw = bw if bw is not None else (0.62 if compact else 0.74)
    for xi_, v, e, c, ok in zip(x, ys, es, cs, present):
        if ok:
            # yerr=None, not yerr=[nan]: a cell with no interval gets NO whisker at all, which
            # is visibly different from a short one.
            ax.bar([xi_], [v], width=bw, color=c, edgecolor=INK, linewidth=0.7,
                   yerr=([e] if np.isfinite(e) else None),
                   error_kw=dict(elinewidth=0.9, capsize=1.8, capthick=0.9,
                                 ecolor=INK, zorder=6), zorder=3)
        else:
            # NOT RUN, drawn as an empty slot rather than omitted. A missing bar and a bar at the
            # axis floor look identical, and here the two mean opposite things -- "no measurement"
            # against "the embedding destroyed the score".
            ax.bar([xi_], [y1 - y0], bottom=y0, width=bw, facecolor="none",
                   edgecolor=STYLE["grid"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
            ax.text(xi_, y0 + 0.5 * (y1 - y0), "not run", rotation=90, ha="center", va="center",
                    fontsize=FS["annot"] - 2, color="#8A8A8A", zorder=4)

    # NO per-bar tick labels. Eleven of them do not fit at any panel width in this set, and they
    # would repeat what the colour already says -- the legend names the three embeddings once and
    # the group name names the classical base. Ticks are the group names alone.
    ax.set_xticks(group_centres)
    # Rotated in BOTH layouts: "desc+ECFP4" beside "ECFP4" needs ~0.9in horizontally and the
    # widest panel in this set gives ~0.55in per group.
    # HORIZONTAL tick labels. With Mordred retired there are three references, not five, and their
    # names fit side by side at this panel width -- the 30-degree rotation was costing roughly a
    # third of an inch of PAGE HEIGHT per row for labels that no longer need it, which is the
    # scarce direction. If a longer reference name is ever added this needs revisiting; the check
    # is whether the labels collide, not whether they look tight.
    ax.set_xticklabels([g for g, _, _ in GROUPS], fontsize=FS["annot"] - 1)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    if compact:
        ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))


# Wrapped forms for the VERTICAL legend. "CLIMB + descriptors + ECFP4" on one line is 3.3in wide,
# which forces the legend into a horizontal strip; broken at the + signs it is 3 short lines and
# the whole key becomes a tall narrow block (user 2026-08-19).
WRAPPED = {"CLIMB alone": "CLIMB\nalone",
           "CLIMB + descriptors": "CLIMB\n+ descriptors",
           "CLIMB + descriptors + ECFP4": "CLIMB\n+ descriptors\n+ ECFP4"}


def legend_handles(skip_anchor=False, wrap=False):
    """Three ROLES, not six bars, plus the dotted baseline.

    The same three colours repeat in both groups -- classical block alone, + CLIMB, + CheMeleon --
    so naming them once is the whole key; which classical block a group uses is written under its
    own bars. Six entries would just be the same three words twice.

    The dotted line gets its OWN entry (user 2026-08-19). It used to be the anchor's bar as well,
    which was redundant; with the bar gone the line is the only thing carrying fp+desc, and an
    unlabelled reference line is a number the reader cannot name.

    `skip_anchor` and `wrap` are kept for the fig_E+F caller's signature; the role legend is short
    enough that neither changes it now.
    """
    # Driven by ROLE_ORDER, so adding or renaming an embedding cannot leave the legend behind --
    # this line was a literal tuple and broke the moment the second CLIMB arm was added.
    # THE CLASSICAL ADDITIONS NEED NAMING TOO, now that the amber bars are "+ RDKit desc",
    # "+ Mordred" and "+ ECFP4" rather than the single "no embedding" base. Built by walking
    # GROUPS and keeping first sight of each (label, colour), so a legend entry cannot exist for a
    # bar the figure does not draw, nor a bar go unnamed -- the failure the literal tuple had.
    seen, out = set(), []
    for _, _, adds in GROUPS:
        for _, lab, colour in adds:
            key = (str(lab), colour)
            if key in seen:
                continue
            seen.add(key)
            out.append(Patch(facecolor=colour, edgecolor=INK, lw=0.8, label=str(lab)))
    return out


def main():
    d = compute()

    # ---- data record: every task the experiment covers, not just the canonical panels ----
    rows = []
    for task, metric in ALL_TASKS.items():
        g = d[(d.task == task) & (d.metric == metric)].set_index("features")
        if BASE not in g.index:
            continue
        sign = -1 if metric in LOWER_BETTER_METRIC else 1
        row = dict(task=task, metric=metric,
                   in_canonical_panels=int(task in {v for v in PANEL_TASK.values() if v}))
        for f, _, _ in FEATURES:
            row[f] = round(float(g.loc[f, "mean"]), 4) if f in g.index else ""
            _sd = float(g.loc[f, "std"]) if f in g.index else float("nan")
            row[f + "_sd"] = round(_sd, 4) if np.isfinite(_sd) else ""
        # one delta per CLIMB arm; an arm with no run leaves its columns blank rather than
        # inheriting the other arm's verdict
        for role, key in CONCAT_ARMS:
            t = CONCAT_TAG[role]
            if key not in g.index:
                row[f"delta_{t}"] = row[f"sd_{t}"] = row[f"beats_sd_{t}"] = ""
                continue
            delta = sign * (float(g.loc[key, "mean"]) - float(g.loc[BASE, "mean"]))
            sd = float(g.loc[key, "std"])
            sd = 0.0 if not np.isfinite(sd) else sd   # 1-replicate cell: no SD to beat
            row[f"delta_{t}"] = round(delta, 4)
            row[f"sd_{t}"] = round(sd, 4)
            row[f"beats_sd_{t}"] = "yes" if delta > sd else "no"
        rows.append(row)
    OUTDIR.mkdir(exist_ok=True)
    cols = ["task", "metric", "in_canonical_panels"] + \
           [c for f, _, _ in FEATURES for c in (f, f + "_sd")] + \
           [c for role, _ in CONCAT_ARMS
            for c in (f"delta_{CONCAT_TAG[role]}", f"sd_{CONCAT_TAG[role]}",
                      f"beats_sd_{CONCAT_TAG[role]}")]
    DATADIR.mkdir(parents=True, exist_ok=True)
    with open(DATADIR / "fig_F.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- the figure: canonical six ----
    # SHORTER, AT THE SAME A4 WIDTH (Leif 2026-08-23: page length is the scarce resource, width is
    # free). 5.1in -> 3.95in. The room comes from dropping five repeated y-labels and tightening
    # the legend gap, not from shrinking the panels: the bars keep their height.
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"], 3.45))
    for i, (ax, p) in enumerate(zip(axes.ravel(), PANEL_ORDER)):
        draw_panel(ax, d, p, ylabel=(i % 3 == 0))

    handles = legend_handles()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.052), ncol=row_ncol(handles),
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.2,
               borderpad=0.30, **LEGEND_BOX, labelcolor=INK)
    # SAY THAT THE INTERVALS ARE MISSING, ON THE CANVAS. Bars with no whiskers read as precise;
    # bars with a stated reason read as pending. The uncertainty on a lift is the spread of the
    # PER-FOLD DIFFERENCE, which the aggregate tables cannot express -- see PAIRED_READY.
    if not PAIRED_READY:
        fig.text(0.5, 0.062, "intervals pending: a lift needs the spread of the per-fold "
                             "difference, not either arm's own SD",
                 ha="center", va="bottom", fontsize=FS["legend"] - 1, color=INK, alpha=0.75)
    fig.tight_layout(rect=(0, 0.055 + (0.022 if not PAIRED_READY else 0), 1, 1), h_pad=0.9, w_pad=1.0)
    # PAPER FIGURE as of 2026-08-23, not a component. fig_E+F was split into two standalone
    # figures, so this belongs in figures_v2/ proper alongside fig_A/fig_B/fig_G rather than in
    # panels/, which holds only the pieces that other figures assemble.
    save(fig, "fig_F")
    plt.close(fig)

    print("\nFig F — does concatenating CLIMB onto the classical features help?")
    print("  (delta signed so + = concatenation helped)")
    # THE SOURCE TAGS ARE AMBIGUOUS AND THE COLUMN NAMES INHERIT THEM. `CLM` is not "CLIMB", it is
    # the UNSUPERVISED arm specifically -- concat_redundancy.py hardcodes unsup_8M -- so a column
    # headed desc+CLM reads as the family when it means one member of it. The tags are kept as-is
    # because they are the join key into the source tables and renaming them would break that
    # trail; the mapping is printed instead, every run, so it travels with the numbers.
    print("  source tag -> arm:  " + ",  ".join(
        f"{ROLE_SUFFIX[r]} = {r.lstrip('+ ')}" for r in ROLE_ORDER if ROLE_SUFFIX[r]) + "\n")
    head = "".join(f"{f:>16}" for f, _, _ in FEATURES)
    head += "".join(f"{'Δ ' + CONCAT_TAG[role]:>11}{'>SD':>5}" for role, _ in CONCAT_ARMS)
    print(f"  {'task':<13}{'canon':<7}{head}")
    for r in rows:
        line = f"  {r['task']:<13}{'yes' if r['in_canonical_panels'] else '—':<7}"
        for f, _, _ in FEATURES:
            v = r.get(f)
            line += (f"{v:>16.4f}" if isinstance(v, (int, float)) else f"{'—':>16}")
        for role, _ in CONCAT_ARMS:
            t = CONCAT_TAG[role]
            dv = r.get(f"delta_{t}")
            line += (f"{dv:>+11.4f}" if isinstance(dv, (int, float)) else f"{'—':>11}")
            line += f"{r.get(f'beats_sd_{t}') or '—':>5}"
        print(line)
    # ONE VERDICT PER CLIMB ARM (user 2026-08-20). Reporting a single number meant silently
    # choosing which arm "the embedding" refers to, and it was the unsupervised one -- the weaker
    # of the two on most panels. The claim is stronger stated for both, and an arm with no runs
    # says "not run" rather than borrowing the other's verdict.
    #
    # SPLIT BY WHETHER THE TASK IS DRAWN. A bare count over `rows` includes BBBP, ESOL and CBS,
    # which are in the table but not in the canonical panel set, and the only positive is BBBP --
    # so unqualified it reads as though concatenation helped somewhere the reader can see.
    canon = [r for r in rows if r["in_canonical_panels"]]
    print()
    for role, _ in CONCAT_ARMS:
        t = CONCAT_TAG[role]
        have = [r for r in rows if r.get(f"beats_sd_{t}") in ("yes", "no")]
        if not have:
            print(f"  {role}: NOT RUN — no concatenation cells for this arm yet")
            continue
        helped = [r["task"] for r in have if r[f"beats_sd_{t}"] == "yes"]
        hc = [r for r in canon if r.get(f"beats_sd_{t}") == "yes"]
        nc = [r for r in canon if r.get(f"beats_sd_{t}") in ("yes", "no")]
        print(f"  {role}: beat its own SD on {len(helped)}/{len(have)} tasks "
              f"({', '.join(helped) or 'nowhere'}), and on {len(hc)}/{len(nc)} "
              f"of the DRAWN canonical panels")
    print("  wrote figures_v2/fig_F.png/pdf + figure_data/fig_F/fig_F.csv")


if __name__ == "__main__":
    main()
