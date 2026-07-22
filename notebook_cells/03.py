# ---- palette, regime registry, task registry: the single source of truth for colour + label ----
# Every figure reads colours from here, so a model has the same colour in A1, A2, B1p1 and B2 and
# the legend transfers between panels without the reader re-learning it.
PALETTE = dict(grey="#8A8A8A", grey2="#C4C4C4", black="#1A1A1A", blue="#4477AA", green="#228833",
               orange="#EE7733", red="#CC3311", teal="#009988", purple="#AA3377", cyan="#66CCEE",
               yellow="#CCBB44", sand="#DDCC77")

sup_recipes = ["dense","sparse_all","dense_plus_sparse","minimol_full","mixed"]
_recipe_pretty = {"dense":"dense (MTR)","sparse_all":"sparse_all","dense_plus_sparse":"dense+sparse",
                  "minimol_full":"minimol_full","mixed":"mixed"}
_recipe_color  = {"dense":PALETTE["orange"],"sparse_all":PALETTE["teal"],"dense_plus_sparse":PALETTE["red"],
                  "minimol_full":PALETTE["purple"],"mixed":PALETTE["cyan"]}
_recipe_marker = {"dense":"^","sparse_all":"v","dense_plus_sparse":"P","minimol_full":"X","mixed":"<"}

REGIME = {  # key : (colour, marker, linestyle, label)
    "random"      :(PALETTE["grey2"], None,(0,(1,1)),  "random (chance)"),
    "no_pretrain" :(PALETTE["grey"],  "o", (0,(3,2)),  "no_pretrain (frozen)"),
    "ecfp4"       :(PALETTE["black"], "D", (0,(4,1)),  "Morgan+XGBoost (ECFP4)"),
    "fp_desc"     :(PALETTE["sand"],  "s", (0,(4,1)),  "Morgan+desc+XGBoost"),
    "unsup_only"  :(PALETTE["blue"],  "o", "-",        "unsup_only (MLM)"),
    "unsup2sup"   :(PALETTE["green"], "s", "-",        "unsup→sup (MLM→SFT)"),
}
for r in sup_recipes:
    REGIME[f"sup_only:{r}"]=(_recipe_color[r],_recipe_marker[r],"-",f"sup_only: {_recipe_pretty[r]}")
# unsup->sup is reported PER SFT recipe (A1 splits it, so it is comparable with sup_only's five
# arms instead of averaging five different recipes into one bar). Green family = the regime, so
# the reader still sees "this is unsup->sup" at a glance; the shade tracks the recipe.
_u2s_color={"dense":"#228833","sparse_all":"#88CCAA","dense_plus_sparse":"#117744",
            "minimol_full":"#44BB99","mixed":"#7FB069"}
for r in sup_recipes:
    REGIME[f"unsup2sup:{r}"]=(_u2s_color[r],_recipe_marker[r],"-",f"unsup→sup: {_recipe_pretty[r]}")
# Random-init encoder FINE-TUNED end-to-end on the eval task (E1). Sits next to the frozen
# no_pretrain bar and answers a different question, so it gets its own grey, not a shared one.
REGIME["no_pretrain_e2e"]=("#555555","o",(0,(3,2)),"no_pretrain (end-to-end)")
rc_color=lambda k:REGIME[k][0]; rc_marker=lambda k:REGIME[k][1]
rc_ls   =lambda k:REGIME[k][2]; rc_label =lambda k:REGIME[k][3]

TASKS = {
    "ESOL"          :dict(metric="RMSE",   higher_better=False,pretty="ESOL", domain="solubility"),
    "BBBP"          :dict(metric="ROC-AUC",higher_better=True, pretty="BBBP", domain="ADMET"),
    "BACE"          :dict(metric="ROC-AUC",higher_better=True, pretty="BACE", domain="binding"),
    "Tox21"         :dict(metric="ROC-AUC",higher_better=True, pretty="Tox21",domain="toxicity"),
    "QM7"           :dict(metric="RMSE",   higher_better=False,pretty="QM7",  domain="quantum"),
    # HIV is scored by NEF1% (top-1% early enrichment) -- the virtual-screening readout. Chance is
    # the active fraction (1443/41120), not 0.5.
    "HIV"           :dict(metric="NEF1%",  higher_better=True, pretty="HIV",  domain="virtual screening",
                          suite_key="HIV_nef1", chance=0.0351),
    "Lipophilicity" :dict(metric="RMSE",   higher_better=False,pretty="Lipo.",domain="ADMET"),
}
CORE_TASKS=["ESOL","BBBP","BACE","Tox21","QM7","HIV"]
BUDGET_FP={"2M":2e6,"8M":8e6,"24M":24e6,"48M":48e6,"96M":96e6}
MATCHED_BUDGET="8M"          # largest budget where every arm ran to completion
U2S_SFT_FP=2e6               # unsup->sup spends its MLM base PLUS a 2M-FP SFT stage

def chance_value(task):
    return TASKS[task].get("chance", 0.5 if TASKS[task]["higher_better"] else 1.0)
def mlabel(task):
    m=TASKS[task]; return f'{m["metric"]} {"↑" if m["higher_better"] else "↓"}'
def ttitle(task,oneline=False):
    sep=" " if oneline else "\n"; return f'{TASKS[task]["pretty"]}{sep}({mlabel(task)})'
def lift(value,task,floor):
    """Improvement over the no_pretrain floor; positive = better for BOTH metric directions."""
    if not np.isfinite(floor) or floor==0 or not np.isfinite(value): return np.nan
    return (value-floor)/floor if TASKS[task]["higher_better"] else (floor-value)/floor

# ---- shared drawing helpers ----
def set_fp_axis(ax,budgets):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(list(budgets)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f"{int(b/1e6)}M" for b in budgets]))
    ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.tick_params(axis="x",which="minor",bottom=False)
def label_all_yticks(ax,n=5):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=n)); ax.yaxis.set_minor_locator(ticker.NullLocator())
def panel_tag(ax,tag,dx=-0.12,dy=1.02):
    ax.text(dx,dy,tag,transform=ax.transAxes,fontsize=STYLE["fs_panel_tag"],fontweight="bold",va="bottom",ha="right")
def add_chance_line(ax,task):
    ax.axhline(chance_value(task),color="#999999",ls=(0,(1,1.5)),lw=1.0,zorder=5)
def no_data_watermark(ax,need):
    ax.text(0.5,0.5,f"NO DATA\n{need}",transform=ax.transAxes,ha="center",va="center",
            fontsize=STYLE["fs_annot"]+1,color="#B00020",fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",fc="#FCE8E8",ec="#B00020",lw=0.8),zorder=5)
    for s in ax.spines.values(): s.set_linestyle((0,(3,3)))
def save_fig(fig,name):
    for ext in STYLE["save_formats"]: fig.savefig(f'{STYLE["outdir"]}/{name}.{ext}')
    print("saved",name)

import numpy as np
print("registry ready:",len(REGIME),"regimes,",len(TASKS),"tasks")