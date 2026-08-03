# ---------- C1J1 + I1 fused · one stacked 5-panel figure ----------
# Re-draws the five subfigures already built above -- (a,b,c) from the C1J1 cell and (d,e) from the
# I1 cell -- into one stacked, left-aligned figure for the paper. It deliberately REUSES the data
# those cells computed (avg, H, _norm/_vmax, MAT_ROWS, pts + X/Y/TK/tcol/a/b/r/rho/p for C1J1;
# pairs, sim/nov/se_s/se_n, binned_lift for I1; the H10 fit line is recomputed locally from
# X/Y because C1J1's global `a` is later reused as a loop variable) instead of recomputing, so
# this panel can never
# disagree with the standalone figures. It MUST run after both of them.
#   row 1:  a  (mean lift per SFT arm)        |  b  (SFT-family -> eval-task transfer matrix)
#   row 2:  c  (H10: lift vs chemical similarity)             [legend sits in the freed right space]
#   row 3:  d  (I1 quartile lift)             |  e  (I1 lift vs corpus similarity)
figF=plt.figure(figsize=(STYLE["col2"],10.8))
_outer=figF.add_gridspec(3,1,height_ratios=[1.05,0.98,0.98],hspace=0.42)
_r0=_outer[0].subgridspec(1,2,width_ratios=[1,1.30],wspace=0.62)
axA=figF.add_subplot(_r0[0]); axB=figF.add_subplot(_r0[1])
_r1=_outer[1].subgridspec(1,2,width_ratios=[1.75,1.0],wspace=0.04)
axC=figF.add_subplot(_r1[0])                    # legend overflows to the right, into the empty col
_r2=_outer[2].subgridspec(1,2,width_ratios=[1,1],wspace=0.38)
axD=figF.add_subplot(_r2[0]); axE=figF.add_subplot(_r2[1])

# ----- (a) mean lift per arm (C1J1 data) -----
_cols=["#555555" if a.startswith("unsup_only") else
       (PALETTE["black"] if a=="Morgan+XGBoost" else PALETTE["blue"]) for a in ARM_ORDER]
axA.barh(range(len(ARM_ORDER)),avg.values,color=_cols,edgecolor="white",lw=0.4)
axA.axvline(0,color="#333",lw=0.8)
axA.set_yticks(range(len(ARM_ORDER))); axA.set_yticklabels(ARM_ORDER); axA.invert_yaxis()
axA.set_xlabel(f"avg lift over {C1J1_FLOOR_LABEL} (%)")
axA.set_title("Which SFT label type helps?\n(all arms: unsup→sup from one 2M-FP MLM base)",pad=6)
panel_tag(axA,"a",dx=-0.62)

# ----- (b) SFT-family transfer matrix (C1J1 data) -----
_im=axB.imshow(H.values,cmap="PuOr_r",norm=_norm,aspect="auto")
axB.set_xticks(range(len(CORE_TASKS)))
axB.set_xticklabels([f'{TASKS[t]["pretty"]}\n[{TASKS[t]["domain"]}]' for t in CORE_TASKS],
                    fontsize=STYLE["fs_annot"]-1.5,rotation=30,ha="right")
axB.set_yticks(range(len(MAT_ROWS))); axB.set_yticklabels(MAT_ROWS); axB.grid(False)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v=H.values[i,j]
        if np.isfinite(v):
            axB.text(j,i,f"{v:+.0f}",ha="center",va="center",fontsize=STYLE["fs_annot"],
                     color="white" if abs(v)>14 else "#222")
_cbt=[t for t in (-_vmax,-20,-5,0,5,20,_vmax) if abs(t)<=_vmax]
_cb=figF.colorbar(_im,ax=axB,fraction=0.046,pad=0.03,ticks=_cbt)
_cb.ax.yaxis.set_major_formatter(ticker.FixedFormatter([f"{t:+.0f}".replace("+0","0") for t in _cbt]))
_cb.set_label("lift (%), symlog scale",fontsize=STYLE["fs_legend"])
_cb.ax.tick_params(labelsize=STYLE["fs_annot"]-1)
axB.set_title("SFT family → eval task",pad=6); panel_tag(axB,"b",dx=-0.22)

# ----- (c) H10: lift vs chemical similarity (C1J1 data) -----
if len(pts)>=4:
    for t in [t for t in CORE_TASKS if t in set(TK)]:
        _m=[k==t for k in TK]
        axC.scatter(X[_m],Y[_m],s=30,color=tcol[t],edgecolor="white",lw=0.5,zorder=3,
                    label=TASKS[t]["pretty"])
    # recompute the fit locally: the C1J1 cell's global `a` (polyfit intercept) is later
    # clobbered to a string by its `for a in ARM_ORDER` print loop, so don't lean on it here.
    _b,_a=np.polyfit(X,Y,1); _xs=np.linspace(X.min(),X.max(),50)
    axC.plot(_xs,_a+_b*_xs,color=PALETTE["black"],lw=STYLE["lw_thin"],ls=(0,(4,2)),zorder=2)
    axC.axhline(0,color=PALETTE["grey2"],lw=STYLE["lw_thin"],zorder=1)
    axC.set_xlabel("family–task ECFP4 Tanimoto similarity  —  right = MORE similar →")
    axC.set_ylabel(f"lift over {C1J1_FLOOR_LABEL} (%)")
    axC.set_title(f"H10: lift vs chemical similarity  (n={len(pts)}, "
                  f"r={r:+.2f}, ρ={rho:+.2f}, p={p:.2f})",pad=6)
    axC.legend(title="eval task",loc="upper left",bbox_to_anchor=(1.02,1.0),frameon=False,
               fontsize=STYLE["fs_legend"],title_fontsize=STYLE["fs_legend"],handletextpad=0.3)
    label_all_yticks(axC)
else:
    no_data_watermark(axC,"run scripts/compute_family_task_similarity.py")
    axC.set_xlabel("family–task Tanimoto similarity  —  right = MORE similar →")
    axC.set_ylabel(f"lift over {C1J1_FLOOR_LABEL} (%)")
panel_tag(axC,"c",dx=-0.09)

# ----- (d) I1 quartile lift: most-similar vs most-novel (I1 data) -----
_ylabI=f"lift over {_I1_BASE_LABEL[I1_BASE_KEY]} (%)"
if pairs:
    axD.bar([0,1],[sim,nov],color=[PALETTE["purple"],PALETTE["teal"]],width=0.6,
            yerr=[se_s,se_n],capsize=STYLE["cap_size"],error_kw=dict(lw=STYLE["lw_thin"]))
    for _xi,_v,_e in zip([0,1],[sim,nov],[se_s,se_n]):
        axD.text(_xi,_v+_e+0.6,f"{_v:+.1f}%",ha="center",fontsize=STYLE["fs_annot"])
    axD.axhline(0,color=PALETTE["black"],lw=0.6)
    _lo,_hi=min(0,sim-se_s,nov-se_n),max(0,sim+se_s,nov+se_n)
    axD.set_ylim(_lo-abs(_lo)*0.35-2,_hi+abs(_hi)*0.35+3)
else:
    axD.set_ylim(-5,15); no_data_watermark(axD,_I1_NEED)
axD.set_xticks([0,1])
axD.set_xticklabels(["most corpus-similar\n(top quartile)","most novel\n(bottom quartile)"])
axD.set_ylabel(_ylabI); label_all_yticks(axD); panel_tag(axD,"d",dx=-0.20)

# ----- (e) I1 lift vs corpus similarity (I1 data) -----
_drew=False
for t in I1_TASKS:
    _xs2,_ys2,_lo2,_hi2,_nn2=binned_lift(t)
    if not _xs2: continue
    _drew=True
    _err=np.vstack([np.array(_ys2)-np.array(_lo2),np.array(_hi2)-np.array(_ys2)])
    axE.errorbar(_xs2,_ys2,yerr=_err,color=col.get(t,PALETTE["grey"]),marker="o",lw=STYLE["lw"],
                 capsize=STYLE["cap_size"],label=f"{t} (n/bin≈{int(np.median(_nn2))})")
if _drew:
    axE.axhline(0,color=PALETTE["black"],lw=0.6); axE.legend(loc="best",fontsize=STYLE["fs_legend"])
else:
    axE.set_ylim(-5,15); no_data_watermark(axE,_I1_NEED)
axE.set_xlabel("max ECFP4 Tanimoto to corpus (bin mean)\nright = MORE similar →")
axE.set_ylabel(_ylabI); label_all_yticks(axE); panel_tag(axE,"e",dx=-0.18)

figF.suptitle("Fig C1J1+I1 — supervised-label transfer (a–c) and who benefits from MLM "
              "pretraining (d–e)",fontsize=STYLE["fs_title"],y=0.995)
figF.subplots_adjust(left=0.155,right=0.945,top=0.945,bottom=0.055)
save_fig(figF,"figC1J1_I1_combined"); plt.show()
print("combined C1J1+I1 panel: a–c reuse the C1J1 cell's data, d–e reuse the I1 cell's data "
      "-> identical to the standalone figures by construction.")
