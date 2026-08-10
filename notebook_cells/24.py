# ---------- C1J1 + I1 fused · one stacked 5-panel figure ----------
# Re-draws the five subfigures already built above -- (a,b,c) from the C1J1 cell and (d,e) from the
# I1 cell -- into one stacked, left-aligned figure for the paper. It deliberately REUSES the data
# those cells computed (avg, H, _norm/_vmax, MAT_ROWS, pts + X/Y/TK/tcol/a/b/r/rho/p for C1J1;
# pairs, sim/nov/se_s/se_n, mpairs/mem/se_m, binned_lift for I1; the H10 fit line is recomputed from
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

# ----- (d) I1 quartile lift (non-memorized) + the excluded corpus-match group (I1 data) -----
_ylabI=f"lift over {_I1_BASE_LABEL[I1_BASE_KEY]} (%)"
if pairs:
    _barsD=[("most corpus-similar\n(top quartile,\nnot identical)",sim,se_s,"#1b5e20"),
            ("most novel\n(bottom quartile)",nov,se_n,"#66bb6a")]
    if mpairs: _barsD.append(("corpus-identical\n(Tanimoto=1.0,\nexcluded)",mem,se_m,"#9e9e9e"))
    _xpD=list(range(len(_barsD)))
    axD.bar(_xpD,[b[1] for b in _barsD],color=[b[3] for b in _barsD],width=0.6,
            yerr=[b[2] for b in _barsD],capsize=STYLE["cap_size"],error_kw=dict(lw=STYLE["lw_thin"]))
    for _xi,_b in zip(_xpD,_barsD):
        axD.text(_xi,_b[1]+_b[2]+0.6,f"{_b[1]:+.1f}%",ha="center",fontsize=STYLE["fs_annot"])
    axD.axhline(0,color=PALETTE["black"],lw=0.6)
    _dv=[b[1] for b in _barsD]; _de=[b[2] for b in _barsD]
    _lo,_hi=min(0,*(v-e for v,e in zip(_dv,_de))),max(0,*(v+e for v,e in zip(_dv,_de)))
    axD.set_ylim(_lo-abs(_lo)*0.35-2,_hi+abs(_hi)*0.35+3)
    axD.set_xticks(_xpD); axD.set_xticklabels([b[0] for b in _barsD],fontsize=STYLE["fs_annot"]-0.5)
else:
    axD.set_ylim(-5,15); no_data_watermark(axD,_I1_NEED); axD.set_xticks([])
axD.set_ylabel(_ylabI); label_all_yticks(axD)

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
axE.set_xlabel("max ECFP4 Tanimoto to corpus (bin mean)")
axE.set_ylabel(_ylabI); label_all_yticks(axE); panel_tag(axE,"e",dx=-0.18)

_suptitle(figF, "Fig C1J1+I1 — supervised-label transfer (a–c) and who benefits from MLM "
              "pretraining (d–e)",fontsize=STYLE["fs_title"],y=0.995)
figF.subplots_adjust(left=0.155,right=0.945,top=0.945,bottom=0.055)
# Left-align the three left-column panels. Panel a is a horizontal bar chart whose long arm-name
# labels overhang further left than c/d's y-axis labels, so on a tight-cropped export a reads as
# flush-left while c/d sit indented. Indent a's axes so its label block lands in the same left band,
# then stack the a/c/d tags in one flush-left column at a shared figure-x.
_shift=0.085
_pa=axA.get_position(); axA.set_position([_pa.x0+_shift,_pa.y0,_pa.width-_shift,_pa.height])
for _ax,_t in [(axA,"a"),(axC,"c"),(axD,"d")]:
    figF.text(0.015,_ax.get_position().y1+0.006,_t,fontsize=STYLE["fs_panel_tag"],
              fontweight="bold",va="bottom",ha="left")
save_fig(figF,"figC1J1_I1_combined"); plt.show()
print("combined C1J1+I1 panel: a–c reuse the C1J1 cell's data, d–e reuse the I1 cell's data "
      "-> identical to the standalone figures by construction.")
