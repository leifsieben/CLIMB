# ---------- inventory: is each exported figure FINAL, or still moving? ----------
EVAL_BUILD = pd.Timestamp("2026-07-21")   # the eval_v2 build every final figure must be scored by

def _sums(pattern,sub="moleculenet"):
    return [Path(q) for q in glob.glob(f"figure_data/{pattern}/{sub}/suite_summary.json")]

# what each figure actually reads off disk
FIG_SOURCES={
  "A1.a": _sums("climb_v2_phase2/*"),
  "A1.b": _sums("climb_v2_phase2/*","moleculenet_cv"),
  "A2.a": _sums("climb_v2_phase2/*","moleculenet_cv"),
  "A2.b": _sums("climb_v2_phase2/*","moleculenet_cv"),
  "B1p1": [Path(q) for q in glob.glob(f"{LE}/*/moleculenet/moleculenet_summary.csv")],
  # E1's frozen line is read per head seed straight from the phase-2 summaries, so its verdict
  # depends on those too -- not only on the cached eval_ceiling.csv.
  "E1"  : [q for q in (CEIL,CEIL_SUP) if q.exists()]
          +[q for r in ("random_baseline_00","unsup_2M","unsup_8M","unsup_24M",
                        f"skip_{E1_SUP_RECIPE}_2M",f"skip_{E1_SUP_RECIPE}_8M",f"skip_{E1_SUP_RECIPE}_24M")
              for q in _sums(f"climb_v2_phase2/{r}")],
  "B2"  : _sums("climb_v2_phase2/*"),
  "C1J1": _sums("climb_v2_ablation_dedup/*")+_sums("climb_v2_phase2/unsup_2M"),
  "I1"  : _sums("climb_v2_phase2/*","moleculenet_cv"),
  "H1"  : _sums("climb_v2/scaling_*"),
}
# blockers that are MISSING work rather than stale scoring
_missing_hiv=sorted(set(DF.run)-set(DF[DF.task=="HIV"].run))
_u2s_seeds=DF[(DF.regime=="unsup2sup")&(DF.budget_label==MATCHED_BUDGET)].seed.nunique()
_missing_hiv_cv=sorted(set(DF_CV.run)-set(DF_CV[DF_CV.task=="HIV"].run))
# Scheme-aware: the hold-out half of the E1 wave landed before the CV half, so a single _e2e flag
# computed from DF would clear A1.b/A2's blocker while their e2e bar is still absent.
_e2e   =len(DF[DF.regime=="no_pretrain_e2e"])
_e2e_cv=len(DF_CV[DF_CV.regime=="no_pretrain_e2e"])
BLOCKERS={
  "A1.a": ([f"{len(_missing_hiv)} runs lack HIV NEF1%"] if _missing_hiv else [])
        +([f"unsup→sup has {_u2s_seeds} pretraining seed: its error bar is head-seed only"]
          if _u2s_seeds<2 else [])
        +([] if _e2e else ["no_pretrain_end_to_end (E1) not run"]),
  "A1.b": ([f"{len(_missing_hiv_cv)} runs lack HIV in the CV eval"] if _missing_hiv_cv else [])
        +([] if _e2e_cv else ["no_pretrain_end_to_end has no CV eval yet "
                             "(only the hold-out half of the E1 wave has landed)"]),
  "A2.a": ([f"{len(_missing_hiv_cv)} runs lack HIV in the CV eval"] if _missing_hiv_cv else [])
        +([] if _e2e_cv else ["no_pretrain_end_to_end has no CV eval yet "
                             "(only the hold-out half of the E1 wave has landed)"])
        # Derived, not asserted: this was a hardcoded string that kept claiming a gap after the
        # runs had landed. A blocker that cannot clear itself is just a stale caption.
        +([] if len(glob.glob("figure_data/climb_v2_phase2/skip_dense_96M/moleculenet_cv/suite_summary.json")
                   +glob.glob("figure_data/climb_v2_phase2/u2s_*_from48M/moleculenet_cv/suite_summary.json"))>=6
           else ["96M rung / unsup→sup 48M rung still lack a CV eval"])
        +([] if glob.glob("figure_data/climb_v2_phase2/unsup_100M/moleculenet_cv/suite_summary.json")
           else ["~100M-molecule unsupervised rung not yet scored"]),
  "A2.b": ([f"{len(_missing_hiv_cv)} runs lack HIV in the CV eval"] if _missing_hiv_cv else [])
        +([] if _e2e_cv else ["no_pretrain_end_to_end has no CV eval yet "
                             "(only the hold-out half of the E1 wave has landed)"])
        +([] if glob.glob("figure_data/climb_v2_phase2/unsup_100M/moleculenet_cv/suite_summary.json")
           else ["~100M-molecule unsupervised rung not yet scored"]),
  "B2": ([] if _have_b2 else ["corrupt_mlm_8M / corrupt_mtr_8M still training"]),
  "E1": list(_E1_PENDING and [f"sup_only ladder not fine-tuned yet: {', '.join(_E1_PENDING)}"] or []),
  # A figure is not FINAL just because its files are consistent -- a declared arm with no runs is
  # a gap too. B1p1 carries no_pretrain_e2e in its legend as NOT RUN; C1J1's whole wave was scored
  # only on the small single hold-out, so its effect sizes are the weakest in the notebook.
  # COVERAGE, not existence. A non-empty glob says ONE cell landed, not that the sweep finished --
  # the same "the output file appeared, so it must be done" reasoning that let truncated runs into
  # earlier analyses. Count against the expected total, and show the fraction.
  "B1p1": ([] if len(glob.glob(f"{LE}/e2e_n*_s*/moleculenet/moleculenet_summary.csv"))>=13 else
           [f"no_pretrain_end_to_end sweep is "
            f"{len(glob.glob(f'{LE}/e2e_n*_s*/moleculenet/moleculenet_summary.csv'))}/13 cells "
            f"(n=100/300/1000/3000 x 3 subsample seeds, + full)"]),
  "C1J1": ([] if len(glob.glob(f"figure_data/{ABL}/*/moleculenet_cv/suite_summary.json"))>=10 else
           [f"ablation-wave CV is "
            f"{len(glob.glob(f'figure_data/{ABL}/*/moleculenet_cv/suite_summary.json'))}/10 runs; "
            f"the panel is still on the single hold-out (113-204 test mols)"]),
  "H1"  : ([] if H1_SUB=="moleculenet_cv"
           else ["round-1 wave has NO CV eval and no saved encoders - retraining under way"]),
  "I1"  : ([] if _I1_HAVE else ["baseline is now no_pretrain_e2e; its CV predictions land with "
                                "the E1 wave"]),
}

FIGS=[("A1.a","figA1a_best_model_headline_holdout","which model performs best (8M, single scaffold hold-out)"),
      ("A1.b","figA1b_best_model_headline_cv","which model performs best (8M, pooled 5-fold scaffold CV)"),
      ("A2.a","figA2a_scaling_tokens","scaling in training tokens"),
      ("A2.b","figA2b_scaling_unique_molecules","scaling in unique molecules seen"),
      ("B1p1","figB1p1_label_efficiency_train_test","label-efficiency + fit/generalize mechanism"),
      ("E1","figE1_H5_eval_ceiling","H5: is the sup/unsup ordering a frozen-probe artifact?"),
      ("B2","figB2_corrupted_control"+("" if _have_b2 else "_PLACEHOLDER"),"corrupted-pretraining control"),
      ("C1J1","figC1J1_sft_family_transfer","SFT family: which helps, how much, does it track chemistry"),
      ("I1","figI1_memorization_vs_representation","memorization vs representation"),
      ("H1","figH1_canonical_vs_enumerated","canonical vs enumerated SMILES")]
out=Path(STYLE["outdir"]); final=[]
print(f"{'fig':<6} {'verdict':<12} {'exported':<44} {'files':<7} status")
for fid,name,what in FIGS:
    both=(out/f"{name}.png").exists() and (out/f"{name}.pdf").exists()
    srcs=[q for q in FIG_SOURCES.get(fid,[]) if q.exists()]
    old=[q for q in srcs if pd.Timestamp(q.stat().st_mtime,unit="s")<EVAL_BUILD]
    blk=list(BLOCKERS.get(fid,[]))
    # What disqualifies a figure is MIXING two scorers, not being scored by the older one. H1
    # compares canonical vs enumerated entirely within one round-1 wave: internally consistent,
    # so it is final even though it predates the current build. A1 mixing 07-19 and 07-22
    # summaries is not.
    stale = old if (old and len(old)<len(srcs)) else []
    if stale: blk.insert(0,f"MIXED scorers: {len(stale)}/{len(srcs)} summaries predate the rest")
    elif old: what += "  [scored by the pre-2026-07-21 eval_v2, but consistently so]"
    verdict="FINAL" if (both and not blk) else ("PROVISIONAL" if both else "NOT EXPORTED")
    if verdict=="FINAL": final.append(fid)
    print(f"{fid:<6} {verdict:<12} {name:<44} {'png+pdf' if both else 'MISSING':<7} "
          f"{'; '.join(blk) if blk else what}")
    for q in sorted(stale)[:3]:
        print(f"{'':<6} {'':<12}   stale: {'/'.join(q.parts[1:3])}")

print(f"\nFINAL - safe to quote: {', '.join(final) if final else '(none)'}")
prov=[f for f,_,_ in FIGS if f not in final]
if prov: print(f"PROVISIONAL - do NOT quote effect sizes yet: {', '.join(prov)}")
stray=sorted(p.name for p in out.glob("*") if p.stem not in {n for _,n,_ in FIGS})
print("stray files in figures_out (not one of the exported set):", stray or "none")