# The BBBP positive is real, reproducible, and not a result

2026-08-23. Recorded so it is not rediscovered and mistaken for a finding.

## What it is

In the fig_F concatenation grid, BBBP is the ONLY cell where a CLIMB embedding adds anything.
v2, ROC-AUC, lift over each classical reference, ± paired SD over 5 folds:

    reference            + CLIMB unsup     + CLIMB sup
    RDKit desc             +5.20 ± 3.16     +5.22 ± 3.29
    Mordred                +5.96 ± 3.40     +5.45 ± 3.28
    ECFP4 + RDKit desc     +5.30 ± 2.62     +5.17 ± 2.87

It is not fold noise. Per-fold lift over ECFP4+desc is +9.2, +4.7, +2.1, +6.3, +4.6 — 5/5 positive,
where BACE, HIV and Tox21 are 0/5. Two independently pretrained encoders agree to within 0.1%, and
it is not absorbed by adding fingerprints to the reference.

## Why it is still not a result (Leif's call, and correct)

**BBBP is a weak instrument, and we showed that ourselves on a different metric.** `nef1` pins at
EXACTLY 1.0 on BBBP for all seven feature blocks including plain ECFP4, because the dataset is
76.5% positive and a top-1% enrichment metric saturates at that prevalence. A dataset that
completely degenerates one metric is not one to hang a lone positive on.

**It is 1 of 65 datasets, and it is the one being looked at BECAUSE it is positive.** That is
selection. Sixty-five datasets will produce a 5/5-fold outlier with no effect present.

**It sits near the ceiling.** Raw fold AUCs go 0.907 -> 0.948: +4.6% relative, +0.041 absolute, at
the top of the scale where percentage lift exaggerates.

MoleculeNet BBBP is independently criticised for label provenance and scaffold-split instability.

## What would change the verdict

A second, independent dataset showing the same direction — ideally one that is not
high-prevalence and not near the AUC ceiling. Until then the honest statement is the one the paper
makes: both CLIMB embeddings are informationally redundant given fingerprints and descriptors,
negative on all six canonical panels.

Do NOT re-add BBBP as a seventh fig_F panel on the strength of this. The data stays in
figure_data/fig_F/fig_F.csv as the per-task record.
