# The classical anchors moved on 2026-08-20, and the movement has two terms

The four XGBoost fingerprint anchors were re-run at the current featurizer (`ecfp4_stereo` for
ECFP4/ECFP4+desc, `morgan_r3_counts` for R3FP/R3FP+desc) and pooled over three head-seed triples
instead of one. Their published numbers therefore moved for **two reasons at once**, and the two
are separable only because the pre-stereo directories were quarantined rather than overwritten.

    featurizer term   PRESTEREO base vs stereo base, SAME seed triple (42/117/709)
    seed term         stereo base vs the pooled three dirs, SAME featurizer

## Measured

| panel | arm | featurizer | seed | net | head-seed SD |
|---|---|---|---|---|---|
| Ames (ROC-AUC, higher better) | ECFP4 | **+0.0038** helps | −0.0014 | +0.0024 | 0.0013 |
| MoleculeACE (macro RMSE, lower better) | ECFP4 | **+0.0015** hurts | −0.0008 | +0.0007 | 0.0010 |
| MoleculeACE (macro RMSE, lower better) | ECFP4+desc | **−0.0020** helps | −0.0004 | −0.0024 | 0.0005 |

## Why there is no cross-panel sentence about the featurizer

The same fingerprint change moves the same arm in **opposite directions** on the two panels — and
the sign convention flips with the metric, so `+0.0038` (Ames, higher-better) and `+0.0015`
(MoleculeACE, lower-better) do not even mean the same thing. Any sentence spanning both panels
would have to get the metric direction right *and* survive the effect reversing. **State the
decomposition per panel or not at all.**

Reporting the NET alone would have been worse than saying nothing. On Ames the net is +0.0024,
which reads as "negligible" — while burying a featurizer effect nearly three times larger under a
partial cancellation against the seed term, which points the other way.

## What does generalise

The seed term is about one head-seed standard deviation on every panel and arm measured
(−0.0014 vs 0.0013, −0.0008 vs 0.0010, −0.0004 vs 0.0005). So the original single triple was not
*wrong*, it was one draw, and pooling corrected it by roughly the amount one draw is expected to be
off. Write it that way round: "the published number changed when we added seeds" reads as
instability; "one triple was unrepresentative by about one SD" is the correction it actually is.

## Provenance

Pre-stereo copies are preserved as `*_PRESTEREO` on S3 on both the MoleculeACE and Polaris tracks —
they are an INPUT to the decomposition above, not only a vintage record, so they must not be
cleaned up. Directories produced in this pass carry `fp_variant` in `verified.json`; directories
untouched by it carry nothing, and were deliberately not backfilled, since their variant is known
from the runner rather than from the artefact. Audit check 18 enforces that an arm never pools two
vintages and reports the unlabelled-beside-labelled case as UNVERIF rather than pass or fail.
