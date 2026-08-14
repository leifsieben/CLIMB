# e2e recipe test — verdict (2026-08-14)

**Question (from LS):** are we deploying the end-to-end CLM in its *best* way? In prior experience
e2e was very competitive when the downstream task had a lot of data — is the CLIMB e2e arm only
losing here because our suite tasks are tiny?

**Test:** re-run the best-supervised e2e arm (`skip_dense_8M`) with a stronger recipe
(lr 5e-5, 40 epochs, patience 8, native-range prediction clip) on the **largest tasks in each
suite** (5–6k train), 3 seeds, and compare against the default recipe and the XGBoost anchors.

## MoleculeACE — 4 largest tasks (overall RMSE, lower = better)

| task | e2e tuned | e2e default | XGBoost (fp+desc) | XGBoost (fp) |
|---|---|---|---|---|
| CHEMBL234_Ki | 0.715 | 0.749 | **0.639** | 0.654 |
| CHEMBL214_Ki | 0.734 | 0.752 | **0.673** | 0.675 |
| CHEMBL233_Ki | 0.847 | 0.870 | **0.751** | 0.808 |
| CHEMBL244_Ki | 0.798 | 0.822 | **0.724** | 0.730 |
| **mean** | 0.774 | 0.798 | **0.697** | 0.717 |

## Polaris — 4 largest tasks (task-defined primary metric)

| task | metric | e2e tuned | e2e default | XGBoost (fp+desc) | XGBoost (fp) |
|---|---|---|---|---|---|
| tdcommons/ames | ROC-AUC ↑ | 0.823 | 0.825 | **0.870** | 0.836 |
| tdcommons/ld50-zhu | MAE ↓ | 0.650 | 0.676 | **0.602** | 0.626 |
| tdcommons/lipophilicity-astrazeneca | MAE ↓ | 0.549 | 0.584 | **0.529** | 0.639 |
| tdcommons/ppbr-az | MAE ↓ | 8.571 | 8.511 | **8.452** | 10.700 |

## Verdict

1. **The stronger recipe helps, modestly and only on regression.** Tuning improves e2e RMSE/MAE by
   ~0.025–0.036 on the regression tasks (MoleculeACE mean 0.798→0.774; ld50-zhu 0.676→0.650;
   lipophilicity 0.584→0.549). On the classification/ROC tasks it is neutral (ames 0.825→0.823;
   ppbr-az essentially unchanged). So the default recipe was **not** grossly mis-deployed — there is
   a real but small amount of headroom, already captured.

2. **Even tuned, and even on the biggest tasks, e2e loses to XGBoost(fp+desc) on all 8 tasks tested.**
   The size hypothesis is not the explanation: these are the largest tasks in each suite (5–6k train),
   and the descriptor-augmented anchor still wins every one.

3. **The e2e CLM is not incompetent — it beats plain Morgan+XGBoost (fp) on the two physchem
   regression tasks** (lipophilicity 0.549 vs 0.639; ppbr-az 8.57 vs 10.70) and is close on the
   others. The model to beat is **Morgan + RDKit descriptors (fp+desc)**, and it is not beaten.

**Bottom line:** the finding is robust across both benchmark families and holds under a stronger
recipe on the largest available tasks — consistent with the paper's headline and with van Tilborg
et al.'s own MoleculeACE result that classical models match or beat deep models on these targets.

Source: `figure_data/chemeleon_suite/{moleculeace,polaris}/skip_dense_8M_e2e_tuned/` (3 seeds),
scored vs `skip_dense_8M_e2e`, `fp_desc`, `ecfp4`.
