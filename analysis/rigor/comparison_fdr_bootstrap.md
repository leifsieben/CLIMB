# Principal 8M comparisons — FDR-corrected + scaffold cluster-bootstrap

Cluster bootstrap: 1000 resamples of whole Bemis-Murcko scaffolds. `point_p` = molecule-level DeLong/Wilcoxon (anti-conservative, indicative only); `point_q`/`boot_q` = Benjamini-Hochberg FDR across all rows; `[ci_lo, ci_hi]` = 95% cluster-bootstrap CI on the metric difference (a - b, oriented so >0 favours a). A CI spanning 0 = no difference detectable.

| pair | task | metric | a_mean | b_mean | delta | fold_t_p | point_p | point_q | ci_lo | ci_hi | boot_p | boot_q | n_scaffolds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unsup_8M vs fp_desc_anchor | ESOL | RMSE↓ | 1.0131 | 0.7204 | 0.2928 | 0.0047 | 0.0 | 0.0 | -0.3739 | -0.1962 | 0.0 | 0.0 | 585 |
| unsup_8M vs fp_desc_anchor | QM7 | RMSE↓ | 199.5783 | 187.2409 | 12.3374 | 0.0141 | 0.0 | 0.0 | -20.1043 | -8.1745 | 0.0 | 0.0 | 3771 |
| unsup_8M vs fp_desc_anchor | BBBP | AUC↑ | 0.9536 | 0.9032 | 0.0503 | 0.03 | 0.0 | 0.0 | 0.029 | 0.0723 | 0.0 | 0.0 | 1123 |
| unsup_8M vs fp_desc_anchor | BACE | AUC↑ | 0.8694 | 0.8668 | 0.0026 | 0.5073 | 0.9427 | 0.9427 | -0.0133 | 0.0138 | 0.956 | 0.98 | 675 |
| unsup_8M vs fp_desc_anchor | Tox21 | AUC↑ | 0.7987 | 0.8249 | -0.0261 | 0.0002 | 0.008 | 0.0159 | -0.0295 | -0.017 | 0.0 | 0.0 | 4099 |
| unsup_8M vs fp_desc_anchor | HIV | AUC↑ | 0.7789 | 0.8159 | -0.037 | 0.0141 | 0.0 | 0.0 | -0.0532 | -0.0216 | 0.0 | 0.0 | 20670 |
| unsup_8M vs fp_desc_anchor | HIV | NEF1%↑ | 0.6819 | 0.7133 | -0.0313 | 0.478 | 0.0 | 0.0 | -0.0711 | 0.0123 | 0.206 | 0.3393 | 20670 |
| unsup_8M vs skip_dense_8M | ESOL | RMSE↓ | 1.0131 | 0.8963 | 0.1168 | 0.0878 | 0.0 | 0.0 | -0.1961 | -0.0597 | 0.0 | 0.0 | 585 |
| unsup_8M vs skip_dense_8M | QM7 | RMSE↓ | 199.5783 | 194.7582 | 4.8201 | 0.201 | 0.0019 | 0.0053 | -11.836 | -0.2862 | 0.044 | 0.0948 | 3771 |
| unsup_8M vs skip_dense_8M | BBBP | AUC↑ | 0.9536 | 0.9404 | 0.0131 | 0.0321 | 0.0002 | 0.0005 | 0.0044 | 0.0242 | 0.006 | 0.0187 | 1123 |
| unsup_8M vs skip_dense_8M | BACE | AUC↑ | 0.8694 | 0.8485 | 0.021 | 0.0514 | 0.0042 | 0.0101 | 0.0003 | 0.0361 | 0.048 | 0.096 | 675 |
| unsup_8M vs skip_dense_8M | Tox21 | AUC↑ | 0.7987 | 0.7961 | 0.0026 | 0.3504 | 0.4734 | 0.5523 | -0.0041 | 0.0094 | 0.396 | 0.4821 | 4099 |
| unsup_8M vs skip_dense_8M | HIV | AUC↑ | 0.7789 | 0.7724 | 0.0065 | 0.5487 | 0.1637 | 0.2467 | -0.0049 | 0.0232 | 0.238 | 0.3702 | 20670 |
| unsup_8M vs skip_dense_8M | HIV | NEF1%↑ | 0.6819 | 0.6578 | 0.0241 | 0.635 | 0.1637 | 0.2467 | -0.0216 | 0.105 | 0.196 | 0.3393 | 20670 |
| u2s_dense_from8M vs skip_dense_8M | ESOL | RMSE↓ | 0.9239 | 0.8963 | 0.0276 | 0.4624 | 0.1806 | 0.2467 | -0.0999 | 0.0307 | 0.456 | 0.532 | 585 |
| u2s_dense_from8M vs skip_dense_8M | QM7 | RMSE↓ | 194.3587 | 194.7582 | -0.3995 | 0.7364 | 0.3275 | 0.4169 | -1.4886 | 2.5329 | 0.786 | 0.8803 | 3771 |
| u2s_dense_from8M vs skip_dense_8M | BBBP | AUC↑ | 0.944 | 0.9404 | 0.0036 | 0.4647 | 0.159 | 0.2467 | -0.0024 | 0.0101 | 0.252 | 0.3714 | 1123 |
| u2s_dense_from8M vs skip_dense_8M | BACE | AUC↑ | 0.8359 | 0.8485 | -0.0126 | 0.1462 | 0.0366 | 0.0684 | -0.0274 | 0.0026 | 0.128 | 0.2389 | 675 |
| u2s_dense_from8M vs skip_dense_8M | Tox21 | AUC↑ | 0.7944 | 0.7961 | -0.0018 | 0.5219 | 0.3733 | 0.4544 | -0.0045 | 0.0042 | 0.98 | 0.98 | 4099 |
| u2s_dense_from8M vs skip_dense_8M | HIV | AUC↑ | 0.7741 | 0.7724 | 0.0017 | 0.7322 | 0.9262 | 0.9427 | -0.0093 | 0.01 | 0.918 | 0.98 | 20670 |
| u2s_dense_from8M vs skip_dense_8M | HIV | NEF1%↑ | 0.6434 | 0.6578 | -0.0145 | 0.4263 | 0.9262 | 0.9427 | -0.0664 | 0.0156 | 0.302 | 0.4228 | 20670 |
| u2s_dense_from8M vs unsup_8M | ESOL | RMSE↓ | 0.9239 | 1.0131 | -0.0892 | 0.1861 | 0.0 | 0.0 | 0.0106 | 0.1492 | 0.024 | 0.0607 | 585 |
| u2s_dense_from8M vs unsup_8M | QM7 | RMSE↓ | 194.3587 | 199.5783 | -5.2196 | 0.1094 | 0.0043 | 0.0101 | 1.4468 | 11.9081 | 0.0 | 0.0 | 3771 |
| u2s_dense_from8M vs unsup_8M | BBBP | AUC↑ | 0.944 | 0.9536 | -0.0095 | 0.1449 | 0.0051 | 0.0109 | -0.0194 | -0.0015 | 0.026 | 0.0607 | 1123 |
| u2s_dense_from8M vs unsup_8M | BACE | AUC↑ | 0.8359 | 0.8694 | -0.0335 | 0.0327 | 0.0 | 0.0 | -0.0455 | -0.0149 | 0.0 | 0.0 | 675 |
| u2s_dense_from8M vs unsup_8M | Tox21 | AUC↑ | 0.7944 | 0.7987 | -0.0044 | 0.3317 | 0.5138 | 0.5754 | -0.0095 | 0.0046 | 0.368 | 0.4684 | 4099 |
| u2s_dense_from8M vs unsup_8M | HIV | AUC↑ | 0.7741 | 0.7789 | -0.0049 | 0.6784 | 0.185 | 0.2467 | -0.024 | 0.0081 | 0.348 | 0.464 | 20670 |
| u2s_dense_from8M vs unsup_8M | HIV | NEF1%↑ | 0.6434 | 0.6819 | -0.0386 | 0.5051 | 0.185 | 0.2467 | -0.1186 | -0.0104 | 0.018 | 0.0504 | 20670 |
