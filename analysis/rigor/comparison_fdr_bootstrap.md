# Principal 8M comparisons — FDR-corrected + scaffold cluster-bootstrap

Cluster bootstrap: 1000 resamples of whole Bemis-Murcko scaffolds. `point_p` = molecule-level DeLong/Wilcoxon (anti-conservative, indicative only); `point_q`/`boot_q` = Benjamini-Hochberg FDR across all rows; `[ci_lo, ci_hi]` = 95% cluster-bootstrap CI on the metric difference (a - b, oriented so >0 favours a). A CI spanning 0 = no difference detectable.

| pair | task | metric | a_mean | b_mean | delta | fold_t_p | point_p | point_q | ci_lo | ci_hi | boot_p | boot_q | n_scaffolds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unsup_8M vs fp_desc_anchor | ESOL | RMSE↓ | 0.4903 | 0.356 | 0.1342 | 0.0113 | 0.0 | 0.0 | -0.1836 | -0.0828 | 0.0 | 0.0 | 585 |
| unsup_8M vs fp_desc_anchor | QM7 | RMSE↓ | 0.8743 | 0.8213 | 0.0531 | 0.028 | 0.0 | 0.0 | -0.0884 | -0.0343 | 0.0 | 0.0 | 3771 |
| unsup_8M vs fp_desc_anchor | BBBP | AUC↑ | 0.9536 | 0.9032 | 0.0503 | 0.03 | 0.0 | 0.0 | 0.029 | 0.0723 | 0.0 | 0.0 | 1123 |
| unsup_8M vs fp_desc_anchor | BACE | AUC↑ | 0.8694 | 0.8668 | 0.0026 | 0.5073 | 0.9427 | 0.9427 | -0.0133 | 0.0138 | 0.956 | 0.9914 | 675 |
| unsup_8M vs fp_desc_anchor | Tox21 | AUC↑ | 0.7752 | 0.7944 | -0.0192 | 0.0014 | 0.1804 | 0.259 | -0.0228 | -0.0103 | 0.0 | 0.0 | 4107 |
| unsup_8M vs fp_desc_anchor | HIV | AUC↑ | 0.7789 | 0.8159 | -0.037 | 0.0141 | 0.0 | 0.0 | -0.0532 | -0.0216 | 0.0 | 0.0 | 20670 |
| unsup_8M vs fp_desc_anchor | HIV | NEF1%↑ | 0.6819 | 0.7133 | -0.0313 | 0.478 | 0.0 | 0.0 | -0.0711 | 0.0123 | 0.206 | 0.3036 | 20670 |
| unsup_8M vs skip_dense_8M | ESOL | RMSE↓ | 0.4903 | 0.4391 | 0.0511 | 0.1109 | 0.0 | 0.0 | -0.0906 | -0.0112 | 0.014 | 0.0392 | 585 |
| unsup_8M vs skip_dense_8M | QM7 | RMSE↓ | 0.8743 | 0.8508 | 0.0236 | 0.1899 | 0.0005 | 0.0015 | -0.0552 | -0.0058 | 0.01 | 0.0311 | 3771 |
| unsup_8M vs skip_dense_8M | BBBP | AUC↑ | 0.9536 | 0.9404 | 0.0131 | 0.0321 | 0.0002 | 0.0005 | 0.0044 | 0.0242 | 0.006 | 0.021 | 1123 |
| unsup_8M vs skip_dense_8M | BACE | AUC↑ | 0.8694 | 0.8485 | 0.021 | 0.0514 | 0.0042 | 0.0098 | 0.0003 | 0.0361 | 0.048 | 0.0896 | 675 |
| unsup_8M vs skip_dense_8M | Tox21 | AUC↑ | 0.7752 | 0.7705 | 0.0047 | 0.139 | 0.5537 | 0.6202 | 0.0001 | 0.012 | 0.046 | 0.0896 | 4107 |
| unsup_8M vs skip_dense_8M | HIV | AUC↑ | 0.7789 | 0.7724 | 0.0065 | 0.5487 | 0.1637 | 0.259 | -0.0049 | 0.0232 | 0.238 | 0.3332 | 20670 |
| unsup_8M vs skip_dense_8M | HIV | NEF1%↑ | 0.6819 | 0.6578 | 0.0241 | 0.635 | 0.1637 | 0.259 | -0.0216 | 0.105 | 0.196 | 0.3036 | 20670 |
| u2s_dense_from8M vs skip_dense_8M | ESOL | RMSE↓ | 0.4422 | 0.4391 | 0.003 | 0.7812 | 0.2188 | 0.2838 | -0.0367 | 0.0188 | 0.902 | 0.9886 | 585 |
| u2s_dense_from8M vs skip_dense_8M | QM7 | RMSE↓ | 0.8508 | 0.8508 | -0.0 | 0.9954 | 0.4691 | 0.5508 | -0.0056 | 0.0053 | 0.996 | 0.996 | 3771 |
| u2s_dense_from8M vs skip_dense_8M | BBBP | AUC↑ | 0.944 | 0.9404 | 0.0036 | 0.4647 | 0.159 | 0.259 | -0.0024 | 0.0101 | 0.252 | 0.336 | 1123 |
| u2s_dense_from8M vs skip_dense_8M | BACE | AUC↑ | 0.8359 | 0.8485 | -0.0126 | 0.1462 | 0.0366 | 0.0733 | -0.0274 | 0.0026 | 0.128 | 0.2108 | 675 |
| u2s_dense_from8M vs skip_dense_8M | Tox21 | AUC↑ | 0.7666 | 0.7705 | -0.0039 | 0.4161 | 0.4721 | 0.5508 | -0.0068 | 0.0037 | 0.576 | 0.672 | 4107 |
| u2s_dense_from8M vs skip_dense_8M | HIV | AUC↑ | 0.7741 | 0.7724 | 0.0017 | 0.7322 | 0.9262 | 0.9427 | -0.0093 | 0.01 | 0.918 | 0.9886 | 20670 |
| u2s_dense_from8M vs skip_dense_8M | HIV | NEF1%↑ | 0.6434 | 0.6578 | -0.0145 | 0.4263 | 0.9262 | 0.9427 | -0.0664 | 0.0156 | 0.302 | 0.3844 | 20670 |
| u2s_dense_from8M vs unsup_8M | ESOL | RMSE↓ | 0.4422 | 0.4903 | -0.0481 | 0.1839 | 0.0 | 0.0 | 0.004 | 0.0777 | 0.032 | 0.0689 | 585 |
| u2s_dense_from8M vs unsup_8M | QM7 | RMSE↓ | 0.8508 | 0.8743 | -0.0236 | 0.1126 | 0.0014 | 0.0035 | 0.0066 | 0.0541 | 0.0 | 0.0 | 3771 |
| u2s_dense_from8M vs unsup_8M | BBBP | AUC↑ | 0.944 | 0.9536 | -0.0095 | 0.1449 | 0.0051 | 0.0109 | -0.0194 | -0.0015 | 0.026 | 0.0607 | 1123 |
| u2s_dense_from8M vs unsup_8M | BACE | AUC↑ | 0.8359 | 0.8694 | -0.0335 | 0.0327 | 0.0 | 0.0 | -0.0455 | -0.0149 | 0.0 | 0.0 | 675 |
| u2s_dense_from8M vs unsup_8M | Tox21 | AUC↑ | 0.7666 | 0.7752 | -0.0086 | 0.0924 | 0.223 | 0.2838 | -0.0149 | 0.0005 | 0.066 | 0.1155 | 4107 |
| u2s_dense_from8M vs unsup_8M | HIV | AUC↑ | 0.7741 | 0.7789 | -0.0049 | 0.6784 | 0.185 | 0.259 | -0.024 | 0.0081 | 0.348 | 0.4237 | 20670 |
| u2s_dense_from8M vs unsup_8M | HIV | NEF1%↑ | 0.6434 | 0.6819 | -0.0386 | 0.5051 | 0.185 | 0.259 | -0.1186 | -0.0104 | 0.018 | 0.0458 | 20670 |
