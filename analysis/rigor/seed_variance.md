# Pretraining-seed variance, principal 8M arms (3 seeds x 5-fold CV)

`seed_std` = std across the 3 pretraining seeds' fold-means; `fold_std_s0` = the within-seed across-fold std currently used for the headline error bars.

| arm | task | metric | seed_mean | seed_std | fold_std_s0 | s0 | s1 | s2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unsup_only | ESOL | rmse | 1.0258 | 0.0106 | 0.0799 | 1.0131 | 1.0392 | 1.0252 |
| unsup_only | QM7 | rmse | 198.8708 | 0.5539 | 7.5238 | 199.5783 | 198.8083 | 198.2257 |
| unsup_only | BBBP | roc_auc | 0.9503 | 0.0034 | 0.0074 | 0.9536 | 0.9456 | 0.9519 |
| unsup_only | BACE | roc_auc | 0.8581 | 0.0115 | 0.0221 | 0.8694 | 0.8625 | 0.8424 |
| unsup_only | Tox21 | roc_auc | 0.7961 | 0.002 | 0.039 | 0.7987 | 0.794 | 0.7957 |
| unsup_only | HIV | roc_auc | 0.7689 | 0.0071 | 0.0341 | 0.7789 | 0.7646 | 0.7631 |
| unsup_only | HIV | nef1 | 0.6618 | 0.0142 | 0.0919 | 0.6819 | 0.6506 | 0.653 |
| sup_only:dense | ESOL | rmse | 0.8914 | 0.0042 | 0.1225 | 0.8963 | 0.892 | 0.886 |
| sup_only:dense | QM7 | rmse | 195.0588 | 0.2443 | 4.4595 | 194.7582 | 195.0618 | 195.3565 |
| sup_only:dense | BBBP | roc_auc | 0.943 | 0.002 | 0.0052 | 0.9404 | 0.9452 | 0.9432 |
| sup_only:dense | BACE | roc_auc | 0.849 | 0.0036 | 0.024 | 0.8485 | 0.8537 | 0.8449 |
| sup_only:dense | Tox21 | roc_auc | 0.7966 | 0.0019 | 0.0429 | 0.7961 | 0.7946 | 0.7991 |
| sup_only:dense | HIV | roc_auc | 0.7688 | 0.0031 | 0.0325 | 0.7724 | 0.7649 | 0.769 |
| sup_only:dense | HIV | nef1 | 0.6474 | 0.0089 | 0.0941 | 0.6578 | 0.6482 | 0.6361 |
| sup_only:mixed | ESOL | rmse | 0.8825 | 0.0347 | 0.0677 | 0.9303 | 0.8488 | 0.8684 |
| sup_only:mixed | QM7 | rmse | 196.7984 | 0.2285 | 5.631 | 196.638 | 197.1215 | 196.6356 |
| sup_only:mixed | BBBP | roc_auc | 0.9416 | 0.0013 | 0.0077 | 0.9401 | 0.9432 | 0.9415 |
| sup_only:mixed | BACE | roc_auc | 0.8275 | 0.0062 | 0.0382 | 0.8201 | 0.8353 | 0.8271 |
| sup_only:mixed | Tox21 | roc_auc | 0.8188 | 0.0016 | 0.0279 | 0.8209 | 0.8172 | 0.8182 |
| sup_only:mixed | HIV | roc_auc | 0.7797 | 0.0064 | 0.0308 | 0.7745 | 0.7888 | 0.7758 |
| sup_only:mixed | HIV | nef1 | 0.6627 | 0.0086 | 0.0605 | 0.6747 | 0.6554 | 0.6578 |
| unsup->sup:dense | ESOL | rmse | 0.9513 | 0.0295 | 0.1012 | 0.9239 | 0.9923 | 0.9376 |
| unsup->sup:dense | QM7 | rmse | 195.786 | 1.0382 | 5.4653 | 194.3587 | 196.7977 | 196.2016 |
| unsup->sup:dense | BBBP | roc_auc | 0.9427 | 0.002 | 0.0066 | 0.944 | 0.9443 | 0.9398 |
| unsup->sup:dense | BACE | roc_auc | 0.8302 | 0.0092 | 0.0242 | 0.8359 | 0.8171 | 0.8375 |
| unsup->sup:dense | Tox21 | roc_auc | 0.7948 | 0.0025 | 0.0458 | 0.7944 | 0.7919 | 0.7981 |
| unsup->sup:dense | HIV | roc_auc | 0.7705 | 0.006 | 0.0408 | 0.7741 | 0.7755 | 0.7621 |
| unsup->sup:dense | HIV | nef1 | 0.6289 | 0.0102 | 0.1007 | 0.6434 | 0.6217 | 0.6217 |
