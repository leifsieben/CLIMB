# Pretraining-seed variance, principal 8M arms (3 seeds x 5-fold CV)

`seed_std` = std across the 3 pretraining seeds' fold-means; `fold_std_s0` = the within-seed across-fold std currently used for the headline error bars.

| arm | task | metric | seed_mean | seed_std | fold_std_s0 | s0 | s1 | s2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unsup_only | ESOL | rmse | 0.5034 | 0.0157 | 0.0509 | 0.4903 | 0.5255 | 0.4946 |
| unsup_only | QM7 | rmse | 0.8714 | 0.003 | 0.0363 | 0.8743 | 0.8727 | 0.8672 |
| unsup_only | BBBP | roc_auc | 0.9503 | 0.0034 | 0.0074 | 0.9536 | 0.9456 | 0.9519 |
| unsup_only | BACE | roc_auc | 0.8581 | 0.0115 | 0.0221 | 0.8694 | 0.8625 | 0.8424 |
| unsup_only | Tox21 | roc_auc | 0.7712 | 0.0034 | 0.0328 | 0.7752 | 0.7715 | 0.767 |
| unsup_only | HIV | roc_auc | 0.7689 | 0.0071 | 0.0341 | 0.7789 | 0.7646 | 0.7631 |
| unsup_only | HIV | nef1 | 0.6618 | 0.0142 | 0.0919 | 0.6819 | 0.6506 | 0.653 |
| sup_only:dense | ESOL | rmse | 0.4293 | 0.0083 | 0.0628 | 0.4391 | 0.4299 | 0.4188 |
| sup_only:dense | QM7 | rmse | 0.8507 | 0.0008 | 0.0219 | 0.8508 | 0.8496 | 0.8516 |
| sup_only:dense | BBBP | roc_auc | 0.943 | 0.002 | 0.0052 | 0.9404 | 0.9452 | 0.9432 |
| sup_only:dense | BACE | roc_auc | 0.849 | 0.0036 | 0.024 | 0.8485 | 0.8537 | 0.8449 |
| sup_only:dense | Tox21 | roc_auc | 0.7712 | 0.0005 | 0.0324 | 0.7705 | 0.7715 | 0.7717 |
| sup_only:dense | HIV | roc_auc | 0.7688 | 0.0031 | 0.0325 | 0.7724 | 0.7649 | 0.769 |
| sup_only:dense | HIV | nef1 | 0.6474 | 0.0089 | 0.0941 | 0.6578 | 0.6482 | 0.6361 |
| sup_only:mixed | ESOL | rmse | 0.4254 | 0.026 | 0.0357 | 0.4622 | 0.4076 | 0.4063 |
| sup_only:mixed | QM7 | rmse | 0.8614 | 0.0012 | 0.0289 | 0.8624 | 0.8619 | 0.8597 |
| sup_only:mixed | BBBP | roc_auc | 0.9416 | 0.0013 | 0.0077 | 0.9401 | 0.9432 | 0.9415 |
| sup_only:mixed | BACE | roc_auc | 0.8275 | 0.0062 | 0.0382 | 0.8201 | 0.8353 | 0.8271 |
| sup_only:mixed | Tox21 | roc_auc | 0.7941 | 0.0017 | 0.0238 | 0.7962 | 0.7941 | 0.7921 |
| sup_only:mixed | HIV | roc_auc | 0.7797 | 0.0064 | 0.0308 | 0.7745 | 0.7888 | 0.7758 |
| sup_only:mixed | HIV | nef1 | 0.6627 | 0.0086 | 0.0605 | 0.6747 | 0.6554 | 0.6578 |
| unsup->sup:dense | ESOL | rmse | 0.4636 | 0.0174 | 0.0547 | 0.4422 | 0.4849 | 0.4636 |
| unsup->sup:dense | QM7 | rmse | 0.856 | 0.0054 | 0.0231 | 0.8508 | 0.8635 | 0.8538 |
| unsup->sup:dense | BBBP | roc_auc | 0.9427 | 0.002 | 0.0066 | 0.944 | 0.9443 | 0.9398 |
| unsup->sup:dense | BACE | roc_auc | 0.8302 | 0.0092 | 0.0242 | 0.8359 | 0.8171 | 0.8375 |
| unsup->sup:dense | Tox21 | roc_auc | 0.7681 | 0.0011 | 0.0399 | 0.7666 | 0.7683 | 0.7694 |
| unsup->sup:dense | HIV | roc_auc | 0.7705 | 0.006 | 0.0408 | 0.7741 | 0.7755 | 0.7621 |
| unsup->sup:dense | HIV | nef1 | 0.6289 | 0.0102 | 0.1007 | 0.6434 | 0.6217 | 0.6217 |
