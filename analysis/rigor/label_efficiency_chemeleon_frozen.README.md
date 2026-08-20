# CheMeleon frozen under the label-efficiency protocol

BACE and Tox21 at fraction 1.0, 3 head seeds, MLP on precomputed CheMeleon vectors with zscore
standardisation — the same probe and the same protocol as the CLIMB lines SI fig a draws beside it.

## QM7 IS QUARANTINED, NOT MISSING

The QM7 cell returned test RMSE **1818.9 ± 45.2** against a train RMSE of 206.9. It is in
`*.QM7_FAILED.csv` and must not be plotted.

It is not a result:

* QM7 targets span −2192 to −405 with SD **228.7**, so an RMSE of 1819 is ~8x worse than predicting
  the training mean. A representation cannot be *that* wrong; a broken cell can.
* Every other arm in this same wave lands at 203–219 (random 212.4, unsup 219.2, sup 202.9,
  unsup2sup 207.8, e2e 206.8).
* Excluded as causes: the features are finite with a sane range (train [0, 4.71], test [0, 2.3]);
  the head is `mlp`, identical to `HEAD` used by the encoder arms; standardisation is the same
  zscore path.

The pattern — train RMSE ≈ target SD, test RMSE ≈ |target mean| — looks like predictions collapsing
toward zero in native units, i.e. a target-scaling failure specific to this arm, but that is a
hypothesis and has not been confirmed.

BACE and Tox21 are unaffected: both sit in the expected band for this wave and their train/test gap
is normal.

## The e2e half does not exist and cannot be produced by this driver

`scripts/label_eff_fractions.py` has three paths: a ModernBertModel encoder branch, a CLASSICAL
featurizer branch, and the PRECOMPUTED branch added for this run. CheMeleon e2e is a chemprop
training subprocess per cell, which none of them can host. It needs either a dedicated runner or a
caption caveat.
