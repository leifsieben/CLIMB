# Superseded — do not read these into fig_F

These are the 2026-08-21 Mordred tables computed on the LAPTOP. They are superseded by
`analysis/rigor/figF/concat_{mordred,rdkit_sameenv}_*.csv`, which came from the AWS box.

They are moved rather than deleted because the numbers are real and the BBBP result was first
seen here. They are moved rather than left in place because **mixing them into fig_F would
change the molecule set mid-figure**: Tox21 parses as 7,823 molecules under this laptop's
RDKit/DeepChem and 7,831 on the box that produced the figF tables. Every fig_F cell must come
from one environment, which is the entire reason the RDKit arm was regenerated instead of reused.

A path that still resolves is a path something will eventually read. This directory exists so
that cannot happen by accident.
