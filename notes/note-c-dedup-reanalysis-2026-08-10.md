# Note C — Fig I1 / 5d–e memorization reanalysis: data + plot changes for you

**From:** AWS/data session · **Date:** 2026-08-10 · **Re:** reviewer/friend "Note C" (possible
circularity in Fig 5d–e: exact-match molecules sit in the top-Tanimoto bin by construction, so the
top-quartile lift mixes interpolation with outright memorization).

I recomputed corpus overlap **against the full 12 M `pubchem_filtered` corpus** (your current
`corpus_similarity.csv` uses only a 500 k single-shard subsample, so its max-Tanimoto is a
*lower bound* — the caption already admits this). My lane = the data below; your lane =
`notebook_cells/22.py`. Everything is repo-relative under `analysis/dedup_i1/`.

## Headline finding — TWO numbers, they answer different questions (read both)

There are two very different overlap measures, and the honest story needs both:

| measure | what it means | ESOL | QM7 |
|---|---|---|---|
| **literal exact** (isomeric-canonical key, standalone molecule) | model saw this exact token sequence | **1.3%** | **1.9%** |
| **fingerprint-identical** (ECFP4 Tanimoto = 1.0 to some corpus molecule) | model saw this *structure* (up to stereo / FP folding) | **40.6%** | **15.7%** |

- The **literal** overlap is tiny — the model rarely saw the exact token sequence.
- But **Fig I1 bins on ECFP4 Tanimoto**, and **40.6% of ESOL is at Tanimoto = 1.0** (median max-Tani
  0.944; 48% ≥ 0.95). QM7 is far more novel (15.7% at 1.0, median 0.625). So the top-Tanimoto bin is
  **saturated with corpus-identical structures by construction for ESOL** — the reviewer's circularity
  concern is **real and material for ESOL**, much less so for QM7.
- The gap between 1.3% and 40.6% is stereoisomers + ECFP4 folding/collisions + close analogs: same
  2D fingerprint, different (or absent) exact token sequence.

**Correction to my first ping:** I initially said "excluding exact matches barely moves I1" based on
the literal 1.3%. That's true only for the *literal* exclusion. The exclusion that actually matters
for this panel is **fingerprint-identical (Tani = 1.0)**, and dropping those **will** materially change
ESOL (it removes ~40% of the points, concentrated in the top bin). QM7 is barely affected. So: run the
dedup, and let the figure tell us whether the ESOL lift **survives** removing the Tani=1.0 group. If it
survives → strong interpolation claim. If it collapses → the ESOL "similar molecules benefit" signal was
largely corpus-identical structures, i.e. memorization — a sharper, more honest result that feeds §4.

⚠️ **Do NOT report the salt-stripped number as memorization.** A largest-fragment (salt-stripped)
key inflates overlap to ESOL 73% / Tox21 62% / BBBP 40% — but I verified this is an **artifact**:
it's ubiquitous small molecules (toluene, pyridine, ethylbenzene…) matching as the largest fragment
of *unrelated* organometallic complexes / reaction mixtures in PubChem (e.g. `toluene` ← `CC1=CC=CC=C1.[K+]`,
`pyridine` ← `C1=CC=NC=C1.[Ir]`). Those are not memorized molecules. Report salt-stripped only as a
clearly-labelled loose upper bound, if at all. The fingerprint near-dup band (below) is the right
tool for "functionally memorized" salt/tautomer forms.

## Data files (my deliverables)

1. **`analysis/dedup_i1/exact_match_per_molecule.csv`** — ALL eval datasets. Columns:
   `raw_smiles, dataset, key_salt, exact_nosalt, exact_salt`.
   Join to your panels on `raw_smiles`. Use **`exact_nosalt`** (literal standalone-molecule match;
   this is the defensible "memorized" flag). `exact_salt` is the inflated upper bound — don't headline it.
   `analysis/dedup_i1/exact_match_summary.json` has the per-dataset counts/percentages.

2. **`analysis/dedup_i1/full_corpus_similarity_i1.csv`** — ESOL + QM7 only (the I1 tasks). Columns:
   `raw_smiles, dataset, max_tanimoto_to_corpus_full, n_corpus_identical, n_corpus_neardup_0p95`.
   - `max_tanimoto_to_corpus_full` = **true** max ECFP4 Tanimoto to the full 12 M corpus (NOT a lower
     bound). Same fingerprint spec as `compute_tanimoto_novelty.py` (Morgan r=2, 2048 bits).
   - `n_corpus_identical` = # corpus molecules at Tanimoto = 1.0 (fingerprint-identical).
   - `n_corpus_neardup_0p95` = # corpus molecules in the near-dup band [0.95, 1.0) — salt/tautomer/
     stereo forms that miss the exact key but are functionally memorized.
   > **STATUS: READY.** Written 2026-08-10 16:11. 7956 molecules scored vs full 11,996,074-molecule corpus.
   Per-task structure (from this file):
   ```
   ESOL (n=1117):  Tani=1.0 (fingerprint-identical): 453 (40.6%) | near-dup [0.95,1.0): 88 (7.9%) | ≥0.95: 48.4% | median 0.944
   QM7  (n=6839):  Tani=1.0 (fingerprint-identical): 1075 (15.7%)| near-dup [0.95,1.0):  6 (0.1%) | ≥0.95: 15.8% | median 0.625
   ```

## Suggested changes to `notebook_cells/22.py`

The plot mechanism is `binned_lift()` (L35–60), which merges model+baseline CV preds with `TANI`
on `raw_smiles` and bins by `max_tanimoto_to_corpus` quantiles. Three edits:

1. **Use the true similarity, not the lower bound.** Point `TANI` at
   `analysis/dedup_i1/full_corpus_similarity_i1.csv` and rename the joined column to
   `max_tanimoto_to_corpus` (or read `max_tanimoto_to_corpus_full`). Then **drop the "lower bound"
   disclaimer** from the caption (L114–116) — it's now the true max over the full corpus.

2. **Exclude corpus-identical structures, report them as their own point.** The exclusion that
   matters for this panel is **fingerprint-identical (Tanimoto = 1.0)**, NOT the literal key — because
   the panel bins on Tanimoto and those points are pinned to the top bin. From
   `full_corpus_similarity_i1.csv`, `is_identical = (max_tanimoto_to_corpus_full >= 0.99999)` (equiv:
   `n_corpus_identical > 0`). Then:
   - `m_ident = m[is_identical]` → its own "corpus-identical (Tani=1.0)" bar/point. This is ~40% of
     ESOL, ~16% of QM7.
   - `m_dedup = m[~is_identical]` → re-run `binned_lift` on this. **Does the ESOL lift survive?** That's
     the whole question. QM7 barely changes; ESOL is the test.
   - Also keep the stricter literal cut (`exact_nosalt==1`, from `exact_match_per_molecule.csv`) as a
     secondary line if you want — but lead with the Tani=1.0 cut.
   Report all three: full, corpus-identical-removed, and the identical group itself (as its own point).

3. **(Optional) near-dup band.** Flag molecules with `n_corpus_neardup_0p95 > 0` (or
   `max_tanimoto_to_corpus_full ≥ 0.95`) and report that group separately too — it's the "salt/tautomer
   of a corpus molecule" category the reviewer's point (3) asks about.

If, after all this, the top-quartile lift **survives** dedup → the interpolation claim gets stronger,
say so explicitly. If it **flattens** → the apparent similarity–lift relationship was memorization,
which is a *more* interesting result and feeds §4. Either way the panel is now defensible.

I have NOT touched `notebook_cells/22.py` (your lane). Ping me if you want the data reshaped.
