# Dino — AMP Challenge 2027 submission

**Team Dino** · [github.com/impala9397-hub/amp-challenge-dino](https://github.com/impala9397-hub/amp-challenge-dino)

## Abstract

A 4.8M-parameter character-level autoregressive transformer, trained only on the
training partition of the organizers' corpus, samples 250,000 candidate peptides.
A distribution-matching selector reduces these to the submitted 50,000-member
library and then to an ordered top-100, inside the challenge's exact-match and
Levenshtein-ratio gates. Selection deliberately does not rank candidates by a
score: we measured that any rule rewarding closeness to the reference
distribution collapses the selection into that distribution's centre, which
inflates FBD even when the mean matches exactly. Instead, reference sequences
spanning the distribution each claim their closest candidate. Generation is pinned
to a single CPU thread with a fixed seed, so `uv run generate` reproduces both
files byte for byte with no network access. The outputs are computational
proposals; no antibacterial or safety measurement has been performed.

## What we would want a reader to take away

**Ranking is the wrong primitive for picking a subset that will be scored against
a distribution.** This is not a design preference, it is a measurement. Selecting
100 sequences, evaluated with ESM2-650M against the organizers' reference set:

| Selection rule | FBD ↓ | MMD ↓ | Precision | Recall | Coverage |
|---|---|---|---|---|---|
| Reward closeness to the reference centroid | 5.11 | 11.58 | 1.00 | **0.11** | 0.78 |
| Force high net charge (hard filter) | 5.13 | 13.07 | 1.00 | 0.14 | 0.74 |
| One pick per reference mode | 1.90 | 2.00 | 0.99 | 0.72 | 1.00 |
| **Match each mode's exemplar 1:1** | **1.79** | **1.08** | 0.96 | **0.75** | 1.00 |
| *For scale:* 100 real AMPs at random | 1.71 | 1.15 | 0.95 | 0.91 | 1.00 |

`Precision 1.00` alongside `Recall 0.11` is the whole diagnosis. Every pick lay
inside the reference distribution; together they covered a sliver of it. FBD
compares means *and* covariances, so a collapsed spread keeps FBD high however
well the mean lines up — we confirmed this by matching the reference charge
distribution bin for bin and watching FBD stay at 5.11.

We also learned to measure small-sample floors before believing a number. Twelve
random draws of 100 real AMPs give FBD 1.48–1.84. A 100-sequence selection cannot
beat that, so 1.79 is at the floor, and our first comparison — a 100-sequence FBD
of 2.61 against a 50,000-sequence FBD of 0.21 — was meaningless.

## Selection

Both stages work in one feature space: net charge at pH 7, length,
amphipathicity, mean hydrophobicity, and the twenty residue fractions. All of it
is pure NumPy on the CPU, because the entry point must run offline and
deterministically — a protein language model can evaluate the result but cannot
produce it.

**Library.** Each of the 39,448 reference sequences claims its nearest unclaimed
candidate from the 250,000 sampled; the remaining slots go to unclaimed candidates
closest to any reference.

**Top-100.** The reference set is split into 100 clusters; each cluster's medoid —
a real reference sequence, not an average — takes the closest candidate clearing
both gates. The list is ordered by distance to the matched exemplar, closest
first, because the organizers' main text and FAQ disagree on whether the
experimental draw comes from the top 100 or the top 50.

## Screens and gates

Mandatory requirements (canonical residues, length 8–50, no exact reference match,
top-100 within Levenshtein ratio 0.80) are enforced in code and tested. We exclude
against MarLys v3 rather than the reference set alone, because a measured run
leaked two sequences otherwise, and a leak fails the organizers' validator.

Our synthesis screens — no cysteine, no three identical residues in a row, fewer
than three K/R per five-residue window — apply **only** to the top-100. The
library is submitted unscreened, and that is measured rather than assumed:

| Set | Screen pass rate |
|---|---|
| Our library | 37.0% |
| Real AMPs | 36.9% |
| UniProt peptides (negatives) | 41.3% |

The negative class passes at a higher rate than real AMPs, so maximising the pass
rate would push the library away from the reference distribution.

Three departures from the HydrAMP filters used by the organizers' starter kit,
each measured: positive residues are K and R only (histidine's pKa is near 6, and
counting it made the screens 3.2 pp stricter for no stated reason); the
"three identical hydrophobic residues" rule is dropped as a strict subset of the
general run rule, with zero unique catches across four datasets; and cysteine is
counted rather than detected, tolerating one for library analysis and none for
top-100 selection.

## Activity prediction

We read the organizers' own published BATTLE-AMP benchmark across its nine
regression tasks rather than trusting model reputations:

| Model | Mean Spearman | Tasks with R²(log2) > 0 |
|---|---|---|
| MBC-Attention | 0.355 | 3 / 9 |
| AMPredictor | 0.337 | 3 / 9 |
| Deep-AMP (best variant) | 0.297 | 3 / 9 |
| APEX (all five variants) | 0.066 – 0.195 | **0 / 9** |

An earlier version of this pipeline ranked candidates by APEX. No APEX variant
reaches a positive R² on any of the nine tasks, so that ranking was noise. And the
best available predictor reaches Spearman 0.355 — not a signal that can order
50,000 candidates.

`uv run generate` therefore contains **no activity predictor**. Selection is
distribution matching only. As a check that this costs nothing, we scored the
selected top-100 with Deep-AMP: median predicted MIC 43.9 µM, against 120.9 µM for
the previous activity-ranked selection and 39.8 µM for 100 real AMPs. Matching the
distribution moved predicted potency toward real AMPs without optimising for it.

## Disclosure

- **Training data:** the training partition of the organizers' corpus only —
  37,935 train / 1,513 validation out of 39,448. No private or unpublished
  dataset at any stage. AMPSphere and other expansion corpora were evaluated in
  development and **not** used for the released checkpoint.
- **External data:** MarLys v3 (CC0) as an exclusion list only, never for
  training.
- **Manual intervention:** none in the generation or selection path. Every
  threshold in this submission is a default in the code, and the values were set
  by the measurements reported above.
- **Licensing:** MIT, with `python-Levenshtein` (GPL-2.0-or-later) and `modlamp`
  (which depends on GPL-2.0 `mysql-connector-python`) deliberately replaced by our
  own implementations, tested for exact agreement against both.

## Not established

No experimental antibacterial or haemolysis result exists for any sequence here.
Generalization after retraining with a different training seed is unmeasured.
Activity-surrogate numbers are their authors' published performance, not
validation of our candidates.
