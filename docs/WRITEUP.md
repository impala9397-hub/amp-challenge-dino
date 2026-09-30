# Dino — AMP Challenge 2027 submission

**Team Dino** · [github.com/impala9397-hub/amp-challenge-dino](https://github.com/impala9397-hub/amp-challenge-dino)

## Abstract

A 4.8M-parameter character-level autoregressive transformer, trained only on the
training partition of the organizers' corpus, samples 50,000 peptides at
temperature 1.0. **That sample is the submitted library — it is not selected.**
An unbiased draw is already the best available estimate of the generator's
distribution, and we measured that choosing a subset of it only makes the library
worse. Selection applies to the top-100 alone, and it does not rank: any rule
rewarding closeness to the reference distribution collapses the picks into that
distribution's centre, which inflates FBD even when the mean matches exactly.
Instead the reference set is split into 100 modes, each mode's exemplar is pushed
outward from the reference centroid to undo the narrowing that taking cluster
centres causes, and each exemplar claims the closest candidate clearing the
challenge's exact-match and Levenshtein-ratio gates. Generation is pinned to a
single CPU thread with a fixed seed, so `uv run generate` reproduces both files
byte for byte with no network access. The outputs are computational proposals; no
antibacterial or safety measurement has been performed.

## What we would want a reader to take away

**Ranking is the wrong primitive for picking a subset that will be scored against
a distribution.** This is not a design preference, it is a measurement. Selecting
100 sequences, evaluated with ESM2-650M against the organizers' reference set:

| Selection rule | FBD ↓ | MMD ↓ | Precision | Recall | Coverage |
|---|---|---|---|---|---|
| Reward closeness to the reference centroid | 5.11 | 11.58 | 1.00 | **0.11** | 0.78 |
| Force high net charge (hard filter) | 5.13 | 13.07 | 1.00 | 0.14 | 0.74 |
| One pick per reference mode | 1.90 | 2.00 | 0.99 | 0.72 | 1.00 |
| Match each mode's exemplar 1:1 | 1.78 | 1.23 | 0.94 | 0.81 | 1.00 |
| **The same, with the exemplar spread applied** | **1.71** | **0.95** | 0.93 | **0.86** | 0.97 |
| *For scale:* 100 real AMPs at random | 1.56–1.71 | 0.60–1.45 | 0.91–0.95 | 0.90–0.94 | 0.89–1.00 |

`Precision 1.00` alongside `Recall 0.11` is the whole diagnosis. Every pick lay
inside the reference distribution; together they covered a sliver of it. FBD
compares means *and* covariances, so a collapsed spread keeps FBD high however
well the mean lines up — we confirmed this by matching the reference charge
distribution bin for bin and watching FBD stay at 5.11.

**A cheap surrogate for the scoring metric is not the scoring metric.** Our
matching space gives the twenty residue fractions 20 of its 24 columns, which
dilutes the four aggregate properties. Down-weighting composition raises the
Spearman correlation between our distances and ESM2's from 0.300 to 0.357, a 19%
gain on the surrogate — and it makes the selection's FBD 16% *worse*. The
selector's job is not to reproduce ESM2's distances but to find the candidate
most like a target, and suppressing composition is exactly what makes candidates
indistinguishable. We now check space changes against FBD and MMD directly.

We also learned to measure small-sample floors before believing a number. Twelve
random draws of 100 real AMPs give FBD 1.48–1.84. A 100-sequence selection is
near that floor, and our first comparison — a 100-sequence FBD of 2.61 against a
50,000-sequence FBD of 0.21 — was meaningless.

## Selection

Both the library decision and the top-100 selection reason in one feature space:
net charge at pH 7, length, amphipathicity, mean hydrophobicity, and the twenty
residue fractions. All of it is pure NumPy on the CPU, because the entry point
must run offline and deterministically — a protein language model can evaluate the
result but cannot produce it.

**Library: no selection at all.** We built a selector that makes each of the
39,448 reference sequences claim its nearest candidate, ran it at library scale,
and rejected it. Every headline metric moved the wrong way: FBD 0.2104 → 0.3573,
MMD 0.3072 → 0.8285, Recall 0.9160 → 0.8735, Diversity 0.8570 → 0.8404. Charge and
coverage did improve, and it was not enough; amphipathicity overshot the reference
(0.408 against 0.368), which is the signature of matching a property distribution
more tightly than a real sample of it would and paying for that in embedding
spread. The selector remains in the code because it is the right tool the moment
the candidate pool stops being an unbiased sample, and because deleting it would
delete the evidence for why the library is submitted unselected.

**Top-100.** The reference set is split into 100 clusters. Each cluster's medoid —
a real reference sequence, not an average — is scaled outward from the reference
centroid by 1.16 and then takes the closest candidate clearing both gates. The
list is ordered by distance to the matched exemplar, closest first, because the
organizers' main text and FAQ disagree on whether the experimental draw comes from
the top 100 or the top 50.

The 1.16 is a correction, not a tuned knob. A medoid sits at its cluster's centre,
so 100 medoids are narrower than the reference: standard deviations of 0.86x for
net charge, 0.84x for length, 0.82x for amphipathicity, 0.90x for hydrophobicity.
The *means* survive — k-means subdivides dense regions, so cluster count carries
the density — but the tails thin, the charge distribution's `<=0` bin falling from
18.9% to 15.0% and its `8+` bin from 4.7% to 2.0%. Since features are standardised
by the reference's own mean, the reference centroid is the origin, and scaling an
exemplar pushes it outward along its own direction: the target set regains its
spread while each target still points into a region dense enough to hold
candidates. `1 / 0.86 = 1.16` is the predicted correction, and across five
independent k-means seeds it improves FBD every time, by 3.9% to 9.2%.

It costs 2.6% of ClippedCoverage, which was saturated at 1.000 and could only
fall. We took the trade because three of the four distributional metrics improve
robustly and the resulting 0.974 still sits above the 0.95 that 100 real
antimicrobial peptides average. The axis weights are withheld, so this is a
judgement rather than a calculation, and the numbers are here so a reader can
disagree with it. Two more obvious repairs were tried first and both lost:
replacing a medoid with a random reference sequence (FBD 1.78 → 2.12) and with a
random member of its own cluster (1.78 → 2.09). A random reference can be an
outlier with no candidate near it, so those bought spread with match quality.

## Screens and gates

Mandatory requirements (canonical residues, length 8–50, no exact reference match,
top-100 within Levenshtein ratio 0.80) are enforced in code and tested. We exclude
against MarLys v3 rather than the reference set alone, because a measured run
leaked two sequences otherwise, and a leak fails the organizers' validator. The
top-100 gate is set at 0.78 rather than 0.80, leaving 0.02 of margin against
implementation differences in the ratio; the submitted list's maximum is 0.7778.

Our synthesis screens — no cysteine, no three identical residues in a row, fewer
than three K/R per five-residue window — apply **only** to the top-100. The
library is submitted unscreened, and that is measured rather than assumed:

| Set | n | Pass, ≤1 cysteine | Pass, no cysteine |
|---|---:|---:|---:|
| Our library | 50,000 | 50.5% | 40.2% |
| Real AMPs (the reference set) | 39,448 | 49.8% | 40.4% |
| UniProt peptides (negatives) | 2,600 | 46.8% | 41.8% |

The screens do not separate antimicrobial peptides from non-antimicrobial ones —
the three sets land within about 4 pp of each other under either cysteine rule,
and which set ranks highest flips between them. So a higher pass rate is not
evidence of a better library, and maximising it would move the library off the
reference distribution for nothing. Our library tracks the reference set to within
1 pp under both rules, which is what we want. The screens earn their place on the
top-100 only, where a peptide nobody can synthesise is not a useful
recommendation.

Three departures from the HydrAMP filters used by the organizers' starter kit,
each measured: positive residues are K and R only, matching HydrAMP's own source
(histidine's pKa is near 6, so it is largely uncharged at pH 7); the "three
identical hydrophobic residues in a row" rule is dropped as a strict logical
subset of the general run rule, with zero unique catches across four datasets; and
cysteine is counted rather than detected, tolerating one for library analysis and
none for top-100 selection, where 20,000 candidates remain and a single avoidable
disulfide risk is not worth taking.

## Activity prediction

We read the organizers' own published BATTLE-AMP benchmark across its nine
regression tasks rather than trusting model reputations:

| Model | Mean Spearman | Tasks with R²(log2) > 0 |
|---|---|---|
| MBC-Attention | 0.355 | 3 / 9 |
| AMPredictor | 0.337 | 3 / 9 |
| Deep-AMP (best variant) | 0.297 | 3 / 9 |
| APEX (all six variants) | 0.066 – 0.195 | **0 / 9** |

An earlier version of this pipeline ranked candidates by APEX. No APEX variant
reaches a positive R² on any of the nine tasks, so that ranking was noise. But the
decisive number is the best one: Spearman 0.355 cannot order 50,000 candidates.

`uv run generate` therefore contains **no activity predictor**. Selection is
distribution matching only. As a check that this costs nothing, we ran the
organizers' own `battleamp-snakemake` wrapper on the selected top-100 — all three
models, 6 min 17 s on one A40. Median predicted MIC: 7.8 µM (Deep-AMP CNN,
Gram-positive), 17.3 µM (AMPredictor), 22.6 µM (MBC-Attention). Matching the
distribution produced candidates these predictors like, without optimising for
them. We report the medians as a sanity check and not as evidence of activity;
Deep-AMP's 90th percentile reaches 14,713 µM on real AMPs as well as on ours,
so the absolute values carry no weight.

## Disclosure

- **Training data:** the training partition of the organizers' corpus only —
  37,940 train / 1,508 validation out of 39,448, split by sha256 bucket with any
  validation candidate more than 0.8 similar to a training sequence moved back
  into training. No private or unpublished dataset at any stage. AMPSphere and
  other expansion corpora were evaluated in development and **not** used for the
  released checkpoint.
- **External data:** MarLys v3 (CC0) as an exclusion list only, never for
  training.
- **Checkpoint choice:** epoch 14, chosen on the scoring axes rather than on
  validation loss, which selects epoch 5. We checked whether that disagreement
  came from a skewed validation set — the move-back step preferentially removes
  high-charge sequences, because those have the largest analogue families — and
  rebuilt a charge-stratified, leak-free validation set. It still selects epoch 5.
  Cross-entropy and FBD measure different things and both measure correctly.
- **Manual intervention:** none in the generation or selection path. Every
  threshold in this submission is a default in the code, and the values were set
  by the measurements reported above.
- **Licensing:** MIT, with `python-Levenshtein` (GPL-2.0-or-later) and `modlamp`
  (which depends on GPL-2.0 `mysql-connector-python`) deliberately replaced by our
  own implementations, tested for exact agreement against both. `pyproject.toml`
  sets `default-groups = []` so that a bare `uv sync` installs neither.

## Not established

No experimental antibacterial or haemolysis result exists for any sequence here.
Generalization after retraining with a different training seed is unmeasured.
Activity-surrogate numbers are their authors' published performance, not
validation of our candidates. The gap between the library's FBD of 0.2104 and the
0.0613 that a fresh draw of real AMPs achieves belongs to the generator; five
attempts to close it from the selection side were measured and rejected.
