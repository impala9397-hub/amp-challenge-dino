# Method

## Summary

A character-level autoregressive transformer, trained only on the training
partition of the organizers' corpus, samples the 50,000-member library. A
distribution-matching selector then picks and orders the top-100 from it, inside
the challenge's similarity gates. The library itself is submitted unselected —
matching it to the reference distribution was tried and measurably hurt.
Generation is pinned to a single CPU thread with a fixed seed, so a repeated run
reproduces the files byte for byte.

The outputs are computational proposals. No antibacterial or safety measurement
has been performed on them.

## Generator and training disclosure

| | |
|---|---|
| Architecture | Character-level causal transformer |
| Hidden dimension | 256 |
| Layers / heads / FFN | 6 / 8 / 1024, GELU, pre-norm (`norm_first=True`) |
| Positional encoding | Learned, up to 64 positions |
| Vocabulary | 23 — `<pad>`, `<bos>`, `<eos>`, and the 20 canonical amino acids |
| Parameters | 4,767,255 (the six blocks are 99.4% of them) |
| Checkpoint | Epoch 14 of the official-only training run, frozen |

The training corpus is the organizers' `antibacterial.fasta`: 39,448 sequences,
split 37,935 train / 1,513 validation. **The corpus size is not the training set
size**, and this repository does not describe it as 39,448 training examples.

Development compared model sizes, an official-only run against data-expansion
alternatives including AMPSphere subsets, and staged training. Those expansion
corpora were **not** used for the released checkpoint. Later work compared
autoregressive training through 32 epochs, length-conditioned autoregressive
through 50, and masked diffusion through 100, across three training seeds each;
none of it justified replacing epoch 14. The epoch was chosen on distribution
measurements and a second generation seed, not on validation loss alone.

Multiple generation seeds are not independent retraining, and performance on
development references does not establish generalization. Performance after
retraining with a different training seed remains unmeasured.

There is no non-public training data. Nothing private, unpublished or restricted
entered the model at any point — see [NOTICE.md](../NOTICE.md).

## Why selection matches a distribution instead of ranking a score

The library is scored against reference distributions, and the challenge's own
description of the aggregation score says it was *"tuned to discriminate between
known potent and weak antimicrobial peptides, as well as negative examples."*
That makes matching the reference distribution the objective on five of the six
axes, and it makes ranking actively harmful.

We measured this. Selecting 100 sequences, evaluated with ESM2-650M against the
organizers' reference set:

| Selection rule | FBD ↓ | MMD ↓ | Precision | Recall | Coverage |
|---|---|---|---|---|---|
| Reward closeness to the reference centroid | 5.11 | 11.58 | 1.00 | **0.11** | 0.78 |
| Force high net charge (a hard filter) | 5.13 | 13.07 | 1.00 | 0.14 | 0.74 |
| One pick per reference mode | 1.90 | 2.00 | 0.99 | 0.72 | 1.00 |
| **Match each mode's exemplar 1:1** | **1.79** | **1.08** | 0.96 | **0.75** | 1.00 |
| *For scale:* 100 real AMPs at random | 1.71 | 1.15 | 0.95 | 0.91 | 1.00 |

Two things are worth extracting from that table.

**`Precision 1.00` with `Recall 0.11` diagnoses the failure.** Every pick sat
inside the reference distribution; together they covered a sliver of it. FBD
compares means *and* covariances, so a collapsed spread keeps FBD high no matter
how well the mean lines up. We confirmed this the expensive way: a variant that
matched the reference charge distribution bin for bin still scored FBD 5.11.

**Small-sample floors have to be measured, not assumed.** Twelve random draws of
100 real AMPs gave FBD 1.48–1.84 (median 1.64). A 100-sequence selection cannot
do better than that, so the headline number 1.79 is at the floor, and comparing a
100-sequence FBD against a 50,000-sequence FBD — 1.79 against 0.21 — is
meaningless. We made that mistake first.

## Where matching helps and where it does not

Matching the reference distribution wins for the top-100 and **loses for the
library**. The difference is what the baseline was.

| Stage | Baseline being replaced | Outcome |
|---|---|---|
| top-100 | a subset picked by score, so biased | FBD 2.61 -> 1.79 |
| library | all 50,000 sampled, so unbiased | FBD 0.2104 -> **0.3573** |

Measured at library scale, with 250,000 candidates and each of the 39,448
reference sequences claiming its closest:

    FBD       0.2104 -> 0.3573     MMD        0.3072 -> 0.8285
    Recall    0.9160 -> 0.8735     Diversity  0.8570 -> 0.8404
    charge     2.096 -> 3.097      Coverage    0.750 -> 0.835

Charge (reference 2.927) and coverage improved and it was not enough.
Amphipathicity overshot the reference — 0.408 against 0.368 — which is the
signature of the failure: the selection tracked the reference's *property*
distribution more tightly than a real sample of it would, and paid for that in
embedding-space spread.

An unbiased sample is already the best available estimate of the generator's
distribution, so choosing a subset of it can only discard information. The
library is therefore the sample itself, in sampling order, and the remaining gap —
library FBD 0.2104 against 0.0613 for a fresh draw of real AMPs — belongs to the
generator, not the selector.

`select_library` stays in the codebase, unused by the pipeline. It is the right
tool the moment the candidate pool stops being an unbiased sample, and deleting it
would delete the evidence for this decision.

## Top-100 selection, concretely

The feature space is net charge at pH 7, length, amphipathicity, mean
hydrophobicity, and the twenty residue fractions, standardised by the reference
set's own spread because the reference is the yardstick.

The reference set is split into 100 clusters. Each cluster's medoid — a real
reference sequence, never an average — takes the closest candidate that clears both
gates. Larger clusters are served first, so an interrupted run still covers the
dominant modes. The list is ordered by ascending distance to the matched exemplar,
so the candidates we are most confident about come first.

That ordering is deliberate: the organizers' main text and FAQ disagree about
whether the experimental draw comes from the top 100 or the top 50, so the order
is kept meaningful rather than arbitrary.

## Why properties and not a protein language model

ESM2 would be the natural feature space and we cannot use it. The entry point
must run without network access, on the CPU, and produce identical output when
repeated; a 650M-parameter model satisfies none of those, and embedding 50,000
sequences on a CPU is not a few minutes' work.

So the split is explicit: **selection uses CPU properties, evaluation uses ESM2.**
Every number in this document that names FBD, MMD, Precision, Recall or Coverage
was produced by ESM2-650M offline. Nothing in `uv run generate` touches it.

The interesting question this raises — can a cheap feature space improve a metric
measured in an expensive one — is the experiment, and the table above is its
result.

## Gates and screens

**Mandatory (the challenge's).** Canonical 20 amino acids; length 8–50; no exact
match against the reference set; top-100 sequences within a Levenshtein ratio of
0.80 against every reference sequence. We exclude against MarLys v3 (103,143
sequences), a superset of the reference set, because a measured run leaked two
sequences when excluding against the reference set alone — and a leak is a
validator failure, not a warning.

The Levenshtein ratio here uses **substitution cost 2**:
`(len(a) + len(b) - distance) / (len(a) + len(b))`. With substitution cost 1 the
values differ and the gate moves, so the definition is pinned in code and tested.

**Ours (synthesis screens).** No cysteine; no three identical residues in a row;
fewer than three K/R in any five-residue window. These apply to the top-100 only.
The library is submitted unscreened, and that is a measured decision:

| Set | Screen pass rate |
|---|---|
| Our library | 37.0% |
| Real AMPs (reference set) | 36.9% |
| UniProt peptides (negative examples) | 41.3% |

The negative class passes at a *higher* rate than real AMPs. Maximising the pass
rate would therefore push the library away from the reference distribution, which
is the opposite of what the aggregation score rewards. The 0.1 pp gap against
real AMPs is left where it is.

Three departures from the HydrAMP filters the organizers' starter kit uses:

1. **Positive residues are K and R, not K, R and H.** HydrAMP's own source counts
   K and R; histidine's pKa is near 6 and it is largely uncharged at pH 7. The
   codebase we inherited counted H, which made the screens 3.2 pp stricter.
2. **"Three identical hydrophobic residues in a row" is dropped.** It is a strict
   logical subset of "three identical residues in a row". Measured across four
   datasets it caught zero sequences the general rule missed.
3. **Cysteine is counted rather than detected.** Library analysis tolerates one;
   top-100 selection rejects any, because with ~20,000 candidates remaining there
   is no reason to accept an avoidable disulfide risk.

A by-product worth stating plainly: the dense-positive-window screen is a charge
ceiling in disguise. Measured pass rates by net charge — 98.3% at charge ≤0, 21.3%
at 4–6, 5.3% at 6–8, 2.2% above 8 — mean it removes cationic peptides, and
cationicity is the defining property of antimicrobial peptides. We did **not**
relax the screen: the screened pool already held 1,308 candidates above charge 4
and 210 above charge 6, against 100 slots. The problem was never the screen. It
was that the old ranking never asked for charge.

## Pairwise similarity ceiling

Set to 0.70, and the reason is not that 0.70 scores better. Under mode matching
the selected 100 reach a maximum pairwise ratio of 0.667 on their own, so every
ceiling from 0.70 upward is inactive — measured FBD, MMD, Recall, Coverage and
Diversity are identical from 0.70 to 1.00. A ceiling of 0.60 rejects two
candidates and perturbs the matching slightly.

So 0.70 is a guardrail rather than an active constraint: it does nothing today and
catches a future generator whose samples cluster. The redundancy cost of the looser
setting was measured too — for a random draw of 25 from the 100, the expected
number of pairs above ratio 0.6 is 0.12, against 0.30 for 100 real AMPs. The
selection is less redundant than nature.

## Activity prediction: measured, then excluded from the pipeline

The challenge names AMPredictor, MBC-Attention and Deep-AMP as activity
surrogates. We read the organizers' own published benchmark
([BATTLE-AMP](https://doi.org/10.64898/2026.06.19.733349),
`results/aggregated/regression_results.tsv`) across its nine regression tasks:

| Model | Mean Spearman | Tasks with R²(log2) > 0 |
|---|---|---|
| MBC-Attention | 0.355 | 3 / 9 |
| AMPredictor | 0.337 | 3 / 9 |
| Deep-AMP (best of four variants) | 0.297 | 3 / 9 |
| Deep-AMP (worst variant) | 0.160 | 0 / 9 |
| APEX (all five variants) | 0.066 – 0.195 | **0 / 9** |

Two conclusions. **APEX is unusable**: no variant reaches a positive R² on any of
the nine tasks, and an earlier version of this pipeline ranked by it. And **the
best available predictor reaches Spearman 0.355**, which is not a signal you can
sort 50,000 candidates by.

We also ran the three named surrogates ourselves through the organizers' wrapper
on 100 selected sequences. Deep-AMP's 90th percentile prediction was 14,712 µM —
on real AMPs as well as ours — so its absolute values are not usable even where
its ranking is.

Given that, `uv run generate` contains **no activity predictor**. This is a change
from an earlier version of the pipeline, which cut the lower half by predicted
activity. The selection is distribution matching only. Adding a surrogate would
mean shipping weights whose own training lineage we could not verify, to gain a
ranking signal the organizers' benchmark shows is weak.

As a check that this does not cost activity, we scored the selected top-100 with
Deep-AMP: median predicted MIC 43.9 µM against 120.9 µM for the previous
activity-ranked selection, and 39.8 µM for 100 real AMPs. Distribution matching
moved predicted potency in the right direction without optimising for it.

## Reproducibility

- Single CPU thread, deterministic kernels, fixed seed, no network access.
- The sampler uses a locally seeded generator per batch, so unrelated RNG use
  elsewhere in a process cannot shift the output.
- No key/value cache in the sampler. It would be arithmetically equivalent but
  not bit-equivalent, and bit-equivalence is the requirement.
- `maximum_attempts` is derived from the requested count and batch size. A fixed
  constant is a trap: with batch 512, a cap of 40 tops out at 20,480 sequences, so
  a request for 50,000 could never succeed.

## What is not established

- No experimental antibacterial or haemolysis measurement on any sequence here.
- Generalization after retraining with a different training seed is unmeasured.
- The activity surrogates' rankings are reported as their authors' published
  performance, not as validation of our candidates.
- A human dose-response comparison remains blocked by seven unresolved
  record-linkage conflicts, and its outputs were never used for selection.
