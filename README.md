# Dino — autoregressive peptide generation for the AMP Challenge 2027

A 4.8M-parameter character-level transformer proposes 50,000 candidate
antimicrobial peptides, and a distribution-matching selector picks the ordered
100 that would go to the bench.

> These are computational proposals. Antibacterial efficacy and human safety have
> **not** been established for any sequence in this repository.

## Run it

```bash
uv run generate
```

That writes `output/library.fasta` (50,000 sequences), `output/top.fasta` (the
ordered 100) and `output/report.json`. Every argument has a default, the seed is
fixed, and nothing reaches the network — the checkpoint and both data files ship
in the repository.

It is slow on purpose: generation is pinned to a single CPU thread because
multi-threaded and GPU reductions are not order-stable, and the challenge asks
for identical output on a repeated run. Expect a few hours, dominated by
sampling.

```bash
uv run pytest            # includes a check against the scoring stack's own descriptors
```

## What the pipeline does

| Step | What happens |
|---|---|
| 1 | Read the organizers' 39,448-sequence reference set and the MarLys exclusion list |
| 2 | Sample 250,000 candidates from the frozen checkpoint, left to right, on the CPU |
| 3 | Keep what meets the mandatory requirements: canonical residues, length 8–50, no exact reference match |
| 4 | Select 50,000 so that their property distribution tracks the reference set |
| 5 | Apply our synthesis screens to form the top-100 candidate pool |
| 6 | Select and order 100, one per mode of the reference distribution, inside both similarity gates |

Step 2 deliberately samples five times the library size. Selection needs a
surplus to choose from; without it, step 4 has nothing to do.

## The one idea worth knowing

**Ranking is the wrong way to pick a subset here, and we measured why.**

Any score that rewards "close to the reference distribution" piles the picks into
the middle of that distribution. Measured with ESM2-650M against the reference
set, selecting 100 sequences:

| Selection rule | FBD ↓ | MMD ↓ | Precision | Recall |
|---|---|---|---|---|
| Reward closeness to the reference centroid | 5.11 | 11.58 | 1.00 | **0.11** |
| One pick per reference mode | 1.90 | 2.00 | 0.99 | 0.72 |
| **Match each mode's exemplar 1:1** | **1.79** | **1.08** | 0.96 | **0.75** |
| *For scale:* 100 real AMPs drawn at random | 1.71 | 1.15 | 0.95 | 0.91 |

`Precision 1.00` with `Recall 0.11` is the tell: every pick landed inside the
reference distribution, and together they covered a sliver of it. FBD compares
means *and* covariances, so a shrunken spread keeps it high however well the mean
lines up — matching the reference charge distribution exactly still left FBD at
5.11.

So selection does not rank. It matches: take reference sequences that span the
distribution, and for each one keep the candidate that resembles it most.

Details, including the failures that led here, are in [docs/METHOD.md](docs/METHOD.md).

## Repository layout

```
src/dino_amp/
  cli.py          uv run generate
  sampling.py     the sampler, and why it is single-threaded CPU
  model.py        the causal transformer (inference only)
  tokenizer.py    23-symbol vocabulary
  properties.py   charge, hydrophobicity, amphipathicity, composition
  similarity.py   Levenshtein ratio and the reference index
  screens.py      mandatory requirements vs our synthesis screens
  selection.py    library and top-100 selection
data/             organizers' reference set (BSD-3), MarLys exclusions (CC0)
weights/          the frozen generator checkpoint
docs/             method and write-up
```

## Licensing

MIT for this repository's code and the checkpoint. The redistributed data keeps
its own terms, and the training-data disclosure is in [NOTICE.md](NOTICE.md) —
in short, the generator saw only the organizers' own corpus, so there is no
non-public dataset to release.

The similarity and property implementations are ours rather than imported
specifically to keep a GPL dependency chain out of an MIT repository; both are
tested for exact agreement against the packages they replace.
