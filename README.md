# Dino — autoregressive peptide generation for the AMP Challenge 2027

**Team Dino** — Jongwon Im, Hannah Kim, Linda Chen ·
contact **impala9397@gmail.com**

A 4.8M-parameter character-level transformer proposes 50,000 candidate
antimicrobial peptides, and a distribution-matching selector picks the ordered
100 that would go to the bench.

> These are computational proposals. Antibacterial efficacy and human safety have
> **not** been established for any sequence in this repository.

## Run it

```bash
uv run generate
```

That writes `generate/library.fasta` (50,000 sequences), `generate/top.fasta`
(the ordered 100) and `generate/report.json`. The directory name is not a
preference: the organizers' `verify_submission.py` looks for exactly those two
paths. Every argument has a default, the seed is fixed, and nothing reaches the
network — the checkpoint and both data files ship in the repository.

It is slow on purpose: generation is pinned to a single CPU thread because
multi-threaded and GPU reductions are not order-stable, and the challenge asks
for identical output on a repeated run. Expect roughly an hour, almost all of it
sampling at the measured 15.3 sequences per second — 54.8 minutes on the machine
that produced the submitted files.

```bash
uv run --group dev pytest
```

56 tests, including a check of every property implementation against the scoring
stack's own descriptors. The `--group dev` is required: `pyproject.toml` sets
`default-groups = []`, so a bare `uv sync` installs the three runtime
dependencies and nothing else. That keeps `modlamp` — and the GPL-2.0
`mysql-connector-python` it pulls in — out of the environment the organizers
install to run `generate`, while still letting us test against it.

## What the pipeline does

| Step | What happens |
|---|---|
| 1 | Read the organizers' 39,448-sequence reference set and the MarLys exclusion list |
| 2 | Sample 50,000 candidates from the frozen checkpoint, left to right, on the CPU |
| 3 | Keep what meets the mandatory requirements: canonical residues, length 8–50, no exact reference match |
| 4 | Take the sample as the library, **unselected** — matching it to the reference made every metric worse |
| 5 | Apply our synthesis screens to form the top-100 candidate pool |
| 6 | Select and order 100, one per mode of the reference distribution, inside both similarity gates |

Step 4 is the surprising one. Distribution-matching the library against the
reference set was tried and made every headline metric worse — FBD 0.2104 to
0.3573, MMD 0.3072 to 0.8285. An unbiased sample is already the best estimate of
the generator's distribution, so choosing a subset of it only discards
information. Matching still wins for the top-100, where the baseline it replaces
was a score-ranked and therefore biased subset.

## The one idea worth knowing

**Ranking is the wrong way to pick a subset here, and we measured why.**

Any score that rewards "close to the reference distribution" piles the picks into
the middle of that distribution. Measured with ESM2-650M against the reference
set, selecting 100 sequences:

| Selection rule | FBD ↓ | MMD ↓ | Precision | Recall |
|---|---|---|---|---|
| Reward closeness to the reference centroid | 5.11 | 11.58 | 1.00 | **0.11** |
| One pick per reference mode | 1.90 | 2.00 | 0.99 | 0.72 |
| Match each mode's exemplar 1:1 | 1.78 | 1.23 | 0.94 | 0.81 |
| **The same, with the exemplar spread applied** | **1.71** | **0.95** | 0.93 | **0.86** |
| *For scale:* 100 real AMPs drawn at random | 1.56–1.71 | 0.60–1.45 | 0.91–0.95 | 0.90–0.94 |

`Precision 1.00` with `Recall 0.11` is the tell: every pick landed inside the
reference distribution, and together they covered a sliver of it. FBD compares
means *and* covariances, so a shrunken spread keeps it high however well the mean
lines up — matching the reference charge distribution exactly still left FBD at
5.11.

So selection does not rank. It matches: take reference sequences that span the
distribution, and for each one keep the candidate that resembles it most. The
last row of that table is the same rule with one correction — a cluster medoid
sits at its cluster's centre, so a set of 100 medoids is 14% narrower than the
reference, and each one is pushed back outward before it claims a candidate.

Details, including the failures that led here, are in [docs/METHOD.md](docs/METHOD.md).
The submission write-up is [docs/WRITEUP.md](docs/WRITEUP.md).

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
tests/            56 tests, including the organizers' submission contract
data/             organizers' reference set (BSD-3), MarLys exclusions (CC0)
weights/          the frozen generator checkpoint
docs/             METHOD.md (why this design) and WRITEUP.md (the submission)
licenses/         the redistributed data's own licences
NOTICE.md         data provenance and training-data disclosure
```

`generate/` is not tracked. It is what `uv run generate` produces, and the
organizers produce it themselves by running this repository.

## Contact

**Team Dino** — Jongwon Im (team lead), Hannah Kim, Linda Chen.
Kaggle team `Dino`, writeup *Matching the distribution instead of ranking it*.

Reach us at **impala9397@gmail.com**, or open an issue on this repository.

Organizers: this is the surest way to reach us. Kaggle's "share your email with
the host" setting on the team page accepts the change and returns success, but
does not persist it — verified three times after the competition closed — so we
cannot rely on it to receive the compliance notice or the 72-hour window to
resolve issues.

## Licensing

MIT for this repository's code and the checkpoint. The redistributed data keeps
its own terms, and the training-data disclosure is in [NOTICE.md](NOTICE.md) —
in short, the generator saw only the organizers' own corpus, so there is no
non-public dataset to release.

The similarity and property implementations are ours rather than imported
specifically to keep a GPL dependency chain out of an MIT repository; both are
tested for exact agreement against the packages they replace.
