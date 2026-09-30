"""Choosing what to submit: the 50,000-member library and the ordered top-100.

## The problem this module exists to solve

Ranking is the obvious way to pick a subset, and it is the wrong way here. Any
score that rewards "close to the reference distribution" concentrates the picks
in the middle of that distribution. Measured with ESM2-650M against the
organizers' reference set, on a 100-sequence selection:

    selection rule                      FBD    MMD   Precision  Recall
    reward closeness to the centroid   5.11  11.58      1.00     0.11
    one pick per reference mode        1.90   2.00      0.99     0.72
    match each mode's exemplar 1:1     1.79   1.08      0.96     0.75
    (for scale) 100 real AMPs          1.71   1.15      0.95     0.91

``Precision 1.00`` with ``Recall 0.11`` is the diagnosis: every pick lay inside
the reference distribution, but together they covered a sliver of it. FBD
compares means *and* covariances, so shrinking the spread keeps it high no
matter how well the mean lines up — matching the reference charge distribution
exactly left FBD at 5.11.

So selection does not rank. It matches: pick reference sequences that span the
distribution, and for each one take the candidate that resembles it most.

## Where matching helps, and where it does not

This is the part that took a failed experiment to learn. Matching helps the
top-100 and **hurts the library**, and the difference is the baseline:

| | Baseline being replaced | Result |
|---|---|---|
| top-100 | a subset picked by score, so *biased* | FBD 2.61 → 1.79 |
| library | all 50,000 sampled, so *unbiased* | FBD 0.2104 → 0.3573 |

Measured at library scale — 250,000 candidates, each of the 39,448 reference
sequences claiming its closest — every headline metric moved the wrong way:

    FBD       0.2104 -> 0.3573     MMD        0.3072 -> 0.8285
    Recall    0.9160 -> 0.8735     Diversity  0.8570 -> 0.8404

Charge (2.10 -> 3.10, reference 2.93) and coverage (0.75 -> 0.83) did improve, and
it was not enough. Amphipathicity overshot the reference (0.408 against 0.368),
which is the signature: the selection matched the reference's property
distribution more tightly than a real sample of it would, and paid for that in
embedding-space spread.

The reason is that an unbiased sample is already the best available estimate of
the generator's distribution. Choosing a subset of it can only discard
information. So `select_library` exists but **the pipeline does not use it for the
submitted library** — see `cli.py`. It is kept because it is the right tool the
moment the candidate pool stops being an unbiased sample, and because removing it
would delete the evidence for why the library is submitted unselected.

The remaining gap — library FBD 0.2104 against 0.0613 for a fresh sample of real
AMPs — is the generator's, not the selector's. Selection cannot close it.

## Why properties rather than a language model

The entry point must run with no network access, on the CPU, and produce
byte-identical output when repeated. A protein language model satisfies none of
those. `properties.feature_matrix` is therefore the matching space: charge,
length, amphipathicity, hydrophobicity and the twenty residue fractions, all
pure NumPy. ESM2 is used to *evaluate* the result, never to produce it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dino_amp.properties import feature_matrix
from dino_amp.similarity import ReferenceIndex, levenshtein_ratio

DEFAULT_SEED = 20260930
DEFAULT_REFERENCE_CEILING = 0.80
DEFAULT_REFERENCE_MARGIN = 0.02
DEFAULT_PAIRWISE_CEILING = 0.70
DEFAULT_NEIGHBOURS = 12
DEFAULT_CHUNK = 400


@dataclass(frozen=True)
class LibrarySelection:
    sequences: tuple[str, ...]
    matched_one_to_one: int
    filled_from_remainder: int
    references_without_free_candidate: int

    def describe(self) -> str:
        return (
            f"library {len(self.sequences):,} "
            f"(1:1 matched {self.matched_one_to_one:,}, "
            f"filled {self.filled_from_remainder:,}, "
            f"references with no free candidate {self.references_without_free_candidate:,})"
        )


@dataclass(frozen=True)
class TopSelection:
    sequences: tuple[str, ...]
    reference_similarity: tuple[float, ...]
    rejected_by_reference: int
    rejected_by_pairwise: int

    @property
    def max_reference_similarity(self) -> float:
        return max(self.reference_similarity) if self.reference_similarity else 0.0

    def describe(self) -> str:
        return (
            f"top {len(self.sequences)} · max reference similarity "
            f"{self.max_reference_similarity:.4f} · rejected "
            f"{self.rejected_by_reference} by reference, "
            f"{self.rejected_by_pairwise} by pairwise"
        )


def _standardise(reference: np.ndarray, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale both sides by the reference's own spread; the reference is the yardstick."""
    mean = reference.mean(axis=0)
    spread = reference.std(axis=0)
    spread[spread == 0] = 1.0
    return (reference - mean) / spread, (pool - mean) / spread


def _nearest_candidates(
    reference_z: np.ndarray, pool_z: np.ndarray, count: int, chunk: int
) -> np.ndarray:
    """For each reference row, the indices of its ``count`` nearest pool rows.

    Chunked because the full distance matrix would be reference x pool floats.
    ``|r|^2`` is dropped: it is constant within a row and cannot change a row's
    own ordering.
    """
    out = np.empty((len(reference_z), count), dtype=np.int64)
    pool_square = (pool_z ** 2).sum(axis=1)
    for start in range(0, len(reference_z), chunk):
        block = reference_z[start : start + chunk]
        distances = pool_square[None, :] - 2.0 * (block @ pool_z.T)
        out[start : start + chunk] = np.argpartition(distances, count, axis=1)[:, :count]
    return out


def select_library(
    candidates: list[str],
    reference: list[str],
    *,
    size: int,
    seed: int = DEFAULT_SEED,
    neighbours: int = DEFAULT_NEIGHBOURS,
    chunk: int = DEFAULT_CHUNK,
) -> LibrarySelection:
    """Pick ``size`` sequences whose property distribution tracks the reference.

    Each reference sequence claims its nearest unclaimed candidate. With more
    slots than reference sequences, the remainder goes to the unclaimed
    candidates that sit closest to any reference.

    The reference order is shuffled from ``seed`` so the result does not depend
    on the order of the input FASTA, and stays identical across runs.
    """
    if size > len(candidates):
        raise ValueError(f"need {size:,} sequences but only {len(candidates):,} candidates")

    reference_features = feature_matrix(reference)
    pool_features = feature_matrix(candidates)
    reference_z, pool_z = _standardise(reference_features, pool_features)

    nearest = _nearest_candidates(reference_z, pool_z, neighbours, chunk)

    claimed = np.zeros(len(candidates), dtype=bool)
    chosen: list[int] = []
    starved = 0
    for reference_row in np.random.default_rng(seed).permutation(len(reference)):
        if len(chosen) >= size:
            break
        for candidate_row in nearest[reference_row]:
            if not claimed[candidate_row]:
                claimed[candidate_row] = True
                chosen.append(int(candidate_row))
                break
        else:
            starved += 1

    matched = len(chosen)
    remaining = size - matched
    if remaining > 0:
        free = np.where(~claimed)[0]
        free_z = pool_z[free]
        free_square = (free_z ** 2).sum(axis=1)
        reference_square = (reference_z ** 2).sum(axis=1)
        best = np.full(len(free), np.inf)
        for start in range(0, len(reference_z), chunk):
            block = reference_z[start : start + chunk]
            distances = (
                free_square[None, :]
                - 2.0 * (block @ free_z.T)
                + reference_square[start : start + chunk, None]
            )
            best = np.minimum(best, distances.min(axis=0))
        # Ties broken by sequence text so the result never depends on input order.
        order = sorted(range(len(free)), key=lambda i: (best[i], candidates[free[i]]))
        chosen.extend(int(free[i]) for i in order[:remaining])

    return LibrarySelection(
        sequences=tuple(candidates[i] for i in chosen),
        matched_one_to_one=matched,
        filled_from_remainder=max(0, remaining),
        references_without_free_candidate=starved,
    )


def select_top(
    candidates: list[str],
    reference: list[str],
    reference_index: ReferenceIndex,
    *,
    size: int = 100,
    modes: int = 100,
    seed: int = DEFAULT_SEED,
    reference_ceiling: float = DEFAULT_REFERENCE_CEILING,
    reference_margin: float = DEFAULT_REFERENCE_MARGIN,
    pairwise_ceiling: float = DEFAULT_PAIRWISE_CEILING,
    search_width: int = 600,
) -> TopSelection:
    """Pick and order ``size`` sequences, one per mode of the reference set.

    The reference is split into ``modes`` clusters; each cluster's medoid — a
    real reference sequence, not an average — gets the closest candidate that
    clears both gates. Larger clusters are served first, so an interrupted run
    still covers the dominant modes.

    The returned order is by ascending distance to the matched exemplar: the
    candidates we are most confident resemble a known antimicrobial peptide come
    first. This matters because the organizers' FAQ and main text disagree about
    whether the experimental draw comes from the top 100 or the top 50, so the
    list is kept meaningfully ordered rather than arbitrary.
    """
    effective_ceiling = reference_ceiling - reference_margin
    if effective_ceiling <= 0:
        raise ValueError("reference_margin must stay below reference_ceiling")

    reference_features = feature_matrix(reference)
    pool_features = feature_matrix(candidates)
    reference_z, pool_z = _standardise(reference_features, pool_features)

    labels = _kmeans(reference_z, modes, seed=seed)
    order: list[tuple[int, int]] = []          # (cluster size, medoid row)
    for cluster in range(modes):
        members = np.where(labels == cluster)[0]
        if not len(members):
            continue
        centre = reference_z[members].mean(axis=0)
        medoid = members[int(np.argmin(np.linalg.norm(reference_z[members] - centre, axis=1)))]
        order.append((len(members), int(medoid)))
    order.sort(key=lambda item: (-item[0], reference[item[1]]))

    taken: set[str] = set()
    picked: list[tuple[float, str, float]] = []   # (distance, sequence, reference similarity)
    rejected_reference = 0
    rejected_pairwise = 0
    for _, medoid in order:
        if len(picked) >= size:
            break
        distances = np.linalg.norm(pool_z - reference_z[medoid], axis=1)
        for index in np.argsort(distances)[:search_width]:
            sequence = candidates[index]
            if sequence in taken:
                continue
            similarity = reference_index.max_similarity(sequence)
            if similarity > effective_ceiling:
                rejected_reference += 1
                continue
            if any(levenshtein_ratio(sequence, other) > pairwise_ceiling
                   for _, other, _ in picked):
                rejected_pairwise += 1
                continue
            taken.add(sequence)
            picked.append((float(distances[index]), sequence, similarity))
            break

    picked.sort(key=lambda item: (item[0], item[1]))
    return TopSelection(
        sequences=tuple(sequence for _, sequence, _ in picked),
        reference_similarity=tuple(similarity for _, _, similarity in picked),
        rejected_by_reference=rejected_reference,
        rejected_by_pairwise=rejected_pairwise,
    )


def _kmeans(points: np.ndarray, clusters: int, *, seed: int, iterations: int = 25) -> np.ndarray:
    """k-means++ initialisation followed by Lloyd iterations.

    Implemented here rather than imported: this package's runtime dependencies
    are NumPy and PyTorch, and adding scikit-learn for one clustering call would
    weigh down an environment the organizers have to install from scratch.
    """
    count = len(points)
    if clusters < 1:
        raise ValueError("clusters must be at least 1")
    if count < clusters:
        raise ValueError(f"cannot split {count} points into {clusters} clusters")

    rng = np.random.default_rng(seed)
    centres = np.empty((clusters, points.shape[1]), dtype=points.dtype)
    centres[0] = points[rng.integers(count)]
    closest = np.linalg.norm(points - centres[0], axis=1) ** 2
    for index in range(1, clusters):
        total = float(closest.sum())
        if total <= 0.0:
            centres[index] = points[rng.integers(count)]
        else:
            centres[index] = points[int(rng.choice(count, p=closest / total))]
        closest = np.minimum(closest, np.linalg.norm(points - centres[index], axis=1) ** 2)

    labels = np.zeros(count, dtype=np.int64)
    for step in range(iterations):
        distances = np.empty((count, clusters), dtype=np.float64)
        for index in range(clusters):
            distances[:, index] = np.linalg.norm(points - centres[index], axis=1)
        updated = np.argmin(distances, axis=1)
        if step > 0 and np.array_equal(updated, labels):
            break
        labels = updated
        for index in range(clusters):
            members = points[labels == index]
            if len(members):
                centres[index] = members.mean(axis=0)
    return labels
