"""The novelty gate is a disqualification boundary, so its definition is pinned.

A submitted top-100 sequence must stay at or below a Levenshtein ratio of 0.80
against every reference sequence. Two things can silently break that:

  * using substitution cost 1 instead of 2, which changes every ratio
  * a length-based pruning bound that is too tight, which would let the reference
    index return a maximum lower than the true one

Both are tested here, with extra weight on pairs near the 0.80 boundary — the
region where a wrong definition changes who passes.
"""

from __future__ import annotations

import random

import pytest

from dino_amp.properties import AMINO_ACIDS
from dino_amp.similarity import (
    BACKEND,
    ReferenceIndex,
    levenshtein_ratio,
    max_similarity_upper_bound,
)


def pure_python_ratio(left: str, right: str) -> float:
    """Reference implementation, written out so the fast path has something to match."""
    length_left, length_right = len(left), len(right)
    total = length_left + length_right
    if total == 0:
        return 1.0
    if length_left == 0 or length_right == 0:
        return 0.0
    previous = list(range(length_right + 1))
    for i in range(1, length_left + 1):
        current = [i] + [0] * length_right
        left_char = left[i - 1]
        for j in range(1, length_right + 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if left_char == right[j - 1] else 2),
            )
        previous = current
    return (total - previous[length_right]) / total


def make_pairs(seed: int = 20260930) -> list[tuple[str, str]]:
    """Unrelated pairs, plus mutated pairs that land near the gate."""
    rng = random.Random(seed)

    def peptide() -> str:
        return "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(8, 50)))

    pairs = [(peptide(), peptide()) for _ in range(400)]
    for _ in range(400):
        base = peptide()
        mutated = list(base)
        for _ in range(rng.randint(1, 4)):
            mutated[rng.randrange(len(mutated))] = rng.choice(AMINO_ACIDS)
        if rng.random() < 0.5 and len(mutated) > 9:
            del mutated[rng.randrange(len(mutated))]
        pairs.append((base, "".join(mutated)))
    return pairs


PAIRS = make_pairs()


def test_backend_matches_the_pure_python_definition() -> None:
    for left, right in PAIRS:
        assert levenshtein_ratio(left, right) == pytest.approx(
            pure_python_ratio(left, right), abs=1e-12
        ), f"{BACKEND} disagrees on {left!r} vs {right!r}"


def test_pairs_near_the_gate_are_actually_covered() -> None:
    """A test that never approaches 0.80 would not test the gate."""
    near = [1 for left, right in PAIRS if 0.70 <= levenshtein_ratio(left, right) <= 0.90]
    assert len(near) >= 20, f"only {len(near)} pairs near the gate; the fixture is too easy"


def test_substitution_costs_two() -> None:
    """One substitution in a 10-mer: distance 2, so ratio is 18/20, not 19/20."""
    assert levenshtein_ratio("AAAAAAAAAA", "AAAAAAAAAC") == pytest.approx(0.9)


def test_identical_and_disjoint_extremes() -> None:
    assert levenshtein_ratio("KWKLFKKI", "KWKLFKKI") == pytest.approx(1.0)
    assert levenshtein_ratio("", "") == pytest.approx(1.0)
    assert levenshtein_ratio("KWKLFKKI", "") == pytest.approx(0.0)


def test_upper_bound_never_understates_the_true_ratio() -> None:
    """If the bound could fall below a real ratio, pruning would discard the answer."""
    for left, right in PAIRS:
        assert max_similarity_upper_bound(len(left), len(right)) >= levenshtein_ratio(
            left, right
        ) - 1e-12


def test_reference_index_agrees_with_brute_force() -> None:
    rng = random.Random(7)

    def peptide() -> str:
        return "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(8, 30)))

    references = [peptide() for _ in range(300)]
    index = ReferenceIndex(references)
    assert len(index) == 300
    for _ in range(40):
        query = peptide()
        expected = max(levenshtein_ratio(query, candidate) for candidate in references)
        assert index.max_similarity(query) == pytest.approx(expected, abs=1e-12)


def test_reference_index_finds_an_exact_match() -> None:
    index = ReferenceIndex(["KWKLFKKIGAVLKVL", "GLFDIIKKIAESF"])
    assert index.max_similarity("KWKLFKKIGAVLKVL") == pytest.approx(1.0)
