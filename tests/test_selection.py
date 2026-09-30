"""The top-100 selector, including the exemplar-spread correction.

A medoid sits at the centre of its cluster, so 100 medoids are narrower than
the reference they came from. ``DEFAULT_EXEMPLAR_SPREAD`` pushes each one
outward from the reference centroid to undo that. These tests pin the
correction's value, check that it actually changes the picks, and confirm that
both gates and determinism survive it — the constant is the kind of thing that
gets "tidied" back to 1.0 by someone who does not know what it buys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dino_amp import selection
from dino_amp.similarity import ReferenceIndex, levenshtein_ratio

REFERENCE = Path(__file__).resolve().parents[1] / "data/antibacterial.fasta"


def read_fasta(path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        elif line.strip():
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return sequences


@pytest.fixture(scope="module")
def reference() -> list[str]:
    # A slice keeps the clustering fast; the properties under test do not
    # depend on how much of the reference set is used.
    return read_fasta(REFERENCE)[:3000]


@pytest.fixture(scope="module")
def candidates(reference: list[str]) -> list[str]:
    """Mutated reference sequences: close enough to match, far enough to pass."""
    rng = np.random.default_rng(7)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    pool: list[str] = []
    for source in reference[:1500]:
        letters = list(source)
        for position in rng.choice(len(letters), max(2, len(letters) // 3), replace=False):
            letters[int(position)] = residues[int(rng.integers(20))]
        pool.append("".join(letters))
    return list(dict.fromkeys(pool))


def test_default_spread_is_the_measured_correction() -> None:
    # The medoids' spread measured 0.82-0.90x of the reference's; 1/0.86 = 1.16.
    assert selection.DEFAULT_EXEMPLAR_SPREAD == pytest.approx(1.16)


def test_spread_changes_which_candidates_are_picked(reference, candidates) -> None:
    index = ReferenceIndex(reference)
    default = selection.select_top(candidates, reference, index, size=20, modes=20)
    neutral = selection.select_top(
        candidates, reference, index, size=20, modes=20, exemplar_spread=1.0
    )
    assert set(default.sequences) != set(neutral.sequences)


def test_spread_widens_the_selection(reference, candidates) -> None:
    """The point of the correction: a wider spread in the matching space."""
    from dino_amp.properties import feature_matrix

    index = ReferenceIndex(reference)
    reference_spread = feature_matrix(reference).std(axis=0)

    def relative_spread(sequences: tuple[str, ...]) -> float:
        return float(np.mean(feature_matrix(list(sequences)).std(axis=0) / reference_spread))

    default = selection.select_top(candidates, reference, index, size=40, modes=40)
    neutral = selection.select_top(
        candidates, reference, index, size=40, modes=40, exemplar_spread=1.0
    )
    assert relative_spread(default.sequences) > relative_spread(neutral.sequences)


def test_gates_hold_with_the_spread_applied(reference, candidates) -> None:
    index = ReferenceIndex(reference)
    result = selection.select_top(candidates, reference, index, size=20, modes=20)
    ceiling = selection.DEFAULT_REFERENCE_CEILING - selection.DEFAULT_REFERENCE_MARGIN
    assert result.max_reference_similarity <= ceiling
    picks = result.sequences
    for left in range(len(picks)):
        for right in range(left + 1, len(picks)):
            assert levenshtein_ratio(picks[left], picks[right]) <= selection.DEFAULT_PAIRWISE_CEILING


def test_selection_is_deterministic(reference, candidates) -> None:
    index = ReferenceIndex(reference)
    first = selection.select_top(candidates, reference, index, size=20, modes=20)
    second = selection.select_top(candidates, reference, index, size=20, modes=20)
    assert first.sequences == second.sequences


def test_selection_does_not_depend_on_candidate_order(reference, candidates) -> None:
    """Ties are broken by sequence text, so a shuffled input gives the same list."""
    index = ReferenceIndex(reference)
    shuffled = list(candidates)
    np.random.default_rng(99).shuffle(shuffled)
    straight = selection.select_top(candidates, reference, index, size=20, modes=20)
    jumbled = selection.select_top(shuffled, reference, index, size=20, modes=20)
    assert set(straight.sequences) == set(jumbled.sequences)
