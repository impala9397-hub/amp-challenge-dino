"""Levenshtein similarity and the reference index the novelty gate needs.

The challenge requires every submitted top-100 sequence to stay at or below a
Levenshtein ratio of 0.80 against every sequence in the organizers' reference
set. Getting the definition wrong changes who passes, so it is stated here:

    ratio = (len(a) + len(b) - distance) / (len(a) + len(b))

where ``distance`` uses **substitution cost 2**. With substitution cost 1 the
numbers differ and the gate moves.

`rapidfuzz` (MIT) computes this exactly, with ``weights=(1, 1, 2)``, and is used
when available. A pure-Python fallback is included so the package has no hard
requirement beyond NumPy and PyTorch; `tests/test_similarity.py` pins the two
implementations to each other. We deliberately do not depend on
`python-Levenshtein`, which is GPL-2.0-or-later.
"""

from __future__ import annotations

from collections.abc import Iterable

REFERENCE_CEILING = 0.80

try:  # pragma: no cover - exercised by whichever path is installed
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein

    def levenshtein_ratio(left: str, right: str) -> float:
        """Similarity in [0, 1]; substitution costs 2."""
        return _RapidLevenshtein.normalized_similarity(left, right, weights=(1, 1, 2))

    BACKEND = "rapidfuzz"
except ImportError:  # pragma: no cover

    def levenshtein_ratio(left: str, right: str) -> float:
        """Similarity in [0, 1]; substitution costs 2. Pure-Python fallback."""
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

    BACKEND = "python"


def max_similarity_upper_bound(length_left: int, length_right: int) -> float:
    """Highest ratio two sequences of these lengths could reach.

    Because ``distance >= |length_left - length_right|``, the ratio cannot
    exceed ``2 * min / (length_left + length_right)``. The reference index uses
    this to skip whole length groups instead of scoring 39,448 pairs per query.
    """
    total = length_left + length_right
    if total == 0:
        return 1.0
    return 2.0 * min(length_left, length_right) / total


class ReferenceIndex:
    """Answers "how close is this to the closest reference sequence?".

    Sequences are grouped by length. For a query we walk the length groups and
    skip any group whose length bound is already at or below the best ratio
    found so far — the bound is exact, so skipping never loses the true maximum.
    """

    def __init__(self, references: Iterable[str]) -> None:
        self._by_length: dict[int, list[str]] = {}
        self._count = 0
        for sequence in references:
            self._by_length.setdefault(len(sequence), []).append(sequence)
            self._count += 1
        self._lengths = sorted(self._by_length)

    def __len__(self) -> int:
        return self._count

    def max_similarity(self, sequence: str) -> float:
        query_length = len(sequence)
        best = 0.0
        # Visit the most promising lengths first so the bound prunes sooner.
        ordered = sorted(
            self._lengths,
            key=lambda length: -max_similarity_upper_bound(query_length, length),
        )
        for length in ordered:
            if max_similarity_upper_bound(query_length, length) <= best:
                break
            for candidate in self._by_length[length]:
                value = levenshtein_ratio(sequence, candidate)
                if value > best:
                    best = value
                    if best >= 1.0:
                        return best
        return best
