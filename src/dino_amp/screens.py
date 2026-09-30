"""Sequence requirements and synthesis-oriented screens.

Two kinds of rule live here and they are not interchangeable.

**Sequence requirements** come from the challenge and are mandatory: canonical
20 amino acids, length 8-50, and no exact match against the reference set. A
library violating any of these fails the organizers' validator.

**Synthesis screens** are ours. The challenge does *not* require them — we
checked the README's Sequence Requirements and the Kaggle constraints, and no
synthesizability rule appears in either. They exist because a peptide nobody can
make is not a useful recommendation, and they are applied only when picking the
top-100 that would actually be synthesized. The 50,000-member library is
submitted unfiltered.

That distinction is load-bearing, and measurement is why. The screens pass:

    our library            37.0%
    real AMPs (reference)  36.9%
    UniProt peptides       41.3%

The negative class passes at a *higher* rate than real AMPs. So "maximize the
pass rate" would push the library away from the reference distribution, which is
the opposite of what the aggregation score rewards. The library therefore stays
unscreened and the measured 0.1 pp gap against real AMPs is left alone.

## Why these particular screens

They follow HydrAMP's filters, as used by the organizers' own starter kit, with
three measured departures:

1. **Positive residues are K and R, not K, R and H.** HydrAMP's source counts
   ``K`` and ``R`` only, and the starter kit calls that function. Histidine's
   pKa is about 6, so it is largely uncharged at pH 7.
2. **The "three identical hydrophobic residues in a row" rule is dropped.** It
   is a strict logical subset of "three identical residues in a row" — any
   sequence it rejects, the more general rule already rejected. Measured across
   four datasets it caught **zero** sequences the general rule missed.
3. **Cysteine is counted, not merely detected.** A single cysteine is ordinary
   in synthesis; two or more invite disulfide scrambling. Library-level analysis
   uses "two or more", while top-100 selection rejects any cysteine, because
   there a single avoidable risk is not worth taking when 20,000 candidates
   remain.
"""

from __future__ import annotations

from collections.abc import Iterable

from dino_amp.properties import AMINO_ACIDS

MINIMUM_LENGTH = 8
MAXIMUM_LENGTH = 50

POSITIVE_RESIDUES = frozenset("KR")
POSITIVE_WINDOW = 5
POSITIVE_LIMIT = 3
IDENTICAL_RUN = 3

_ALLOWED = frozenset(AMINO_ACIDS)


def violates_sequence_requirements(sequence: str) -> tuple[str, ...]:
    """Mandatory challenge requirements. Empty tuple means the sequence is fine."""
    reasons: list[str] = []
    if not sequence:
        return ("empty",)
    if not set(sequence) <= _ALLOWED:
        reasons.append("non_canonical_residue")
    if not MINIMUM_LENGTH <= len(sequence) <= MAXIMUM_LENGTH:
        reasons.append("length_out_of_range")
    return tuple(reasons)


def has_dense_positive_window(sequence: str) -> bool:
    """Three or more K/R inside any window of five residues."""
    for start in range(max(0, len(sequence) - POSITIVE_WINDOW + 1)):
        window = sequence[start : start + POSITIVE_WINDOW]
        if sum(1 for residue in window if residue in POSITIVE_RESIDUES) >= POSITIVE_LIMIT:
            return True
    return False


def has_identical_run(sequence: str) -> bool:
    """Three identical residues in a row."""
    for start in range(max(0, len(sequence) - IDENTICAL_RUN + 1)):
        window = sequence[start : start + IDENTICAL_RUN]
        if len(set(window)) == 1:
            return True
    return False


def violates_synthesis_screens(sequence: str, *, cysteine_limit: int = 1) -> tuple[str, ...]:
    """Our synthesis screens. ``cysteine_limit`` is the largest tolerated count.

    ``cysteine_limit=0`` rejects any cysteine (used for top-100 selection);
    ``cysteine_limit=1`` tolerates a single one (used for library analysis).
    """
    reasons: list[str] = []
    if sequence.count("C") > cysteine_limit:
        reasons.append("cysteine_count")
    if has_dense_positive_window(sequence):
        reasons.append("dense_positive_window")
    if has_identical_run(sequence):
        reasons.append("identical_residue_run")
    return tuple(reasons)


def screen_library(
    sequences: Iterable[str],
    *,
    excluded: frozenset[str],
) -> tuple[list[str], dict[str, int]]:
    """Apply only the mandatory requirements, de-duplicate, and drop exclusions.

    No synthesis screen is applied here on purpose — see the module docstring.
    """
    counts = {"seen": 0, "duplicate": 0, "excluded": 0, "requirements": 0, "kept": 0}
    kept: list[str] = []
    seen: set[str] = set()
    for sequence in sequences:
        counts["seen"] += 1
        if sequence in seen:
            counts["duplicate"] += 1
            continue
        seen.add(sequence)
        if sequence in excluded:
            counts["excluded"] += 1
            continue
        if violates_sequence_requirements(sequence):
            counts["requirements"] += 1
            continue
        kept.append(sequence)
        counts["kept"] += 1
    return kept, counts
