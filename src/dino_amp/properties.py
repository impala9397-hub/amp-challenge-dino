"""Physicochemical properties, computed to match the challenge's scoring stack.

The organizers evaluate with `seqme`, which delegates these three descriptors to
`modlamp`. We reimplement them instead of depending on `modlamp` for one
concrete reason: `modlamp` is BSD-3 itself, but its dependency list includes
`mysql-connector-python`, which is GPL-2.0. This repository is MIT and is meant
to be installable without pulling a copyleft dependency chain.

Equality with `modlamp` is not assumed — it is tested. `tests/test_properties.py`
compares 300+ sequences against `modlamp` (a development-only extra) and
requires exact agreement. Two details matter and are easy to get wrong:

  * `modlamp` rounds the net charge to three decimals; we round identically.
  * `modlamp`'s Eisenberg table carries the rounded values (F 1.20, I 1.40,
    L 1.10, R -2.50, V 1.10) rather than the two-decimal values printed in the
    original paper. The scoring stack uses `modlamp`, so we follow `modlamp`.

Sources
  charge  Bjellqvist method; pKa values from the CRC Handbook of Chemistry and
          Physics, 96th edition, as used by `modlamp`.
  scale   Eisenberg et al., consensus hydrophobicity.
  moment  Eisenberg et al.; sliding window of 11 residues at 100 degrees, which
          is the alpha-helical periodicity.
"""

from __future__ import annotations

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

EISENBERG: dict[str, float] = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.20,
    "G": 0.48, "H": -0.40, "I": 1.40, "K": -1.50, "L": 1.10,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.50,
    "S": -0.18, "T": -0.05, "V": 1.10, "W": 0.81, "Y": 0.26,
}

POSITIVE_PKS: dict[str, float] = {"Nterm": 9.38, "K": 10.67, "R": 12.10, "H": 6.04}
NEGATIVE_PKS: dict[str, float] = {"Cterm": 2.15, "D": 3.71, "E": 4.15, "C": 8.14, "Y": 10.10}

DEFAULT_PH = 7.0
MOMENT_WINDOW = 11
MOMENT_ANGLE = 100


def net_charge(sequence: str, ph: float = DEFAULT_PH) -> float:
    """Net charge at the given pH. Each terminus counts once."""
    positive = 0.0
    for residue, pk in POSITIVE_PKS.items():
        count = 1.0 if residue in ("Nterm", "Cterm") else float(sequence.count(residue))
        ratio = 10.0 ** (pk - ph)
        positive += count * (ratio / (ratio + 1.0))
    negative = 0.0
    for residue, pk in NEGATIVE_PKS.items():
        count = 1.0 if residue in ("Nterm", "Cterm") else float(sequence.count(residue))
        ratio = 10.0 ** (ph - pk)
        negative += count * (ratio / (ratio + 1.0))
    return round(positive - negative, 3)


def hydrophobicity(sequence: str) -> float:
    """Mean Eisenberg hydrophobicity over the residues."""
    return float(np.mean([EISENBERG[residue] for residue in sequence]))


def hydrophobic_moment(
    sequence: str, window: int = MOMENT_WINDOW, angle: int = MOMENT_ANGLE
) -> float:
    """Mean hydrophobic moment over sliding windows — the amphipathicity.

    A sequence shorter than the window is treated as a single window, which is
    what `modlamp` does.
    """
    span = min(window, len(sequence))
    values = np.array([EISENBERG[residue] for residue in sequence], dtype=float)
    radians = angle * (np.pi / 180.0) * np.arange(span)
    cosine, sine = np.cos(radians), np.sin(radians)
    moments = [
        np.sqrt((values[start : start + span] * sine).sum() ** 2
                + (values[start : start + span] * cosine).sum() ** 2) / span
        for start in range(len(values) - span + 1)
    ]
    return float(np.mean(moments))


FEATURE_NAMES: tuple[str, ...] = (
    "charge", "length", "amphipathicity", "hydrophobicity",
    *(f"fraction_{residue}" for residue in AMINO_ACIDS),
)


def feature_matrix(sequences: list[str], ph: float = DEFAULT_PH) -> np.ndarray:
    """The space in which library selection matches the reference distribution.

    Columns: charge, length, amphipathicity, hydrophobicity, then the twenty
    residue fractions. Everything here is pure NumPy on the CPU and depends on
    nothing but the sequence, which is what the reproducibility requirement
    needs — a protein language model would need network access and a GPU, and
    would not give byte-identical output across machines.

    Composition is included because it is the only thing in this feature set
    that carries *which* residues are present rather than an aggregate of them.
    """
    index = {residue: position for position, residue in enumerate(AMINO_ACIDS)}
    out = np.zeros((len(sequences), 4 + len(AMINO_ACIDS)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        out[row, 0] = net_charge(sequence, ph)
        out[row, 1] = len(sequence)
        out[row, 2] = hydrophobic_moment(sequence)
        out[row, 3] = hydrophobicity(sequence)
        counts = np.zeros(len(AMINO_ACIDS))
        for residue in sequence:
            counts[index[residue]] += 1.0
        out[row, 4:] = counts / len(sequence)
    return out
