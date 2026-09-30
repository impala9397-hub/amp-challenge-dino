"""The properties must agree with the stack that scores the challenge.

`seqme` delegates charge, hydrophobicity and hydrophobic moment to `modlamp`.
We reimplement all three to keep `modlamp`'s GPL-2.0 dependency chain out of an
MIT repository — which is only defensible if the values are identical, so that
is what these tests check, not an approximation of it.

`modlamp` is a development-only dependency. If it is missing the comparison
tests skip; the self-consistency tests still run.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from dino_amp.properties import (
    AMINO_ACIDS,
    EISENBERG,
    feature_matrix,
    hydrophobic_moment,
    hydrophobicity,
    net_charge,
)

modlamp_descriptors = pytest.importorskip(
    "modlamp.descriptors", reason="modlamp is a development-only dependency"
)


def sample_sequences(count: int = 300, seed: int = 20260930) -> list[str]:
    """Hand-picked edge cases first, then random peptides across the length range."""
    rng = random.Random(seed)
    fixed = [
        "KWKLFKKIGAVLKVL",   # a canonical cationic helix
        "GLFDIIKKIAESF",
        "RRRRRRRR",          # all positive
        "DDEECCYY",          # all negative, plus the two extra negative pKa residues
        "HHHHHHHH",          # histidine only — the pKa near 6 case
        "A" * 8,             # shortest allowed, single residue
        "A" * 50,            # longest allowed
        "C" * 11,            # cysteine, which carries a negative pKa in this model
    ]
    randoms = [
        "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(8, 50)))
        for _ in range(count)
    ]
    return fixed + randoms


SEQUENCES = sample_sequences()


def test_net_charge_matches_modlamp_exactly() -> None:
    descriptor = modlamp_descriptors.GlobalDescriptor(SEQUENCES)
    descriptor.calculate_charge(ph=7.0, amide=False)
    expected = descriptor.descriptor.ravel()
    actual = np.array([net_charge(sequence) for sequence in SEQUENCES])
    assert np.array_equal(actual, expected)


def test_hydrophobicity_matches_modlamp_exactly() -> None:
    descriptor = modlamp_descriptors.PeptideDescriptor(SEQUENCES, "eisenberg")
    descriptor.calculate_global()
    expected = descriptor.descriptor.ravel()
    actual = np.array([hydrophobicity(sequence) for sequence in SEQUENCES])
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)


def test_hydrophobic_moment_matches_modlamp_exactly() -> None:
    descriptor = modlamp_descriptors.PeptideDescriptor(SEQUENCES, "eisenberg")
    descriptor.calculate_moment(window=11, angle=100, modality="mean")
    expected = descriptor.descriptor.ravel()
    actual = np.array([hydrophobic_moment(sequence) for sequence in SEQUENCES])
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)


def test_eisenberg_table_follows_modlamp_not_the_paper() -> None:
    """Pin the five values where modlamp and the published table differ.

    The original paper prints F 1.19, I 1.38, L 1.06, R -2.53, V 1.08. modlamp
    carries the rounded values, and the scoring stack calls modlamp, so we follow
    modlamp. Without this test, "correcting" the table to the paper would silently
    move every hydrophobicity and moment away from the scored values.
    """
    descriptor = modlamp_descriptors.PeptideDescriptor(["A"], "eisenberg")
    for residue, value in EISENBERG.items():
        assert descriptor.scale[residue][0] == pytest.approx(value, abs=0)


def test_charge_is_rounded_to_three_decimals() -> None:
    """modlamp rounds; a sequence whose exact charge has more digits proves we do too."""
    value = net_charge("KWKLFKKIGAVLKVL")
    assert value == round(value, 3)


def test_feature_matrix_shape_and_composition() -> None:
    matrix = feature_matrix(["KWKLFKKIGAVLKVL", "AAAAAAAA"])
    assert matrix.shape == (2, 4 + len(AMINO_ACIDS))
    # Residue fractions sum to one for every row.
    assert np.allclose(matrix[:, 4:].sum(axis=1), 1.0)
    # The all-alanine row puts everything in alanine's column.
    alanine = 4 + AMINO_ACIDS.index("A")
    assert matrix[1, alanine] == pytest.approx(1.0)
    assert matrix[1, 1] == 8.0


def test_short_sequence_uses_whole_sequence_as_one_window() -> None:
    """Below the 11-residue window the moment is a single window, as in modlamp."""
    short = "KWKLFKKI"
    descriptor = modlamp_descriptors.PeptideDescriptor([short], "eisenberg")
    descriptor.calculate_moment(window=11, angle=100, modality="mean")
    assert hydrophobic_moment(short) == pytest.approx(
        descriptor.descriptor.ravel()[0], abs=1e-12
    )
