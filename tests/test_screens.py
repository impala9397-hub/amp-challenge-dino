"""The screens decide what can be submitted, so each rule gets a golden case.

The original HydrAMP filter code contained an inverted condition that went
unnoticed because no test asserted per-rule behaviour. These tests exist so the
same thing cannot happen here: every rule has a sequence that must trip it and a
sequence that must not.
"""

from __future__ import annotations

import pytest

from dino_amp.screens import (
    has_dense_positive_window,
    has_identical_run,
    screen_library,
    violates_sequence_requirements,
    violates_synthesis_screens,
)

CLEAN = "KWLRNVGKLFA"   # passes every rule


def test_clean_sequence_passes_everything() -> None:
    assert violates_sequence_requirements(CLEAN) == ()
    assert violates_synthesis_screens(CLEAN) == ()
    assert violates_synthesis_screens(CLEAN, cysteine_limit=0) == ()


@pytest.mark.parametrize(
    ("sequence", "reason"),
    [
        ("KWLRNVGKLFAX", "non_canonical_residue"),
        ("KWLRNVG", "length_out_of_range"),
        ("K" * 51, "length_out_of_range"),
    ],
)
def test_requirement_violations(sequence: str, reason: str) -> None:
    assert reason in violates_sequence_requirements(sequence)


def test_histidine_is_not_a_positive_residue() -> None:
    """The departure from the inherited code, stated as a test.

    HydrAMP's own source counts K and R only. Counting H as positive made the
    screens 3.2 pp stricter and, because the window rule is effectively a charge
    ceiling, removed cationic peptides for no stated reason.
    """
    assert has_dense_positive_window("AKRKAVGLFNW") is True    # K, R, K in one window
    assert has_dense_positive_window("AKHKAVGLFNW") is False   # K, H, K — H does not count
    assert has_dense_positive_window("AHHHAVGLFNW") is False


def test_dense_positive_window_boundaries() -> None:
    assert has_dense_positive_window("KRKAA") is True        # three in exactly five
    assert has_dense_positive_window("KRAAK") is True        # spread across the window
    assert has_dense_positive_window("KRAAAK") is False      # two per window only
    assert has_dense_positive_window("KR") is False          # shorter than the window


def test_identical_run_covers_the_hydrophobic_case() -> None:
    """The dropped fourth rule was a subset of this one, which is why it was dropped.

    Any sequence the "three identical hydrophobic residues" rule rejected is
    already rejected here, so keeping both caught nothing extra — measured as zero
    unique catches across four datasets.
    """
    assert has_identical_run("KWLLLNVGKFA") is True     # LLL, hydrophobic
    assert has_identical_run("KWNQQQGKFAT") is True     # QQQ, not hydrophobic
    assert has_identical_run("KWLLNVGKFA") is False     # only two in a row


def test_cysteine_limit_separates_library_from_top_selection() -> None:
    one = "KWLRNVGKLFAC"
    two = "KWLRCVGKLFAC"
    # Library analysis tolerates a single cysteine.
    assert violates_synthesis_screens(one, cysteine_limit=1) == ()
    assert "cysteine_count" in violates_synthesis_screens(two, cysteine_limit=1)
    # Top-100 selection rejects any.
    assert "cysteine_count" in violates_synthesis_screens(one, cysteine_limit=0)


def test_screen_library_applies_no_synthesis_rule() -> None:
    """The library is submitted unscreened; a cysteine-bearing sequence stays in.

    This is the measured decision recorded in screens.py: the negative class
    passes the synthesis screens at a higher rate than real AMPs, so screening the
    library would move it away from the reference distribution.
    """
    sequences = ["KWLRCVGKLFAC", "KKKRRKKK", CLEAN]
    kept, counts = screen_library(sequences, excluded=frozenset())
    assert set(kept) == set(sequences)
    assert counts["kept"] == 3


def test_screen_library_removes_duplicates_and_exclusions() -> None:
    kept, counts = screen_library(
        [CLEAN, CLEAN, "GLFDIIKKIAESF", "KWLRNVGKLFAX"],
        excluded=frozenset({"GLFDIIKKIAESF"}),
    )
    assert kept == [CLEAN]
    assert counts["duplicate"] == 1
    assert counts["excluded"] == 1
    assert counts["requirements"] == 1
