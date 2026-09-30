"""The challenge requires identical output on a repeated run, so it is tested.

Full-track eligibility asks for a "fixed default random seed (identical output on
repeated runs)". That is a property of the code, not a promise, so these tests
sample twice and compare. They use a small count to stay fast; the guarantee they
check does not depend on the count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dino_amp import sampling
from dino_amp.model import load
from dino_amp.selection import select_library
from dino_amp.similarity import ReferenceIndex

WEIGHTS = Path(__file__).resolve().parents[1] / "weights/generator.pt"
REFERENCE = Path(__file__).resolve().parents[1] / "data/antibacterial.fasta"

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="generator checkpoint not present"
)


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
def model():
    return load(WEIGHTS)


def test_checkpoint_has_the_documented_shape(model) -> None:
    assert model.parameter_count == 4_767_255
    assert model.config.model_dimension == 256
    assert model.config.layers == 6
    assert model.config.heads == 8


def test_sampling_is_reproducible(model) -> None:
    first = sampling.sample(model, count=64, seed=7, batch_size=64)
    second = sampling.sample(model, count=64, seed=7, batch_size=64)
    assert first == second


def test_sampling_depends_on_the_seed(model) -> None:
    """A test that only checks equality would pass on a sampler that ignores the seed."""
    first = sampling.sample(model, count=64, seed=7, batch_size=64)
    other = sampling.sample(model, count=64, seed=8, batch_size=64)
    assert first != other


def test_sampling_ignores_the_global_random_state(model) -> None:
    """Unrelated RNG use elsewhere in the process must not shift the output."""
    torch.manual_seed(1234)
    np.random.seed(1234)
    first = sampling.sample(model, count=64, seed=7, batch_size=64)
    torch.manual_seed(999)
    np.random.seed(999)
    _ = torch.rand(100)
    second = sampling.sample(model, count=64, seed=7, batch_size=64)
    assert first == second


def test_sampled_sequences_respect_the_length_range(model) -> None:
    sequences = sampling.sample(
        model, count=64, seed=7, batch_size=64, minimum_length=12, maximum_length=20
    )
    assert sequences
    assert all(12 <= len(sequence) <= 20 for sequence in sequences)


def test_excluded_sequences_never_appear(model) -> None:
    sequences = sampling.sample(model, count=128, seed=11, batch_size=128)
    blocked = frozenset(sequences[:10])
    again = sampling.sample(model, count=128, seed=11, batch_size=128, excluded=blocked)
    assert not (set(again) & blocked)


def test_thread_count_is_restored(model) -> None:
    """Determinism pins threads to one; leaving it pinned would slow the caller down."""
    before = torch.get_num_threads()
    sampling.sample(model, count=8, seed=3, batch_size=8)
    assert torch.get_num_threads() == before


def test_library_selection_is_reproducible() -> None:
    reference = read_fasta(REFERENCE)[:2000]
    rng = np.random.default_rng(0)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    candidates = [
        "".join(rng.choice(alphabet, size=int(rng.integers(8, 51))))
        for _ in range(4000)
    ]
    first = select_library(candidates, reference, size=500, seed=42)
    second = select_library(candidates, reference, size=500, seed=42)
    assert first.sequences == second.sequences
    assert len(set(first.sequences)) == 500


def test_library_selection_refuses_an_impossible_request() -> None:
    with pytest.raises(ValueError, match="candidates"):
        select_library(["KWKLFKKI"], ["KWKLFKKIGAVLKVL"], size=10)


def test_reference_index_covers_the_shipped_reference() -> None:
    reference = read_fasta(REFERENCE)
    assert len(reference) == 39_448
    index = ReferenceIndex(reference)
    assert len(index) == 39_448
    # A sequence taken from the reference must come back as an exact match; if it
    # did not, the novelty gate would pass sequences the validator rejects.
    assert index.max_similarity(reference[0]) == pytest.approx(1.0)
