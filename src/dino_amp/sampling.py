"""Drawing candidate sequences from the generator.

Every choice here serves one requirement: **repeating the run must produce the
same file, on any machine.** That is stricter than it sounds and it rules things
out.

* **CPU only.** Matrix reductions on MPS and CUDA are not order-stable, so the
  same weights can give different logits and therefore different samples.
* **One thread.** Multi-threaded CPU reductions have the same problem. This is
  the main reason generation is slow, and it is not negotiable.
* **A local generator object, seeded per batch.** Sampling never reads the global
  RNG, so an unrelated `numpy` or `torch` call elsewhere cannot shift the output.
* **No network.** The checkpoint ships in this repository.

There is no key/value cache: the model re-reads the whole prefix at every step.
That is roughly a 50x cost at the longest length. A cache would be arithmetically
equivalent but not bit-equivalent, and bit-equivalence is the requirement, so the
slow loop stays.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from dino_amp import tokenizer
from dino_amp.model import MAXIMUM_POSITIONS, CausalTransformer

DEVICE = "cpu"
THREADS = 1
DEFAULT_SEED = 20260917
DEFAULT_TEMPERATURE = 1.0
DEFAULT_BATCH = 512
ATTEMPT_HEADROOM = 3.0


@contextmanager
def deterministic() -> Iterator[None]:
    """Pin thread count and deterministic kernels, then restore what was there."""
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)


def sample(
    model: CausalTransformer,
    *,
    count: int,
    seed: int = DEFAULT_SEED,
    temperature: float = DEFAULT_TEMPERATURE,
    minimum_length: int = 8,
    maximum_length: int = 50,
    batch_size: int = DEFAULT_BATCH,
    excluded: frozenset[str] | None = None,
    maximum_attempts: int | None = None,
    progress: bool = False,
) -> list[str]:
    """Draw ``count`` distinct sequences, left to right, one token at a time.

    ``excluded`` is dropped during sampling rather than afterwards, so a run
    cannot come up short after filtering.

    ``maximum_attempts`` is derived from ``count`` and ``batch_size`` when not
    given. It must never be a fixed constant: with batch 512 a cap of 40 tops out
    at 20,480 sequences, so asking for 50,000 could not succeed no matter how
    long it ran.
    """
    if maximum_attempts is None:
        batches_needed = math.ceil(count / max(1, batch_size))
        maximum_attempts = max(4, math.ceil(batches_needed * ATTEMPT_HEADROOM))

    blocked = excluded or frozenset()
    width = min(maximum_length + 2, MAXIMUM_POSITIONS)
    seen: set[str] = set()
    produced: list[str] = []

    model = model.to(DEVICE).eval()
    with deterministic(), torch.no_grad():
        for attempt in range(maximum_attempts):
            if len(produced) >= count:
                break
            generator = torch.Generator(device=DEVICE)
            generator.manual_seed(seed + attempt)
            tokens = torch.full((batch_size, 1), tokenizer.BOS_ID, dtype=torch.long)
            finished = torch.zeros(batch_size, dtype=torch.bool)
            for step in range(width - 1):
                logits = model(tokens)[:, -1, :]
                logits[:, tokenizer.PAD_ID] = -math.inf
                logits[:, tokenizer.BOS_ID] = -math.inf
                if step + 1 < minimum_length:
                    logits[:, tokenizer.EOS_ID] = -math.inf
                probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                choice = torch.multinomial(probabilities, 1, generator=generator)
                choice[finished] = tokenizer.PAD_ID
                tokens = torch.cat([tokens, choice], dim=1)
                finished = finished | (choice.squeeze(1) == tokenizer.EOS_ID)
                if bool(finished.all()):
                    break
            for row in tokens.tolist():
                sequence = tokenizer.decode(row)
                if not minimum_length <= len(sequence) <= maximum_length:
                    continue
                if sequence in seen or sequence in blocked:
                    continue
                seen.add(sequence)
                produced.append(sequence)
                if len(produced) >= count:
                    break
            if progress and (attempt + 1) % 25 == 0:
                print(f"  batch {attempt + 1}: {len(produced):,} sequences", flush=True)
    return produced[:count]
