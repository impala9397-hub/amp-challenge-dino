"""``uv run generate`` — the challenge's required entry point.

    uv run generate

writes ``generate/library.fasta`` (50,000 sequences), ``generate/top.fasta`` (the
ordered 100) and ``generate/report.json``. The directory name is fixed by the
organizers' verifier, which looks for exactly those two paths. Every argument has a default, the seed
is fixed, nothing reaches the network, and a repeat run produces identical files.

The pipeline:

    1. read the organizers' reference set and the MarLys exclusion list
    2. sample candidates from the frozen checkpoint (CPU, single thread)
    3. keep only what meets the mandatory sequence requirements
    4. take the first 50,000 as the library, **unselected**
    5. apply our synthesis screens to build the top-100 candidate pool
    6. select and order the top-100, one per reference mode, inside both gates

Step 4 is deliberately not a selection, and that is a measured decision rather
than a shortcut. Distribution-matching the library against the reference set was
tried and made every headline metric worse (FBD 0.2104 -> 0.3573, MMD 0.3072 ->
0.8285); an unbiased sample is already the best estimate of the generator's
distribution, so choosing a subset of it only discards information. The reasoning
is in ``selection.py``.

The library is also submitted **unscreened**: the synthesis screens in step 5
apply only to the 100 sequences that would actually be synthesised, for the reason
given in ``screens.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections.abc import Sequence
from pathlib import Path

from dino_amp import model as generator_model
from dino_amp import sampling, selection
from dino_amp.screens import screen_library, violates_synthesis_screens
from dino_amp.similarity import BACKEND, ReferenceIndex

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PACKAGE_ROOT / "weights/generator.pt"
DEFAULT_REFERENCE = PACKAGE_ROOT / "data/antibacterial.fasta"
DEFAULT_EXCLUSIONS = PACKAGE_ROOT / "data/marlys-v3-sequences.txt.gz"
# The organizers' verify_submission.py hard-codes ENTRY_POINT = "generate" and
# looks for <repo>/generate/library.fasta and <repo>/generate/top.fasta. Writing
# anywhere else fails verification at step [4], so this name is not a preference.
DEFAULT_OUTPUT = PACKAGE_ROOT / "generate"

LIBRARY_SIZE = 50_000
TOP_SIZE = 100
# The library is the sample itself, so one library's worth is all that is needed.
# The sampler already adds internal headroom for duplicates and exclusions. Raising
# this only costs time: at the measured 15.3 sequences/second on one CPU thread,
# every extra 50,000 candidates is another 55 minutes for no measured gain.
CANDIDATE_MULTIPLE = 1
MINIMUM_EXCLUSIONS = 100_000     # guards against a truncated exclusion file


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


def read_exclusions(path: Path) -> tuple[str, ...]:
    """One sequence per line, gzipped. A short file is an error, not a warning.

    A silently smaller exclusion set means reference sequences leak into the
    library, and the organizers' validator rejects the submission for it.
    """
    with gzip.open(path, "rt") as handle:
        sequences = tuple(line.strip() for line in handle if line.strip())
    if len(sequences) < MINIMUM_EXCLUSIONS:
        raise ValueError(
            f"exclusion list holds {len(sequences):,} sequences, expected at least "
            f"{MINIMUM_EXCLUSIONS:,} — the file looks truncated"
        )
    return sequences


def write_fasta(path: Path, sequences: Sequence[str], prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f">{prefix}_{index:05d}\n{sequence}\n"
                for index, sequence in enumerate(sequences, start=1))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the AMP Challenge library and ordered top list."
    )
    parser.add_argument("--library-size", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-size", type=int, default=TOP_SIZE)
    parser.add_argument("--candidate-multiple", type=int, default=CANDIDATE_MULTIPLE)
    parser.add_argument("--seed", type=int, default=sampling.DEFAULT_SEED)
    parser.add_argument("--selection-seed", type=int, default=selection.DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=sampling.DEFAULT_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, default=sampling.DEFAULT_BATCH)
    parser.add_argument("--minimum-length", type=int, default=8)
    parser.add_argument("--maximum-length", type=int, default=50)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pairwise-ceiling", type=float, default=selection.DEFAULT_PAIRWISE_CEILING
    )
    parser.add_argument(
        "--reference-margin", type=float, default=selection.DEFAULT_REFERENCE_MARGIN
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - started:7.1f}s] {message}", flush=True)

    reference = read_fasta(args.reference)
    exclusions = read_exclusions(args.exclusions)
    # The union guards against the reference file drifting out of the exclusion
    # list. Measured today they coincide, but a future reference update must not
    # quietly leak sequences into the library.
    excluded = frozenset(reference) | frozenset(exclusions)
    log(f"reference {len(reference):,} · exclusions {len(exclusions):,} "
        f"· union {len(excluded):,} · similarity backend {BACKEND}")

    model = generator_model.load(args.weights)
    log(f"generator {model.parameter_count:,} parameters · CPU, {sampling.THREADS} thread")

    wanted = args.library_size * args.candidate_multiple
    log(f"sampling {wanted:,} candidates (this is the slow step, "
        f"roughly {wanted / 15.3 / 60:.0f} minutes at the measured rate)")
    raw = sampling.sample(
        model,
        count=wanted,
        seed=args.seed,
        temperature=args.temperature,
        minimum_length=args.minimum_length,
        maximum_length=args.maximum_length,
        batch_size=args.batch_size,
        excluded=excluded,
        progress=True,
    )
    candidates, counts = screen_library(raw, excluded=excluded)
    log(f"sampled {len(raw):,} → meet requirements {len(candidates):,} {counts}")

    # The library is the sample itself, in sampling order. Not a selection —
    # see the module docstring and selection.py for the measurement that settled
    # this. The top-100 pool is then drawn from the library, because the challenge
    # asks for a ranked subset of the submitted library rather than a separate set.
    library = candidates[: args.library_size]
    log(f"library {len(library):,} taken unselected from {len(candidates):,} candidates")

    top_pool = [s for s in library if not violates_synthesis_screens(s, cysteine_limit=0)]
    log(f"top-100 pool after synthesis screens: {len(top_pool):,} of {len(library):,}")

    index = ReferenceIndex(reference)
    top = selection.select_top(
        top_pool,
        reference,
        index,
        size=args.top_size,
        seed=args.selection_seed,
        reference_margin=args.reference_margin,
        pairwise_ceiling=args.pairwise_ceiling,
    )
    log(top.describe())

    if len(library) != args.library_size:
        raise RuntimeError(f"library holds {len(library):,}, expected {args.library_size:,}")
    if len(top.sequences) != args.top_size:
        raise RuntimeError(f"top list holds {len(top.sequences)}, expected {args.top_size}")
    leaked = sorted(set(library) & excluded)
    if leaked:
        raise RuntimeError(f"{len(leaked)} library sequences appear in the exclusion set")

    write_fasta(args.output_dir / "library.fasta", library, "dino_lib")
    write_fasta(args.output_dir / "top.fasta", top.sequences, "dino_top")
    report = {
        "library_size": len(library),
        "top_size": len(top.sequences),
        "candidates_sampled": len(raw),
        "candidates_after_requirements": len(candidates),
        "screen_counts": counts,
        "library_selection": "none — the library is the sample in sampling order",
        "top_pool_size": len(top_pool),
        "max_reference_similarity": top.max_reference_similarity,
        "rejected_by_reference": top.rejected_by_reference,
        "rejected_by_pairwise": top.rejected_by_pairwise,
        "seed": args.seed,
        "selection_seed": args.selection_seed,
        "temperature": args.temperature,
        "similarity_backend": BACKEND,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {args.output_dir}/library.fasta, top.fasta, report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
