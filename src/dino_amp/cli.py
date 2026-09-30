"""``uv run generate`` — the challenge's required entry point.

    uv run generate

writes ``output/library.fasta`` (50,000 sequences), ``output/top.fasta`` (the
ordered 100) and ``output/report.json``. Every argument has a default, the seed
is fixed, nothing reaches the network, and a repeat run produces identical files.

The pipeline:

    1. read the organizers' reference set and the MarLys exclusion list
    2. sample candidates from the frozen checkpoint (CPU, single thread)
    3. keep only what meets the mandatory sequence requirements
    4. select the library so its property distribution tracks the reference
    5. apply our synthesis screens to build the top-100 candidate pool
    6. select and order the top-100, one per reference mode, inside both gates

Step 4 is why more candidates are sampled than the library needs: selection
needs a surplus to choose from. Step 5 is applied to the top-100 only — the
library is submitted unscreened, for the reason given in ``screens.py``.
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
DEFAULT_OUTPUT = PACKAGE_ROOT / "output"

LIBRARY_SIZE = 50_000
TOP_SIZE = 100
CANDIDATE_MULTIPLE = 5           # sample 5x the library size, then select
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
    log(f"sampling {wanted:,} candidates (this is the slow step)")
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

    chosen = selection.select_library(
        candidates, reference, size=args.library_size, seed=args.selection_seed
    )
    log(chosen.describe())
    library = list(chosen.sequences)

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
        "library_selection": {
            "matched_one_to_one": chosen.matched_one_to_one,
            "filled_from_remainder": chosen.filled_from_remainder,
            "references_without_free_candidate": chosen.references_without_free_candidate,
        },
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
