"""Pin the parts of the submission contract that silently disqualify if broken.

These are read off the organizers' `scripts/verify_submission.py` rather than
inferred. Each one is a thing that fails verification without any obvious symptom
in our own output, which is why it gets a test instead of a comment.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from dino_amp import cli

REPO = Path(__file__).resolve().parents[1]


def test_output_directory_is_named_generate() -> None:
    """The verifier looks for <repo>/generate/library.fasta and top.fasta.

    `ENTRY_POINT = "generate"` is hard-coded in verify_submission.py and used both
    as the `uv run` target and as the output directory name. Writing to `output/`
    passes our own eyes and fails their step [4].
    """
    assert cli.DEFAULT_OUTPUT == REPO / "generate"


def test_entry_point_is_registered_as_generate() -> None:
    """The verifier runs `uv run --no-sync generate`."""
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    scripts = config["project"]["scripts"]
    assert "generate" in scripts
    assert scripts["generate"] == "dino_amp.cli:main"


def test_every_cli_argument_has_a_default() -> None:
    """Full-track requires that any additional argument have a default.

    The verifier invokes the entry point with no arguments at all, so a required
    argument is an immediate failure.
    """
    parser = cli.build_parser()
    for action in parser._actions:          # noqa: SLF001 - argparse exposes no public view
        if action.dest == "help":
            continue
        assert action.required is False, f"--{action.dest} is required"
        assert action.default is not None, f"--{action.dest} has no default"


def test_default_run_needs_no_arguments() -> None:
    """Parsing an empty argument list must succeed and fill every field."""
    args = cli.build_parser().parse_args([])
    assert args.library_size == 50_000
    assert args.top_size == 100
    assert args.seed == 20260917


def test_uv_sync_does_not_pull_the_development_group() -> None:
    """A bare `uv sync` is what the verifier runs; uv includes `dev` by default.

    Leaving that default in place would install modlamp, and modlamp depends on
    GPL-2.0 mysql-connector-python — in an environment that only needs to run
    `generate`. Emptying the default groups is what keeps this repository's install
    footprint permissive.
    """
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert config["tool"]["uv"]["default-groups"] == []


def test_runtime_dependencies_are_permissive_and_few() -> None:
    """The runtime set is the licensing claim; a new entry here needs a decision."""
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    names = {
        entry.split(">")[0].split("<")[0].split("=")[0].strip().lower()
        for entry in config["project"]["dependencies"]
    }
    assert names == {"numpy", "torch", "rapidfuzz"}


def test_gpl_packages_are_absent_from_the_runtime_set() -> None:
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    joined = " ".join(config["project"]["dependencies"]).lower()
    for forbidden in ("levenshtein", "modlamp", "mysql-connector"):
        assert forbidden not in joined


def test_shipped_data_is_present_and_complete() -> None:
    """No network access at generate time, so the inputs have to be in the repo."""
    assert (REPO / "weights/generator.pt").exists()
    assert (REPO / "data/antibacterial.fasta").exists()
    assert (REPO / "data/marlys-v3-sequences.txt.gz").exists()


def test_exclusion_list_is_not_truncated() -> None:
    """A short exclusion list leaks reference sequences into the library.

    That is the overlap check at the verifier's step [6], and a measured run
    leaked two sequences when excluding against the reference set alone.
    """
    sequences = cli.read_exclusions(REPO / "data/marlys-v3-sequences.txt.gz")
    assert len(sequences) == 103_143


def test_reference_set_is_the_expected_size() -> None:
    assert len(cli.read_fasta(REPO / "data/antibacterial.fasta")) == 39_448


def test_fasta_writer_emits_non_empty_headers() -> None:
    """The verifier rejects a record whose header is blank."""
    target = Path("/tmp") / "dino_header_check.fasta"
    cli.write_fasta(target, ["KWKLFKKI", "GLFDIIKKIAESF"], "dino_top")
    lines = target.read_text().splitlines()
    assert lines[0] == ">dino_top_00001"
    assert lines[2] == ">dino_top_00002"
    assert all(line.strip() for line in lines)
    target.unlink()


def test_licence_metadata_is_mit() -> None:
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert config["project"]["license"] == "MIT"
    assert (REPO / "LICENSE").read_text().startswith("MIT License")


@pytest.mark.parametrize("document", ["README.md", "NOTICE.md", "docs/METHOD.md", "docs/WRITEUP.md"])
def test_required_documents_exist(document: str) -> None:
    assert (REPO / document).exists()
