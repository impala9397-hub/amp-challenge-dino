"""Character-level tokenizer for peptide sequences.

The vocabulary is fixed at 23 symbols: three special tokens plus the twenty
canonical amino acids. It must not change, because the released checkpoint's
embedding table is indexed by these exact ids.
"""

from __future__ import annotations

import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
VOCABULARY = (PAD, BOS, EOS, *ALPHABET)
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

_TO_ID = {symbol: index for index, symbol in enumerate(VOCABULARY)}
_TO_SYMBOL = {index: symbol for symbol, index in _TO_ID.items()}


def encode(sequence: str) -> list[int]:
    """Token ids wrapped in ``<bos> ... <eos>``."""
    return [BOS_ID, *(_TO_ID[symbol] for symbol in sequence), EOS_ID]


def decode(token_ids: list[int] | tuple[int, ...]) -> str:
    """Drop special tokens and stop at ``<eos>``."""
    symbols: list[str] = []
    for token_id in token_ids:
        if token_id in (BOS_ID, PAD_ID):
            continue
        if token_id == EOS_ID:
            break
        symbols.append(_TO_SYMBOL[token_id])
    return "".join(symbols)


def pad_batch(
    sequences: list[str], *, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(tokens, mask)``; the mask marks positions that carry a real token."""
    encoded = [encode(sequence) for sequence in sequences]
    width = max(len(item) for item in encoded)
    tokens = torch.full((len(encoded), width), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool, device=device)
    for row, item in enumerate(encoded):
        tokens[row, : len(item)] = torch.tensor(item, dtype=torch.long, device=device)
        mask[row, : len(item)] = True
    return tokens, mask
