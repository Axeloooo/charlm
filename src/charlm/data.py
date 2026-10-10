"""Names dataset, vocabulary and train/dev/test splits (shared by all lectures)."""

from __future__ import annotations

import random
from pathlib import Path

import torch

DEFAULT_NAMES = Path(__file__).resolve().parents[2] / "data" / "names.txt"
END = "."


def load_words(path: str | Path = DEFAULT_NAMES) -> list[str]:
    return Path(path).read_text().split()


def build_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Map characters to ints; the end/start token "." is always index 0."""
    chars = sorted(set("".join(words)) - {END})
    stoi = {END: 0, **{c: i + 1 for i, c in enumerate(chars)}}
    return stoi, {i: c for c, i in stoi.items()}


def split_words(
    words: list[str], seed: int = 42, fractions: tuple[float, float] = (0.8, 0.9)
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle, then split 80/10/10 into train, dev and test words."""
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    n1, n2 = int(fractions[0] * len(shuffled)), int(fractions[1] * len(shuffled))
    return shuffled[:n1], shuffled[n1:n2], shuffled[n2:]


def build_dataset(
    words: list[str], stoi: dict[str, int], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn words into (context of block_size chars, next char) pairs."""
    xs: list[list[int]] = []
    ys: list[int] = []
    for w in words:
        context = [0] * block_size
        for ch in w + END:
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = [*context[1:], ix]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)
