import pytest
import torch

from charlm.data import build_vocab

WORDS = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper", "evelyn", "abby"]


@pytest.fixture
def words() -> list[str]:
    return WORDS * 20


@pytest.fixture
def stoi(words: list[str]) -> dict[str, int]:
    return build_vocab(words)[0]


@pytest.fixture
def gen() -> torch.Generator:
    return torch.Generator().manual_seed(42)
