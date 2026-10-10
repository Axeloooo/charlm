import math

import torch
import torch.nn.functional as F

from charlm.bigram import CountBigram, NeuralBigram, count_bigrams
from charlm.data import build_dataset


def test_counts(stoi: dict[str, int]) -> None:
    n = count_bigrams(["ab", "ab"], {".": 0, "a": 1, "b": 2})
    assert n[0, 1] == 2 and n[1, 2] == 2 and n[2, 0] == 2


def test_probability_rows_sum_to_one(words: list[str], stoi: dict[str, int]) -> None:
    model = CountBigram(count_bigrams(words, stoi))
    assert torch.allclose(model.probs.sum(1), torch.ones(len(stoi)))


def test_count_model_beats_uniform(words: list[str], stoi: dict[str, int]) -> None:
    model = CountBigram(count_bigrams(words, stoi))
    assert model.nll(words, stoi) < math.log(len(stoi))


def test_sampling_terminates(words: list[str], stoi: dict[str, int], gen: torch.Generator) -> None:
    model = CountBigram(count_bigrams(words, stoi))
    itos = {i: c for c, i in stoi.items()}
    assert isinstance(model.sample(itos, gen), str)


def test_neural_bigram_approaches_count_model(words: list[str], stoi: dict[str, int]) -> None:
    g = torch.Generator().manual_seed(1)
    x, y = build_dataset(words, stoi, 1)
    net = NeuralBigram(len(stoi), g)
    net.fit(x.squeeze(1), y, steps=300)
    trained = float(F.cross_entropy(net.logits(x.squeeze(1)), y).detach())
    counts = CountBigram(count_bigrams(words, stoi), smoothing=1.0)
    assert trained < counts.nll(words, stoi) + 0.1
