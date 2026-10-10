import pytest
import torch

from charlm.data import build_dataset, build_vocab
from charlm.layers import BatchNorm1d, FlattenConsecutive
from charlm.models import build_mlp, build_wavenet
from charlm.train import evaluate, sample, train


def test_batchnorm_normalises_in_training(gen: torch.Generator) -> None:
    bn = BatchNorm1d(5)
    out = bn(torch.randn(256, 5, generator=gen) * 4 + 3)
    assert torch.allclose(out.mean(0), torch.zeros(5), atol=1e-4)
    assert torch.allclose(out.std(0), torch.ones(5), atol=1e-2)


def test_batchnorm_3d_and_running_stats(gen: torch.Generator) -> None:
    bn = BatchNorm1d(4, momentum=1.0)
    x = torch.randn(32, 6, 4, generator=gen)
    bn(x)
    assert torch.allclose(bn.running_mean, x.mean((0, 1)), atol=1e-5)
    bn.training = False
    assert bn(x).shape == x.shape


def test_flatten_consecutive_shapes() -> None:
    assert FlattenConsecutive(2)(torch.zeros(4, 8, 3)).shape == (4, 4, 6)
    assert FlattenConsecutive(8)(torch.zeros(4, 8, 3)).shape == (4, 24)


def test_output_shapes(gen: torch.Generator) -> None:
    x = torch.randint(0, 27, (5, 8), generator=gen)
    assert build_wavenet(27, 8, gen)(x).shape == (5, 27)
    assert build_mlp(27, 8, gen)(x).shape == (5, 27)
    assert build_mlp(27, 8, gen, batchnorm=False)(x).shape == (5, 27)


def test_initial_loss_is_near_uniform(words: list[str]) -> None:
    g = torch.Generator().manual_seed(0)
    stoi, _ = build_vocab(words)
    x, y = build_dataset(words, stoi, 3)
    model = build_mlp(len(stoi), 3, g)
    uniform = float(torch.log(torch.tensor(float(len(stoi)))))
    assert abs(evaluate(model, x, y) - uniform) < 0.6


def test_training_reduces_loss_and_samples(words: list[str]) -> None:
    g = torch.Generator().manual_seed(0)
    stoi, itos = build_vocab(words)
    x, y = build_dataset(words, stoi, 8)
    model = build_wavenet(len(stoi), 8, g, n_embd=8, n_hidden=32)
    before = evaluate(model, x, y)
    train(model, x, y, g, steps=300, batch_size=32, lr=0.1)
    assert evaluate(model, x, y) < before - 0.5
    names = sample(model, itos, 8, g, count=3)
    assert len(names) == 3 and all(isinstance(n, str) for n in names)


def test_wavenet_rejects_a_block_size_that_is_not_a_power_of_group(gen: torch.Generator) -> None:
    with pytest.raises(ValueError, match="power of group"):
        build_wavenet(27, 6, gen)
    with pytest.raises(ValueError, match="at least 2"):
        build_wavenet(27, 8, gen, group=1)
    assert build_wavenet(27, 9, gen, group=3)(torch.zeros(2, 9, dtype=torch.long)).shape == (2, 27)
