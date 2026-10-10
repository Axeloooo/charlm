"""Lectures 2-5: the MLP language model, with and without BatchNorm, and the WaveNet-style tree."""

from __future__ import annotations

import torch

from charlm.layers import (
    BatchNorm1d,
    Embedding,
    FlattenConsecutive,
    Linear,
    Module,
    Sequential,
    Tanh,
)


def _finish(layers: list[Module], last: Linear) -> Sequential:
    with torch.no_grad():
        last.weight *= 0.1  # make the last layer less confident at init
    return Sequential([*layers, last])


def build_mlp(
    vocab_size: int,
    block_size: int,
    generator: torch.Generator,
    n_embd: int = 10,
    n_hidden: int = 200,
    batchnorm: bool = True,
) -> Sequential:
    """Bengio et al. 2003 character MLP (lecture 3). Without BN this is lecture 3's part 2."""
    layers: list[Module] = [
        Embedding(vocab_size, n_embd, generator),
        FlattenConsecutive(block_size),
        Linear(n_embd * block_size, n_hidden, generator, bias=not batchnorm),
    ]
    if batchnorm:
        layers.append(BatchNorm1d(n_hidden))
    layers.append(Tanh())
    return _finish(layers, Linear(n_hidden, vocab_size, generator))


def build_wavenet(
    vocab_size: int,
    block_size: int,
    generator: torch.Generator,
    n_embd: int = 24,
    n_hidden: int = 128,
    group: int = 2,
) -> Sequential:
    """Lecture 5: fuse the context hierarchically, `group` characters at a time."""
    layers: list[Module] = [Embedding(vocab_size, n_embd, generator)]
    fan_in, remaining = n_embd, block_size
    while remaining > 1:
        layers += [
            FlattenConsecutive(group),
            Linear(fan_in * group, n_hidden, generator, bias=False),
            BatchNorm1d(n_hidden),
            Tanh(),
        ]
        fan_in, remaining = n_hidden, remaining // group
    return _finish(layers, Linear(n_hidden, vocab_size, generator))
