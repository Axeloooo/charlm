"""Lectures 3 and 5: hand-written layers in the style of torch.nn."""

from __future__ import annotations

import torch


class Module:
    training = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return []


class Linear(Module):
    def __init__(
        self, fan_in: int, fan_out: int, generator: torch.Generator, bias: bool = True
    ) -> None:
        # Kaiming-style 1/sqrt(fan_in) scaling keeps activations roughly unit variance.
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x @ self.weight
        return out if self.bias is None else out + self.bias

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            dims = 0 if x.ndim == 2 else (0, 1)  # (N, C) or (N, L, C)
            mean = x.mean(dims, keepdim=True)
            var = x.var(dims, keepdim=True)
            with torch.no_grad():
                m = self.momentum
                self.running_mean = (1 - m) * self.running_mean + m * mean.squeeze()
                self.running_var = (1 - m) * self.running_var + m * var.squeeze()
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]


class Tanh(Module):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int, generator: torch.Generator) -> None:
        self.weight = torch.randn((num_embeddings, dim), generator=generator)

    def __call__(self, ix: torch.Tensor) -> torch.Tensor:
        return self.weight[ix]

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight]


class FlattenConsecutive(Module):
    """Concatenate n consecutive embeddings: (B, T, C) -> (B, T/n, C*n). WaveNet's tree."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        x = x.view(b, t // self.n, c * self.n)
        return x.squeeze(1) if x.shape[1] == 1 else x


class Sequential(Module):
    def __init__(self, layers: list[Module]) -> None:
        self.layers = layers

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[torch.Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, mode: bool = True) -> None:
        for layer in self.layers:
            layer.training = mode
