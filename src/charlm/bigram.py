"""Lecture 2: the bigram language model, by counting and as a one-layer neural net."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def count_bigrams(words: list[str], stoi: dict[str, int]) -> torch.Tensor:
    """Count how often each character follows each other character."""
    n = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for c1, c2 in zip(chs, chs[1:], strict=False):
            n[stoi[c1], stoi[c2]] += 1
    return n


class CountBigram:
    """P(next | current) from smoothed counts."""

    def __init__(self, counts: torch.Tensor, smoothing: float = 1.0) -> None:
        p = counts.float() + smoothing
        self.probs = p / p.sum(1, keepdim=True)

    def nll(self, words: list[str], stoi: dict[str, int]) -> float:
        """Average negative log likelihood per character."""
        total, n = 0.0, 0
        for w in words:
            chs = ["."] + list(w) + ["."]
            for c1, c2 in zip(chs, chs[1:], strict=False):
                total -= float(torch.log(self.probs[stoi[c1], stoi[c2]]))
                n += 1
        return total / n

    def sample(self, itos: dict[int, str], generator: torch.Generator) -> str:
        out: list[str] = []
        ix = 0
        while True:
            ix = int(torch.multinomial(self.probs[ix], 1, generator=generator))
            if ix == 0:
                return "".join(out)
            out.append(itos[ix])


class NeuralBigram:
    """The same model as one linear layer on one-hot inputs, trained by gradient descent."""

    def __init__(self, vocab_size: int, generator: torch.Generator) -> None:
        self.vocab_size = vocab_size
        self.W = torch.randn((vocab_size, vocab_size), generator=generator, requires_grad=True)

    def logits(self, xs: torch.Tensor) -> torch.Tensor:
        return F.one_hot(xs, self.vocab_size).float() @ self.W

    def loss(self, xs: torch.Tensor, ys: torch.Tensor, reg: float = 0.01) -> torch.Tensor:
        counts = self.logits(xs).exp()
        probs = counts / counts.sum(1, keepdim=True)
        nll = -probs[torch.arange(len(ys)), ys].log().mean()
        return nll + reg * (self.W**2).mean()

    def fit(self, xs: torch.Tensor, ys: torch.Tensor, steps: int = 100, lr: float = 50.0) -> float:
        for _ in range(steps):
            self.W.grad = None
            loss = self.loss(xs, ys)
            loss.backward()
            with torch.no_grad():
                self.W -= lr * self.W.grad  # type: ignore[operator]
        return float(loss.detach())
