"""Training, evaluation and sampling shared by the MLP and WaveNet models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from charlm.layers import Sequential


def train(
    model: Sequential,
    x: torch.Tensor,
    y: torch.Tensor,
    generator: torch.Generator,
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 0.1,
    lr_decay_at: float = 0.75,
) -> list[float]:
    """Minibatch SGD; the learning rate drops 10x after `lr_decay_at` of the steps."""
    params = model.parameters()
    for p in params:
        p.requires_grad = True
    model.train(True)
    losses: list[float] = []
    for i in range(steps):
        ix = torch.randint(0, x.shape[0], (batch_size,), generator=generator)
        loss = F.cross_entropy(model(x[ix]), y[ix])
        for p in params:
            p.grad = None
        loss.backward()
        step_lr = lr if i < steps * lr_decay_at else lr / 10
        with torch.no_grad():
            for p in params:
                p -= step_lr * p.grad  # type: ignore[operator]
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def evaluate(model: Sequential, x: torch.Tensor, y: torch.Tensor) -> float:
    model.train(False)
    return float(F.cross_entropy(model(x), y))


@torch.no_grad()
def sample(
    model: Sequential,
    itos: dict[int, str],
    block_size: int,
    generator: torch.Generator,
    count: int = 10,
) -> list[str]:
    model.train(False)
    names: list[str] = []
    for _ in range(count):
        out: list[str] = []
        context = [0] * block_size
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = int(torch.multinomial(probs, 1, generator=generator))
            context = [*context[1:], ix]
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names
