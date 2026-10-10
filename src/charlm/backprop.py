"""Lecture 4: backpropagate through a 2-layer MLP with BatchNorm by hand, no loss.backward()."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Params:
    C: torch.Tensor
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    bngain: torch.Tensor
    bnbias: torch.Tensor

    def as_list(self) -> list[torch.Tensor]:
        return [self.C, self.W1, self.b1, self.W2, self.b2, self.bngain, self.bnbias]


def init_params(
    vocab_size: int,
    block_size: int,
    generator: torch.Generator,
    n_embd: int = 10,
    n_hidden: int = 64,
) -> Params:
    def rnd(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * scale

    fan_in = n_embd * block_size
    return Params(
        C=rnd(vocab_size, n_embd),
        W1=rnd(fan_in, n_hidden, scale=(5 / 3) / fan_in**0.5),
        b1=rnd(n_hidden, scale=0.1),
        W2=rnd(n_hidden, vocab_size, scale=0.1),
        b2=rnd(vocab_size, scale=0.1),
        bngain=1.0 + rnd(n_hidden, scale=0.1),
        bnbias=rnd(n_hidden, scale=0.1),
    )


def forward(
    p: Params, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward pass keeping every intermediate the backward pass needs."""
    n = x.shape[0]
    emb = p.C[x]
    embcat = emb.view(n, -1)
    hprebn = embcat @ p.W1 + p.b1
    bnmean = hprebn.mean(0, keepdim=True)
    bndiff = hprebn - bnmean
    bnvar = bndiff.pow(2).sum(0, keepdim=True) / (n - 1)  # Bessel's correction
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = p.bngain * bnraw + p.bnbias
    h = torch.tanh(hpreact)
    logits = h @ p.W2 + p.b2
    loss = F.cross_entropy(logits, y)
    cache = {
        "emb": emb,
        "embcat": embcat,
        "bnvar_inv": bnvar_inv,
        "bnraw": bnraw,
        "h": h,
        "logits": logits,
    }
    return loss, cache


def manual_backward(
    p: Params, x: torch.Tensor, y: torch.Tensor, cache: dict[str, torch.Tensor]
) -> list[torch.Tensor]:
    """Gradients for Params.as_list() order, derived by hand."""
    n = x.shape[0]
    h, bnraw, bnvar_inv = cache["h"], cache["bnraw"], cache["bnvar_inv"]

    # cross entropy: softmax(logits) - one_hot(y), averaged over the batch
    dlogits = F.softmax(cache["logits"], 1)
    dlogits[range(n), y] -= 1
    dlogits /= n
    # linear layer 2
    dh = dlogits @ p.W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    # tanh
    dhpreact = (1.0 - h**2) * dh
    # batchnorm, collapsed into one expression (the lecture derives it step by step)
    dbngain = (bnraw * dhpreact).sum(0)
    dbnbias = dhpreact.sum(0)
    dhprebn = (
        p.bngain
        * bnvar_inv
        / n
        * (n * dhpreact - dhpreact.sum(0) - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0))
    )
    # linear layer 1
    dembcat = dhprebn @ p.W1.T
    dW1 = cache["embcat"].T @ dhprebn
    db1 = dhprebn.sum(0)
    # embedding lookup: scatter-add the gradient back to the rows that were used
    demb = dembcat.view(cache["emb"].shape)
    dC = torch.zeros_like(p.C)
    dC.index_add_(0, x.reshape(-1), demb.reshape(-1, p.C.shape[1]))
    return [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
