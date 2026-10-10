import torch

from charlm.backprop import forward, init_params, manual_backward
from charlm.data import build_dataset, build_vocab


def test_manual_gradients_match_autograd(words: list[str]) -> None:
    g = torch.Generator().manual_seed(7)
    stoi, _ = build_vocab(words)
    x, y = build_dataset(words, stoi, 3)
    ix = torch.randint(0, x.shape[0], (32,), generator=g)
    xb, yb = x[ix], y[ix]

    p = init_params(len(stoi), 3, g, n_hidden=16)
    for t in p.as_list():
        t.requires_grad = True
    loss, cache = forward(p, xb, yb)
    cache = {k: v.detach() for k, v in cache.items()}
    loss.backward()

    manual = manual_backward(p, xb, yb, cache)
    for name, auto, mine in zip(vars(p), p.as_list(), manual, strict=True):
        assert auto.grad is not None
        assert torch.allclose(auto.grad, mine, atol=1e-5), name
