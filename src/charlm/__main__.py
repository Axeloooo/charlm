"""Train a model from the command line: python -m charlm --model wavenet --steps 2000."""

from __future__ import annotations

import argparse

import torch

from charlm.data import build_dataset, build_vocab, load_words, split_words
from charlm.models import build_mlp, build_wavenet
from charlm.train import evaluate, sample, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Character-level name generator")
    parser.add_argument("--model", choices=["mlp", "mlp-bn", "wavenet"], default="wavenet")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2147483647)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args(argv)

    g = torch.Generator().manual_seed(args.seed)
    words = load_words()
    stoi, itos = build_vocab(words)
    block_size = 8 if args.model == "wavenet" else 3
    train_w, dev_w, _ = split_words(words)
    xtr, ytr = build_dataset(train_w, stoi, block_size)
    xdev, ydev = build_dataset(dev_w, stoi, block_size)

    if args.model == "wavenet":
        model = build_wavenet(len(stoi), block_size, g)
    else:
        model = build_mlp(len(stoi), block_size, g, batchnorm=args.model == "mlp-bn")
    train(model, xtr, ytr, g, steps=args.steps, batch_size=args.batch_size, lr=args.lr)
    print(f"train loss {evaluate(model, xtr, ytr):.4f}  dev loss {evaluate(model, xdev, ydev):.4f}")
    for name in sample(model, itos, block_size, g, args.samples):
        print(name)


if __name__ == "__main__":
    main()
