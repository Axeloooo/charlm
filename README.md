# charlm

Character-level language models, from a bigram table to an MLP, BatchNorm and a WaveNet-style tree,
trained to generate names. It follows Andrej Karpathy's makemore lectures from the
Neural Networks: Zero to Hero series:

1. [Building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo): bigram model
2. [Part 2: MLP](https://youtu.be/TCH_1BHY58I)
3. [Part 3: Activations, gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc)
4. [Part 4: Becoming a backprop ninja](https://youtu.be/q8SA3rM6ckI)
5. [Part 5: Building a WaveNet](https://youtu.be/t3YJ5hKiMQ0)

The dataset is `data/names.txt`, the 32,033 names from the makemore repository.

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Where each lecture lives

| Lecture                            | Code                                                                |
| ---------------------------------- | ------------------------------------------------------------------- |
| Bigram by counting, then as a net  | `CountBigram`, `NeuralBigram` in `src/charlm/bigram.py`             |
| MLP (Bengio et al. 2003)           | `build_mlp(..., batchnorm=False)` in `models.py`                    |
| BatchNorm, init and diagnostics    | `Linear`, `BatchNorm1d`, `Tanh` in `layers.py`; `build_mlp`         |
| Manual backward pass               | `forward`, `manual_backward` in `backprop.py` (tested vs autograd)  |
| WaveNet-style hierarchical context | `FlattenConsecutive` in `layers.py`; `build_wavenet` in `models.py` |
| Data, splits, training, sampling   | `data.py`, `train.py`                                               |

## Run

```
python -m charlm --model mlp          # Part 2
python -m charlm --model mlp-bn       # Part 3
python -m charlm --model wavenet --steps 20000
```

It prints train and dev loss and ten generated names. The CLI defaults are small so it runs on a
CPU in seconds; the lectures train for 100k-200k steps for the best dev loss, which is a job for a
GPU or a patient CPU.

## Tests

```
pytest -q
```

The tests check the vocabulary and splits, that the count and neural bigram agree, BatchNorm
statistics, tensor shapes, the initial loss being near `ln(27)`, that training lowers loss, and that
the hand-derived gradients match `loss.backward()`.

## Development

See `CLAUDE.md`. Commits follow Conventional Commits; work lands on `devel` through pull requests.

## License

MIT
