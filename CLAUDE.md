# charlm

Character-level language models (Karpathy Zero to Hero, makemore lectures 2-6). PyTorch.

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
npm install                       # commitlint, prettier, semantic-release
pre-commit install --install-hooks
```

## Layout

- `src/charlm/`: `data.py`, `bigram.py`, `layers.py`, `models.py`, `backprop.py`, `train.py`; CLI in `__main__.py`
- `tests/`: pytest, manual gradients checked against autograd
- Root: tooling (`.pre-commit-config.yaml`, `.github/`, `package.json`, commitlint, semantic-release, prettier)

## Commands

- Format: `black .`; lint: `ruff check .` (ruff's formatter is not used)
- Types: `mypy` (strict)
- Tests: `pytest -q`
- Other files: `npx prettier --write .`

## Commits (Conventional Commits, enforced by commitlint)

Types: feat, fix, docs, infra, refactor, test, chore (also perf, build, ci, revert).

Authorship: Claude is never the author or co-author of a commit. Commits carry the author's name
and email only, with no `Co-Authored-By` or `Claude-Session` trailers. The author and committer
on this repo are Axel. commitlint rejects messages that name Claude in a trailer.

## Branches and releases

`feature/short-description`, `fix/...`, `docs/...`, `infra/...`. Branch from `devel`; work lands on
`devel` via PR. Feature PRs and release PRs are squash-merged, so the PR title must be a valid
conventional commit. Releases: cut `release/YYYY-MM-DD` from `devel`, open a PR into `main` and
squash-merge it. semantic-release runs on `main`, tags the version and updates `CHANGELOG.md`.
Never commit to `main` directly.
