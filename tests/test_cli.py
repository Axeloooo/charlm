from pathlib import Path

import pytest

from charlm.__main__ import main


def test_cli_trains_and_prints(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--model", "mlp-bn", "--steps", "20", "--samples", "2"])
    out = capsys.readouterr().out
    assert "train loss" in out and "dev loss" in out and "test loss" in out


def test_cli_bigram_counts_and_samples(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--model", "bigram", "--samples", "3"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert "train loss" in lines[0] and len(lines) == 4


def test_cli_accepts_a_custom_word_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    words = tmp_path / "words.txt"
    words.write_text("\n".join(["anna", "bella", "carla", "diana", "elena"] * 40))
    main(["--data", str(words), "--model", "mlp", "--steps", "10", "--samples", "1"])
    assert "test loss" in capsys.readouterr().out
