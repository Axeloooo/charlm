import pytest

from charlm.__main__ import main


def test_cli_trains_and_prints(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--model", "mlp-bn", "--steps", "20", "--samples", "2"])
    out = capsys.readouterr().out
    assert "train loss" in out and "dev loss" in out
