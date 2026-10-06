"""Tests for the Day 1 package entry point."""

from pytest import CaptureFixture

from vigil.__main__ import main


def test_main_reports_scaffold_status(capsys: CaptureFixture[str]) -> None:
    """The scaffold command should exit cleanly and describe its limited scope."""
    assert main() == 0

    captured = capsys.readouterr()
    assert captured.out == (
        "Vigil scaffold ready. Runtime features are not implemented yet.\n"
    )
