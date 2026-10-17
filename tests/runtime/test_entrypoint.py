"""Tests for the package entry point."""

from collections.abc import Sequence

import pytest

import vigil.__main__ as entrypoint


def test_main_forwards_arguments_to_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[Sequence[str] | None] = []

    def fake_demo_main(argv: Sequence[str] | None = None) -> int:
        received.append(argv)
        return 7

    monkeypatch.setattr(entrypoint, "demo_main", fake_demo_main)

    assert entrypoint.main(["--webcam", "0"]) == 7
    assert received == [["--webcam", "0"]]
