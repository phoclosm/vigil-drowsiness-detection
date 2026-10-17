"""Tests for public package release metadata."""

import vigil


def test_package_exposes_v010_version() -> None:
    assert vigil.__version__ == "0.1.0"
