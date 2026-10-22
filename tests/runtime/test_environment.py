"""Telemetry and real OpenCV operations without camera or UI access."""

from types import SimpleNamespace

import numpy as np
import pytest

from vigil.runtime import environment


def test_version_reports_imported_build(monkeypatch):
    monkeypatch.setattr(
        environment, "import_module", lambda _: SimpleNamespace(__version__="5.0.0")
    )
    assert environment.opencv_version() == "5.0.0"


def test_version_handles_missing_dependency(monkeypatch):
    def missing(_):
        raise ImportError("missing")

    monkeypatch.setattr(environment, "import_module", missing)
    assert environment.opencv_version() == "unavailable"


def test_real_opencv_color_conversion_and_overlay():
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert rgb[0, 0].tolist() == [30, 20, 10]
    assert image[0, 0].tolist() == [10, 20, 30]
    cv2.putText(image, "ACTIVE", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1)
    assert np.count_nonzero(image[10:60]) > 0
