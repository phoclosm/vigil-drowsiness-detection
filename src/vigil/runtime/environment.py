"""Version telemetry without importing optional vision libraries."""

from importlib import import_module


def opencv_version() -> str:
    """Report the loaded OpenCV build, or an explicit unavailable status."""
    try:
        return str(import_module("cv2").__version__)
    except (ImportError, AttributeError):
        return "unavailable"
