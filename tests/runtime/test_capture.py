"""Deterministic tests for video capture configuration and lifecycle."""

from pathlib import Path
from typing import Any

import pytest

from vigil.runtime.capture import (
    CaptureConfig,
    CaptureError,
    CaptureOpenError,
    CaptureReadError,
    CaptureStream,
)


class FakeCaptureBackend:
    """Minimal in-memory replacement for ``cv2.VideoCapture``."""

    def __init__(
        self,
        *,
        opened: bool = True,
        reads: list[tuple[bool, Any]] | None = None,
    ) -> None:
        self.opened = opened
        self.reads = list(reads or [])
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return self.opened and not self.released

    def read(self) -> tuple[bool, Any]:
        if not self.reads:
            return False, None
        return self.reads.pop(0)

    def release(self) -> None:
        self.released = True


class RecordingFactory:
    """Return one fake backend and record the source used to open it."""

    def __init__(self, backend: FakeCaptureBackend) -> None:
        self.backend = backend
        self.sources: list[int | str] = []

    def __call__(self, source: int | str) -> FakeCaptureBackend:
        self.sources.append(source)
        return self.backend


def test_webcam_config_defaults_to_device_zero() -> None:
    config = CaptureConfig.webcam()

    assert config.source == 0
    assert config.backend_source == 0
    assert config.is_webcam
    assert not config.is_video_file


@pytest.mark.parametrize("index", [-1, -10])
def test_webcam_config_rejects_negative_index(index: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CaptureConfig.webcam(index)


@pytest.mark.parametrize("source", [True, False, "0", 1.5])
def test_capture_config_rejects_ambiguous_source(source: object) -> None:
    with pytest.raises(TypeError, match="capture source"):
        CaptureConfig(source=source)  # type: ignore[arg-type]


def test_video_config_normalizes_path_for_backend(tmp_path: Path) -> None:
    video_path = tmp_path / "drive.mp4"
    config = CaptureConfig.video_file(video_path)

    assert config.source == video_path
    assert config.backend_source == str(video_path)
    assert config.is_video_file
    assert not config.is_webcam


def test_video_config_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CaptureConfig.video_file("   ")


def test_missing_video_is_rejected_before_backend_open(tmp_path: Path) -> None:
    backend = FakeCaptureBackend()
    factory = RecordingFactory(backend)
    stream = CaptureStream(
        CaptureConfig.video_file(tmp_path / "missing.mp4"),
        capture_factory=factory,
    )

    with pytest.raises(CaptureOpenError, match="does not exist"):
        stream.open()

    assert factory.sources == []
    assert not backend.released


def test_open_failure_releases_backend() -> None:
    backend = FakeCaptureBackend(opened=False)
    stream = CaptureStream(
        CaptureConfig.webcam(2),
        capture_factory=RecordingFactory(backend),
    )

    with pytest.raises(CaptureOpenError, match="webcam device 2"):
        stream.open()

    assert backend.released


def test_read_requires_an_open_stream() -> None:
    stream = CaptureStream(CaptureConfig.webcam())

    with pytest.raises(CaptureError, match="opened before reading"):
        stream.read()


def test_webcam_read_failure_is_explicit() -> None:
    backend = FakeCaptureBackend(reads=[(False, None)])
    stream = CaptureStream(
        CaptureConfig.webcam(),
        capture_factory=RecordingFactory(backend),
    ).open()

    with pytest.raises(CaptureReadError, match="webcam device 0"):
        stream.read()


def test_video_frames_stop_at_eof_and_include_timing(tmp_path: Path) -> None:
    video_path = tmp_path / "drive.mp4"
    video_path.touch()
    backend = FakeCaptureBackend(
        reads=[(True, "frame-0"), (True, "frame-1"), (False, None)]
    )
    factory = RecordingFactory(backend)
    clock_values = iter([10.0, 10.25])
    stream = CaptureStream(
        CaptureConfig.video_file(video_path),
        capture_factory=factory,
        clock=clock_values.__next__,
    ).open()

    frames = list(stream.frames())

    assert factory.sources == [str(video_path)]
    assert [frame.image for frame in frames] == ["frame-0", "frame-1"]
    assert [frame.index for frame in frames] == [0, 1]
    assert frames[0].timing.timestamp_seconds == 10.0
    assert frames[0].timing.delta_seconds is None
    assert frames[0].timing.frames_per_second is None
    assert frames[1].timing.timestamp_seconds == 10.25
    assert frames[1].timing.delta_seconds == pytest.approx(0.25)
    assert frames[1].timing.frames_per_second == pytest.approx(4.0)


def test_context_manager_releases_backend() -> None:
    backend = FakeCaptureBackend(reads=[(True, "frame")])

    with CaptureStream(
        CaptureConfig.webcam(),
        capture_factory=RecordingFactory(backend),
    ) as stream:
        assert stream.is_open
        assert stream.read() is not None

    assert backend.released
    assert not stream.is_open
