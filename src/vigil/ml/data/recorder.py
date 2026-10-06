"""Deterministic session data recorder for Vigil M2.

This module implements the SessionRecorder class, which consumes structured
M1 feature records along with caller-supplied ground-truth labels and
serializes them to disk according to docs/m2-data-contract.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType
from typing import TextIO

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from vigil.ml.data.schema import SampleRecord, SessionMetadata


class SessionRecorder:
    """Records session metadata and sample sequences in canonical JSON Lines format.

    Canonical storage layout:
        {session_dir}/session_meta.json
        {session_dir}/samples.jsonl

    Responsibilities:
        1. Creates the session directory.
        2. Writes session_meta.json deterministically once upon initialization.
        3. Appends strictly validated SampleRecord instances to samples.jsonl.
        4. Maintains the previous frame index and timestamp to enforce temporal monotonicity.
        5. Refuses writes after close.
        6. Operates as a Python context manager.
    """

    def __init__(
        self,
        session_dir: Path | str | None = None,
        metadata: SessionMetadata | None = None,
        *,
        base_dir: Path | str | None = None,
    ) -> None:
        """Initialize a SessionRecorder.

        Args:
            session_dir: Explicit path to the session directory.
            metadata: SessionMetadata instance specifying session and camera configuration.
            base_dir: Optional base directory under which {session_id} directory is created
                if session_dir is not provided.
        """
        if metadata is None:
            raise ValueError("metadata must be provided.")
        if not isinstance(metadata, SessionMetadata):
            raise TypeError(f"metadata must be a SessionMetadata instance, got {type(metadata).__name__}.")

        metadata.validate()
        self._metadata = metadata

        if session_dir is not None:
            self._session_dir = Path(session_dir)
        elif base_dir is not None:
            self._session_dir = Path(base_dir) / metadata.session_id
        else:
            self._session_dir = Path("data") / "recordings" / metadata.session_id

        self._meta_path = self._session_dir / "session_meta.json"
        self._samples_path = self._session_dir / "samples.jsonl"

        # Create session directory
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Write session metadata once
        if not self._meta_path.exists():
            self._meta_path.write_text(self._metadata.to_json(), encoding="utf-8")

        # Open samples file for appending; lifecycle is managed by close() and __exit__()
        self._file: TextIO | None = open(self._samples_path, "a", encoding="utf-8")  # noqa: SIM115
        self._is_closed: bool = False
        self._prev_sample: SampleRecord | None = None
        self._sample_count: int = 0

    @property
    def session_dir(self) -> Path:
        """Directory path where session files are stored."""
        return self._session_dir

    @property
    def meta_path(self) -> Path:
        """Path to session_meta.json."""
        return self._meta_path

    @property
    def samples_path(self) -> Path:
        """Path to samples.jsonl."""
        return self._samples_path

    @property
    def metadata(self) -> SessionMetadata:
        """Session metadata associated with this recording."""
        return self._metadata

    @property
    def is_closed(self) -> bool:
        """True if the recorder has been closed."""
        return self._is_closed

    @property
    def sample_count(self) -> int:
        """Number of samples successfully appended in this recording session."""
        return self._sample_count

    @property
    def prev_sample(self) -> SampleRecord | None:
        """The most recently appended sample, or None if no samples appended."""
        return self._prev_sample

    def append(self, sample: SampleRecord) -> None:
        """Validate and append a sample record to samples.jsonl.

        Enforces strict identity, schema, and temporal monotonicity invariants:
            - Recorder must be open.
            - Sample must be a valid SampleRecord instance.
            - Sample session_id and subject_id must match session metadata.
            - Frame 0 must have frame_index=0, timestamp_ms=0.0, and frame_delta_ms=None.
            - Subsequent frames must have frame_index > previous frame_index.
            - Timestamps must be strictly monotonically increasing.
            - Frame deltas must match timestamp differences within floating-point tolerance.
            - Missing landmark features must be null, not 0.0.
            - Real capture timing is preserved truthfully: large positive deltas
              representing capture stalls are recorded as measured and are NOT clamped.
            - Ambiguous labels (label_id=-1) are recorded as supplied without filtering.
        """
        if self._is_closed:
            raise RuntimeError("Cannot append to a closed SessionRecorder.")

        if not isinstance(sample, SampleRecord):
            raise TypeError(f"sample must be a SampleRecord instance, got {type(sample).__name__}.")

        # Validate sample internal schema invariants
        sample.validate()

        # Enforce metadata identity matching
        if sample.session_id != self._metadata.session_id:
            raise ValueError(
                f"Sample session_id '{sample.session_id}' does not match recorder session_id "
                f"'{self._metadata.session_id}'."
            )
        if sample.subject_id != self._metadata.subject_id:
            raise ValueError(
                f"Sample subject_id '{sample.subject_id}' does not match recorder subject_id "
                f"'{self._metadata.subject_id}'."
            )

        # Enforce temporal and sequential invariants
        if self._prev_sample is None:
            # First sample in the session
            if sample.frame_index != 0:
                raise ValueError(
                    f"First sample must have frame_index == 0, got {sample.frame_index}."
                )
            if sample.timestamp_ms != 0.0:
                raise ValueError(
                    f"First sample must have timestamp_ms == 0.0, got {sample.timestamp_ms}."
                )
            if sample.frame_delta_ms is not None:
                raise ValueError(
                    f"First sample must have frame_delta_ms is None, got {sample.frame_delta_ms}."
                )
        else:
            # Subsequent samples in the session
            prev = self._prev_sample

            # 1. Frame index monotonicity
            if sample.frame_index == prev.frame_index:
                raise ValueError(
                    f"Duplicate frame_index {sample.frame_index} encountered."
                )
            if sample.frame_index < prev.frame_index:
                raise ValueError(
                    f"frame_index went backwards: {sample.frame_index} < {prev.frame_index}."
                )

            # 2. Strict timestamp monotonicity
            if sample.timestamp_ms <= prev.timestamp_ms:
                raise ValueError(
                    f"timestamp_ms must strictly increase: current {sample.timestamp_ms} <= "
                    f"previous {prev.timestamp_ms}."
                )

            # 3. Frame delta consistency
            if sample.frame_delta_ms is None:
                raise ValueError(
                    f"Sample at frame_index {sample.frame_index} must have a non-null frame_delta_ms."
                )
            if sample.frame_delta_ms <= 0.0:
                raise ValueError(
                    f"frame_delta_ms must be positive for frame_index {sample.frame_index}, "
                    f"got {sample.frame_delta_ms}."
                )

            # Gap handling note:
            # Real capture timing is preserved truthfully. Large positive frame deltas
            # (e.g. from camera stalls or system scheduling delays) are recorded as measured
            # and are NOT clamped or replaced with synthetic values. Downstream quality or
            # evaluation logic may detect or handle timing gaps.
            expected_delta = sample.timestamp_ms - prev.timestamp_ms
            if abs(sample.frame_delta_ms - expected_delta) > 1e-3:
                raise ValueError(
                    f"frame_delta_ms ({sample.frame_delta_ms}) does not match timestamp difference "
                    f"({expected_delta:.4f}) for frame {sample.frame_index}."
                )

        # Write sample to JSON Lines deterministically
        if self._file is not None and not self._file.closed:
            self._file.write(sample.to_json() + "\n")
            self._file.flush()

        self._prev_sample = sample
        self._sample_count += 1

    def close(self) -> None:
        """Close the recorder and flush file handles. Idempotent."""
        if not self._is_closed:
            self._is_closed = True
            if self._file is not None and not self._file.closed:
                self._file.flush()
                self._file.close()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and close recorder."""
        self.close()
