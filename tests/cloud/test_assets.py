"""S3 configuration and integrity checks without AWS credentials or traffic."""

import hashlib
import io
import sys
from types import SimpleNamespace

import pytest

from vigil.cloud.assets import resolve_bundle


def test_local_bundle_precedes_s3(monkeypatch, tmp_path):
    local = tmp_path / "approved.pt"
    monkeypatch.setenv("VIGIL_BUNDLE_PATH", str(local))
    monkeypatch.setenv("VIGIL_BUNDLE_S3_URI", "invalid")
    assert resolve_bundle() == local


def test_s3_requires_digest_before_network(monkeypatch):
    monkeypatch.delenv("VIGIL_BUNDLE_PATH", raising=False)
    monkeypatch.setenv("VIGIL_BUNDLE_S3_URI", "s3://bucket/model.pt")
    monkeypatch.delenv("VIGIL_BUNDLE_SHA256", raising=False)
    with pytest.raises(ValueError):
        resolve_bundle()


@pytest.mark.parametrize("correct_digest", [True, False])
def test_s3_verifies_download_digest(monkeypatch, correct_digest):
    content = b"not an actual model; integrity fixture only"
    monkeypatch.delenv("VIGIL_BUNDLE_PATH", raising=False)
    monkeypatch.setenv("VIGIL_BUNDLE_S3_URI", "s3://approved-bucket/immutable.pt")
    monkeypatch.setenv(
        "VIGIL_BUNDLE_SHA256",
        hashlib.sha256(content).hexdigest() if correct_digest else "0" * 64,
    )

    def get_object(**kwargs):
        assert kwargs == {"Bucket": "approved-bucket", "Key": "immutable.pt"}
        return {"Body": io.BytesIO(content)}

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda *_, **__: SimpleNamespace(get_object=get_object)),
    )
    if correct_digest:
        path = resolve_bundle()
        try:
            assert path.read_bytes() == content
        finally:
            path.unlink()
    else:
        with pytest.raises(ValueError, match="digest mismatch"):
            resolve_bundle()
