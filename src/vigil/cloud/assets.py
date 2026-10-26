"""Resolve an operator-controlled bundle locally or from private S3."""

import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit


def resolve_bundle():
    """Require a digest for downloaded artifacts; use instance IAM credentials."""
    local = os.environ.get("VIGIL_BUNDLE_PATH")
    if local:
        return Path(local)
    uri = os.environ.get("VIGIL_BUNDLE_S3_URI", "")
    parts = urlsplit(uri)
    digest = os.environ.get("VIGIL_BUNDLE_SHA256", "")
    import re

    if (
        parts.scheme != "s3"
        or not parts.netloc
        or not parts.path.strip("/")
        or parts.query
        or parts.fragment
        or parts.username
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
    ):
        raise ValueError("a local bundle or S3 object with SHA-256 is required")
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}),
    )
    body = client.get_object(Bucket=parts.netloc, Key=parts.path.lstrip("/"))["Body"]
    import tempfile

    handle, filename = tempfile.mkstemp(prefix="vigil-bundle-", suffix=".pt")
    os.close(handle)
    path = Path(filename)
    checksum = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as stream, body:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                size += len(chunk)
                if size > 128 * 1024 * 1024:
                    raise ValueError("bundle exceeds configured size limit")
                checksum.update(chunk)
                stream.write(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError("bundle digest mismatch")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise
