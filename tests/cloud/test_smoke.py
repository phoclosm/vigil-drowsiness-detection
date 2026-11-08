"""The operator smoke verifies the deployed bundle and locked API contract."""

import io
import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient

from scripts import cloud_smoke
from vigil.cloud.app import create_app

API_TOKEN = "vigil-test-token-0123456789abcdef0123456789"


def test_authenticated_smoke_verifies_digest_and_contract(bundle_path, monkeypatch):
    monkeypatch.setenv("VIGIL_API_TOKEN", API_TOKEN)
    with TestClient(create_app(bundle_path, api_token=API_TOKEN)) as api:

        def local_urlopen(request, timeout):
            assert timeout == 10
            if isinstance(request, Request):
                url = request.full_url
                headers = dict(request.header_items())
                if request.data is None:
                    response = api.get(url, headers=headers)
                else:
                    response = api.post(url, content=request.data, headers=headers)
            else:
                url = request
                response = api.get(url)
            if response.status_code >= 400:
                raise HTTPError(url, response.status_code, "request failed", {}, None)
            return io.BytesIO(response.content)

        monkeypatch.setattr(cloud_smoke, "urlopen", local_urlopen)
        digest = api.get("/health").json()["bundle_id"]
        cloud_smoke.check("http://127.0.0.1:18080", digest)
        with pytest.raises(ValueError, match="approved digest"):
            cloud_smoke.check("http://127.0.0.1:18080", "0" * 64)


def test_smoke_rejects_inconsistent_metadata(bundle_path, monkeypatch):
    monkeypatch.setenv("VIGIL_API_TOKEN", API_TOKEN)
    with TestClient(create_app(bundle_path, api_token=API_TOKEN)) as api:

        def local_urlopen(request, timeout):
            url = request.full_url if isinstance(request, Request) else request
            if url.endswith("/health"):
                return io.BytesIO(api.get("/health").content)
            if isinstance(request, Request) and request.has_header("Authorization"):
                metadata = api.get(
                    "/model-info", headers={"Authorization": f"Bearer {API_TOKEN}"}
                ).json()
                metadata["stride"] = 1
                return io.BytesIO(json.dumps(metadata).encode())
            raise HTTPError(url, 401, "unauthenticated", {}, None)

        monkeypatch.setattr(cloud_smoke, "urlopen", local_urlopen)
        with pytest.raises(ValueError, match="locked contract"):
            cloud_smoke.check("http://127.0.0.1:18080")


@pytest.mark.parametrize("url", ["http://remote.example", "https://user:pass@host", "https://host/?token=x"])
def test_smoke_rejects_unsafe_urls_before_network(url, monkeypatch):
    monkeypatch.setattr(
        cloud_smoke, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used")
    )
    with pytest.raises(ValueError, match="smoke URL"):
        cloud_smoke.check(url)
