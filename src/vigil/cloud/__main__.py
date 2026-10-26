"""Container entrypoint with an explicit listen port."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "vigil.cloud.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        access_log=False,
    )
