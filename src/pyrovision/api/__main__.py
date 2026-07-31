"""Run the PyroVision backend with its validated environment configuration."""

from __future__ import annotations

import uvicorn

from .config import load_backend_config


def main() -> None:
    config = load_backend_config()
    uvicorn.run(
        "pyrovision.api.app:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
