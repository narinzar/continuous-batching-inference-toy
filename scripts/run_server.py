"""Launch the FastAPI batching server with uvicorn.

Usage:
    python scripts/run_server.py
    HOST=0.0.0.0 PORT=8000 python scripts/run_server.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `import src...` work when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
