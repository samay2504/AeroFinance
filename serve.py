#!/usr/bin/env python
"""
Minimal uvicorn wrapper that respects .env FASTAPI_PORT without CLI args.
Usage: python serve.py
"""

import os
from pathlib import Path

# Load .env first
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from app.config import settings
import uvicorn

if __name__ == "__main__":
    import asyncio
    import sys

    # uvloop: Cython event loop — 2-4x faster than default asyncio (Linux / macOS / Docker).
    # Falls back silently on Windows (ProactorEventLoop is used automatically there).
    if sys.platform != "win32":
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print("[serve] uvloop enabled — high-performance event loop active")
        except ImportError:
            pass  # uvloop not installed; standard asyncio is fine

    config = uvicorn.Config(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.logging.level.lower(),
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
