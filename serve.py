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
    config = uvicorn.Config(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.logging.level.lower(),
    )
    server = uvicorn.Server(config)
    import asyncio
    asyncio.run(server.serve())
