"""
ID Controller — CRUD for session/document IDs.

ID generation delegates to app.core.id_generator (single source of truth).
Storage: thread-safe in-memory dict + JSON sidecar on disk (no extra deps).
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DATA_DIR
from app.core.id_generator import generate_short_id, get_iso_timestamp

logger = logging.getLogger("ai-ca.id_controller")

# ── Persistent store ──────────────────────────────────────────────────────────
_STORE_PATH = Path(DATA_DIR) / "ids_store.json"
_lock = threading.RLock()
_cache: Dict[str, Dict] = {}
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    if _STORE_PATH.exists():
        try:
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _cache.update(data)
            logger.info("ID store loaded: %d records", len(_cache))
        except Exception as e:
            logger.warning("Could not load ID store: %s", e)
    _loaded = True


def _persist() -> None:
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STORE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, indent=2, default=str)
        tmp.replace(_STORE_PATH)
    except Exception as e:
        logger.warning("Could not persist ID store: %s", e)


# ── Controllers ───────────────────────────────────────────────────────────────

async def create_id_controller(request) -> Dict[str, Any]:
    """Create or return an existing ID for user+namespace (idempotent)."""
    user_id: str = request.user_id
    namespace: str = request.namespace or "default"
    meta: Optional[Dict] = request.meta

    with _lock:
        _load()

        # Idempotent: return existing ID for same user+namespace
        for sid, rec in _cache.items():
            if rec.get("user_id") == user_id and rec.get("namespace") == namespace:
                return {
                    "success": True,
                    "id": sid,
                    "user_id": rec["user_id"],
                    "namespace": rec["namespace"],
                    "created_at": rec["created_at"],
                    "meta": rec.get("meta"),
                    "reused": True,
                }

        # Generate new ID using id_generator (single source of truth)
        sid = generate_short_id("id")
        rec = {
            "id": sid,
            "user_id": user_id,
            "namespace": namespace,
            "created_at": get_iso_timestamp(),
            "meta": meta or {},
        }
        _cache[sid] = rec
        _persist()
        logger.info("Created ID %s for user=%s namespace=%s", sid, user_id, namespace)

        return {
            "success": True,
            "id": sid,
            "user_id": rec["user_id"],
            "namespace": rec["namespace"],
            "created_at": rec["created_at"],
            "meta": rec["meta"],
            "reused": False,
        }


async def list_ids_controller(user_id: str) -> Dict[str, Any]:
    """Return all IDs belonging to a user."""
    with _lock:
        _load()
        records = sorted(
            [
                {
                    "id": sid,
                    "namespace": rec.get("namespace", "default"),
                    "created_at": rec.get("created_at"),
                    "meta": rec.get("meta"),
                }
                for sid, rec in _cache.items()
                if rec.get("user_id") == user_id
            ],
            key=lambda r: r.get("created_at") or "",
        )
        return {"user_id": user_id, "count": len(records), "ids": records}


async def validate_id_controller(request) -> Dict[str, Any]:
    """Validate that an ID exists and belongs to the requesting user."""
    user_id: str = request.user_id
    id_val: str = request.id

    with _lock:
        _load()
        rec = _cache.get(id_val)
        if not rec:
            return {"valid": False, "id": id_val, "user_id": user_id, "reason": "ID not found"}
        if rec.get("user_id") != user_id:
            return {"valid": False, "id": id_val, "user_id": user_id, "reason": "ID belongs to a different user"}
        return {
            "valid": True,
            "id": id_val,
            "user_id": rec["user_id"],
            "namespace": rec.get("namespace", "default"),
            "created_at": rec.get("created_at"),
            "meta": rec.get("meta"),
        }

