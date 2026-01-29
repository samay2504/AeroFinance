"""ID service for create/list/validate endpoints."""

from typing import Any, Dict, List

from app.core.id_generator import (
    generate_short_id,
    normalize_client_id,
    validate_user_id,
)

_id_store: Dict[str, List[Dict[str, Any]]] = {}


def create_id(user_id: str, namespace: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    safe_user = normalize_client_id(user_id)
    if not validate_user_id(safe_user):
        return {"success": False, "error": "Invalid user_id"}

    new_id = generate_short_id(namespace or "id")
    entry = {
        "id": new_id,
        "user_id": safe_user,
        "namespace": namespace or "default",
        "meta": meta or {},
    }
    _id_store.setdefault(safe_user, []).append(entry)
    return {"success": True, "id": new_id, "user_id": safe_user}


def list_ids(user_id: str) -> Dict[str, Any]:
    safe_user = normalize_client_id(user_id)
    return {"success": True, "user_id": safe_user, "ids": _id_store.get(safe_user, [])}


def validate_id(user_id: str, id_value: str) -> Dict[str, Any]:
    safe_user = normalize_client_id(user_id)
    entries = _id_store.get(safe_user, [])
    match = next((entry for entry in entries if entry["id"] == id_value), None)
    return {
        "success": True,
        "valid": match is not None,
        "meta": match.get("meta", {}) if match else {},
    }
