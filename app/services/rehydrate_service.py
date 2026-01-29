"""Dataset rehydration service."""

from typing import Dict, Any, List, Optional
from fastapi import HTTPException


def rehydrate_datasets(
    client_id: str, dataset_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load datasets from disk cache into memory for a client.
    If dataset_ids is None, rehydrate all datasets for the client.
    """
    try:
        from app.core.data_registry import get_data_registry

        registry = get_data_registry()
        if dataset_ids is None:
            metas = registry.list_for_client(client_id)
            dataset_ids = [m.get("dataset_id") for m in metas if m.get("dataset_id")]

        loaded = []
        missing = []

        for dataset_id in dataset_ids:
            df = registry.get(dataset_id, client_id=client_id)
            if df is not None:
                loaded.append(dataset_id)
            else:
                missing.append(dataset_id)

        return {
            "success": True,
            "client_id": client_id,
            "loaded": loaded,
            "missing": missing,
            "count_loaded": len(loaded),
            "count_missing": len(missing),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
