"""Dataset listing service."""
from fastapi import HTTPException


def list_datasets(client_id: str) -> dict:
    """List datasets for a client."""
    try:
        from app.core.data_registry import get_data_registry

        registry = get_data_registry()
        datasets = registry.list_for_client(client_id)

        return {
            "success": True,
            "client_id": client_id,
            "datasets": datasets,
            "count": len(datasets)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
