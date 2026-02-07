"""Dataset listing service."""
from fastapi import HTTPException
import re


def _extract_doc_id(dataset_id: str) -> str:
    """Extract doc_id from dataset_id format: client:doc_XXXXX_hash:sheet"""
    match = re.search(r'(doc_\d+_[a-f0-9]+)', dataset_id)
    return match.group(1) if match else ""


def list_datasets(client_id: str) -> dict:
    """List datasets for a client."""
    try:
        from app.core.data_registry import get_data_registry

        registry = get_data_registry()
        datasets = registry.list_for_client(client_id)
        
        # Add explicit doc_id field to each dataset for consistency
        for dataset in datasets:
            dataset["doc_id"] = _extract_doc_id(dataset.get("dataset_id", ""))

        return {
            "success": True,
            "client_id": client_id,
            "datasets": datasets,
            "count": len(datasets)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
