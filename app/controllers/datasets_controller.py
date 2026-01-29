"""Datasets controller."""
from app.services.datasets_service import list_datasets as list_datasets_service


async def list_datasets(client_id: str) -> dict:
    """Controller for listing datasets."""
    return list_datasets_service(client_id)
