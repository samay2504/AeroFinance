"""ID controller."""

from app.services.id_service import create_id, list_ids, validate_id


async def create_id_controller(request):
    return create_id(request.user_id, request.namespace, request.meta)


async def list_ids_controller(user_id: str):
    return list_ids(user_id)


async def validate_id_controller(request):
    return validate_id(request.user_id, request.id)
