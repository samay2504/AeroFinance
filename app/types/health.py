"""Health-related typed models."""

from pydantic import BaseModel


class HealthComponents(BaseModel):
    api: str
    sql_engine: str
    vector_db: str
    llm: str
