"""Ingest module initialization."""
from app.ingest.excel_ingest import ExcelIngestor
from app.ingest.json_ingest import JSONIngestor

__all__ = ["ExcelIngestor", "JSONIngestor"]
