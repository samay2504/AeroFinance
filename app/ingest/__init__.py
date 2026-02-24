"""Ingest module initialization."""
from app.ingest.excel_ingest import ExcelIngestor
from app.ingest.json_ingest import JSONIngestor

# PDF ingestion — lazy import to avoid hard dependency on pdfplumber/rank_bm25
try:
    from app.ingest.pdf_ingest import SmartPDFIngestor
except ImportError:
    SmartPDFIngestor = None

__all__ = ["ExcelIngestor", "JSONIngestor", "SmartPDFIngestor"]
