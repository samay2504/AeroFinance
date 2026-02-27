"""Ingest module initialization."""
from app.ingest.excel_ingest import ExcelIngestor
from app.ingest.json_ingest import JSONIngestor

# PDF ingestion — lazy import to avoid hard dependency on pdfplumber/rank_bm25.
# Catches Exception (not just ImportError) to handle Windows WinError 1114 DLL
# failures gracefully — server starts fine without PDF support in that case.
try:
    from app.ingest.pdf_ingest import SmartPDFIngestor
except Exception:
    SmartPDFIngestor = None

__all__ = ["ExcelIngestor", "JSONIngestor", "SmartPDFIngestor"]
